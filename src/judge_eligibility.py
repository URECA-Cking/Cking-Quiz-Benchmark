"""Read-only Judge planning from stored Pilot results.

Source integrity is a precondition: any violation stops planning before a Judge call.
Pointwise eligibility is per question (original questionIndex kept); Pairwise eligibility is
all-or-nothing per A/B pair. Direct Quiz (C) is out of Judge scope and is never read here.
"""

import hashlib
import json
from pathlib import Path

from src.approval_tracking import require_valid_approval_tracking
from src.human_evaluation import require_valid_human_evaluation


QUIZ_FILE = "quiz-generation.jsonl"
GROUNDING_FILE = "video-grounding.jsonl"
PAIR_IDENTITY_FIELDS = ("videoId", "repetition", "sourceGroundingRunId", "contentTextSha256", "promptVersion")


class SourceIntegrityError(ValueError):
    """The stored Pilot inputs cannot be trusted as Judge input; no Judge call may follow."""


def _strict_loads(text):
    def reject_constant(_value):
        raise ValueError("Non-finite JSON value")
    return json.loads(text, parse_constant=reject_constant)


def _file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path):
    try:
        rows = [_strict_loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, UnicodeError, ValueError) as exc:
        raise SourceIntegrityError("Cannot read " + path.name) from exc
    if not all(isinstance(row, dict) for row in rows):
        raise SourceIntegrityError("Invalid row in " + path.name)
    return rows


def _read_evaluation(results, run_id, snapshot):
    path = results / "evaluation" / (run_id + ".json")
    try:
        data = _strict_loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise SourceIntegrityError("Cannot read evaluation for " + run_id) from exc
    snapshot["evaluation/" + path.name] = _file_sha256(path)
    return data


def question_structure_problem(question):
    """Minimum Judge-readable structure; validator contract rules are intentionally not checked."""
    if not isinstance(question, dict):
        return "notAnObject"
    for key in ("question", "explanation", "sourceEvidence"):
        if not isinstance(question.get(key), str) or not question[key].strip():
            return "missingOrEmpty:" + key
    options = question.get("options")
    if (not isinstance(options, list) or not options
            or any(not isinstance(option, str) or not option.strip() for option in options)):
        return "invalidOptions"
    index = question.get("correctOptionIndex")
    if type(index) is not int or not 0 <= index < len(options):
        return "invalidCorrectOptionIndex"
    return None


def build_plan(results_dir, conditions):
    """``conditions`` maps A/B to their generation model IDs (from configs/judge.yaml)."""
    results = Path(results_dir)
    models = {model: condition for condition, model in conditions.items()}
    if set(conditions) != {"A", "B"} or len(models) != 2:
        raise ValueError("Judge conditions must map A and B to two distinct models")
    snapshot = {}
    quiz_path, grounding_path = results / QUIZ_FILE, results / GROUNDING_FILE
    quiz_rows, grounding_rows = _read_jsonl(quiz_path), _read_jsonl(grounding_path)
    snapshot[QUIZ_FILE] = _file_sha256(quiz_path)
    snapshot[GROUNDING_FILE] = _file_sha256(grounding_path)

    completed = {}
    for row in quiz_rows:
        if (row.get("benchmarkType") == "quiz_generation" and row.get("method") == "fixed_content_text"
                and row.get("model") in models and row.get("apiStatus") == "success"):
            key = (row.get("videoId"), row.get("repetition"), models[row["model"]])
            if key in completed:
                # The Pilot protocol never reruns a successful condition, so two are ambiguous.
                raise SourceIntegrityError("More than one successful Quiz for %s" % (key,))
            completed[key] = row

    units = []
    for (video_id, repetition, condition), row in sorted(completed.items(), key=lambda item: item[0]):
        units.append(_pointwise_unit(results, row, condition, grounding_rows, snapshot))
    pairs = _pairwise_units(units)
    return {"pointwise": units, "pairwise": pairs, "sourceSnapshot": snapshot}


def _pointwise_unit(results, row, condition, grounding_rows, snapshot):
    run_id = row.get("runId")
    try:
        require_valid_human_evaluation(row)
    except ValueError as exc:
        raise SourceIntegrityError("Invalid Human Evaluation in " + str(run_id)) from exc
    unit = {"unitType": "pointwise_set", "videoId": row["videoId"], "repetition": row["repetition"],
            "condition": condition, "quizRunId": run_id, "model": row["model"],
            "sourceGroundingRunId": row.get("sourceGroundingRunId"),
            "contentTextSha256": row.get("contentTextSha256"), "promptVersion": row.get("promptVersion"),
            "questionReviews": row.get("questionReviews", []), "status": "excluded",
            "exclusionReason": None, "eligibleQuestionIndexes": [], "excludedQuestions": [],
            "questions": {}, "contentText": None}
    if row.get("parseStatus") != "pass":
        unit["exclusionReason"] = "notParsed"
        return unit
    source = [item for item in grounding_rows if item.get("runId") == unit["sourceGroundingRunId"]]
    if len(source) != 1:
        raise SourceIntegrityError("Quiz %s needs exactly one source Grounding" % run_id)
    grounding = source[0]
    if (grounding.get("benchmarkType") != "video_grounding" or grounding.get("apiStatus") != "success"
            or grounding.get("videoId") != row["videoId"]
            or grounding.get("contentTextApprovalStatus") != "approved"):
        raise SourceIntegrityError("Quiz %s source Grounding is not an approved success" % run_id)
    if grounding.get("repetition") != row["repetition"]:
        # Same Grounding/Quiz identity rule as offline_aggregator.aggregate_pipeline.
        raise SourceIntegrityError("Quiz %s and its source Grounding differ in repetition" % run_id)
    try:
        require_valid_approval_tracking(grounding)
    except ValueError as exc:
        raise SourceIntegrityError("Invalid approval tracking for " + grounding["runId"]) from exc
    evaluation = _read_evaluation(results, grounding["runId"], snapshot)
    content = evaluation.get("contentText") if isinstance(evaluation, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise SourceIntegrityError("Approved contentText is missing for " + grounding["runId"])
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    if digest != grounding.get("contentTextSha256") or digest != unit["contentTextSha256"]:
        raise SourceIntegrityError("contentText SHA-256 mismatch for Quiz " + run_id)
    quiz = _read_evaluation(results, run_id, snapshot)
    questions = quiz.get("questions") if isinstance(quiz, dict) else None
    if not isinstance(questions, list):
        raise SourceIntegrityError("Quiz evaluation has no questions list for " + run_id)
    unit["contentText"] = content
    for index, question in enumerate(questions):
        problem = question_structure_problem(question)
        if problem is None:
            unit["questions"][index] = question
            unit["eligibleQuestionIndexes"].append(index)
        else:
            unit["excludedQuestions"].append({"questionIndex": index, "reason": problem})
    if unit["eligibleQuestionIndexes"]:
        unit["status"] = "eligible"
    else:
        unit["exclusionReason"] = "noEligibleQuestion"
    return unit


def _pairwise_units(units):
    by_key = {}
    for unit in units:
        by_key.setdefault((unit["videoId"], unit["repetition"]), {})[unit["condition"]] = unit
    pairs = []
    for (video_id, repetition), members in sorted(by_key.items()):
        pair = {"unitType": "pairwise_pair", "pairId": "%s:%s" % (video_id, repetition),
                "videoId": video_id, "repetition": repetition, "status": "excluded", "exclusionReason": None,
                "quizRunIds": {condition: unit["quizRunId"] for condition, unit in members.items()},
                "questionCounts": {condition: len(unit["eligibleQuestionIndexes"])
                                   for condition, unit in members.items()}}
        if set(members) != {"A", "B"}:
            pair["exclusionReason"] = "missingCondition"
        else:
            a, b = members["A"], members["B"]
            if any(a.get(field) != b.get(field) for field in PAIR_IDENTITY_FIELDS):
                raise SourceIntegrityError("A/B source identity mismatch for " + pair["pairId"])
            if a["status"] != "eligible" or b["status"] != "eligible":
                pair["exclusionReason"] = "notPointwiseEligible"
            elif a["excludedQuestions"] or b["excludedQuestions"]:
                pair["exclusionReason"] = "excludedQuestionPresent"
            elif len(a["eligibleQuestionIndexes"]) != len(b["eligibleQuestionIndexes"]):
                pair["exclusionReason"] = "questionCountMismatch"
            else:
                pair["status"] = "eligible"
        pairs.append(pair)
    return pairs


def eligibility_records(plan):
    """Persistable view: hashes and indexes only, never contentText or Human Evaluation values."""
    records = []
    for unit in plan["pointwise"]:
        records.append({key: unit[key] for key in (
            "unitType", "videoId", "repetition", "condition", "quizRunId", "sourceGroundingRunId",
            "contentTextSha256", "status", "exclusionReason", "eligibleQuestionIndexes",
            "excludedQuestions")})
    for pair in plan["pairwise"]:
        records.append(dict(pair))
    return records
