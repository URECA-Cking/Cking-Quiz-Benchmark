"""Blind Human Evaluation of Pilot v2 Quiz sets (A/B/C). Local only: no Provider call.

A session is built from stored Pilot v2 results and kept under ``results/human/<sessionId>/``:

- ``session.json``: PRIVATE. Seed, the blind order, blindId -> original run mapping and the
  reasons sets were not eligible. Written last; a session without it is incomplete.
- ``blind-export.json``: given to the evaluator. Only allowlisted fields (YouTube URL and Quiz
  questions) plus empty verdict slots; no condition, model, run or contentText metadata.
- ``evaluation.json``: the imported evaluation. Written once and never overwritten.

Pilot summary rows are only read. Pilot v2 Human verdicts are never written to their
``questionReviews``; ``evaluation.json`` is the source of truth for Pilot v2.
"""

import argparse
import hashlib
import json
import os
import secrets
import tempfile
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.approval_tracking import normalize_approved_by
from src.human_evaluation import QUESTION_REVIEW_VALUE_KEYS
from src.judge_contract import QUESTION_FIELDS
from src.judge_eligibility import question_structure_problem
from src.offline_aggregator import aggregate_pipeline
from src.pilot_runner import PilotRunner
from src.provider_adapters import validate_quiz_config


EXPORT_FORMAT = "human-quiz-blind-v1"
SESSION_FORMAT = "human-quiz-session-v1"
EVALUATION_FORMAT = "human-quiz-evaluation-v1"
RUBRIC_VERSION = "human-quiz-rubric-v1"
ORDERING_ALGORITHM = "sha256-rank-v1"
PROMPT_VERSION = "pilot-v2"
QUESTION_COUNT = 3
VERDICTS = ("pass", "fail", "uncertain")
QUESTION_ITEMS = QUESTION_REVIEW_VALUE_KEYS
SET_ITEMS = ("coverage", "redundancy", "learningValue")
CONDITIONS = ("A", "B", "C")
DIRECT_METHOD = "gemini_direct_quiz"
# notRun: no recorded attempt (never started, e.g. A/B after a Grounding rejection); not a model failure.
# noApiSuccess: at least one recorded attempt, none with apiStatus=success.
NOT_ELIGIBLE_REASONS = ("notRun", "noApiSuccess", "parseFailed", "questionCountMismatch", "invalidQuestionStructure")
QUIZ_METHOD = "fixed_content_text"
# Minimal attempt facts kept for a set that was not evaluated, so its failure stays reproducible.
ATTEMPT_FIELDS = ("runId", "attempt", "apiStatus", "errorCategory", "httpStatus", "parseStatus",
                  "validatorStatus", "questionCount")
SEED_MAX_LENGTH = 200


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _dumps(value):
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n"


def _rank(seed, kind, key):
    """Seeded ordering key independent of Python's random implementation."""
    return _sha256((seed + "|" + kind + "|" + key).encode("utf-8"))


def _valid_session_id(session_id):
    return (isinstance(session_id, str) and len(session_id) == 32
            and all(char in "0123456789abcdef" for char in session_id))


def _valid_seed(seed):
    return (isinstance(seed, str) and seed == seed.strip() and 0 < len(seed) <= SEED_MAX_LENGTH
            and not any(unicodedata.category(char) in ("Cc", "Cf", "Zl", "Zp", "Cs") for char in seed))


def _blind_questions(questions):
    """Evaluator-visible questions: the Judge-visible generated fields plus their index (allowlist)."""
    return [dict({"questionIndex": index}, **{field: question[field] for field in QUESTION_FIELDS})
            for index, question in enumerate(questions)]


def _empty_set_review():
    return dict({item: None for item in SET_ITEMS}, reviewNote=None)


def _empty_question_review(index):
    return dict({"questionIndex": index}, **{item: None for item in QUESTION_ITEMS}, reviewNote=None)


class HumanQuizEvaluation:
    """Builds blind sessions and imports evaluations for one Pilot results directory."""

    def __init__(self, repository, results="results", config=None):
        self.repository = Path(repository).resolve()
        config = Path(config) if config else self.repository / "configs" / "pilot-v2.yaml"
        self.runner = PilotRunner(self.repository, results, config)
        validate_quiz_config(self.runner.config)
        if self.runner.config["prompt_version"] != PROMPT_VERSION:
            raise ValueError("Blind Human Quiz Evaluation is for Pilot v2 results only")
        self.results = self.runner.results
        self.root = self.results / "human"
        models = {item["provider"]: item["id"] for item in self.runner.config["quiz_generation"]["models"]}
        self.quiz_conditions = {models["gemini"]: "A", models["openai"]: "B"}
        direct = next(item for item in self.runner.config["end_to_end"]["methods"] if item["id"] == DIRECT_METHOD)
        self.direct_model = direct["quiz_model"]

    # ----- reading Pilot results (read-only) ---------------------------------------------

    def _summary(self, benchmark_type):
        path = self.results / PilotRunner.FILES[benchmark_type]
        if not path.exists():
            return []
        return [self.runner._strict_json(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]

    def _condition(self, benchmark_type, row):
        if benchmark_type == "quiz_generation":
            if row.get("method") != QUIZ_METHOD or row.get("model") not in self.quiz_conditions:
                raise ValueError("Unexpected Pilot v2 Quiz result: start from valid Pilot v2 results")
            return self.quiz_conditions[row["model"]]
        if row.get("model") != self.direct_model:
            raise ValueError("Unexpected Pilot v2 Direct result: start from valid Pilot v2 results")
        return "C"

    def _pilot_v2_rows(self):
        """{(condition, videoId, repetition): [(benchmarkType, row), ...]} for Pilot v2 A/B/C rows."""
        groups = {}
        candidates = [("quiz_generation", row) for row in self._summary("quiz_generation")]
        candidates += [("end_to_end", row) for row in self._summary("end_to_end")
                       if row.get("method") == DIRECT_METHOD]
        for benchmark_type, row in candidates:
            if row.get("promptVersion") != PROMPT_VERSION:
                continue  # Pilot v1 and other experiments are never part of a v2 session.
            key = (self._condition(benchmark_type, row), row["videoId"], row["repetition"])
            groups.setdefault(key, []).append((benchmark_type, row))
        return groups

    def _evaluation_payload(self, run_id):
        path = self.results / "evaluation" / (run_id + ".json")
        data = path.read_bytes()
        return _sha256(data), self.runner._strict_json(data.decode("utf-8"))

    # ----- session creation --------------------------------------------------------------

    def _classify(self, key, rows):
        """Return ("eligible", item) or ("notEligible", record) for one A/B/C condition."""
        condition, video_id, repetition = key
        attempts = sorted(({field: row.get(field) for field in ATTEMPT_FIELDS} for _, row in rows),
                          key=lambda item: (item["attempt"] if type(item["attempt"]) is int else 1, item["runId"]))
        record = {"condition": condition, "videoId": video_id, "repetition": repetition, "reason": None,
                  "runId": None, "attempts": attempts}
        successes = [(benchmark_type, row) for benchmark_type, row in rows if row.get("apiStatus") == "success"]
        if len(successes) > 1:
            # The Pilot v2 terminal rule allows one API success per condition; never pick one.
            raise ValueError("More than one API success for a Pilot v2 Quiz condition: %s" % (key,))
        if not successes:
            record["reason"] = "noApiSuccess" if rows else "notRun"
            return "notEligible", record
        benchmark_type, row = successes[0]
        record["runId"] = row["runId"]
        if condition in ("A", "B"):
            try:
                # Approval, review, source version, single candidate and SHA, as for offline aggregation.
                aggregate_pipeline(self.results, row.get("sourceGroundingRunId"), row["runId"])
            except ValueError as exc:
                raise ValueError("Pilot v2 Quiz %s has an unverifiable source: %s" % (row["runId"], exc)) from exc
        if row.get("parseStatus") != "pass":
            record["reason"] = "parseFailed"
            return "notEligible", record
        if row.get("questionCount") != QUESTION_COUNT:
            record["reason"] = "questionCountMismatch"
            return "notEligible", record
        evaluation_sha256, payload = self._evaluation_payload(row["runId"])
        questions = payload.get("questions") if isinstance(payload, dict) else None
        if not isinstance(questions, list) or len(questions) != row["questionCount"]:
            raise ValueError("Pilot result %s does not match its evaluation output" % row["runId"])
        problems = [{"questionIndex": index, "problem": problem}
                    for index, problem in enumerate(question_structure_problem(question) for question in questions)
                    if problem is not None]
        if problems:
            record.update(reason="invalidQuestionStructure", questionProblems=problems)
            return "notEligible", record
        item = {"benchmarkType": benchmark_type, "runId": row["runId"], "condition": condition,
                "videoId": video_id, "repetition": repetition, "promptVersion": row["promptVersion"],
                "evaluationSha256": evaluation_sha256}
        if condition in ("A", "B"):
            item.update(sourceGroundingRunId=row["sourceGroundingRunId"], contentTextSha256=row["contentTextSha256"])
        return "eligible", (item, questions)

    def _build(self, seed):
        groups = self._pilot_v2_rows()
        eligible, not_eligible = [], []
        repetitions = range(1, self.runner.config["repetitions_per_condition"] + 1)
        for video_id in sorted(self.runner.videos):
            for repetition in repetitions:
                for condition in CONDITIONS:
                    key = (condition, video_id, repetition)
                    status, value = self._classify(key, groups.pop(key, []))
                    (eligible if status == "eligible" else not_eligible).append(value)
        if groups:
            raise ValueError("Pilot v2 results include a video or repetition outside the Pilot configuration")
        pairs = {}
        for item, _ in eligible:
            if item["condition"] in ("A", "B"):
                pairs.setdefault((item["videoId"], item["repetition"]), []).append(item)
        for members in pairs.values():
            if len({(item["sourceGroundingRunId"], item["contentTextSha256"]) for item in members}) != 1:
                raise ValueError("Pilot v2 Quiz A/B do not share one approved contentText")
        if not eligible:
            raise ValueError("No Pilot v2 Quiz set is eligible for Human Evaluation")

        by_video = {}
        for item, questions in eligible:
            by_video.setdefault(item["videoId"], []).append((item, questions))
        items, videos = [], []
        for video_number, video_id in enumerate(sorted(by_video, key=lambda value: _rank(seed, "video", value)), 1):
            video_ref = "V%02d" % video_number
            sets = []
            for item, questions in sorted(by_video[video_id], key=lambda entry: _rank(seed, "set", entry[0]["runId"])):
                blind_id = "S%03d" % (len(items) + 1)
                items.append(dict(blindId=blind_id, videoRef=video_ref, **item))
                sets.append({
                    "blindId": blind_id,
                    "questions": _blind_questions(questions),
                    "setReview": _empty_set_review(),
                    "questionReviews": [_empty_question_review(index) for index in range(len(questions))]})
            videos.append({"videoRef": video_ref, "youtubeUrl": self.runner.videos[video_id]["youtubeUrl"],
                           "sets": sets})
        return items, videos, not_eligible

    def create_session(self, seed=None):
        """Create a blind session from the current Pilot v2 results; returns the sessionId."""
        if seed is None:
            seed = secrets.token_hex(16)
        if not _valid_seed(seed):
            raise ValueError("The seed must be a non-empty printable string of at most %d characters" % SEED_MAX_LENGTH)
        with self.runner._results_lock():
            self.runner._check_storage_integrity()
            items, videos, not_eligible = self._build(seed)
            session_id = uuid.uuid4().hex
            export = {"format": EXPORT_FORMAT, "sessionId": session_id, "rubricVersion": RUBRIC_VERSION,
                      "videos": videos}
            export_bytes = _dumps(export).encode("utf-8")
            session = {"format": SESSION_FORMAT, "sessionId": session_id,
                       "createdAt": datetime.now(timezone.utc).isoformat(), "rubricVersion": RUBRIC_VERSION,
                       "promptVersion": PROMPT_VERSION,
                       "sourceResults": self.results.relative_to(self.repository).as_posix(),
                       "seed": seed, "orderingAlgorithm": ORDERING_ALGORITHM,
                       "blindExportSha256": _sha256(export_bytes), "items": items, "notEligible": not_eligible}
            directory = self._session_dir(session_id)
            directory.mkdir(parents=True)
            # session.json is written last: a directory without it is an incomplete session.
            PilotRunner._atomic_write(directory / "blind-export.json", export_bytes)
            PilotRunner._atomic_write(directory / "session.json", _dumps(session).encode("utf-8"))
        return session_id

    # ----- session access ----------------------------------------------------------------

    def _session_dir(self, session_id):
        if not _valid_session_id(session_id):
            raise ValueError("Invalid Human Evaluation sessionId")
        if self.root.is_symlink() or self.root.exists() and not self.root.is_dir():
            raise ValueError("results/human must be a real directory")
        return self.root / session_id

    def _load_session(self, session_id):
        directory = self._session_dir(session_id)
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("Unknown Human Evaluation session")
        session_path, export_path = directory / "session.json", directory / "blind-export.json"
        if not session_path.is_file() or not export_path.is_file() or session_path.is_symlink() or export_path.is_symlink():
            raise ValueError("Incomplete Human Evaluation session: create a new session")
        try:
            session = self.runner._strict_json(session_path.read_text(encoding="utf-8"))
            export_bytes = export_path.read_bytes()
            export = self.runner._strict_json(export_bytes.decode("utf-8"))
        except (UnicodeError, ValueError) as exc:
            raise ValueError("Malformed Human Evaluation session") from exc
        items = session.get("items") if isinstance(session, dict) else None
        if (not isinstance(session, dict) or session.get("format") != SESSION_FORMAT
                or session.get("sessionId") != session_id or session.get("rubricVersion") != RUBRIC_VERSION
                or session.get("promptVersion") != PROMPT_VERSION
                or not _valid_seed(session.get("seed")) or session.get("orderingAlgorithm") != ORDERING_ALGORITHM
                or session.get("blindExportSha256") != _sha256(export_bytes)
                or not isinstance(items, list) or not items
                or any(not isinstance(item, dict) or not isinstance(item.get("blindId"), str)
                       or not isinstance(item.get("videoRef"), str) or item.get("condition") not in CONDITIONS
                       or item.get("benchmarkType") not in ("quiz_generation", "end_to_end")
                       or not isinstance(item.get("runId"), str) or not isinstance(item.get("evaluationSha256"), str)
                       for item in items)
                or len({item["blindId"] for item in items}) != len(items)
                or not isinstance(export, dict) or export.get("sessionId") != session_id):
            raise ValueError("Malformed Human Evaluation session")
        return directory, session, export

    # ----- evaluation import -------------------------------------------------------------

    def import_evaluation(self, session_id, submission, reviewed_by):
        """Validate a filled blind export and store it once as evaluation.json."""
        with self.runner._results_lock():
            directory, session, export = self._load_session(session_id)
            target = directory / "evaluation.json"
            if target.exists() or target.is_symlink():
                raise ValueError("This session already has an imported evaluation; create a new session to re-evaluate")
            sets = self._validate_submission(submission, session, export)
            try:
                reviewer = normalize_approved_by(reviewed_by)
            except ValueError:
                raise ValueError("A valid reviewed_by is required for Human Quiz Evaluation") from None
            self.runner._check_storage_integrity()
            self._verify_sources(session["items"], export)
            self._verify_mapping(session)
            evaluation = {"format": EVALUATION_FORMAT, "sessionId": session_id, "rubricVersion": RUBRIC_VERSION,
                          "reviewedBy": reviewer, "reviewedAt": datetime.now(timezone.utc).isoformat(),
                          "blindExportSha256": session["blindExportSha256"],
                          "sets": [dict(item, setReview=sets[item["blindId"]]["setReview"],
                                        questionReviews=sets[item["blindId"]]["questionReviews"])
                                   for item in session["items"]]}
            self._write_once(target, _dumps(evaluation).encode("utf-8"))
        return evaluation

    def _validate_submission(self, submission, session, export):
        if (not isinstance(submission, dict) or set(submission) != set(export)
                or any(submission[key] != export[key] for key in ("format", "sessionId", "rubricVersion"))
                or not isinstance(submission.get("videos"), list)):
            raise ValueError("The evaluation does not match this session's blind export")
        try:
            exported = {(video["videoRef"], blind["blindId"]): (video, blind)
                        for video in export["videos"] for blind in video["sets"]}
        except (KeyError, TypeError) as exc:
            raise ValueError("Malformed Human Evaluation session") from exc
        expected = {(item["videoRef"], item["blindId"]) for item in session["items"]}
        if set(exported) != expected or len(exported) != len(session["items"]):
            raise ValueError("Malformed Human Evaluation session")
        exported_videos = {video["videoRef"]: video for video in export["videos"]}
        seen, seen_videos, result = set(), set(), {}
        for video in submission["videos"]:
            if (not isinstance(video, dict) or set(video) != {"videoRef", "youtubeUrl", "sets"}
                    or not isinstance(video["videoRef"], str) or not isinstance(video["sets"], list)):
                raise ValueError("Unexpected video entry in the evaluation")
            if video["videoRef"] in seen_videos:
                raise ValueError("Duplicate video %s in the evaluation" % video["videoRef"])
            seen_videos.add(video["videoRef"])
            if video["videoRef"] not in exported_videos:
                raise ValueError("Unknown video in the evaluation")
            original = exported_videos[video["videoRef"]]
            if video["youtubeUrl"] != original["youtubeUrl"]:
                raise ValueError("The evaluated video differs from the blind export")
            blind_ids = [blind.get("blindId") if isinstance(blind, dict) else None for blind in video["sets"]]
            if sorted(map(str, blind_ids)) != sorted(blind["blindId"] for blind in original["sets"]):
                raise ValueError("Video %s must contain exactly its exported blind sets" % video["videoRef"])
            for blind in video["sets"]:
                if (not isinstance(blind, dict) or set(blind) != {"blindId", "questions", "setReview", "questionReviews"}
                        or not isinstance(blind["blindId"], str)):
                    raise ValueError("Unexpected set entry in the evaluation")
                key = (video["videoRef"], blind["blindId"])
                if key in seen:
                    raise ValueError("Duplicate blind set %s in the evaluation" % blind["blindId"])
                seen.add(key)
                if key not in exported:
                    raise ValueError("Unknown blind set in the evaluation")
                exported_video, exported_set = exported[key]
                if (video["youtubeUrl"] != exported_video["youtubeUrl"]
                        or blind["questions"] != exported_set["questions"]):
                    raise ValueError("The evaluated Quiz set differs from the blind export")
                result[blind["blindId"]] = {"setReview": self._set_review(blind["setReview"]),
                                            "questionReviews": self._question_reviews(blind["questionReviews"])}
        if seen_videos != set(exported_videos):
            raise ValueError("The evaluation is missing videos")
        if seen != expected:
            raise ValueError("The evaluation is missing blind sets")
        return result

    @staticmethod
    def _verdict(value):
        if type(value) is not str or value not in VERDICTS:
            raise ValueError("Every verdict must be pass, fail or uncertain")
        return value

    @staticmethod
    def _note(value):
        if value is not None and type(value) is not str:
            raise ValueError("reviewNote must be a string or null")
        return value

    def _set_review(self, review):
        if not isinstance(review, dict) or set(review) != set(SET_ITEMS) | {"reviewNote"}:
            raise ValueError("setReview needs exactly coverage, redundancy, learningValue and reviewNote")
        return dict({item: self._verdict(review[item]) for item in SET_ITEMS}, reviewNote=self._note(review["reviewNote"]))

    def _question_reviews(self, reviews):
        keys = set(QUESTION_ITEMS) | {"questionIndex", "reviewNote"}
        if not isinstance(reviews, list) or any(not isinstance(review, dict) or set(review) != keys for review in reviews):
            raise ValueError("questionReviews entries need exactly questionIndex, the six verdicts and reviewNote")
        indexes = [review["questionIndex"] for review in reviews]
        if any(type(index) is not int for index in indexes) or sorted(indexes) != list(range(QUESTION_COUNT)):
            raise ValueError("questionReviews must cover questionIndex 0, 1 and 2 exactly once")
        return [dict({"questionIndex": review["questionIndex"]},
                     **{item: self._verdict(review[item]) for item in QUESTION_ITEMS},
                     reviewNote=self._note(review["reviewNote"]))
                for review in sorted(reviews, key=lambda review: review["questionIndex"])]

    def _verify_sources(self, items, export):
        rows = {"quiz_generation": self._summary("quiz_generation"), "end_to_end": self._summary("end_to_end")}
        shown = {blind["blindId"]: blind["questions"] for video in export["videos"] for blind in video["sets"]}
        for item in items:
            matches = [row for row in rows[item["benchmarkType"]] if row.get("runId") == item["runId"]]
            if len(matches) != 1:
                raise ValueError("The original Pilot run of blind set %s is missing" % item["blindId"])
            row = matches[0]
            # The method follows from the result file and the model from the condition; not exported.
            expected_method = QUIZ_METHOD if item["benchmarkType"] == "quiz_generation" else DIRECT_METHOD
            expected_model = (self.direct_model if item["condition"] == "C" else
                              next(model for model, condition in self.quiz_conditions.items()
                                   if condition == item["condition"]))
            if (row.get("method") != expected_method or row.get("model") != expected_model
                    or row.get("questionCount") != QUESTION_COUNT):
                raise ValueError("The original Pilot run of blind set %s changed" % item["blindId"])
            identity = {"videoId": row.get("videoId"), "repetition": row.get("repetition"),
                        "promptVersion": row.get("promptVersion"),
                        "condition": self._condition(item["benchmarkType"], row)}
            if (row.get("apiStatus") != "success" or row.get("parseStatus") != "pass"
                    or any(item.get(key) != value for key, value in identity.items())
                    or item["condition"] in ("A", "B") and (
                        row.get("sourceGroundingRunId") != item.get("sourceGroundingRunId")
                        or row.get("contentTextSha256") != item.get("contentTextSha256"))):
                raise ValueError("The original Pilot run of blind set %s changed" % item["blindId"])
            try:
                evaluation_sha256, payload = self._evaluation_payload(item["runId"])
            except OSError as exc:
                raise ValueError("The original Quiz output of blind set %s is missing" % item["blindId"]) from exc
            if evaluation_sha256 != item["evaluationSha256"]:
                raise ValueError("The original Quiz output of blind set %s changed" % item["blindId"])
            # Bind blindId -> mapped run -> its actual questions -> the questions the evaluator saw.
            questions = payload.get("questions") if isinstance(payload, dict) else None
            try:
                bound = isinstance(questions, list) and _blind_questions(questions) == shown.get(item["blindId"])
            except (KeyError, TypeError):
                bound = False
            if not bound:
                raise ValueError("Blind set %s does not show the questions of its mapped source run" % item["blindId"])

    def _verify_mapping(self, session):
        """Recompute the creation-time (blindId, videoRef, source run) assignment and require it unchanged.

        The stored seed and the Pilot v2 sources are run through the same classification and SHA-256
        ranking as session creation. Question content alone cannot prove the assignment, because two
        runs can show identical questions. List order is not significant.
        """
        expected, _, _ = self._build(session["seed"])
        by_blind_id = lambda item: item["blindId"]
        if sorted(expected, key=by_blind_id) != sorted(session["items"], key=by_blind_id):
            raise ValueError("Session mapping integrity mismatch: blind sets are not assigned as at session creation")

    @staticmethod
    def _write_once(path, data):
        """Write a complete file that never replaces an existing one (same-directory hard link)."""
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=".human-", suffix=".tmp",
                                             delete=False) as stream:
                temp_path = Path(stream.name)
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temp_path, path)
            except FileExistsError:
                raise ValueError("This session already has an imported evaluation; create a new session to re-evaluate")
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Blind Human Evaluation of Pilot v2 Quiz sets (local, no Provider call)")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="Create a blind session from the stored Pilot v2 results")
    create.add_argument("--seed", help="Ordering seed; generated once and stored when omitted")
    imported = commands.add_parser("import", help="Import a filled blind export once")
    imported.add_argument("--session-id", required=True)
    imported.add_argument("--evaluation-file", type=Path, required=True)
    imported.add_argument("--reviewed-by", required=True)
    args = parser.parse_args()
    tool = HumanQuizEvaluation(Path(__file__).resolve().parents[1], args.results_dir)
    if args.command == "create":
        session_id = tool.create_session(args.seed)
        session = json.loads((tool.root / session_id / "session.json").read_text(encoding="utf-8"))
        print(json.dumps({"sessionId": session_id, "eligibleSets": len(session["items"]),
                          "notEligible": len(session["notEligible"]),
                          "blindExport": (tool.root / session_id / "blind-export.json").as_posix()}, ensure_ascii=False))
    else:
        submission = tool.runner._strict_json(args.evaluation_file.read_text(encoding="utf-8"))
        evaluation = tool.import_evaluation(args.session_id, submission, args.reviewed_by)
        print(json.dumps({"sessionId": args.session_id, "evaluatedSets": len(evaluation["sets"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
