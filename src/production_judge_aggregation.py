"""Deterministic aggregation of a completed production Pointwise Judge evaluation for Creator review (Issue #43).

Execution status (the Judge evaluation state), semantic summary (what the verdicts say) and review
routing are kept apart, and none of them is a Creator approval: ``ALL_PASS`` is not approval, ``HAS_FAIL``
does not discard or regenerate the Quiz, and a missing evaluation is not a semantic fail. No item is a
mandatory pass gate. The Judge evaluates against the contentText only, never the original video.

Only a verified completed evaluation (``ProductionJudgeRunner.require_completed``) is aggregated. The
result is stored once under ``operations/<quizOperationId>/judge/<evaluationId>/aggregations/<id>.json``,
where the id is derived from the exact Quiz and Judge provenance and the aggregation policy. A stored
aggregation is returned only after the upstream evaluation is verified again and the aggregation is
recomputed and found identical. Unavailable evaluations are answered read-only and never stored.
"""

import re
from datetime import datetime, timezone

from src.approval_tracking import valid_utc_timestamp
from src.judge_contract import POINTWISE_ITEMS, VERDICTS, canonical_json, sha256_hex
from src.production_judge import ProductionJudgeRunner, ProductionJudgeUnavailable


AGGREGATION_POLICY_VERSION = "production-judge-aggregation-v1"
ARTIFACT_FORMAT = "production-judge-aggregation-artifact-v1"
SUMMARY_ROUTING = {"ALL_PASS": "READY_FOR_CREATOR_REVIEW", "HAS_FAIL": "ATTENTION_REQUIRED",
                   "UNCERTAIN_ONLY": "ATTENTION_REQUIRED"}
UNAVAILABLE_SUMMARY, UNAVAILABLE_ROUTING = "UNAVAILABLE", "JUDGE_UNAVAILABLE"
# Judge states without a usable evaluation that still allow manual Creator review.
UNAVAILABLE_STATES = frozenset({"not_started", "retryable", "exhausted", "terminal", "uncertain"})
# States that are never routed to review: the records or the Quiz eligibility do not verify.
FAIL_CLOSED_STATES = frozenset({"corrupted", "quiz_not_judge_ready"})
ARTIFACT_KEYS = frozenset(("format", "aggregationId", "provenance", "aggregation", "generatedAt"))
_AGGREGATION_ID = re.compile(r"[0-9a-f]{64}", re.ASCII)


class ProductionReviewRefused(ValueError):
    """No review information may be given (fail-closed); ``reasons`` says why."""

    def __init__(self, reasons):
        self.reasons = tuple(reasons)
        super().__init__("Production Judge review information is refused: " + ", ".join(self.reasons))


def aggregation_policy_sha256():
    """Digest of everything that decides the aggregation's meaning; a change makes a separate artifact."""
    return sha256_hex({"policyVersion": AGGREGATION_POLICY_VERSION, "rubricItems": list(POINTWISE_ITEMS),
                       "verdicts": list(VERDICTS), "summaryRouting": SUMMARY_ROUTING,
                       "summaryRule": "HAS_FAIL if any fail; else UNCERTAIN_ONLY if any uncertain; else ALL_PASS"})


def _validated_questions(questions):
    """The completed Pointwise verdicts ordered by questionIndex; ValueError on any malformed entry."""
    if not isinstance(questions, list) or not questions:
        raise ValueError("Pointwise questions must be a non-empty list")
    keys = {"questionIndex"} | set(POINTWISE_ITEMS)
    for question in questions:
        if (not isinstance(question, dict) or set(question) != keys
                or type(question["questionIndex"]) is not int or question["questionIndex"] < 0):
            raise ValueError("Pointwise question entry is malformed")
        for item in POINTWISE_ITEMS:
            value = question[item]
            if (not isinstance(value, dict) or set(value) != {"verdict", "reason"} or value["verdict"] not in VERDICTS
                    or not isinstance(value["reason"], str) or not value["reason"].strip()):
                raise ValueError("Pointwise verdict entry is malformed")
    indexes = [question["questionIndex"] for question in questions]
    if len(set(indexes)) != len(indexes):
        raise ValueError("Pointwise questionIndex is repeated")
    return sorted(questions, key=lambda question: question["questionIndex"])


def aggregate_pointwise(questions):
    """Pure aggregation of completed Pointwise verdicts; the same verdicts in any order give the same result.

    Every original verdict and reason is kept (``questions``), and each fail or uncertain item names its
    questionIndex, rubric field, verdict and reason. Nothing is added, gated or decided for the Creator.
    """
    ordered = _validated_questions(questions)
    zero = lambda: {verdict: 0 for verdict in VERDICTS}
    counts, rubric_counts, question_counts = zero(), {item: zero() for item in POINTWISE_ITEMS}, []
    flagged = {"fail": [], "uncertain": []}
    for question in ordered:
        per_question = dict(zero(), questionIndex=question["questionIndex"])
        for item in POINTWISE_ITEMS:
            verdict, reason = question[item]["verdict"], question[item]["reason"]
            counts[verdict] += 1
            rubric_counts[item][verdict] += 1
            per_question[verdict] += 1
            if verdict in flagged:
                flagged[verdict].append({"questionIndex": question["questionIndex"], "rubric": item,
                                         "verdict": verdict, "reason": reason})
        question_counts.append(per_question)
    summary = "HAS_FAIL" if counts["fail"] else "UNCERTAIN_ONLY" if counts["uncertain"] else "ALL_PASS"
    return {"policyVersion": AGGREGATION_POLICY_VERSION, "policySha256": aggregation_policy_sha256(),
            "questionCount": len(ordered), "evaluationCount": len(ordered) * len(POINTWISE_ITEMS),
            "counts": counts, "rubricCounts": rubric_counts, "questionCounts": question_counts,
            "failures": flagged["fail"], "uncertainties": flagged["uncertain"],
            "semanticSummary": summary, "reviewRouting": SUMMARY_ROUTING[summary],
            "questions": [{key: question[key] for key in ["questionIndex", *POINTWISE_ITEMS]} for question in ordered]}


class ProductionJudgeAggregator:
    def __init__(self, repository, results=None, quiz_config=None, judge_config=None):
        self.judge = ProductionJudgeRunner(repository, results, quiz_config, judge_config)
        self.quiz = self.judge.quiz

    def _provenance(self, completed):
        identity = completed["identity"]
        return {"quiz": identity["quiz"], "judge": identity["judge"],
                "judgeRequestBodySha256": identity["requestBodySha256"],
                "judgeEvaluation": {"evaluationId": completed["evaluationId"], "attempt": completed["attempt"],
                                    "inputFingerprint": completed["inputFingerprint"],
                                    "outputSha256": completed["outputSha256"], "resultSha256": completed["resultSha256"]},
                "aggregationPolicy": {"version": AGGREGATION_POLICY_VERSION, "sha256": aggregation_policy_sha256()}}

    def expected_aggregation(self, quiz_operation_id, evaluation_id):
        """(aggregationId, provenance, aggregation) recomputed from the verified completed evaluation; no file
        is read or written for the aggregation itself."""
        try:
            completed = self.judge.require_completed(quiz_operation_id, evaluation_id)
        except ProductionJudgeUnavailable as exc:
            raise ProductionReviewRefused(exc.reasons) from None
        try:
            aggregation = aggregate_pointwise(completed["questions"])
            provenance = self._provenance(completed)
        except (KeyError, TypeError, ValueError):
            raise ProductionReviewRefused(["judgeResultMalformed"]) from None
        return sha256_hex(provenance), provenance, aggregation

    def _artifact_path(self, quiz_operation_id, evaluation_id, aggregation_id):
        return self.quiz._path("operations", quiz_operation_id, "judge", evaluation_id, "aggregations",
                               aggregation_id + ".json")

    def _stored_artifact(self, path, aggregation_id, provenance, aggregation):
        """The stored artifact if it is a regular file equal to the recomputed aggregation; else refused."""
        try:
            stored = self.quiz._read_json(path)
        except FileNotFoundError:  # also a link or a non-file at the path
            raise ProductionReviewRefused(["aggregationArtifactMissing"]) from None
        except (OSError, UnicodeError, ValueError):
            raise ProductionReviewRefused(["aggregationArtifactUnreadable"]) from None
        if (not isinstance(stored, dict) or set(stored) != ARTIFACT_KEYS or stored["format"] != ARTIFACT_FORMAT
                or stored["aggregationId"] != aggregation_id or not _AGGREGATION_ID.fullmatch(aggregation_id)
                or stored["provenance"] != provenance or stored["aggregation"] != aggregation
                or not valid_utc_timestamp(stored["generatedAt"])):
            raise ProductionReviewRefused(["aggregationArtifactMismatch"])
        return stored

    def aggregate(self, quiz_operation_id, evaluation_id):
        """The stored aggregation of a verified completed evaluation, created once and then only reused
        after the upstream evaluation and the recomputed aggregation match it; else ProductionReviewRefused."""
        aggregation_id, provenance, aggregation = self.expected_aggregation(quiz_operation_id, evaluation_id)
        artifact = {"format": ARTIFACT_FORMAT, "aggregationId": aggregation_id, "provenance": provenance,
                    "aggregation": aggregation, "generatedAt": datetime.now(timezone.utc).isoformat()}
        try:
            path = self._artifact_path(quiz_operation_id, evaluation_id, aggregation_id)
            if not path.exists():
                # Atomic create-once: a failed write leaves no artifact, so the next call can simply retry.
                self.quiz._create_once(path, canonical_json(artifact).encode("utf-8"))
        except (OSError, ValueError) as exc:
            raise ProductionReviewRefused(["aggregationArtifactWriteFailed"]) from exc
        return self._stored_artifact(path, aggregation_id, provenance, aggregation)

    def verify_aggregation(self, quiz_operation_id, evaluation_id, expected=None):
        """Read-only ``aggregate``: the existing stored aggregation after the same checks, never created or
        repaired. A missing artifact is refused (``aggregationArtifactMissing``).

        ``expected`` is an ``expected_aggregation`` result the caller already computed and compared; the
        artifact is then checked against exactly that value instead of a fresh recomputation."""
        if expected is None:
            expected = self.expected_aggregation(quiz_operation_id, evaluation_id)
        aggregation_id, provenance, aggregation = expected
        try:
            path = self._artifact_path(quiz_operation_id, evaluation_id, aggregation_id)
        except ValueError:
            raise ProductionReviewRefused(["aggregationArtifactUnreadable"]) from None
        return self._stored_artifact(path, aggregation_id, provenance, aggregation)

    def review_status(self, quiz_operation_id, evaluation_id):
        """Review information for one evaluation, read-only apart from creating its aggregation once.

        Completed: the verified aggregation with its semantic summary and routing. A usable evaluation
        that is not there (not started, retryable, exhausted, terminal, uncertain): UNAVAILABLE /
        JUDGE_UNAVAILABLE with the execution status, which allows manual Creator review. Corrupted
        records and a Quiz that is no longer Judge-ready are refused. Nothing here calls a Provider.
        """
        state = self.judge.evaluation_state(quiz_operation_id, evaluation_id)
        if state == "completed":
            artifact = self.aggregate(quiz_operation_id, evaluation_id)
            return {"quizOperationId": quiz_operation_id, "evaluationId": evaluation_id, "executionStatus": state,
                    "semanticSummary": artifact["aggregation"]["semanticSummary"],
                    "reviewRouting": artifact["aggregation"]["reviewRouting"],
                    "manualReviewAllowed": True, "aggregationId": artifact["aggregationId"],
                    "aggregation": artifact["aggregation"], "provenance": artifact["provenance"]}
        if state in UNAVAILABLE_STATES:
            return {"quizOperationId": quiz_operation_id, "evaluationId": evaluation_id, "executionStatus": state,
                    "semanticSummary": UNAVAILABLE_SUMMARY, "reviewRouting": UNAVAILABLE_ROUTING,
                    "manualReviewAllowed": True, "unavailableReason": "evaluationState:" + state}
        raise ProductionReviewRefused(["evaluationState:" + state])
