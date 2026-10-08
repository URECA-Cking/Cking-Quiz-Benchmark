"""Production Creator review and the single final approve/reject decision per Quiz operation (Issue #45).

The Creator reviews the Quiz (contentText, questions) with the Judge result or its absence, then records
one create-once decision bound to what was reviewed. Judge results and aggregations are never changed.
A decision whose reviewed Quiz, Judge evaluation or aggregation no longer matches is stale and is never
an approval. ``decidedBy`` is a caller-supplied name, not an authenticated identity. Publishing is a
later step that uses ``require_creator_approved``.

Side effects: ``review_packet`` and ``record_decision`` may create the evaluation's aggregation artifact
(``ProductionJudgeAggregator.review_status``); ``record_decision`` also creates
``operations/<operationId>/creator-decision.json`` once. ``decision_status`` and
``require_creator_approved`` never create or change a file.
"""

import re
from datetime import datetime, timezone

from src.approval_tracking import normalize_approved_by, valid_utc_timestamp
from src.judge_contract import POINTWISE_ITEMS, canonical_json
from src.production_judge_aggregation import (FAIL_CLOSED_STATES, SUMMARY_ROUTING, UNAVAILABLE_ROUTING,
                                              UNAVAILABLE_STATES, UNAVAILABLE_SUMMARY, ProductionJudgeAggregator,
                                              ProductionReviewRefused)
from src.production_quiz import ProductionQuizNotJudgeReady


DECISION_FORMAT = "production-creator-decision-v1"
DECISIONS = ("APPROVE", "REJECT")
# The kind of an approval follows the routing the Creator reviewed; it is never chosen by the caller.
APPROVAL_KINDS = {"READY_FOR_CREATOR_REVIEW": "JUDGE_AGREED", "ATTENTION_REQUIRED": "JUDGE_OVERRIDE",
                  UNAVAILABLE_ROUTING: "MANUAL_WITHOUT_JUDGE"}
REJECTION_KIND = "CREATOR_REJECTION"
PENDING, APPROVED, REJECTED = "PENDING_CREATOR_REVIEW", "CREATOR_APPROVED", "CREATOR_REJECTED"
STALE, CORRUPTED = "CREATOR_DECISION_STALE", "CREATOR_DECISION_CORRUPTED"
DECISION_KEYS = frozenset(("format", "quizOperationId", "evaluationId", "decision", "decisionKind", "decidedBy",
                           "decidedAt", "reason", "acknowledgedItems", "review", "quiz", "judge"))
REVIEW_KEYS = frozenset(("executionStatus", "semanticSummary", "reviewRouting", "aggregationId", "aggregationPolicy"))
QUIZ_KEYS = ("attempt", "inputFingerprint", "quizOutputSha256", "sourceGroundingRunId", "contentTextSha256",
             "sourceSnapshotSha256", "groundingValidationVersion")
JUDGE_KEYS = ("attempt", "inputFingerprint", "outputSha256", "resultSha256")
QUESTION_KEYS = ("questionIndex", "question", "options", "correctOptionIndex", "explanation", "sourceEvidence")
_ID = re.compile(r"[0-9a-f]{32}", re.ASCII)
_SHA256 = re.compile(r"[0-9a-f]{64}", re.ASCII)


class CreatorReviewRefused(ValueError):
    """The review, decision or approval is refused (fail-closed); ``reasons`` says why."""

    def __init__(self, reasons):
        self.reasons = tuple(reasons)
        super().__init__("Creator review is refused: " + ", ".join(self.reasons))


def _nonblank(value):
    return isinstance(value, str) and bool(value.strip())


def _problem_items(aggregation):
    """The (questionIndex, rubric) items an override must acknowledge: every fail and uncertain verdict."""
    return sorted((item["questionIndex"], item["rubric"])
                  for item in aggregation["failures"] + aggregation["uncertainties"])


def _acknowledged(items):
    """Acknowledged items as sorted (questionIndex, rubric) pairs; ValueError on malformed or repeated items."""
    if items is None:
        return []
    if not isinstance(items, list):
        raise ValueError("acknowledgedItems must be a list")
    pairs = []
    for item in items:
        if (not isinstance(item, dict) or set(item) != {"questionIndex", "rubric"}
                or type(item["questionIndex"]) is not int or item["questionIndex"] < 0
                or item["rubric"] not in POINTWISE_ITEMS):
            raise ValueError("acknowledged item is malformed")
        pairs.append((item["questionIndex"], item["rubric"]))
    if len(set(pairs)) != len(pairs):
        raise ValueError("acknowledged item is repeated")
    return sorted(pairs)


def _sha256_text(value):
    return type(value) is str and _SHA256.fullmatch(value) is not None


def _quiz_and_judge_types_valid(quiz, judge, review):
    """Exact JSON types of the recorded provenance (``True`` is not the integer 1, 0 is not ``False``)."""
    quiz_valid = (type(quiz["attempt"]) is int and quiz["attempt"] >= 1
                  and all(_sha256_text(quiz[key]) for key in ("inputFingerprint", "quizOutputSha256",
                                                              "contentTextSha256", "sourceSnapshotSha256"))
                  and type(quiz["sourceGroundingRunId"]) is str and _ID.fullmatch(quiz["sourceGroundingRunId"]) is not None
                  and type(quiz["groundingValidationVersion"]) is str and bool(quiz["groundingValidationVersion"].strip()))
    if judge is None:
        return quiz_valid
    policy = review["aggregationPolicy"]
    return (quiz_valid and type(judge["attempt"]) is int and judge["attempt"] >= 1
            and all(_sha256_text(judge[key]) for key in ("inputFingerprint", "outputSha256", "resultSha256"))
            and _sha256_text(review["aggregationId"])
            and type(policy) is dict and set(policy) == {"version", "sha256"}
            and type(policy["version"]) is str and _sha256_text(policy["sha256"]))


class ProductionCreatorReview:
    def __init__(self, repository, results=None, quiz_config=None, judge_config=None):
        self.aggregator = ProductionJudgeAggregator(repository, results, quiz_config, judge_config)
        self.judge = self.aggregator.judge
        self.quiz = self.aggregator.quiz

    def _decision_path(self, operation_id):
        return self.quiz._path("operations", operation_id, "creator-decision.json")

    # Review

    def review_packet(self, operation_id, evaluation_id):
        """What the Creator reviews: the Quiz and the Judge result (or a warning that there is none).

        Reuses require_judge_ready and review_status; corrupted records and a Quiz that is not Judge-ready
        are refused, never shown as reviewable. May create the aggregation artifact (see module notes).
        """
        if not isinstance(operation_id, str) or _ID.fullmatch(operation_id) is None:
            raise CreatorReviewRefused(["invalidOperationId"])
        try:
            quiz = self.quiz.require_judge_ready(operation_id)
        except ProductionQuizNotJudgeReady as exc:
            raise CreatorReviewRefused(["quizNotJudgeReady"] + list(exc.reasons)) from None
        try:
            review = self.aggregator.review_status(operation_id, evaluation_id)
        except ProductionReviewRefused as exc:
            raise CreatorReviewRefused(exc.reasons) from None
        quiz_provenance = {key: quiz[key] for key in QUIZ_KEYS}
        if review["executionStatus"] == "completed":
            reviewed_quiz = review["provenance"]["quiz"]
            if (reviewed_quiz["operationId"] != operation_id or reviewed_quiz["attempt"] != quiz["attempt"]
                    or reviewed_quiz["inputFingerprint"] != quiz["inputFingerprint"]
                    or reviewed_quiz["quizOutputSha256"] != quiz["quizOutputSha256"]):
                raise CreatorReviewRefused(["quizJudgeMismatch"])
        judge = {key: review[key] for key in ("evaluationId", "executionStatus", "semanticSummary", "reviewRouting",
                                              "manualReviewAllowed")}
        if review["executionStatus"] == "completed":
            aggregation = review["aggregation"]
            judge.update(aggregationId=review["aggregationId"], aggregation=aggregation,
                         problemItems=[{"questionIndex": index, "rubric": rubric}
                                       for index, rubric in _problem_items(aggregation)],
                         judgeEvaluation=review["provenance"]["judgeEvaluation"],
                         aggregationPolicy=review["provenance"]["aggregationPolicy"], unavailableWarning=None)
        else:
            judge.update(aggregationId=None, aggregation=None, problemItems=[], judgeEvaluation=None,
                         aggregationPolicy=None,
                         unavailableWarning="No completed Judge evaluation (%s); review without Judge results."
                                            % review["executionStatus"])
        return {"quizOperationId": operation_id, "contentText": quiz["contentText"],
                "questions": [{key: question[key] for key in QUESTION_KEYS} for question in quiz["questions"]],
                "quizProvenance": quiz_provenance, "judge": judge}

    # Decision

    def _decision_record(self, operation_id, evaluation_id, decision, decided_by, reason, acknowledged_items):
        """The decision record for the current review, or CreatorReviewRefused when a policy is not met."""
        if decision not in DECISIONS:
            raise CreatorReviewRefused(["unknownDecision"])
        try:
            decided_by = normalize_approved_by(decided_by)
        except ValueError:
            raise CreatorReviewRefused(["invalidDecidedBy"]) from None
        if reason is not None and not _nonblank(reason):
            raise CreatorReviewRefused(["blankReason"])
        try:
            acknowledged = _acknowledged(acknowledged_items)
        except ValueError:
            raise CreatorReviewRefused(["invalidAcknowledgedItems"]) from None
        packet = self.review_packet(operation_id, evaluation_id)
        judge = packet["judge"]
        routing = judge["reviewRouting"]
        if decision == "REJECT":
            kind = REJECTION_KIND
            if reason is None:
                raise CreatorReviewRefused(["rejectionReasonRequired"])
            if acknowledged:
                raise CreatorReviewRefused(["acknowledgedItemsNotExpected"])
        else:
            kind = APPROVAL_KINDS[routing]
            if kind == "MANUAL_WITHOUT_JUDGE" and reason is None:
                raise CreatorReviewRefused(["manualApprovalReasonRequired"])
            if kind == "JUDGE_OVERRIDE":
                if reason is None:
                    raise CreatorReviewRefused(["overrideReasonRequired"])
                expected = [(item["questionIndex"], item["rubric"]) for item in judge["problemItems"]]
                if acknowledged != expected:
                    raise CreatorReviewRefused(["acknowledgedItemsMismatch"])
            elif acknowledged:
                raise CreatorReviewRefused(["acknowledgedItemsNotExpected"])
        completed = judge["executionStatus"] == "completed"
        return {"format": DECISION_FORMAT, "quizOperationId": operation_id, "evaluationId": evaluation_id,
                "decision": decision, "decisionKind": kind, "decidedBy": decided_by,
                "decidedAt": datetime.now(timezone.utc).isoformat(), "reason": reason,
                "acknowledgedItems": [{"questionIndex": index, "rubric": rubric} for index, rubric in acknowledged],
                "review": {"executionStatus": judge["executionStatus"], "semanticSummary": judge["semanticSummary"],
                           "reviewRouting": routing, "aggregationId": judge["aggregationId"],
                           "aggregationPolicy": judge["aggregationPolicy"]},
                "quiz": packet["quizProvenance"],
                "judge": {key: judge["judgeEvaluation"][key] for key in JUDGE_KEYS} if completed else None}

    def record_decision(self, operation_id, evaluation_id, decision, decided_by, reason=None, acknowledged_items=None):
        """Record the one final decision for this Quiz operation, or return it if this exact request was
        already recorded. A different decision, decider, reason, acknowledgment or reviewed provenance is
        refused, and an existing record is never replaced, even a damaged one."""
        record = self._decision_record(operation_id, evaluation_id, decision, decided_by, reason, acknowledged_items)
        try:
            path = self._decision_path(operation_id)
            exists = path.exists() or path.is_symlink()
            if not exists:
                exists = not self.quiz._create_once(path, canonical_json(record).encode("utf-8"))
        except (OSError, ValueError) as exc:
            raise CreatorReviewRefused(["decisionWriteFailed"]) from exc
        stored = self._read_record(operation_id)
        if stored is None:
            raise CreatorReviewRefused(["decisionRecordCorrupted"])
        same = {key: value for key, value in stored.items() if key != "decidedAt"} == \
            {key: value for key, value in record.items() if key != "decidedAt"}
        if not same:
            raise CreatorReviewRefused(["decisionAlreadyRecorded" if exists else "decisionRecordCorrupted"])
        return stored

    # Status and gate (read-only)

    def _read_record(self, operation_id):
        """The stored decision if its own content is valid, else None; nothing upstream is checked here."""
        try:
            record = self.quiz._read_json(self._decision_path(operation_id))
        except (OSError, UnicodeError, ValueError):
            return None
        return record if self._record_valid(operation_id, record) else None

    @staticmethod
    def _record_valid(operation_id, record):
        """Whether a decision record (a stored file or a snapshot of one) meets the record contract on its
        own, with exact JSON types; nothing upstream is read."""
        try:
            review, quiz, judge = record["review"], record["quiz"], record["judge"]
            acknowledged = _acknowledged(record["acknowledgedItems"])
            completed = review["executionStatus"] == "completed"
            # Only a completed evaluation or one of the states review_status offers for manual review;
            # corrupted, quiz_not_judge_ready or anything unknown is never a reviewable decision.
            if not completed and review["executionStatus"] not in UNAVAILABLE_STATES:
                return False
            routing = review["reviewRouting"]
            valid = (set(record) == DECISION_KEYS and record["format"] == DECISION_FORMAT
                     and record["quizOperationId"] == operation_id and isinstance(record["evaluationId"], str)
                     and _ID.fullmatch(record["evaluationId"]) is not None
                     and record["decision"] in DECISIONS
                     and normalize_approved_by(record["decidedBy"]) == record["decidedBy"]
                     and valid_utc_timestamp(record["decidedAt"])
                     and (record["reason"] is None or _nonblank(record["reason"]))
                     and [{"questionIndex": index, "rubric": rubric} for index, rubric in acknowledged]
                     == record["acknowledgedItems"]
                     and isinstance(review, dict) and set(review) == REVIEW_KEYS
                     and isinstance(quiz, dict) and set(quiz) == set(QUIZ_KEYS)
                     and (routing == UNAVAILABLE_ROUTING) == (not completed)
                     and (SUMMARY_ROUTING.get(review["semanticSummary"]) == routing if completed
                          else review["semanticSummary"] == UNAVAILABLE_SUMMARY
                          and review["aggregationId"] is None and review["aggregationPolicy"] is None and judge is None)
                     and (not completed or isinstance(judge, dict) and set(judge) == set(JUDGE_KEYS))
                     and all(type(review[key]) is str for key in ("executionStatus", "semanticSummary", "reviewRouting"))
                     and _quiz_and_judge_types_valid(quiz, judge, review))
            if valid and record["decision"] == "REJECT":
                valid = record["decisionKind"] == REJECTION_KIND and record["reason"] is not None and not acknowledged
            elif valid:
                valid = (record["decisionKind"] == APPROVAL_KINDS.get(routing)
                         and (record["reason"] is not None or record["decisionKind"] == "JUDGE_AGREED")
                         and (bool(acknowledged) == (record["decisionKind"] == "JUDGE_OVERRIDE")))
        except (AttributeError, KeyError, TypeError, ValueError):
            valid = False
        return valid

    def _stale_reasons(self, record):
        """Why the reviewed Quiz, Judge evaluation or aggregation no longer match the record; [] if they do.

        Read-only: the aggregation is recomputed and its stored artifact is only read, never created.
        """
        operation_id, evaluation_id = record["quizOperationId"], record["evaluationId"]
        try:
            quiz = self.quiz.require_judge_ready(operation_id)
        except ProductionQuizNotJudgeReady:
            return ["quizNotJudgeReady"]
        if {key: quiz[key] for key in QUIZ_KEYS} != record["quiz"]:
            return ["quizChanged"]
        state = self.judge.evaluation_state(operation_id, evaluation_id)
        if state in FAIL_CLOSED_STATES:
            return ["evaluationState:" + state]
        recorded_state = record["review"]["executionStatus"]
        if state != recorded_state:
            return ["judgeCompletedAfterManualApproval" if state == "completed" else "judgeStateChanged"]
        if state != "completed":
            return []
        try:
            expected = self.aggregator.expected_aggregation(operation_id, evaluation_id)
        except ProductionReviewRefused:
            return ["judgeResultUnavailable"]
        aggregation_id, provenance, aggregation = expected
        if {key: provenance["judgeEvaluation"][key] for key in JUDGE_KEYS} != record["judge"]:
            return ["judgeResultChanged"]
        if (aggregation_id != record["review"]["aggregationId"]
                or provenance["aggregationPolicy"] != record["review"]["aggregationPolicy"]
                or aggregation["semanticSummary"] != record["review"]["semanticSummary"]):
            return ["aggregationChanged"]
        if record["decisionKind"] == "JUDGE_OVERRIDE" and _acknowledged(record["acknowledgedItems"]) != _problem_items(aggregation):
            return ["acknowledgedItemsChanged"]
        try:
            # Read-only, and against exactly the value compared above: never created, repaired or recomputed.
            stored = self.aggregator.verify_aggregation(operation_id, evaluation_id, expected=expected)
        except ProductionReviewRefused as exc:
            return ["aggregationArtifactMissing" if "aggregationArtifactMissing" in exc.reasons
                    else "aggregationArtifactMismatch"]
        if (stored["aggregationId"] != record["review"]["aggregationId"]
                or stored["provenance"]["aggregationPolicy"] != record["review"]["aggregationPolicy"]
                or stored["provenance"]["quiz"]["quizOutputSha256"] != record["quiz"]["quizOutputSha256"]
                or stored["provenance"]["quiz"]["inputFingerprint"] != record["quiz"]["inputFingerprint"]
                or {key: stored["provenance"]["judgeEvaluation"][key] for key in JUDGE_KEYS} != record["judge"]
                or stored["aggregation"]["semanticSummary"] != record["review"]["semanticSummary"]):
            return ["aggregationArtifactMismatch"]
        # Nothing upstream may have changed while it was being verified: the decision is approved only
        # when one consistent state was seen from the first comparison to the end.
        try:
            unchanged = self.aggregator.expected_aggregation(operation_id, evaluation_id) == expected
        except ProductionReviewRefused:
            unchanged = False
        return [] if unchanged else ["judgeResultChangedDuringVerification"]

    def decision_status(self, operation_id):
        """``PENDING_CREATOR_REVIEW`` (no decision recorded), ``CREATOR_APPROVED`` / ``CREATOR_REJECTED`` (a valid
        decision whose reviewed Quiz, Judge evaluation and aggregation still match), ``CREATOR_DECISION_STALE``
        (a valid decision whose reviewed provenance no longer matches or no longer verifies) or
        ``CREATOR_DECISION_CORRUPTED`` (the decision record itself is invalid). Never creates or changes a file."""
        if not isinstance(operation_id, str) or _ID.fullmatch(operation_id) is None:
            return {"quizOperationId": operation_id, "status": CORRUPTED, "reasons": ["invalidOperationId"],
                    "decision": None}
        try:
            path = self._decision_path(operation_id)
            if not path.exists() and not path.is_symlink():
                return {"quizOperationId": operation_id, "status": PENDING, "reasons": [], "decision": None}
        except (OSError, ValueError):
            return {"quizOperationId": operation_id, "status": CORRUPTED, "reasons": ["decisionPathInvalid"],
                    "decision": None}
        record = self._read_record(operation_id)
        if record is None:
            return {"quizOperationId": operation_id, "status": CORRUPTED, "reasons": ["decisionRecordInvalid"],
                    "decision": None}
        try:
            reasons = self._stale_reasons(record)
        except (OSError, ValueError):
            reasons = ["upstreamUnreadable"]
        if reasons:
            return {"quizOperationId": operation_id, "status": STALE, "reasons": reasons, "decision": record}
        status = APPROVED if record["decision"] == "APPROVE" else REJECTED
        return {"quizOperationId": operation_id, "status": status, "reasons": [], "decision": record}

    def require_creator_approved(self, operation_id):
        """The approved Quiz for a later Publish step, or CreatorReviewRefused. Only a valid APPROVE decision
        whose reviewed Quiz, Judge evaluation and aggregation still match passes. Never creates or changes a file."""
        status = self.decision_status(operation_id)
        if status["status"] != APPROVED:
            raise CreatorReviewRefused(["creatorStatus:" + status["status"]] + status["reasons"])
        record = status["decision"]
        try:
            quiz = self.quiz.require_judge_ready(operation_id)
        except ProductionQuizNotJudgeReady:
            raise CreatorReviewRefused(["quizNotJudgeReady"]) from None
        if {key: quiz[key] for key in QUIZ_KEYS} != record["quiz"]:
            raise CreatorReviewRefused(["quizChanged"])
        return {"quizOperationId": operation_id, "contentText": quiz["contentText"],
                "questions": [{key: question[key] for key in QUESTION_KEYS} for question in quiz["questions"]],
                "decision": record}
