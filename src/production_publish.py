"""Local publication records of Creator-approved production Quizzes (Issue #47).

A local publication record is a file stating that one approved Quiz, with the exact approval it relied
on, was recorded as published by ``publishedBy``. It does not publish anything to the Cking service, and
nothing here talks to an external service, database or Provider. ``publishedBy`` is a caller-supplied
name, not an authenticated identity.

Eligibility is ``ProductionCreatorReview.require_creator_approved``; Creator and Judge policies are not
re-implemented here. The record is created once per Quiz operation at
``operations/<operationId>/publication.json`` and never changed. Whether it is still current is
computed on every read: a valid record whose approval or Quiz no longer verifies is stale, not rewritten.
``publication_status`` never creates or changes a file.
"""

import hashlib
import json
import re
from datetime import datetime, timezone

from src.approval_tracking import normalize_approved_by, valid_utc_timestamp
from src.judge_contract import canonical_json, sha256_hex
from src.production_creator_review import CreatorReviewRefused, ProductionCreatorReview
from src.production_quiz import evaluate_quiz_output


PUBLICATION_FORMAT = "production-local-publication-v1"
PUBLICATION_POLICY_VERSION = "production-local-publication-policy-v1"
NOT_RECORDED = "LOCAL_PUBLICATION_NOT_RECORDED"
CURRENT = "LOCAL_PUBLICATION_CURRENT"
STALE = "LOCAL_PUBLICATION_STALE"
CORRUPTED = "LOCAL_PUBLICATION_CORRUPTED"
RECORD_KEYS = frozenset(("format", "publishId", "policyVersion", "quizOperationId", "creatorDecision",
                         "creatorDecisionSha256", "payload", "payloadSha256", "publishedBy", "publishedAt"))
PAYLOAD_KEYS = frozenset(("contentText", "questions"))
_ID = re.compile(r"[0-9a-f]{32}", re.ASCII)
_SHA256 = re.compile(r"[0-9a-f]{64}", re.ASCII)


class LocalPublicationRefused(ValueError):
    """No local publication record may be created or returned (fail-closed); ``reasons`` says why."""

    def __init__(self, reasons):
        self.reasons = tuple(reasons)
        super().__init__("Local publication is refused: " + ", ".join(self.reasons))


def _strict_json(data):
    def reject_constant(_value):
        raise ValueError("Non-finite JSON value")
    return json.loads(data, parse_constant=reject_constant)


def _same_json(left, right):
    """Equality of JSON values including their types: unlike ``==``, 0 is not ``false`` and 1 is not ``true``."""
    return canonical_json(left) == canonical_json(right)


def publish_id(operation_id, creator_decision_sha256, payload_sha256, policy_version=PUBLICATION_POLICY_VERSION):
    """Deterministic identity of one local publication: operation, the exact Creator decision file bytes,
    the published payload and the publication policy."""
    return sha256_hex({"policyVersion": policy_version, "quizOperationId": operation_id,
                       "creatorDecisionSha256": creator_decision_sha256, "payloadSha256": payload_sha256})


class ProductionLocalPublisher:
    def __init__(self, repository, results=None, quiz_config=None, judge_config=None):
        self.review = ProductionCreatorReview(repository, results, quiz_config, judge_config)
        self.quiz = self.review.quiz

    def _path(self, operation_id, name):
        return self.quiz._path("operations", operation_id, name)

    def _regular_file_bytes(self, path):
        """The bytes of a regular, non-link file; FileNotFoundError for anything else or a vanished file."""
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path.name)
        return path.read_bytes()

    def _decision_bytes(self, operation_id):
        return self._regular_file_bytes(self._path(operation_id, "creator-decision.json"))

    def _approved(self, operation_id):
        """(decision bytes, approved Quiz) from one verification: the Creator gate's decision record must be
        exactly the content of the bytes read before it, and those bytes must be unchanged after it."""
        try:
            before = self._decision_bytes(operation_id)
        except (OSError, ValueError):
            before = None
        try:
            approved = self.review.require_creator_approved(operation_id)
        except CreatorReviewRefused as exc:
            raise LocalPublicationRefused(["creatorNotApproved"] + list(exc.reasons)) from None
        if before is None:
            raise LocalPublicationRefused(["creatorDecisionUnreadable"])
        try:
            after = self._decision_bytes(operation_id)
            same_record = _same_json(_strict_json(before.decode("utf-8")), approved["decision"])
        except (OSError, UnicodeError, ValueError):
            raise LocalPublicationRefused(["creatorDecisionUnreadable"]) from None
        if after != before or not same_record:
            raise LocalPublicationRefused(["creatorDecisionChangedDuringVerification"])
        return before, approved

    def _payload_valid(self, payload):
        """The published payload meets the production Quiz output contract on its own: the output gate's
        structural and exact-type rules (evaluate_quiz_output) and questionIndex 0..n-1 as integers."""
        if type(payload) is not dict or set(payload) != PAYLOAD_KEYS:
            return False
        content, questions = payload["contentText"], payload["questions"]
        if type(content) is not str or not content.strip() or type(questions) is not list:
            return False
        if any(type(question) is not dict or type(question.get("questionIndex")) is not int
               or question["questionIndex"] != position for position, question in enumerate(questions)):
            return False
        quiz = self.quiz.quiz
        output = {"promptVersion": quiz["prompt_version"],
                  "questions": [{key: value for key, value in question.items() if key != "questionIndex"}
                                for question in questions]}
        gate = evaluate_quiz_output(output, content, quiz)
        return gate["parseStatus"] == "pass" and not gate["outputGateReasons"]

    def _record_problem(self, operation_id, record):
        """Why the record itself breaks the storage contract, else None. Nothing upstream is read."""
        try:
            if (not isinstance(record, dict) or set(record) != RECORD_KEYS or record["format"] != PUBLICATION_FORMAT
                    or record["policyVersion"] != PUBLICATION_POLICY_VERSION or record["quizOperationId"] != operation_id
                    or not isinstance(record["creatorDecisionSha256"], str)
                    or _SHA256.fullmatch(record["creatorDecisionSha256"]) is None
                    or not self._payload_valid(record["payload"])
                    or record["payloadSha256"] != sha256_hex(record["payload"])
                    or record["publishId"] != publish_id(operation_id, record["creatorDecisionSha256"], record["payloadSha256"])
                    or normalize_approved_by(record["publishedBy"]) != record["publishedBy"]
                    or not valid_utc_timestamp(record["publishedAt"])
                    or not self.review._record_valid(operation_id, record["creatorDecision"])
                    or record["creatorDecision"]["decision"] != "APPROVE"):
                return "publicationRecordInvalid"
        except (AttributeError, KeyError, TypeError, ValueError):
            return "publicationRecordInvalid"
        return None

    def _read_record(self, operation_id):
        """(record, None) for a valid stored record, (None, reason) otherwise; ``None, None`` when absent."""
        path = self._path(operation_id, "publication.json")
        if not path.exists() and not path.is_symlink():
            return None, None
        try:
            record = _strict_json(self._regular_file_bytes(path).decode("utf-8"))
        except (OSError, UnicodeError, ValueError):
            return None, "publicationRecordUnreadable"
        problem = self._record_problem(operation_id, record)
        return (None, problem) if problem else (record, None)

    def publication_status(self, operation_id):
        """``LOCAL_PUBLICATION_NOT_RECORDED`` (no record), ``LOCAL_PUBLICATION_CURRENT`` (a valid record whose
        approval and payload still verify), ``LOCAL_PUBLICATION_STALE`` (a valid record that no longer matches
        the current approval or Quiz) or ``LOCAL_PUBLICATION_CORRUPTED`` (the record itself is invalid).
        Never creates or changes a file, and never calls a Provider."""
        if not isinstance(operation_id, str) or _ID.fullmatch(operation_id) is None:
            return {"quizOperationId": operation_id, "status": CORRUPTED, "reasons": ["invalidOperationId"],
                    "publication": None}
        try:
            record, problem = self._read_record(operation_id)
        except (OSError, ValueError):
            record, problem = None, "publicationPathInvalid"
        if problem:
            return {"quizOperationId": operation_id, "status": CORRUPTED, "reasons": [problem], "publication": None}
        if record is None:
            return {"quizOperationId": operation_id, "status": NOT_RECORDED, "reasons": [], "publication": None}
        # The decision file still having the recorded bytes but other content than the snapshot means the
        # record's own snapshot is wrong, not that the approval moved on.
        try:
            current_bytes = self._decision_bytes(operation_id)
            if (hashlib.sha256(current_bytes).hexdigest() == record["creatorDecisionSha256"]
                    and not _same_json(_strict_json(current_bytes.decode("utf-8")), record["creatorDecision"])):
                return {"quizOperationId": operation_id, "status": CORRUPTED,
                        "reasons": ["creatorDecisionSnapshotMismatch"], "publication": None}
        except (OSError, UnicodeError, ValueError):
            pass  # the approval check below reports a missing or unreadable decision as stale
        reasons = []
        try:
            decision_bytes, approved = self._approved(operation_id)
            if hashlib.sha256(decision_bytes).hexdigest() != record["creatorDecisionSha256"]:
                reasons.append("creatorDecisionChanged")
            elif not _same_json(approved["decision"], record["creatorDecision"]):
                reasons.append("creatorDecisionChanged")
            elif not _same_json({"contentText": approved["contentText"], "questions": approved["questions"]},
                                record["payload"]):
                reasons.append("payloadChanged")
        except LocalPublicationRefused as exc:
            reasons = list(exc.reasons)
        status = STALE if reasons else CURRENT
        return {"quizOperationId": operation_id, "status": status, "reasons": reasons, "publication": record}

    def record_publication(self, operation_id, published_by):
        """Create the one local publication record of an approved Quiz, or return it for an identical request.

        Refused unless the Creator gate approves the Quiz, the decision file bytes are stable across that
        check and again just before writing, and the record is current right after it is written. A
        different publisher, approval, payload or policy is refused, and an existing record, even a damaged
        one, is never replaced. A record written before a later check failed is kept, not deleted.
        """
        if not isinstance(operation_id, str) or _ID.fullmatch(operation_id) is None:
            raise LocalPublicationRefused(["invalidOperationId"])
        try:
            published_by = normalize_approved_by(published_by)
        except ValueError:
            raise LocalPublicationRefused(["invalidPublishedBy"]) from None
        decision_bytes, approved = self._approved(operation_id)
        payload = {"contentText": approved["contentText"], "questions": approved["questions"]}
        decision_sha = hashlib.sha256(decision_bytes).hexdigest()
        payload_sha = sha256_hex(payload)
        record = {"format": PUBLICATION_FORMAT, "publishId": publish_id(operation_id, decision_sha, payload_sha),
                  "policyVersion": PUBLICATION_POLICY_VERSION, "quizOperationId": operation_id,
                  "creatorDecision": approved["decision"], "creatorDecisionSha256": decision_sha,
                  "payload": payload, "payloadSha256": payload_sha, "publishedBy": published_by,
                  "publishedAt": datetime.now(timezone.utc).isoformat()}
        # Last check before writing: the approval the record names is still the file on disk.
        try:
            current = self._decision_bytes(operation_id)
        except (OSError, ValueError):
            raise LocalPublicationRefused(["creatorDecisionUnreadable"]) from None
        if current != decision_bytes:
            raise LocalPublicationRefused(["creatorDecisionChangedBeforeRecording"])
        try:
            path = self._path(operation_id, "publication.json")
            existed = path.exists() or path.is_symlink()
            if not existed:
                existed = not self.quiz._create_once(path, canonical_json(record).encode("utf-8"))
        except (OSError, ValueError) as exc:
            raise LocalPublicationRefused(["publicationWriteFailed"]) from exc
        stored, problem = self._read_record(operation_id)
        if problem or stored is None:
            raise LocalPublicationRefused([problem or "publicationRecordUnreadable"])
        if not _same_json({key: value for key, value in stored.items() if key != "publishedAt"},
                          {key: value for key, value in record.items() if key != "publishedAt"}):
            raise LocalPublicationRefused(["publicationAlreadyRecorded" if existed else "publicationRecordUnreadable"])
        status = self.publication_status(operation_id)
        if status["status"] != CURRENT or status["publication"]["publishId"] != stored["publishId"]:
            # The record stays: it states what was published; its current validity is reported separately.
            raise LocalPublicationRefused(["publicationRecordedButNotCurrent"] + status["reasons"])
        return stored
