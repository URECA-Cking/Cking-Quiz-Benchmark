"""Production automatic Grounding: versioned machine validation and production eligibility.

Machine validation is not Human approval. A ``pass`` record only means that one stored Grounding
artifact met this version's structural, source identity, contentText hash and storage integrity
contract. It never checks the artifact against the original video: factual accuracy, key information
coverage, timestamp positions and evidenceType are not verified, and the contentText is not ground
truth. The Pilot Human approval fields (``contentTextApprovalStatus``, ``approvedBy``, ``approvedAt``,
``groundingReview``) are never read or written here, so a machine pass cannot become a Pilot approval.

A Grounding artifact is the summary row, its ``evaluation/<runId>.json`` (contentText and facts) and
its ``raw/<runId>.json`` Provider metadata, in the Pilot results layout. The validation record is a
separate JSON object; it is not stored in the Grounding row.
"""

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

from src.approval_tracking import valid_utc_timestamp
from src.provider_adapters import FIXTURE_RAW_METADATA, valid_grounding_fact, validate_raw_metadata


PRODUCTION_GROUNDING_VALIDATION_VERSION = "production-grounding-validation-v1"
# Grounding prompt versions whose stored artifact each validation version understands.
VALIDATED_GROUNDING_PROMPT_VERSIONS = {PRODUCTION_GROUNDING_VALIDATION_VERSION: frozenset({"video-grounding-v3"})}
# Grounding models each validation version understands: the gemini_video model of configs/pilot-v2.yaml,
# the only one that produced video-grounding-v3 artifacts. Another model needs a new validation version.
VALIDATED_GROUNDING_MODELS = {PRODUCTION_GROUNDING_VALIDATION_VERSION: frozenset({"gemini-3.8-flash"})}
VALIDATION_STATUSES = ("pass", "fail", "not_run")
VALIDATION_RECORD_KEYS = frozenset(("status", "version", "groundingRunId", "contentTextSha256",
                                    "validatedAt", "failureReasons"))
STORED_FACT_KEYS = frozenset(("fact", "evidenceType", "evidence", "timestampStartSeconds", "timestampEndSeconds"))
GROUNDING_FILE = "video-grounding.jsonl"
_RUN_ID = re.compile(r"[0-9a-f]{32}", re.ASCII)
_SHA256 = re.compile(r"[0-9a-f]{64}", re.ASCII)


class ProductionGroundingNotEligible(ValueError):
    """The Grounding must not be used as a production Quiz input; ``reasons`` says why."""

    def __init__(self, reasons):
        self.reasons = tuple(reasons)
        super().__init__("Grounding is not eligible for production: " + ", ".join(self.reasons))


def _strict_json(text):
    def reject_constant(_value):
        raise ValueError("Non-finite JSON value")
    return json.loads(text, parse_constant=reject_constant)


def _sha256(text):
    """SHA-256 of text already known to be UTF-8 encodable (see _content_text)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _utf8_encodable(text):
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:  # e.g. a lone surrogate from a JSON \ud800 escape
        return False
    return True


def _raw_content_text(artifact):
    evaluation = artifact.get("evaluation") if isinstance(artifact, dict) else None
    content = evaluation.get("contentText") if isinstance(evaluation, dict) else None
    return content if isinstance(content, str) and content.strip() else None


def _content_text(artifact):
    """The non-empty, UTF-8 encodable contentText, else None; only this text is ever hashed."""
    content = _raw_content_text(artifact)
    return content if content is not None and _utf8_encodable(content) else None


def load_grounding_artifact(results_dir, run_id):
    """Read one Grounding artifact by runId; missing, ambiguous, linked or corrupted storage fails closed."""
    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise ProductionGroundingNotEligible(["invalidRunId"])
    results = Path(results_dir)
    summary = results / GROUNDING_FILE
    if summary.is_symlink() or not summary.is_file():
        raise ProductionGroundingNotEligible(["missingResultRow"])
    try:
        rows = [_strict_json(line) for line in summary.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (UnicodeError, ValueError):
        raise ProductionGroundingNotEligible(["corruptedArtifact"]) from None
    matches = [row for row in rows if isinstance(row, dict) and row.get("runId") == run_id]
    if len(matches) != 1:
        raise ProductionGroundingNotEligible(["missingResultRow" if not matches else "ambiguousResultRow"])
    artifact = {"row": matches[0]}
    for name, missing in (("evaluation", "missingEvaluation"), ("raw", "missingRawMetadata")):
        path = results / name / (run_id + ".json")
        if path.is_symlink() or not path.is_file():
            raise ProductionGroundingNotEligible([missing])
        try:
            artifact[name] = _strict_json(path.read_text(encoding="utf-8"))
        except (UnicodeError, ValueError):
            raise ProductionGroundingNotEligible(["corruptedArtifact"]) from None
    return artifact


def grounding_artifact_problems(artifact, duration_seconds, version=PRODUCTION_GROUNDING_VALIDATION_VERSION):
    """Reasons this artifact fails the validation contract of ``version``; empty when it passes.

    ``duration_seconds`` is the video length when known, else None (no upper timestamp bound,
    as in the current Grounding contract).
    """
    row = artifact.get("row") if isinstance(artifact, dict) else None
    if not isinstance(row, dict):
        return ["missingResultRow"]
    problems = []
    if (row.get("benchmarkType") != "video_grounding" or row.get("method") != "gemini_video"
            or not isinstance(row.get("promptVersion"), str)
            or row["promptVersion"] not in VALIDATED_GROUNDING_PROMPT_VERSIONS.get(version, ())):
        problems.append("unsupportedGroundingArtifact")
    model = row.get("model")
    if not isinstance(model, str) or not model.strip():
        problems.append("missingModel")
    elif model not in VALIDATED_GROUNDING_MODELS.get(version, ()):
        problems.append("unsupportedGroundingArtifact")
    if not isinstance(row.get("runId"), str) or _RUN_ID.fullmatch(row["runId"]) is None:
        problems.append("invalidRunId")
    raw = artifact.get("raw")
    # A fixture run is recorded as not_run with {"source": "fixture"} raw metadata; it is never live.
    if (row.get("apiStatus") != "success" or row.get("errorCategory") is not None
            or raw is None or raw == FIXTURE_RAW_METADATA):
        problems.append("providerResultNotCompleted")
    if raw is not None and raw != FIXTURE_RAW_METADATA:
        try:
            validate_raw_metadata(raw)  # the same contract Pilot storage integrity enforces
        except ValueError:
            problems.append("invalidRawMetadata")
        else:
            if raw["provider"] != "gemini":
                problems.append("unsupportedGroundingArtifact")
    if duration_seconds is not None and (type(duration_seconds) not in (int, float)
                                         or not math.isfinite(duration_seconds) or duration_seconds <= 0):
        problems.append("invalidDurationMetadata")
        duration_seconds = None
    evaluation = artifact.get("evaluation")
    content = _content_text(artifact)
    if content is None:
        problems.append("missingContentText" if _raw_content_text(artifact) is None else "contentTextNotUtf8")
    if not isinstance(evaluation, dict) or "facts" not in evaluation:
        problems.append("missingFacts")
    elif not isinstance(evaluation["facts"], list):
        problems.append("factsNotArray")
    else:
        facts = evaluation["facts"]
        if any(not isinstance(fact, dict) or set(fact) != STORED_FACT_KEYS
               or not valid_grounding_fact(fact, fact["timestampStartSeconds"], fact["timestampEndSeconds"],
                                           duration_seconds)
               for fact in facts):
            problems.append("invalidFact")
        stored = row.get("groundingFacts")
        if (not isinstance(stored, list) or len(stored) != len(facts)
                or any(not isinstance(item, dict) or {key: item.get(key) for key in STORED_FACT_KEYS} != fact
                       for item, fact in zip(stored, facts))):
            problems.append("factsMismatch")
    row_hash = row.get("contentTextSha256")
    if not isinstance(row_hash, str) or _SHA256.fullmatch(row_hash) is None:
        problems.append("missingContentTextSha256")
    elif content is not None and _sha256(content) != row_hash:
        problems.append("contentTextSha256Mismatch")
    return list(dict.fromkeys(problems))


def validate_production_grounding(artifact, duration_seconds, validated_at=None):
    """Run machine validation and return its record (never stored in the Grounding row).

    ``validated_at`` defaults to now; a caller-supplied value must be a valid canonical UTC timestamp.
    """
    if validated_at is None:
        validated_at = datetime.now(timezone.utc).isoformat()
    elif not valid_utc_timestamp(validated_at):
        raise ValueError("validated_at must be a canonical UTC timestamp")
    problems = grounding_artifact_problems(artifact, duration_seconds)
    row = artifact.get("row") if isinstance(artifact, dict) else None
    run_id = row.get("runId") if isinstance(row, dict) else None
    content = _content_text(artifact)
    return {
        "status": "fail" if problems else "pass",
        "version": PRODUCTION_GROUNDING_VALIDATION_VERSION,
        "groundingRunId": run_id if isinstance(run_id, str) and _RUN_ID.fullmatch(run_id) else None,
        "contentTextSha256": _sha256(content) if content is not None else None,
        "validatedAt": validated_at,
        "failureReasons": problems,
    }


def _record_problems(record):
    if record is None:
        return ["missingValidationRecord"]
    if (not isinstance(record, dict) or set(record) != VALIDATION_RECORD_KEYS
            or record["status"] not in VALIDATION_STATUSES
            # Same canonical UTC timestamp contract as the rest of the results; no approval meaning.
            or not valid_utc_timestamp(record["validatedAt"])
            or not isinstance(record["failureReasons"], list)
            or any(not isinstance(reason, str) for reason in record["failureReasons"])
            # Only a fail carries reasons.
            or bool(record["failureReasons"]) != (record["status"] == "fail")):
        return ["invalidValidationRecord"]
    if record["version"] != PRODUCTION_GROUNDING_VALIDATION_VERSION:
        return ["unsupportedValidationVersion"]
    if record["status"] == "not_run":
        return ["validationNotRun"]
    if record["status"] == "fail":
        return ["validationFailed"]
    return []


def require_production_grounding_eligible(artifact, record, video_id, duration_seconds):
    """Return the validated contentText if this Grounding may be a production Quiz input; else raise.

    The single production gate: the stored record must be a supported-version ``pass`` bound to this
    run and to the exact contentText SHA-256, and the artifact is re-checked here instead of trusting
    the stored status. Human approval is neither required nor accepted as a substitute.
    """
    reasons = _record_problems(record)
    reasons += grounding_artifact_problems(artifact, duration_seconds)
    row = artifact.get("row") if isinstance(artifact, dict) else None
    if isinstance(row, dict) and row.get("videoId") != video_id:
        reasons.append("sourceVideoMismatch")
    content = _content_text(artifact)
    if isinstance(record, dict) and set(record) == VALIDATION_RECORD_KEYS:
        if not isinstance(row, dict) or record["groundingRunId"] != row.get("runId"):
            reasons.append("validationRecordRunMismatch")
        if content is None or record["contentTextSha256"] != _sha256(content):
            reasons.append("validationRecordContentTextSha256Mismatch")
    if reasons:
        raise ProductionGroundingNotEligible(reasons)
    return content
