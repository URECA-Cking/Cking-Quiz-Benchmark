"""Read-only views of approved Grounding plus separately executed Quiz runs.

This module does not invoke Providers or persist derived results.
"""

import hashlib
import json
import math
import re
from pathlib import Path

from src.approval_tracking import require_valid_approval_tracking


_SHA256 = re.compile(r"[a-f0-9]{64}\Z")
_RUN_ID = re.compile(r"[A-Za-z0-9_-]+\Z")
_USAGE_FIELDS = ("inputTokens", "outputTokens", "thinkingTokens")
_HISTORY_FIELDS = ("runId", "attempt", "apiStatus", "errorCategory", "httpStatus",
                   "providerErrorCode", "retryStopReason")


def _rows(path):
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line) for line in lines if line.strip()]
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError("Cannot read result summary") from exc
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("Invalid result summary row")
    return rows


def _unique(rows, run_id, name):
    matches = [row for row in rows if row.get("runId") == run_id]
    if len(matches) != 1:
        raise ValueError(name + " must identify exactly one result")
    return matches[0]


def _attempt(row):
    value = row.get("attempt", 1)
    if type(value) is not int or value < 1:
        raise ValueError("Invalid attempt in result history")
    return value


def _history(rows, selected, condition_fields):
    selected_attempt = _attempt(selected)
    matches = []
    for row in rows:
        if all(row.get(field) == selected.get(field) for field in condition_fields):
            attempt = _attempt(row)
            if attempt <= selected_attempt:
                item = {key: row[key] for key in _HISTORY_FIELDS if key in row}
                item["attempt"] = attempt  # Legacy rows without attempt mean attempt 1.
                matches.append(item)
    return sorted(matches, key=lambda item: (item["attempt"], item["runId"]))


def _measurement(row, key):
    value = row.get(key)
    if value is not None and (type(value) not in (int, float)
                              or not math.isfinite(value) or value < 0):
        raise ValueError("Invalid measurement in result summary")
    return value


def aggregate_pipeline(results_dir, grounding_run_id, quiz_run_id):
    """Return one derived grounded pipeline; never write to ``results_dir``."""
    if (not isinstance(grounding_run_id, str) or _RUN_ID.fullmatch(grounding_run_id) is None
            or not isinstance(quiz_run_id, str) or _RUN_ID.fullmatch(quiz_run_id) is None
            or grounding_run_id == quiz_run_id):
        raise ValueError("Distinct safe Grounding and Quiz runIds are required")
    results = Path(results_dir)
    grounding_rows = _rows(results / "video-grounding.jsonl")
    quiz_rows = _rows(results / "quiz-generation.jsonl")
    grounding = _unique(grounding_rows, grounding_run_id, "Grounding runId")
    quiz = _unique(quiz_rows, quiz_run_id, "Quiz runId")

    if (grounding.get("benchmarkType") != "video_grounding"
            or grounding.get("method") not in ("gemini_video", "authorized_transcript")
            or quiz.get("benchmarkType") != "quiz_generation"
            or quiz.get("method") != "fixed_content_text"):
        raise ValueError("Incorrect result type or method")
    if grounding.get("apiStatus") != "success" or quiz.get("apiStatus") != "success":
        raise ValueError("Both selected runs must have successful API status")
    if (quiz.get("sourceGroundingRunId") != grounding_run_id
            or grounding.get("videoId") != quiz.get("videoId")
            or grounding.get("repetition") != quiz.get("repetition")):
        raise ValueError("Grounding and Quiz identity mismatch")
    if grounding.get("contentTextApprovalStatus") != "approved":
        raise ValueError("Grounding contentText is not human approved")
    require_valid_approval_tracking(grounding)
    approved_hash = grounding.get("contentTextSha256")
    if (not isinstance(approved_hash, str) or _SHA256.fullmatch(approved_hash) is None
            or quiz.get("contentTextSha256") != approved_hash):
        raise ValueError("Approved contentText SHA-256 mismatch")
    if not isinstance(quiz.get("promptVersion"), str) or not quiz["promptVersion"].strip():
        raise ValueError("Quiz promptVersion is required")

    evaluation_path = results / "evaluation" / (grounding_run_id + ".json")
    try:
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError("Grounding evaluation is unavailable") from exc
    content = evaluation.get("contentText") if isinstance(evaluation, dict) else None
    if (not isinstance(content, str) or not content.strip()
            or hashlib.sha256(content.encode("utf-8")).hexdigest() != approved_hash):
        raise ValueError("Grounding evaluation contentText SHA-256 mismatch")

    grounding_latency = _measurement(grounding, "latencyMs")
    quiz_latency = _measurement(quiz, "latencyMs")
    grounding_cost = _measurement(grounding, "estimatedCostUsd")
    quiz_cost = _measurement(quiz, "estimatedCostUsd")
    return {
        "videoId": grounding["videoId"], "repetition": grounding["repetition"],
        "groundingRunId": grounding_run_id, "quizRunId": quiz_run_id,
        "groundingMethod": grounding["method"], "groundingModel": grounding.get("model"),
        "quizMethod": quiz["method"], "quizModel": quiz.get("model"),
        "promptVersion": quiz["promptVersion"], "contentTextSha256": approved_hash,
        "groundingApiStatus": grounding["apiStatus"], "quizApiStatus": quiz["apiStatus"],
        "pipelineApiStatus": "success",
        "selectedGroundingAttempt": _attempt(grounding),
        "selectedQuizAttempt": _attempt(quiz),
        "groundingAttemptHistory": _history(
            grounding_rows, grounding, ("videoId", "method", "model", "repetition")),
        "quizAttemptHistory": _history(
            quiz_rows, quiz, ("videoId", "method", "model", "promptVersion",
                              "contentTextSha256", "repetition")),
        "groundingLatencyMs": grounding_latency, "quizLatencyMs": quiz_latency,
        "apiLatencySumMs": (grounding_latency + quiz_latency
                            if grounding_latency is not None and quiz_latency is not None else None),
        "groundingUsage": {key: grounding.get(key) for key in _USAGE_FIELDS},
        "quizUsage": {key: quiz.get(key) for key in _USAGE_FIELDS},
        "groundingEstimatedCostUsd": grounding_cost,
        "quizEstimatedCostUsd": quiz_cost,
        "selectedPipelineEstimatedCostUsd": (
            grounding_cost + quiz_cost if grounding_cost is not None and quiz_cost is not None
            else None),
        "groundingPricingReference": grounding.get("pricingReference"),
        "quizPricingReference": quiz.get("pricingReference"),
        "contentTextApprovalStatus": grounding["contentTextApprovalStatus"],
        "groundingHumanEvaluation": {
            "groundingFacts": grounding.get("groundingFacts"),
            "omission": grounding.get("omission"),
            "hallucination": grounding.get("hallucination"),
        },
        "quizHumanEvaluation": quiz.get("questionReviews"),
    }


def compare_pair(results_dir, quiz_a_run_id, quiz_b_run_id):
    """Validate two Quiz runs used the same approved Grounding and input."""
    quiz_rows = _rows(Path(results_dir) / "quiz-generation.jsonl")
    quiz_a = _unique(quiz_rows, quiz_a_run_id, "Quiz A runId")
    quiz_b = _unique(quiz_rows, quiz_b_run_id, "Quiz B runId")
    if quiz_a_run_id == quiz_b_run_id:
        raise ValueError("A/B comparison requires distinct Quiz runs")
    fields = ("videoId", "repetition", "sourceGroundingRunId", "contentTextSha256",
              "promptVersion")
    if any(quiz_a.get(field) != quiz_b.get(field) for field in fields):
        raise ValueError("A/B Quiz input identity mismatch")
    source_id = quiz_a.get("sourceGroundingRunId")
    return {"a": aggregate_pipeline(results_dir, source_id, quiz_a_run_id),
            "b": aggregate_pipeline(results_dir, source_id, quiz_b_run_id)}
