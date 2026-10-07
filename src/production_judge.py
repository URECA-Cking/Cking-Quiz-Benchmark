"""Production GPT Pointwise Judge for a Judge-ready production Quiz (Issue #41).

Flow: ProductionQuizRunner.require_judge_ready -> Pointwise request -> strict validation ->
create-once result. It reuses the Pilot Judge's OpenAI client, cost guard, HTTP retry, Pointwise
prompt, schema and validator unchanged; Pairwise, A/B, Human values and Pilot results are not used.

Meaning boundaries:
- The Judge evaluates the Quiz against its exact contentText only. It never sees the video, so it
  does not verify the Grounding against the video (facts, omissions, timestamps); a Quiz that is
  faithful to a wrong Grounding can pass.
- A valid ``fail`` or ``uncertain`` verdict is a completed evaluation. It is never retried, and
  nothing here regenerates a Quiz or aggregates verdicts into an overall PASS/FAIL.
- A completed evaluation is not Creator approval.

Storage (``results/production/operations/<quizOperationId>/judge/<evaluationId>/``), created once:
- ``evaluation.json``: identity (exact Quiz provenance and Judge settings), input fingerprint and the
  execution policy (fixed retry policy plus the run's timeout, cost caps, input estimate and price).
- ``request.json``: the exact request body; its SHA-256 is part of the identity.
- ``attempts/<n>/started.json``, ``attempts/<n>/requests/<k>.json`` (each HTTP send's reservation,
  written before the send), ``attempts/<n>/output.json`` (model text) and ``attempts/<n>/result.json``
  (written last). An attempt without its result is uncertain and is never replayed automatically.
"""

import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.approval_tracking import valid_utc_timestamp
from src.judge_client import JudgeCallFailure, JudgeHttpClient, JudgePolicy, build_request, estimate_cost
from src.judge_contract import (POINTWISE_INSTRUCTIONS, QUESTION_FIELDS, RUBRIC_VERSION, SCHEMA_VERSION,
                                canonical_json, pointwise_prompt, pointwise_schema, sha256_hex)
from src.judge_validation import JudgeOutputError, validate_pointwise
from src.production_quiz import ProductionQuizNotJudgeReady, ProductionQuizRunner


PROMPT_VERSIONS = frozenset({"production-pointwise-judge-v1"})
KIND = "pointwise"
PROVIDER = "openai"
# The fixed execution policy the Judge client implements: one technical attempt sends at most one
# HTTP retry, only for 429/5xx with a Retry-After; two attempts at most, never started automatically.
EXECUTION_POLICY = {"http_retry_max": 1, "retry_after_required": True, "max_technical_attempts": 2,
                    "automatic_logical_retry": False, "request_limit": 4}
CONFIG_KEYS = frozenset(("provider", "model", "reasoning", "api_key_environment_variable", "max_output_tokens",
                         "prompt_version", "rubric_version", "schema_version", "execution_policy"))
RUNTIME_KEYS = frozenset(("timeoutSeconds", "perCallCostLimitUsd", "totalCostLimitUsd", "estimatedInputTokens", "price"))
PRICE_KEYS = frozenset(("input", "output", "cachedInput", "reference", "checkedAt"))

MANIFEST_FORMAT = "production-judge-evaluation-v1"
STARTED_FORMAT = "production-judge-attempt-started-v1"
REQUEST_FORMAT = "production-judge-request-intent-v1"
RESULT_FORMAT = "production-judge-attempt-result-v1"
# Every attempt record carries the SHA-256 of the evaluation's execution policy, so a stored policy that is
# later changed (even to other valid values) no longer matches the records written under it.
STARTED_KEYS = frozenset(("format", "evaluationId", "attempt", "inputFingerprint", "requestBodySha256",
                          "executionPolicySha256", "startedAt"))
REQUEST_KEYS = frozenset(("format", "evaluationId", "attempt", "request", "requestId", "requestBodySha256",
                          "executionPolicySha256", "reservedUsd", "createdAt"))
RESULT_KEYS = frozenset(("format", "evaluationId", "attempt", "inputFingerprint", "requestBodySha256",
                         "executionPolicySha256", "startedAt",
                         "completedAt", "outcome", "errorCategory", "httpStatus", "providerErrorCode", "reportedModel",
                         "httpRequests", "usage", "estimatedCostUsd", "outputSha256", "pointwise"))

# Outcome of one technical attempt, derived only from its error category and the HTTP sends it made.
UNCERTAIN_CATEGORIES = frozenset({"timeout", "network_error"})  # the request may have been processed
RETRYABLE_CATEGORIES = frozenset({"parse_error", "schema_error", "semantic_error", "incomplete",
                                  "invalid_provider_response"})
HTTP_WAIT_CATEGORIES = frozenset({"rate_limit", "server_error"})
HTTP_REQUEST_KEYS = frozenset(("requestId", "request", "isHttpRetry", "status", "outcome", "retryAfterSeconds",
                               "waitedSeconds", "reservedUsd", "settledUsd"))
FIXED_EXECUTION_KEYS = {"httpRetryMax": "http_retry_max", "retryAfterRequired": "retry_after_required",
                        "maxTechnicalAttempts": "max_technical_attempts",
                        "automaticLogicalRetry": "automatic_logical_retry", "requestLimit": "request_limit"}
EXECUTION_KEYS = frozenset(FIXED_EXECUTION_KEYS) | RUNTIME_KEYS
_ID = re.compile(r"[0-9a-f]{32}", re.ASCII)


class ProductionJudgeUnavailable(ValueError):
    """No usable Judge result for this evaluation; ``reasons`` says why."""

    def __init__(self, reasons):
        self.reasons = tuple(reasons)
        super().__init__("Production Judge result is not available: " + ", ".join(self.reasons))


def valid_retry_after(value):
    """A usable Retry-After wait as the Judge client stores it (seconds); a missing or unparseable header is None."""
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def classify_attempt(category, http_requests):
    """``completed``, ``retryable_failure`` (an explicit next attempt may follow), ``terminal_failure``
    or ``uncertain``, from the error category and the attempt's stored HTTP request records.

    A 429/5xx is retryable only when the attempt used its one HTTP retry and the last response still
    carried a valid Retry-After; a 429/5xx whose last response has no valid Retry-After is terminal.
    Refusals, client errors and guards are terminal; timeouts and network errors are uncertain."""
    if category is None:
        return "completed"
    if category in UNCERTAIN_CATEGORIES:
        return "uncertain"
    if category in RETRYABLE_CATEGORIES:
        return "retryable_failure"
    if (category in HTTP_WAIT_CATEGORIES and isinstance(http_requests, list) and len(http_requests) == 2
            and isinstance(http_requests[-1], dict) and valid_retry_after(http_requests[-1].get("retryAfterSeconds"))):
        return "retryable_failure"
    return "terminal_failure"


def production_contract_sha256(prompt_version):
    """The Production Pointwise contract: the Pilot Pointwise instructions and schema under the
    production prompt version. Pairwise text and schema are deliberately not part of it."""
    return sha256_hex({"promptVersion": prompt_version, "rubricVersion": RUBRIC_VERSION,
                       "schemaVersion": SCHEMA_VERSION, "pointwiseInstructions": POINTWISE_INSTRUCTIONS,
                       "pointwiseSchema": pointwise_schema()})


def validate_production_judge_config(config):
    """The configs/production-judge.yaml contract; raise ValueError otherwise."""
    policy = config.get("execution_policy") if isinstance(config, dict) else None
    if (not isinstance(config, dict) or set(config) != CONFIG_KEYS or config["provider"] != PROVIDER
            or any(not isinstance(config[key], str) or not config[key].strip()
                   for key in ("model", "reasoning", "api_key_environment_variable"))
            or type(config["max_output_tokens"]) is not int or config["max_output_tokens"] < 1
            or config["prompt_version"] not in PROMPT_VERSIONS
            or config["rubric_version"] != RUBRIC_VERSION or config["schema_version"] != SCHEMA_VERSION
            or not isinstance(policy, dict) or set(policy) != set(EXECUTION_POLICY)
            # type() too, so a YAML true never stands in for 1 or the reverse.
            or any(type(policy[key]) is not type(value) or policy[key] != value
                   for key, value in EXECUTION_POLICY.items())):
        raise ValueError("Invalid production Judge configuration")
    return config


def _now():
    return datetime.now(timezone.utc).isoformat()


def _strict_json(data):
    def reject_constant(_value):
        raise ValueError("Non-finite JSON value")
    return json.loads(data, parse_constant=reject_constant)


def _pointwise_list(validated):
    return [dict({"questionIndex": index}, **validated[index]) for index in sorted(validated)]


class ProductionJudgeRunner:
    def __init__(self, repository, results=None, quiz_config=None, judge_config=None):
        self.quiz = ProductionQuizRunner(repository, results, quiz_config)
        config_file = Path(judge_config) if judge_config else self.quiz.repository / "configs" / "production-judge.yaml"
        self.config = validate_production_judge_config(
            yaml.safe_load(config_file.read_text(encoding="utf-8"))["production_judge"])
        self.contract_sha256 = production_contract_sha256(self.config["prompt_version"])

    # Quiz binding and request

    def _bind(self, quiz_operation_id):
        """Judge input, request body and identity re-derived from the stored Quiz artifacts.

        ``require_judge_ready`` is the only eligibility boundary (it raises ProductionQuizNotJudgeReady);
        nothing a caller passes is trusted. The prompt holds only the contentText and question fields.
        """
        judge_input = self.quiz.require_judge_ready(quiz_operation_id)
        attempt = judge_input["attempt"]
        operation = _strict_json(self.quiz._path("operations", quiz_operation_id, "operation.json").read_text(encoding="utf-8"))
        result_bytes = self.quiz._path("operations", quiz_operation_id, "attempts", str(attempt), "result.json").read_bytes()
        result = _strict_json(result_bytes.decode("utf-8"))
        if (operation["inputFingerprint"] != judge_input["inputFingerprint"]
                or result["outputSha256"] != judge_input["quizOutputSha256"]):
            raise ValueError("The Quiz artifacts changed while being read")
        questions = {question["questionIndex"]: {key: question[key] for key in QUESTION_FIELDS}
                     for question in judge_input["questions"]}
        prompt = pointwise_prompt(judge_input["contentText"], questions)
        body = build_request(PROVIDER, self.config["model"], KIND, prompt, pointwise_schema(),
                             self.config["reasoning"], self.config["max_output_tokens"])
        body_bytes = canonical_json(body).encode("utf-8")
        identity = {
            "quiz": {"operationId": quiz_operation_id, "attempt": attempt,
                     "inputFingerprint": judge_input["inputFingerprint"],
                     "quizOutputSha256": judge_input["quizOutputSha256"],
                     "quizResultSha256": hashlib.sha256(result_bytes).hexdigest(),
                     "sourceGroundingRunId": judge_input["sourceGroundingRunId"],
                     "contentTextSha256": judge_input["contentTextSha256"],
                     "sourceSnapshotSha256": judge_input["sourceSnapshotSha256"],
                     "groundingValidationVersion": judge_input["groundingValidationVersion"],
                     "promptVersion": operation["input"]["promptVersion"],
                     "outputContractVersion": operation["input"]["outputContractVersion"]},
            "judge": {"provider": PROVIDER, "model": self.config["model"], "reasoning": self.config["reasoning"],
                      "maxOutputTokens": self.config["max_output_tokens"],
                      "promptVersion": self.config["prompt_version"], "rubricVersion": self.config["rubric_version"],
                      "schemaVersion": self.config["schema_version"], "contractSha256": self.contract_sha256},
            "requestBodySha256": hashlib.sha256(body_bytes).hexdigest()}
        return body, body_bytes, identity, sorted(questions)

    def _runtime_policy(self, runtime):
        """JudgePolicy and recorded execution policy from explicit run values; no defaults exist."""
        price = runtime.get("price") if isinstance(runtime, dict) else None
        if (not isinstance(runtime, dict) or set(runtime) != RUNTIME_KEYS
                or not isinstance(price, dict) or set(price) != PRICE_KEYS):
            raise ValueError("Production Judge needs timeoutSeconds, perCallCostLimitUsd, totalCostLimitUsd, "
                             "estimatedInputTokens and price (input, output, cachedInput, reference, checkedAt)")
        policy = JudgePolicy(live=True, http_request_limit=EXECUTION_POLICY["request_limit"],
                             per_call_cost_limit=runtime["perCallCostLimitUsd"],
                             total_cost_limit=runtime["totalCostLimitUsd"], timeout_seconds=runtime["timeoutSeconds"],
                             estimated_input_tokens={KIND: runtime["estimatedInputTokens"]},
                             max_output_tokens={KIND: self.config["max_output_tokens"]}, prices={PROVIDER: price})
        try:
            policy.authorize(KIND, PROVIDER, 0, 0.0)  # every guard value is present, valid and fits one call
        except JudgeCallFailure:
            raise ValueError("Production Judge runtime guard values are missing, invalid or too small") from None
        execution = {"httpRetryMax": EXECUTION_POLICY["http_retry_max"],
                     "retryAfterRequired": EXECUTION_POLICY["retry_after_required"],
                     "maxTechnicalAttempts": EXECUTION_POLICY["max_technical_attempts"],
                     "automaticLogicalRetry": EXECUTION_POLICY["automatic_logical_retry"],
                     "requestLimit": EXECUTION_POLICY["request_limit"],
                     "timeoutSeconds": runtime["timeoutSeconds"], "perCallCostLimitUsd": runtime["perCallCostLimitUsd"],
                     "totalCostLimitUsd": runtime["totalCostLimitUsd"],
                     "estimatedInputTokens": runtime["estimatedInputTokens"], "price": dict(price)}
        return policy, execution

    def _stored_policy(self, execution):
        """The JudgePolicy of a stored execution policy, validated by the same rules as a new run's; else ValueError."""
        if (not isinstance(execution, dict) or set(execution) != EXECUTION_KEYS
                or any(type(execution[key]) is not type(EXECUTION_POLICY[name]) or execution[key] != EXECUTION_POLICY[name]
                       for key, name in FIXED_EXECUTION_KEYS.items())):
            raise ValueError("stored execution policy is invalid")
        policy, rebuilt = self._runtime_policy({key: execution[key] for key in RUNTIME_KEYS})
        if rebuilt != execution:
            raise ValueError("stored execution policy is invalid")
        return policy

    # Stored attempts

    def _root(self, quiz_operation_id, evaluation_id):
        return ("operations", quiz_operation_id, "judge", evaluation_id)

    def _attempts(self, root):
        """{n: {started, requests: {k: intent}, result, output}} read strictly; ValueError on bad entries."""
        base = self.quiz._path(*root, "attempts")
        attempts = {}
        if not base.exists():
            return attempts
        for entry in base.iterdir():
            if entry.name.startswith("."):
                continue
            if not entry.is_dir() or entry.is_symlink() or not re.fullmatch(r"[1-9][0-9]*", entry.name):
                raise ValueError("unexpected attempt entry")
            read = lambda path: self.quiz._read_json(path) if path.exists() else None
            requests = {}
            if (entry / "requests").exists():
                for item in (entry / "requests").iterdir():
                    if item.name.startswith("."):
                        continue
                    match = re.fullmatch(r"([1-9][0-9]*)\.json", item.name)
                    if match is None or item.is_symlink():
                        raise ValueError("unexpected request entry")
                    requests[int(match.group(1))] = self.quiz._read_json(item)
            output = entry / "output.json"
            record = {"started": read(entry / "started.json"), "requests": requests,
                      "result": read(entry / "result.json"),
                      "output": output.read_bytes() if output.exists() else None}
            if record["started"] is None and not requests and record["result"] is None and record["output"] is None:
                continue  # created before started.json was linked: nothing was sent
            attempts[int(entry.name)] = record
        return attempts

    def _attempt_problem(self, manifest, policy, number, record, indexes):
        """Why this attempt breaks the manifest -> started -> request intents -> output -> result chain;
        ``uncertain`` for an attempt that may have reached the Provider without a usable outcome; else None.

        Reservations and settled costs are recomputed from the stored execution policy and usage with
        the Judge cost rules: an unknown cost must stay ``null`` and keep its reservation."""
        started, requests, result = record["started"], record["requests"], record["result"]
        if started is None:
            return "missingStartedRecord"
        expected = {"evaluationId": manifest["evaluationId"], "attempt": number,
                    "requestBodySha256": manifest["identity"]["requestBodySha256"],
                    "executionPolicySha256": sha256_hex(manifest["executionPolicy"])}
        reservation = policy.estimate(KIND, PROVIDER)
        if (not isinstance(started, dict) or set(started) != STARTED_KEYS or started["format"] != STARTED_FORMAT
                or any(started[key] != value for key, value in expected.items())
                or started["inputFingerprint"] != manifest["inputFingerprint"]
                or not valid_utc_timestamp(started["startedAt"])):
            return "startedRecordMismatch"
        if sorted(requests) != list(range(1, len(requests) + 1)) or len(requests) > 2:
            return "requestJournalMismatch"
        for k, intent in requests.items():
            if (not isinstance(intent, dict) or set(intent) != REQUEST_KEYS or intent["format"] != REQUEST_FORMAT
                    or any(intent[key] != value for key, value in expected.items())
                    or intent["request"] != k or intent["requestId"] != "%s-a%d-r%d" % (manifest["evaluationId"], number, k)
                    or type(intent["reservedUsd"]) is not float or intent["reservedUsd"] != reservation
                    or not valid_utc_timestamp(intent["createdAt"])):
                return "requestJournalMismatch"
        if result is None:
            return "uncertain"  # started or journalled, maybe sent, no recorded outcome
        if (not isinstance(result, dict) or set(result) != RESULT_KEYS or result["format"] != RESULT_FORMAT
                or any(result[key] != value for key, value in expected.items())
                or result["inputFingerprint"] != manifest["inputFingerprint"]
                or result["startedAt"] != started["startedAt"] or not valid_utc_timestamp(result["completedAt"])
                or not isinstance(result["httpRequests"], list) or len(result["httpRequests"]) != len(requests)
                or self._http_log_problem(result, requests, policy)
                or result["outcome"] != classify_attempt(result["errorCategory"], result["httpRequests"])):
            return "resultRecordMismatch"
        output = record["output"]
        if result["outputSha256"] is None:
            if output is not None:
                return "resultRecordMismatch"
        elif output is None or hashlib.sha256(output).hexdigest() != result["outputSha256"]:
            return "outputIntegrity"
        if result["outcome"] == "completed":
            try:
                text = json.loads(output.decode("ascii"))["text"]
                validated = _pointwise_list(validate_pointwise(text, indexes))  # re-parse the stored model text
            except (AttributeError, KeyError, TypeError, UnicodeError, ValueError, JudgeOutputError):
                return "outputIntegrity"
            if result["pointwise"] != validated:
                return "resultRecordMismatch"
        elif result["pointwise"] is not None:
            return "resultRecordMismatch"
        if result["outcome"] == "uncertain":
            return "uncertain"
        return None

    @staticmethod
    def _http_log_problem(result, requests, policy):
        """True when the result's HTTP request records do not match the journal and the client's rules.

        Only the last request can have a response that was read, so only it may be settled, and its
        settled cost must equal estimate_cost() of the stored usage (``null`` when that is unknown).
        A request followed by an HTTP retry must be a 429/5xx with a valid Retry-After that was waited."""
        log = result["httpRequests"]
        for k, sent in enumerate(log, 1):
            last = k == len(log)
            if not isinstance(sent, dict) or set(sent) != HTTP_REQUEST_KEYS:
                return True
            settled = None
            if last and sent["outcome"] == "response" and type(sent["status"]) is int and 200 <= sent["status"] < 300:
                settled = estimate_cost(PROVIDER, result["usage"], policy.prices.get(PROVIDER))
            if (sent["requestId"] != requests[k]["requestId"] or sent["request"] != k
                    or sent["reservedUsd"] != requests[k]["reservedUsd"] or sent["isHttpRetry"] is not (k > 1)
                    or (sent["settledUsd"] is None) != (settled is None)
                    or (settled is not None and (type(sent["settledUsd"]) is not float or sent["settledUsd"] != settled))):
                return True
            if not last and (sent["outcome"] != "response"
                             or (sent["status"] != 429 and sent["status"] not in range(500, 600))
                             or not valid_retry_after(sent["retryAfterSeconds"])
                             or sent["waitedSeconds"] != sent["retryAfterSeconds"]):
                return True
            if last and sent["waitedSeconds"] is not None:
                return True
        if log and result["httpStatus"] != log[-1]["status"]:
            return True
        last_settled = log[-1]["settledUsd"] if log else None
        return result["estimatedCostUsd"] != last_settled or type(result["estimatedCostUsd"]) is not type(last_settled)

    def _state(self, manifest, attempts, indexes):
        """not_started | retryable | exhausted | completed | terminal | uncertain | corrupted."""
        numbers = sorted(attempts)
        if numbers != list(range(1, len(numbers) + 1)) or len(numbers) > EXECUTION_POLICY["max_technical_attempts"]:
            return "corrupted"
        try:
            policy = self._stored_policy(manifest["executionPolicy"])
        except ValueError:
            return "corrupted"
        problems = [self._attempt_problem(manifest, policy, n, attempts[n], indexes) for n in numbers]
        if any(problem not in (None, "uncertain") for problem in problems):
            return "corrupted"
        if "uncertain" in problems:
            return "uncertain"
        outcomes = [attempts[n]["result"]["outcome"] for n in numbers]
        if any(outcome != "retryable_failure" for outcome in outcomes[:-1]):
            return "corrupted"  # only a retryable failure may be followed by another attempt
        if not outcomes:
            return "not_started"
        last = outcomes[-1]
        if last == "completed":
            return "completed"
        if last == "terminal_failure":
            return "terminal"
        return "exhausted" if len(outcomes) >= EXECUTION_POLICY["max_technical_attempts"] else "retryable"

    @staticmethod
    def _used_budget(attempts):
        """HTTP sends and reserved/settled USD from the journal; unknown costs keep their reservation."""
        requests, cost = 0, 0.0
        for record in attempts.values():
            sent = record["result"]["httpRequests"] if record["result"] else []
            for k, intent in record["requests"].items():
                requests += 1
                settled = sent[k - 1].get("settledUsd") if len(sent) >= k else None
                cost += settled if settled is not None else intent["reservedUsd"]
        return requests, cost

    def _load(self, quiz_operation_id, evaluation_id):
        if not isinstance(evaluation_id, str) or _ID.fullmatch(evaluation_id) is None:
            raise ValueError("A production Judge evaluationId is 32 lowercase hex characters")
        body, body_bytes, identity, indexes = self._bind(quiz_operation_id)
        return body, body_bytes, identity, indexes, self._root(quiz_operation_id, evaluation_id)

    # Evaluation

    def evaluate(self, quiz_operation_id, evaluation_id, runtime, transport=None, sleep=None, api_keys=None):
        """Run one technical attempt of a Pointwise evaluation and return its result record.

        Refused before any write or HTTP request unless the Quiz is Judge-ready, the runtime guards are
        valid and an API key is available. A resumed evaluation must keep the same Quiz artifact,
        Judge settings, request and execution policy. Only a verified retryable failure allows the
        next (explicit) attempt; completed, terminal, uncertain and exhausted evaluations are not run.
        """
        policy, execution = self._runtime_policy(runtime)
        key = api_keys.get(PROVIDER) if api_keys is not None else os.getenv(self.config["api_key_environment_variable"])
        if not key:
            raise ValueError("The OpenAI API key environment variable is not set")
        body, body_bytes, identity, indexes, root = self._load(quiz_operation_id, evaluation_id)
        manifest = {"format": MANIFEST_FORMAT, "evaluationId": evaluation_id, "quizOperationId": quiz_operation_id,
                    "inputFingerprint": sha256_hex(identity), "identity": identity, "executionPolicy": execution}
        manifest_path = self.quiz._path(*root, "evaluation.json")
        if not self.quiz._create_once(manifest_path, canonical_json(manifest).encode("utf-8")):
            if self.quiz._read_json(manifest_path) != manifest:
                raise ValueError("This evaluation was created for a different Quiz artifact, Judge config, request "
                                 "or execution policy; it is never reused")
        request_path = self.quiz._path(*root, "request.json")
        if not self.quiz._create_once(request_path, body_bytes) and request_path.read_bytes() != body_bytes:
            raise ValueError("Production Judge storage integrity error: stored request body differs")

        attempts = self._attempts(root)
        state = self._state(manifest, attempts, indexes)
        if state not in ("not_started", "retryable"):
            raise ValueError("This evaluation is %s; no further request is sent" % state)
        attempt = len(attempts) + 1
        requests_used, reserved = self._used_budget(attempts)
        try:
            policy.authorize(KIND, PROVIDER, requests_used, reserved)
        except JudgeCallFailure:
            raise ValueError("The evaluation's request limit or cost cap leaves no room for another request") from None
        client = JudgeHttpClient(policy, {PROVIDER: self.config["api_key_environment_variable"]}, transport=transport,
                                 sleep=sleep, api_keys=api_keys)
        client.restore_usage(requests_used, reserved)

        attempt_root = root + ("attempts", str(attempt))
        policy_sha = sha256_hex(execution)
        started = {"format": STARTED_FORMAT, "evaluationId": evaluation_id, "attempt": attempt,
                   "inputFingerprint": manifest["inputFingerprint"],
                   "requestBodySha256": identity["requestBodySha256"], "executionPolicySha256": policy_sha,
                   "startedAt": _now()}
        if not self.quiz._create_once(self.quiz._path(*attempt_root, "started.json"), canonical_json(started).encode("utf-8")):
            raise ValueError("Another process started this attempt")

        def before_send(intent):
            record = {"format": REQUEST_FORMAT, "evaluationId": evaluation_id, "attempt": attempt,
                      "request": intent["request"], "requestId": intent["requestId"],
                      "requestBodySha256": identity["requestBodySha256"], "executionPolicySha256": policy_sha,
                      "reservedUsd": intent["reservedUsd"], "createdAt": _now()}
            path = self.quiz._path(*attempt_root, "requests", "%d.json" % intent["request"])
            if not self.quiz._create_once(path, canonical_json(record).encode("utf-8")):
                raise ValueError("Production Judge storage integrity error: request intent already exists")

        http_status = provider_error_code = None
        pointwise = None
        try:
            response = client.call(PROVIDER, KIND, body, attempt_id="%s-a%d" % (evaluation_id, attempt),
                                   before_send=before_send)
            category, text = None, response["text"]
            log, usage, cost, reported = (response["httpRequests"], response["usage"],
                                          response["estimatedCostUsd"], response["reportedModel"])
            http_status = log[-1]["status"] if log else None
            try:
                pointwise = _pointwise_list(validate_pointwise(text, indexes))
            except JudgeOutputError as failure:
                category = failure.category  # a usable answer never arrived: the text is kept for diagnosis
        except JudgeCallFailure as failure:
            category, text, log, usage, cost, reported = (failure.category, failure.raw_text, failure.http_requests,
                                                          failure.usage, failure.estimated_cost_usd, failure.reported_model)
            http_status, provider_error_code = failure.http_status, failure.provider_error_code
            if http_status is None and log:
                http_status = log[-1]["status"]
        output_sha = None
        if text is not None:
            output_bytes = json.dumps({"text": text}, ensure_ascii=True, sort_keys=True).encode("ascii")
            if not self.quiz._create_once(self.quiz._path(*attempt_root, "output.json"), output_bytes):
                raise ValueError("Production Judge storage integrity error: attempt output already exists")
            output_sha = hashlib.sha256(output_bytes).hexdigest()
        result = {"format": RESULT_FORMAT, "evaluationId": evaluation_id, "attempt": attempt,
                  "inputFingerprint": manifest["inputFingerprint"], "requestBodySha256": identity["requestBodySha256"],
                  "executionPolicySha256": policy_sha, "startedAt": started["startedAt"], "completedAt": _now(),
                  "outcome": classify_attempt(category, log), "errorCategory": category,
                  "httpStatus": http_status, "providerErrorCode": provider_error_code, "reportedModel": reported,
                  "httpRequests": log, "usage": usage, "estimatedCostUsd": cost, "outputSha256": output_sha,
                  "pointwise": pointwise}
        # The result record is the completion marker: written only after the output it names.
        if not self.quiz._create_once(self.quiz._path(*attempt_root, "result.json"), canonical_json(result).encode("utf-8")):
            raise ValueError("Production Judge storage integrity error: attempt result already exists")
        return result

    # Reading

    def evaluation_state(self, quiz_operation_id, evaluation_id):
        """The evaluation state re-derived from storage: ``quiz_not_judge_ready`` when the Quiz no longer
        passes require_judge_ready, ``corrupted`` when the evaluation records do not verify."""
        try:
            _, body_bytes, identity, indexes, root = self._load(quiz_operation_id, evaluation_id)
        except ProductionQuizNotJudgeReady:
            return "quiz_not_judge_ready"
        except (KeyError, OSError, TypeError, UnicodeError, ValueError):
            return "corrupted"
        try:
            manifest = self.quiz._read_json(self.quiz._path(*root, "evaluation.json"))
        except FileNotFoundError:
            return "not_started"
        except (KeyError, TypeError, UnicodeError, ValueError):
            return "corrupted"
        try:
            if (set(manifest) != {"format", "evaluationId", "quizOperationId", "inputFingerprint", "identity",
                                  "executionPolicy"}
                    or manifest["format"] != MANIFEST_FORMAT or manifest["evaluationId"] != evaluation_id
                    or manifest["quizOperationId"] != quiz_operation_id
                    or manifest["identity"] != identity or manifest["inputFingerprint"] != sha256_hex(identity)
                    or self.quiz._path(*root, "request.json").read_bytes() != body_bytes):
                return "corrupted"
            self._stored_policy(manifest["executionPolicy"])  # every runtime value, by the new-run rules
            return self._state(manifest, self._attempts(root), indexes)
        except (AttributeError, KeyError, OSError, TypeError, UnicodeError, ValueError):
            return "corrupted"

    def require_completed(self, quiz_operation_id, evaluation_id):
        """Return the completed Pointwise result, or raise ProductionJudgeUnavailable.

        Re-checks that the Quiz is still Judge-ready and unchanged, that the stored manifest and
        request match what would be built now, the whole attempt record chain, and re-validates the
        stored model text. Per-question verdicts only; no overall PASS/FAIL.
        """
        try:
            _, body_bytes, identity, indexes, root = self._load(quiz_operation_id, evaluation_id)
        except ValueError as exc:  # includes ProductionQuizNotJudgeReady
            raise ProductionJudgeUnavailable(["quizNotJudgeReady: %s" % exc]) from None
        state = self.evaluation_state(quiz_operation_id, evaluation_id)
        if state != "completed":
            raise ProductionJudgeUnavailable(["evaluationState:" + state])
        manifest = self.quiz._read_json(self.quiz._path(*root, "evaluation.json"))
        attempts = self._attempts(root)
        number = max(attempts)
        result = attempts[number]["result"]
        return {"evaluationId": evaluation_id, "quizOperationId": quiz_operation_id, "attempt": number,
                "inputFingerprint": manifest["inputFingerprint"], "identity": identity,
                "outputSha256": result["outputSha256"], "estimatedCostUsd": result["estimatedCostUsd"],
                "questions": result["pointwise"]}
