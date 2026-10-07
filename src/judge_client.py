"""Judge-only Provider access, separate from the Pilot ProviderRouter.

One ``call`` is one technical attempt: up to two HTTP requests (one retry for 429/5xx).
A retry waits for an official retry hint (``Retry-After``). Without a hint the wait policy is
not decided yet, so the attempt fails with ``retryWaitUnresolved`` and the runner stops the
run instead of retrying immediately. This is a temporary fail-closed state, not a protocol rule.
"""

import email.utils
import json
import math
import os
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.provider_adapters import GEMINI_URL, OPENAI_URL, _gemini_http_error_code


RUN_STOP_CATEGORIES = frozenset({"client_error", "api_key_missing", "live_guard"})
WAIT_CATEGORIES = frozenset({"rate_limit", "server_error"})


class JudgeCallFailure(Exception):
    """A failed technical attempt. ``stop_run`` marks failures that must stop the whole run."""

    def __init__(self, category, http_status=None, provider_error_code=None, usage=None,
                 http_requests=None, retry_after=None, reported_model=None, raw_text=None,
                 retry_wait_unresolved=False):
        super().__init__(category)
        self.category = category
        self.http_status = http_status
        self.provider_error_code = provider_error_code
        self.usage = usage
        self.http_requests = http_requests or []
        self.retry_after = retry_after
        self.reported_model = reported_model
        self.raw_text = raw_text
        self.retry_wait_unresolved = retry_wait_unresolved
        self.estimated_cost_usd = None
        self.stop_run = category in RUN_STOP_CATEGORIES


def _valid_cached_price(value):
    """A cached-input price is optional; when given it must be a finite, non-negative number."""
    return value is None or (type(value) in (int, float) and math.isfinite(value) and value >= 0)


@dataclass
class JudgePolicy:
    """Explicit live guards. Prices come from the operator with their source and check date."""

    live: bool = False
    http_request_limit: int = None
    per_call_cost_limit: float = None
    total_cost_limit: float = None
    timeout_seconds: float = None
    estimated_input_tokens: dict = field(default_factory=dict)
    max_output_tokens: dict = field(default_factory=dict)
    # provider -> {"input": float, "output": float, "cachedInput": float|None,
    #              "reference": str, "checkedAt": str}
    prices: dict = field(default_factory=dict)

    def estimate(self, kind, provider):
        price = self.prices.get(provider) or {}
        tokens_in = self.estimated_input_tokens.get(kind)
        tokens_out = self.max_output_tokens.get(kind)
        return (tokens_in * price["input"] + tokens_out * price["output"]) / 1000000

    def authorize(self, kind, provider, requests_made, reserved_cost):
        price = self.prices.get(provider)
        numbers = (self.per_call_cost_limit, self.total_cost_limit, self.timeout_seconds)
        tokens_in = self.estimated_input_tokens.get(kind)
        tokens_out = self.max_output_tokens.get(kind)
        # type() checks also reject bool, which is an int subclass.
        if (self.live is not True or type(self.http_request_limit) is not int or self.http_request_limit < 1
                or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                       for value in numbers)
                or type(tokens_in) is not int or tokens_in < 0
                or type(tokens_out) is not int or tokens_out < 1
                or not isinstance(price, dict)
                or any(type(price.get(key)) not in (int, float) or not math.isfinite(price[key]) or price[key] <= 0
                       for key in ("input", "output"))
                or not _valid_cached_price(price.get("cachedInput"))
                or not isinstance(price.get("reference"), str) or not price["reference"].strip()
                or not isinstance(price.get("checkedAt"), str) or not price["checkedAt"].strip()):
            raise JudgeCallFailure("live_guard")
        try:
            estimate = self.estimate(kind, provider)
        except OverflowError:
            raise JudgeCallFailure("live_guard")
        if (not math.isfinite(estimate) or estimate < 0 or requests_made >= self.http_request_limit
                or estimate > self.per_call_cost_limit or reserved_cost + estimate > self.total_cost_limit):
            raise JudgeCallFailure("live_guard")
        return estimate


def build_request(provider, model, kind, prompt, schema, reasoning, max_output_tokens):
    """Provider request body. Its canonical JSON is the measurement inputHash."""
    if type(max_output_tokens) is not int or max_output_tokens < 1:
        raise ValueError("max_output_tokens must be decided before building a Judge request")
    if provider == "openai":
        return {"model": model, "input": prompt, "store": False, "reasoning": {"effort": reasoning},
                "max_output_tokens": max_output_tokens,
                "text": {"format": {"type": "json_schema", "name": "judge_" + kind, "strict": True,
                                    "schema": schema}}}
    if provider == "gemini":
        return {"model": model, "input": [{"type": "text", "text": prompt}], "store": False,
                "response_format": {"type": "text", "mime_type": "application/json", "schema": schema},
                "generation_config": {"max_output_tokens": max_output_tokens, "thinking_level": reasoning}}
    raise ValueError("Unknown Judge provider")


def parse_retry_after(value, now=None):
    """Seconds from a Retry-After header (delta-seconds or HTTP-date); None if absent/invalid."""
    if value is None:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        moment = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None
    if moment is None:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return max(0.0, (moment - (now or datetime.now(timezone.utc))).total_seconds())


def judge_transport(url, headers, body, timeout):
    """Return (status, body, retryAfterSeconds). Only Retry-After is read; headers are never kept."""
    request = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers,
                                     method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8")), None
    except urllib.error.HTTPError as error:
        retry_after = parse_retry_after(error.headers.get("Retry-After") if error.headers else None)
        code = _gemini_http_error_code(error) if url == GEMINI_URL else None
        return error.code, ({"providerErrorCode": code} if url == GEMINI_URL else {}), retry_after


def _int_or_none(value):
    return value if type(value) is int and value >= 0 else None


def _usage_field(container, key, name, invalid):
    """Non-negative int, or None. A present but malformed value is recorded in ``invalid``."""
    value = container.get(key) if isinstance(container, dict) else None
    parsed = _int_or_none(value)
    if value is not None and parsed is None:
        invalid.append(name)
    return parsed


def _usage_details(container, key, name, invalid):
    """A nested usage-details object. Absent (or null) means no detail was provided; a present
    non-object value is malformed and recorded in ``invalid`` so its cache usage is never trusted."""
    value = container.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        invalid.append(name)
        return {}
    return value


def _model_texts(provider, response):
    """(refused, texts) from the allowed model-output text fields only."""
    if provider == "openai":
        items = [item for output in response["output"] if output.get("type") == "message"
                 for item in output["content"]]
        return (any(item.get("type") == "refusal" for item in items),
                [item["text"] for item in items if item.get("type") == "output_text"])
    return False, [item["text"] for step in response["steps"] if step.get("type") == "model_output"
                   for item in step["content"] if item.get("type") == "text"]


def parse_response(provider, response):
    """Return {text, reportedModel, usage}; raise JudgeCallFailure for incomplete/refusal/invalid."""
    if not isinstance(response, dict):
        raise JudgeCallFailure("invalid_provider_response")
    usage_source = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    invalid = []
    if provider == "openai":
        details_in = _usage_details(usage_source, "input_tokens_details", "inputTokensDetails", invalid)
        details_out = _usage_details(usage_source, "output_tokens_details", "outputTokensDetails", invalid)
        usage = {"inputTokens": _usage_field(usage_source, "input_tokens", "inputTokens", invalid),
                 "cachedInputTokens": _usage_field(details_in, "cached_tokens", "cachedInputTokens", invalid),
                 "outputTokens": _usage_field(usage_source, "output_tokens", "outputTokens", invalid),
                 "reasoningTokens": _usage_field(details_out, "reasoning_tokens", "reasoningTokens", invalid)}
    else:
        # Interactions API: total_cached_tokens is "the cached part of the prompt", so it cannot exceed
        # total_input_tokens (checked below).
        usage = {"inputTokens": _usage_field(usage_source, "total_input_tokens", "inputTokens", invalid),
                 "cachedInputTokens": _usage_field(usage_source, "total_cached_tokens", "cachedInputTokens", invalid),
                 "outputTokens": _usage_field(usage_source, "total_output_tokens", "outputTokens", invalid),
                 "reasoningTokens": _usage_field(usage_source, "total_thought_tokens", "reasoningTokens", invalid)}
    if (usage["cachedInputTokens"] is not None and usage["inputTokens"] is not None
            and usage["cachedInputTokens"] > usage["inputTokens"]):
        invalid.append("cachedInputTokensExceedInput")
    if invalid:
        usage["invalidFields"] = invalid
    model = response.get("model") if isinstance(response.get("model"), str) else None
    if response.get("status") != "completed":
        # Keep only the allowed model text fields as a diagnostic; this is never a usable result.
        try:
            partial = [text for text in _model_texts(provider, response)[1] if isinstance(text, str)]
        except (AttributeError, KeyError, TypeError):
            partial = []
        raise JudgeCallFailure("incomplete", usage=usage, reported_model=model,
                               raw_text="\n".join(partial) if partial else None)
    try:
        refused, texts = _model_texts(provider, response)
    except (AttributeError, KeyError, TypeError):
        raise JudgeCallFailure("invalid_provider_response", usage=usage, reported_model=model)
    if refused:
        raise JudgeCallFailure("refusal", usage=usage, reported_model=model)
    if len(texts) != 1 or not isinstance(texts[0], str):
        raise JudgeCallFailure("invalid_provider_response", usage=usage, reported_model=model)
    return {"text": texts[0], "reportedModel": model, "usage": usage}


def estimate_cost(provider, usage, price):
    """USD estimate from returned usage and operator prices.

    Returns None (unknown, never zero) when usage or prices are missing or malformed, cached tokens
    exceed input tokens, or the result is not a finite non-negative number.
    """
    if (not usage or not price or usage.get("invalidFields")
            or usage.get("inputTokens") is None or usage.get("outputTokens") is None):
        return None
    if (any(type(price.get(key)) not in (int, float) or not math.isfinite(price[key]) or price[key] < 0
            for key in ("input", "output")) or not _valid_cached_price(price.get("cachedInput"))):
        return None
    try:
        if provider == "openai":
            cached = usage.get("cachedInputTokens") or 0
            if cached > usage["inputTokens"] or (cached and price.get("cachedInput") is None):
                return None
            # OpenAI output_tokens already include reasoning tokens.
            cost = ((usage["inputTokens"] - cached) * price["input"] + cached * (price.get("cachedInput") or 0)
                    + usage["outputTokens"] * price["output"]) / 1000000
        else:
            # The official docs do not state whether cached tokens are part of total_input_tokens or
            # how the cached price applies to implicit caching, so a cached call has no known cost.
            if usage.get("cachedInputTokens"):
                return None
            # Gemini bills thinking tokens as output in addition to output tokens.
            cost = (usage["inputTokens"] * price["input"]
                    + (usage["outputTokens"] + (usage.get("reasoningTokens") or 0)) * price["output"]) / 1000000
    except OverflowError:
        return None
    return cost if math.isfinite(cost) and cost >= 0 else None


class JudgeHttpClient:
    is_actual_api = True

    def __init__(self, policy, api_key_variables, transport=None, sleep=None, api_keys=None):
        self.policy = policy
        self.api_key_variables = api_key_variables
        self.transport = transport or judge_transport
        self.sleep = sleep or time.sleep
        self.api_keys = api_keys
        self.requests_made = 0
        self.reserved_cost = 0.0

    def _key(self, provider):
        if self.api_keys is not None:
            return self.api_keys.get(provider)
        return os.getenv(self.api_key_variables.get(provider, ""))

    def restore_usage(self, requests_made, reserved_cost):
        """Carry a resumed run's persisted usage so run-level guards never restart from zero."""
        if self.requests_made or self.reserved_cost:
            raise ValueError("Run usage can only be restored on a client that has not sent requests")
        self.requests_made = requests_made
        self.reserved_cost = reserved_cost

    def call_measurement(self, measurement, body, attempt_id=None, before_send=None):
        return self.call(measurement["provider"], measurement["kind"], body, attempt_id, before_send)

    def call(self, provider, kind, body, attempt_id=None, before_send=None):
        """``before_send(intent)`` must durably record each request's reservation; it runs before the
        transport is entered, so a request is never sent without a persisted reservation."""
        url = OPENAI_URL if provider == "openai" else GEMINI_URL
        key = self._key(provider)
        log = []
        for http_try in range(2):
            try:
                estimate = self.policy.authorize(kind, provider, self.requests_made, self.reserved_cost)
                if not key:
                    raise JudgeCallFailure("api_key_missing")
            except JudgeCallFailure as failure:
                # Requests already sent in this attempt stay on the record when a retry is refused.
                failure.http_requests = log
                raise
            headers = {"Content-Type": "application/json"}
            headers["Authorization" if provider == "openai" else "x-goog-api-key"] = (
                "Bearer " + key if provider == "openai" else key)
            self.requests_made += 1
            self.reserved_cost += estimate
            # Created before the transport call so every request actually sent is traceable. reservedUsd is
            # this request's pre-call reservation; settledUsd replaces it only once its cost is known.
            request_id = "%s-r%d" % (attempt_id, http_try + 1) if attempt_id else None
            entry = {"requestId": request_id, "request": http_try + 1, "isHttpRetry": http_try > 0,
                     "status": None, "outcome": "sent", "retryAfterSeconds": None, "waitedSeconds": None,
                     "reservedUsd": estimate, "settledUsd": None}
            log.append(entry)
            if before_send is not None:
                # Durable reservation first; if this write fails the request is never sent.
                before_send({"requestId": request_id, "request": http_try + 1, "provider": provider,
                             "kind": kind, "reservedUsd": estimate})
            try:
                status, response, retry_after = self.transport(url, headers, body, self.policy.timeout_seconds)
            except (TimeoutError, socket.timeout):
                entry["outcome"] = "timeout"
                raise JudgeCallFailure("timeout", http_requests=log)
            except urllib.error.URLError as error:
                category = "timeout" if isinstance(error.reason, (TimeoutError, socket.timeout)) else "network_error"
                entry["outcome"] = category
                raise JudgeCallFailure(category, http_requests=log)
            except OSError:
                entry["outcome"] = "network_error"
                raise JudgeCallFailure("network_error", http_requests=log)
            except (ValueError, TypeError):
                entry["outcome"] = "invalid_provider_response"
                raise JudgeCallFailure("invalid_provider_response", http_requests=log)
            if type(status) is not int:
                entry["outcome"] = "invalid_provider_response"
                raise JudgeCallFailure("invalid_provider_response", http_requests=log)
            code = response.get("providerErrorCode") if isinstance(response, dict) else None
            entry.update(status=status, outcome="response", retryAfterSeconds=retry_after)
            if status == 429 or 500 <= status < 600:
                category = "rate_limit" if status == 429 else "server_error"
                if http_try == 0 and retry_after is not None:
                    self.sleep(retry_after)
                    entry["waitedSeconds"] = retry_after
                    continue
                raise JudgeCallFailure(category, status, code, http_requests=log, retry_after=retry_after,
                                       retry_wait_unresolved=retry_after is None)
            if 400 <= status < 500:
                raise JudgeCallFailure("client_error", status, code, http_requests=log)
            if not 200 <= status < 300:
                raise JudgeCallFailure("provider_error", status, code, http_requests=log)
            try:
                result = parse_response(provider, response)
            except JudgeCallFailure as failure:
                failure.http_requests = log
                failure.estimated_cost_usd = entry["settledUsd"] = self._settle(provider, failure.usage, estimate)
                raise
            result["httpRequests"] = log
            result["estimatedCostUsd"] = entry["settledUsd"] = self._settle(provider, result["usage"], estimate)
            return result
        raise JudgeCallFailure("provider_error", http_requests=log)

    def _settle(self, provider, usage, estimate):
        # An unknown cost keeps the full pre-call reservation; it is never settled as zero.
        cost = estimate_cost(provider, usage, self.policy.prices.get(provider))
        if cost is not None:
            self.reserved_cost += cost - estimate
        return cost


class JudgeFixtureClient:
    """Offline client returning scripted attempt outcomes per logical measurement, in order.

    An outcome is either ``{"text": str, "usage": {...}?, "reportedModel": str?}`` or
    ``{"failure": category, "retryAfter": seconds?, "rawText": str?}``.
    """

    is_actual_api = False

    def __init__(self, script, default=None):
        self.script = {key: list(value) for key, value in script.items()}
        self.default = default
        self.calls = []

    def call_measurement(self, measurement, body, attempt_id=None, before_send=None):
        # No HTTP request is made, so there is no reservation to journal.
        self.calls.append(measurement["logicalMeasurementId"])
        queue = self.script.get(measurement["logicalMeasurementId"])
        if queue is None:
            queue = self.script.get(measurement.get("fixtureKey"))
        outcome = queue.pop(0) if queue else (self.default(measurement) if self.default else None)
        if outcome is None:
            raise JudgeCallFailure("fixture_missing")
        if "failure" in outcome:
            raise JudgeCallFailure(outcome["failure"], retry_after=outcome.get("retryAfter"),
                                   raw_text=outcome.get("rawText"),
                                   retry_wait_unresolved=(outcome["failure"] in WAIT_CATEGORIES
                                                          and outcome.get("retryAfter") is None))
        return {"text": outcome["text"], "reportedModel": outcome.get("reportedModel"),
                "usage": outcome.get("usage"), "httpRequests": [], "estimatedCostUsd": None}
