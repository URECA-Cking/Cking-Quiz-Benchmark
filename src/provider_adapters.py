"""Explicitly gated Pilot HTTP adapters. Responses are allowlisted before persistence."""

import json
import math
import os
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.provider_failure import ProviderFailure


GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
OPENAI_URL = "https://api.openai.com/v1/responses"


def _string():
    return {"type": "string"}


def quiz_schema():
    question = {"type": "object", "additionalProperties": False,
                "properties": {"question": _string(), "options": {"type": "array", "items": _string()},
                               "correctOptionIndex": {"type": "integer"},
                               "explanation": _string(), "sourceEvidence": _string()},
                "required": ["question", "options", "correctOptionIndex", "explanation", "sourceEvidence"]}
    return {"type": "object", "additionalProperties": False,
            "properties": {"promptVersion": _string(), "questions": {"type": "array", "items": question}},
            "required": ["promptVersion", "questions"]}


def grounding_schema():
    fact = {"type": "object", "additionalProperties": False,
            "properties": {"fact": _string(), "evidenceType": _string(), "evidence": _string(),
                           "timestampStartSeconds": {"type": ["number", "null"]},
                           "timestampEndSeconds": {"type": ["number", "null"]}},
            "required": ["fact", "evidenceType", "evidence", "timestampStartSeconds", "timestampEndSeconds"]}
    return {"type": "object", "additionalProperties": False,
            "properties": {"contentText": _string(), "facts": {"type": "array", "items": fact}},
            "required": ["contentText", "facts"]}


@dataclass
class LivePolicy:
    enabled: bool = False
    call_limit: int = None
    per_call_cost_limit: float = None
    total_cost_limit: float = None
    timeout_seconds: float = None
    retry_attempts: int = None
    reasoning_effort: str = None
    video_processing: str = None
    max_output_tokens: int = None
    video_estimated_input_tokens: int = None
    quiz_estimated_input_tokens: int = None
    input_price_per_million: float = None
    output_price_per_million: float = None
    pricing_reference: str = None
    provider_prices: dict = None

    def prices(self, provider):
        if self.provider_prices is not None:
            return self.provider_prices.get(provider, (None, None, None))
        return (self.input_price_per_million, self.output_price_per_million,
                self.pricing_reference)

    def authorize(self, kind, calls, reserved_cost, provider):
        input_price, output_price, reference = self.prices(provider)
        if kind in ("grounding", "direct"):
            estimated_input_tokens = self.video_estimated_input_tokens
        elif kind == "quiz":
            estimated_input_tokens = self.quiz_estimated_input_tokens
        else:
            raise ProviderFailure("live_guard")
        values = (self.per_call_cost_limit, self.total_cost_limit, self.timeout_seconds,
                  input_price, output_price)
        if (self.enabled is not True or type(self.call_limit) is not int or self.call_limit < 1
                or type(self.retry_attempts) is not int or self.retry_attempts < 0
                or type(self.max_output_tokens) is not int or self.max_output_tokens < 1
                or type(estimated_input_tokens) is not int or estimated_input_tokens < 1
                or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values)
                or not isinstance(reference, str) or not reference.strip()
                or (provider == "openai" and self.reasoning_effort not in
                    ("none", "low", "medium", "high", "xhigh"))
                or (kind in ("grounding", "direct") and self.video_processing not in ("static", "agentic"))):
            raise ProviderFailure("live_guard")
        estimate = (estimated_input_tokens * input_price
                    + self.max_output_tokens * output_price) / 1000000
        if (calls >= self.call_limit or estimate > self.per_call_cost_limit
                or reserved_cost + estimate > self.total_cost_limit):
            raise ProviderFailure("live_guard")
        return estimate


def urllib_transport(url, headers, body, timeout):
    request = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers,
                                     method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return error.code, {}


class ProviderRouter:
    is_actual_api = True

    def __init__(self, policy, transport=None, api_keys=None, pilot_config=None):
        self.policy = policy
        self.transport = transport or urllib_transport
        self.api_keys = api_keys
        if pilot_config is None:
            config_path = Path(__file__).resolve().parents[1] / "configs" / "pilot.yaml"
            pilot_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["pilot"]
        self.models = self._model_registry(pilot_config)
        self.calls = 0
        self.reserved_cost = 0.0

    @staticmethod
    def _model_registry(config):
        registry = {}
        def register(model, provider, key_name, kind):
            if (provider not in ("gemini", "openai") or not isinstance(key_name, str)
                    or not key_name or not isinstance(model, str) or not model):
                raise ValueError("Invalid Pilot provider configuration")
            entry = registry.setdefault(model, {"provider": provider, "key_name": key_name,
                                                "kinds": set()})
            if entry["provider"] != provider or entry["key_name"] != key_name:
                raise ValueError("Conflicting Pilot model configuration")
            entry["kinds"].add(kind)
        for method in config["video_grounding"]["methods"]:
            if "model" in method:
                register(method["model"], method["provider"],
                         method["api_key_environment_variable"], "grounding")
        for model in config["quiz_generation"]["models"]:
            register(model["id"], model["provider"],
                     model["api_key_environment_variable"], "quiz")
        for method in config["end_to_end"]["methods"]:
            if method["grounding"] == "direct_video":
                model = method["quiz_model"]
                if model not in registry or registry[model]["provider"] != "gemini":
                    raise ValueError("Direct video requires a configured Gemini model")
                registry[model]["kinds"].add("direct")
        return registry

    def invoke(self, kind, **kwargs):
        model = kwargs.get("model")
        entry = self.models.get(model)
        if entry is None or kind not in entry["kinds"]:
            raise ProviderFailure("unsupported_provider")
        provider = entry["provider"]
        estimate = self.policy.authorize(kind, self.calls, self.reserved_cost, provider)
        key = (self.api_keys or {}).get(provider) if self.api_keys is not None else os.getenv(
            entry["key_name"])
        if not key:
            raise ProviderFailure("api_key_missing")
        body = self._request(provider, kind, **kwargs)
        url = GEMINI_URL if provider == "gemini" else OPENAI_URL
        headers = {"Content-Type": "application/json"}
        headers["x-goog-api-key" if provider == "gemini" else "Authorization"] = (
            key if provider == "gemini" else "Bearer " + key)
        status = None
        last_http_failure = None
        for attempt in range(self.policy.retry_attempts + 1):
            # Count every HTTP attempt, including retries, against the explicit caps.
            try:
                self.policy.authorize(kind, self.calls, self.reserved_cost, provider)
            except ProviderFailure:
                if last_http_failure is not None:
                    raise last_http_failure
                raise
            self.calls += 1
            self.reserved_cost += estimate
            try:
                status, response = self.transport(url, headers, body, self.policy.timeout_seconds)
            except (TimeoutError, socket.timeout):
                raise ProviderFailure("timeout")
            except urllib.error.URLError as error:
                if isinstance(error.reason, (TimeoutError, socket.timeout)):
                    raise ProviderFailure("timeout")
                raise ProviderFailure("network_error")
            except OSError:
                raise ProviderFailure("network_error")
            except (ValueError, TypeError):
                raise ProviderFailure("invalid_provider_response")
            if type(status) is not int:
                raise ProviderFailure("invalid_provider_response")
            if status == 429:
                category = "rate_limit"
            elif 500 <= status < 600:
                category = "server_error"
            elif 400 <= status < 500:
                raise ProviderFailure("client_error", status)
            elif not 200 <= status < 300:
                raise ProviderFailure("provider_error", status)
            else:
                result = self._extract(provider, response)
                input_tokens, output_tokens = result["inputTokens"], result["outputTokens"]
                if input_tokens is not None and output_tokens is not None:
                    input_price, output_price, reference = self.policy.prices(provider)
                    billed_output = output_tokens + (result["thinkingTokens"] or 0) if provider == "gemini" else output_tokens
                    result["estimatedCostUsd"] = (input_tokens * input_price
                                                  + billed_output * output_price) / 1000000
                    result["pricingReference"] = reference
                    self.reserved_cost += result["estimatedCostUsd"] - estimate
                return result
            if attempt == self.policy.retry_attempts:
                raise ProviderFailure(category, status)
            last_http_failure = ProviderFailure(category, status)
        raise ProviderFailure("provider_error")

    def _request(self, provider, kind, **kwargs):
        model = kwargs["model"]
        if kind == "quiz":
            text = kwargs.get("contentText")
            if not isinstance(text, str) or not text.strip():
                raise ProviderFailure("invalid_input")
            source = "아래 contentText의 근거 문구만 사용하세요. sourceEvidence는 원문에서 그대로 인용하세요.\n" + text
        else:
            video = kwargs.get("video")
            if not isinstance(video, dict) or not isinstance(video.get("youtubeUrl"), str) or not video["youtubeUrl"].startswith("https://www.youtube.com/watch?"):
                raise ProviderFailure("invalid_input")
            source = video["youtubeUrl"]
        if kind == "grounding":
            prompt = "영상에서 확인 가능한 사실과 발화/화면 근거, 제시된 timestamp를 추출하세요. 추측은 제외하세요."
            schema = grounding_schema()
        else:
            question_count, option_count = kwargs.get("questionCount"), kwargs.get("optionCount")
            prompt_version = kwargs.get("promptVersion")
            if (type(question_count) is not int or question_count < 1 or type(option_count) is not int
                    or option_count < 2 or not isinstance(prompt_version, str) or not prompt_version):
                raise ProviderFailure("invalid_input")
            prompt = (f"한국어 4지선다 퀴즈 {question_count}개, 각 {option_count}개 보기를 생성하세요. "
                      f"correctOptionIndex는 0-based입니다. promptVersion은 {prompt_version}입니다. "
                      "정답 근거인 sourceEvidence를 포함하세요.")
            schema = quiz_schema()
        if provider == "gemini":
            input_parts = ([{"type": "video", "uri": source, "processing": self.policy.video_processing}]
                           if kind != "quiz" else [])
            input_parts.append({"type": "text", "text": prompt + ("\n" + source if kind == "quiz" else "")})
            return {"model": model, "input": input_parts, "store": False,
                    "response_format": {"type": "text", "mime_type": "application/json", "schema": schema},
                    "generation_config": {"max_output_tokens": self.policy.max_output_tokens}}
        return {"model": model, "input": prompt + "\n" + source, "store": False,
                "reasoning": {"effort": self.policy.reasoning_effort},
                "max_output_tokens": self.policy.max_output_tokens,
                "text": {"format": {"type": "json_schema", "name": "pilot_quiz",
                                     "strict": True, "schema": schema}}}

    @staticmethod
    def _extract(provider, response):
        if not isinstance(response, dict):
            raise ProviderFailure("invalid_provider_response")
        if response.get("status") != "completed":
            raise ProviderFailure("incomplete_response")
        try:
            if provider == "gemini":
                steps = response["steps"]
                outputs = [item["text"] for step in steps if step.get("type") == "model_output"
                           for item in step["content"] if item.get("type") == "text"]
                usage = response.get("usage", {})
                tokens = (usage.get("total_input_tokens"), usage.get("total_output_tokens"),
                          usage.get("total_thought_tokens"))
                tool_tokens = usage.get("total_tool_use_tokens")
            else:
                outputs = [item["text"] for output in response["output"] if output.get("type") == "message"
                           for item in output["content"] if item.get("type") == "output_text"]
                usage = response.get("usage", {})
                tokens = (usage.get("input_tokens"), usage.get("output_tokens"),
                          usage.get("output_tokens_details", {}).get("reasoning_tokens"))
                tool_tokens = None
            if len(outputs) != 1 or not isinstance(outputs[0], str) or not outputs[0].strip():
                raise ValueError()
            if any(value is not None and (type(value) is not int or value < 0) for value in tokens):
                raise ValueError()
            if tool_tokens is not None and (type(tool_tokens) is not int or tool_tokens < 0):
                raise ValueError()
        except (AttributeError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            raise ProviderFailure("invalid_provider_response")
        try:
            normalized = json.loads(outputs[0])
        except json.JSONDecodeError:
            # An HTTP-complete Quiz with malformed model JSON is a parse failure in the runner.
            normalized = outputs[0]
        # Generated text belongs to normalized evaluation data, never persisted raw metadata.
        return {"normalized": normalized, "responseBody": {"status": "completed", "provider": provider,
                                                                "usage": {"inputTokens": tokens[0],
                                                                          "outputTokens": tokens[1],
                                                                          "thinkingTokens": tokens[2],
                                                                          "toolUseTokens": tool_tokens}},
                "inputTokens": tokens[0], "outputTokens": tokens[1], "thinkingTokens": tokens[2]}
