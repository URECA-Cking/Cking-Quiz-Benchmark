"""One-condition Pilot runner. Live HTTP requires explicit safety options."""

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml


class ProviderFailure(Exception):
    def __init__(self, category):
        super().__init__(category)
        self.category = category


class FixtureProvider:
    """Offline provider boundary. A future API adapter can implement the same invoke method."""

    is_actual_api = False

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def invoke(self, kind, **kwargs):
        self.calls.append(dict(kind=kind, **kwargs))
        response = self.responses.get(kind)
        if response is None:
            raise ProviderFailure("fixture_missing")
        if response.get("errorCategory"):
            raise ProviderFailure(response["errorCategory"])
        return response


class PilotRunner:
    FILES = {
        "video_grounding": "video-grounding.jsonl",
        "quiz_generation": "quiz-generation.jsonl",
        "end_to_end": "end-to-end.jsonl",
    }

    def __init__(self, repository, results, config=None):
        self.repository = Path(repository).resolve()
        self.results = self._safe_results_path(results)
        config_file = Path(config) if config else self.repository / "configs" / "pilot.yaml"
        self.config = yaml.safe_load(config_file.read_text(encoding="utf-8"))["pilot"]
        self.videos = {row["videoId"]: row for row in self._jsonl(self.repository / "data" / "videos.jsonl")}

    def _safe_results_path(self, results):
        safe_root = self.repository / "results"
        if safe_root.resolve() != safe_root:
            raise ValueError("The results directory must not be a link to another location")
        requested = Path(results)
        resolved = (requested if requested.is_absolute() else self.repository / requested).resolve()
        if resolved != safe_root and safe_root not in resolved.parents:
            raise ValueError("The results directory must be inside repository/results")
        return resolved

    @staticmethod
    def _jsonl(path):
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _check(self, video_id, repetition):
        if video_id not in self.videos:
            raise ValueError("Unknown Pilot videoId")
        if not 1 <= repetition <= self.config["repetitions_per_condition"]:
            raise ValueError("Repetition is outside pilot configuration")

    def _base(self, benchmark_type, video_id, method, model, repetition):
        return {
            "benchmarkType": benchmark_type, "runId": uuid.uuid4().hex,
            "videoId": video_id, "method": method, "model": model,
            "promptVersion": self.config["prompt_version"] if benchmark_type != "video_grounding" else None,
            "repetition": repetition, "startedAt": datetime.now(timezone.utc).isoformat(),
            "apiStatus": "not_run", "errorCategory": None, "latencyMs": None,
            "inputTokens": None, "outputTokens": None, "thinkingTokens": None,
            "estimatedCostUsd": None, "pricingReference": None,
        }

    def _save(self, row, raw=None, evaluation=None):
        self._safe_results_path(self.results)
        self._validate_raw_metadata(raw)
        for key in ("latencyMs", "estimatedCostUsd", "totalLatencyMs", "totalEstimatedCostUsd"):
            value = row.get(key)
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
                raise ValueError("Invalid measurement in result row")
        for key in ("inputTokens", "outputTokens", "thinkingTokens"):
            value = row.get(key)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("Invalid token usage in result row")
        self.results.mkdir(parents=True, exist_ok=True)
        if raw is not None:
            raw_dir = self.results / "raw"
            if raw_dir.resolve() != raw_dir:
                raise ValueError("Raw results directory must not be a link")
            raw_dir.mkdir(exist_ok=True)
            raw_file = raw_dir / (row["runId"] + ".json")
            if raw_file.resolve() != raw_file:
                raise ValueError("Raw result file must not be a link")
            raw_file.write_text(
                json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
        if evaluation is not None:
            evaluation_dir = self.results / "evaluation"
            if evaluation_dir.resolve() != evaluation_dir:
                raise ValueError("Evaluation results directory must not be a link")
            evaluation_dir.mkdir(exist_ok=True)
            evaluation_file = evaluation_dir / (row["runId"] + ".json")
            if evaluation_file.resolve() != evaluation_file:
                raise ValueError("Evaluation result file must not be a link")
            evaluation_file.write_text(
                json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")
        result_file = self.results / self.FILES[row["benchmarkType"]]
        if result_file.resolve() != result_file:
            raise ValueError("Result file must not be a link")
        with result_file.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    @staticmethod
    def _validate_raw_metadata(value):
        if value is None or value == {"source": "fixture"}:
            return
        if (not isinstance(value, dict) or set(value) != {"status", "provider", "usage"}
                or value["status"] != "completed" or value["provider"] not in ("gemini", "openai")
                or not isinstance(value["usage"], dict)
                or set(value["usage"]) != {"inputTokens", "outputTokens", "thinkingTokens", "toolUseTokens"}
                or any(token is not None and (type(token) is not int or token < 0)
                       for token in value["usage"].values())):
            raise ValueError("Raw results may contain allowlisted provider metadata only")

    @staticmethod
    def _metrics(row, response, elapsed_ms):
        values = {"latencyMs": response.get("latencyMs", elapsed_ms)}
        for key in ("inputTokens", "outputTokens", "thinkingTokens", "estimatedCostUsd", "pricingReference"):
            values[key] = response.get(key)
        for key in ("latencyMs", "estimatedCostUsd"):
            value = values[key]
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
                raise ProviderFailure("invalid_measurement")
        for key in ("inputTokens", "outputTokens", "thinkingTokens"):
            value = values[key]
            if value is not None and (type(value) is not int or value < 0):
                raise ProviderFailure("invalid_measurement")
        if values["pricingReference"] is not None and not isinstance(values["pricingReference"], str):
            raise ProviderFailure("invalid_measurement")
        row.update(values)

    @staticmethod
    def _actual(provider):
        return getattr(provider, "is_actual_api", False) is True

    def _success(self, row, provider, response, elapsed):
        if self._actual(provider):
            self._metrics(row, response, elapsed)
            row["apiStatus"] = "success"
        # Offline fixtures have no measured API latency, usage, cost or API success.

    def _failure(self, row, provider, failure, started):
        actual = self._actual(provider)
        row["apiStatus"] = "error" if actual else "not_run"
        category = failure.category
        category = category if isinstance(category, str) and category else "provider_error"
        row["errorCategory"] = category if actual else "fixture_" + category
        if actual:
            row["latencyMs"] = round((time.monotonic() - started) * 1000, 3)

    def _require_fixed_content(self, video_id, content_hash):
        root = self.repository / "data" / "restricted" / "fixed-content"
        if root.parent.exists() and root.parent.resolve() != root.parent:
            raise ValueError("Fixed content directory must not be a link")
        root.mkdir(parents=True, exist_ok=True)
        if root.resolve() != root:
            raise ValueError("Fixed content manifest directory must not be a link")
        prompt_version = self.config["prompt_version"]
        key = hashlib.sha256((video_id + "\0" + prompt_version).encode("utf-8")).hexdigest()
        manifest = root / (key + ".json")
        if manifest.resolve() != manifest:
            raise ValueError("Fixed content manifest must not be a link")
        record = {"videoId": video_id, "promptVersion": prompt_version, "contentTextSha256": content_hash}
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=root,
                                             prefix=".fixed-", suffix=".tmp", delete=False) as stream:
                temp_path = Path(stream.name)
                json.dump(record, stream)
                stream.flush()
                os.fsync(stream.fileno())
            # Same-directory hard-link creation is atomic and never replaces the first writer.
            try:
                os.link(temp_path, manifest)
            except FileExistsError:
                if json.loads(manifest.read_text(encoding="utf-8")) != record:
                    raise ValueError("A different fixed contentText is registered for this video and promptVersion")
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    @staticmethod
    def _call(provider, kind, **kwargs):
        start = time.monotonic()
        try:
            response = provider.invoke(kind, **kwargs)
        except ProviderFailure:
            raise
        except Exception as exc:
            # Never include provider exception text: it could contain URLs, headers or secrets.
            raise ProviderFailure("provider_error") from exc
        if not isinstance(response, dict):
            raise ProviderFailure("invalid_provider_response")
        return response, round((time.monotonic() - start) * 1000, 3)

    def _run_grounding(self, video_id, method, repetition, provider, authorized_transcript):
        self._check(video_id, repetition)
        methods = {item["id"]: item for item in self.config["video_grounding"]["methods"]}
        if method not in methods:
            raise ValueError("Unknown grounding method")
        row = self._base("video_grounding", video_id, method, methods[method].get("model"), repetition)
        row.update(groundingFacts=[], omission=None, hallucination=None)
        if method == "authorized_transcript" and not authorized_transcript:
            row["errorCategory"] = "authorized_transcript_unavailable"
            return self._save(row), None
        started = time.monotonic()
        raw = None
        normalized = None
        try:
            if method == "authorized_transcript":
                # The caller must supply text whose use rights have been checked. No AI call.
                content = authorized_transcript
                raw = None
            else:
                response, elapsed = self._call(
                    provider, "grounding", video=self.videos[video_id], model=row["model"])
                normalized = response.get("normalized") if self._actual(provider) else response.get("raw")
                raw = response.get("responseBody") if self._actual(provider) else {"source": "fixture"}
                if not isinstance(normalized, dict) or not isinstance(normalized.get("contentText"), str) or not normalized["contentText"].strip():
                    raise ProviderFailure("invalid_grounding_response")
                content = normalized["contentText"]
                facts = normalized.get("facts", [])
                if not isinstance(facts, list):
                    raise ProviderFailure("invalid_grounding_response")
                for fact in facts:
                    if (not isinstance(fact, dict)
                            or any(not isinstance(fact.get(key), str) or not fact[key].strip()
                                   for key in ("fact", "evidenceType", "evidence"))
                            or fact["evidenceType"] not in ("speech", "visual", "unknown")
                            or any(value is not None and (type(value) not in (int, float) or value < 0)
                                   for value in (fact.get("timestampStartSeconds"),
                                                 fact.get("timestampEndSeconds")))):
                        raise ProviderFailure("invalid_grounding_response")
                    row["groundingFacts"].append({
                        "fact": fact["fact"], "evidenceType": fact["evidenceType"],
                        "evidence": fact["evidence"],
                        "timestampStartSeconds": fact.get("timestampStartSeconds"),
                        "timestampEndSeconds": fact.get("timestampEndSeconds"),
                        "factExists": None, "evidenceTypeCorrect": None,
                        "timestampAccurate": None, "reviewNote": None,
                    })
                self._success(row, provider, response, elapsed)
            return self._save(row, raw, normalized), content
        except ProviderFailure as exc:
            self._failure(row, provider, exc, started)
            return self._save(row, raw, normalized), None

    def run_grounding(self, video_id, method, repetition, provider, authorized_transcript=None):
        return self._run_grounding(video_id, method, repetition, provider, authorized_transcript)[0]

    @staticmethod
    def _question_reviews(count):
        return [{"questionIndex": i, "answerAccuracy": None, "uniqueAnswer": None,
                 "evidenceSupportsAnswer": None, "videoGrounding": None,
                 "koreanQuality": None, "hallucination": None, "reviewNote": None}
                for i in range(count)]

    def _evaluate_quiz(self, row, raw, content_text):
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else raw
            if not isinstance(parsed, dict) or not isinstance(parsed.get("questions"), list):
                raise ValueError()
            if "promptVersion" in parsed and parsed["promptVersion"] is not None and not isinstance(parsed["promptVersion"], str):
                raise ValueError()
            for question in parsed["questions"]:
                if not isinstance(question, dict) or any(not isinstance(question.get(key), str)
                                                        for key in ("question", "explanation", "sourceEvidence")):
                    raise ValueError()
                if not isinstance(question.get("options"), list) or any(
                        not isinstance(option, str) for option in question["options"]):
                    raise ValueError()
                index = question.get("correctOptionIndex")
                if type(index) is not int or not -(2 ** 31) <= index <= 2 ** 31 - 1:
                    raise ValueError()
        except (ValueError, TypeError):
            row.update(parseStatus="fail", validatorStatus="not_run", errorCategory="parse_error")
            return
        row["parseStatus"] = "pass"
        questions = parsed["questions"]
        row["questionCount"] = len(questions)
        row["questionReviews"] = self._question_reviews(len(questions))
        try:
            if "promptVersion" not in parsed or parsed["promptVersion"] != row["promptVersion"]:
                raise ValueError()
            if len(questions) != self.config["questions_per_video"]:
                raise ValueError()
            if content_text is not None and len(content_text) > 50000:
                raise ValueError()
            contained = True
            seen_questions = set()
            for question in questions:
                options = question.get("options")
                index = question.get("correctOptionIndex")
                normalized_question = question["question"].strip().lower()
                normalized_options = [option.strip().lower() for option in options]
                if (not normalized_question or normalized_question in seen_questions
                        or len(options) != self.config["options_per_question"]
                        or any(not option for option in normalized_options)
                        or len(set(normalized_options)) != len(options)
                        or not 0 <= index < len(options)
                        or not question["explanation"].strip()
                        or not question["sourceEvidence"].strip()):
                    raise ValueError()
                seen_questions.add(normalized_question)
                if content_text is not None and question["sourceEvidence"] not in content_text:
                    contained = False
            row["evidenceTextContained"] = contained if content_text is not None else None
            row["validatorStatus"] = ("not_run" if content_text is None
                                      else "pass" if contained else "fail")
            if not contained:
                row["errorCategory"] = "evidence_not_in_content"
        except ValueError:
            row["validatorStatus"] = "fail"
            row["errorCategory"] = "quiz_contract_error"

    def run_quiz(self, video_id, model, repetition, content_text, provider, source_grounding_run_id=None):
        self._check(video_id, repetition)
        models = {item["id"] for item in self.config["quiz_generation"]["models"]}
        if model not in models or not isinstance(content_text, str) or not content_text.strip():
            raise ValueError("Configured quiz model and nonempty fixed contentText are required")
        content_hash = hashlib.sha256(content_text.encode("utf-8")).hexdigest()
        if source_grounding_run_id is None:
            self._require_fixed_content(video_id, content_hash)
        row = self._base("quiz_generation", video_id, "fixed_content_text", model, repetition)
        row.update(contentTextSha256=content_hash,
                   sourceGroundingRunId=source_grounding_run_id, parseStatus="not_run",
                   validatorStatus="not_run", beCompatibility="not_run", questionCount=None,
                   evidenceTextContained=None, questionReviews=[])
        started = time.monotonic()
        try:
            response, elapsed = self._call(provider, "quiz", contentText=content_text,
                                           model=model, promptVersion=row["promptVersion"],
                                           questionCount=self.config["questions_per_video"],
                                           optionCount=self.config["options_per_question"])
            normalized = response.get("normalized") if self._actual(provider) else response.get("raw")
            raw = response.get("responseBody") if self._actual(provider) else {"source": "fixture"}
            self._success(row, provider, response, elapsed)
            self._evaluate_quiz(row, normalized, content_text)
            row["beCompatibility"] = "pass" if row["validatorStatus"] == "pass" else "fail"
            return self._save(row, raw, normalized)
        except ProviderFailure as exc:
            self._failure(row, provider, exc, started)
            return self._save(row)

    def run_end_to_end(self, video_id, method, repetition, provider, authorized_transcript=None):
        self._check(video_id, repetition)
        methods = {item["id"]: item for item in self.config["end_to_end"]["methods"]}
        if method not in methods:
            raise ValueError("Unknown end-to-end method")
        setting = methods[method]
        row = self._base("end_to_end", video_id, method, setting["quiz_model"], repetition)
        row.update(groundingRunId=None, quizRunId=None, beCompatibility="not_run",
                   questionReviews=[], totalLatencyMs=None, totalEstimatedCostUsd=None,
                   videoGrounding=None)
        if method == "gemini_direct_quiz":
            row["beCompatibility"] = "not_applicable"
            row.update(parseStatus="not_run", validatorStatus="not_run", questionCount=None,
                       evidenceTextContained=None)
            started = time.monotonic()
            try:
                response, elapsed = self._call(provider, "direct", video=self.videos[video_id],
                                               model=row["model"], promptVersion=row["promptVersion"],
                                               questionCount=self.config["questions_per_video"],
                                               optionCount=self.config["options_per_question"])
                normalized = response.get("normalized") if self._actual(provider) else response.get("raw")
                raw = response.get("responseBody") if self._actual(provider) else {"source": "fixture"}
                self._success(row, provider, response, elapsed)
                self._evaluate_quiz(row, normalized, None)
                # A video-only direct Quiz cannot run the BE contentText validator.
                row["validatorStatus"] = "not_run"
                row["totalLatencyMs"] = row["latencyMs"]
                row["totalEstimatedCostUsd"] = row["estimatedCostUsd"]
                return self._save(row, raw, normalized)
            except ProviderFailure as exc:
                self._failure(row, provider, exc, started)
                return self._save(row)
        grounding, content = self._run_grounding(video_id, setting["grounding"], repetition,
                                                 provider, authorized_transcript)
        row["groundingRunId"] = grounding["runId"]
        if content is None:
            row.update(apiStatus=grounding["apiStatus"], errorCategory=grounding["errorCategory"])
            row["totalLatencyMs"] = grounding["latencyMs"]
            row["totalEstimatedCostUsd"] = grounding["estimatedCostUsd"]
            return self._save(row)
        quiz = self.run_quiz(video_id, row["model"], repetition, content, provider,
                             source_grounding_run_id=grounding["runId"])
        row["quizRunId"] = quiz["runId"]
        row["apiStatus"] = quiz["apiStatus"]
        row["errorCategory"] = quiz["errorCategory"]
        row["beCompatibility"] = quiz["beCompatibility"]
        row["questionReviews"] = quiz["questionReviews"]
        latencies = (grounding["latencyMs"], quiz["latencyMs"])
        row["totalLatencyMs"] = sum(latencies) if all(value is not None for value in latencies) else None
        costs = (grounding["estimatedCostUsd"], quiz["estimatedCostUsd"])
        row["totalEstimatedCostUsd"] = sum(costs) if all(value is not None for value in costs) else None
        return self._save(row)


def main():
    parser = argparse.ArgumentParser(description="Run exactly one Pilot condition")
    parser.add_argument("mode", choices=("grounding", "quiz", "end-to-end"))
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--method")
    parser.add_argument("--model")
    parser.add_argument("--repetition", type=int, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--fixture", type=Path)
    source.add_argument("--live", action="store_true")
    parser.add_argument("--call-limit", type=int)
    parser.add_argument("--per-call-cost-limit", type=float)
    parser.add_argument("--total-cost-limit", type=float,
                        help="Estimated cost guard for this CLI run/ProviderRouter only; not a persistent billing cap")
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--retry-attempts", type=int)
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--video-processing")
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--estimated-input-tokens", type=int)
    parser.add_argument("--input-price-per-million", type=float)
    parser.add_argument("--output-price-per-million", type=float)
    parser.add_argument("--pricing-reference", help="Source URL/date for the supplied token prices")
    for provider_name in ("gemini", "openai"):
        parser.add_argument(f"--{provider_name}-input-price-per-million", type=float)
        parser.add_argument(f"--{provider_name}-output-price-per-million", type=float)
        parser.add_argument(f"--{provider_name}-pricing-reference")
    parser.add_argument("--content-file", type=Path)
    parser.add_argument("--authorized-transcript-file", type=Path)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    runner = PilotRunner(Path(__file__).resolve().parents[1], args.results_dir)
    if args.live:
        from src.provider_adapters import LivePolicy, ProviderRouter
        provider_prices = {}
        for provider_name in ("gemini", "openai"):
            provider_input = getattr(args, f"{provider_name}_input_price_per_million")
            provider_output = getattr(args, f"{provider_name}_output_price_per_million")
            provider_reference = getattr(args, f"{provider_name}_pricing_reference")
            if any(value is not None for value in (provider_input, provider_output, provider_reference)):
                provider_prices[provider_name] = (provider_input, provider_output, provider_reference)
        if provider_prices and (args.input_price_per_million is not None
                                or args.output_price_per_million is not None
                                or args.pricing_reference is not None):
            parser.error("Use either generic or provider-specific pricing options")
        policy = LivePolicy(enabled=True, call_limit=args.call_limit,
                            per_call_cost_limit=args.per_call_cost_limit,
                            total_cost_limit=args.total_cost_limit,
                            timeout_seconds=args.timeout_seconds, retry_attempts=args.retry_attempts,
                            reasoning_effort=args.reasoning_effort, video_processing=args.video_processing,
                            max_output_tokens=args.max_output_tokens,
                            estimated_input_tokens=args.estimated_input_tokens,
                            input_price_per_million=args.input_price_per_million,
                            output_price_per_million=args.output_price_per_million,
                            pricing_reference=args.pricing_reference,
                            provider_prices=provider_prices or None)
        provider = ProviderRouter(policy, pilot_config=runner.config)
        if args.mode == "quiz":
            selected = [("quiz", args.model)]
        elif args.mode == "grounding":
            method = next((item for item in runner.config["video_grounding"]["methods"]
                           if item["id"] == args.method), None)
            selected = [("grounding", method["model"])] if method and "model" in method else []
        else:
            method = next((item for item in runner.config["end_to_end"]["methods"]
                           if item["id"] == args.method), None)
            if method is None:
                parser.error("Unknown end-to-end method")
            if method["grounding"] == "direct_video":
                selected = [("direct", method["quiz_model"])]
            else:
                grounding = next((item for item in runner.config["video_grounding"]["methods"]
                                  if item["id"] == method["grounding"]), None)
                selected = ([("grounding", grounding["model"])] if grounding and "model" in grounding else [])
                selected.append(("quiz", method["quiz_model"]))
        providers = {provider.models[model]["provider"] for _, model in selected
                     if model in provider.models}
        if len(providers) > 1 and not provider_prices:
            parser.error("Mixed-provider live runs require provider-specific pricing")
        for kind, model in selected:
            entry = provider.models.get(model)
            if entry is None or kind not in entry["kinds"]:
                parser.error("Unsupported live provider/model for the selected condition")
            policy.authorize(kind, 0, 0, entry["provider"])
    else:
        provider = FixtureProvider(json.loads(args.fixture.read_text(encoding="utf-8")))
    transcript = args.authorized_transcript_file.read_text(encoding="utf-8") if args.authorized_transcript_file else None
    if args.mode == "grounding":
        if not args.method:
            parser.error("--method is required for grounding")
        row = runner.run_grounding(args.video_id, args.method, args.repetition, provider, transcript)
    elif args.mode == "quiz":
        if not args.model or not args.content_file:
            parser.error("--model and --content-file are required for quiz")
        row = runner.run_quiz(args.video_id, args.model, args.repetition,
                              args.content_file.read_text(encoding="utf-8"), provider)
    else:
        if not args.method:
            parser.error("--method is required for end-to-end")
        row = runner.run_end_to_end(args.video_id, args.method, args.repetition, provider, transcript)
    print(json.dumps({"runId": row["runId"], "apiStatus": row["apiStatus"],
                      "errorCategory": row["errorCategory"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
