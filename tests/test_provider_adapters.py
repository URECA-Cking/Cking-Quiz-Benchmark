import contextlib
import hashlib
import io
import json
import runpy
import shutil
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from src.pilot_runner import PilotRunner, ProviderFailure
from src.provider_adapters import (GROUNDING_PROMPTS, LivePolicy, ProviderRouter, grounding_schema,
                                   urllib_transport, video_timestamp_seconds)


CONTENT = "NASA measures global rain and snow every 30 minutes."
APPROVER = "pilot-reviewer"
VIDEO = {"videoId": "nasa-water-cycle-2019", "youtubeUrl": "https://www.youtube.com/watch?v=mCNcxu8MXzo"}
QUIZ = {"promptVersion": "pilot-v1", "questions": [
    {"question": f"질문 {i}?", "options": ["가", "나", "다", "라"],
     "correctOptionIndex": 0, "explanation": "자료 기준", "sourceEvidence": "global rain and snow"}
    for i in range(3)]}
GROUNDING_V1 = "video-grounding-v1"
GROUNDING = {"contentText": CONTENT, "facts": [{
    "fact": "NASA measures rainfall", "evidenceType": "speech", "evidence": "global rain and snow",
    "timestampStartSeconds": 61, "timestampEndSeconds": 73}]}
# video-grounding-v3 (configs/pilot-v2.yaml) takes "MM:SS" string timestamps.
GROUNDING_V3 = {"contentText": CONTENT, "facts": [{
    "fact": "NASA measures rainfall", "evidenceType": "speech", "evidence": "global rain and snow",
    "timestampStart": "01:01", "timestampEnd": "01:13"}]}


def approve_test_source(runner, content=CONTENT):
    """Create an explicitly approved source only in a temporary test repository."""
    row = runner._base("video_grounding", VIDEO["videoId"], "gemini_video",
                       "gemini-3.8-flash", 1)
    row.update(apiStatus="success", groundingFacts=[], omission=None, hallucination=None,
               contentTextSha256=None, contentTextApprovalStatus=None)
    runner._save(row, raw={"source": "fixture"},
                 evaluation={"contentText": content, "facts": []})
    runner.approve_content(row["runId"], APPROVER)
    return row["runId"]


def policy(**overrides):
    values = dict(enabled=True, call_limit=1, per_call_cost_limit=1.0, total_cost_limit=1.0,
                  timeout_seconds=10, retry_attempts=0, reasoning_effort="none",
                  video_processing="static", max_output_tokens=2000,
                  video_estimated_input_tokens=10000, quiz_estimated_input_tokens=10000,
                  input_price_per_million=1.0,
                  output_price_per_million=1.0, pricing_reference="operator-test-pricing")
    values.update(overrides)
    return LivePolicy(**values)


class FakeTransport:
    def __init__(self, response, status=200):
        self.response, self.status = response, status
        self.calls = []

    def __call__(self, url, headers, body, timeout):
        self.calls.append((url, headers, body, timeout))
        return self.status, self.response


class QueueTransport(FakeTransport):
    def __init__(self, responses):
        super().__init__(None)
        self.responses = iter(responses)

    def __call__(self, url, headers, body, timeout):
        self.calls.append((url, headers, body, timeout))
        return 200, next(self.responses)


def gemini_response(payload, status="completed"):
    return {"status": status, "model": "gemini-3.8-flash", "steps": [{
        "type": "model_output", "content": [{"type": "text", "text": json.dumps(payload)}]}],
        "usage": {"total_input_tokens": 10, "total_output_tokens": 20, "total_thought_tokens": 3}}


class PilotV2ControlsAdapterTest(unittest.TestCase):
    def config(self):
        import yaml
        return yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/pilot-v2.yaml").read_text(encoding="utf-8"))["pilot"]

    def router(self, transport, config=None, **overrides):
        return ProviderRouter(policy(**overrides), transport=transport,
                              api_keys={"gemini": "test-key", "openai": "test-key"}, pilot_config=config)

    def arguments(self, kind="quiz", model="gemini-3.8-flash", version="pilot-v2"):
        return dict(model=model, promptVersion=version, questionCount=3, optionCount=4,
                    **({"contentText": CONTENT} if kind == "quiz" else {"video": VIDEO}))

    def test_v1_quiz_and_direct_request_literals_are_unchanged(self):
        prompt = ("한국어 4지선다 퀴즈 3개, 각 4개 보기를 생성하세요. "
                  "correctOptionIndex는 0-based입니다. promptVersion은 pilot-v1입니다. "
                  "정답 근거인 sourceEvidence를 포함하세요.")
        source = "아래 contentText의 근거 문구만 사용하세요. sourceEvidence는 원문에서 그대로 인용하세요.\n" + CONTENT
        for kind, model in (("quiz", "gemini-3.8-flash"), ("quiz", "gpt-5.4-mini"), ("direct", "gemini-3.8-flash")):
            transport = FakeTransport(gemini_response(QUIZ) if model.startswith("gemini") else {
                "status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(QUIZ)}]}],
                "usage": {"input_tokens": 10, "output_tokens": 20, "output_tokens_details": {"reasoning_tokens": 3}}})
            self.router(transport).invoke(kind, **self.arguments(kind, model, "pilot-v1"))
            body = transport.calls[0][2]
            if model.startswith("gemini"):
                expected_input = ([{"type": "video", "uri": VIDEO["youtubeUrl"], "processing": "static"}] if kind == "direct" else [])
                expected_input.append({"type": "text", "text": prompt if kind == "direct" else prompt + "\n" + source})
                self.assertEqual(body["input"], expected_input)
                self.assertEqual(body["generation_config"], {"max_output_tokens": 2000})
            else:
                self.assertEqual(body["input"], prompt + "\n" + source)
                self.assertEqual(body["reasoning"], {"effort": "none"})

    def test_v2_requests_use_medium_without_sampling_or_exact_three_schema(self):
        for kind, model in (("quiz", "gemini-3.8-flash"), ("quiz", "gpt-5.4-mini"), ("direct", "gemini-3.8-flash")):
            output = dict(QUIZ, promptVersion="pilot-v2")
            transport = FakeTransport(gemini_response(output) if model.startswith("gemini") else {
                "status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(output)}]}],
                "usage": {"input_tokens": 10, "output_tokens": 20, "output_tokens_details": {"reasoning_tokens": 3}}})
            router = self.router(transport, self.config(), reasoning_effort=None)
            self.assertEqual(router.invoke(kind, **self.arguments(kind, model))["normalized"], output)
            body = transport.calls[0][2]
            if model.startswith("gemini"):
                self.assertEqual(body["generation_config"], {"max_output_tokens": 2000, "thinking_level": "medium"})
                schema = body["response_format"]["schema"]
                prompt = body["input"][-1]["text"]
            else:
                self.assertEqual(body["reasoning"], {"effort": "medium"})
                schema = body["text"]["format"]["schema"]
                prompt = body["input"]
            self.assertIn("정확히 3", prompt)
            self.assertNotIn("minItems", schema["properties"]["questions"])
            self.assertNotIn("maxItems", schema["properties"]["questions"])
            self.assertNotIn("temperature", json.dumps(body))
            self.assertNotIn("top_p", json.dumps(body))

    def test_v2_retry_and_settings_conflicts_are_rejected_before_http(self):
        for overrides in ({"retry_attempts": 1}, {"reasoning_effort": "low"}):
            transport = FakeTransport({})
            with self.assertRaises((ValueError, ProviderFailure)):
                self.router(transport, self.config(), **dict({"reasoning_effort": None}, **overrides)).invoke("quiz", **self.arguments(model="gpt-5.4-mini"))
            self.assertEqual(transport.calls, [])

    def test_grounding_technical_retry_is_preserved_for_v1_and_v2(self):
        for config in (None, self.config()):
            for status in (429, 503):
                with self.subTest(version=config and config["prompt_version"], status=status):
                    transport = FakeTransport(gemini_response(GROUNDING))
                    responses = iter(((status, {}), (200, gemini_response(GROUNDING))))
                    def send(url, headers, body, timeout):
                        transport.calls.append((url, headers, body, timeout))
                        return next(responses)
                    router = self.router(send, config, retry_attempts=1, call_limit=2)
                    result = router.invoke("grounding", video=VIDEO, model="gemini-3.8-flash",
                                           promptVersion="video-grounding-v2" if config else GROUNDING_V1)
                    self.assertEqual(result["normalized"], GROUNDING)
                    self.assertEqual((router.calls, len(transport.calls)), (2, 2))
                    self.assertEqual(transport.calls[0][2], transport.calls[1][2])
                    self.assertNotIn("thinking_level", transport.calls[0][2]["generation_config"])

    def test_v2_quiz_and_direct_nonzero_retry_fail_before_transport(self):
        for kind, model in (("quiz", "gemini-3.8-flash"), ("quiz", "gpt-5.4-mini"),
                            ("direct", "gemini-3.8-flash")):
            with self.subTest(kind=kind, model=model):
                transport = FakeTransport({})
                router = self.router(transport, self.config(), retry_attempts=1, call_limit=2)
                with self.assertRaises(ProviderFailure) as caught:
                    router.invoke(kind, **self.arguments(kind, model))
                self.assertEqual(caught.exception.category, "live_guard")
                self.assertEqual((router.calls, router.reserved_cost, transport.calls), (0, 0, []))

    def test_unknown_prompt_version_is_rejected_before_http(self):
        for kind in ("quiz", "direct"):
            transport = FakeTransport({})
            with self.assertRaises((ValueError, ProviderFailure)):
                self.router(transport).invoke(kind, **self.arguments(kind, version="pilot-v99"))
            self.assertEqual(transport.calls, [])

    def test_v2_missing_settings_and_wrong_question_count_fail_closed(self):
        config = self.config()
        config["questions_per_video"] = 2
        transport = FakeTransport({})
        with self.assertRaises((ValueError, ProviderFailure)):
            self.router(transport, config).invoke("quiz", **self.arguments())
        self.assertEqual(transport.calls, [])

    def test_v2_grounding_request_is_identical_and_has_no_thinking_override(self):
        config = self.config()
        v2 = self.router(FakeTransport({}), config, reasoning_effort=None)
        legacy = self.router(FakeTransport({}))
        arguments = dict(video=VIDEO, model="gemini-3.8-flash", promptVersion="video-grounding-v2")
        transport = FakeTransport(gemini_response(GROUNDING))
        v2.transport = transport
        v2.invoke("grounding", **arguments)
        self.assertEqual(transport.calls[0][2], legacy._request("gemini", "grounding", **arguments))
        self.assertEqual(transport.calls[0][2]["generation_config"], {"max_output_tokens": 2000})

    def test_v2_gemini_thinking_override_mismatch_and_match(self):
        for kind in ("quiz", "direct"):
            transport = FakeTransport(gemini_response(dict(QUIZ, promptVersion="pilot-v2")))
            with self.assertRaises(ProviderFailure):
                self.router(transport, self.config(), thinking_level="high").invoke(kind, **self.arguments(kind))
            self.assertEqual(transport.calls, [])
            self.router(transport, self.config(), thinking_level="medium").invoke(kind, **self.arguments(kind))
            self.assertEqual(len(transport.calls), 1)

    def test_v2_invalid_generation_settings_are_rejected_without_reservation(self):
        for settings in ({"thinking_level": "high"}, {"thinking_level": None},
                         {"thinking_level": "medium", "temperature": 1}):
            config = self.config()
            config["quiz_generation"]["models"][0]["generation_settings"] = settings
            transport = FakeTransport({})
            router = self.router(transport, config)
            with self.assertRaises(ValueError):
                router.invoke("direct", **self.arguments("direct"))
            self.assertEqual((router.calls, router.reserved_cost, transport.calls), (0, 0, []))
        config = self.config()
        config["quiz_generation"]["models"][0].pop("generation_settings", None)
        with self.assertRaises((ValueError, ProviderFailure)):
            self.router(transport, config).invoke("quiz", **self.arguments())
        self.assertEqual(transport.calls, [])

    def test_cli_v2_settings_are_resolved_or_rejected_before_runner(self):
        from src.pilot_runner import main
        for model, option in (("gpt-5.4-mini", "--reasoning-effort"), ("gemini-3.8-flash", "--thinking-level")):
            for value in (None, "medium", "low"):
                config = self.config()
                runner = mock.Mock(config=config)
                runner.run_quiz.return_value = {"runId": "offline-test", "apiStatus": "not_run", "errorCategory": None}
                with tempfile.TemporaryDirectory() as folder:
                    content_file = Path(folder) / "content.txt"
                    content_file.write_text(CONTENT, encoding="utf-8")
                    argv = ["pilot_runner", "quiz", "--video-id", VIDEO["videoId"], "--model", model,
                            "--repetition", "1", "--live", "--content-file", str(content_file),
                            "--source-grounding-run-id", "a" * 32, "--call-limit", "1", "--retry-attempts", "0",
                            "--timeout-seconds", "10", "--per-call-cost-limit", "1", "--total-cost-limit", "1",
                            "--max-output-tokens", "2000", "--quiz-estimated-input-tokens", "10000",
                            "--input-price-per-million", "1", "--output-price-per-million", "1", "--pricing-reference", "test"]
                    if value is not None:
                        argv += [option, value]
                    with mock.patch.object(sys, "argv", argv), mock.patch("src.pilot_runner.PilotRunner", return_value=runner), \
                         mock.patch("src.provider_adapters.urllib_transport") as transport, contextlib.redirect_stdout(io.StringIO()):
                        if value == "low":
                            with self.assertRaises(ProviderFailure):
                                main()
                            runner.run_quiz.assert_not_called()
                        else:
                            main()
                            router = runner.run_quiz.call_args.args[4]
                            effective = router.execution_policy("quiz", model, "pilot-v2")
                            self.assertEqual(getattr(effective, "reasoning_effort" if model.startswith("gpt") else "thinking_level"), "medium")
                        transport.assert_not_called()


class ProviderAdapterTest(unittest.TestCase):
    @staticmethod
    def temporary_runner(folder):
        root = Path(__file__).resolve().parents[1]
        repository = Path(folder)
        (repository / "configs").mkdir()
        (repository / "data").mkdir()
        shutil.copyfile(root / "configs" / "pilot.yaml", repository / "configs" / "pilot.yaml")
        shutil.copyfile(root / "data" / "videos.jsonl", repository / "data" / "videos.jsonl")
        return PilotRunner(repository, repository / "results")

    def test_grounding_validation_failure_keeps_returned_usage_and_estimated_cost(self):
        invalid = dict(GROUNDING, facts=[dict(GROUNDING["facts"][0], evidenceType="narration")])
        with tempfile.TemporaryDirectory() as folder:
            runner = self.temporary_runner(folder)
            row = runner.run_grounding(VIDEO["videoId"], "gemini_video", 1,
                                       self.router(FakeTransport(gemini_response(invalid))))
            self.assertEqual((row["apiStatus"], row["errorCategory"]),
                             ("error", "invalid_grounding_response"))
            self.assertEqual((row["inputTokens"], row["outputTokens"], row["thinkingTokens"]),
                             (10, 20, 3))
            self.assertAlmostEqual(row["estimatedCostUsd"], 33 / 1000000)
            self.assertEqual(row["pricingReference"], "operator-test-pricing")

    def test_grounding_validation_failure_with_usage_passes_next_storage_preflight(self):
        invalid = dict(GROUNDING, facts=[dict(GROUNDING["facts"][0], evidenceType="narration")])
        with tempfile.TemporaryDirectory() as folder:
            repository = Path(folder)
            (repository / "configs").mkdir()
            (repository / "data").mkdir()
            shutil.copyfile(Path(__file__).resolve().parents[1] / "configs" / "pilot.yaml",
                            repository / "configs" / "pilot.yaml")
            (repository / "data" / "videos.jsonl").write_text(
                json.dumps({"videoId": VIDEO["videoId"],
                            "youtubeUrl": "https://www.youtube.com/watch?v=mock-video"}) + "\n",
                encoding="utf-8")
            runner = PilotRunner(repository, repository / "results")

            first_transport = FakeTransport(gemini_response(invalid))
            failed = runner.run_grounding(VIDEO["videoId"], "gemini_video", 1,
                                          self.router(first_transport))
            self.assertEqual(len(first_transport.calls), 1)
            self.assertEqual((failed["apiStatus"], failed["errorCategory"]),
                             ("error", "invalid_grounding_response"))
            self.assertEqual((failed["inputTokens"], failed["outputTokens"]), (10, 20))
            self.assertIsNotNone(failed["estimatedCostUsd"])
            self.assertTrue((repository / "results" / "raw" / f'{failed["runId"]}.json').is_file())
            self.assertTrue((repository / "results" / "evaluation" / f'{failed["runId"]}.json').is_file())

            next_transport = FakeTransport(gemini_response(GROUNDING))
            next_row = runner.run_grounding(VIDEO["videoId"], "gemini_video", 1,
                                            self.router(next_transport))
            self.assertEqual(len(next_transport.calls), 1)
            self.assertEqual((next_row["apiStatus"], next_row["attempt"]), ("success", 2))
            rows = [json.loads(line) for line in
                    (repository / "results" / "video-grounding.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["runId"] for row in rows], [failed["runId"], next_row["runId"]])
            self.assertEqual(rows[0]["errorCategory"], "invalid_grounding_response")

    def test_incomplete_response_keeps_safe_usage_for_both_providers(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = self.temporary_runner(folder)
            gemini = gemini_response(GROUNDING, status="in_progress")
            grounding = runner.run_grounding(VIDEO["videoId"], "gemini_video", 1,
                                             self.router(FakeTransport(gemini)))
            self.assertEqual((grounding["errorCategory"], grounding["inputTokens"],
                              grounding["outputTokens"], grounding["thinkingTokens"]),
                             ("incomplete_response", 10, 20, 3))
            self.assertAlmostEqual(grounding["estimatedCostUsd"], 33 / 1000000)

            source_id = approve_test_source(runner)
            openai = {"status": "incomplete", "usage": {"input_tokens": 12, "output_tokens": 18,
                       "output_tokens_details": {"reasoning_tokens": 4}}}
            quiz = runner.run_quiz(VIDEO["videoId"], "gpt-5.4-mini", 1, CONTENT,
                                   self.router(FakeTransport(openai)),
                                   source_grounding_run_id=source_id)
            self.assertEqual((quiz["errorCategory"], quiz["inputTokens"],
                              quiz["outputTokens"], quiz["thinkingTokens"]),
                             ("incomplete_response", 12, 18, 4))
            self.assertAlmostEqual(quiz["estimatedCostUsd"], 30 / 1000000)
            self.assertEqual(quiz["pricingReference"], "operator-test-pricing")

    def test_http_error_and_malformed_or_missing_usage_do_not_create_metrics(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = self.temporary_runner(folder)
            responses = ((503, {}, "server_error"),
                         (200, {"status": "incomplete"}, "incomplete_response"),
                         (200, {"status": "incomplete", "usage": {"total_input_tokens": "10",
                                  "total_output_tokens": 20}}, "incomplete_response"))
            for status, response, category in responses:
                with self.subTest(status=status, response=response):
                    row = runner.run_grounding(VIDEO["videoId"], "gemini_video", 1,
                                               self.router(FakeTransport(response, status)))
                    self.assertEqual(row["errorCategory"], category)
                    for key in ("inputTokens", "outputTokens", "thinkingTokens",
                                "estimatedCostUsd", "pricingReference"):
                        self.assertIsNone(row[key])

    def test_partial_incomplete_usage_preserves_only_returned_tokens(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = self.temporary_runner(folder)
            response = {"status": "in_progress", "usage": {"total_input_tokens": 11}}
            row = runner.run_grounding(VIDEO["videoId"], "gemini_video", 1,
                                       self.router(FakeTransport(response)))
            self.assertEqual(row["errorCategory"], "incomplete_response")
            self.assertEqual(row["inputTokens"], 11)
            for key in ("outputTokens", "thinkingTokens", "estimatedCostUsd", "pricingReference"):
                self.assertIsNone(row[key])

    def test_completed_malformed_output_and_direct_quiz_keep_returned_usage(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = self.temporary_runner(folder)
            response = gemini_response(GROUNDING)
            response["steps"] = []
            direct = runner.run_end_to_end(VIDEO["videoId"], "gemini_direct_quiz", 1,
                                           self.router(FakeTransport(response)))
            self.assertEqual((direct["apiStatus"], direct["errorCategory"]),
                             ("error", "invalid_provider_response"))
            self.assertEqual((direct["inputTokens"], direct["outputTokens"],
                              direct["thinkingTokens"]), (10, 20, 3))
            self.assertAlmostEqual(direct["estimatedCostUsd"], 33 / 1000000)

            source_id = approve_test_source(runner)
            malformed_quiz = gemini_response(QUIZ)
            malformed_quiz["steps"][0]["content"][0]["text"] = "{broken"
            quiz = runner.run_quiz(VIDEO["videoId"], "gemini-3.8-flash", 1, CONTENT,
                                   self.router(FakeTransport(malformed_quiz)),
                                   source_grounding_run_id=source_id)
            self.assertEqual((quiz["apiStatus"], quiz["parseStatus"]), ("success", "fail"))
            self.assertEqual((quiz["inputTokens"], quiz["outputTokens"],
                              quiz["thinkingTokens"]), (10, 20, 3))
            self.assertAlmostEqual(quiz["estimatedCostUsd"], 33 / 1000000)

    @staticmethod
    def cli_runner_namespace():
        captured = {}

        def capture(frame, event, _arg):
            if (event == "call" and frame.f_code.co_name == "main"
                    and frame.f_globals.get("__name__") == "__main__"):
                captured.update(runner=frame.f_globals["PilotRunner"],
                                failure=frame.f_globals["ProviderFailure"])

        previous_profile = sys.getprofile()
        try:
            sys.setprofile(capture)
            with mock.patch.object(sys, "argv", ["src.pilot_runner", "--help"]):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    try:
                        runpy.run_module("src.pilot_runner", run_name="__main__", alter_sys=True)
                    except SystemExit as exc:
                        if exc.code != 0:
                            raise
        finally:
            sys.setprofile(previous_profile)
        return captured["runner"], captured["failure"]

    def test_python_m_runner_and_adapter_share_provider_failure_class(self):
        _, cli_failure = self.cli_runner_namespace()
        self.assertIs(cli_failure, ProviderFailure)

    def test_python_m_runner_preserves_http_failure_categories_without_secrets(self):
        cli_runner, _ = self.cli_runner_namespace()
        root = Path(__file__).resolve().parents[1]
        for status, expected in ((400, "client_error"), (401, "client_error"),
                                 (403, "client_error"), (429, "rate_limit"),
                                 (500, "server_error")):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as folder:
                repository = Path(folder)
                (repository / "configs").mkdir()
                (repository / "data").mkdir()
                shutil.copyfile(root / "configs" / "pilot.yaml", repository / "configs" / "pilot.yaml")
                shutil.copyfile(root / "data" / "videos.jsonl", repository / "data" / "videos.jsonl")
                row = cli_runner(repository, repository / "results").run_grounding(
                    VIDEO["videoId"], "gemini_video", 1,
                    self.router(FakeTransport({"error": "Bearer unrelated-secret"}, status)))
                self.assertEqual(row["errorCategory"], expected)
                self.assertEqual(row["httpStatus"], status)
                saved = (repository / "results" / "video-grounding.jsonl").read_text(encoding="utf-8")
                self.assertNotIn("gemini-test-secret", saved)
                self.assertNotIn("unrelated-secret", saved)
                self.assertNotIn("Authorization", saved)
                self.assertNotIn("request", saved)
                self.assertEqual(json.loads(saved)["httpStatus"], status)

    def test_python_m_runner_keeps_generic_exception_as_provider_error(self):
        cli_runner, _ = self.cli_runner_namespace()

        class BrokenProvider:
            is_actual_api = True

            def invoke(self, _kind, **_kwargs):
                raise RuntimeError("Bearer unrelated-secret")

        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder:
            repository = Path(folder)
            (repository / "configs").mkdir()
            (repository / "data").mkdir()
            shutil.copyfile(root / "configs" / "pilot.yaml", repository / "configs" / "pilot.yaml")
            shutil.copyfile(root / "data" / "videos.jsonl", repository / "data" / "videos.jsonl")
            row = cli_runner(repository, repository / "results").run_grounding(
                VIDEO["videoId"], "gemini_video", 1, BrokenProvider())
            self.assertEqual(row["errorCategory"], "provider_error")
            self.assertIsNone(row["httpStatus"])
            saved = (repository / "results" / "video-grounding.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("unrelated-secret", saved)

    def router(self, transport, **overrides):
        return ProviderRouter(policy(**overrides), transport=transport,
                              api_keys={"gemini": "gemini-test-secret", "openai": "openai-test-secret"})

    def test_gemini_grounding_uses_youtube_and_extracts_normalized_result(self):
        transport = FakeTransport(gemini_response(GROUNDING))
        result = self.router(transport).invoke("grounding", video=VIDEO, promptVersion=GROUNDING_V1, model="gemini-3.8-flash")
        url, headers, body, _ = transport.calls[0]
        self.assertEqual(url, "https://generativelanguage.googleapis.com/v1beta/interactions")
        self.assertEqual(body["input"][0], {"type": "video", "uri": VIDEO["youtubeUrl"], "processing": "static"})
        self.assertEqual(result["normalized"], GROUNDING)
        self.assertEqual(result["inputTokens"], 10)
        self.assertEqual(result["thinkingTokens"], 3)
        self.assertNotIn("gemini-test-secret", json.dumps(result))

    def test_grounding_request_limits_and_defines_evidence_types(self):
        transport = FakeTransport(gemini_response(GROUNDING))
        self.router(transport).invoke("grounding", video=VIDEO, promptVersion=GROUNDING_V1, model="gemini-3.8-flash")
        body = transport.calls[0][2]
        evidence_type = body["response_format"]["schema"]["properties"]["facts"]["items"]["properties"]["evidenceType"]
        self.assertEqual(evidence_type, {"type": "string", "enum": ["speech", "visual", "unknown"]})
        prompt = body["input"][1]["text"]
        for value in ("speech", "visual", "unknown"):
            self.assertIn(value, prompt)
        self.assertIn("주된 근거", prompt)
        self.assertIn("발화", prompt)
        self.assertIn("나레이션", prompt)
        self.assertIn("화면", prompt)
        self.assertIn("신뢰성 있게", prompt)

    def test_grounding_request_defines_timestamps_as_elapsed_seconds(self):
        transport = FakeTransport(gemini_response(GROUNDING))
        self.router(transport).invoke("grounding", video=VIDEO, promptVersion=GROUNDING_V1, model="gemini-3.8-flash")
        body = transport.calls[0][2]
        prompt = body["input"][1]["text"]
        for text in ("timestampStartSeconds", "timestampEndSeconds", "경과", "MM:SS",
                     "01:30.5 → 90.5 seconds", "null"):
            self.assertIn(text, prompt)
        properties = body["response_format"]["schema"]["properties"]["facts"]["items"]["properties"]
        for key in ("timestampStartSeconds", "timestampEndSeconds"):
            self.assertEqual(properties[key]["type"], ["number", "null"])
            for text in ("경과 초", "MM:SS", "01:30.5 → 90.5", "null"):
                self.assertIn(text, properties[key]["description"])

    def grounding_request(self, version):
        transport = FakeTransport(gemini_response(GROUNDING))
        self.router(transport).invoke("grounding", video=VIDEO, promptVersion=version, model="gemini-3.8-flash")
        return transport.calls[0][2]

    def test_v1_grounding_prompt_is_unchanged(self):
        # SHA-256 of the Grounding prompt text sent before Grounding prompts were versioned.
        body = self.grounding_request(GROUNDING_V1)
        prompt = body["input"][1]["text"]
        self.assertEqual(hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                         "717c36f5762f20824728e6ce41a276a2c53faa68a41d10fb0a019f21bc9a5fab")
        self.assertEqual(body["response_format"]["schema"], grounding_schema())

    def test_v2_grounding_prompt_adds_the_content_text_contract(self):
        v1, v2 = self.grounding_request(GROUNDING_V1), self.grounding_request("video-grounding-v2")
        prompt = v2["input"][1]["text"]
        self.assertTrue(prompt.startswith(v1["input"][1]["text"]))  # facts rules are kept as in v1
        for text in ("핵심 정보", "외부 지식", "숫자", "고유명사", "인과관계", "서로 구별되는",
                     "한국어", "facts와 모순", "메타 문장"):
            self.assertIn(text, prompt)
        self.assertNotIn("최소", prompt)  # no fixed minimum length
        self.assertEqual((v2["input"][0], v2["response_format"]), (v1["input"][0], v1["response_format"]))
        self.assertEqual(GROUNDING_PROMPTS["video-grounding-v2"], prompt)

    def test_v1_and_v2_grounding_prompts_and_schema_are_frozen(self):
        # SHA-256 at main df71ba7, before video-grounding-v3; historical Grounding requests keep their text.
        sha = lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest()
        self.assertEqual(sha(GROUNDING_PROMPTS["video-grounding-v1"]),
                         "717c36f5762f20824728e6ce41a276a2c53faa68a41d10fb0a019f21bc9a5fab")
        self.assertEqual(sha(GROUNDING_PROMPTS["video-grounding-v2"]),
                         "b9e8c1d3e175879743802a3289a6f0b54df1c3771021905d1b70191173acd543")
        for version in (None, GROUNDING_V1, "video-grounding-v2"):
            with self.subTest(version=version):
                self.assertEqual(sha(json.dumps(grounding_schema(version), ensure_ascii=False, sort_keys=True)),
                                 "bacaab94a50239ef13a4d8c5e53605cec75328c3a4d9d6be84419c6f4843ec5b")
                self.assertEqual(self.grounding_request(version or GROUNDING_V1)["response_format"]["schema"],
                                 grounding_schema())

    def test_v3_grounding_requests_mmss_string_timestamps(self):
        v2, v3 = self.grounding_request("video-grounding-v2"), self.grounding_request("video-grounding-v3")
        prompt = v3["input"][1]["text"]
        self.assertEqual(prompt, GROUNDING_PROMPTS["video-grounding-v3"])
        for text in ('"MM:SS"', '"MM:SS.s"', '"01:41.2"', '"03:04.5"', "00~59", "계산하지 마세요", "null"):
            self.assertIn(text, prompt)
        self.assertNotIn("timestampStartSeconds", prompt)  # the Provider never writes seconds
        self.assertTrue(prompt.endswith(GROUNDING_PROMPTS["video-grounding-v2"][len(GROUNDING_PROMPTS["video-grounding-v1"]):]))
        fact = v3["response_format"]["schema"]["properties"]["facts"]["items"]
        self.assertEqual(fact["required"], ["fact", "evidenceType", "evidence", "timestampStart", "timestampEnd"])
        self.assertEqual(set(fact["properties"]), {"fact", "evidenceType", "evidence", "timestampStart", "timestampEnd"})
        for key in ("timestampStart", "timestampEnd"):
            self.assertEqual(fact["properties"][key]["type"], ["string", "null"])
            self.assertNotIn("pattern", fact["properties"][key])  # not in Gemini's documented schema subset
            for text in ('"MM:SS"', '"01:41.2"', "00~59", "null"):
                self.assertIn(text, fact["properties"][key]["description"])
        self.assertEqual(v3["input"][0], v2["input"][0])

    def test_video_timestamp_conversion_is_deterministic(self):
        for text, seconds in (("00:57.2", 57.2), ("01:41.2", 101.2), ("02:26.7", 146.7), ("03:04.5", 184.5),
                              ("00:00", 0.0), ("01:00", 60.0), ("1:41.2", 101.2), ("59:59.999", 3599.999),
                              ("01:53.25", 113.25), ("100:00", 6000.0)):
            with self.subTest(text=text):
                self.assertEqual(video_timestamp_seconds(text), seconds)
                self.assertEqual(json.dumps(video_timestamp_seconds(text)), json.dumps(seconds))
        self.assertIsNone(video_timestamp_seconds(None))  # an unlocatable position stays null

    def test_malformed_video_timestamps_are_rejected(self):
        for value in ("01:75.0", "01:60", "-01:41.2", "01:-41.2", "+01:41.2", "abc", "01:4a.2", "141.2", "0141.2",
                      "01:41:20", "01:41.", "01:41.2345", ":41.2", "01:", "", " 01:41.2", "01:41.2\n",
                      "０１:４１.２", "1000:00", 141.2, 101, True, [], {}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                video_timestamp_seconds(value)

    def test_unknown_or_missing_grounding_prompt_version_is_rejected_before_http(self):
        for arguments in ({}, {"promptVersion": None}, {"promptVersion": "video-grounding-v9"},
                          {"promptVersion": "pilot-v2"}, {"promptVersion": [GROUNDING_V1]}):
            transport = FakeTransport(gemini_response(GROUNDING))
            with self.subTest(arguments=arguments), self.assertRaises(ProviderFailure) as failure:
                self.router(transport).invoke("grounding", video=VIDEO, model="gemini-3.8-flash", **arguments)
            self.assertEqual(failure.exception.category, "invalid_input")
            self.assertEqual(transport.calls, [])

    def test_runner_sends_the_configured_grounding_prompt_version(self):
        root = Path(__file__).resolve().parents[1]
        for config, version, payload in (("pilot.yaml", GROUNDING_V1, GROUNDING),
                                         ("pilot-v2.yaml", "video-grounding-v3", GROUNDING_V3)):
            with self.subTest(config=config), tempfile.TemporaryDirectory() as folder:
                runner = self.temporary_runner(folder)
                if config != "pilot.yaml":
                    shutil.copyfile(root / "configs" / config, runner.repository / "configs" / config)
                    runner = PilotRunner(runner.repository, runner.results, runner.repository / "configs" / config)
                transport = FakeTransport(gemini_response(payload))
                row = runner.run_grounding(VIDEO["videoId"], "gemini_video", 1, self.router(transport))
                self.assertEqual((row["apiStatus"], row["promptVersion"]), ("success", version))
                self.assertEqual(transport.calls[0][2]["input"][1]["text"], GROUNDING_PROMPTS[version])

    def test_unknown_configured_grounding_version_never_reaches_the_transport(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = self.temporary_runner(folder)
            runner.config["video_grounding"]["prompt_version"] = "video-grounding-v9"
            transport = FakeTransport(gemini_response(GROUNDING))
            with self.assertRaisesRegex(ValueError, "not a known Grounding prompt version"):
                runner.run_grounding(VIDEO["videoId"], "gemini_video", 1, self.router(transport))
            self.assertEqual(transport.calls, [])
            self.assertFalse((runner.results / "video-grounding.jsonl").exists())

    def test_configured_model_alias_uses_configured_provider_and_key_name(self):
        config = {"video_grounding": {"methods": [{"id": "gemini_video", "model": "gemini-pilot-alias",
                    "provider": "gemini", "api_key_environment_variable": "PILOT_GEMINI_KEY"}]},
                  "quiz_generation": {"models": [{"id": "gemini-pilot-alias", "provider": "gemini",
                    "api_key_environment_variable": "PILOT_GEMINI_KEY"}]},
                  "end_to_end": {"methods": [{"id": "gemini_direct_quiz", "grounding": "direct_video",
                    "quiz_model": "gemini-pilot-alias"}]}}
        transport = FakeTransport(gemini_response(GROUNDING))
        with mock.patch.dict("os.environ", {"PILOT_GEMINI_KEY": "alias-secret"}):
            router = ProviderRouter(policy(reasoning_effort=None), transport=transport, pilot_config=config)
            result = router.invoke("grounding", video=VIDEO, promptVersion=GROUNDING_V1, model="gemini-pilot-alias")
        self.assertEqual(result["normalized"], GROUNDING)
        self.assertEqual(transport.calls[0][1]["x-goog-api-key"], "alias-secret")
        self.assertEqual(transport.calls[0][2]["model"], "gemini-pilot-alias")

    def test_actual_usage_produces_cost_with_reference_and_gemini_thinking(self):
        result = self.router(FakeTransport(gemini_response(GROUNDING)),
                             input_price_per_million=2.0, output_price_per_million=3.0).invoke(
                                 "grounding", video=VIDEO, promptVersion=GROUNDING_V1, model="gemini-3.8-flash")
        self.assertAlmostEqual(result["estimatedCostUsd"], (10 * 2 + (20 + 3) * 3) / 1000000)
        self.assertEqual(result["pricingReference"], "operator-test-pricing")

    def test_mixed_provider_prices_are_applied_to_each_response(self):
        prices = {"gemini": (2.0, 3.0, "gemini-price-source"),
                  "openai": (5.0, 7.0, "openai-price-source")}
        gemini = self.router(FakeTransport(gemini_response(GROUNDING)),
                             provider_prices=prices).invoke("grounding", video=VIDEO, promptVersion=GROUNDING_V1,
                                                             model="gemini-3.8-flash")
        response = {"status": "completed", "output": [{"type": "message", "content": [
                    {"type": "output_text", "text": json.dumps(QUIZ)}]}],
                    "usage": {"input_tokens": 11, "output_tokens": 22,
                              "output_tokens_details": {"reasoning_tokens": 2}}}
        openai = self.router(FakeTransport(response), provider_prices=prices).invoke(
            "quiz", model="gpt-5.4-mini", contentText=CONTENT, promptVersion="pilot-v1",
            questionCount=3, optionCount=4)
        self.assertAlmostEqual(gemini["estimatedCostUsd"], (10 * 2 + 23 * 3) / 1000000)
        self.assertAlmostEqual(openai["estimatedCostUsd"], (11 * 5 + 22 * 7) / 1000000)
        self.assertEqual(openai["pricingReference"], "openai-price-source")

    def test_actual_usage_replaces_preflight_reservation_before_next_call(self):
        transport = FakeTransport(gemini_response(GROUNDING))
        router = self.router(transport, call_limit=2, video_estimated_input_tokens=1,
                             max_output_tokens=1, total_cost_limit=0.000034)
        router.invoke("grounding", video=VIDEO, promptVersion=GROUNDING_V1, model="gemini-3.8-flash")
        with self.assertRaises(ProviderFailure) as failure:
            router.invoke("grounding", video=VIDEO, promptVersion=GROUNDING_V1, model="gemini-3.8-flash")
        self.assertEqual(failure.exception.category, "live_guard")
        self.assertEqual(len(transport.calls), 1)

    def test_pre_call_estimate_uses_the_selected_input_stage(self):
        guard = policy(video_estimated_input_tokens=1000, quiz_estimated_input_tokens=100,
                       max_output_tokens=1)
        self.assertAlmostEqual(guard.authorize("grounding", 0, 0, "gemini"), 0.001001)
        self.assertAlmostEqual(guard.authorize("direct", 0, 0, "gemini"), 0.001001)
        self.assertAlmostEqual(guard.authorize("quiz", 0, 0, "gemini"), 0.000101)

    def test_each_kind_requires_only_its_own_estimate_before_http(self):
        grounding = FakeTransport(gemini_response(GROUNDING))
        self.router(grounding, quiz_estimated_input_tokens=None).invoke(
            "grounding", video=VIDEO, promptVersion=GROUNDING_V1, model="gemini-3.8-flash")
        self.assertEqual(len(grounding.calls), 1)

        quiz = FakeTransport(gemini_response(QUIZ))
        self.router(quiz, video_estimated_input_tokens=None).invoke(
            "quiz", model="gemini-3.8-flash", contentText=CONTENT,
            promptVersion="pilot-v1", questionCount=3, optionCount=4)
        self.assertEqual(len(quiz.calls), 1)

        direct = FakeTransport(gemini_response(QUIZ))
        self.router(direct, quiz_estimated_input_tokens=None).invoke(
            "direct", video=VIDEO, model="gemini-3.8-flash",
            promptVersion="pilot-v1", questionCount=3, optionCount=4)
        self.assertEqual(len(direct.calls), 1)

        for kind, missing, arguments in (
                ("grounding", {"video_estimated_input_tokens": None},
                 {"video": VIDEO, "promptVersion": GROUNDING_V1}),
                ("quiz", {"quiz_estimated_input_tokens": None},
                 {"contentText": CONTENT, "promptVersion": "pilot-v1",
                  "questionCount": 3, "optionCount": 4}),
                ("direct", {"video_estimated_input_tokens": None},
                 {"video": VIDEO, "promptVersion": "pilot-v1",
                  "questionCount": 3, "optionCount": 4})):
            transport = FakeTransport({})
            with self.subTest(kind=kind), self.assertRaises(ProviderFailure) as failure:
                self.router(transport, **missing).invoke(
                    kind, model="gemini-3.8-flash", **arguments)
            self.assertEqual(failure.exception.category, "live_guard")
            self.assertEqual(transport.calls, [])

    def test_approved_separate_grounding_and_quiz_use_stage_estimates(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder:
            repository = Path(folder)
            (repository / "configs").mkdir()
            (repository / "data").mkdir()
            shutil.copyfile(root / "configs" / "pilot.yaml", repository / "configs" / "pilot.yaml")
            shutil.copyfile(root / "data" / "videos.jsonl", repository / "data" / "videos.jsonl")
            runner = PilotRunner(repository, repository / "results")
            transport = QueueTransport((gemini_response(GROUNDING), gemini_response(QUIZ)))
            router = self.router(transport, call_limit=2, max_output_tokens=1,
                                 video_estimated_input_tokens=1000,
                                 quiz_estimated_input_tokens=100,
                                 total_cost_limit=0.00102)
            grounding = runner.run_grounding(VIDEO["videoId"], "gemini_video", 1, router)
            runner.approve_content(grounding["runId"], APPROVER)
            row = runner.run_quiz(VIDEO["videoId"], "gemini-3.8-flash", 1, CONTENT,
                                  router, source_grounding_run_id=grounding["runId"])
            self.assertEqual(row["apiStatus"], "success")
            self.assertEqual(len(transport.calls), 2)
            self.assertAlmostEqual(router.reserved_cost, 0.000066)

    def test_openai_requires_reasoning_but_gemini_does_not(self):
        gemini = self.router(FakeTransport(gemini_response(GROUNDING)), reasoning_effort=None)
        self.assertEqual(gemini.invoke("grounding", video=VIDEO, promptVersion=GROUNDING_V1,
                                      model="gemini-3.8-flash")["normalized"], GROUNDING)
        transport = FakeTransport({})
        with self.assertRaises(ProviderFailure) as failure:
            self.router(transport, reasoning_effort=None).invoke(
                "quiz", model="gpt-5.4-mini", contentText=CONTENT, promptVersion="pilot-v1",
                questionCount=3, optionCount=4)
        self.assertEqual(failure.exception.category, "live_guard")
        self.assertEqual(transport.calls, [])

    def test_gemini_quiz_and_direct_use_structured_output(self):
        for kind, arguments in (("quiz", {"contentText": CONTENT}), ("direct", {"video": VIDEO})):
            with self.subTest(kind=kind):
                transport = FakeTransport(gemini_response(QUIZ))
                router = self.router(transport)
                result = router.invoke(kind, model="gemini-3.8-flash", promptVersion="pilot-v1",
                                       questionCount=3, optionCount=4, **arguments)
                body = transport.calls[0][2]
                self.assertEqual(body["response_format"]["type"], "text")
                self.assertEqual(body["response_format"]["mime_type"], "application/json")
                self.assertIn("schema", body["response_format"])
                self.assertEqual(result["normalized"], QUIZ)
                self.assertIn(CONTENT if kind == "quiz" else "youtube.com", json.dumps(body))

    def test_openai_responses_quiz_uses_strict_schema_and_extracts_usage(self):
        transport = FakeTransport({"status": "completed", "model": "gpt-5.4-mini", "output": [{
            "type": "message", "content": [{"type": "output_text", "text": json.dumps(QUIZ)}]}],
            "usage": {"input_tokens": 11, "output_tokens": 22,
                      "output_tokens_details": {"reasoning_tokens": 2}}})
        result = self.router(transport).invoke("quiz", contentText=CONTENT, model="gpt-5.4-mini",
                                               promptVersion="pilot-v1", questionCount=3, optionCount=4)
        url, headers, body, _ = transport.calls[0]
        self.assertEqual(url, "https://api.openai.com/v1/responses")
        self.assertEqual(body["text"]["format"]["type"], "json_schema")
        self.assertTrue(body["text"]["format"]["strict"])
        self.assertEqual(body["reasoning"]["effort"], "none")
        self.assertIn(CONTENT, body["input"])
        self.assertEqual(result["normalized"], QUIZ)
        self.assertEqual(result["outputTokens"], 22)
        self.assertNotIn("openai-test-secret", json.dumps(result))

    def test_live_guard_blocks_before_transport(self):
        transport = FakeTransport(gemini_response(QUIZ))
        for config in ({"enabled": False}, {"per_call_cost_limit": None},
                       {"pricing_reference": None}, {"video_processing": None}):
            with self.subTest(config=config), self.assertRaises(ProviderFailure):
                self.router(transport, **config).invoke("direct", video=VIDEO,
                    model="gemini-3.8-flash", promptVersion="pilot-v1", questionCount=3, optionCount=4)
        self.assertEqual(transport.calls, [])

    def test_bad_status_and_http_errors_are_classified(self):
        for status, category in ((429, "rate_limit"), (400, "client_error"), (500, "server_error")):
            with self.subTest(status=status), self.assertRaises(ProviderFailure) as failure:
                self.router(FakeTransport({}, status)).invoke("grounding", video=VIDEO, promptVersion=GROUNDING_V1,
                                                              model="gemini-3.8-flash")
            self.assertEqual(failure.exception.category, category)
        with self.assertRaises(ProviderFailure) as failure:
            self.router(FakeTransport(gemini_response(QUIZ, "incomplete"))).invoke(
                "quiz", model="gemini-3.8-flash", contentText=CONTENT,
                promptVersion="pilot-v1", questionCount=3, optionCount=4)
        self.assertEqual(failure.exception.category, "incomplete_response")

    def test_gemini_http_error_keeps_only_allowlisted_code(self):
        cases = (
            (b'{"error":{"code":"service_unavailable","message":"Bearer unrelated-secret"}}',
             "service_unavailable"),
            (b"<html>Bearer unrelated-secret</html>", None),
            (b'{"error":{"code":', None),
            (b'{"error":{"code":"Bearer unrelated-secret","message":"secret"}}', None),
            (b'{"error":{"code":503,"message":"secret"}}', None),
            (b'{"error":{"code":"service_unavailable","message":"' + b"x" * 17000 + b'"}}', None),
        )
        root = Path(__file__).resolve().parents[1]
        for body, expected in cases:
            with self.subTest(expected=expected, body=body[:12]), tempfile.TemporaryDirectory() as folder:
                repository = Path(folder)
                (repository / "configs").mkdir()
                (repository / "data").mkdir()
                shutil.copyfile(root / "configs" / "pilot.yaml", repository / "configs" / "pilot.yaml")
                shutil.copyfile(root / "data" / "videos.jsonl", repository / "data" / "videos.jsonl")
                error = urllib.error.HTTPError("https://generativelanguage.googleapis.com/v1beta/interactions",
                                               503, "error", {"x-test": "header-secret"}, io.BytesIO(body))
                with mock.patch("src.provider_adapters.urllib.request.urlopen", side_effect=error):
                    row = PilotRunner(repository, repository / "results").run_end_to_end(
                        VIDEO["videoId"], "gemini_direct_quiz", 1, self.router(urllib_transport))
                self.assertEqual((row["errorCategory"], row["httpStatus"]), ("server_error", 503))
                self.assertEqual(row["providerErrorCode"], expected)
                saved = (repository / "results" / "end-to-end.jsonl").read_text(encoding="utf-8")
                self.assertNotIn("unrelated-secret", saved)
                self.assertNotIn("header-secret", saved)
                self.assertNotIn("gemini-test-secret", saved)
                self.assertNotIn("message", saved)
                self.assertFalse((repository / "results" / "raw" / (row["runId"] + ".json")).exists())

    def test_provider_failure_old_constructor_and_retry_keep_safe_code(self):
        old = ProviderFailure("server_error", 503)
        self.assertIsNone(old.provider_error_code)
        transport = FakeTransport({"providerErrorCode": "service_unavailable"}, 503)
        with self.assertRaises(ProviderFailure) as failure:
            self.router(transport, retry_attempts=2, call_limit=1).invoke(
                "grounding", video=VIDEO, promptVersion=GROUNDING_V1, model="gemini-3.8-flash")
        self.assertEqual((failure.exception.category, failure.exception.http_status,
                          failure.exception.provider_error_code),
                         ("server_error", 503, "service_unavailable"))
        self.assertEqual(len(transport.calls), 1)

    def test_gemini_client_error_keeps_code_and_success_has_null_code(self):
        transport = FakeTransport({"providerErrorCode": "invalid_request"}, 400)
        with self.assertRaises(ProviderFailure) as failure:
            self.router(transport).invoke("grounding", video=VIDEO, promptVersion=GROUNDING_V1, model="gemini-3.8-flash")
        self.assertEqual((failure.exception.category, failure.exception.http_status,
                          failure.exception.provider_error_code),
                         ("client_error", 400, "invalid_request"))
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder:
            repository = Path(folder)
            (repository / "configs").mkdir()
            (repository / "data").mkdir()
            shutil.copyfile(root / "configs" / "pilot.yaml", repository / "configs" / "pilot.yaml")
            shutil.copyfile(root / "data" / "videos.jsonl", repository / "data" / "videos.jsonl")
            row = PilotRunner(repository, repository / "results").run_grounding(
                VIDEO["videoId"], "gemini_video", 1,
                self.router(FakeTransport(gemini_response(GROUNDING))))
            self.assertIsNone(row["providerErrorCode"])

    def test_provider_error_code_schema_is_optional_for_legacy_rows(self):
        schema = json.loads((Path(__file__).resolve().parents[1] / "docs" /
                             "run-result.schema.json").read_text(encoding="utf-8"))
        self.assertNotIn("providerErrorCode", schema["required"])
        self.assertEqual(schema["properties"]["providerErrorCode"]["type"], ["string", "null"])
        self.assertIn("providerErrorCode", schema["properties"])
        legacy = {"benchmarkType": "end_to_end", "runId": "legacy", "videoId": VIDEO["videoId"],
                  "method": "gemini_direct_quiz", "model": "gemini-3.8-flash",
                  "promptVersion": "pilot-v1", "repetition": 1, "startedAt": "2026-10-01T00:00:00Z",
                  "apiStatus": "error", "errorCategory": "server_error", "httpStatus": 503,
                  "groundingRunId": None, "quizRunId": None,
                  "beCompatibility": "not_applicable", "questionReviews": []}
        self.assertTrue(set(schema["required"]).issubset(legacy))
        self.assertTrue(set(schema["oneOf"][2]["required"]).issubset(legacy))
        self.assertTrue(set(legacy).issubset(schema["properties"]))

    def test_malformed_envelope_and_network_error_are_classified(self):
        with self.assertRaises(ProviderFailure) as failure:
            self.router(FakeTransport({"status": "completed", "steps": []})).invoke(
                "grounding", video=VIDEO, promptVersion=GROUNDING_V1, model="gemini-3.8-flash")
        self.assertEqual(failure.exception.category, "invalid_provider_response")
        def broken(*args):
            raise OSError("secret in exception must not escape")
        with self.assertRaises(ProviderFailure) as failure:
            self.router(broken).invoke("grounding", video=VIDEO, promptVersion=GROUNDING_V1, model="gemini-3.8-flash")
        self.assertEqual(failure.exception.category, "network_error")

    def test_mock_live_router_integrates_with_runner_and_keeps_human_reviews_null(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder:
            repository = Path(folder)
            (repository / "configs").mkdir()
            (repository / "data").mkdir()
            shutil.copyfile(root / "configs" / "pilot.yaml", repository / "configs" / "pilot.yaml")
            shutil.copyfile(root / "data" / "videos.jsonl", repository / "data" / "videos.jsonl")
            runner = PilotRunner(repository, repository / "results")
            source_id = approve_test_source(runner)
            for model, response in (("gemini-3.8-flash", gemini_response(QUIZ)),
                                    ("gpt-5.4-mini", {"status": "completed", "output": [{
                                        "type": "message", "content": [{"type": "output_text", "text": json.dumps(QUIZ)}]}],
                                        "usage": {"input_tokens": 10, "output_tokens": 20}})):
                with self.subTest(model=model):
                    router = self.router(FakeTransport(response))
                    row = runner.run_quiz(VIDEO["videoId"], model, 1, CONTENT, router,
                                          source_grounding_run_id=source_id)
                    self.assertEqual(row["apiStatus"], "success")
                    self.assertIsNotNone(row["estimatedCostUsd"])
                    self.assertEqual(row["pricingReference"], "operator-test-pricing")
                    self.assertEqual(row["validatorStatus"], "pass")
                    self.assertEqual(row["beCompatibility"], "pass")
                    self.assertTrue(all(review["answerAccuracy"] is None and
                                        review["evidenceSupportsAnswer"] is None
                                        for review in row["questionReviews"]))
                    saved = json.loads((repository / "results" / "raw" /
                                        (row["runId"] + ".json")).read_text(encoding="utf-8"))
                    self.assertEqual(saved["status"], "completed")
                    self.assertNotIn("request", saved)
                    self.assertNotIn("gemini-test-secret", json.dumps(saved))
                    self.assertNotIn("openai-test-secret", json.dumps(saved))

    def test_retry_cannot_exceed_http_call_limit(self):
        for status, category in ((429, "rate_limit"), (500, "server_error")):
            with self.subTest(status=status):
                transport = FakeTransport({"error": "Bearer unrelated-secret"}, status)
                with self.assertRaises(ProviderFailure) as failure:
                    self.router(transport, retry_attempts=2, call_limit=1).invoke(
                        "grounding", video=VIDEO, promptVersion=GROUNDING_V1, model="gemini-3.8-flash")
                self.assertEqual((failure.exception.category, failure.exception.http_status),
                                 (category, status))
                self.assertEqual(failure.exception.retry_stop_reason, "live_guard")
                self.assertEqual(len(transport.calls), 1)
                with tempfile.TemporaryDirectory() as folder:
                    repository = Path(folder)
                    (repository / "configs").mkdir()
                    (repository / "data").mkdir()
                    root = Path(__file__).resolve().parents[1]
                    shutil.copyfile(root / "configs" / "pilot.yaml", repository / "configs" / "pilot.yaml")
                    shutil.copyfile(root / "data" / "videos.jsonl", repository / "data" / "videos.jsonl")
                    row = PilotRunner(repository, repository / "results").run_grounding(
                        VIDEO["videoId"], "gemini_video", 1,
                        self.router(FakeTransport({"error": "Bearer unrelated-secret"}, status),
                                    retry_attempts=2, call_limit=1))
                    self.assertEqual((row["errorCategory"], row["httpStatus"]), (category, status))
                    self.assertEqual(row["retryStopReason"], "live_guard")
                    saved = (repository / "results" / "video-grounding.jsonl").read_text(encoding="utf-8")
                    self.assertNotIn("unrelated-secret", saved)
                    self.assertNotIn("gemini-test-secret", saved)

    def test_first_call_guard_rejection_remains_live_guard_without_http_status(self):
        transport = FakeTransport({}, 429)
        with self.assertRaises(ProviderFailure) as failure:
            self.router(transport, retry_attempts=2, call_limit=0).invoke(
                "grounding", video=VIDEO, promptVersion=GROUNDING_V1, model="gemini-3.8-flash")
        self.assertEqual((failure.exception.category, failure.exception.http_status),
                         ("live_guard", None))
        self.assertIsNone(failure.exception.retry_stop_reason)
        self.assertEqual(transport.calls, [])
        with tempfile.TemporaryDirectory() as folder:
            row = self.temporary_runner(folder).run_grounding(
                VIDEO["videoId"], "gemini_video", 1,
                self.router(transport, retry_attempts=2, call_limit=0))
        self.assertEqual((row["errorCategory"], row["httpStatus"], row["retryStopReason"]),
                         ("live_guard", None, None))
        self.assertEqual(transport.calls, [])

    def test_retry_stop_keeps_last_provider_failure_for_each_runner_path(self):
        schema_path = Path(__file__).resolve().parents[1] / "docs" / "run-result.schema.json"
        rules = json.loads(schema_path.read_text(encoding="utf-8"))["allOf"]
        invariant = next(rule["then"]["properties"] for rule in rules
                         if "retryStopReason" in rule["if"].get("required", []))
        for status, category, code in ((429, "rate_limit", "rate_limit_exceeded"),
                                       (503, "server_error", "service_unavailable")):
            for path in ("grounding", "quiz", "direct"):
                with self.subTest(status=status, path=path), \
                        tempfile.TemporaryDirectory() as folder:
                    runner = self.temporary_runner(folder)
                    source_id = approve_test_source(runner) if path == "quiz" else None
                    transport = FakeTransport({"providerErrorCode": code}, status)
                    router = self.router(transport, retry_attempts=2, call_limit=1)
                    if path == "grounding":
                        row = runner.run_grounding(VIDEO["videoId"], "gemini_video", 1, router)
                    elif path == "quiz":
                        row = runner.run_quiz(VIDEO["videoId"], "gemini-3.8-flash", 1, CONTENT,
                                              router, source_grounding_run_id=source_id)
                    else:
                        row = runner.run_end_to_end(VIDEO["videoId"], "gemini_direct_quiz", 1,
                                                    router)
                    self.assertEqual((row["apiStatus"], row["errorCategory"], row["httpStatus"],
                                      row["providerErrorCode"], row["retryStopReason"]),
                                     ("error", category, status, code, "live_guard"))
                    self.assertEqual(len(transport.calls), 1)
                    # The production row satisfies the schema's retryStopReason invariant.
                    self.assertEqual(row["apiStatus"], invariant["apiStatus"]["const"])
                    self.assertIs(type(row["httpStatus"]), int)
                    self.assertNotEqual(row["errorCategory"], invariant["errorCategory"]["not"]["const"])

    def test_cost_guard_stops_retry_after_server_error(self):
        # Each call reserves (10000 + 2000) / 1e6 = 0.012 USD, so the second call exceeds 0.02.
        transport = FakeTransport({"providerErrorCode": "service_unavailable"}, 503)
        with self.assertRaises(ProviderFailure) as failure:
            self.router(transport, retry_attempts=2, call_limit=5, total_cost_limit=0.02).invoke(
                "grounding", video=VIDEO, promptVersion=GROUNDING_V1, model="gemini-3.8-flash")
        self.assertEqual((failure.exception.category, failure.exception.http_status,
                          failure.exception.provider_error_code, failure.exception.retry_stop_reason),
                         ("server_error", 503, "service_unavailable", "live_guard"))
        self.assertEqual(len(transport.calls), 1)

    def test_exhausted_retries_do_not_record_retry_stop_reason(self):
        transport = FakeTransport({"providerErrorCode": "service_unavailable"}, 503)
        with tempfile.TemporaryDirectory() as folder:
            row = self.temporary_runner(folder).run_grounding(
                VIDEO["videoId"], "gemini_video", 1,
                self.router(transport, retry_attempts=1, call_limit=5))
        self.assertEqual((row["errorCategory"], row["httpStatus"], row["retryStopReason"]),
                         ("server_error", 503, None))
        self.assertEqual(len(transport.calls), 2)

    def test_successful_live_row_has_null_retry_stop_reason(self):
        with tempfile.TemporaryDirectory() as folder:
            row = self.temporary_runner(folder).run_grounding(
                VIDEO["videoId"], "gemini_video", 1,
                self.router(FakeTransport(gemini_response(GROUNDING))))
        self.assertEqual(row["apiStatus"], "success")
        self.assertIn("retryStopReason", row)
        self.assertIsNone(row["retryStopReason"])

    def test_provider_echo_keeps_evaluation_output_but_not_persisted_metadata(self):
        for echoed in ("gemini-test-secret", "Bearer unrelated-credential"):
            with self.subTest(echoed=echoed):
                payload = dict(QUIZ, questions=[dict(question, sourceEvidence=echoed)
                                                for question in QUIZ["questions"]])
                result = self.router(FakeTransport(gemini_response(payload))).invoke(
                    "quiz", contentText=CONTENT + " " + echoed, model="gemini-3.8-flash",
                    promptVersion="pilot-v1", questionCount=3, optionCount=4)
                self.assertEqual(result["normalized"], payload)
                self.assertNotIn(echoed, json.dumps(result["responseBody"]))

    def test_echoed_credentials_never_enter_raw_file_while_quiz_remains_reviewable(self):
        root = Path(__file__).resolve().parents[1]
        for echoed in ("gemini-test-secret", "Bearer unrelated-credential"):
            with self.subTest(echoed=echoed), tempfile.TemporaryDirectory() as folder:
                repository = Path(folder)
                (repository / "configs").mkdir()
                (repository / "data").mkdir()
                shutil.copyfile(root / "configs" / "pilot.yaml", repository / "configs" / "pilot.yaml")
                shutil.copyfile(root / "data" / "videos.jsonl", repository / "data" / "videos.jsonl")
                payload = dict(QUIZ, questions=[dict(question, sourceEvidence=echoed)
                                                for question in QUIZ["questions"]])
                runner = PilotRunner(repository, repository / "results")
                source_id = approve_test_source(runner, CONTENT + " " + echoed)
                row = runner.run_quiz(VIDEO["videoId"], "gemini-3.8-flash", 1,
                                      CONTENT + " " + echoed,
                                      self.router(FakeTransport(gemini_response(payload))),
                                      source_grounding_run_id=source_id)
                self.assertEqual(row["validatorStatus"], "pass")
                raw_file = repository / "results" / "raw" / (row["runId"] + ".json")
                evaluation_file = repository / "results" / "evaluation" / (row["runId"] + ".json")
                self.assertNotIn(echoed, raw_file.read_text(encoding="utf-8"))
                self.assertEqual(json.loads(evaluation_file.read_text(encoding="utf-8")), payload)

    def test_separate_approved_results_link_runs_and_direct_never_claims_be_validation(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder:
            repository = Path(folder)
            (repository / "configs").mkdir()
            (repository / "data").mkdir()
            shutil.copyfile(root / "configs" / "pilot.yaml", repository / "configs" / "pilot.yaml")
            shutil.copyfile(root / "data" / "videos.jsonl", repository / "data" / "videos.jsonl")
            runner = PilotRunner(repository, repository / "results")
            transport = QueueTransport((gemini_response(GROUNDING), gemini_response(QUIZ)))
            router = self.router(transport, call_limit=2)
            grounding = runner.run_grounding(VIDEO["videoId"], "gemini_video", 1, router)
            runner.approve_content(grounding["runId"], APPROVER)
            quiz = runner.run_quiz(VIDEO["videoId"], "gemini-3.8-flash", 1,
                                   CONTENT, router, source_grounding_run_id=grounding["runId"])
            self.assertEqual(quiz["sourceGroundingRunId"], grounding["runId"])
            self.assertEqual(quiz["beCompatibility"], "pass")
            self.assertIsNotNone(quiz["estimatedCostUsd"])
            direct = runner.run_end_to_end(VIDEO["videoId"], "gemini_direct_quiz", 1,
                                           self.router(FakeTransport(gemini_response(QUIZ))))
            self.assertEqual(direct["beCompatibility"], "not_applicable")
            self.assertEqual(direct["validatorStatus"], "not_run")

    def test_malformed_model_quiz_json_is_parse_failure_not_http_failure(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder:
            repository = Path(folder)
            (repository / "configs").mkdir()
            (repository / "data").mkdir()
            shutil.copyfile(root / "configs" / "pilot.yaml", repository / "configs" / "pilot.yaml")
            shutil.copyfile(root / "data" / "videos.jsonl", repository / "data" / "videos.jsonl")
            runner = PilotRunner(repository, repository / "results")
            source_id = approve_test_source(runner)
            envelope = {"status": "completed", "steps": [{"type": "model_output",
                        "content": [{"type": "text", "text": "{broken"}]}]}
            row = runner.run_quiz(VIDEO["videoId"], "gemini-3.8-flash", 1, CONTENT,
                                  self.router(FakeTransport(envelope)),
                                  source_grounding_run_id=source_id)
            self.assertEqual(row["apiStatus"], "success")
            self.assertEqual(row["parseStatus"], "fail")
            self.assertEqual(row["errorCategory"], "parse_error")

    def test_retry_only_for_429_and_5xx_and_timeout_is_classified(self):
        for first_status in (429, 500):
            class RecoveringTransport(FakeTransport):
                def __call__(self, url, headers, body, timeout):
                    self.calls.append((url, headers, body, timeout))
                    return ((first_status, {}) if len(self.calls) == 1
                            else (200, gemini_response(GROUNDING)))
            with self.subTest(first_status=first_status):
                transport = RecoveringTransport(None)
                result = self.router(transport, call_limit=2, retry_attempts=1).invoke(
                    "grounding", video=VIDEO, promptVersion=GROUNDING_V1, model="gemini-3.8-flash")
                self.assertEqual(result["normalized"], GROUNDING)
                self.assertEqual(len(transport.calls), 2)
        client_error = FakeTransport({}, 400)
        with self.assertRaises(ProviderFailure):
            self.router(client_error, call_limit=2, retry_attempts=1).invoke(
                "grounding", video=VIDEO, promptVersion=GROUNDING_V1, model="gemini-3.8-flash")
        self.assertEqual(len(client_error.calls), 1)
        def timeout(*args):
            raise TimeoutError()
        with self.assertRaises(ProviderFailure) as failure:
            self.router(timeout, call_limit=2, retry_attempts=1).invoke(
                "grounding", video=VIDEO, promptVersion=GROUNDING_V1, model="gemini-3.8-flash")
        self.assertEqual(failure.exception.category, "timeout")

    def test_model_cannot_override_fixed_prompt_version(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder:
            repository = Path(folder)
            (repository / "configs").mkdir()
            (repository / "data").mkdir()
            shutil.copyfile(root / "configs" / "pilot.yaml", repository / "configs" / "pilot.yaml")
            shutil.copyfile(root / "data" / "videos.jsonl", repository / "data" / "videos.jsonl")
            runner = PilotRunner(repository, repository / "results")
            source_id = approve_test_source(runner)
            row = runner.run_quiz(
                VIDEO["videoId"], "gemini-3.8-flash", 1, CONTENT,
                self.router(FakeTransport(gemini_response(dict(QUIZ, promptVersion="other")))),
                source_grounding_run_id=source_id)
            self.assertEqual(row["validatorStatus"], "fail")
            self.assertEqual(row["beCompatibility"], "fail")


if __name__ == "__main__":
    unittest.main()
