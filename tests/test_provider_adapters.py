import contextlib
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
from src.provider_adapters import LivePolicy, ProviderRouter, urllib_transport


CONTENT = "NASA measures global rain and snow every 30 minutes."
APPROVER = "pilot-reviewer"
VIDEO = {"videoId": "nasa-water-cycle-2019", "youtubeUrl": "https://www.youtube.com/watch?v=mCNcxu8MXzo"}
QUIZ = {"promptVersion": "pilot-v1", "questions": [
    {"question": f"질문 {i}?", "options": ["가", "나", "다", "라"],
     "correctOptionIndex": 0, "explanation": "자료 기준", "sourceEvidence": "global rain and snow"}
    for i in range(3)]}
GROUNDING = {"contentText": CONTENT, "facts": [{
    "fact": "NASA measures rainfall", "evidenceType": "speech", "evidence": "global rain and snow",
    "timestampStartSeconds": 61, "timestampEndSeconds": 73}]}


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
        result = self.router(transport).invoke("grounding", video=VIDEO, model="gemini-3.8-flash")
        url, headers, body, _ = transport.calls[0]
        self.assertEqual(url, "https://generativelanguage.googleapis.com/v1beta/interactions")
        self.assertEqual(body["input"][0], {"type": "video", "uri": VIDEO["youtubeUrl"], "processing": "static"})
        self.assertEqual(result["normalized"], GROUNDING)
        self.assertEqual(result["inputTokens"], 10)
        self.assertEqual(result["thinkingTokens"], 3)
        self.assertNotIn("gemini-test-secret", json.dumps(result))

    def test_grounding_request_limits_and_defines_evidence_types(self):
        transport = FakeTransport(gemini_response(GROUNDING))
        self.router(transport).invoke("grounding", video=VIDEO, model="gemini-3.8-flash")
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
        self.router(transport).invoke("grounding", video=VIDEO, model="gemini-3.8-flash")
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
            result = router.invoke("grounding", video=VIDEO, model="gemini-pilot-alias")
        self.assertEqual(result["normalized"], GROUNDING)
        self.assertEqual(transport.calls[0][1]["x-goog-api-key"], "alias-secret")
        self.assertEqual(transport.calls[0][2]["model"], "gemini-pilot-alias")

    def test_actual_usage_produces_cost_with_reference_and_gemini_thinking(self):
        result = self.router(FakeTransport(gemini_response(GROUNDING)),
                             input_price_per_million=2.0, output_price_per_million=3.0).invoke(
                                 "grounding", video=VIDEO, model="gemini-3.8-flash")
        self.assertAlmostEqual(result["estimatedCostUsd"], (10 * 2 + (20 + 3) * 3) / 1000000)
        self.assertEqual(result["pricingReference"], "operator-test-pricing")

    def test_mixed_provider_prices_are_applied_to_each_response(self):
        prices = {"gemini": (2.0, 3.0, "gemini-price-source"),
                  "openai": (5.0, 7.0, "openai-price-source")}
        gemini = self.router(FakeTransport(gemini_response(GROUNDING)),
                             provider_prices=prices).invoke("grounding", video=VIDEO,
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
        router.invoke("grounding", video=VIDEO, model="gemini-3.8-flash")
        with self.assertRaises(ProviderFailure) as failure:
            router.invoke("grounding", video=VIDEO, model="gemini-3.8-flash")
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
            "grounding", video=VIDEO, model="gemini-3.8-flash")
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
                 {"video": VIDEO}),
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
        self.assertEqual(gemini.invoke("grounding", video=VIDEO,
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
                self.router(FakeTransport({}, status)).invoke("grounding", video=VIDEO,
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
                "grounding", video=VIDEO, model="gemini-3.8-flash")
        self.assertEqual((failure.exception.category, failure.exception.http_status,
                          failure.exception.provider_error_code),
                         ("server_error", 503, "service_unavailable"))
        self.assertEqual(len(transport.calls), 1)

    def test_gemini_client_error_keeps_code_and_success_has_null_code(self):
        transport = FakeTransport({"providerErrorCode": "invalid_request"}, 400)
        with self.assertRaises(ProviderFailure) as failure:
            self.router(transport).invoke("grounding", video=VIDEO, model="gemini-3.8-flash")
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
                "grounding", video=VIDEO, model="gemini-3.8-flash")
        self.assertEqual(failure.exception.category, "invalid_provider_response")
        def broken(*args):
            raise OSError("secret in exception must not escape")
        with self.assertRaises(ProviderFailure) as failure:
            self.router(broken).invoke("grounding", video=VIDEO, model="gemini-3.8-flash")
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
                        "grounding", video=VIDEO, model="gemini-3.8-flash")
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
                "grounding", video=VIDEO, model="gemini-3.8-flash")
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
                "grounding", video=VIDEO, model="gemini-3.8-flash")
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
                    "grounding", video=VIDEO, model="gemini-3.8-flash")
                self.assertEqual(result["normalized"], GROUNDING)
                self.assertEqual(len(transport.calls), 2)
        client_error = FakeTransport({}, 400)
        with self.assertRaises(ProviderFailure):
            self.router(client_error, call_limit=2, retry_attempts=1).invoke(
                "grounding", video=VIDEO, model="gemini-3.8-flash")
        self.assertEqual(len(client_error.calls), 1)
        def timeout(*args):
            raise TimeoutError()
        with self.assertRaises(ProviderFailure) as failure:
            self.router(timeout, call_limit=2, retry_attempts=1).invoke(
                "grounding", video=VIDEO, model="gemini-3.8-flash")
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
