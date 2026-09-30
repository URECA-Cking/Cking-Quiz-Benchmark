import json
import shutil
import tempfile
import unittest
from pathlib import Path

from src.pilot_runner import PilotRunner, ProviderFailure
from src.provider_adapters import LivePolicy, ProviderRouter


CONTENT = "NASA measures global rain and snow every 30 minutes."
VIDEO = {"videoId": "nasa-water-cycle-2019", "youtubeUrl": "https://www.youtube.com/watch?v=mCNcxu8MXzo"}
QUIZ = {"promptVersion": "pilot-v1", "questions": [
    {"question": f"질문 {i}?", "options": ["가", "나", "다", "라"],
     "correctOptionIndex": 0, "explanation": "자료 기준", "sourceEvidence": "global rain and snow"}
    for i in range(3)]}
GROUNDING = {"contentText": CONTENT, "facts": [{
    "fact": "NASA measures rainfall", "evidenceType": "speech", "evidence": "global rain and snow",
    "timestampStartSeconds": 61, "timestampEndSeconds": 73}]}


def policy(**overrides):
    values = dict(enabled=True, call_limit=1, per_call_cost_limit=1.0, total_cost_limit=1.0,
                  timeout_seconds=10, retry_attempts=0, reasoning_effort="none",
                  video_processing="static", max_output_tokens=2000,
                  estimated_input_tokens=10000, input_price_per_million=1.0,
                  output_price_per_million=1.0)
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
                       {"reasoning_effort": None}, {"video_processing": None}):
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
            for model, response in (("gemini-3.8-flash", gemini_response(QUIZ)),
                                    ("gpt-5.4-mini", {"status": "completed", "output": [{
                                        "type": "message", "content": [{"type": "output_text", "text": json.dumps(QUIZ)}]}],
                                        "usage": {"input_tokens": 10, "output_tokens": 20}})):
                with self.subTest(model=model):
                    router = self.router(FakeTransport(response))
                    row = runner.run_quiz(VIDEO["videoId"], model, 1, CONTENT, router)
                    self.assertEqual(row["apiStatus"], "success")
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
        transport = FakeTransport({}, 429)
        with self.assertRaises(ProviderFailure) as failure:
            self.router(transport, retry_attempts=2, call_limit=1).invoke(
                "grounding", video=VIDEO, model="gemini-3.8-flash")
        self.assertEqual(failure.exception.category, "live_guard")
        self.assertEqual(len(transport.calls), 1)

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
                row = runner.run_quiz(VIDEO["videoId"], "gemini-3.8-flash", 1,
                                      CONTENT + " " + echoed,
                                      self.router(FakeTransport(gemini_response(payload))))
                self.assertEqual(row["validatorStatus"], "pass")
                raw_file = repository / "results" / "raw" / (row["runId"] + ".json")
                evaluation_file = repository / "results" / "evaluation" / (row["runId"] + ".json")
                self.assertNotIn(echoed, raw_file.read_text(encoding="utf-8"))
                self.assertEqual(json.loads(evaluation_file.read_text(encoding="utf-8")), payload)

    def test_two_stage_mock_results_link_runs_and_direct_never_claims_be_validation(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder:
            repository = Path(folder)
            (repository / "configs").mkdir()
            (repository / "data").mkdir()
            shutil.copyfile(root / "configs" / "pilot.yaml", repository / "configs" / "pilot.yaml")
            shutil.copyfile(root / "data" / "videos.jsonl", repository / "data" / "videos.jsonl")
            runner = PilotRunner(repository, repository / "results")
            transport = QueueTransport((gemini_response(GROUNDING), gemini_response(QUIZ)))
            row = runner.run_end_to_end(VIDEO["videoId"], "gemini_grounding_gemini_quiz", 1,
                                        self.router(transport, call_limit=2))
            self.assertIsNotNone(row["groundingRunId"])
            self.assertIsNotNone(row["quizRunId"])
            self.assertNotEqual(row["groundingRunId"], row["quizRunId"])
            self.assertEqual(row["beCompatibility"], "pass")
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
            envelope = {"status": "completed", "steps": [{"type": "model_output",
                        "content": [{"type": "text", "text": "{broken"}]}]}
            row = runner.run_quiz(VIDEO["videoId"], "gemini-3.8-flash", 1, CONTENT,
                                  self.router(FakeTransport(envelope)))
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
            row = PilotRunner(repository, repository / "results").run_quiz(
                VIDEO["videoId"], "gemini-3.8-flash", 1, CONTENT,
                self.router(FakeTransport(gemini_response(dict(QUIZ, promptVersion="other")))))
            self.assertEqual(row["validatorStatus"], "fail")
            self.assertEqual(row["beCompatibility"], "fail")


if __name__ == "__main__":
    unittest.main()
