import copy
import hashlib
import json
import shutil
import socket
import tempfile
import unittest
import urllib.error
from unittest import mock
from pathlib import Path

import yaml

from src.judge_contract import POINTWISE_ITEMS, RUBRIC_VERSION, SCHEMA_VERSION, pointwise_schema
from src.judge_client import parse_retry_after
from src.pilot_runner import PilotRunner
from src.production_grounding import load_grounding_artifact, validate_production_grounding
from src.production_judge import (ProductionJudgeRunner, ProductionJudgeUnavailable, classify_attempt,
                                  production_contract_sha256, validate_production_judge_config)
from src.production_quiz import ProductionQuizNotJudgeReady, new_operation_id
from tests.test_pilot_runner import CONTENT, GROUNDING_V3, ROOT, VIDEO, SimulatedApiFixture
from tests.test_production_grounding import COMPLETED_RAW, DURATION, VALIDATED_AT
from tests.test_production_quiz import FakeQuizProvider, quiz_with


RUNTIME = {"timeoutSeconds": 30, "perCallCostLimitUsd": 1.0, "totalCostLimitUsd": 5.0, "estimatedInputTokens": 3000,
           "maxRetryAfterSeconds": 60,
           "price": {"input": 2.0, "output": 10.0, "cachedInput": 0.2, "reference": "operator-test-pricing",
                     "checkedAt": "2026-10-08"}}
# (3000 * 2 + 8192 * 10) / 1e6 per request reservation; a known response settles to (1000 * 2 + 300 * 10) / 1e6.
RESERVATION = (3000 * 2.0 + 8192 * 10.0) / 1000000
KEYS = {"openai": "test-key"}


def verdicts(changes=None, indexes=(0, 1, 2)):
    """A valid Pointwise answer: all pass, with ``changes`` = {(index, item): verdict}."""
    changes = changes or {}
    return {"questions": [dict({"questionIndex": index}, **{
        item: {"verdict": changes.get((index, item), "pass"), "reason": "%s 문항 %d 근거" % (item, index)}
        for item in POINTWISE_ITEMS}) for index in indexes]}


def openai_response(text, status="completed", refusal=False, usage=None):
    content = [{"type": "refusal", "refusal": "거부"}] if refusal else [{"type": "output_text", "text": text}]
    return (200, {"status": status, "model": "gpt-6.1-sol-2026",
                  "usage": usage if usage is not None else {
                      "input_tokens": 1000, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 300,
                      "output_tokens_details": {"reasoning_tokens": 100}},
                  "output": [{"type": "message", "content": content}]}, None)


def ok(answer=None):
    return openai_response(json.dumps(answer if answer is not None else verdicts(), ensure_ascii=False))


class FakeOpenAI:
    """Scripted Judge transport; records every request body. Never opens a network connection."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.bodies = []

    def __call__(self, url, headers, body, timeout):
        self.bodies.append(copy.deepcopy(body))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class ProductionJudgeTest(unittest.TestCase):
    def test_yaml_container_errors_are_value_errors_without_side_effects(self):
        cases = ("", "null", "[]", "text", "3", "other: {}",
                 "production_judge: null", "production_judge: []", "production_judge: text",
                 "production_judge: {}")
        for number, text in enumerate(cases):
            with self.subTest(yaml=text):
                path = self.repository / "configs" / ("invalid-%d.yaml" % number)
                path.write_text(text, encoding="utf-8")
                before = {p.relative_to(self.repository): p.read_bytes()
                          for p in self.repository.rglob("*") if p.is_file()}
                with mock.patch("src.judge_client.urllib.request.urlopen") as transport:
                    with self.assertRaises(ValueError):
                        ProductionJudgeRunner(self.repository, judge_config=path)
                    transport.assert_not_called()
                after = {p.relative_to(self.repository): p.read_bytes()
                         for p in self.repository.rglob("*") if p.is_file()}
                self.assertEqual(after, before)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repository = Path(self.temp.name)
        (self.repository / "configs").mkdir()
        (self.repository / "data").mkdir()
        for name in ("pilot-v2.yaml", "production.yaml", "production-judge.yaml"):
            shutil.copyfile(ROOT / "configs" / name, self.repository / "configs" / name)
        shutil.copyfile(ROOT / "data" / "videos.jsonl", self.repository / "data" / "videos.jsonl")
        self.grounding_results = self.repository / "results" / "grounding-source"
        pilot = PilotRunner(self.repository, self.grounding_results, self.repository / "configs" / "pilot-v2.yaml")
        self.run_id = pilot.run_grounding(VIDEO, "gemini_video", 1, SimulatedApiFixture(
            {"grounding": {"normalized": GROUNDING_V3, "responseBody": COMPLETED_RAW}}))["runId"]
        artifact = load_grounding_artifact(self.grounding_results, self.run_id)
        self.record = validate_production_grounding(artifact, DURATION, VALIDATED_AT)
        self.runner = ProductionJudgeRunner(self.repository)
        self.production = self.repository / "results" / "production"
        self.quiz_id = self.quiz()
        self.sleeps = []

    def tearDown(self):
        self.temp.cleanup()

    def quiz(self, output=None):
        operation_id = new_operation_id()
        provider = FakeQuizProvider(output) if output is not None else FakeQuizProvider()
        self.runner.quiz.generate(operation_id, self.grounding_results, self.run_id, self.record, VIDEO, DURATION,
                                  provider)
        return operation_id

    def evaluate(self, transport, evaluation_id=None, runtime=RUNTIME, runner=None, quiz_id=None, api_keys=KEYS):
        evaluation_id = evaluation_id or new_operation_id()
        result = (runner or self.runner).evaluate(quiz_id or self.quiz_id, evaluation_id, runtime,
                                                  transport=transport, sleep=self.sleeps.append, api_keys=api_keys)
        return evaluation_id, result

    def judge_dir(self, evaluation_id, quiz_id=None):
        return self.production / "operations" / (quiz_id or self.quiz_id) / "judge" / evaluation_id

    def state(self, evaluation_id):
        return self.runner.evaluation_state(self.quiz_id, evaluation_id)

    def runner_with(self, **changes):
        config = yaml.safe_load((ROOT / "configs" / "production-judge.yaml").read_text(encoding="utf-8"))
        config["production_judge"].update(changes)
        path = self.repository / "configs" / ("production-judge-%d.yaml" % len(list((self.repository / "configs").iterdir())))
        path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
        return ProductionJudgeRunner(self.repository, judge_config=path)

    @staticmethod
    def rewrite(path, change):
        data = json.loads(path.read_text(encoding="utf-8"))
        change(data)
        path.unlink()
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    # Semantic verdicts

    def test_valid_verdicts_are_completed_results_and_never_retried(self):
        cases = {
            "all pass": {},
            "answer fail": {(0, "textAnswerCorrect"): "fail"},
            "unique answer fail": {(1, "uniqueAnswer"): "fail"},
            "evidence fail": {(2, "evidenceSupportsAnswer"): "fail"},
            "clarity uncertain": {(0, "questionClarity"): "uncertain"},
            "korean fail": {(1, "koreanQuality"): "fail"},
            "distractor fail": {(2, "distractorQuality"): "fail"},
            "faithfulness fail": {(0, "textFaithfulness"): "fail"},
            "mixed": {(0, "textAnswerCorrect"): "fail", (1, "questionClarity"): "uncertain",
                      (2, "textFaithfulness"): "fail", (2, "koreanQuality"): "uncertain"},
        }
        quiz_files_before = sorted(path.name for path in (self.production / "operations").iterdir())
        for name, changes in cases.items():
            with self.subTest(case=name):
                answer = verdicts(changes)
                transport = FakeOpenAI(ok(answer))
                evaluation_id, result = self.evaluate(transport)
                self.assertEqual((result["outcome"], result["errorCategory"], len(transport.bodies)),
                                 ("completed", None, 1))
                completed = self.runner.require_completed(self.quiz_id, evaluation_id)
                self.assertEqual(completed["questions"], answer["questions"])  # verdicts and reasons kept as given
                self.assertNotIn("overall", json.dumps(completed))
                # A fail or uncertain verdict is final: the same evaluation never calls again.
                again = FakeOpenAI(ok())
                with self.assertRaisesRegex(ValueError, "completed"):
                    self.evaluate(again, evaluation_id)
                self.assertEqual(again.bodies, [])
        # No Quiz was regenerated or touched by any verdict.
        self.assertEqual(sorted(path.name for path in (self.production / "operations").iterdir()), quiz_files_before)
        self.assertEqual(self.runner.quiz.operation_state(self.quiz_id), "completed")

    def test_the_request_holds_only_the_exact_content_text_and_questions(self):
        transport = FakeOpenAI(ok())
        evaluation_id, _ = self.evaluate(transport)
        body = transport.bodies[0]
        self.assertEqual((body["model"], body["reasoning"], body["max_output_tokens"], body["store"]),
                         ("gpt-6.1-sol", {"effort": "medium"}, 8192, False))
        self.assertEqual(body["text"]["format"], {"type": "json_schema", "name": "judge_pointwise", "strict": True,
                                                  "schema": pointwise_schema()})
        prompt = body["input"]
        self.assertIn("\n[contentText]\n" + CONTENT + "\n[/contentText]\n", prompt)
        judge_input = self.runner.quiz.require_judge_ready(self.quiz_id)
        shown = json.loads(prompt.split("[questions]\n")[1].split("\n[/questions]")[0])
        self.assertEqual(shown, [{key: question[key] for key in ("questionIndex", "question", "options",
                                                                  "correctOptionIndex", "explanation", "sourceEvidence")}
                                 for question in judge_input["questions"]])
        # Provenance stays in storage, never in the prompt.
        for hidden in (self.quiz_id, evaluation_id, self.run_id, "gemini", judge_input["contentTextSha256"],
                       judge_input["quizOutputSha256"], judge_input["sourceSnapshotSha256"],
                       "production-grounding-validation", "production-quiz", "approved", "pilot"):
            self.assertNotIn(hidden, json.dumps(body, ensure_ascii=False))
        manifest = json.loads((self.judge_dir(evaluation_id) / "evaluation.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["identity"]["quiz"]["operationId"], self.quiz_id)
        self.assertEqual(manifest["identity"]["quiz"]["contentTextSha256"], judge_input["contentTextSha256"])
        stored_body = json.loads((self.judge_dir(evaluation_id) / "request.json").read_text(encoding="utf-8"))
        self.assertEqual(stored_body, body)

    # Eligibility and binding

    def test_a_quiz_that_is_not_judge_ready_makes_no_request(self):
        cases = {"output gate fail": self.quiz(quiz_with(sourceEvidence='"global rain and snow"')),
                 "unknown operation": new_operation_id()}
        for name, quiz_id in cases.items():
            with self.subTest(case=name):
                transport = FakeOpenAI(ok())
                with self.assertRaises(ProductionQuizNotJudgeReady):
                    self.evaluate(transport, quiz_id=quiz_id)
                self.assertEqual(transport.bodies, [])
                self.assertFalse((self.production / "operations" / quiz_id / "judge").exists())

    def test_changed_quiz_artifacts_block_new_requests_and_stored_results(self):
        tampering = {
            "quiz output": lambda quiz_id: (self.production / "operations" / quiz_id / "attempts" / "1"
                                            / "output.json").write_text('{"output": {}}', encoding="ascii"),
            "source snapshot": lambda quiz_id: next((self.production / "sources").glob("*.json")).write_bytes(b"{}"),
        }
        for name, tamper in tampering.items():
            with self.subTest(case=name):
                quiz_id = self.quiz()
                evaluation_id, _ = self.evaluate(FakeOpenAI(ok()), quiz_id=quiz_id)
                self.runner.require_completed(quiz_id, evaluation_id)
                tamper(quiz_id)
                with self.assertRaises(ProductionJudgeUnavailable):
                    self.runner.require_completed(quiz_id, evaluation_id)
                self.assertEqual(self.runner.evaluation_state(quiz_id, evaluation_id), "quiz_not_judge_ready")
                transport = FakeOpenAI(ok())
                with self.assertRaises(ProductionQuizNotJudgeReady):
                    self.evaluate(transport, quiz_id=quiz_id)
                self.assertEqual(transport.bodies, [])

    def test_an_evaluation_bound_to_another_quiz_operation_is_refused(self):
        evaluation_id, _ = self.evaluate(FakeOpenAI(ok()))
        manifest_path = self.judge_dir(evaluation_id) / "evaluation.json"
        self.rewrite(manifest_path, lambda m: m["identity"]["quiz"].update(operationId=new_operation_id()))
        self.assertEqual(self.state(evaluation_id), "corrupted")
        with self.assertRaises(ProductionJudgeUnavailable):
            self.runner.require_completed(self.quiz_id, evaluation_id)
        transport = FakeOpenAI(ok())
        with self.assertRaisesRegex(ValueError, "never reused"):
            self.evaluate(transport, evaluation_id)
        self.assertEqual(transport.bodies, [])

    def test_judge_settings_and_contract_are_in_the_fingerprint(self):
        fingerprints = {}
        for name, runner in {"default": self.runner, "model": self.runner_with(model="gpt-6.1-sol-mini"),
                             "reasoning": self.runner_with(reasoning="high"),
                             "max_output_tokens": self.runner_with(max_output_tokens=4096)}.items():
            evaluation_id, _ = self.evaluate(FakeOpenAI(ok()), runner=runner)
            manifest = json.loads((self.judge_dir(evaluation_id) / "evaluation.json").read_text(encoding="utf-8"))
            fingerprints[name] = (manifest["inputFingerprint"], manifest["identity"]["requestBodySha256"])
        self.assertEqual(len({value[0] for value in fingerprints.values()}), 4)
        self.assertEqual(len({value[1] for value in fingerprints.values()}), 4)  # each changes the exact request
        self.assertNotEqual(production_contract_sha256("production-pointwise-judge-v1"),
                            production_contract_sha256("production-pointwise-judge-v2"))
        # Same evaluation, changed Judge setting: refused before any request.
        evaluation_id, _ = self.evaluate(FakeOpenAI((500, {}, None)))
        transport = FakeOpenAI(ok())
        with self.assertRaisesRegex(ValueError, "never reused"):
            self.evaluate(transport, evaluation_id, runner=self.runner_with(reasoning="high"))
        self.assertEqual(transport.bodies, [])

    def test_config_contract(self):
        config = yaml.safe_load((ROOT / "configs" / "production-judge.yaml").read_text(encoding="utf-8"))["production_judge"]
        validate_production_judge_config(config)
        self.assertEqual((config["rubric_version"], config["schema_version"]), (RUBRIC_VERSION, SCHEMA_VERSION))
        bad = {"gemini provider": dict(config, provider="gemini"),
               "unknown prompt": dict(config, prompt_version="judge-prompt-v1"),
               "other rubric": dict(config, rubric_version="judge-rubric-v2"),
               "other schema": dict(config, schema_version="judge-schema-v2"),
               "bool tokens": dict(config, max_output_tokens=True),
               "three attempts": dict(config, execution_policy=dict(config["execution_policy"], max_technical_attempts=3)),
               "automatic retry": dict(config, execution_policy=dict(config["execution_policy"], automatic_logical_retry=True)),
               "bool retry max": dict(config, execution_policy=dict(config["execution_policy"], http_retry_max=True)),
               "pairwise key": dict(config, pairwise=True)}
        for name, changed in bad.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                validate_production_judge_config(changed)

    # Contract failures and terminal / uncertain states

    def test_unusable_responses_are_kept_and_only_an_explicit_attempt_follows(self):
        answer = verdicts()
        missing_item = copy.deepcopy(answer)
        del missing_item["questions"][0]["koreanQuality"]
        extra_item = copy.deepcopy(answer)
        extra_item["questions"][0]["overallScore"] = {"verdict": "pass", "reason": "x"}
        unknown = copy.deepcopy(answer)
        unknown["questions"][1]["uniqueAnswer"]["verdict"] = "partial"
        empty_reason = copy.deepcopy(answer)
        empty_reason["questions"][2]["textFaithfulness"]["reason"] = "  "
        cases = {
            "malformed JSON": (openai_response('{"questions": ['), "parse_error"),
            "duplicate JSON key": (openai_response('{"questions": [], "questions": []}'), "parse_error"),
            "missing rubric item": (ok(missing_item), "schema_error"),
            "extra rubric item": (ok(extra_item), "schema_error"),
            "unknown verdict": (ok(unknown), "schema_error"),
            "missing question index": (ok(verdicts(indexes=(0, 1))), "semantic_error"),
            "duplicate question index": (ok(verdicts(indexes=(0, 1, 1))), "semantic_error"),
            "empty reason": (ok(empty_reason), "semantic_error"),
            "incomplete": (openai_response('{"questions', status="incomplete"), "incomplete"),
        }
        for name, (response, category) in cases.items():
            with self.subTest(case=name):
                transport = FakeOpenAI(response)
                evaluation_id, result = self.evaluate(transport)
                self.assertEqual((result["outcome"], result["errorCategory"], result["pointwise"], len(transport.bodies)),
                                 ("retryable_failure", category, None, 1))  # no automatic second attempt
                self.assertEqual(self.state(evaluation_id), "retryable")
                output = json.loads((self.judge_dir(evaluation_id) / "attempts" / "1" / "output.json").read_text(encoding="ascii"))
                self.assertEqual(output["text"], response[1]["output"][0]["content"][0]["text"])  # raw text kept
                with self.assertRaises(ProductionJudgeUnavailable):
                    self.runner.require_completed(self.quiz_id, evaluation_id)
                # Explicit attempt 2 runs once with the same request body; then the evaluation can complete.
                retry = FakeOpenAI(ok())
                _, second = self.evaluate(retry, evaluation_id)
                self.assertEqual((second["attempt"], second["outcome"]), (2, "completed"))
                self.assertEqual(retry.bodies, transport.bodies)
                self.assertEqual(self.runner.require_completed(self.quiz_id, evaluation_id)["attempt"], 2)

    def test_two_failed_attempts_exhaust_the_evaluation(self):
        evaluation_id, _ = self.evaluate(FakeOpenAI(openai_response("{")))
        self.evaluate(FakeOpenAI(openai_response("{")), evaluation_id)
        self.assertEqual(self.state(evaluation_id), "exhausted")
        transport = FakeOpenAI(ok())
        with self.assertRaisesRegex(ValueError, "exhausted"):
            self.evaluate(transport, evaluation_id)  # attempt 3 never runs
        self.assertEqual(transport.bodies, [])

    def test_terminal_responses_are_never_retried(self):
        cases = {"refusal": (openai_response("", refusal=True), "refusal", 1),
                 "client error": ((400, {}, None), "client_error", 1),
                 "429 without Retry-After": ((429, {}, None), "rate_limit", 1),
                 "503 without Retry-After": ((503, {}, None), "server_error", 1)}
        for name, (response, category, sends) in cases.items():
            with self.subTest(case=name):
                transport = FakeOpenAI(response)
                evaluation_id, result = self.evaluate(transport)
                self.assertEqual((result["outcome"], result["errorCategory"], len(transport.bodies)),
                                 ("terminal_failure", category, sends))
                self.assertEqual(self.state(evaluation_id), "terminal")
                again = FakeOpenAI(ok())
                with self.assertRaisesRegex(ValueError, "terminal"):
                    self.evaluate(again, evaluation_id)
                self.assertEqual(again.bodies, [])

    def test_retry_after_allows_one_http_retry_inside_an_attempt(self):
        transport = FakeOpenAI((429, {}, 2.0), ok())
        evaluation_id, result = self.evaluate(transport)
        self.assertEqual((result["outcome"], len(transport.bodies), self.sleeps), ("completed", 2, [2.0]))
        self.assertEqual([sent["isHttpRetry"] for sent in result["httpRequests"]], [False, True])
        # 5xx with Retry-After twice: one HTTP retry, then a retryable failure; attempt 2 does the same; limit 4.
        transport = FakeOpenAI((503, {}, 1.0), (503, {}, 1.0))
        evaluation_id, first = self.evaluate(transport)
        self.assertEqual((first["outcome"], len(transport.bodies)), ("retryable_failure", 2))
        transport = FakeOpenAI((503, {}, 1.0), (503, {}, 1.0))
        _, second = self.evaluate(transport, evaluation_id)
        self.assertEqual((second["outcome"], len(transport.bodies)), ("retryable_failure", 2))
        sends = sum(len(list((path / "requests").glob("*.json")))
                    for path in (self.judge_dir(evaluation_id) / "attempts").iterdir())
        self.assertEqual((sends, self.state(evaluation_id)), (4, "exhausted"))

    def test_timeouts_network_errors_and_interruptions_are_uncertain_and_never_replayed(self):
        cases = {"timeout": socket.timeout(), "network": urllib.error.URLError("connection reset"),
                 "interrupted after the request intent": KeyboardInterrupt()}
        for name, error in cases.items():
            with self.subTest(case=name):
                evaluation_id = new_operation_id()
                if isinstance(error, KeyboardInterrupt):
                    with self.assertRaises(KeyboardInterrupt):
                        self.evaluate(FakeOpenAI(error), evaluation_id)
                    attempt = self.judge_dir(evaluation_id) / "attempts" / "1"
                    self.assertTrue((attempt / "requests" / "1.json").is_file())
                    self.assertFalse((attempt / "result.json").exists())
                else:
                    _, result = self.evaluate(FakeOpenAI(error), evaluation_id)
                    self.assertEqual(result["outcome"], "uncertain")
                self.assertEqual(self.state(evaluation_id), "uncertain")
                transport = FakeOpenAI(ok())
                with self.assertRaisesRegex(ValueError, "uncertain"):
                    self.evaluate(transport, evaluation_id)
                self.assertEqual(transport.bodies, [])
        # Started only (no request intent): also uncertain, never replayed.
        evaluation_id = new_operation_id()
        with self.assertRaises(KeyboardInterrupt):
            self.evaluate(FakeOpenAI(KeyboardInterrupt()), evaluation_id)
        shutil.rmtree(self.judge_dir(evaluation_id) / "attempts" / "1" / "requests")
        self.assertEqual(self.state(evaluation_id), "uncertain")

    def test_a_valid_verdict_with_malformed_usage_completes_with_unknown_cost(self):
        transport = FakeOpenAI(openai_response(json.dumps(verdicts()), usage={"input_tokens": -1, "output_tokens": "x"}))
        evaluation_id, result = self.evaluate(transport)
        self.assertEqual((result["outcome"], result["estimatedCostUsd"]), ("completed", None))
        self.assertIn("invalidFields", result["usage"])
        self.assertEqual(result["httpRequests"][0]["settledUsd"], None)
        self.runner.require_completed(self.quiz_id, evaluation_id)

    # Guards and budget

    def test_runtime_guards_are_required_before_any_request(self):
        without = lambda key: {name: value for name, value in RUNTIME.items() if name != key}
        cases = {"missing timeout": without("timeoutSeconds"), "zero timeout": dict(RUNTIME, timeoutSeconds=0),
                 "nan timeout": dict(RUNTIME, timeoutSeconds=float("nan")), "bool timeout": dict(RUNTIME, timeoutSeconds=True),
                 "missing per-call cap": without("perCallCostLimitUsd"), "missing total cap": without("totalCostLimitUsd"),
                 "per-call cap too small": dict(RUNTIME, perCallCostLimitUsd=0.0001),
                 "missing price": without("price"),
                 "missing price reference": dict(RUNTIME, price=dict(RUNTIME["price"], reference=" ")),
                 "missing input estimate": without("estimatedInputTokens")}
        for name, runtime in cases.items():
            with self.subTest(case=name):
                transport = FakeOpenAI(ok())
                with self.assertRaises(ValueError):
                    self.evaluate(transport, runtime=runtime)
                self.assertEqual(transport.bodies, [])
        transport = FakeOpenAI(ok())
        with self.assertRaisesRegex(ValueError, "API key"):
            self.evaluate(transport, api_keys={})
        self.assertEqual(transport.bodies, [])
        self.assertFalse((self.production / "operations" / self.quiz_id / "judge").exists())

    def test_a_retry_refused_by_the_cost_cap_is_terminal(self):
        runtime = dict(RUNTIME, totalCostLimitUsd=RESERVATION * 1.5)  # room for one request only
        transport = FakeOpenAI((429, {}, 1.0), ok())
        evaluation_id, result = self.evaluate(transport, runtime=runtime)
        self.assertEqual((result["outcome"], result["errorCategory"], len(transport.bodies)),
                         ("terminal_failure", "live_guard", 1))

    def test_resume_restores_the_journalled_budget(self):
        runtime = dict(RUNTIME, totalCostLimitUsd=RESERVATION * 1.5)
        # An unknown cost keeps its full reservation, so attempt 2 no longer fits the evaluation cap.
        evaluation_id, _ = self.evaluate(FakeOpenAI(openai_response("{", usage={"input_tokens": "x"})), runtime=runtime)
        transport = FakeOpenAI(ok())
        with self.assertRaisesRegex(ValueError, "cost cap"):
            self.evaluate(transport, evaluation_id, runtime=runtime)
        self.assertEqual(transport.bodies, [])
        # A known settled cost releases the rest of its reservation, so attempt 2 fits.
        evaluation_id, _ = self.evaluate(FakeOpenAI(openai_response("{")), runtime=runtime)
        _, second = self.evaluate(FakeOpenAI(ok()), evaluation_id, runtime=runtime)
        self.assertEqual(second["outcome"], "completed")
        # A changed execution policy (here the timeout) is never accepted on resume.
        evaluation_id, _ = self.evaluate(FakeOpenAI(openai_response("{")))
        transport = FakeOpenAI(ok())
        with self.assertRaisesRegex(ValueError, "never reused"):
            self.evaluate(transport, evaluation_id, runtime=dict(RUNTIME, timeoutSeconds=60))
        self.assertEqual(transport.bodies, [])

    # Stored result tampering

    def test_tampered_judge_records_are_never_used_or_resumed(self):
        tampering = {
            "verdict": ("result.json", lambda r: r["pointwise"][0]["textAnswerCorrect"].update(verdict="fail")),
            "outcome": ("result.json", lambda r: r.update(outcome="retryable_failure")),
            "output text": ("output.json", lambda r: r.update(text=json.dumps(verdicts({(0, "uniqueAnswer"): "fail"})))),
            "request journal reservation": ("requests/1.json", lambda r: r.update(reservedUsd=0)),
            "started fingerprint": ("started.json", lambda r: r.update(inputFingerprint="f" * 64)),
            "result request body hash": ("result.json", lambda r: r.update(requestBodySha256="f" * 64)),
            "started missing": ("started.json", None),
        }
        for name, (file_name, change) in tampering.items():
            with self.subTest(case=name):
                evaluation_id, _ = self.evaluate(FakeOpenAI(ok()))
                path = self.judge_dir(evaluation_id) / "attempts" / "1" / file_name
                if change is None:
                    path.unlink()
                elif file_name == "output.json":
                    data = json.loads(path.read_text(encoding="ascii"))
                    change(data)
                    path.unlink()
                    path.write_text(json.dumps(data), encoding="ascii")
                else:
                    self.rewrite(path, change)
                self.assertEqual(self.state(evaluation_id), "corrupted")
                with self.assertRaises(ProductionJudgeUnavailable):
                    self.runner.require_completed(self.quiz_id, evaluation_id)
                transport = FakeOpenAI(ok())
                with self.assertRaisesRegex(ValueError, "corrupted"):
                    self.evaluate(transport, evaluation_id)
                self.assertEqual(transport.bodies, [])
        evaluation_id, _ = self.evaluate(FakeOpenAI(ok()))
        (self.judge_dir(evaluation_id) / "request.json").write_bytes(b"{}")
        self.assertEqual(self.state(evaluation_id), "corrupted")
        for change in (lambda m: m["executionPolicy"].update(requestLimit=40),
                       lambda m: m["executionPolicy"].update(maxTechnicalAttempts=3),
                       lambda m: m.update(evaluationId="0" * 32)):
            evaluation_id, _ = self.evaluate(FakeOpenAI(ok()))
            self.rewrite(self.judge_dir(evaluation_id) / "evaluation.json", change)
            self.assertEqual(self.state(evaluation_id), "corrupted")

    def test_attempt_outcomes_follow_the_issue_policy(self):
        expected = {None: "completed", "parse_error": "retryable_failure", "schema_error": "retryable_failure",
                    "semantic_error": "retryable_failure", "incomplete": "retryable_failure",
                    "invalid_provider_response": "retryable_failure", "refusal": "terminal_failure",
                    "client_error": "terminal_failure", "provider_error": "terminal_failure",
                    "live_guard": "terminal_failure", "api_key_missing": "terminal_failure",
                    "timeout": "uncertain", "network_error": "uncertain"}
        one = [{"retryAfterSeconds": None}]
        for category, outcome in expected.items():
            self.assertEqual(classify_attempt(category, one, 60), outcome)
        waited = {"retryAfterSeconds": 1.0}
        for category in ("rate_limit", "server_error"):
            self.assertEqual(classify_attempt(category, [{"retryAfterSeconds": None}], 60), "terminal_failure")
            self.assertEqual(classify_attempt(category, [waited], 60), "terminal_failure")  # no HTTP retry was used
            self.assertEqual(classify_attempt(category, [waited, {"retryAfterSeconds": 2.0}], 60), "retryable_failure")
            self.assertEqual(classify_attempt(category, [waited, {"retryAfterSeconds": 60}], 60), "retryable_failure")
            for hint in (None, -1, float("nan"), True, "1", 60.5, 86400):
                self.assertEqual(classify_attempt(category, [waited, {"retryAfterSeconds": hint}], 60), "terminal_failure")

    # Stored cost settlement, execution policy integrity and Retry-After classification

    def result_path(self, evaluation_id, attempt=1):
        return self.judge_dir(evaluation_id) / "attempts" / str(attempt) / "result.json"

    def assert_corrupted_and_never_resumed(self, evaluation_id, runtime=RUNTIME, message="corrupted|never reused"):
        self.assertEqual(self.state(evaluation_id), "corrupted")
        with self.assertRaises(ProductionJudgeUnavailable):
            self.runner.require_completed(self.quiz_id, evaluation_id)
        transport = FakeOpenAI(ok())
        with self.assertRaisesRegex(ValueError, message):
            self.evaluate(transport, evaluation_id, runtime=runtime)
        self.assertEqual(transport.bodies, [])

    def test_an_unknown_settled_cost_cannot_be_rewritten_to_release_its_reservation(self):
        runtime = dict(RUNTIME, totalCostLimitUsd=RESERVATION * 1.5)
        evaluation_id, first = self.evaluate(FakeOpenAI(openai_response("{", usage={"input_tokens": "x"})), runtime=runtime)
        self.assertEqual((first["httpRequests"][0]["settledUsd"], first["estimatedCostUsd"]), (None, None))
        transport = FakeOpenAI(ok())
        with self.assertRaisesRegex(ValueError, "cost cap"):  # the reservation is kept: attempt 2 does not fit
            self.evaluate(transport, evaluation_id, runtime=runtime)
        self.assertEqual(transport.bodies, [])
        for change in (lambda r: r["httpRequests"][0].update(settledUsd=0),
                       lambda r: (r["httpRequests"][0].update(settledUsd=0.0), r.update(estimatedCostUsd=0.0))):
            with self.subTest(change=change):
                path = self.result_path(evaluation_id)
                original = path.read_bytes()
                self.rewrite(path, change)
                self.assert_corrupted_and_never_resumed(evaluation_id, runtime)
                path.unlink()
                path.write_bytes(original)
        self.assertEqual(self.state(evaluation_id), "retryable")

    def test_a_known_settled_cost_must_equal_the_recomputed_cost(self):
        evaluation_id, first = self.evaluate(FakeOpenAI(openai_response("{")))
        self.assertEqual(first["httpRequests"][0]["settledUsd"], (1000 * 2.0 + 300 * 10.0) / 1000000)
        tampering = {"settled only": lambda r: r["httpRequests"][0].update(settledUsd=0.0),
                     "settled and summary": lambda r: (r["httpRequests"][0].update(settledUsd=0.001),
                                                       r.update(estimatedCostUsd=0.001)),
                     "summary only": lambda r: r.update(estimatedCostUsd=0.0),
                     "usage and settled kept": lambda r: r["usage"].update(outputTokens=1)}
        for name, change in tampering.items():
            with self.subTest(case=name):
                path = self.result_path(evaluation_id)
                original = path.read_bytes()
                self.rewrite(path, change)
                self.assert_corrupted_and_never_resumed(evaluation_id)
                path.unlink()
                path.write_bytes(original)
        # The reservation itself is recomputed from the stored price, so it cannot be lowered either.
        journal = self.judge_dir(evaluation_id) / "attempts" / "1" / "requests" / "1.json"
        self.rewrite(journal, lambda r: r.update(reservedUsd=0.0))
        self.rewrite(self.result_path(evaluation_id), lambda r: r["httpRequests"][0].update(reservedUsd=0.0))
        self.assert_corrupted_and_never_resumed(evaluation_id)

    def test_a_stored_execution_policy_is_fully_validated_and_bound_to_the_attempts(self):
        tampering = {
            "timeout 0": lambda e: e.update(timeoutSeconds=0),
            "total cap -1": lambda e: e.update(totalCostLimitUsd=-1),
            "price {}": lambda e: e.update(price={}),
            "bool timeout": lambda e: e.update(timeoutSeconds=True),
            "missing input estimate": lambda e: e.pop("estimatedInputTokens"),
            "extra key": lambda e: e.update(maxWaitSeconds=60),
            "empty price reference": lambda e: e["price"].update(reference=""),
            "other valid timeout": lambda e: e.update(timeoutSeconds=45),
            "other valid total cap": lambda e: e.update(totalCostLimitUsd=6.0),
            "other valid per-call cap": lambda e: e.update(perCallCostLimitUsd=0.5),
            "other valid input price": lambda e: e["price"].update(input=3.0),
            "other valid checked date": lambda e: e["price"].update(checkedAt="2026-10-09"),
            "request limit": lambda e: e.update(requestLimit=40),
        }
        for name, change in tampering.items():
            with self.subTest(case=name):
                evaluation_id, _ = self.evaluate(FakeOpenAI(ok()))
                self.assertEqual(self.state(evaluation_id), "completed")
                self.rewrite(self.judge_dir(evaluation_id) / "evaluation.json",
                             lambda manifest: change(manifest["executionPolicy"]))
                self.assert_corrupted_and_never_resumed(evaluation_id)
        untouched, _ = self.evaluate(FakeOpenAI(ok()))
        self.assertEqual(self.runner.require_completed(self.quiz_id, untouched)["evaluationId"], untouched)

    def test_a_429_or_5xx_without_a_retry_after_on_its_last_response_is_terminal(self):
        # judge_transport stores an unparseable Retry-After exactly like a missing one: None.
        self.assertIsNone(parse_retry_after("soon"))
        for status in (429, 503):
            with self.subTest(status=status):
                transport = FakeOpenAI((status, {}, 1.0), (status, {}, 2.0))
                evaluation_id, first = self.evaluate(transport)
                self.assertEqual((first["outcome"], len(transport.bodies)), ("retryable_failure", 2))
                self.assertEqual(self.state(evaluation_id), "retryable")
                # Tampering the last hint flips the derived meaning, so the record no longer verifies.
                path = self.result_path(evaluation_id)
                original = path.read_bytes()
                self.rewrite(path, lambda r: r["httpRequests"][1].update(retryAfterSeconds=None))
                self.assert_corrupted_and_never_resumed(evaluation_id)
                path.unlink()
                path.write_bytes(original)
                _, second = self.evaluate(FakeOpenAI(ok()), evaluation_id)  # explicit attempt 2 within the cap
                self.assertEqual(second["outcome"], "completed")

                for name, last_hint in (("missing", None), ("invalid header", parse_retry_after("soon"))):
                    transport = FakeOpenAI((status, {}, 1.0), (status, {}, last_hint))
                    evaluation_id, result = self.evaluate(transport)
                    self.assertEqual((result["outcome"], len(transport.bodies), self.state(evaluation_id)),
                                     ("terminal_failure", 2, "terminal"), name)
                    again = FakeOpenAI(ok())
                    with self.assertRaisesRegex(ValueError, "terminal"):
                        self.evaluate(again, evaluation_id)
                    self.assertEqual(again.bodies, [])
                path = self.result_path(evaluation_id)
                self.rewrite(path, lambda r: r["httpRequests"][1].update(retryAfterSeconds=5.0))
                self.assert_corrupted_and_never_resumed(evaluation_id)

                transport = FakeOpenAI((status, {}, None))
                evaluation_id, result = self.evaluate(transport)
                self.assertEqual((result["outcome"], len(transport.bodies)), ("terminal_failure", 1))


    # Retry-After wait cap (maxRetryAfterSeconds)

    def test_a_retry_after_within_the_cap_is_waited_and_retried_once(self):
        for hint in (1.0, 60, 60.0):  # below and exactly at the cap of 60
            with self.subTest(hint=hint):
                self.sleeps.clear()
                transport = FakeOpenAI((429, {}, hint), ok())
                evaluation_id, result = self.evaluate(transport)
                self.assertEqual((result["outcome"], len(transport.bodies), self.sleeps), ("completed", 2, [hint]))
                self.assertEqual(self.state(evaluation_id), "completed")

    def test_a_retry_after_above_the_cap_is_terminal_without_waiting_or_retrying(self):
        for status, hint, cap in ((429, 60.5, 60), (503, 61, 60), (503, 86400.0, 60), (429, 1.0, 0)):
            with self.subTest(status=status, hint=hint, cap=cap):
                self.sleeps.clear()
                runtime = dict(RUNTIME, maxRetryAfterSeconds=cap)
                transport = FakeOpenAI((status, {}, hint), ok())
                evaluation_id, result = self.evaluate(transport, runtime=runtime)
                self.assertEqual(self.sleeps, [])  # never waited
                self.assertEqual(len(transport.bodies), 1)  # no HTTP retry
                self.assertEqual((result["outcome"], result["errorCategory"], result["httpStatus"]),
                                 ("terminal_failure", "rate_limit" if status == 429 else "server_error", status))
                self.assertNotEqual(result["outcome"], "uncertain")
                sent = result["httpRequests"]
                self.assertEqual((len(sent), sent[0]["retryAfterSeconds"], sent[0]["waitedSeconds"], sent[0]["settledUsd"]),
                                 (1, hint, None, None))
                self.assertEqual(sent[0]["reservedUsd"], RESERVATION)  # the reservation is kept
                journal = sorted((self.judge_dir(evaluation_id) / "attempts" / "1" / "requests").glob("*.json"))
                self.assertEqual(len(journal), 1)
                self.assertEqual(self.state(evaluation_id), "terminal")
                again = FakeOpenAI(ok())
                with self.assertRaisesRegex(ValueError, "terminal"):  # no explicit attempt 2
                    self.evaluate(again, evaluation_id, runtime=runtime)
                self.assertEqual(again.bodies, [])
        # Within the cap on the first response, above it on the retry's response: terminal too.
        transport = FakeOpenAI((503, {}, 1.0), (503, {}, 120.0))
        evaluation_id, result = self.evaluate(transport)
        self.assertEqual((result["outcome"], len(transport.bodies)), ("terminal_failure", 2))
        self.assertEqual(self.state(evaluation_id), "terminal")

    def test_a_missing_or_invalid_retry_after_is_still_terminal_with_a_cap(self):
        for hint in (None, parse_retry_after("soon")):
            with self.subTest(hint=hint):
                self.sleeps.clear()
                transport = FakeOpenAI((503, {}, hint))
                _, result = self.evaluate(transport)
                self.assertEqual((result["outcome"], len(transport.bodies), self.sleeps), ("terminal_failure", 1, []))

    def test_the_retry_after_cap_is_a_required_runtime_value(self):
        without = {key: value for key, value in RUNTIME.items() if key != "maxRetryAfterSeconds"}
        cases = {"missing": without, "negative": dict(RUNTIME, maxRetryAfterSeconds=-1),
                 "bool": dict(RUNTIME, maxRetryAfterSeconds=True), "nan": dict(RUNTIME, maxRetryAfterSeconds=float("nan")),
                 "infinity": dict(RUNTIME, maxRetryAfterSeconds=float("inf")),
                 "string": dict(RUNTIME, maxRetryAfterSeconds="60"), "null": dict(RUNTIME, maxRetryAfterSeconds=None)}
        for name, runtime in cases.items():
            with self.subTest(case=name):
                transport = FakeOpenAI(ok())
                with self.assertRaises(ValueError):
                    self.evaluate(transport, runtime=runtime)
                self.assertEqual(transport.bodies, [])
        self.assertFalse((self.production / "operations" / self.quiz_id / "judge").exists())

    def test_the_stored_retry_after_cap_is_validated_bound_and_fixed_on_resume(self):
        evaluation_id, _ = self.evaluate(FakeOpenAI(ok()))
        manifest = json.loads((self.judge_dir(evaluation_id) / "evaluation.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["executionPolicy"]["maxRetryAfterSeconds"], 60)
        again = FakeOpenAI(ok())
        with self.assertRaisesRegex(ValueError, "completed"):  # a completed evaluation is never re-called
            self.evaluate(again, evaluation_id)
        self.assertEqual(again.bodies, [])
        for cap in (120, -1, None):
            with self.subTest(stored=cap):
                evaluation_id, _ = self.evaluate(FakeOpenAI(ok()))
                self.rewrite(self.judge_dir(evaluation_id) / "evaluation.json",
                             lambda m: m["executionPolicy"].update(maxRetryAfterSeconds=cap))
                self.assert_corrupted_and_never_resumed(evaluation_id)
        # A retryable evaluation cannot be resumed with another cap.
        evaluation_id, _ = self.evaluate(FakeOpenAI(openai_response("{")))
        transport = FakeOpenAI(ok())
        with self.assertRaisesRegex(ValueError, "never reused"):
            self.evaluate(transport, evaluation_id, runtime=dict(RUNTIME, maxRetryAfterSeconds=120))
        self.assertEqual(transport.bodies, [])
        # A stored over-cap hint on a waited request does not verify.
        evaluation_id, _ = self.evaluate(FakeOpenAI((503, {}, 1.0), (503, {}, 2.0)))
        self.rewrite(self.result_path(evaluation_id), lambda r: r["httpRequests"][0].update(
            retryAfterSeconds=120.0, waitedSeconds=120.0))
        self.assert_corrupted_and_never_resumed(evaluation_id)

    def test_require_completed_reports_unreadable_storage_as_unavailable(self):
        evaluation_id, _ = self.evaluate(FakeOpenAI(ok()))
        # Errors reading stored files while binding the Quiz never escape as raw exceptions.
        for error in (OSError("unreadable"), KeyError("input"), TypeError("shape"), UnicodeDecodeError("utf-8", b"", 0, 1, "x")):
            with self.subTest(error=type(error).__name__), mock.patch.object(self.runner, "_bind", side_effect=error):
                with self.assertRaises(ProductionJudgeUnavailable):
                    self.runner.require_completed(self.quiz_id, evaluation_id)
        self.assertEqual(self.runner.require_completed(self.quiz_id, evaluation_id)["evaluationId"], evaluation_id)

    # Result byte identity and manifest classification

    def test_the_result_sha_is_the_sha_of_the_verified_file_bytes(self):
        evaluation_id, _ = self.evaluate(FakeOpenAI(ok()))
        path = self.judge_dir(evaluation_id) / "attempts" / "1" / "result.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        layouts = {"canonical": path.read_bytes(),
                   "pretty-printed": json.dumps(record, ensure_ascii=False, indent=2).encode("utf-8"),
                   "trailing newline": path.read_bytes() + b"\n",
                   "reordered keys": json.dumps(dict(reversed(list(record.items()))), ensure_ascii=False).encode("utf-8")}
        seen = set()
        for name, data in layouts.items():
            with self.subTest(layout=name):
                path.unlink()
                path.write_bytes(data)
                # The Judge contract parses the record; its serialization is not part of the contract.
                self.assertEqual(self.state(evaluation_id), "completed")
                completed = self.runner.require_completed(self.quiz_id, evaluation_id)
                self.assertEqual(completed["resultSha256"], hashlib.sha256(path.read_bytes()).hexdigest())
                seen.add(completed["resultSha256"])
        self.assertEqual(len(seen), 4)

    def test_a_missing_or_damaged_manifest_is_corrupted_not_unstarted(self):
        self.assertEqual(self.state(new_operation_id()), "not_started")  # never recorded anything
        cases = {"manifest deleted": lambda path: path.unlink(),
                 "manifest replaced by a directory": lambda path: (path.unlink(), path.mkdir()),
                 "manifest JSON damaged": lambda path: (path.unlink(), path.write_text("{", encoding="utf-8"))}
        for name, damage in cases.items():
            with self.subTest(case=name):
                evaluation_id, _ = self.evaluate(FakeOpenAI(ok()))
                self.assertEqual(self.state(evaluation_id), "completed")
                damage(self.judge_dir(evaluation_id) / "evaluation.json")
                self.assertTrue((self.judge_dir(evaluation_id) / "attempts").is_dir())
                self.assertEqual(self.state(evaluation_id), "corrupted")
                with self.assertRaises(ProductionJudgeUnavailable):
                    self.runner.require_completed(self.quiz_id, evaluation_id)
                transport = FakeOpenAI(ok())
                with self.assertRaisesRegex(ValueError, "corrupted|never reused"):
                    self.evaluate(transport, evaluation_id)
                self.assertEqual(transport.bodies, [])
                self.assertFalse((self.judge_dir(evaluation_id) / "evaluation.json").is_file()
                                 and name == "manifest deleted")  # not silently recreated

    def test_a_manifest_create_that_never_completed_is_still_unstarted_and_resumable(self):
        evaluation_id = new_operation_id()
        folder = self.judge_dir(evaluation_id)
        folder.mkdir(parents=True)
        self.assertEqual(self.state(evaluation_id), "not_started")  # empty directory
        # Exactly what _create_once leaves when it stops before linking the manifest.
        with tempfile.NamedTemporaryFile("wb", dir=folder, prefix=".production-", suffix=".tmp", delete=False) as stream:
            stream.write(b"{")
        self.assertEqual(self.state(evaluation_id), "not_started")
        _, result = self.evaluate(FakeOpenAI(ok()), evaluation_id)
        self.assertEqual((result["outcome"], self.state(evaluation_id)), ("completed", "completed"))
        retry_id, _ = self.evaluate(FakeOpenAI(openai_response("{")))
        self.assertEqual(self.evaluate(FakeOpenAI(ok()), retry_id)[1]["attempt"], 2)  # resume is unchanged

if __name__ == "__main__":
    unittest.main()
