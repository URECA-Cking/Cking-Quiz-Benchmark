import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.judge_client import (JudgeCallFailure, JudgeFixtureClient, JudgeHttpClient, JudgePolicy, estimate_cost,
                              parse_response)
from src.judge_eligibility import SourceIntegrityError
from src.judge_runner import JudgeRunner, cost_summary, measurement_states, preflight_live_policy
from src.pilot_runner import PilotRunner
from tests.judge_fixtures import build_synthetic_pilot, default_outcome, pairwise_output, pointwise_output


TOKENS = {"pointwise": 4000, "pairwise": 6000}
WATER_A = "pointwise:openai_judge:nasa-water-cycle-2019:1:A"


def snapshot(results):
    return {path.relative_to(results).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in results.rglob("*") if path.is_file() and "judge" not in path.relative_to(results).parts}


class JudgeRunnerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repository = Path(self.temp.name)
        self.results = build_synthetic_pilot(self.repository)
        self.sleeps = []
        self.runner = JudgeRunner(self.repository, max_output_tokens=TOKENS, sleep=self.sleeps.append)

    def tearDown(self):
        self.temp.cleanup()

    def client(self, script=None):
        return JudgeFixtureClient(script or {}, default=default_outcome)

    def attempts(self, run_id):
        path = self.results / "judge" / run_id / "attempts.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def measurement_id(self, run_id, fixture_key):
        path = self.results / "judge" / run_id / "measurements.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        return next(row["logicalMeasurementId"] for row in rows if row["fixtureKey"] == fixture_key)

    def assert_stopped_at(self, run_id, categories):
        # WATER_A is the 7th Pointwise measurement (sets are ordered by videoId); nothing runs after the stop.
        attempts = self.attempts(run_id)
        mid = self.measurement_id(run_id, WATER_A)
        self.assertEqual([row["errorCategory"] for row in attempts if row["logicalMeasurementId"] == mid], categories)
        self.assertEqual(len(attempts), 6 + len(categories))
        self.assertEqual(attempts[-1]["logicalMeasurementId"], mid)

    def test_full_fixture_run_stores_only_under_results_judge_and_keeps_pilot_intact(self):
        before = snapshot(self.results)
        manifest = self.runner.start(self.client())
        self.assertEqual((manifest["status"], manifest["stopReason"]), ("completed", None))
        run_dir = self.results / "judge" / manifest["judgeRunId"]
        self.assertEqual(len(self.attempts(manifest["judgeRunId"])), 28)
        for name in ("run.json", "eligibility.jsonl", "measurements.jsonl", "attempts.jsonl"):
            self.assertTrue((run_dir / name).is_file())
        self.assertEqual(snapshot(self.results), before)
        PilotRunner(self.repository, self.results)._check_storage_integrity()

        report = self.runner.derive(manifest["judgeRunId"])
        self.assertEqual((report["eligibility"]["pointwiseEligibleSets"], report["eligibility"]["pointwiseEligibleQuestions"],
                          report["eligibility"]["pairwiseEligiblePairs"]), (8, 22, 3))
        self.assertEqual(report["measurementStatus"], {"SELECTED": 28})
        pairs = [json.loads(line) for line in (run_dir / "derived" / "pairwise-pairs.jsonl")
                 .read_text(encoding="utf-8").splitlines()]
        h2a = {pair["videoId"]: pair["humanDerivedValidity"]["result"] for pair in pairs}
        self.assertEqual(h2a, {"nasa-water-cycle-2019": "NO_VALIDITY_DIFFERENCE",
                               "nasa-methane-2020": "NO_VALIDITY_DIFFERENCE", "kari-microgravity-2024": "A"})
        # All default calls are TIE decided by "tie", so none is comparable with H2a.
        self.assertTrue(all(pair["h2aComparison"]["comparability"] == "NOT_COMPARABLE_DECISION_BASIS"
                            for pair in pairs))
        self.assertTrue(all(pair["status"] == "RESOLVED" and pair["preference"] == "TIE" for pair in pairs))
        PilotRunner(self.repository, self.results)._check_storage_integrity()

    def test_first_valid_success_after_semantic_failure(self):
        manifest = self.runner.start(self.client({WATER_A: [{"text": pointwise_output([0, 1])},
                                                           {"text": pointwise_output([0, 1, 2], "fail")},
                                                           {"text": pointwise_output([0, 1, 2])}]}))
        mid = self.measurement_id(manifest["judgeRunId"], WATER_A)
        history = [row for row in self.attempts(manifest["judgeRunId"]) if row["logicalMeasurementId"] == mid]
        self.assertEqual([(row["attempt"], row["outcome"], row["errorCategory"]) for row in history],
                         [(1, "failure", "semantic_error"), (2, "success", None)])
        self.runner.derive(manifest["judgeRunId"])
        verdicts = (self.results / "judge" / manifest["judgeRunId"] / "derived" / "pointwise-verdicts.jsonl")
        rows = [json.loads(line) for line in verdicts.read_text(encoding="utf-8").splitlines()]
        water_a = [row for row in rows if row["judgeId"] == "openai_judge" and row["videoId"] == "nasa-water-cycle-2019"
                   and row["condition"] == "A"]
        self.assertTrue(water_a and all(row["verdict"] == "fail" for row in water_a))

    def test_retry_waits_for_hint_and_stops_without_one(self):
        manifest = self.runner.start(self.client({WATER_A: [{"failure": "rate_limit", "retryAfter": 7}]}))
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(self.sleeps, [7])
        stopped = JudgeRunner(self.repository, max_output_tokens=TOKENS, sleep=self.sleeps.append).start(
            self.client({WATER_A: [{"failure": "server_error"}]}))
        self.assertEqual((stopped["status"], stopped["stopReason"]), ("stopped", "retryWaitUnresolved"))
        self.assert_stopped_at(stopped["judgeRunId"], ["server_error"])

    def test_repeated_incomplete_stops_but_mixed_exhaustion_continues(self):
        stopped = self.runner.start(self.client({WATER_A: [{"failure": "incomplete"}] * 3}))
        self.assertEqual((stopped["status"], stopped["stopReason"]), ("stopped", "repeatedIncomplete"))
        self.assert_stopped_at(stopped["judgeRunId"], ["incomplete"] * 3)

        mixed = self.runner.start(self.client({WATER_A: [{"failure": "incomplete"}, {"failure": "timeout"},
                                                        {"failure": "server_error", "retryAfter": 1}]}))
        self.assertEqual(mixed["status"], "completed")
        states = self.runner.derive(mixed["judgeRunId"])["measurementStatus"]
        self.assertEqual(states, {"SELECTED": 27, "INCOMPLETE": 1})

    def test_pairwise_order_failure_makes_judge_and_pair_incomplete(self):
        key = "pairwise:gemini_judge:nasa-methane-2020:1:BA"
        manifest = self.runner.start(self.client({key: [{"failure": "refusal"}] * 3}))
        self.assertEqual(manifest["status"], "completed")
        self.runner.derive(manifest["judgeRunId"])
        derived = self.results / "judge" / manifest["judgeRunId"] / "derived"
        pairs = {row["videoId"]: row for row in (json.loads(line) for line in
                 (derived / "pairwise-pairs.jsonl").read_text(encoding="utf-8").splitlines())}
        methane = pairs["nasa-methane-2020"]
        self.assertEqual((methane["status"], methane["preference"]), ("INCOMPLETE", None))
        self.assertEqual(methane["judgeResults"]["gemini_judge"]["status"], "INCOMPLETE")
        self.assertEqual(methane["judgeResults"]["openai_judge"]["status"], "RESOLVED")

    def test_validity_decided_pairs_are_compared_with_h2a(self):
        script = {}
        for judge in ("openai_judge", "gemini_judge"):
            # KARI: Judge finds a defect in B only; SET_2 is B for AB and SET_1 is B for BA.
            script["pairwise:%s:kari-microgravity-2024:1:AB" % judge] = [{"text": pairwise_output([], [1])}]
            script["pairwise:%s:kari-microgravity-2024:1:BA" % judge] = [{"text": pairwise_output([1], [])}]
        manifest = self.runner.start(self.client(script))
        self.runner.derive(manifest["judgeRunId"])
        derived = self.results / "judge" / manifest["judgeRunId"] / "derived"
        kari = next(row for row in (json.loads(line) for line in
                    (derived / "pairwise-pairs.jsonl").read_text(encoding="utf-8").splitlines())
                    if row["videoId"] == "kari-microgravity-2024")
        self.assertEqual((kari["status"], kari["preference"]), ("RESOLVED", "A"))
        self.assertEqual(kari["h2aComparison"], {"comparability": "COMPARABLE", "reason": None, "outcome": "AGREE"})

    def test_run_stopping_failures_and_resume_with_same_configuration(self):
        manifest = self.runner.start(self.client({WATER_A: [{"failure": "client_error"}]}))
        self.assertEqual((manifest["status"], manifest["stopReason"]), ("stopped", "client_error"))
        run_id = manifest["judgeRunId"]
        resumed = self.runner.resume(run_id, self.client())
        self.assertEqual(resumed["status"], "completed")
        mid = self.measurement_id(run_id, WATER_A)
        self.assertEqual([row["attempt"] for row in self.attempts(run_id) if row["logicalMeasurementId"] == mid], [1, 2])
        self.assertEqual(len(self.attempts(run_id)), 29)

    def test_changed_configuration_requires_a_new_run(self):
        manifest = self.runner.start(self.client({WATER_A: [{"failure": "client_error"}]}))
        changed = JudgeRunner(self.repository, max_output_tokens={"pointwise": 5000, "pairwise": 6000})
        self.assertNotEqual(changed.config_hash, self.runner.config_hash)
        with self.assertRaises(ValueError):
            changed.resume(manifest["judgeRunId"], self.client())
        second = changed.start(self.client())
        self.assertNotEqual(second["judgeRunId"], manifest["judgeRunId"])
        first_ids = {row["logicalMeasurementId"] for row in self.attempts(manifest["judgeRunId"])}
        second_ids = {row["logicalMeasurementId"] for row in self.attempts(second["judgeRunId"])}
        self.assertFalse(first_ids & second_ids)
        # Source changes also block resume.
        quiz = self.results / "quiz-generation.jsonl"
        quiz.write_text(quiz.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.runner.resume(manifest["judgeRunId"], self.client())

    def test_incomplete_output_is_preserved_but_not_selected(self):
        script = {WATER_A: [{"failure": "incomplete", "rawText": '{"questions": [ partial'},
                            {"failure": "incomplete"}, {"text": pointwise_output([0, 1, 2], "uncertain")}]}
        manifest = self.runner.start(self.client(script))
        run_id = manifest["judgeRunId"]
        mid = self.measurement_id(run_id, WATER_A)
        history = [row for row in self.attempts(run_id) if row["logicalMeasurementId"] == mid]
        self.assertEqual([row["outcome"] for row in history], ["failure", "failure", "success"])
        kept = json.loads((self.results / "judge" / run_id / history[0]["outputRef"]).read_text(encoding="utf-8"))
        self.assertEqual((kept["rawText"], kept["parsed"]), ('{"questions": [ partial', None))
        self.assertIsNone(history[1]["outputRef"])
        self.runner.derive(run_id)
        verdicts = self.results / "judge" / run_id / "derived" / "pointwise-verdicts.jsonl"
        rows = [json.loads(line) for line in verdicts.read_text(encoding="utf-8").splitlines()]
        water_a = {row["verdict"] for row in rows if row["judgeId"] == "openai_judge"
                   and row["videoId"] == "nasa-water-cycle-2019" and row["condition"] == "A"}
        self.assertEqual(water_a, {"uncertain"})

    def test_duplicate_successes_select_the_first(self):
        measurements = [{"logicalMeasurementId": "m"}]
        attempts = [{"logicalMeasurementId": "m", "attemptId": "m-2", "attempt": 2, "outcome": "success"},
                    {"logicalMeasurementId": "m", "attemptId": "m-1", "attempt": 1, "outcome": "success"}]
        state = measurement_states(measurements, attempts)["m"]
        self.assertEqual((state["selectedAttemptId"], state["duplicateSuccessAttemptIds"]), ("m-1", ["m-2"]))


class FakeTransport:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, url, headers, body, timeout):
        self.calls.append(url)
        return self.responses.pop(0)


def openai_ok(status="completed", content=None):
    return (200, {"status": status, "model": "gpt-6.1-sol-2026", "usage": {
        "input_tokens": 1000, "input_tokens_details": {"cached_tokens": 200}, "output_tokens": 300,
        "output_tokens_details": {"reasoning_tokens": 100}},
        "output": [{"type": "message", "content": content or [{"type": "output_text", "text": "{}"}]}]}, None)


class JudgeHttpClientTest(unittest.TestCase):
    def policy(self, **overrides):
        values = dict(live=True, http_request_limit=10, per_call_cost_limit=1.0, total_cost_limit=5.0,
                      timeout_seconds=30, estimated_input_tokens={"pointwise": 1000},
                      max_output_tokens={"pointwise": 1000},
                      prices={"openai": {"input": 2.0, "output": 10.0, "cachedInput": 0.1, "reference": "doc",
                                         "checkedAt": "2026-10-06"}})
        values.update(overrides)
        return JudgePolicy(**values)

    def client(self, transport, **overrides):
        self.sleeps = []
        return JudgeHttpClient(self.policy(**overrides), {"openai": "OPENAI_API_KEY"}, transport=transport,
                               sleep=self.sleeps.append, api_keys={"openai": "test-key"})

    def failure(self, client):
        with self.assertRaises(JudgeCallFailure) as caught:
            client.call("openai", "pointwise", {})
        return caught.exception

    def test_success_records_model_usage_and_cost(self):
        result = self.client(FakeTransport(openai_ok())).call("openai", "pointwise", {})
        self.assertEqual(result["reportedModel"], "gpt-6.1-sol-2026")
        self.assertEqual(result["usage"]["cachedInputTokens"], 200)
        self.assertAlmostEqual(result["estimatedCostUsd"], (800 * 2.0 + 200 * 0.1 + 300 * 10.0) / 1000000)

    def test_429_uses_retry_after_once(self):
        transport = FakeTransport((429, {}, 3.0), openai_ok())
        client = self.client(transport)
        result = client.call("openai", "pointwise", {})
        self.assertEqual((self.sleeps, len(transport.calls)), ([3.0], 2))
        self.assertEqual(result["httpRequests"][0]["waitedSeconds"], 3.0)
        failure = self.failure(self.client(FakeTransport((503, {}, 2.0), (503, {}, 4.0))))
        self.assertEqual((failure.category, failure.retry_after, failure.stop_run), ("server_error", 4.0, False))

    def test_missing_hint_fails_closed_without_immediate_retry(self):
        transport = FakeTransport((429, {}, None))
        failure = self.failure(self.client(transport))
        self.assertEqual((failure.category, failure.retry_wait_unresolved, len(transport.calls)),
                         ("rate_limit", True, 1))

    def test_terminal_and_measurement_failures(self):
        self.assertTrue(self.failure(self.client(FakeTransport((400, {}, None)))).stop_run)
        self.assertEqual(self.failure(self.client(FakeTransport(openai_ok(status="incomplete")))).category,
                         "incomplete")
        refusal = openai_ok(content=[{"type": "refusal", "refusal": "no"}])
        self.assertEqual(self.failure(self.client(FakeTransport(refusal))).category, "refusal")
        guard = self.failure(self.client(FakeTransport(openai_ok()), live=False))
        self.assertEqual((guard.category, guard.stop_run), ("live_guard", True))
        limited = self.client(FakeTransport(openai_ok(), openai_ok()), http_request_limit=1)
        limited.call("openai", "pointwise", {})
        self.assertEqual(self.failure(limited).category, "live_guard")

    def test_gemini_cost_adds_thinking_tokens(self):
        usage = {"inputTokens": 1000, "outputTokens": 200, "reasoningTokens": 300, "cachedInputTokens": None}
        self.assertAlmostEqual(estimate_cost("gemini", usage, {"input": 0.75, "output": 3.75}),
                               (1000 * 0.75 + 500 * 3.75) / 1000000)
        self.assertIsNone(estimate_cost("openai", {"inputTokens": 10, "outputTokens": 1, "cachedInputTokens": 5},
                                        {"input": 2.0, "output": 10.0, "cachedInput": None}))


def live_policy(**overrides):
    values = dict(live=True, http_request_limit=50, per_call_cost_limit=1.0, total_cost_limit=10.0,
                  timeout_seconds=30, estimated_input_tokens={"pointwise": 1000, "pairwise": 1000},
                  max_output_tokens=dict(TOKENS),
                  prices={provider: {"input": 1.0, "output": 2.0, "cachedInput": 0.1, "reference": "doc",
                                     "checkedAt": "2026-10-06"} for provider in ("openai", "gemini")})
    values.update(overrides)
    return JudgePolicy(**values)


class LiveGuardAndResumeBoundaryTest(unittest.TestCase):
    JUDGES = [{"provider": "openai"}, {"provider": "gemini"}]

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repository = Path(self.temp.name)
        self.results = build_synthetic_pilot(self.repository)
        self.runner = JudgeRunner(self.repository, max_output_tokens=TOKENS, sleep=lambda seconds: None)

    def tearDown(self):
        self.temp.cleanup()

    def live_client(self, transport):
        return JudgeHttpClient(live_policy(), {}, transport=transport, sleep=lambda seconds: None,
                               api_keys={"openai": "k", "gemini": "k"})

    def attempt_count(self, run_id):
        path = self.results / "judge" / run_id / "attempts.jsonl"
        return len(path.read_text(encoding="utf-8").splitlines())

    def test_negative_bool_or_overflowing_token_estimates_never_pass_the_guard(self):
        for tokens in (-1, True, 10 ** 400):
            policy = live_policy(estimated_input_tokens={"pointwise": tokens, "pairwise": 1000})
            with self.assertRaises(JudgeCallFailure) as caught:
                policy.authorize("pointwise", "openai", 0, 0.0)
            self.assertEqual(caught.exception.category, "live_guard")
        with self.assertRaises(JudgeCallFailure):
            live_policy(max_output_tokens={"pointwise": 0, "pairwise": 10}).authorize("pointwise", "openai", 0, 0.0)
        self.assertGreaterEqual(live_policy().authorize("pointwise", "openai", 0, 0.0), 0)

    def test_failed_preflight_creates_no_run_or_attempt_and_calls_no_provider(self):
        transport = FakeTransport()
        policy = live_policy(estimated_input_tokens={"pointwise": -5, "pairwise": 1000})
        with self.assertRaises(ValueError):
            preflight_live_policy(policy, self.JUDGES)
        self.assertEqual(transport.calls, [])
        self.assertFalse((self.results / "judge").exists())
        preflight_live_policy(live_policy(), self.JUDGES)

    def test_resume_requires_the_same_execution_mode(self):
        fixture_run = self.runner.start(JudgeFixtureClient({WATER_A: [{"failure": "client_error"}]},
                                                           default=default_outcome))
        run_id = fixture_run["judgeRunId"]
        before = self.attempt_count(run_id)
        transport = FakeTransport()
        with self.assertRaises(ValueError):
            self.runner.resume(run_id, self.live_client(transport))
        self.assertEqual((transport.calls, self.attempt_count(run_id)), ([], before))
        manifest = json.loads((self.results / "judge" / run_id / "run.json").read_text(encoding="utf-8"))
        self.assertEqual((manifest["executionMode"], manifest["status"]), ("fixture", "stopped"))

        live_run = self.runner.start(self.live_client(FakeTransport((400, {}, None))))
        live_id = live_run["judgeRunId"]
        self.assertEqual((live_run["stopReason"], self.attempt_count(live_id)), ("client_error", 1))
        fixture = JudgeFixtureClient({}, default=default_outcome)
        with self.assertRaises(ValueError):
            self.runner.resume(live_id, fixture)
        self.assertEqual((fixture.calls, self.attempt_count(live_id)), ([], 1))
        again = FakeTransport((400, {}, None))
        resumed = self.runner.resume(live_id, self.live_client(again))
        self.assertEqual((resumed["stopReason"], len(again.calls), self.attempt_count(live_id)),
                         ("client_error", 1, 2))


class HttpHistoryCostAndPartialOutputTest(unittest.TestCase):
    def client(self, transport, **overrides):
        return JudgeHttpClient(live_policy(**overrides), {}, transport=transport, sleep=lambda seconds: None,
                               api_keys={"openai": "secret-key"})

    def failure(self, client):
        with self.assertRaises(JudgeCallFailure) as caught:
            client.call("openai", "pointwise", {})
        return caught.exception

    def test_request_history_is_kept_for_timeout_network_and_refused_retry(self):
        def raises(error):
            def transport(url, headers, body, timeout):
                raise error
            return transport
        timeout = self.failure(self.client(raises(TimeoutError())))
        self.assertEqual((timeout.category, [entry["outcome"] for entry in timeout.http_requests]),
                         ("timeout", ["timeout"]))
        network = self.failure(self.client(raises(ConnectionResetError())))
        self.assertEqual((network.category, [entry["outcome"] for entry in network.http_requests]),
                         ("network_error", ["network_error"]))
        refused = self.failure(self.client(FakeTransport((429, {}, 2.0)), http_request_limit=1))
        self.assertEqual((refused.category, refused.stop_run), ("live_guard", True))
        self.assertEqual([(entry["request"], entry["status"], entry["waitedSeconds"])
                          for entry in refused.http_requests], [(1, 429, 2.0)])
        for failure in (timeout, network, refused):
            self.assertNotIn("secret-key", json.dumps(failure.http_requests))

    def test_incomplete_keeps_only_allowed_partial_text_as_diagnostic(self):
        partial_content = [{"type": "output_text", "text": '{"questions'}]
        partial = self.failure(self.client(FakeTransport(openai_ok(status="incomplete", content=partial_content))))
        self.assertEqual((partial.category, partial.raw_text), ("incomplete", '{"questions'))
        empty = self.failure(self.client(FakeTransport((200, {"status": "incomplete", "usage": {}}, None))))
        self.assertEqual((empty.category, empty.raw_text), ("incomplete", None))

    def test_cost_summary_never_turns_unknown_into_zero(self):
        attempts = [{"attemptId": "a", "estimatedCostUsd": 0.5}, {"attemptId": "b", "estimatedCostUsd": 0.25},
                    {"attemptId": "c", "estimatedCostUsd": None}]
        self.assertEqual(cost_summary(attempts[:2], {"a", "b"})["selectedMeasurements"],
                         {"usd": 0.75, "knownUsd": 0.75, "unknownCostCount": 0})
        partial = cost_summary(attempts, {"a", "c"})
        self.assertEqual(partial["selectedMeasurements"], {"usd": None, "knownUsd": 0.5, "unknownCostCount": 1})
        self.assertEqual(partial["allAttempts"], {"usd": None, "knownUsd": 0.75, "unknownCostCount": 1})
        self.assertEqual(cost_summary(attempts[2:], {"c"})["selectedMeasurements"],
                         {"usd": None, "knownUsd": 0, "unknownCostCount": 1})


def strict_json(text):
    def reject(value):
        raise AssertionError("non-finite JSON value written: " + value)
    return json.loads(text, parse_constant=reject)


def openai_usage_response(cached, inputs=1000, text="{}"):
    return (200, {"status": "completed", "model": "gpt-6.1-sol", "usage": {
        "input_tokens": inputs, "input_tokens_details": {"cached_tokens": cached}, "output_tokens": 300,
        "output_tokens_details": {"reasoning_tokens": 100}},
        "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]}, None)


class CachedCostGuardTest(unittest.TestCase):
    JUDGES = [{"provider": "openai"}, {"provider": "gemini"}]

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repository = Path(self.temp.name)
        self.results = build_synthetic_pilot(self.repository)
        self.runner = JudgeRunner(self.repository, max_output_tokens=TOKENS, sleep=lambda seconds: None)

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def policy_with_cached(value, **overrides):
        policy = live_policy(**overrides)
        policy.prices["openai"]["cachedInput"] = value
        return policy

    def test_invalid_cached_price_is_rejected_by_guard_and_preflight(self):
        for value in (float("nan"), float("inf"), float("-inf"), -0.1, True, "0.1"):
            with self.subTest(cachedInput=value):
                policy = self.policy_with_cached(value)
                with self.assertRaises(JudgeCallFailure) as caught:
                    policy.authorize("pointwise", "openai", 0, 0.0)
                self.assertEqual(caught.exception.category, "live_guard")
                transport = FakeTransport()
                with self.assertRaises(ValueError):
                    preflight_live_policy(policy, self.JUDGES)
                self.assertEqual(transport.calls, [])
        self.assertFalse((self.results / "judge").exists())
        for value in (None, 0, 0.1):
            preflight_live_policy(self.policy_with_cached(value), self.JUDGES)

    def test_cached_usage_validation_and_cost(self):
        price = {"input": 2.0, "output": 10.0, "cachedInput": 0.1}
        normal = parse_response("openai", openai_usage_response(200)[1])["usage"]
        self.assertNotIn("invalidFields", normal)
        self.assertAlmostEqual(estimate_cost("openai", normal, price), (800 * 2.0 + 200 * 0.1 + 300 * 10.0) / 1e6)
        for cached, reason in ((-1, "cachedInputTokens"), (True, "cachedInputTokens"),
                               (1500, "cachedInputTokensExceedInput")):
            with self.subTest(cached=cached):
                usage = parse_response("openai", openai_usage_response(cached)[1])["usage"]
                self.assertIn(reason, usage["invalidFields"])
                self.assertIsNone(estimate_cost("openai", usage, price))
        for bad_price in (dict(price, cachedInput=float("nan")), dict(price, cachedInput=-1.0),
                          dict(price, input=float("inf"))):
            self.assertIsNone(estimate_cost("openai", normal, bad_price))

    def test_unknown_cost_keeps_reservation_so_the_total_guard_still_applies(self):
        policy = live_policy()
        estimate = policy.estimate("pointwise", "openai")
        policy.total_cost_limit = estimate * 1.5
        client = JudgeHttpClient(policy, {}, transport=FakeTransport(openai_usage_response(1500),
                                                                     openai_usage_response(0)),
                                 sleep=lambda seconds: None, api_keys={"openai": "k"})
        result = client.call("openai", "pointwise", {})
        self.assertIsNone(result["estimatedCostUsd"])
        self.assertEqual(client.reserved_cost, estimate)
        with self.assertRaises(JudgeCallFailure) as caught:
            client.call("openai", "pointwise", {})
        self.assertEqual(caught.exception.category, "live_guard")

    def test_unknown_cost_is_stored_as_null_not_nan_or_zero(self):
        # The first measurement is the OpenAI Pointwise judgement of KARI A (questions 0-2).
        transport = FakeTransport(openai_usage_response(1500, text=pointwise_output([0, 1, 2])), (400, {}, None))
        client = JudgeHttpClient(live_policy(), {}, transport=transport, sleep=lambda seconds: None,
                                 api_keys={"openai": "k", "gemini": "k"})
        manifest = self.runner.start(client)
        self.assertEqual(manifest["stopReason"], "client_error")
        run_dir = self.results / "judge" / manifest["judgeRunId"]
        attempts = [strict_json(line) for line in (run_dir / "attempts.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual((attempts[0]["outcome"], attempts[0]["estimatedCostUsd"]), ("success", None))
        self.assertEqual(attempts[0]["usage"]["invalidFields"], ["cachedInputTokensExceedInput"])
        report = self.runner.derive(manifest["judgeRunId"])
        self.assertEqual(report["cost"]["selectedMeasurements"], {"usd": None, "knownUsd": 0, "unknownCostCount": 1})
        strict_json((run_dir / "derived" / "report-counts.json").read_text(encoding="utf-8"))

    @staticmethod
    def details_response(details=None, present=True, text="{}"):
        usage = {"input_tokens": 1000, "output_tokens": 300, "output_tokens_details": {"reasoning_tokens": 100}}
        if present:
            usage["input_tokens_details"] = details
        return (200, {"status": "completed", "model": "gpt-6.1-sol", "usage": usage,
                      "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]}, None)

    def test_malformed_input_tokens_details_is_never_read_as_zero_cache(self):
        price = {"input": 2.0, "output": 10.0, "cachedInput": 0.1}
        for details in ("invalid", [], 123, True):
            with self.subTest(details=details):
                usage = parse_response("openai", self.details_response(details)[1])["usage"]
                self.assertEqual(usage["invalidFields"], ["inputTokensDetails"])
                self.assertIsNone(usage["cachedInputTokens"])
                self.assertIsNone(estimate_cost("openai", usage, price))
        # Absent (or null) details mean no cache detail was provided, not a malformed container.
        for present, details in ((False, None), (True, None)):
            with self.subTest(present=present):
                usage = parse_response("openai", self.details_response(details, present)[1])["usage"]
                self.assertNotIn("invalidFields", usage)
                self.assertAlmostEqual(estimate_cost("openai", usage, price), (1000 * 2.0 + 300 * 10.0) / 1e6)
        usage = parse_response("openai", self.details_response({"cached_tokens": 200})[1])["usage"]
        self.assertNotIn("invalidFields", usage)
        self.assertAlmostEqual(estimate_cost("openai", usage, price), (800 * 2.0 + 200 * 0.1 + 300 * 10.0) / 1e6)

    def test_malformed_details_keep_reservation_and_null_json_cost(self):
        policy = live_policy()
        estimate = policy.estimate("pointwise", "openai")
        policy.total_cost_limit = estimate * 1.5
        client = JudgeHttpClient(policy, {}, transport=FakeTransport(self.details_response("invalid"),
                                                                     self.details_response({"cached_tokens": 0})),
                                 sleep=lambda seconds: None, api_keys={"openai": "k"})
        self.assertIsNone(client.call("openai", "pointwise", {})["estimatedCostUsd"])
        self.assertEqual(client.reserved_cost, estimate)
        with self.assertRaises(JudgeCallFailure) as caught:
            client.call("openai", "pointwise", {})
        self.assertEqual(caught.exception.category, "live_guard")

        transport = FakeTransport(self.details_response([], text=pointwise_output([0, 1, 2])), (400, {}, None))
        runner_client = JudgeHttpClient(live_policy(), {}, transport=transport, sleep=lambda seconds: None,
                                        api_keys={"openai": "k", "gemini": "k"})
        manifest = self.runner.start(runner_client)
        run_dir = self.results / "judge" / manifest["judgeRunId"]
        first = strict_json((run_dir / "attempts.jsonl").read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual((first["outcome"], first["estimatedCostUsd"], first["usage"]["invalidFields"]),
                         ("success", None, ["inputTokensDetails"]))
        # Two requests were sent (malformed-usage success, then the 400); neither cost is known,
        # so both pre-call reservations remain in force.
        self.assertEqual(runner_client.reserved_cost, 2 * live_policy().estimate("pointwise", "openai"))
        report = self.runner.derive(manifest["judgeRunId"])
        self.assertEqual(report["cost"]["selectedMeasurements"], {"usd": None, "knownUsd": 0, "unknownCostCount": 1})

    def test_repetition_mismatch_stops_before_any_provider_call(self):
        path = self.results / "video-grounding.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        path.write_text("".join(json.dumps(dict(row, repetition=2) if row.get("apiStatus") == "success" else row)
                                + "\n" for row in rows), encoding="utf-8")
        client = JudgeFixtureClient({}, default=default_outcome)
        with self.assertRaises(SourceIntegrityError):
            self.runner.start(client)
        self.assertEqual(client.calls, [])
        self.assertFalse((self.results / "judge").exists())


if __name__ == "__main__":
    unittest.main()
