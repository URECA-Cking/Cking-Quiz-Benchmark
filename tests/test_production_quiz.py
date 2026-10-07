import copy
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from src.pilot_runner import FixtureProvider, PilotRunner
from src.production_grounding import (ProductionGroundingNotEligible, load_grounding_artifact,
                                      validate_production_grounding)
from src.production_quiz import (ProductionQuizNotJudgeReady, ProductionQuizRunner, new_operation_id,
                                 router_config, validate_production_config)
from src.provider_adapters import (PRODUCTION_QUIZ_PROMPTS, ProviderRouter, validate_quiz_config)
from src.provider_failure import ProviderFailure
from tests.test_pilot_runner import CONTENT, GROUNDING_V3, ROOT, VIDEO, SimulatedApiFixture
from tests.test_production_grounding import COMPLETED_RAW, DURATION, VALIDATED_AT
from tests.test_provider_adapters import FakeTransport, gemini_response, policy


PROMPT_VERSION = "production-quiz-v1"
VALID_QUIZ = {"promptVersion": PROMPT_VERSION, "questions": [
    {"question": f"질문 {i}?", "options": ["가", "나", "다", "라"], "correctOptionIndex": i,
     "explanation": "contentText 기준 설명", "sourceEvidence": "global rain and snow"} for i in range(3)]}
SETTINGS = {"thinking_level": "medium", "max_output_tokens": 8192}  # configs/production.yaml
USAGE = {"inputTokens": 100, "outputTokens": 50, "thinkingTokens": 7, "estimatedCostUsd": 0.001,
         "pricingReference": "operator-test-pricing"}


def quiz_with(**changes):
    """VALID_QUIZ with the first question changed (or the whole object for promptVersion/questions)."""
    quiz = copy.deepcopy(VALID_QUIZ)
    for key, value in changes.items():
        if key in ("promptVersion", "questions", "extra"):
            quiz[key if key != "extra" else "notes"] = value
        else:
            quiz["questions"][0][key] = value
    return quiz


class FakeQuizProvider:
    """Offline double recorded as a live Provider; it never opens a network connection."""

    is_actual_api = True

    def __init__(self, output=VALID_QUIZ, failure=None, crash=False, raw=COMPLETED_RAW, settings=SETTINGS):
        self.output, self.failure, self.crash, self.raw, self.settings = output, failure, crash, raw, settings
        self.calls = []

    def request_settings(self, kind, model, prompt_version):
        return dict(self.settings)

    def invoke(self, kind, **kwargs):
        self.calls.append(dict(kind=kind, **kwargs))
        if self.crash:
            raise KeyboardInterrupt  # stands in for a process dying after the request was sent
        if self.failure is not None:
            raise self.failure
        return dict(USAGE, normalized=copy.deepcopy(self.output), responseBody=copy.deepcopy(self.raw))


class ProductionQuizTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repository = Path(self.temp.name)
        (self.repository / "configs").mkdir()
        (self.repository / "data").mkdir()
        for name in ("pilot-v2.yaml", "production.yaml"):
            shutil.copyfile(ROOT / "configs" / name, self.repository / "configs" / name)
        shutil.copyfile(ROOT / "data" / "videos.jsonl", self.repository / "data" / "videos.jsonl")
        # The Grounding source is an artifact in the Pilot results layout, made by the real runner offline.
        self.grounding_results = self.repository / "results" / "grounding-source"
        pilot = PilotRunner(self.repository, self.grounding_results, self.repository / "configs" / "pilot-v2.yaml")
        self.grounding_row = pilot.run_grounding(VIDEO, "gemini_video", 1, SimulatedApiFixture(
            {"grounding": {"normalized": GROUNDING_V3, "responseBody": COMPLETED_RAW}}))
        self.run_id = self.grounding_row["runId"]
        self.artifact = load_grounding_artifact(self.grounding_results, self.run_id)
        self.record = validate_production_grounding(self.artifact, DURATION, VALIDATED_AT)
        self.runner = ProductionQuizRunner(self.repository)
        self.production = self.repository / "results" / "production"

    def tearDown(self):
        self.temp.cleanup()

    def generate(self, provider=None, operation_id=None, record=None, video_id=VIDEO, runner=None):
        provider = provider or FakeQuizProvider()
        operation_id = operation_id or new_operation_id()
        result = (runner or self.runner).generate(operation_id, self.grounding_results, self.run_id,
                                                  self.record if record is None else record, video_id,
                                                  DURATION, provider)
        return operation_id, result, provider

    def assert_not_judge_ready(self, operation_id, reason):
        with self.assertRaises(ProductionQuizNotJudgeReady) as caught:
            self.runner.require_judge_ready(operation_id)
        self.assertIn(reason, caught.exception.reasons)

    def files(self):
        return sorted(str(path.relative_to(self.production)) for path in self.production.rglob("*") if path.is_file()) \
            if self.production.exists() else []

    def runner_with(self, **settings):
        """A runner whose config changes the given generation settings, and those settings."""
        config = yaml.safe_load((self.repository / "configs" / "production.yaml").read_text(encoding="utf-8"))
        config["production"]["quiz_generation"]["generation_settings"].update(settings)
        path = self.repository / "configs" / ("production-" + "-".join(map(str, settings.values())) + ".yaml")
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return ProductionQuizRunner(self.repository, config=path), config["production"]["quiz_generation"]["generation_settings"]

    def attempt_file(self, operation_id, name, attempt=1):
        return self.production / "operations" / operation_id / "attempts" / str(attempt) / name

    @staticmethod
    def rewrite(path, change):
        """Tamper with one stored JSON record (files are created once, so replace it)."""
        data = json.loads(path.read_text(encoding="utf-8"))
        change(data)
        path.unlink()
        path.write_text(json.dumps(data), encoding="utf-8")

    # Grounding input and exact contentText

    def test_eligible_grounding_produces_a_judge_ready_quiz_from_the_exact_content_text(self):
        operation_id, result, provider = self.generate()
        self.assertEqual((result["apiStatus"], result["parseStatus"], result["validatorStatus"],
                          result["outputGateStatus"], result["outputGateReasons"]),
                         ("success", "pass", "pass", "pass", []))
        self.assertEqual((result["inputTokens"], result["estimatedCostUsd"]), (100, 0.001))
        sent = provider.calls[0]
        self.assertEqual((sent["kind"], sent["model"], sent["promptVersion"], sent["questionCount"], sent["optionCount"]),
                         ("quiz", "gemini-3.8-flash", PROMPT_VERSION, 3, 4))
        # The Provider got exactly the stored, validated text: same object content, bytes and SHA-256.
        self.assertEqual(sent["contentText"], self.artifact["evaluation"]["contentText"])
        self.assertEqual(sent["contentText"].encode("utf-8"), CONTENT.encode("utf-8"))
        content_hash = hashlib.sha256(sent["contentText"].encode("utf-8")).hexdigest()
        self.assertEqual(content_hash, self.record["contentTextSha256"])
        judge_input = self.runner.require_judge_ready(operation_id)
        self.assertEqual(judge_input["contentText"], sent["contentText"])
        self.assertEqual((judge_input["contentTextSha256"], judge_input["sourceGroundingRunId"],
                          judge_input["groundingValidationVersion"]),
                         (content_hash, self.run_id, self.record["version"]))
        self.assertEqual([question["questionIndex"] for question in judge_input["questions"]], [0, 1, 2])
        self.assertEqual(judge_input["questions"][1]["correctOptionIndex"], 1)
        # Provenance only: no generation model identity reaches a future Judge input.
        self.assertNotIn("model", json.dumps(judge_input))
        self.assertEqual(self.runner.operation_state(operation_id), "completed")
        # No Pilot fixed-content manifest and no Pilot result is involved.
        self.assertFalse((self.repository / "data" / "restricted").exists())
        self.assertFalse((self.grounding_results / "quiz-generation.jsonl").exists())

    def test_ineligible_groundings_never_reach_the_quiz_provider(self):
        cases = {"missingValidationRecord": (False, VIDEO),
                 "validationFailed": (dict(self.record, status="fail", failureReasons=["invalidFact"]), VIDEO),
                 "validationNotRun": (dict(self.record, status="not_run"), VIDEO),
                 "validationRecordContentTextSha256Mismatch":
                     (dict(self.record, contentTextSha256=hashlib.sha256(b"other").hexdigest()), VIDEO),
                 "sourceVideoMismatch": (self.record, "kari-microgravity-2024")}
        for reason, (record, video_id) in cases.items():
            with self.subTest(reason=reason):
                provider = FakeQuizProvider()
                with self.assertRaises(ProductionGroundingNotEligible) as caught:
                    self.runner.generate(new_operation_id(), self.grounding_results, self.run_id,
                                         None if record is False else record, video_id, DURATION, provider)
                self.assertIn(reason, caught.exception.reasons)
                self.assertEqual(provider.calls, [])
        self.assertEqual(self.files(), [])  # nothing is stored for an ineligible source
        # A changed source contentText (row hash no longer matches) is refused the same way.
        evaluation = self.grounding_results / "evaluation" / (self.run_id + ".json")
        evaluation.write_text(json.dumps({"contentText": CONTENT + " changed", "facts": self.artifact["evaluation"]["facts"]}),
                              encoding="utf-8")
        provider = FakeQuizProvider()
        with self.assertRaises(ProductionGroundingNotEligible) as caught:
            self.generate(provider)
        self.assertIn("contentTextSha256Mismatch", caught.exception.reasons)
        self.assertEqual(provider.calls, [])

    def test_a_fixture_provider_is_never_a_production_generation(self):
        with self.assertRaisesRegex(ValueError, "live Provider"):
            self.generate(FixtureProvider({"quiz": {"raw": VALID_QUIZ}}))
        self.assertEqual(self.files(), [])

    # Persistence and provenance

    def test_validation_record_and_evidence_snapshot_are_persisted_and_outlive_the_source(self):
        operation_id, result, _ = self.generate()
        operation = json.loads((self.production / "operations" / operation_id / "operation.json").read_text(encoding="utf-8"))
        snapshot_path = self.production / "sources" / (operation["sourceSnapshotSha256"] + ".json")
        snapshot_bytes = snapshot_path.read_bytes()
        self.assertEqual(hashlib.sha256(snapshot_bytes).hexdigest(), operation["sourceSnapshotSha256"])
        snapshot = json.loads(snapshot_bytes.decode("utf-8"))
        self.assertEqual(snapshot["validationRecord"], self.record)
        self.assertEqual(snapshot["grounding"]["row"], self.artifact["row"])
        self.assertEqual(snapshot["grounding"]["evaluation"]["contentText"], CONTENT)
        self.assertEqual(snapshot["grounding"]["raw"], COMPLETED_RAW)
        self.assertEqual((snapshot["videoId"], snapshot["sourceGroundingRunId"], snapshot["contentTextSha256"]),
                         (VIDEO, self.run_id, self.record["contentTextSha256"]))
        self.assertEqual((result["sourceSnapshotSha256"], result["inputFingerprint"]),
                         (operation["sourceSnapshotSha256"], operation["inputFingerprint"]))
        self.assertEqual(operation["input"], {
            "videoId": VIDEO, "sourceGroundingRunId": self.run_id, "contentTextSha256": self.record["contentTextSha256"],
            "groundingValidationVersion": "production-grounding-validation-v1", "model": "gemini-3.8-flash",
            "method": "fixed_content_text", "promptVersion": PROMPT_VERSION,
            "outputContractVersion": "production-quiz-output-v1", "generationSettings": SETTINGS})
        # The original Grounding is later changed or deleted: the stored provenance still stands.
        shutil.rmtree(self.grounding_results)
        self.assertEqual(self.runner.require_judge_ready(operation_id)["contentText"], CONTENT)

    def test_tampered_snapshot_or_attempt_artifacts_are_not_judge_ready(self):
        tampering = {
            "attemptArtifactIntegrity": lambda op, operation: (
                self.production / "operations" / op / "attempts" / "1" / "output.json").write_text(
                json.dumps({"output": VALID_QUIZ}, indent=1), encoding="ascii"),
            "sourceSnapshotIntegrity": lambda op, operation: (
                self.production / "sources" / (operation["sourceSnapshotSha256"] + ".json")).write_bytes(b"{}"),
        }
        for reason, tamper in tampering.items():
            with self.subTest(reason=reason):
                operation_id, _, _ = self.generate()
                operation = json.loads((self.production / "operations" / operation_id / "operation.json").read_text(encoding="utf-8"))
                self.runner.require_judge_ready(operation_id)
                tamper(operation_id, operation)
                self.assert_not_judge_ready(operation_id, reason)
        # A corrupted shared snapshot is also refused before any new generation uses it.
        provider = FakeQuizProvider()
        with self.assertRaisesRegex(ValueError, "snapshot does not match"):
            self.generate(provider)
        self.assertEqual(provider.calls, [])

    def test_a_snapshot_whose_evidence_no_longer_passes_the_gate_is_not_judge_ready(self):
        operation_id, _, _ = self.generate()
        operation_path = self.production / "operations" / operation_id / "operation.json"
        operation = json.loads(operation_path.read_text(encoding="utf-8"))
        snapshot = json.loads((self.production / "sources" / (operation["sourceSnapshotSha256"] + ".json")).read_text(encoding="utf-8"))
        # A self-consistent forged snapshot (hash recomputed) whose evidence fails the Grounding gate.
        snapshot["grounding"]["evaluation"]["facts"][0]["evidenceType"] = "narration"
        from src.judge_contract import canonical_json
        data = canonical_json(snapshot).encode("utf-8")
        forged = hashlib.sha256(data).hexdigest()
        (self.production / "sources" / (forged + ".json")).write_bytes(data)
        operation_path.unlink()
        operation_path.write_text(canonical_json(dict(operation, sourceSnapshotSha256=forged)), encoding="utf-8")
        # Keep the record chain self-consistent so only the snapshot's evidence is wrong.
        for name in ("started.json", "result.json"):
            self.rewrite(self.attempt_file(operation_id, name), lambda record: record.update(sourceSnapshotSha256=forged))
        self.assert_not_judge_ready(operation_id, "sourceProvenanceMismatch")

    # Output gate

    def test_provider_success_with_a_bad_quiz_is_stored_but_never_judge_ready(self):
        quote = '"global rain and snow"'  # the Pilot B methane case: quoted evidence, never auto-unquoted
        cases = {
            "malformed JSON": ("{not json", "parseFailed", "parse_error"),
            "wrong question count": (dict(VALID_QUIZ, questions=VALID_QUIZ["questions"][:2]), "questionCountMismatch", "quiz_contract_error"),
            "three options": (quiz_with(options=["가", "나", "다"]), "invalidQuestionStructure", "quiz_contract_error"),
            "duplicate options": (quiz_with(options=["가", "가 ", "다", "라"]), "invalidQuestionStructure", "quiz_contract_error"),
            "index out of range": (quiz_with(correctOptionIndex=4), "invalidQuestionStructure", "quiz_contract_error"),
            "negative index": (quiz_with(correctOptionIndex=-1), "invalidQuestionStructure", "quiz_contract_error"),
            "string index": (quiz_with(correctOptionIndex="0"), "parseFailed", "parse_error"),
            "empty explanation": (quiz_with(explanation=" "), "invalidQuestionStructure", "quiz_contract_error"),
            "empty sourceEvidence": (quiz_with(sourceEvidence=""), "invalidQuestionStructure", "quiz_contract_error"),
            "empty question": (quiz_with(question=" "), "invalidQuestionStructure", "quiz_contract_error"),
            "evidence not in content": (quiz_with(sourceEvidence=quote), "evidenceNotInContent", "evidence_not_in_content"),
            "pilot promptVersion": (quiz_with(promptVersion="pilot-v2"), "promptVersionMismatch", "quiz_contract_error"),
            "extra top-level field": (quiz_with(extra="note"), "unexpectedFields", "quiz_contract_error"),
            "extra question field": (quiz_with(difficulty="easy"), "unexpectedFields", "quiz_contract_error"),
        }
        for name, (output, reason, category) in cases.items():
            with self.subTest(case=name):
                operation_id, result, _ = self.generate(FakeQuizProvider(output))
                self.assertEqual((result["apiStatus"], result["outputGateStatus"], result["errorCategory"]),
                                 ("success", "fail", category))
                self.assertIn(reason, result["outputGateReasons"])
                stored = json.loads((self.production / "operations" / operation_id / "attempts" / "1"
                                     / "output.json").read_text(encoding="ascii"))
                self.assertEqual(stored["output"], output)  # preserved as generated, for diagnosis
                self.assert_not_judge_ready(operation_id, reason)
                # The Provider success is final for this operation: no automatic regeneration.
                self.assertEqual(self.runner.operation_state(operation_id), "completed")
                with self.assertRaisesRegex(ValueError, "new operation"):
                    self.generate(FakeQuizProvider(), operation_id)

    # Identity, retry and regeneration

    def test_a_completed_operation_is_never_run_again(self):
        operation_id, _, _ = self.generate()
        provider = FakeQuizProvider()
        with self.assertRaisesRegex(ValueError, "new operation"):
            self.generate(provider, operation_id)
        self.assertEqual(provider.calls, [])

    def test_an_operation_is_never_reused_with_another_input_config_or_evidence(self):
        operation_id, _, _ = self.generate(FakeQuizProvider(failure=ProviderFailure("server_error", 503)))
        high, high_settings = self.runner_with(thinking_level="high")
        capped, capped_settings = self.runner_with(max_output_tokens=1000)
        changes = {"thinking level": (dict(runner=high), high_settings),
                   "output token cap": (dict(runner=capped), capped_settings),
                   "re-validated record": (dict(record=dict(self.record, validatedAt="2026-10-09T00:00:00+00:00")), SETTINGS)}
        for name, (change, settings) in changes.items():
            with self.subTest(change=name):
                provider = FakeQuizProvider(settings=settings)
                with self.assertRaisesRegex(ValueError, "different input, config or evidence"):
                    self.generate(provider, operation_id, **change)
                self.assertEqual(provider.calls, [])

    def test_a_technical_failure_allows_an_explicit_retry_of_the_same_operation(self):
        operation_id, failed, _ = self.generate(FakeQuizProvider(failure=ProviderFailure("server_error", 503)))
        self.assertEqual((failed["apiStatus"], failed["errorCategory"], failed["httpStatus"], failed["outputGateStatus"]),
                         ("error", "server_error", 503, "fail"))
        self.assertEqual(self.runner.operation_state(operation_id), "failed")
        self.assert_not_judge_ready(operation_id, "noCompletedGeneration")
        _, retried, _ = self.generate(operation_id=operation_id)
        self.assertEqual((retried["attempt"], retried["inputFingerprint"]), (2, failed["inputFingerprint"]))
        self.assertEqual(self.runner.require_judge_ready(operation_id)["attempt"], 2)

    def test_a_new_operation_is_a_separate_generation_with_the_same_fingerprint(self):
        first, first_result, _ = self.generate()
        second, second_result, _ = self.generate()
        self.assertNotEqual(first, second)
        self.assertEqual(first_result["inputFingerprint"], second_result["inputFingerprint"])
        self.assertEqual(self.runner.require_judge_ready(second)["operationId"], second)
        self.assertEqual(len(list((self.production / "sources").glob("*.json"))), 1)  # one shared snapshot

    # Partial and uncertain state

    def test_an_interrupted_request_is_uncertain_and_never_replayed(self):
        operation_id = new_operation_id()
        with self.assertRaises(KeyboardInterrupt):
            self.generate(FakeQuizProvider(crash=True), operation_id)
        attempt = self.production / "operations" / operation_id / "attempts" / "1"
        self.assertTrue((attempt / "started.json").is_file())
        self.assertFalse((attempt / "result.json").exists())
        self.assertEqual(self.runner.operation_state(operation_id), "uncertain")
        self.assert_not_judge_ready(operation_id, "operationUncertain")
        provider = FakeQuizProvider()
        with self.assertRaisesRegex(ValueError, "not replayed"):
            self.generate(provider, operation_id)
        self.assertEqual(provider.calls, [])

    def test_an_attempt_directory_without_a_started_record_is_not_an_attempt(self):
        operation_id = new_operation_id()
        with self.assertRaises(KeyboardInterrupt):
            self.generate(FakeQuizProvider(crash=True), operation_id)
        (self.production / "operations" / operation_id / "attempts" / "1" / "started.json").unlink()
        self.assertEqual(self.runner.operation_state(operation_id), "not_started")
        _, result, _ = self.generate(operation_id=operation_id)
        self.assertEqual(result["attempt"], 1)

    def test_artifacts_without_the_result_record_are_not_completed(self):
        operation_id, _, _ = self.generate()
        attempt = self.production / "operations" / operation_id / "attempts" / "1"
        (attempt / "result.json").unlink()  # as if the process died after raw/output but before the result
        self.assertTrue((attempt / "output.json").is_file())
        self.assertEqual(self.runner.operation_state(operation_id), "uncertain")
        self.assert_not_judge_ready(operation_id, "operationUncertain")

    # Codex review: request settings, provenance chain and technical retry

    def test_output_token_cap_and_thinking_level_are_fingerprinted_and_bound_to_the_provider(self):
        fingerprints = {}
        for settings in ({"max_output_tokens": 1000}, {"max_output_tokens": 2000}, {"thinking_level": "high"}):
            runner, expected = self.runner_with(**settings)
            transport = FakeTransport(gemini_response(VALID_QUIZ))
            router = ProviderRouter(policy(max_output_tokens=expected["max_output_tokens"]), transport=transport,
                                    api_keys={"gemini": "test-key"}, pilot_config=router_config(runner.config))
            _, result, _ = self.generate(router, runner=runner)
            self.assertEqual(transport.calls[0][2]["generation_config"], expected)  # what was actually sent
            fingerprints[json.dumps(settings)] = result["inputFingerprint"]
        _, default, _ = self.generate()
        fingerprints["default"] = default["inputFingerprint"]
        self.assertEqual(len(set(fingerprints.values())), 4)

    def test_a_provider_whose_request_settings_differ_from_the_config_is_refused_before_any_call(self):
        runner, expected = self.runner_with(max_output_tokens=1000)
        transport = FakeTransport(gemini_response(VALID_QUIZ))
        router = ProviderRouter(policy(max_output_tokens=2000), transport=transport,
                                api_keys={"gemini": "test-key"}, pilot_config=router_config(runner.config))
        cases = {"live policy cap 2000, config 1000": router,
                 "reported thinking level": FakeQuizProvider(settings=dict(SETTINGS, thinking_level="high")),
                 "reported output cap": FakeQuizProvider(settings=dict(SETTINGS, max_output_tokens=4096))}
        for name, provider in cases.items():
            with self.subTest(case=name):
                with self.assertRaisesRegex(ValueError, "request settings differ"):
                    self.generate(provider, runner=runner if provider is router else None)
        self.assertEqual(transport.calls, [])
        self.assertTrue(all(not provider.calls for provider in cases.values() if provider is not router))
        self.assertEqual(self.files(), [])
        with self.assertRaises(ProviderFailure):  # the router itself also refuses the mismatched cap
            router.invoke("quiz", model="gemini-3.8-flash", promptVersion=PROMPT_VERSION, questionCount=3,
                          optionCount=4, contentText=CONTENT)
        self.assertEqual(transport.calls, [])

        class Unverifiable(FakeQuizProvider):
            request_settings = None

        provider = Unverifiable()
        with self.assertRaisesRegex(ValueError, "cannot report its request settings"):
            self.generate(provider)
        self.assertEqual(provider.calls, [])

    def test_judge_ready_verifies_the_whole_started_to_result_provenance_chain(self):
        other_run = "0" * 32
        tampering = {
            "result model": ("result.json", lambda r: r.update(model="gemini-3.8-pro")),
            "result sourceGroundingRunId": ("result.json", lambda r: r.update(sourceGroundingRunId=other_run)),
            "result promptVersion": ("result.json", lambda r: r.update(promptVersion="pilot-v2")),
            "result contentTextSha256": ("result.json", lambda r: r.update(contentTextSha256="f" * 64)),
            "result inputFingerprint": ("result.json", lambda r: r.update(inputFingerprint="f" * 64)),
            "result sourceSnapshotSha256": ("result.json", lambda r: r.update(sourceSnapshotSha256="f" * 64)),
            "result operationId": ("result.json", lambda r: r.update(operationId="f" * 32)),
            "result attempt": ("result.json", lambda r: r.update(attempt=2)),
            "result startedAt": ("result.json", lambda r: r.update(startedAt="2026-01-01T00:00:00+00:00")),
            "result completedAt invalid": ("result.json", lambda r: r.update(completedAt="2026-99-99T25:61:61+00:00")),
            "result extra field": ("result.json", lambda r: r.update(judgeReady=True)),
            "result outputSha256": ("result.json", lambda r: r.update(outputSha256="f" * 64)),
            "started operationId": ("started.json", lambda r: r.update(operationId="f" * 32)),
            "started attempt": ("started.json", lambda r: r.update(attempt=2)),
            "started inputFingerprint": ("started.json", lambda r: r.update(inputFingerprint="f" * 64)),
            "started sourceSnapshotSha256": ("started.json", lambda r: r.update(sourceSnapshotSha256="f" * 64)),
            "started startedAt": ("started.json", lambda r: r.update(startedAt="2026-01-01T00:00:00+00:00")),
            "started missing": ("started.json", None),
        }
        for name, (file_name, change) in tampering.items():
            with self.subTest(tamper=name):
                operation_id, _, _ = self.generate()  # a fresh operation per case; the snapshot stays intact
                self.runner.require_judge_ready(operation_id)
                path = self.attempt_file(operation_id, file_name)
                if change is None:
                    path.unlink()
                else:
                    self.rewrite(path, change)
                with self.assertRaises(ProductionQuizNotJudgeReady):
                    self.runner.require_judge_ready(operation_id)
                self.assertEqual(self.runner.operation_state(operation_id), "corrupted")
        untouched, _, _ = self.generate()
        self.assertEqual(self.runner.require_judge_ready(untouched)["operationId"], untouched)

    def test_only_a_verified_technical_error_allows_the_next_attempt(self):
        failure = ProviderFailure("server_error", 503)
        tampering = {
            "apiStatus missing": ("result.json", lambda r: r.pop("apiStatus")),
            "apiStatus null": ("result.json", lambda r: r.update(apiStatus=None)),
            "apiStatus not_run": ("result.json", lambda r: r.update(apiStatus="not_run")),
            "apiStatus unknown": ("result.json", lambda r: r.update(apiStatus="retryable")),
            "error without category": ("result.json", lambda r: r.update(errorCategory=None)),
            "result model": ("result.json", lambda r: r.update(model="gemini-3.8-pro")),
            "result sourceGroundingRunId": ("result.json", lambda r: r.update(sourceGroundingRunId="0" * 32)),
            "started inputFingerprint": ("started.json", lambda r: r.update(inputFingerprint="f" * 64)),
            "started attempt": ("started.json", lambda r: r.update(attempt=3)),
            "result only (started missing)": ("started.json", None),
            "malformed result": ("result.json", "{not json"),
            "error with a stray output file": ("output.json", "{}"),
        }
        for name, (file_name, change) in tampering.items():
            with self.subTest(tamper=name):
                operation_id, _, _ = self.generate(FakeQuizProvider(failure=failure))
                path = self.attempt_file(operation_id, file_name)
                if change is None:
                    path.unlink()
                elif isinstance(change, str):
                    path.unlink(missing_ok=True)
                    path.write_text(change, encoding="utf-8")
                else:
                    self.rewrite(path, change)
                provider = FakeQuizProvider()
                with self.assertRaises(ValueError):
                    self.generate(provider, operation_id)
                self.assertEqual(provider.calls, [])
                self.assertFalse(self.attempt_file(operation_id, "started.json", attempt=2).exists())
        # Untouched technical error: an explicit second attempt runs.
        operation_id, _, _ = self.generate(FakeQuizProvider(failure=failure))
        _, retried, provider = self.generate(operation_id=operation_id)
        self.assertEqual((retried["attempt"], len(provider.calls)), (2, 1))

    def test_a_provider_success_with_a_failed_output_gate_is_never_retried(self):
        operation_id, result, _ = self.generate(FakeQuizProvider(quiz_with(sourceEvidence='"global rain and snow"')))
        self.assertEqual((result["apiStatus"], result["errorCategory"]), ("success", "evidence_not_in_content"))
        provider = FakeQuizProvider()
        with self.assertRaisesRegex(ValueError, "new operation"):
            self.generate(provider, operation_id)
        self.assertEqual(provider.calls, [])

    # Router, config and Pilot isolation

    def test_router_sends_the_production_prompt_settings_and_exact_content_text(self):
        transport = FakeTransport(gemini_response(VALID_QUIZ))
        router = ProviderRouter(policy(max_output_tokens=8192), transport=transport, api_keys={"gemini": "test-key"},
                                pilot_config=router_config(self.runner.config))
        operation_id, result, _ = self.generate(router)
        self.assertEqual(result["outputGateStatus"], "pass")
        body = transport.calls[0][2]
        self.assertEqual((body["model"], body["generation_config"]), ("gemini-3.8-flash", SETTINGS))
        text = body["input"][0]["text"]
        self.assertTrue(text.startswith(PRODUCTION_QUIZ_PROMPTS[PROMPT_VERSION]))
        self.assertTrue(text.endswith("\n" + CONTENT))
        self.runner.require_judge_ready(operation_id)
        # The production router serves no Pilot prompt, Grounding or Direct; Pilot routers no production prompt.
        for kind, version in (("quiz", "pilot-v2"), ("direct", PROMPT_VERSION)):
            with self.subTest(kind=kind, version=version), self.assertRaises(ProviderFailure):
                router.invoke(kind, model="gemini-3.8-flash", promptVersion=version, questionCount=3, optionCount=4,
                              contentText=CONTENT, video={"youtubeUrl": "https://www.youtube.com/watch?v=x"})
        pilot_config = yaml.safe_load((ROOT / "configs" / "pilot-v2.yaml").read_text(encoding="utf-8"))["pilot"]
        pilot_router = ProviderRouter(policy(), transport=FakeTransport(gemini_response(VALID_QUIZ)),
                                      api_keys={"gemini": "test-key", "openai": "test-key"}, pilot_config=pilot_config)
        with self.assertRaises(ProviderFailure):
            pilot_router.invoke("quiz", model="gemini-3.8-flash", promptVersion=PROMPT_VERSION, questionCount=3,
                                optionCount=4, contentText=CONTENT)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            validate_quiz_config(dict(pilot_config, prompt_version=PROMPT_VERSION))

    def test_production_config_and_results_location_are_checked(self):
        config = yaml.safe_load((ROOT / "configs" / "production.yaml").read_text(encoding="utf-8"))["production"]
        validate_production_config(config)
        bad = {"pilot prompt": ("prompt_version", "pilot-v2"), "openai": ("provider", "openai"),
               "two questions": ("questions_per_quiz", 2), "bool options": ("options_per_question", True),
               "unknown output contract": ("output_contract_version", "production-quiz-output-v2"),
               "no thinking level": ("generation_settings", {"max_output_tokens": 8192}),
               "no output cap": ("generation_settings", {"thinking_level": "medium"}),
               "bool output cap": ("generation_settings", {"thinking_level": "medium", "max_output_tokens": True})}
        for name, (key, value) in bad.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                changed = copy.deepcopy(config)
                changed["quiz_generation"][key] = value
                validate_production_config(changed)
        with self.assertRaises(ValueError):
            validate_production_config(dict(config, grounding_validation_version="production-grounding-validation-v2"))
        with self.assertRaisesRegex(ValueError, "results/production"):
            ProductionQuizRunner(self.repository, self.repository / "results" / "pilot-v2-r2")


if __name__ == "__main__":
    unittest.main()
