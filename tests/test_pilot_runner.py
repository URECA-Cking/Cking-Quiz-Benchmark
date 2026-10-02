import json
import hashlib
import os
import shutil
import tempfile
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from unittest import mock

from src.pilot_runner import FixtureProvider, PilotRunner


ROOT = Path(__file__).resolve().parents[1]
VIDEO = "nasa-water-cycle-2019"
CONTENT = "NASA measures global rain and snow every 30 minutes."
QUIZ = {
    "promptVersion": "pilot-v1",
    "questions": [
        {"question": f"Question {i}?", "options": ["A", "B", "C", "D"],
         "correctOptionIndex": 0, "explanation": "Because NASA observes it.",
         "sourceEvidence": "global rain and snow"}
        for i in range(3)
    ],
}
GROUNDING = {
    "contentText": CONTENT,
    "facts": [{"fact": "NASA measures rainfall", "evidenceType": "speech",
               "evidence": "global rain and snow", "timestampStartSeconds": 61,
               "timestampEndSeconds": 73}],
}


FIXTURE_RAW = {"source": "fixture"}


class SimulatedApiFixture(FixtureProvider):
    """Offline double whose responses are recorded as successful API runs."""

    is_actual_api = True


class PilotRunnerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repository = Path(self.temp.name)
        (self.repository / "configs").mkdir()
        (self.repository / "data").mkdir()
        shutil.copyfile(ROOT / "configs" / "pilot.yaml", self.repository / "configs" / "pilot.yaml")
        shutil.copyfile(ROOT / "data" / "videos.jsonl", self.repository / "data" / "videos.jsonl")
        self.results = self.repository / "results"
        self.runner = PilotRunner(self.repository, self.results)

    def tearDown(self):
        self.temp.cleanup()

    def rows(self, name):
        return [json.loads(line) for line in (self.results / name).read_text(encoding="utf-8").splitlines()]

    def approved_source(self, content=CONTENT, video_id=VIDEO, approval="approved",
                        api_status="success", evaluation_content=None, runner=None):
        runner = runner or self.runner
        source = runner._base("video_grounding", video_id, "gemini_video",
                              "gemini-3.8-flash", 1)
        source.update(runId=uuid.uuid4().hex, apiStatus=api_status,
                      contentTextSha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
                      contentTextApprovalStatus=approval, groundingFacts=[],
                      omission=None, hallucination=None)
        runner.results.mkdir(parents=True, exist_ok=True)
        with (runner.results / "video-grounding.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(source) + "\n")
        evaluation_dir = runner.results / "evaluation"
        evaluation_dir.mkdir(exist_ok=True)
        (evaluation_dir / (source["runId"] + ".json")).write_text(
            json.dumps({"contentText": content if evaluation_content is None else evaluation_content,
                        "facts": []}), encoding="utf-8")
        raw_dir = runner.results / "raw"
        raw_dir.mkdir(exist_ok=True)
        (raw_dir / (source["runId"] + ".json")).write_text(
            json.dumps({"source": "fixture"}), encoding="utf-8")
        return source["runId"]

    def run_approved_quiz(self, video_id, model, repetition, content, provider, runner=None):
        runner = runner or self.runner
        source_id = self.approved_source(content, video_id, runner=runner)
        return runner.run_quiz(video_id, model, repetition, content, provider,
                               source_grounding_run_id=source_id)

    def test_quiz_rejects_missing_human_approval_before_provider_call(self):
        provider = FixtureProvider({"quiz": {"raw": QUIZ}})
        with self.assertRaisesRegex(ValueError, "approved"):
            self.runner.run_quiz(VIDEO, "gemini-3.8-flash", 1, CONTENT, provider)
        self.assertEqual(provider.calls, [])

    def test_quiz_accepts_only_matching_approved_grounding_content(self):
        source_id = self.approved_source()
        provider = FixtureProvider({"quiz": {"raw": QUIZ}})
        row = self.runner.run_quiz(VIDEO, "gemini-3.8-flash", 1, CONTENT, provider,
                                   source_grounding_run_id=source_id)
        self.assertEqual(row["sourceGroundingRunId"], source_id)
        self.assertEqual(row["contentTextSha256"], hashlib.sha256(CONTENT.encode()).hexdigest())
        self.assertEqual(provider.calls[0]["contentText"], CONTENT)

    def test_quiz_rejects_unapproved_legacy_and_mismatched_sources(self):
        provider = FixtureProvider({"quiz": {"raw": QUIZ}})
        legacy_id = self.approved_source(approval=None)
        result_file = self.results / "video-grounding.jsonl"
        legacy = self.rows("video-grounding.jsonl")
        legacy[0].pop("contentTextApprovalStatus")
        result_file.write_text(json.dumps(legacy[0]) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "approved"):
            self.runner.run_quiz(VIDEO, "gemini-3.8-flash", 1, CONTENT, provider,
                                 source_grounding_run_id=legacy_id)
        for kwargs in ({"approval": None}, {"api_status": "error"},
                       {"evaluation_content": CONTENT + " changed"}):
            with self.subTest(kwargs=kwargs):
                source_id = self.approved_source(**kwargs)
                with self.assertRaisesRegex(ValueError, "approved"):
                    self.runner.run_quiz(VIDEO, "gemini-3.8-flash", 1, CONTENT, provider,
                                         source_grounding_run_id=source_id)
        source_id = self.approved_source()
        for video_id, content in (("nasa-methane-2020", CONTENT), (VIDEO, CONTENT + " changed")):
            with self.subTest(video_id=video_id, content=content):
                with self.assertRaisesRegex(ValueError, "approved"):
                    self.runner.run_quiz(video_id, "gemini-3.8-flash", 1, content, provider,
                                         source_grounding_run_id=source_id)
        self.assertEqual(provider.calls, [])

    def test_explicit_approval_registration_preserves_existing_run_data(self):
        source_id = self.approved_source(approval=None)
        result_file = self.results / "video-grounding.jsonl"
        legacy = self.rows("video-grounding.jsonl")[0]
        legacy.pop("contentTextSha256")
        legacy.pop("contentTextApprovalStatus")
        result_file.write_text(json.dumps(legacy) + "\n", encoding="utf-8")
        before = self.rows("video-grounding.jsonl")[0]
        evaluation_file = self.results / "evaluation" / (source_id + ".json")
        evaluation_before = evaluation_file.read_bytes()
        self.runner.approve_content(source_id)
        after = self.rows("video-grounding.jsonl")[0]
        self.assertEqual(after["contentTextApprovalStatus"], "approved")
        self.assertEqual(after["contentTextSha256"], hashlib.sha256(CONTENT.encode()).hexdigest())
        for key in before:
            if key not in ("contentTextApprovalStatus", "contentTextSha256"):
                self.assertEqual(after[key], before[key])
        self.assertEqual(evaluation_file.read_bytes(), evaluation_before)
        self.assertEqual(len(self.rows("video-grounding.jsonl")), 1)

    def test_explicit_approval_rejects_changed_evaluation_without_rewriting_result(self):
        source_id = self.approved_source(approval=None, evaluation_content=CONTENT + " changed")
        result_file = self.results / "video-grounding.jsonl"
        before = result_file.read_bytes()
        with self.assertRaisesRegex(ValueError, "hash does not match"):
            self.runner.approve_content(source_id)
        self.assertEqual(result_file.read_bytes(), before)

    def test_two_stage_e2e_does_not_call_grounding_or_quiz(self):
        provider = FixtureProvider({"grounding": {"raw": GROUNDING}, "quiz": {"raw": QUIZ}})
        with self.assertRaisesRegex(ValueError, "approval"):
            self.runner.run_end_to_end(VIDEO, "gemini_grounding_openai_quiz", 1, provider)
        self.assertEqual(provider.calls, [])

    def test_grounding_keeps_human_reviews_unset_and_separates_raw(self):
        provider = FixtureProvider({"grounding": {"raw": GROUNDING, "inputTokens": 10}})
        row = self.runner.run_grounding(VIDEO, "gemini_video", 1, provider)
        self.assertEqual(row["apiStatus"], "not_run")
        self.assertIsNone(row["latencyMs"])
        self.assertIsNone(row["inputTokens"])
        self.assertIsNone(row["groundingFacts"][0]["factExists"])
        self.assertIsNone(row["groundingFacts"][0]["timestampAccurate"])
        self.assertIsNone(row["hallucination"])
        self.assertEqual(row["contentTextSha256"], hashlib.sha256(CONTENT.encode()).hexdigest())
        self.assertIsNone(row["contentTextApprovalStatus"])
        self.assertEqual(self.rows("video-grounding.jsonl")[0]["runId"], row["runId"])
        self.assertTrue((self.results / "raw" / (row["runId"] + ".json")).exists())
        self.assertNotIn("contentText", row)

    def test_grounding_evidence_type_contract_accepts_only_three_values(self):
        for value in ("speech", "visual", "unknown", "narration", "combined"):
            with self.subTest(evidence_type=value):
                fact = dict(GROUNDING["facts"][0], evidenceType=value)
                payload = dict(GROUNDING, facts=[fact])
                row = self.runner.run_grounding(
                    VIDEO, "gemini_video", 1, FixtureProvider({"grounding": {"raw": payload}}))
                if value in ("speech", "visual", "unknown"):
                    self.assertIsNone(row["errorCategory"])
                    self.assertEqual(row["groundingFacts"][0]["evidenceType"], value)
                else:
                    self.assertEqual(row["errorCategory"], "fixture_invalid_grounding_response")
                    self.assertNotIn("contentTextSha256", row)

    def test_existing_three_attempts_are_preserved_before_fourth(self):
        self.results.mkdir()
        result_file = self.results / "video-grounding.jsonl"
        existing = "".join(json.dumps({"videoId": VIDEO, "method": "gemini_video",
                                        "model": "gemini-3.8-flash", "repetition": 1,
                                        "runId": "previous-" + str(number), "apiStatus": "error",
                                        **({} if number == 1 else {"attempt": number})}) + "\n"
                           for number in (1, 2, 3))
        result_file.write_text(existing, encoding="utf-8")
        row = self.runner.run_grounding(
            VIDEO, "gemini_video", 1, FixtureProvider({"grounding": {"raw": GROUNDING}}))
        self.assertEqual(row["attempt"], 4)
        self.assertTrue(result_file.read_text(encoding="utf-8").startswith(existing))

    def test_attempt_starts_at_one_and_failed_run_increments_without_overwrite(self):
        failed = self.runner.run_grounding(
            VIDEO, "gemini_video", 1, FixtureProvider({"grounding": {"errorCategory": "rate_limit"}}))
        retried = self.runner.run_grounding(
            VIDEO, "gemini_video", 1, FixtureProvider({"grounding": {"raw": GROUNDING}}))
        rows = self.rows("video-grounding.jsonl")
        self.assertEqual([row["attempt"] for row in rows], [1, 2])
        self.assertEqual([row["runId"] for row in rows], [failed["runId"], retried["runId"]])
        self.assertNotEqual(failed["runId"], retried["runId"])
        self.assertEqual(rows[0]["errorCategory"], "fixture_rate_limit")

    def test_legacy_result_without_attempt_counts_as_first_attempt(self):
        self.results.mkdir()
        existing = {"videoId": VIDEO, "method": "gemini_video", "repetition": 1,
                    "model": "gemini-3.8-flash", "runId": "existing-failure", "apiStatus": "error"}
        result_file = self.results / "video-grounding.jsonl"
        original = json.dumps(existing) + "\n"
        result_file.write_text(original, encoding="utf-8")
        row = self.runner.run_grounding(VIDEO, "gemini_video", 1,
                                        FixtureProvider({"grounding": {"raw": GROUNDING}}))
        self.assertEqual(row["attempt"], 2)
        self.assertTrue(result_file.read_text(encoding="utf-8").startswith(original))
        self.assertNotIn("attempt", self.rows("video-grounding.jsonl")[0])

    def test_attempt_sequence_is_separate_for_repetition_video_method_and_model(self):
        provider = FixtureProvider({"grounding": {"raw": GROUNDING}, "quiz": {"raw": QUIZ}})
        self.runner.run_grounding(VIDEO, "gemini_video", 1, provider)
        second_repetition = self.runner.run_grounding(VIDEO, "gemini_video", 2, provider)
        other_video = self.runner.run_grounding("nasa-methane-2020", "gemini_video", 1, provider)
        other_method = self.runner.run_grounding(VIDEO, "authorized_transcript", 1, provider,
                                                 authorized_transcript=CONTENT)
        gemini = self.run_approved_quiz(VIDEO, "gemini-3.8-flash", 1, CONTENT, provider)
        openai = self.run_approved_quiz(VIDEO, "gpt-5.4-mini", 1, CONTENT, provider)
        for row in (second_repetition, other_video, other_method, gemini, openai):
            self.assertEqual(row["attempt"], 1)

    def test_grounding_attempt_uses_video_method_model_and_repetition(self):
        provider = FixtureProvider({"grounding": {"raw": GROUNDING}})
        first = self.runner.run_grounding(VIDEO, "gemini_video", 1, provider)
        same = self.runner.run_grounding(VIDEO, "gemini_video", 1, provider)
        other_repetition = self.runner.run_grounding(VIDEO, "gemini_video", 2, provider)
        self.assertEqual((first["attempt"], same["attempt"], other_repetition["attempt"]), (1, 2, 1))

        # A previously used model must not consume this model's attempt sequence.
        other_model = dict(first, model="another-video-model", attempt=5)
        with (self.results / "video-grounding.jsonl").open("a", encoding="utf-8") as target:
            target.write(json.dumps(other_model) + "\n")
        self.assertEqual(self.runner.run_grounding(VIDEO, "gemini_video", 1, provider)["attempt"], 3)

    def test_quiz_attempt_uses_model_prompt_version_content_hash_and_repetition(self):
        provider = FixtureProvider({"quiz": {"raw": QUIZ}})
        first = self.run_approved_quiz(VIDEO, "gemini-3.8-flash", 1, CONTENT, provider)
        self.assertEqual(first["attempt"], 1)
        self.assertEqual(self.run_approved_quiz(VIDEO, "gemini-3.8-flash", 1, CONTENT, provider)["attempt"], 2)
        self.assertEqual(self.run_approved_quiz(VIDEO, "gpt-5.4-mini", 1, CONTENT, provider)["attempt"], 1)
        self.assertEqual(self.run_approved_quiz(VIDEO, "gemini-3.8-flash", 2, CONTENT, provider)["attempt"], 1)

        result_file = self.results / "quiz-generation.jsonl"
        with result_file.open("a", encoding="utf-8") as target:
            target.write(json.dumps(dict(first, promptVersion="pilot-v2", attempt=8)) + "\n")
            target.write(json.dumps(dict(first, contentTextSha256="0" * 64, attempt=9)) + "\n")
        self.assertEqual(self.run_approved_quiz(VIDEO, "gemini-3.8-flash", 1, CONTENT, provider)["attempt"], 3)

        for changed in ({"promptVersion": "pilot-v2"}, {"contentTextSha256": "0" * 64}):
            with self.subTest(changed=changed):
                isolated = self.results / ("changed-prompt" if "promptVersion" in changed else "changed-content")
                isolated.mkdir()
                (isolated / "quiz-generation.jsonl").write_text(
                    json.dumps(dict(first, attempt=7, **changed)) + "\n", encoding="utf-8")
                for directory in ("raw", "evaluation"):
                    target = isolated / directory
                    target.mkdir()
                    shutil.copyfile(self.results / directory / (first["runId"] + ".json"),
                                    target / (first["runId"] + ".json"))
                row = self.run_approved_quiz(VIDEO, "gemini-3.8-flash", 1, CONTENT, provider,
                                             runner=PilotRunner(self.repository, isolated))
                self.assertEqual(row["attempt"], 1)

    def test_two_models_use_exact_same_fixed_content_and_hash(self):
        provider = FixtureProvider({"quiz": {"raw": QUIZ}})
        source_id = self.approved_source()
        gemini = self.runner.run_quiz(VIDEO, "gemini-3.8-flash", 1, CONTENT, provider,
                                      source_grounding_run_id=source_id)
        openai = self.runner.run_quiz(VIDEO, "gpt-5.4-mini", 1, CONTENT, provider,
                                      source_grounding_run_id=source_id)
        self.assertEqual(provider.calls[0]["contentText"], provider.calls[1]["contentText"])
        self.assertEqual(gemini["contentTextSha256"], openai["contentTextSha256"])
        self.assertEqual(gemini["sourceGroundingRunId"], openai["sourceGroundingRunId"])
        self.assertEqual(gemini["beCompatibility"], "pass")
        self.assertIsNone(gemini["questionReviews"][0]["answerAccuracy"])

    def test_fixed_content_cannot_differ_between_models(self):
        provider = FixtureProvider({"quiz": {"raw": QUIZ}})
        self.run_approved_quiz(VIDEO, "gemini-3.8-flash", 1, CONTENT, provider)
        with self.assertRaisesRegex(ValueError, "fixed contentText"):
            self.run_approved_quiz(VIDEO, "gpt-5.4-mini", 1, CONTENT + " changed", provider)
        self.assertEqual(len(provider.calls), 1)

    def test_canonical_input_survives_different_result_directories(self):
        first = PilotRunner(self.repository, self.repository / "results" / "first")
        second = PilotRunner(self.repository, self.repository / "results" / "second")
        self.run_approved_quiz(VIDEO, "gemini-3.8-flash", 1, CONTENT,
                               FixtureProvider({"quiz": {"raw": QUIZ}}), runner=first)
        provider = FixtureProvider({"quiz": {"raw": QUIZ}})
        with self.assertRaisesRegex(ValueError, "fixed contentText"):
            self.run_approved_quiz(VIDEO, "gpt-5.4-mini", 1, CONTENT + " changed",
                                   provider, runner=second)
        self.assertFalse(provider.calls)

    def test_canonical_input_is_scoped_to_video_and_prompt_version(self):
        provider = FixtureProvider({"quiz": {"raw": QUIZ}})
        self.run_approved_quiz(VIDEO, "gemini-3.8-flash", 1, CONTENT, provider)
        other_video = "nasa-methane-2020"
        self.run_approved_quiz(other_video, "gpt-5.4-mini", 1, CONTENT + " other", provider)
        config = self.repository / "configs" / "pilot.yaml"
        config.write_text(config.read_text(encoding="utf-8").replace("pilot-v1", "pilot-v2"), encoding="utf-8")
        next_version = PilotRunner(self.repository, self.repository / "results" / "next")
        self.run_approved_quiz(VIDEO, "gpt-5.4-mini", 1, CONTENT + " changed",
                               provider, runner=next_version)
        self.assertEqual(len(provider.calls), 3)

    def test_custom_results_directory_links_grounding_approval_and_quiz(self):
        custom = PilotRunner(self.repository, self.results / "custom")
        grounding = custom.run_grounding(VIDEO, "gemini_video", 1,
                                         SimulatedApiFixture({"grounding": {
                                             "normalized": GROUNDING, "responseBody": FIXTURE_RAW}}))
        self.assertEqual(grounding["apiStatus"], "success")
        self.assertFalse((self.results / "video-grounding.jsonl").exists())
        custom.approve_content(grounding["runId"])
        saved = [json.loads(line) for line in
                 (custom.results / "video-grounding.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(saved[0]["contentTextApprovalStatus"], "approved")
        self.assertEqual(saved[0]["contentTextSha256"], hashlib.sha256(CONTENT.encode()).hexdigest())
        provider = SimulatedApiFixture({"quiz": {"normalized": QUIZ, "responseBody": FIXTURE_RAW}})
        row = custom.run_quiz(VIDEO, "gemini-3.8-flash", 1, CONTENT, provider,
                              source_grounding_run_id=grounding["runId"])
        self.assertEqual((row["apiStatus"], row["sourceGroundingRunId"]),
                         ("success", grounding["runId"]))
        self.assertEqual(len(provider.calls), 1)
        self.assertTrue((custom.results / "quiz-generation.jsonl").is_file())
        self.assertFalse((self.results / "quiz-generation.jsonl").exists())

    def test_grounding_from_another_results_directory_is_rejected(self):
        def snapshot(directory):
            return {path: path.read_bytes() for path in directory.rglob("*") if path.is_file()}

        custom = PilotRunner(self.repository, self.results / "custom")
        default_source = self.approved_source()
        default_before = snapshot(self.results)
        provider = SimulatedApiFixture({"quiz": {"normalized": QUIZ, "responseBody": FIXTURE_RAW}})
        with self.assertRaisesRegex(ValueError, "approved"):
            custom.run_quiz(VIDEO, "gemini-3.8-flash", 1, CONTENT, provider,
                            source_grounding_run_id=default_source)
        with self.assertRaisesRegex(ValueError, "human approval"):
            custom.approve_content(default_source)
        self.assertEqual(snapshot(self.results), default_before)

        grounding = custom.run_grounding(VIDEO, "gemini_video", 1,
                                         SimulatedApiFixture({"grounding": {
                                             "normalized": GROUNDING, "responseBody": FIXTURE_RAW}}))
        custom.approve_content(grounding["runId"])
        custom_before = snapshot(custom.results)
        with self.assertRaisesRegex(ValueError, "human approval"):
            self.runner.approve_content(grounding["runId"])
        with self.assertRaisesRegex(ValueError, "approved"):
            self.runner.run_quiz(VIDEO, "gemini-3.8-flash", 1, CONTENT, provider,
                                 source_grounding_run_id=grounding["runId"])
        self.assertEqual(snapshot(custom.results), custom_before)
        self.assertEqual(provider.calls, [])

    def test_unsafe_result_paths_are_rejected(self):
        for path in (self.repository / "data" / "output", Path("results/../data/output"),
                     Path("..") / "outside"):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "results directory"):
                PilotRunner(self.repository, path)

    def test_invalid_measurements_cannot_be_saved(self):
        class SimulatedApiProvider:
            is_actual_api = True

            def invoke(self, kind, **kwargs):
                return {"raw": QUIZ, "inputTokens": "10", "estimatedCostUsd": -1}

        row = self.run_approved_quiz(VIDEO, "gemini-3.8-flash", 1, CONTENT, SimulatedApiProvider())
        self.assertEqual(row["apiStatus"], "error")
        self.assertEqual(row["errorCategory"], "invalid_measurement")
        self.assertIsNone(row["inputTokens"])
        self.assertIsNone(row["estimatedCostUsd"])

    def test_all_measurement_types_and_ranges_are_checked(self):
        class SimulatedApiProvider:
            is_actual_api = True

            def __init__(self, measurement):
                self.measurement = measurement

            def invoke(self, kind, **kwargs):
                return {"raw": QUIZ, **self.measurement}

        for measurement in ({"latencyMs": -1}, {"inputTokens": False},
                            {"outputTokens": -3}, {"thinkingTokens": "2"},
                            {"estimatedCostUsd": "0.01"}, {"estimatedCostUsd": float("inf")}):
            with self.subTest(measurement=measurement):
                row = self.run_approved_quiz(VIDEO, "gemini-3.8-flash", 1, CONTENT,
                                           SimulatedApiProvider(measurement))
                self.assertEqual(row["errorCategory"], "invalid_measurement")
                self.assertIsNone(row["inputTokens"])
                self.assertIsNone(row["estimatedCostUsd"])

    def test_raw_response_with_authorization_header_is_not_saved(self):
        row = self.run_approved_quiz(VIDEO, "gemini-3.8-flash", 1, CONTENT,
                                   FixtureProvider({"quiz": {"raw": dict(QUIZ, Authorization="Bearer example")}}))
        self.assertNotIn("Bearer example", (self.results / "raw" / (row["runId"] + ".json")).read_text(encoding="utf-8"))

    def test_raw_metadata_rejects_freeform_provider_output(self):
        row = self.runner._base("quiz_generation", VIDEO, "fixed_content_text", "gemini-3.8-flash", 1)
        for unsafe in ({"status": "completed", "provider": "gemini", "usage": {
                "inputTokens": 1, "outputTokens": 1, "thinkingTokens": None, "toolUseTokens": None},
                "outputText": "Bearer unrelated-credential"},
                       {"source": "fixture", "Authorization": "Bearer unrelated-credential"}):
            with self.subTest(unsafe=unsafe), self.assertRaisesRegex(ValueError, "allowlisted"):
                self.runner._save(row, unsafe)
        self.assertFalse(self.results.exists())

    def test_concurrent_manifest_publication_is_complete_and_rejects_different_content(self):
        original_link = os.link
        barrier = Barrier(2)
        def synchronized_link(source, target):
            barrier.wait(timeout=5)
            return original_link(source, target)
        hashes = ["a" * 64, "b" * 64]
        with mock.patch("src.pilot_runner.os.link", side_effect=synchronized_link):
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(self.runner._require_fixed_content, VIDEO, item) for item in hashes]
                outcomes = []
                for future in futures:
                    try:
                        future.result()
                        outcomes.append("accepted")
                    except ValueError:
                        outcomes.append("mismatch")
        self.assertEqual(sorted(outcomes), ["accepted", "mismatch"])
        root = self.repository / "data" / "restricted" / "fixed-content"
        manifests = list(root.glob("*.json"))
        self.assertEqual(len(manifests), 1)
        self.assertIn(json.loads(manifests[0].read_text(encoding="utf-8"))["contentTextSha256"], hashes)

    def test_concurrent_manifest_publication_accepts_same_content(self):
        original_link = os.link
        barrier = Barrier(2)
        def synchronized_link(source, target):
            barrier.wait(timeout=5)
            return original_link(source, target)
        with mock.patch("src.pilot_runner.os.link", side_effect=synchronized_link):
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(self.runner._require_fixed_content, VIDEO, "a" * 64)
                           for _ in range(2)]
                for future in futures:
                    future.result()
        root = self.repository / "data" / "restricted" / "fixed-content"
        self.assertEqual(len(list(root.glob("*.json"))), 1)
        self.assertEqual(len(list(root.glob("*.tmp"))), 0)

    def test_evidence_substring_is_only_an_automatic_contract_check(self):
        provider = FixtureProvider({"quiz": {"raw": dict(QUIZ, questions=[
            dict(question, sourceEvidence="not in source") for question in QUIZ["questions"]
        ])}})
        row = self.run_approved_quiz(VIDEO, "gemini-3.8-flash", 1, CONTENT, provider)
        self.assertEqual(row["parseStatus"], "pass")
        self.assertEqual(row["validatorStatus"], "fail")
        self.assertEqual(row["beCompatibility"], "fail")
        self.assertIsNone(row["questionReviews"][0]["evidenceSupportsAnswer"])

    def test_duplicate_options_fail_validator_without_human_quality_claim(self):
        questions = [dict(item) for item in QUIZ["questions"]]
        questions[0]["options"] = ["A", "a ", "C", "D"]
        row = self.run_approved_quiz(VIDEO, "gemini-3.8-flash", 1, CONTENT,
                                   FixtureProvider({"quiz": {"raw": dict(QUIZ, questions=questions)}}))
        self.assertEqual(row["validatorStatus"], "fail")
        self.assertIsNone(row["questionReviews"][0]["uniqueAnswer"])

    def test_direct_quiz_never_claims_be_compatibility(self):
        provider = FixtureProvider({"direct": {"raw": QUIZ}})
        row = self.runner.run_end_to_end(VIDEO, "gemini_direct_quiz", 1, provider)
        self.assertEqual(row["beCompatibility"], "not_applicable")
        self.assertEqual(row["validatorStatus"], "not_run")
        self.assertIsNone(row["groundingRunId"])
        self.assertIsNone(row["quizRunId"])
        self.assertIsNone(row["questionReviews"][0]["answerAccuracy"])

    def test_fixture_rows_always_record_null_retry_stop_reason(self):
        success = FixtureProvider({"grounding": {"raw": GROUNDING}, "quiz": {"raw": QUIZ},
                                   "direct": {"raw": QUIZ}})
        failing = FixtureProvider({"grounding": {"errorCategory": "rate_limit"},
                                   "direct": {"errorCategory": "server_error"}})
        rows = [self.runner.run_grounding(VIDEO, "gemini_video", 1, success),
                self.run_approved_quiz(VIDEO, "gemini-3.8-flash", 1, CONTENT, success),
                self.runner.run_end_to_end(VIDEO, "gemini_direct_quiz", 1, success),
                self.runner.run_grounding(VIDEO, "gemini_video", 2, failing),
                self.runner.run_end_to_end(VIDEO, "gemini_direct_quiz", 2, failing)]
        for row in rows:
            with self.subTest(benchmarkType=row["benchmarkType"], apiStatus=row["apiStatus"]):
                self.assertIn("retryStopReason", row)
                self.assertIsNone(row["retryStopReason"])

    def test_retry_stop_reason_schema_allows_only_null_or_live_guard(self):
        schema = json.loads((ROOT / "docs" / "run-result.schema.json").read_text(encoding="utf-8"))
        self.assertEqual(schema["properties"]["retryStopReason"], {"enum": ["live_guard", None]})
        self.assertNotIn("retryStopReason", schema["required"])
        self.runner.run_grounding(VIDEO, "gemini_video", 1,
                                  FixtureProvider({"grounding": {"raw": GROUNDING}}))
        for row in self.rows("video-grounding.jsonl"):
            self.assertIn(row["retryStopReason"], schema["properties"]["retryStopReason"]["enum"])
            legacy = {key: value for key, value in row.items() if key != "retryStopReason"}
            self.assertTrue(set(schema["required"]).issubset(legacy))

    def test_failed_call_and_blocked_two_stage_e2e_are_not_falsely_successful(self):
        provider = FixtureProvider({"grounding": {"errorCategory": "rate_limit"}})
        failed = self.runner.run_grounding(VIDEO, "gemini_video", 1, provider)
        self.assertEqual(failed["apiStatus"], "not_run")
        self.assertEqual(failed["errorCategory"], "fixture_rate_limit")
        with self.assertRaisesRegex(ValueError, "approval"):
            self.runner.run_end_to_end(VIDEO, "transcript_gemini_quiz", 1,
                                       FixtureProvider({}), authorized_transcript=None)
        self.assertEqual(len(provider.calls), 1)

    def test_result_rows_follow_schema_top_level_contract(self):
        provider = FixtureProvider({"grounding": {"raw": GROUNDING}, "quiz": {"raw": QUIZ},
                                    "direct": {"raw": QUIZ}})
        self.runner.run_grounding(VIDEO, "gemini_video", 1, provider)
        self.run_approved_quiz(VIDEO, "gemini-3.8-flash", 1, CONTENT, provider)
        self.runner.run_end_to_end(VIDEO, "gemini_direct_quiz", 1, provider)
        schema = json.loads((ROOT / "docs" / "run-result.schema.json").read_text(encoding="utf-8"))
        self.assertEqual(schema["properties"]["contentTextApprovalStatus"]["enum"],
                         ["approved", None])
        self.assertEqual(schema["oneOf"][0]["then"]["required"], ["contentTextSha256"])
        for filename in ("video-grounding.jsonl", "quiz-generation.jsonl", "end-to-end.jsonl"):
            for row in self.rows(filename):
                self.assertTrue(set(schema["required"]).issubset(row))
                self.assertTrue(set(row).issubset(schema["properties"]))
                self.assertIn(row["benchmarkType"], schema["properties"]["benchmarkType"]["enum"])
                self.assertIn(row["apiStatus"], schema["properties"]["apiStatus"]["enum"])


if __name__ == "__main__":
    unittest.main()
