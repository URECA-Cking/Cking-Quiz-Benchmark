import json
import shutil
import tempfile
import unittest
from pathlib import Path

from src.pilot_runner import FixtureProvider, PilotRunner


ROOT = Path(__file__).resolve().parents[1]
VIDEO = "nasa-water-cycle-2019"
CONTENT = "NASA measures global rain and snow every 30 minutes."
QUIZ = {
    "promptVersion": "quiz-mcq-v1",
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

    def test_grounding_keeps_human_reviews_unset_and_separates_raw(self):
        provider = FixtureProvider({"grounding": {"raw": GROUNDING, "inputTokens": 10}})
        row = self.runner.run_grounding(VIDEO, "gemini_video", 1, provider)
        self.assertEqual(row["apiStatus"], "not_run")
        self.assertIsNone(row["latencyMs"])
        self.assertIsNone(row["inputTokens"])
        self.assertIsNone(row["groundingFacts"][0]["factExists"])
        self.assertIsNone(row["groundingFacts"][0]["timestampAccurate"])
        self.assertIsNone(row["hallucination"])
        self.assertEqual(self.rows("video-grounding.jsonl")[0]["runId"], row["runId"])
        self.assertTrue((self.results / "raw" / (row["runId"] + ".json")).exists())
        self.assertNotIn("contentText", row)

    def test_two_models_use_exact_same_fixed_content_and_hash(self):
        provider = FixtureProvider({"quiz": {"raw": QUIZ}})
        gemini = self.runner.run_quiz(VIDEO, "gemini-3.8-flash", 1, CONTENT, provider)
        deepseek = self.runner.run_quiz(VIDEO, "deepseek-flash", 1, CONTENT, provider)
        self.assertEqual(provider.calls[0]["contentText"], provider.calls[1]["contentText"])
        self.assertEqual(gemini["contentTextSha256"], deepseek["contentTextSha256"])
        self.assertEqual(gemini["beCompatibility"], "pass")
        self.assertIsNone(gemini["questionReviews"][0]["answerAccuracy"])

    def test_fixed_content_cannot_differ_between_models(self):
        provider = FixtureProvider({"quiz": {"raw": QUIZ}})
        self.runner.run_quiz(VIDEO, "gemini-3.8-flash", 1, CONTENT, provider)
        with self.assertRaisesRegex(ValueError, "fixed contentText"):
            self.runner.run_quiz(VIDEO, "deepseek-flash", 1, CONTENT + " changed", provider)
        self.assertEqual(len(provider.calls), 1)

    def test_canonical_input_survives_different_result_directories(self):
        first = PilotRunner(self.repository, self.repository / "results" / "first")
        second = PilotRunner(self.repository, self.repository / "results" / "second")
        first.run_quiz(VIDEO, "gemini-3.8-flash", 1, CONTENT, FixtureProvider({"quiz": {"raw": QUIZ}}))
        provider = FixtureProvider({"quiz": {"raw": QUIZ}})
        with self.assertRaisesRegex(ValueError, "fixed contentText"):
            second.run_quiz(VIDEO, "deepseek-flash", 1, CONTENT + " changed", provider)
        self.assertFalse(provider.calls)

    def test_canonical_input_is_scoped_to_video_and_prompt_version(self):
        provider = FixtureProvider({"quiz": {"raw": QUIZ}})
        self.runner.run_quiz(VIDEO, "gemini-3.8-flash", 1, CONTENT, provider)
        other_video = "nasa-methane-2020"
        self.runner.run_quiz(other_video, "deepseek-flash", 1, CONTENT + " other", provider)
        config = self.repository / "configs" / "pilot.yaml"
        config.write_text(config.read_text(encoding="utf-8").replace("pilot-v1", "pilot-v2"), encoding="utf-8")
        next_version = PilotRunner(self.repository, self.repository / "results" / "next")
        next_version.run_quiz(VIDEO, "deepseek-flash", 1, CONTENT + " changed", provider)
        self.assertEqual(len(provider.calls), 3)

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

        row = self.runner.run_quiz(VIDEO, "gemini-3.8-flash", 1, CONTENT, SimulatedApiProvider())
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
                row = self.runner.run_quiz(VIDEO, "gemini-3.8-flash", 1, CONTENT,
                                           SimulatedApiProvider(measurement))
                self.assertEqual(row["errorCategory"], "invalid_measurement")
                self.assertIsNone(row["inputTokens"])
                self.assertIsNone(row["estimatedCostUsd"])

    def test_raw_response_with_authorization_header_is_not_saved(self):
        with self.assertRaisesRegex(ValueError, "secret"):
            self.runner.run_quiz(VIDEO, "gemini-3.8-flash", 1, CONTENT,
                                 FixtureProvider({"quiz": {"raw": dict(QUIZ, Authorization="Bearer example")}}))
        self.assertFalse((self.results / "quiz-generation.jsonl").exists())

    def test_evidence_substring_is_only_an_automatic_contract_check(self):
        provider = FixtureProvider({"quiz": {"raw": dict(QUIZ, questions=[
            dict(question, sourceEvidence="not in source") for question in QUIZ["questions"]
        ])}})
        row = self.runner.run_quiz(VIDEO, "gemini-3.8-flash", 1, CONTENT, provider)
        self.assertEqual(row["parseStatus"], "pass")
        self.assertEqual(row["validatorStatus"], "fail")
        self.assertEqual(row["beCompatibility"], "fail")
        self.assertIsNone(row["questionReviews"][0]["evidenceSupportsAnswer"])

    def test_duplicate_options_fail_validator_without_human_quality_claim(self):
        questions = [dict(item) for item in QUIZ["questions"]]
        questions[0]["options"] = ["A", "a ", "C", "D"]
        row = self.runner.run_quiz(VIDEO, "gemini-3.8-flash", 1, CONTENT,
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

    def test_two_stage_results_are_linked_and_separate(self):
        provider = FixtureProvider({"grounding": {"raw": GROUNDING}, "quiz": {"raw": QUIZ}})
        row = self.runner.run_end_to_end(VIDEO, "gemini_grounding_deepseek_quiz", 1, provider)
        grounding = self.rows("video-grounding.jsonl")[0]
        quiz = self.rows("quiz-generation.jsonl")[0]
        self.assertEqual(row["groundingRunId"], grounding["runId"])
        self.assertEqual(row["quizRunId"], quiz["runId"])
        self.assertEqual(quiz["sourceGroundingRunId"], grounding["runId"])
        self.assertEqual(provider.calls[1]["contentText"], CONTENT)

    def test_failed_call_and_missing_transcript_are_recorded_without_fake_success(self):
        provider = FixtureProvider({"grounding": {"errorCategory": "rate_limit"}})
        failed = self.runner.run_grounding(VIDEO, "gemini_video", 1, provider)
        self.assertEqual(failed["apiStatus"], "not_run")
        self.assertEqual(failed["errorCategory"], "fixture_rate_limit")
        skipped = self.runner.run_end_to_end(VIDEO, "transcript_gemini_quiz", 1,
                                             FixtureProvider({}), authorized_transcript=None)
        self.assertEqual(skipped["apiStatus"], "not_run")
        self.assertEqual(skipped["beCompatibility"], "not_run")
        self.assertEqual(len(provider.calls), 1)

    def test_result_rows_follow_schema_top_level_contract(self):
        provider = FixtureProvider({"grounding": {"raw": GROUNDING}, "quiz": {"raw": QUIZ},
                                    "direct": {"raw": QUIZ}})
        self.runner.run_grounding(VIDEO, "gemini_video", 1, provider)
        self.runner.run_quiz(VIDEO, "gemini-3.8-flash", 1, CONTENT, provider)
        self.runner.run_end_to_end(VIDEO, "gemini_direct_quiz", 1, provider)
        schema = json.loads((ROOT / "docs" / "run-result.schema.json").read_text(encoding="utf-8"))
        for filename in ("video-grounding.jsonl", "quiz-generation.jsonl", "end-to-end.jsonl"):
            for row in self.rows(filename):
                self.assertTrue(set(schema["required"]).issubset(row))
                self.assertTrue(set(row).issubset(schema["properties"]))
                self.assertIn(row["benchmarkType"], schema["properties"]["benchmarkType"]["enum"])
                self.assertIn(row["apiStatus"], schema["properties"]["apiStatus"]["enum"])


if __name__ == "__main__":
    unittest.main()
