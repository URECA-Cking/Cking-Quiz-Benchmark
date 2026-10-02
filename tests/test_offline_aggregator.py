"""Read-only, temporary-file tests for the offline pipeline view."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.offline_aggregator import aggregate_pipeline, compare_pair


CONTENT = "Human approved content."
SHA = hashlib.sha256(CONTENT.encode("utf-8")).hexdigest()


class OfflineAggregatorTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.results = Path(self.temp.name) / "results"
        (self.results / "evaluation").mkdir(parents=True)
        self.grounding = self.row("video_grounding", "ground-5", "gemini_video",
                                  "gemini-3.8-flash", 5, latencyMs=10,
                                  inputTokens=100, outputTokens=20,
                                  estimatedCostUsd=0.02, pricingReference="gemini-price",
                                  contentTextSha256=SHA, contentTextApprovalStatus="approved",
                                  groundingFacts=[{"factExists": "pass"}],
                                  omission="fail", hallucination="pass")
        self.quiz = self.row("quiz_generation", "quiz-a", "fixed_content_text",
                             "gemini-3.8-flash", 1, latencyMs=4,
                             inputTokens=5, outputTokens=9,
                             estimatedCostUsd=0.003, pricingReference="quiz-price",
                             contentTextSha256=SHA, sourceGroundingRunId="ground-5",
                             promptVersion="pilot-v1", questionReviews=[{"answerAccuracy": "pass"}])
        self.other_quiz = self.row("quiz_generation", "quiz-b", "fixed_content_text",
                                   "gpt-5.4-mini", 1, contentTextSha256=SHA,
                                   sourceGroundingRunId="ground-5", promptVersion="pilot-v1",
                                   questionReviews=[])
        self.write([self.row("video_grounding", "ground-1", "gemini_video",
                             "gemini-3.8-flash", 1, apiStatus="error", errorCategory="server_error",
                             httpStatus=503), self.grounding], [self.quiz, self.other_quiz])
        (self.results / "evaluation" / "ground-5.json").write_text(
            json.dumps({"contentText": CONTENT}), encoding="utf-8")

    @staticmethod
    def row(kind, run_id, method, model, attempt, **extra):
        return {"benchmarkType": kind, "runId": run_id, "videoId": "video-1",
                "method": method, "model": model, "repetition": 1, "attempt": attempt,
                "apiStatus": "success", "latencyMs": None, "inputTokens": None,
                "outputTokens": None, "thinkingTokens": None,
                "estimatedCostUsd": None, "pricingReference": None, **extra}

    def write(self, grounding, quiz):
        for name, rows in (("video-grounding.jsonl", grounding),
                           ("quiz-generation.jsonl", quiz)):
            (self.results / name).write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def aggregate(self, quiz_id="quiz-a"):
        return aggregate_pipeline(self.results, "ground-5", quiz_id)

    def test_aggregates_selected_runs_without_persisting_or_combining_usage(self):
        before = {p: p.read_bytes() for p in self.results.rglob("*") if p.is_file()}
        result = self.aggregate()
        self.assertEqual((result["groundingRunId"], result["quizRunId"]), ("ground-5", "quiz-a"))
        self.assertEqual(result["pipelineApiStatus"], "success")
        self.assertEqual(result["apiLatencySumMs"], 14)
        self.assertEqual(result["selectedPipelineEstimatedCostUsd"], 0.023)
        self.assertEqual(result["groundingUsage"], {"inputTokens": 100, "outputTokens": 20,
                                                     "thinkingTokens": None})
        self.assertEqual(result["quizUsage"], {"inputTokens": 5, "outputTokens": 9,
                                                "thinkingTokens": None})
        self.assertNotIn("totalTokens", result)
        self.assertNotIn("overallQuality", result)
        self.assertEqual(result["groundingHumanEvaluation"]["omission"], "fail")
        self.assertEqual(result["quizHumanEvaluation"], [{"answerAccuracy": "pass"}])
        self.assertEqual(result["groundingAttemptHistory"][0]["apiStatus"], "error")
        self.assertEqual(result["groundingAttemptHistory"][0]["httpStatus"], 503)
        self.assertEqual(result["selectedGroundingAttempt"], 5)
        self.assertEqual(result["groundingPricingReference"], "gemini-price")
        self.assertEqual(result["quizPricingReference"], "quiz-price")
        self.assertEqual(result["contentTextApprovalStatus"], "approved")
        self.assertEqual(before, {p: p.read_bytes() for p in self.results.rglob("*") if p.is_file()})

    def test_rejects_ambiguous_or_missing_run_ids_and_wrong_types(self):
        with self.assertRaisesRegex(ValueError, "Quiz runId"):
            self.aggregate("absent")
        self.write([self.grounding, self.grounding], [self.quiz])
        with self.assertRaisesRegex(ValueError, "Grounding runId"):
            self.aggregate()
        self.write([self.quiz], [self.grounding])
        with self.assertRaises(ValueError):
            self.aggregate()

    def test_rejects_mismatched_link_video_repetition_and_hash(self):
        for field, value in (("sourceGroundingRunId", "wrong"),
                             ("videoId", "other"), ("repetition", 2),
                             ("contentTextSha256", "a" * 64)):
            with self.subTest(field=field):
                changed = {**self.quiz, field: value}
                self.write([self.grounding], [changed])
                with self.assertRaises(ValueError):
                    self.aggregate()

    def test_rejects_unapproved_failed_missing_prompt_and_invalid_sha(self):
        for g_field, g_value, q_field, q_value in (
                ("contentTextApprovalStatus", None, None, None),
                ("apiStatus", "error", None, None),
                (None, None, "apiStatus", "error"),
                (None, None, "promptVersion", None),
                ("contentTextSha256", "invalid", None, None)):
            with self.subTest(g_field=g_field, q_field=q_field):
                g, q = dict(self.grounding), dict(self.quiz)
                if g_field:
                    g[g_field] = g_value
                if q_field:
                    q[q_field] = q_value
                self.write([g], [q])
                with self.assertRaises(ValueError):
                    self.aggregate()

    def test_rejects_evaluation_content_hash_mismatch(self):
        (self.results / "evaluation" / "ground-5.json").write_text(
            json.dumps({"contentText": "altered"}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "evaluation"):
            self.aggregate()

    def test_rejects_missing_evaluation_without_writing_results(self):
        (self.results / "evaluation" / "ground-5.json").unlink()
        before = {p: p.read_bytes() for p in self.results.rglob("*") if p.is_file()}
        with self.assertRaisesRegex(ValueError, "evaluation"):
            self.aggregate()
        self.assertEqual(before, {p: p.read_bytes() for p in self.results.rglob("*") if p.is_file()})

    def test_absent_metrics_stay_unknown(self):
        g = {**self.grounding, "latencyMs": None, "estimatedCostUsd": None,
             "inputTokens": None}
        self.write([g], [self.quiz])
        result = self.aggregate()
        self.assertIsNone(result["apiLatencySumMs"])
        self.assertIsNone(result["selectedPipelineEstimatedCostUsd"])
        self.assertIsNone(result["groundingUsage"]["inputTokens"])

    def test_attempt_history_uses_exact_condition_key_and_legacy_attempt(self):
        old = self.row("video_grounding", "legacy", "gemini_video", "gemini-3.8-flash", 1,
                       apiStatus="error")
        old.pop("attempt")
        other = self.row("video_grounding", "other-model", "gemini_video", "other", 1)
        self.write([old, other, self.grounding], [self.quiz])
        result = self.aggregate()
        self.assertEqual([item["runId"] for item in result["groundingAttemptHistory"]],
                         ["legacy", "ground-5"])
        self.assertEqual(result["groundingAttemptHistory"][0]["attempt"], 1)

    def test_attempt_history_exposes_retry_stop_reason_and_accepts_legacy_rows(self):
        legacy = self.row("video_grounding", "ground-1", "gemini_video", "gemini-3.8-flash", 1,
                          apiStatus="error", errorCategory="server_error", httpStatus=503)
        stopped = self.row("video_grounding", "ground-2", "gemini_video", "gemini-3.8-flash", 2,
                           apiStatus="error", errorCategory="rate_limit", httpStatus=429,
                           providerErrorCode="rate_limit_exceeded", retryStopReason="live_guard")
        self.write([legacy, stopped, {**self.grounding, "retryStopReason": None}], [self.quiz])
        history = self.aggregate()["groundingAttemptHistory"]
        self.assertEqual([item["runId"] for item in history], ["ground-1", "ground-2", "ground-5"])
        self.assertNotIn("retryStopReason", history[0])
        self.assertEqual((history[1]["errorCategory"], history[1]["httpStatus"],
                          history[1]["providerErrorCode"], history[1]["retryStopReason"]),
                         ("rate_limit", 429, "rate_limit_exceeded", "live_guard"))
        self.assertIsNone(history[2]["retryStopReason"])

    def test_quiz_failure_history_excludes_other_prompt_and_content(self):
        prior = {**self.quiz, "runId": "quiz-failed", "attempt": 1,
                 "apiStatus": "error", "errorCategory": "rate_limit"}
        selected = {**self.quiz, "attempt": 2}
        other_prompt = {**prior, "runId": "quiz-other-prompt", "promptVersion": "pilot-v2"}
        other_content = {**prior, "runId": "quiz-other-content", "contentTextSha256": "a" * 64}
        self.write([self.grounding], [prior, selected, other_prompt, other_content])
        result = self.aggregate()
        self.assertEqual([item["runId"] for item in result["quizAttemptHistory"]],
                         ["quiz-failed", "quiz-a"])
        self.assertEqual(result["quizAttemptHistory"][0]["errorCategory"], "rate_limit")

    def test_pair_requires_same_source_video_repetition_sha_and_prompt(self):
        valid = compare_pair(self.results, "quiz-a", "quiz-b")
        self.assertEqual(valid["a"]["groundingRunId"], valid["b"]["groundingRunId"])
        for field, value in (("sourceGroundingRunId", "other"), ("videoId", "other"),
                             ("repetition", 2), ("contentTextSha256", "b" * 64),
                             ("promptVersion", "pilot-v2")):
            with self.subTest(field=field):
                self.write([self.grounding], [self.quiz, {**self.other_quiz, field: value}])
                with self.assertRaises(ValueError):
                    compare_pair(self.results, "quiz-a", "quiz-b")


if __name__ == "__main__":
    unittest.main()
