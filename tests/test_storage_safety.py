import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.pilot_runner import FixtureProvider, PilotRunner


ROOT = Path(__file__).resolve().parents[1]
VIDEO = "nasa-water-cycle-2019"
GROUNDING = {"contentText": "Water cycle overview", "facts": []}


class StorageSafetyTest(unittest.TestCase):
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

    def grounding(self, provider=None):
        provider = provider or FixtureProvider({"grounding": {"raw": GROUNDING}})
        return self.runner.run_grounding(VIDEO, "gemini_video", 1, provider)

    def test_raw_replace_failure_keeps_existing_target(self):
        row = self.runner._base("video_grounding", VIDEO, "gemini_video", "gemini-3.8-flash", 1)
        target = self.results / "raw" / (row["runId"] + ".json")
        target.parent.mkdir(parents=True)
        target.write_text('{"old":true}', encoding="utf-8")
        with mock.patch("src.pilot_runner.os.replace", side_effect=OSError("replace blocked")):
            with self.assertRaises(OSError):
                self.runner._save(row, raw={"source": "fixture"})
        self.assertEqual(target.read_text(encoding="utf-8"), '{"old":true}')
        self.assertFalse((self.results / "video-grounding.jsonl").exists())

    def test_evaluation_replace_failure_keeps_existing_target(self):
        row = self.runner._base("video_grounding", VIDEO, "gemini_video", "gemini-3.8-flash", 1)
        target = self.results / "evaluation" / (row["runId"] + ".json")
        target.parent.mkdir(parents=True)
        target.write_text('{"old":true}', encoding="utf-8")
        original_replace = __import__("os").replace

        def fail_evaluation(source, destination):
            if Path(destination) == target:
                raise OSError("replace blocked")
            return original_replace(source, destination)

        with mock.patch("src.pilot_runner.os.replace", side_effect=fail_evaluation):
            with self.assertRaises(OSError):
                self.runner._save(row, raw={"source": "fixture"}, evaluation=GROUNDING)
        self.assertEqual(target.read_text(encoding="utf-8"), '{"old":true}')
        self.assertFalse((self.results / "video-grounding.jsonl").exists())

    def test_malformed_summary_blocks_before_provider(self):
        self.results.mkdir()
        (self.results / "video-grounding.jsonl").write_text('{"runId":', encoding="utf-8")
        provider = FixtureProvider({"grounding": {"raw": GROUNDING}})
        with self.assertRaisesRegex(ValueError, "storage integrity"):
            self.grounding(provider)
        self.assertEqual(provider.calls, [])

    def test_incomplete_summary_structure_blocks_before_provider(self):
        self.results.mkdir()
        (self.results / "video-grounding.jsonl").write_text('{}\n', encoding="utf-8")
        provider = FixtureProvider({"grounding": {"raw": GROUNDING}})
        with self.assertRaisesRegex(ValueError, "storage integrity"):
            self.grounding(provider)
        self.assertEqual(provider.calls, [])

    def test_nonfinite_json_constant_blocks_before_provider(self):
        self.results.mkdir()
        row = {"runId": "legacy-one", "videoId": VIDEO, "method": "gemini_video",
               "model": "gemini-3.8-flash", "repetition": 1, "apiStatus": "error"}
        (self.results / "video-grounding.jsonl").write_text(
            json.dumps(row)[:-1] + ',"latencyMs":NaN}\n', encoding="utf-8")
        provider = FixtureProvider({"grounding": {"raw": GROUNDING}})
        with self.assertRaisesRegex(ValueError, "storage integrity"):
            self.grounding(provider)
        self.assertEqual(provider.calls, [])

    def test_legacy_error_row_without_attempt_remains_readable(self):
        self.results.mkdir()
        legacy = {"runId": "legacy-one", "videoId": VIDEO, "method": "gemini_video",
                  "model": "gemini-3.8-flash", "repetition": 1, "apiStatus": "error"}
        target = self.results / "video-grounding.jsonl"
        original = json.dumps(legacy) + "\n"
        target.write_text(original, encoding="utf-8")
        row = self.grounding()
        self.assertEqual(row["attempt"], 2)
        self.assertTrue(target.read_text(encoding="utf-8").startswith(original))

    def test_missing_success_payload_blocks_before_provider(self):
        first = self.grounding()
        (self.results / "evaluation" / (first["runId"] + ".json")).unlink()
        provider = FixtureProvider({"grounding": {"raw": GROUNDING}})
        with self.assertRaisesRegex(ValueError, "storage integrity"):
            self.grounding(provider)
        self.assertEqual(provider.calls, [])

    def test_missing_success_raw_blocks_before_provider(self):
        first = self.grounding()
        (self.results / "raw" / (first["runId"] + ".json")).unlink()
        provider = FixtureProvider({"grounding": {"raw": GROUNDING}})
        with self.assertRaisesRegex(ValueError, "storage integrity"):
            self.grounding(provider)
        self.assertEqual(provider.calls, [])

    def test_malformed_linked_payload_blocks_before_provider(self):
        first = self.grounding()
        (self.results / "raw" / (first["runId"] + ".json")).write_text('{"partial":', encoding="utf-8")
        provider = FixtureProvider({"grounding": {"raw": GROUNDING}})
        with self.assertRaisesRegex(ValueError, "storage integrity"):
            self.grounding(provider)
        self.assertEqual(provider.calls, [])

    def test_error_without_payload_is_not_mistaken_for_corruption(self):
        first = self.grounding(FixtureProvider({"grounding": {"errorCategory": "rate_limit"}}))
        self.assertEqual(first["apiStatus"], "not_run")
        self.assertEqual(self.grounding()["attempt"], 2)

    def test_provider_http_failure_row_without_payload_is_valid(self):
        row = self.runner._base("video_grounding", VIDEO, "gemini_video", "gemini-3.8-flash", 1)
        row.update(apiStatus="error", errorCategory="server_error", httpStatus=503,
                   groundingFacts=[], omission=None, hallucination=None)
        self.runner._save(row)
        self.assertEqual(self.grounding()["attempt"], 2)

    def test_invalid_measurement_summary_without_raw_is_valid(self):
        row = self.runner._base("video_grounding", VIDEO, "gemini_video", "gemini-3.8-flash", 1)
        row.update(apiStatus="error", errorCategory="invalid_measurement")
        self.runner._save(row)
        self.runner._check_storage_integrity()
        self.assertEqual(self.grounding()["attempt"], 2)

    def test_invalid_measurement_with_valid_raw_is_valid(self):
        row = self.runner._base("video_grounding", VIDEO, "gemini_video", "gemini-3.8-flash", 1)
        row.update(apiStatus="error", errorCategory="invalid_measurement")
        raw = {"status": "completed", "provider": "gemini", "usage": {
            "inputTokens": 10, "outputTokens": 20, "thinkingTokens": 0, "toolUseTokens": None}}
        self.runner._save(row, raw=raw)
        self.runner._check_storage_integrity()
        self.assertEqual(self.grounding()["attempt"], 2)

    def test_invalid_measurement_with_corrupt_raw_blocks_preflight(self):
        row = self.runner._base("video_grounding", VIDEO, "gemini_video", "gemini-3.8-flash", 1)
        row.update(apiStatus="error", errorCategory="invalid_measurement")
        self.runner._save(row, raw={"source": "fixture"})
        (self.results / "raw" / (row["runId"] + ".json")).write_text('{"partial":', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "malformed linked result payload"):
            self.runner._check_storage_integrity()

    def test_invalid_measurement_with_disallowed_raw_metadata_blocks_preflight(self):
        row = self.runner._base("video_grounding", VIDEO, "gemini_video", "gemini-3.8-flash", 1)
        row.update(apiStatus="error", errorCategory="invalid_measurement")
        self.runner._save(row, raw={"source": "fixture"})
        (self.results / "raw" / (row["runId"] + ".json")).write_text(
            json.dumps({"source": "fixture", "unexpected": "disallowed"}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "storage integrity"):
            self.runner._check_storage_integrity()

    def test_existing_raw_file_cannot_contain_json_null(self):
        row = self.runner._base("video_grounding", VIDEO, "gemini_video", "gemini-3.8-flash", 1)
        row.update(apiStatus="error", errorCategory="invalid_measurement")
        self.runner._save(row, raw={"source": "fixture"})
        (self.results / "raw" / (row["runId"] + ".json")).write_text("null", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "storage integrity"):
            self.runner._check_storage_integrity()

    def test_invalid_grounding_response_still_requires_raw(self):
        row = self.runner._base("video_grounding", VIDEO, "gemini_video", "gemini-3.8-flash", 1)
        row.update(apiStatus="error", errorCategory="invalid_grounding_response")
        self.runner._save(row)
        with self.assertRaisesRegex(ValueError, "missing linked result payload"):
            self.runner._check_storage_integrity()

    def test_orphaned_payload_from_interrupted_save_blocks_next_run(self):
        self.results.mkdir()
        raw = self.results / "raw"
        raw.mkdir()
        (raw / "orphan.json").write_text('{"source":"fixture"}', encoding="utf-8")
        provider = FixtureProvider({"grounding": {"raw": GROUNDING}})
        with self.assertRaisesRegex(ValueError, "storage integrity"):
            self.grounding(provider)
        self.assertEqual(provider.calls, [])

    def test_unfinished_temporary_file_blocks_next_run(self):
        self.results.mkdir()
        (self.results / ".result-leftover.tmp").write_text("partial", encoding="utf-8")
        provider = FixtureProvider({"grounding": {"raw": GROUNDING}})
        with self.assertRaisesRegex(ValueError, "storage integrity"):
            self.grounding(provider)
        self.assertEqual(provider.calls, [])

    def test_success_keeps_summary_raw_and_evaluation_linked(self):
        row = self.grounding()
        self.assertTrue((self.results / "raw" / (row["runId"] + ".json")).is_file())
        self.assertTrue((self.results / "evaluation" / (row["runId"] + ".json")).is_file())
        saved = json.loads((self.results / "video-grounding.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(saved["runId"], row["runId"])
        self.runner._check_storage_integrity()

    def test_summary_replace_failure_keeps_prior_complete_content(self):
        self.grounding(FixtureProvider({"grounding": {"errorCategory": "rate_limit"}}))
        target = self.results / "video-grounding.jsonl"
        original = target.read_bytes()
        original_replace = __import__("os").replace

        def fail_summary(source, destination):
            if Path(destination) == target:
                raise OSError("replace blocked")
            return original_replace(source, destination)

        with mock.patch("src.pilot_runner.os.replace", side_effect=fail_summary):
            with self.assertRaises(OSError):
                self.grounding()
        self.assertEqual(target.read_bytes(), original)
        provider = FixtureProvider({"grounding": {"raw": GROUNDING}})
        with self.assertRaisesRegex(ValueError, "storage integrity"):
            self.grounding(provider)
        self.assertEqual(provider.calls, [])


if __name__ == "__main__":
    unittest.main()
