import errno
import json
import hashlib
import math
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Barrier
from unittest import mock

from src import approval_tracking, grounding_review
from src.pilot_runner import FixtureProvider, PilotRunner


ROOT = Path(__file__).resolve().parents[1]
VIDEO = "nasa-water-cycle-2019"
CONTENT = "NASA measures global rain and snow every 30 minutes."
APPROVER = "pilot-reviewer"
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
        self.runner.approve_content(source_id, APPROVER)
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
            self.runner.approve_content(source_id, APPROVER)
        self.assertEqual(result_file.read_bytes(), before)

    def update_grounding(self, run_id, remove=(), **fields):
        result_file = self.results / "video-grounding.jsonl"
        rows = self.rows("video-grounding.jsonl")
        for row in rows:
            if row["runId"] == run_id:
                for key in remove:
                    row.pop(key, None)
                row.update(fields)
        result_file.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def test_approval_records_trimmed_approver_and_utc_time(self):
        source_id = self.approved_source(approval=None)
        before = self.rows("video-grounding.jsonl")[0]
        started = datetime.now(timezone.utc)
        returned = self.runner.approve_content(source_id, "  Reviewer Kim  ")
        after = self.rows("video-grounding.jsonl")[0]
        self.assertEqual(returned, after)
        self.assertEqual(after["approvedBy"], "Reviewer Kim")
        approved_at = datetime.fromisoformat(after["approvedAt"])
        self.assertEqual(approved_at.utcoffset(), timedelta(0))
        self.assertTrue(started <= approved_at <= datetime.now(timezone.utc))
        self.assertEqual(after["approvedAt"][-6:], before["startedAt"][-6:])
        self.assertIsNotNone(approval_tracking.APPROVED_AT_PATTERN.fullmatch(after["approvedAt"]))
        for key in before:
            if key != "contentTextApprovalStatus":
                self.assertEqual(after[key], before[key])
        provider = FixtureProvider({"quiz": {"raw": QUIZ}})
        self.runner.run_quiz(VIDEO, "gemini-3.8-flash", 1, CONTENT, provider,
                             source_grounding_run_id=source_id)
        self.assertEqual(len(provider.calls), 1)

    def write_grounding_lines(self, lines_and_endings):
        (self.results / "video-grounding.jsonl").write_bytes(b"".join(
            line + ending for line, ending in lines_and_endings))

    def grounding_lines(self):
        return (self.results / "video-grounding.jsonl").read_bytes().splitlines(keepends=True)

    def test_approval_keeps_other_rows_and_target_newlines(self):
        for other_ending, target_ending in ((b"\r\n", b"\n"), (b"\n", b"\r\n")):
            with self.subTest(other=other_ending, target=target_ending):
                shutil.rmtree(self.results, ignore_errors=True)
                self.approved_source(approval=None)
                target_id = self.approved_source(approval=None)
                other, target = (line.rstrip(b"\r\n") for line in self.grounding_lines())
                self.write_grounding_lines([(other, other_ending), (target, target_ending)])
                before = json.loads(target)
                self.runner.approve_content(target_id, APPROVER)
                after_other, after_target = self.grounding_lines()
                self.assertEqual(after_other, other + other_ending)
                self.assertTrue(after_target.endswith(target_ending))
                self.assertEqual(after_target[-len(target_ending) - 1:-len(target_ending)], b"}")
                after = json.loads(after_target)
                self.assertEqual(after["runId"], target_id)
                self.assertEqual(after["contentTextApprovalStatus"], "approved")
                self.assertEqual(after["approvedBy"], APPROVER)
                self.assertEqual({key: after[key] for key in before
                                  if key != "contentTextApprovalStatus"},
                                 {key: before[key] for key in before
                                  if key != "contentTextApprovalStatus"})
                self.assertEqual(set(after) - set(before), {"approvedBy", "approvedAt"})

    def test_approved_by_length_allows_100_code_points_after_strip(self):
        source_id = self.approved_source(approval=None)
        self.runner.approve_content(source_id, "  " + "가" * 100 + "  ")
        self.assertEqual(self.rows("video-grounding.jsonl")[0]["approvedBy"], "가" * 100)

    def test_invalid_approved_by_is_rejected_without_rewriting_result(self):
        source_id = self.approved_source(approval=None)
        result_file = self.results / "video-grounding.jsonl"
        before = result_file.read_bytes()
        for value in (None, 123, b"reviewer", "", "   ", "a\nb", "reviewer\n", "\treviewer",
                      "a\x00b", "a b", "a b", "a​b", "a‮b", "﻿reviewer",
                      "a\ud800b",
                      "x" * 101, "  " + "x" * 101 + "  "):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "approved_by"):
                    self.runner.approve_content(source_id, value)
                self.assertEqual(result_file.read_bytes(), before)

    def test_reapproval_of_tracked_row_is_quiet_noop(self):
        source_id = self.approved_source(approval=None)
        self.runner.approve_content(source_id, "first reviewer")
        result_file = self.results / "video-grounding.jsonl"
        before = result_file.read_bytes()
        self.runner.approve_content(source_id, "second reviewer")
        self.assertEqual(result_file.read_bytes(), before)
        self.assertEqual(self.rows("video-grounding.jsonl")[0]["approvedBy"], "first reviewer")

    def test_reapproval_rejects_invalid_approved_by_without_changing_tracking(self):
        source_id = self.approved_source(approval=None)
        self.runner.approve_content(source_id, "first reviewer")
        tracked = self.rows("video-grounding.jsonl")[0]
        result_file = self.results / "video-grounding.jsonl"
        before = result_file.read_bytes()
        for value in ("", "bad\nreviewer"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "approved_by"):
                    self.runner.approve_content(source_id, value)
                self.assertEqual(result_file.read_bytes(), before)
        after = self.rows("video-grounding.jsonl")[0]
        self.assertEqual((after["approvedBy"], after["approvedAt"]),
                         (tracked["approvedBy"], tracked["approvedAt"]))

    def test_reapproval_of_approved_row_still_checks_evaluation_hash(self):
        tracking = {"approvedBy": APPROVER, "approvedAt": datetime.now(timezone.utc).isoformat()}
        for fields in ({}, tracking):
            with self.subTest(tracked=bool(fields)):
                source_id = self.approved_source(evaluation_content=CONTENT + " changed")
                self.update_grounding(source_id, **fields)
                result_file = self.results / "video-grounding.jsonl"
                before = result_file.read_bytes()
                with self.assertRaisesRegex(ValueError, "hash does not match"):
                    self.runner.approve_content(source_id, APPROVER)
                self.assertEqual(result_file.read_bytes(), before)
                shutil.rmtree(self.results)

    def test_approved_at_accepts_only_canonical_utc_isoformat(self):
        cases = (("2026-10-02T05:00:00+00:00", True), ("2026-10-02T05:00:00.123456+00:00", True),
                 ("2026-10-02T05:00:00Z", False), ("2026-10-02 05:00:00+00:00", False),
                 ("2026-10-02T05:00+00:00", False), ("2026-10-02T05:00:00-00:00", False),
                 ("20261002T050000+0000", False), ("2026-10-02T05:00:00.123+00:00", False),
                 ("2026-02-30T05:00:00+00:00", False), ("2026-10-02T24:00:00+00:00", False),
                 ("٢٠٢٦-10-02T05:00:00+00:00", False))
        for approved_at, valid in cases:
            with self.subTest(approved_at=approved_at):
                source_id = self.approved_source()
                self.update_grounding(source_id, approvedBy=APPROVER, approvedAt=approved_at)
                result_file = self.results / "video-grounding.jsonl"
                before = result_file.read_bytes()
                provider = FixtureProvider({"quiz": {"raw": QUIZ}})
                if valid:
                    self.runner.approve_content(source_id, APPROVER)
                    self.runner.run_quiz(VIDEO, "gemini-3.8-flash", 1, CONTENT, provider,
                                         source_grounding_run_id=source_id)
                    self.assertEqual(len(provider.calls), 1)
                else:
                    with self.assertRaisesRegex(ValueError, "approval tracking"):
                        self.runner.approve_content(source_id, APPROVER)
                    with self.assertRaisesRegex(ValueError, "approval tracking"):
                        self.runner.run_quiz(VIDEO, "gemini-3.8-flash", 1, CONTENT, provider,
                                             source_grounding_run_id=source_id)
                    self.assertEqual(provider.calls, [])
                self.assertEqual(result_file.read_bytes(), before)
                shutil.rmtree(self.results)

    def test_reapproval_of_legacy_approved_row_is_noop_without_backfill(self):
        source_id = self.approved_source()
        result_file = self.results / "video-grounding.jsonl"
        before = result_file.read_bytes()
        self.runner.approve_content(source_id, APPROVER)
        self.assertEqual(result_file.read_bytes(), before)
        row = self.rows("video-grounding.jsonl")[0]
        self.assertNotIn("approvedBy", row)
        self.assertNotIn("approvedAt", row)
        provider = FixtureProvider({"quiz": {"raw": QUIZ}})
        self.runner.run_quiz(VIDEO, "gemini-3.8-flash", 1, CONTENT, provider,
                             source_grounding_run_id=source_id)
        self.assertEqual(len(provider.calls), 1)

    def test_invalid_approval_tracking_fails_closed_for_approval_and_quiz(self):
        valid_at = datetime.now(timezone.utc).isoformat()
        cases = (
            ({"approvedBy": APPROVER}, ()),
            ({"approvedAt": valid_at}, ()),
            ({"approvedBy": APPROVER, "approvedAt": valid_at, "contentTextApprovalStatus": None}, ()),
            ({"approvedBy": APPROVER, "approvedAt": valid_at}, ("contentTextApprovalStatus",)),
        ) + tuple(({"approvedBy": value, "approvedAt": valid_at}, ())
                  for value in (None, 123, "", " padded ", "a\nb", "a​b", "a\ud800b",
                                "x" * 101)) \
          + tuple(({"approvedBy": APPROVER, "approvedAt": value}, ())
                  for value in (None, 123, "not-a-date", "2026-10-02T05:00:00",
                                "2026-10-02T14:00:00+09:00"))
        for fields, remove in cases:
            with self.subTest(fields=fields, remove=remove):
                source_id = self.approved_source()
                self.update_grounding(source_id, remove=remove, **fields)
                result_file = self.results / "video-grounding.jsonl"
                before = result_file.read_bytes()
                with self.assertRaisesRegex(ValueError, "approv"):
                    self.runner.approve_content(source_id, APPROVER)
                self.assertEqual(result_file.read_bytes(), before)
                provider = FixtureProvider({"quiz": {"raw": QUIZ}})
                with self.assertRaisesRegex(ValueError, "approv"):
                    self.runner.run_quiz(VIDEO, "gemini-3.8-flash", 1, CONTENT, provider,
                                         source_grounding_run_id=source_id)
                self.assertEqual(provider.calls, [])
                shutil.rmtree(self.results)

    def test_approval_is_independent_of_human_evaluation(self):
        for omission, hallucination in ((None, None), ("fail", "fail"), ("uncertain", None)):
            with self.subTest(omission=omission, hallucination=hallucination):
                source_id = self.approved_source(approval=None)
                self.update_grounding(source_id, omission=omission, hallucination=hallucination)
                self.runner.approve_content(source_id, APPROVER)
                row = [row for row in self.rows("video-grounding.jsonl")
                       if row["runId"] == source_id][0]
                self.assertEqual((row["contentTextApprovalStatus"], row["approvedBy"],
                                  row["omission"], row["hallucination"]),
                                 ("approved", APPROVER, omission, hallucination))

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

    def test_grounding_timestamp_order_and_recorded_duration_contract(self):
        # KARI has durationSeconds=227; the NASA water-cycle video has no recorded duration.
        cases = [("kari-microgravity-2024", 104.5, 104.5, True),
                 ("kari-microgravity-2024", 111.3, 104.5, False),
                 ("kari-microgravity-2024", 200, 227, True),
                 ("kari-microgravity-2024", 222.8, 229.8, False),
                 ("kari-microgravity-2024", 200, 227.1, False),
                 ("kari-microgravity-2024", None, 228, False),
                 ("kari-microgravity-2024", None, 50, True),
                 ("kari-microgravity-2024", 50, None, True),
                 (VIDEO, 100, 999, True),
                 (VIDEO, 999, None, True)]
        for video_id, start, end, accepted in cases:
            with self.subTest(video_id=video_id, start=start, end=end):
                fact = dict(GROUNDING["facts"][0], timestampStartSeconds=start, timestampEndSeconds=end)
                row = self.runner.run_grounding(
                    video_id, "gemini_video", 1,
                    FixtureProvider({"grounding": {"raw": dict(GROUNDING, facts=[fact])}}))
                if accepted:
                    self.assertIsNone(row["errorCategory"])
                    stored = row["groundingFacts"][0]
                    self.assertEqual((stored["timestampStartSeconds"], stored["timestampEndSeconds"]),
                                     (start, end))
                else:
                    self.assertEqual(row["errorCategory"], "fixture_invalid_grounding_response")
                    self.assertEqual(row["groundingFacts"], [])
                    self.assertNotIn("contentTextSha256", row)

    def run_timestamps(self, video_id, start, end, runner=None):
        fact = dict(GROUNDING["facts"][0], timestampStartSeconds=start, timestampEndSeconds=end)
        return (runner or self.runner).run_grounding(
            video_id, "gemini_video", 1,
            FixtureProvider({"grounding": {"raw": dict(GROUNDING, facts=[fact])}}))

    def assert_timestamp_rejected(self, row, start, end, runner=None):
        runner = runner or self.runner
        self.assertEqual(row["errorCategory"], "fixture_invalid_grounding_response")
        self.assertEqual(row["groundingFacts"], [])
        self.assertNotIn("contentTextSha256", row)
        self.assertEqual(self.rows("video-grounding.jsonl")[-1]["runId"], row["runId"])
        self.assertTrue((self.results / "raw" / (row["runId"] + ".json")).is_file())
        # A standard-JSON invalid response keeps its diagnostic payload; NaN/Infinity does not.
        standard_json = all(value is None or math.isfinite(value) for value in (start, end))
        self.assertEqual((self.results / "evaluation" / (row["runId"] + ".json")).is_file(),
                         standard_json)
        runner._check_storage_integrity()

    def test_grounding_rejects_non_finite_timestamps(self):
        for video_id in ("kari-microgravity-2024", VIDEO):
            for value in (float("nan"), float("inf"), float("-inf")):
                for start, end in ((value, None), (None, value), (value, 50), (10, value)):
                    with self.subTest(video_id=video_id, start=start, end=end):
                        self.assert_timestamp_rejected(self.run_timestamps(video_id, start, end),
                                                       start, end)
        # The same results directory stays usable after every non-finite rejection.
        self.assertIsNone(self.run_timestamps(VIDEO, 61, 73)["errorCategory"])

    def test_non_finite_actual_api_response_keeps_error_row_without_evaluation(self):
        fact = dict(GROUNDING["facts"][0], timestampStartSeconds=float("nan"))
        provider = SimulatedApiFixture({"grounding": {"normalized": dict(GROUNDING, facts=[fact]),
                                                      "responseBody": FIXTURE_RAW}})
        row = self.runner.run_grounding(VIDEO, "gemini_video", 1, provider)
        self.assertEqual((row["apiStatus"], row["errorCategory"], row["attempt"]),
                         ("error", "invalid_grounding_response", 1))
        self.assertTrue((self.results / "raw" / (row["runId"] + ".json")).is_file())
        self.assertFalse((self.results / "evaluation" / (row["runId"] + ".json")).exists())
        self.runner._check_storage_integrity()
        retry = self.runner.run_grounding(VIDEO, "gemini_video", 1, SimulatedApiFixture(
            {"grounding": {"normalized": GROUNDING, "responseBody": FIXTURE_RAW}}))
        self.assertEqual((retry["apiStatus"], retry["attempt"]), ("success", 2))

    def test_standard_json_invalid_response_keeps_evaluation_diagnostic(self):
        fact = dict(GROUNDING["facts"][0], timestampStartSeconds=80, timestampEndSeconds=70)
        provider = SimulatedApiFixture({"grounding": {"normalized": dict(GROUNDING, facts=[fact]),
                                                      "responseBody": FIXTURE_RAW}})
        row = self.runner.run_grounding(VIDEO, "gemini_video", 1, provider)
        self.assertEqual(row["errorCategory"], "invalid_grounding_response")
        evaluation = json.loads((self.results / "evaluation" / (row["runId"] + ".json"))
                                .read_text(encoding="utf-8"))
        self.assertEqual(evaluation["facts"][0]["timestampStartSeconds"], 80)
        self.runner._check_storage_integrity()

    def test_recorded_duration_bounds_start_and_end_independently(self):
        kari = "kari-microgravity-2024"
        cases = [(228, None, False), (227, None, True), (None, 228, False), (None, 227, True),
                 (227, 227, True), (228, 228, False), (100, 50, False), (100, 227, True)]
        for start, end, accepted in cases:
            with self.subTest(start=start, end=end):
                row = self.run_timestamps(kari, start, end)
                if accepted:
                    self.assertIsNone(row["errorCategory"])
                    stored = row["groundingFacts"][0]
                    self.assertEqual((stored["timestampStartSeconds"], stored["timestampEndSeconds"]),
                                     (start, end))
                else:
                    self.assert_timestamp_rejected(row, start, end)

    def test_unrecorded_duration_keeps_finite_and_non_negative_checks(self):
        kari = "kari-microgravity-2024"
        for duration in ("absent", None):
            runner = PilotRunner(self.repository, self.results)
            if duration == "absent":
                del runner.videos[kari]["durationSeconds"]
            else:
                runner.videos[kari]["durationSeconds"] = None
            for start, end, accepted in ((None, 999, True), (999, None, True), (-1, None, False),
                                         (None, -1, False), (float("nan"), None, False),
                                         (None, float("inf"), False), (20, 10, False)):
                with self.subTest(duration=duration, start=start, end=end):
                    row = self.run_timestamps(kari, start, end, runner)
                    if accepted:
                        self.assertIsNone(row["errorCategory"])
                    else:
                        self.assert_timestamp_rejected(row, start, end, runner)

    def test_recorded_duration_is_checked_before_ai_grounding_provider_call(self):
        kari = "kari-microgravity-2024"
        cases = [("absent", True), (None, True), (227, True), (227.0, True),
                 ("227", False), (True, False), (0, False), (-227, False)]
        for duration, accepted in cases:
            with self.subTest(duration=duration):
                runner = PilotRunner(self.repository, self.results)
                if duration == "absent":
                    del runner.videos[kari]["durationSeconds"]
                else:
                    runner.videos[kari]["durationSeconds"] = duration
                provider = FixtureProvider({"grounding": {"raw": GROUNDING}})
                if accepted:
                    row = runner.run_grounding(kari, "gemini_video", 1, provider)
                    self.assertIsNone(row["errorCategory"])
                    self.assertEqual(len(provider.calls), 1)
                else:
                    with self.assertRaisesRegex(ValueError, "durationSeconds"):
                        runner.run_grounding(kari, "gemini_video", 1, provider)
                    self.assertEqual(provider.calls, [])
                    self.assertFalse((self.results / "video-grounding.jsonl").exists())
                    # authorized_transcript makes no AI call and does not use durationSeconds.
                    transcript = runner.run_grounding(kari, "authorized_transcript", 1, provider,
                                                      authorized_transcript=CONTENT)
                    self.assertIsNone(transcript["errorCategory"])
                    self.assertEqual(provider.calls, [])
                if self.results.exists():
                    shutil.rmtree(self.results)

    def test_valid_recorded_duration_keeps_strict_end_bound(self):
        kari = "kari-microgravity-2024"
        for duration in (227, 227.0):
            for end, accepted in ((227, True), (227.5, False)):
                with self.subTest(duration=duration, end=end):
                    runner = PilotRunner(self.repository, self.results)
                    runner.videos[kari]["durationSeconds"] = duration
                    fact = dict(GROUNDING["facts"][0], timestampStartSeconds=200, timestampEndSeconds=end)
                    row = runner.run_grounding(
                        kari, "gemini_video", 1,
                        FixtureProvider({"grounding": {"raw": dict(GROUNDING, facts=[fact])}}))
                    self.assertEqual(row["errorCategory"],
                                     None if accepted else "fixture_invalid_grounding_response")

    def test_grounding_prompt_version_is_recorded_only_for_ai_grounding(self):
        provider = FixtureProvider({"grounding": {"raw": GROUNDING}})
        ai = self.runner.run_grounding(VIDEO, "gemini_video", 1, provider)
        transcript = self.runner.run_grounding(VIDEO, "authorized_transcript", 1, provider,
                                               authorized_transcript=CONTENT)
        self.assertEqual(ai["promptVersion"], "video-grounding-v1")
        self.assertIsNone(transcript["promptVersion"])
        self.assertEqual([row["promptVersion"] for row in self.rows("video-grounding.jsonl")],
                         ["video-grounding-v1", None])

    def test_ai_grounding_without_prompt_version_config_fails_before_provider_call(self):
        for value in (None, "", "  ", 1):
            with self.subTest(prompt_version=value):
                runner = PilotRunner(self.repository, self.results)
                if value is None:
                    del runner.config["video_grounding"]["prompt_version"]
                else:
                    runner.config["video_grounding"]["prompt_version"] = value
                provider = FixtureProvider({"grounding": {"raw": GROUNDING}})
                with self.assertRaisesRegex(ValueError, "video_grounding.prompt_version"):
                    runner.run_grounding(VIDEO, "gemini_video", 1, provider)
                self.assertEqual(provider.calls, [])
                self.assertFalse((self.results / "video-grounding.jsonl").exists())
                transcript = runner.run_grounding(VIDEO, "authorized_transcript", 1, provider,
                                                  authorized_transcript=CONTENT)
                self.assertIsNone(transcript["promptVersion"])
                self.assertEqual(provider.calls, [])
                shutil.rmtree(self.results)

    def test_grounding_prompt_version_leaves_quiz_and_direct_pilot_version_unchanged(self):
        self.assertEqual(self.runner.config["prompt_version"], "pilot-v1")
        self.runner.run_grounding(VIDEO, "gemini_video", 1,
                                  FixtureProvider({"grounding": {"raw": GROUNDING}}))
        quiz = self.run_approved_quiz(VIDEO, "gemini-3.8-flash", 1, CONTENT,
                                      FixtureProvider({"quiz": {"raw": QUIZ}}))
        direct = self.runner.run_end_to_end(VIDEO, "gemini_direct_quiz", 1,
                                            FixtureProvider({"direct": {"raw": QUIZ}}))
        self.assertEqual((quiz["promptVersion"], direct["promptVersion"]), ("pilot-v1", "pilot-v1"))
        self.assertEqual(quiz["validatorStatus"], "pass")
        self.assertEqual(direct["validatorStatus"], "not_run")

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
        # pilot-v2 Quiz needs a reviewed video-grounding-v2 source; its manifest entry is separate.
        shutil.copyfile(ROOT / "configs" / "pilot-v2.yaml", self.repository / "configs" / "pilot-v2.yaml")
        next_version = PilotRunner(self.repository, self.repository / "results" / "next",
                                   self.repository / "configs" / "pilot-v2.yaml")
        changed = CONTENT + " changed"
        source = next_version.run_grounding(VIDEO, "gemini_video", 1, SimulatedApiFixture(
            {"grounding": {"normalized": dict(GROUNDING, contentText=changed), "responseBody": FIXTURE_RAW}}))
        next_version.review_grounding(source["runId"], APPROVER,
                                      {item: "pass" for item in grounding_review.REVIEW_ITEMS})
        next_version.run_quiz(VIDEO, "gpt-5.4-mini", 1, changed, provider, source_grounding_run_id=source["runId"])
        self.assertEqual(len(provider.calls), 3)

    def test_custom_results_directory_links_grounding_approval_and_quiz(self):
        custom = PilotRunner(self.repository, self.results / "custom")
        grounding = custom.run_grounding(VIDEO, "gemini_video", 1,
                                         SimulatedApiFixture({"grounding": {
                                             "normalized": GROUNDING, "responseBody": FIXTURE_RAW}}))
        self.assertEqual(grounding["apiStatus"], "success")
        self.assertFalse((self.results / "video-grounding.jsonl").exists())
        custom.approve_content(grounding["runId"], APPROVER)
        saved = [json.loads(line) for line in
                 (custom.results / "video-grounding.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(saved[0]["contentTextApprovalStatus"], "approved")
        self.assertEqual(saved[0]["contentTextSha256"], hashlib.sha256(CONTENT.encode()).hexdigest())
        self.assertEqual(saved[0]["approvedBy"], APPROVER)
        self.assertIn("approvedAt", saved[0])
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
            # The results lock file is not a result; a refused operation may still create it.
            return {path: path.read_bytes() for path in directory.rglob("*")
                    if path.is_file() and path.name != PilotRunner.LOCK_FILE}

        custom = PilotRunner(self.repository, self.results / "custom")
        default_source = self.approved_source()
        default_before = snapshot(self.results)
        provider = SimulatedApiFixture({"quiz": {"normalized": QUIZ, "responseBody": FIXTURE_RAW}})
        with self.assertRaisesRegex(ValueError, "approved"):
            custom.run_quiz(VIDEO, "gemini-3.8-flash", 1, CONTENT, provider,
                            source_grounding_run_id=default_source)
        with self.assertRaisesRegex(ValueError, "human approval"):
            custom.approve_content(default_source, APPROVER)
        self.assertEqual(snapshot(self.results), default_before)

        grounding = custom.run_grounding(VIDEO, "gemini_video", 1,
                                         SimulatedApiFixture({"grounding": {
                                             "normalized": GROUNDING, "responseBody": FIXTURE_RAW}}))
        custom.approve_content(grounding["runId"], APPROVER)
        custom_before = snapshot(custom.results)
        with self.assertRaisesRegex(ValueError, "human approval"):
            self.runner.approve_content(grounding["runId"], APPROVER)
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

    def run_direct(self, normalized):
        provider = SimulatedApiFixture({"direct": {"normalized": normalized,
                                                   "responseBody": FIXTURE_RAW}})
        return self.runner.run_end_to_end(VIDEO, "gemini_direct_quiz", 1, provider)

    def test_direct_quiz_valid_structure_records_validator_not_run(self):
        row = self.run_direct(QUIZ)
        self.assertEqual(row["apiStatus"], "success")
        self.assertEqual(row["parseStatus"], "pass")
        self.assertEqual(row["validatorStatus"], "not_run")
        self.assertIsNone(row["errorCategory"])
        self.assertIsNone(row["evidenceTextContained"])
        self.assertEqual(row["beCompatibility"], "not_applicable")

    def test_direct_quiz_structure_contract_failure_keeps_validator_fail(self):
        duplicate_options = [dict(item) for item in QUIZ["questions"]]
        duplicate_options[0]["options"] = ["A", "a ", "C", "D"]
        cases = {"question_count": dict(QUIZ, questions=QUIZ["questions"][:2]),
                 "duplicate_options": dict(QUIZ, questions=duplicate_options),
                 "prompt_version": dict(QUIZ, promptVersion="other-version")}
        for case, output in cases.items():
            with self.subTest(case=case):
                row = self.run_direct(output)
                self.assertEqual(row["apiStatus"], "success")
                self.assertEqual(row["parseStatus"], "pass")
                self.assertEqual(row["validatorStatus"], "fail")
                self.assertEqual(row["errorCategory"], "quiz_contract_error")
                self.assertIsNone(row["evidenceTextContained"])
                self.assertEqual(row["beCompatibility"], "not_applicable")

    def test_direct_quiz_parse_failure_records_validator_not_run(self):
        row = self.run_direct("not json")
        self.assertEqual(row["apiStatus"], "success")
        self.assertEqual(row["parseStatus"], "fail")
        self.assertEqual(row["validatorStatus"], "not_run")
        self.assertEqual(row["errorCategory"], "parse_error")
        self.assertEqual(row["beCompatibility"], "not_applicable")

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

    @staticmethod
    def retry_stop_rule_accepts(rule, row):
        """Evaluate only the retryStopReason if/then rule; not general JSON Schema validation."""
        condition = rule["if"]
        if not (all(key in row for key in condition["required"])
                and all(row[key] == spec["const"] for key, spec in condition["properties"].items())):
            return True
        then = rule["then"]
        if any(key not in row for key in then["required"]):
            return False
        for key, spec in then["properties"].items():
            value = row[key]
            if ("const" in spec and value != spec["const"]
                    or spec.get("type") == "integer" and type(value) is not int
                    or spec.get("type") == "string" and not isinstance(value, str)
                    or "not" in spec and value == spec["not"]["const"]):
                return False
        return True

    def test_retry_stop_reason_schema_invariant_matches_runtime_contract(self):
        schema = json.loads((ROOT / "docs" / "run-result.schema.json").read_text(encoding="utf-8"))
        rule = {"if": {"required": ["retryStopReason"],
                       "properties": {"retryStopReason": {"const": "live_guard"}}},
                "then": {"required": ["apiStatus", "httpStatus", "errorCategory"],
                         "properties": {"apiStatus": {"const": "error"},
                                        "httpStatus": {"type": "integer"},
                                        "errorCategory": {"type": "string",
                                                          "not": {"const": "live_guard"}}}}}
        self.assertIn(rule, schema["allOf"])
        self.assertNotIn("providerErrorCode", rule["then"]["required"])
        blocked = {"apiStatus": "error", "httpStatus": 429, "errorCategory": "rate_limit",
                   "providerErrorCode": None, "retryStopReason": "live_guard"}
        valid = ({"apiStatus": "error", "httpStatus": 503, "errorCategory": "server_error"},
                 {"apiStatus": "success", "errorCategory": None, "httpStatus": None,
                  "retryStopReason": None},
                 {"apiStatus": "error", "errorCategory": "live_guard", "httpStatus": None,
                  "retryStopReason": None},
                 blocked)
        invalid = ({**blocked, "apiStatus": "success"},
                   {key: value for key, value in blocked.items() if key != "apiStatus"},
                   {key: value for key, value in blocked.items() if key != "httpStatus"},
                   {**blocked, "httpStatus": None},
                   {key: value for key, value in blocked.items() if key != "errorCategory"},
                   {**blocked, "errorCategory": "live_guard"})
        for row in valid:
            with self.subTest(valid=row):
                self.assertTrue(self.retry_stop_rule_accepts(rule, row))
        for row in invalid:
            with self.subTest(invalid=row):
                self.assertFalse(self.retry_stop_rule_accepts(rule, row))

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
        self.runner.approve_content(self.approved_source(approval=None), APPROVER)
        schema = json.loads((ROOT / "docs" / "run-result.schema.json").read_text(encoding="utf-8"))
        self.assertEqual(schema["properties"]["contentTextApprovalStatus"]["enum"],
                         ["approved", "rejected", None])  # rejected only via the Pilot v2 review
        self.assertEqual(schema["oneOf"][0]["then"]["required"], ["contentTextSha256"])
        self.assertEqual(schema["properties"]["approvedBy"],
                         {"type": "string", "minLength": 1,
                          "maxLength": approval_tracking.APPROVED_BY_MAX_LENGTH})
        self.assertEqual({key: value for key, value in schema["properties"]["approvedAt"].items()
                          if key != "pattern"}, {"type": "string", "format": "date-time"})
        self.assertEqual(schema["dependentRequired"],
                         {"approvedBy": ["approvedAt"], "approvedAt": ["approvedBy"]})
        grounding_rows = self.rows("video-grounding.jsonl")
        self.assertTrue(any("approvedBy" in row for row in grounding_rows))
        self.assertTrue(any(row.get("contentTextApprovalStatus") == "approved"
                            and "approvedBy" not in row for row in grounding_rows))
        for filename in ("video-grounding.jsonl", "quiz-generation.jsonl", "end-to-end.jsonl"):
            for row in self.rows(filename):
                self.assertTrue(set(schema["required"]).issubset(row))
                self.assertTrue(set(row).issubset(schema["properties"]))
                self.assertEqual("approvedBy" in row, "approvedAt" in row)
                self.assertIn(row["benchmarkType"], schema["properties"]["benchmarkType"]["enum"])
                self.assertIn(row["apiStatus"], schema["properties"]["apiStatus"]["enum"])


ALL_PASS = {item: "pass" for item in grounding_review.REVIEW_ITEMS}


class PilotV2GroundingReviewTest(unittest.TestCase):
    """Pilot v2 (video-grounding-v2) human Grounding checklist and its A/B gate."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repository = Path(self.temp.name)
        (self.repository / "configs").mkdir()
        (self.repository / "data").mkdir()
        for name in ("pilot.yaml", "pilot-v2.yaml"):
            shutil.copyfile(ROOT / "configs" / name, self.repository / "configs" / name)
        shutil.copyfile(ROOT / "data" / "videos.jsonl", self.repository / "data" / "videos.jsonl")
        self.results = self.repository / "results"
        self.v1 = PilotRunner(self.repository, self.results)
        self.v2 = PilotRunner(self.repository, self.results, self.repository / "configs" / "pilot-v2.yaml")

    def tearDown(self):
        self.temp.cleanup()

    def grounding_file(self):
        return self.results / "video-grounding.jsonl"

    def rows(self):
        return [json.loads(line) for line in self.grounding_file().read_text(encoding="utf-8").splitlines()]

    def snapshot(self):
        # The results lock file is not a result; a refused operation may still create it.
        return {path: path.read_bytes() for path in self.repository.rglob("*")
                if path.is_file() and path.name != PilotRunner.LOCK_FILE}

    def grounding(self, runner=None, video_id=VIDEO, repetition=1):
        provider = SimulatedApiFixture({"grounding": {"normalized": GROUNDING, "responseBody": FIXTURE_RAW}})
        row = (runner or self.v2).run_grounding(video_id, "gemini_video", repetition, provider)
        self.assertEqual(row["apiStatus"], "success")
        return row

    def reviewed(self, checklist=ALL_PASS):
        row = self.grounding()
        return self.v2.review_grounding(row["runId"], APPROVER, checklist)

    def quiz(self, model, source_id, runner=None, quiz=None):
        provider = SimulatedApiFixture({"quiz": {"normalized": quiz or dict(QUIZ, promptVersion="pilot-v2"),
                                                 "responseBody": FIXTURE_RAW}})
        return (runner or self.v2).run_quiz(VIDEO, model, 1, CONTENT, provider,
                                            source_grounding_run_id=source_id), provider

    def test_v2_config_changes_only_the_two_versions(self):
        self.assertEqual((self.v1.config["prompt_version"], self.v1.config["video_grounding"]["prompt_version"]),
                         ("pilot-v1", "video-grounding-v1"))
        self.assertEqual((self.v2.config["prompt_version"], self.v2.config["video_grounding"]["prompt_version"]),
                         ("pilot-v2", "video-grounding-v2"))
        for config in (self.v1.config, self.v2.config):
            config.pop("prompt_version")
            config["video_grounding"].pop("prompt_version")
        self.assertEqual(self.v1.config, self.v2.config)

    def test_runner_passes_the_grounding_prompt_version_to_the_provider(self):
        for runner, version in ((self.v1, "video-grounding-v1"), (self.v2, "video-grounding-v2")):
            provider = FixtureProvider({"grounding": {"raw": GROUNDING}})
            row = runner.run_grounding(VIDEO, "gemini_video", 1, provider)
            self.assertEqual((row["promptVersion"], provider.calls[0]["promptVersion"]), (version, version))

    def test_legacy_and_v1_grounding_attempts_keep_their_sequence_and_v2_starts_at_one(self):
        self.results.mkdir()
        legacy = "".join(json.dumps({"videoId": VIDEO, "method": "gemini_video", "model": "gemini-3.8-flash",
                                     "repetition": 1, "runId": "legacy-%d" % number, "apiStatus": "error",
                                     "promptVersion": None, **({} if number == 1 else {"attempt": number})}) + "\n"
                         for number in (1, 2))
        self.grounding_file().write_text(legacy, encoding="utf-8")
        provider = FixtureProvider({"grounding": {"raw": GROUNDING}})
        self.assertEqual(self.v1.run_grounding(VIDEO, "gemini_video", 1, provider)["attempt"], 3)
        self.assertEqual(self.v2.run_grounding(VIDEO, "gemini_video", 1, provider)["attempt"], 1)
        self.assertEqual(self.v2.run_grounding(VIDEO, "gemini_video", 1, provider)["attempt"], 2)
        self.assertEqual(self.v1.run_grounding(VIDEO, "gemini_video", 1, provider)["attempt"], 4)
        self.assertTrue(self.grounding_file().read_text(encoding="utf-8").startswith(legacy))

    def test_all_pass_checklist_approves_and_keeps_approval_tracking_consistent(self):
        row = self.reviewed(dict(ALL_PASS, reviewNote="검토 메모"))
        stored = self.rows()[0]
        self.assertEqual(stored, row)
        self.assertEqual(stored["contentTextApprovalStatus"], "approved")
        review = stored["groundingReview"]
        self.assertEqual({key: review[key] for key in grounding_review.REVIEW_ITEMS}, ALL_PASS)
        self.assertEqual((review["reviewNote"], review["reviewedBy"]), ("검토 메모", APPROVER))
        self.assertEqual((stored["approvedBy"], stored["approvedAt"]), (APPROVER, review["reviewedAt"]))
        without_note = self.grounding(repetition=2)
        self.assertIsNone(self.v2.review_grounding(without_note["runId"], APPROVER, ALL_PASS)
                          ["groundingReview"]["reviewNote"])  # reviewNote is optional

    def test_any_fail_or_uncertain_rejects(self):
        for item in grounding_review.REVIEW_ITEMS:
            for verdict in ("fail", "uncertain"):
                with self.subTest(item=item, verdict=verdict):
                    row = self.reviewed(dict(ALL_PASS, **{item: verdict}))
                    self.assertEqual(row["contentTextApprovalStatus"], "rejected")
                    self.assertEqual(row["groundingReview"][item], verdict)
                    self.assertNotIn("approvedBy", row)
                    self.assertNotIn("approvedAt", row)
                    shutil.rmtree(self.results)

    def test_malformed_or_incomplete_checklist_is_rejected_without_writing(self):
        row = self.grounding()
        missing = {key: value for key, value in ALL_PASS.items() if key != "factsConsistency"}
        cases = [(missing, APPROVER), (dict(ALL_PASS, extra="pass"), APPROVER),
                 (dict(ALL_PASS, factualAccuracy=None), APPROVER), (dict(ALL_PASS, factualAccuracy="PASS"), APPROVER),
                 (dict(ALL_PASS, factualAccuracy=True), APPROVER), (dict(ALL_PASS, reviewNote=1), APPROVER),
                 (dict(ALL_PASS, reviewedBy="someone"), APPROVER), (list(ALL_PASS), APPROVER), (None, APPROVER),
                 (ALL_PASS, ""), (ALL_PASS, "  "), (ALL_PASS, "line\nbreak"), (ALL_PASS, None)]
        before = self.snapshot()
        for checklist, reviewer in cases:
            with self.subTest(checklist=checklist, reviewer=reviewer), self.assertRaises(ValueError):
                self.v2.review_grounding(row["runId"], reviewer, checklist)
        self.assertEqual(self.snapshot(), before)

    def test_review_targets_only_unreviewed_successful_v2_grounding(self):
        v1 = self.grounding(self.v1)
        failed = self.v2.run_grounding(VIDEO, "gemini_video", 2, SimulatedApiFixture(
            {"grounding": {"errorCategory": "server_error"}}))
        invalid = dict(GROUNDING, facts=[dict(GROUNDING["facts"][0], evidenceType="narration")])
        invalid_row = self.v2.run_grounding("kari-microgravity-2024", "gemini_video", 1, SimulatedApiFixture(
            {"grounding": {"normalized": invalid, "responseBody": FIXTURE_RAW}}))
        self.assertEqual(invalid_row["errorCategory"], "invalid_grounding_response")
        tampered = self.grounding(video_id="nasa-methane-2020")
        (self.results / "evaluation" / (tampered["runId"] + ".json")).write_text(
            json.dumps({"contentText": CONTENT + " changed", "facts": []}), encoding="utf-8")
        reviewed = self.reviewed()
        before = self.snapshot()
        # Failed and unknown runs have no evaluation output to review.
        for run_id, message in ((v1["runId"], "does not use"), (failed["runId"], "evaluation are required"),
                                (invalid_row["runId"], "successful"),
                                (tampered["runId"], "hash"), (reviewed["runId"], "never overwritten"),
                                (uuid.uuid4().hex, "evaluation are required")):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.v2.review_grounding(run_id, APPROVER, dict(ALL_PASS, factualAccuracy="fail"))
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.rows()[-1], reviewed)

    def test_v2_grounding_cannot_bypass_the_checklist_through_approve_content(self):
        row = self.grounding()
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "review_grounding"):
            self.v2.approve_content(row["runId"], APPROVER)
        with self.assertRaisesRegex(ValueError, "review_grounding"):
            self.v1.approve_content(row["runId"], APPROVER)  # The row's version decides, not the config.
        self.assertEqual(self.snapshot(), before)

    def test_v1_approve_content_is_unchanged(self):
        row = self.grounding(self.v1)
        approved = self.v1.approve_content(row["runId"], APPROVER)
        self.assertEqual((approved["contentTextApprovalStatus"], approved["approvedBy"]), ("approved", APPROVER))
        self.assertNotIn("groundingReview", approved)
        quiz, provider = self.quiz("gemini-3.8-flash", row["runId"], runner=self.v1, quiz=QUIZ)
        self.assertEqual((quiz["apiStatus"], quiz["promptVersion"], len(provider.calls)), ("success", "pilot-v1", 1))

    def test_rejected_grounding_never_reaches_ab_quiz(self):
        rejected = self.reviewed(dict(ALL_PASS, keyInformationCoverage="uncertain"))
        before = self.snapshot()
        for model in ("gemini-3.8-flash", "gpt-5.4-mini"):
            provider = SimulatedApiFixture({"quiz": {"normalized": dict(QUIZ, promptVersion="pilot-v2"),
                                                     "responseBody": FIXTURE_RAW}})
            with self.subTest(model=model), self.assertRaisesRegex(ValueError, "approved"):
                self.v2.run_quiz(VIDEO, model, 1, CONTENT, provider, source_grounding_run_id=rejected["runId"])
            self.assertEqual(provider.calls, [])
        self.assertFalse((self.results / "quiz-generation.jsonl").exists())
        self.assertEqual(self.snapshot(), before)  # rejected state kept; no manifest or Quiz row
        self.assertEqual(self.rows(), [rejected])  # no Grounding was regenerated

    def assert_generation_refused(self, runner=None):
        before = self.snapshot()
        for provider in (FixtureProvider({"grounding": {"raw": GROUNDING}}),
                         SimulatedApiFixture({"grounding": {"normalized": GROUNDING, "responseBody": FIXTURE_RAW}})):
            with self.assertRaisesRegex(ValueError, "successful Grounding already exists"):
                (runner or self.v2).run_grounding(VIDEO, "gemini_video", 1, provider)
            self.assertEqual(provider.calls, [])
        self.assertEqual(self.snapshot(), before)  # no row, evaluation or raw file; nothing changed

    def test_rejected_condition_refuses_manual_grounding_rerun_before_provider_call(self):
        rejected = self.reviewed(dict(ALL_PASS, factsConsistency="fail"))
        self.assert_generation_refused()
        self.assert_generation_refused(PilotRunner(self.repository, self.results,
                                                   self.repository / "configs" / "pilot-v2.yaml"))
        self.assertEqual(self.rows(), [rejected])

    def test_technical_failure_then_success_then_rejection_is_terminal(self):
        failed = self.v2.run_grounding(VIDEO, "gemini_video", 1, SimulatedApiFixture(
            {"grounding": {"errorCategory": "server_error"}}))
        self.assertEqual((failed["apiStatus"], failed["attempt"]), ("error", 1))
        success = self.grounding()  # A technical failure allows the next attempt.
        self.assertEqual(success["attempt"], 2)
        rejected = self.v2.review_grounding(success["runId"], APPROVER, dict(ALL_PASS, koreanConsistency="fail"))
        self.assert_generation_refused()
        self.assertEqual(self.rows(), [failed, rejected])

    def test_unreviewed_or_approved_success_is_terminal_for_its_condition(self):
        success = self.grounding()
        self.assert_generation_refused()  # Unreviewed: no second successful candidate.
        self.assertEqual(self.rows(), [success])
        approved = self.v2.review_grounding(success["runId"], APPROVER, ALL_PASS)
        self.assert_generation_refused()
        self.assertEqual(self.rows(), [approved])

    def test_other_conditions_stay_independent_of_a_terminal_condition(self):
        self.reviewed(dict(ALL_PASS, factsConsistency="fail"))
        self.assertEqual(self.grounding(repetition=2)["attempt"], 1)
        self.assertEqual(self.grounding(video_id="nasa-methane-2020")["attempt"], 1)
        self.assertEqual(self.grounding(self.v1)["promptVersion"], "video-grounding-v1")
        other_model = dict(self.rows()[0], model="another-video-model")
        condition = grounding_review.grounding_condition(self.rows()[0])
        self.assertEqual(grounding_review.successful_review_candidates([other_model], condition), [])

    def test_pilot_v1_keeps_its_retry_semantics_after_success(self):
        first = self.grounding(self.v1)
        second = self.grounding(self.v1)  # The terminal-success rule is Pilot v2 only.
        self.assertEqual((first["attempt"], second["attempt"]), (1, 2))
        self.v1.approve_content(second["runId"], APPROVER)
        self.v1._check_storage_integrity()

    def two_successful_candidates(self):
        """Historical/tampered storage: two successful candidates of one Pilot v2 condition."""
        first = self.grounding()
        second = dict(first, runId=uuid.uuid4().hex, attempt=2)
        with self.grounding_file().open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(second) + "\n")
        for directory in ("raw", "evaluation"):
            shutil.copyfile(self.results / directory / (first["runId"] + ".json"),
                            self.results / directory / (second["runId"] + ".json"))
        return first, second

    def test_multiple_successful_candidates_block_every_action_without_choosing_one(self):
        first, second = self.two_successful_candidates()
        before = self.snapshot()
        for action in (lambda: self.v2.review_grounding(first["runId"], APPROVER, dict(ALL_PASS, factualAccuracy="fail")),
                       lambda: self.v2.review_grounding(second["runId"], APPROVER, ALL_PASS),
                       lambda: self.quiz("gemini-3.8-flash", second["runId"]),
                       lambda: self.v2.run_grounding(VIDEO, "gemini_video", 1,
                                                     FixtureProvider({"grounding": {"raw": GROUNDING}}))):
            with self.assertRaisesRegex(ValueError, "more than one successful Grounding"):
                action()
        self.assertEqual(self.snapshot(), before)

    def test_review_refuses_a_condition_with_another_candidate_even_if_integrity_is_bypassed(self):
        first, second = self.two_successful_candidates()
        before = self.snapshot()
        with mock.patch.object(self.v2, "_check_storage_integrity"):
            for run_id, checklist in ((first["runId"], dict(ALL_PASS, factualAccuracy="fail")),
                                      (second["runId"], ALL_PASS)):
                with self.assertRaisesRegex(ValueError, "more than one successful Grounding candidate"):
                    self.v2.review_grounding(run_id, APPROVER, checklist)
        self.assertEqual(self.snapshot(), before)

    def approved_v1_source(self, legacy=False):
        row = self.grounding(self.v1)
        self.v1.approve_content(row["runId"], APPROVER)
        if legacy:
            rows = self.rows()
            rows[-1]["promptVersion"] = None  # Like the stored pre-versioning Pilot v1 results.
            self.grounding_file().write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")
        return row["runId"]

    def test_quiz_source_must_belong_to_the_configured_grounding_version(self):
        sources = {"legacy": self.approved_v1_source(legacy=True)}
        v2_source = self.reviewed()["runId"]
        sources["v1"] = self.approved_v1_source()
        refused = (("pilot-v2 + legacy", self.v2, sources["legacy"], "pilot-v2"),
                   ("pilot-v2 + v1", self.v2, sources["v1"], "pilot-v2"),
                   ("pilot-v1 + v2", self.v1, v2_source, "pilot-v1"))
        for name, runner, source, quiz_version in refused:
            with self.subTest(name=name):
                before = self.snapshot()
                provider = SimulatedApiFixture({"quiz": {"normalized": dict(QUIZ, promptVersion=quiz_version),
                                                         "responseBody": FIXTURE_RAW}})
                with self.assertRaisesRegex(ValueError, "does not belong to this Pilot experiment"):
                    runner.run_quiz(VIDEO, "gemini-3.8-flash", 1, CONTENT, provider, source_grounding_run_id=source)
                self.assertEqual(provider.calls, [])
                self.assertFalse((self.results / "quiz-generation.jsonl").exists())
                self.assertFalse((self.repository / "data" / "restricted").exists())  # no manifest
                self.assertEqual(self.snapshot(), before)
        for name, runner, source, quiz_version in (("pilot-v1 + legacy", self.v1, sources["legacy"], "pilot-v1"),
                                                   ("pilot-v1 + v1", self.v1, sources["v1"], "pilot-v1"),
                                                   ("pilot-v2 + v2", self.v2, v2_source, "pilot-v2")):
            with self.subTest(name=name):
                row, provider = self.quiz("gpt-5.4-mini", source, runner=runner,
                                          quiz=dict(QUIZ, promptVersion=quiz_version))
                self.assertEqual((row["apiStatus"], row["promptVersion"], len(provider.calls)),
                                 ("success", quiz_version, 1))

    def test_unknown_grounding_prompt_version_is_refused_before_any_record(self):
        for value in ("video-grounding-v9", "pilot-v2"):
            with self.subTest(value=value):
                runner = PilotRunner(self.repository, self.results, self.repository / "configs" / "pilot-v2.yaml")
                runner.config["video_grounding"]["prompt_version"] = value
                for provider in (FixtureProvider({"grounding": {"raw": GROUNDING}}),
                                 SimulatedApiFixture({"grounding": {"normalized": GROUNDING,
                                                                    "responseBody": FIXTURE_RAW}})):
                    with self.assertRaisesRegex(ValueError, "not a known Grounding prompt version"):
                        runner.run_grounding(VIDEO, "gemini_video", 1, provider)
                    self.assertEqual(provider.calls, [])
                self.assertEqual([path.name for path in self.results.rglob("*")], [PilotRunner.LOCK_FILE])

    def test_stored_grounding_prompt_version_must_be_legacy_null_or_known(self):
        stored = self.grounding(self.v1, video_id="nasa-methane-2020")  # rows[0], the row edited below
        quiz_source = self.grounding(self.v1)
        self.v1.approve_content(quiz_source["runId"], APPROVER)
        self.quiz("gemini-3.8-flash", quiz_source["runId"], runner=self.v1, quiz=QUIZ)  # a pilot-v1 Quiz row

        def store(value):
            rows = self.rows()
            rows[0]["promptVersion"] = value
            self.grounding_file().write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

        for value in ([], {}, 1, True, "", "unknown-grounding-version", "pilot-v1"):
            with self.subTest(value=value):
                store(value)
                before = self.snapshot()
                with self.assertRaisesRegex(ValueError, "integrity error: invalid video-grounding.jsonl line 1"):
                    self.v1._check_storage_integrity()
                provider = FixtureProvider({"grounding": {"raw": GROUNDING}})
                for action in (lambda: self.v1.run_grounding("kari-microgravity-2024", "gemini_video", 1, provider),
                               lambda: self.v2.review_grounding(stored["runId"], APPROVER, ALL_PASS),
                               lambda: self.v1.approve_content(stored["runId"], APPROVER)):
                    with self.assertRaisesRegex(ValueError, "integrity error"):
                        action()
                self.assertEqual(provider.calls, [])
                self.assertEqual(self.snapshot(), before)
        # Legacy null and known versions stay valid; the Quiz row keeps its Pilot version (pilot-v1).
        for value in (None, "video-grounding-v1", "video-grounding-v2"):
            with self.subTest(value=value):
                store(value)
                self.v1._check_storage_integrity()
        self.assertEqual(json.loads((self.results / "quiz-generation.jsonl").read_text(encoding="utf-8"))
                         ["promptVersion"], "pilot-v1")

    def test_cli_selects_pilot_v1_by_default_and_pilot_v2_with_config(self):
        import src.pilot_runner as pilot_runner

        class Stop(Exception):
            pass

        base = ["src.pilot_runner", "grounding", "--video-id", VIDEO, "--repetition", "1",
                "--fixture", "unused.json"]
        for extra, expected in (([], None), (["--config", "configs/pilot-v2.yaml"], Path("configs/pilot-v2.yaml"))):
            with self.subTest(extra=extra), mock.patch.object(sys, "argv", base + extra), \
                    mock.patch.object(pilot_runner, "PilotRunner", side_effect=Stop) as runner:
                with self.assertRaises(Stop):
                    pilot_runner.main()
                self.assertEqual(runner.call_args[0], (ROOT, Path("results"), expected))
        # PilotRunner without a config keeps reading configs/pilot.yaml (pilot-v1).
        self.assertEqual(PilotRunner(self.repository, self.results, None).config["prompt_version"], "pilot-v1")

    def test_approved_v2_grounding_runs_ab_quiz_and_coexists_with_pilot_v1(self):
        v1_source = self.grounding(self.v1)
        self.v1.approve_content(v1_source["runId"], APPROVER)
        self.quiz("gemini-3.8-flash", v1_source["runId"], runner=self.v1, quiz=QUIZ)
        approved = self.reviewed()
        rows = [self.quiz(model, approved["runId"]) for model in ("gemini-3.8-flash", "gpt-5.4-mini")]
        for row, provider in rows:
            self.assertEqual((row["apiStatus"], row["validatorStatus"], row["promptVersion"]),
                             ("success", "pass", "pilot-v2"))
            self.assertEqual((row["sourceGroundingRunId"], row["contentTextSha256"]),
                             (approved["runId"], approved["contentTextSha256"]))
            self.assertEqual(provider.calls[0]["contentText"], CONTENT)  # contentText only, no facts
        manifests = [json.loads(path.read_text(encoding="utf-8")) for path in
                     (self.repository / "data" / "restricted" / "fixed-content").glob("*.json")]
        self.assertEqual(sorted(item["promptVersion"] for item in manifests), ["pilot-v1", "pilot-v2"])

    def test_tampered_approval_or_review_fails_closed(self):
        def tamper(row):
            self.grounding_file().write_text(json.dumps(row) + "\n", encoding="utf-8")

        approved = self.reviewed()
        rejected_review = dict(approved["groundingReview"], factualAccuracy="fail")
        cases = {
            "approved without review": {k: v for k, v in approved.items() if k != "groundingReview"},
            "approved with a fail item": dict(approved, groundingReview=rejected_review),
            "rejected with all pass": {k: v for k, v in dict(approved, contentTextApprovalStatus="rejected").items()
                                       if k not in ("approvedBy", "approvedAt")},
            "rejected without review": dict(
                {k: v for k, v in approved.items() if k not in ("groundingReview", "approvedBy", "approvedAt")},
                contentTextApprovalStatus="rejected"),
            "review on a v1 row": dict(approved, promptVersion="video-grounding-v1"),
            "approver differs from reviewer": dict(approved, approvedBy="someone-else"),
            "review with an extra key": dict(approved, groundingReview=dict(approved["groundingReview"], x=1)),
        }
        for name, row in cases.items():
            with self.subTest(name=name):
                tamper(row)
                provider = SimulatedApiFixture({"quiz": {"normalized": dict(QUIZ, promptVersion="pilot-v2"),
                                                         "responseBody": FIXTURE_RAW}})
                with self.assertRaisesRegex(ValueError, "integrity|approved|review"):
                    self.v2.run_quiz(VIDEO, "gemini-3.8-flash", 1, CONTENT, provider,
                                     source_grounding_run_id=approved["runId"])
                self.assertEqual(provider.calls, [])
                with self.assertRaisesRegex(ValueError, "integrity"):
                    self.v2._check_storage_integrity()

    def test_pilot_v1_rows_need_no_review(self):
        legacy = self.grounding(self.v1)
        self.v1.approve_content(legacy["runId"], APPROVER)
        rows = self.rows()
        rows[0].pop("approvedBy")
        rows[0].pop("approvedAt")
        rows[0]["promptVersion"] = None  # Pre-versioning approved row, like the stored Pilot v1 results.
        self.grounding_file().write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
        self.v1._check_storage_integrity()
        quiz, provider = self.quiz("gpt-5.4-mini", legacy["runId"], runner=self.v1, quiz=QUIZ)
        self.assertEqual((quiz["apiStatus"], len(provider.calls)), ("success", 1))


class BlockingProvider(SimulatedApiFixture):
    """Fake API that holds the operation inside its critical section until released."""

    def __init__(self, responses):
        super().__init__(responses)
        self.entered, self.release = threading.Event(), threading.Event()

    def invoke(self, kind, **kwargs):
        self.entered.set()
        if not self.release.wait(timeout=10):
            raise AssertionError("BlockingProvider was never released")
        return super().invoke(kind, **kwargs)


BUSY = "already in progress for this results directory"
GROUNDED = {"grounding": {"normalized": GROUNDING, "responseBody": FIXTURE_RAW}}


class PilotResultsLockTest(unittest.TestCase):
    """One Pilot mutation per results directory; a concurrent one fails fast (fake Providers only)."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repository = Path(self.temp.name)
        (self.repository / "configs").mkdir()
        (self.repository / "data").mkdir()
        for name in ("pilot.yaml", "pilot-v2.yaml"):
            shutil.copyfile(ROOT / "configs" / name, self.repository / "configs" / name)
        shutil.copyfile(ROOT / "data" / "videos.jsonl", self.repository / "data" / "videos.jsonl")
        self.results = self.repository / "results"
        self.pool = ThreadPoolExecutor(max_workers=2)

    def tearDown(self):
        self.pool.shutdown(wait=True)
        self.temp.cleanup()

    def runner(self, results=None):
        # A separate instance per caller, like a separate CLI process using the same directory.
        return PilotRunner(self.repository, results or self.results, self.repository / "configs" / "pilot-v2.yaml")

    def grounding_rows(self, results=None):
        path = (results or self.results) / "video-grounding.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def artifacts(self, results=None):
        results = results or self.results
        return {name: sorted(path.stem for path in (results / name).glob("*.json")) for name in ("raw", "evaluation")}

    def hold(self, call, provider):
        """Start ``call`` in another thread and wait until it is inside the locked operation."""
        future = self.pool.submit(call)
        self.assertTrue(provider.entered.wait(timeout=10))
        return future

    def test_concurrent_same_condition_grounding_runs_only_once(self):
        first, second = BlockingProvider(GROUNDED), SimulatedApiFixture(GROUNDED)
        future = self.hold(lambda: self.runner().run_grounding(VIDEO, "gemini_video", 1, first), first)
        with self.assertRaisesRegex(ValueError, BUSY):
            self.runner().run_grounding(VIDEO, "gemini_video", 1, second)
        first.release.set()
        row = future.result(timeout=10)
        self.assertEqual((len(first.calls), second.calls), (1, []))
        rows = self.grounding_rows()
        self.assertEqual([(item["runId"], item["apiStatus"], item["attempt"]) for item in rows],
                         [(row["runId"], "success", 1)])
        self.assertEqual(self.artifacts(), {"raw": [row["runId"]], "evaluation": [row["runId"]]})
        self.assertEqual(len(grounding_review.successful_review_candidates(
            rows, grounding_review.grounding_condition(row))), 1)
        self.runner()._check_storage_integrity()  # no orphan artifact
        # After the lock is released, the terminal-success rule still refuses the same condition.
        with self.assertRaisesRegex(ValueError, "successful Grounding already exists"):
            self.runner().run_grounding(VIDEO, "gemini_video", 1, second)
        self.assertEqual(second.calls, [])

    def test_simultaneous_start_allows_exactly_one_generation(self):
        barrier = threading.Barrier(2)
        providers = [SimulatedApiFixture(GROUNDED), SimulatedApiFixture(GROUNDED)]

        def start(provider):
            runner = self.runner()
            barrier.wait(timeout=10)
            try:
                return runner.run_grounding(VIDEO, "gemini_video", 1, provider)["apiStatus"]
            except ValueError as exc:
                return str(exc)

        outcomes = [future.result(timeout=10) for future in
                    [self.pool.submit(start, provider) for provider in providers]]
        # The loser is refused by the lock, or, if the winner already finished, by terminal success.
        self.assertEqual(sum(len(provider.calls) for provider in providers), 1)
        self.assertEqual(outcomes.count("success"), 1)
        self.assertTrue(any(BUSY in item or "successful Grounding already exists" in item
                            for item in outcomes if item != "success"))
        self.assertEqual(len(self.grounding_rows()), 1)
        self.runner()._check_storage_integrity()

    def test_concurrent_reviews_store_exactly_one_review(self):
        candidate = self.runner().run_grounding(VIDEO, "gemini_video", 1, SimulatedApiFixture(GROUNDED))
        entered, release = threading.Event(), threading.Event()
        original = PilotRunner._rewrite_grounding_row

        def held_rewrite(result_file, lines, index, row):
            entered.set()
            self.assertTrue(release.wait(timeout=10))
            return original(result_file, lines, index, row)

        with mock.patch.object(PilotRunner, "_rewrite_grounding_row", staticmethod(held_rewrite)):
            future = self.pool.submit(self.runner().review_grounding, candidate["runId"], "reviewer-a", ALL_PASS)
            self.assertTrue(entered.wait(timeout=10))
            with self.assertRaisesRegex(ValueError, BUSY):
                self.runner().review_grounding(candidate["runId"], "reviewer-b",
                                               dict(ALL_PASS, factualAccuracy="fail"))
            release.set()
            reviewed = future.result(timeout=10)
        stored = self.grounding_rows()
        self.assertEqual(stored, [reviewed])
        review = stored[0]["groundingReview"]
        self.assertEqual((review["reviewedBy"], stored[0]["contentTextApprovalStatus"],
                          stored[0]["approvedBy"], stored[0]["approvedAt"]),
                         ("reviewer-a", "approved", "reviewer-a", review["reviewedAt"]))
        self.runner()._check_storage_integrity()
        with self.assertRaisesRegex(ValueError, "never overwritten"):
            self.runner().review_grounding(candidate["runId"], "reviewer-b", dict(ALL_PASS, factualAccuracy="fail"))
        self.assertEqual(self.grounding_rows(), [reviewed])

    def test_different_operations_never_overlap_and_keep_every_row(self):
        source = self.runner().run_grounding(VIDEO, "gemini_video", 1, SimulatedApiFixture(GROUNDED))
        self.runner().review_grounding(source["runId"], APPROVER, ALL_PASS)
        quiz = {"quiz": {"normalized": dict(QUIZ, promptVersion="pilot-v2"), "responseBody": FIXTURE_RAW}}
        quiz_a, quiz_b = BlockingProvider(quiz), SimulatedApiFixture(quiz)
        future = self.hold(lambda: self.runner().run_quiz(VIDEO, "gemini-3.8-flash", 1, CONTENT, quiz_a,
                                                          source_grounding_run_id=source["runId"]), quiz_a)
        other = SimulatedApiFixture(GROUNDED)
        for blocked in (lambda: self.runner().run_quiz(VIDEO, "gpt-5.4-mini", 1, CONTENT, quiz_b,
                                                       source_grounding_run_id=source["runId"]),
                        lambda: self.runner().run_grounding("nasa-methane-2020", "gemini_video", 1, other),
                        lambda: self.runner().approve_content(source["runId"], APPROVER),
                        lambda: self.runner().run_end_to_end(VIDEO, "gemini_direct_quiz", 1, other)):
            with self.assertRaisesRegex(ValueError, BUSY):
                blocked()
        quiz_a.release.set()
        first = future.result(timeout=10)
        self.assertEqual((quiz_b.calls, other.calls), ([], []))
        second = self.runner().run_quiz(VIDEO, "gpt-5.4-mini", 1, CONTENT, quiz_b,
                                        source_grounding_run_id=source["runId"])
        quiz_rows = [json.loads(line) for line in
                     (self.results / "quiz-generation.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["runId"] for row in quiz_rows], [first["runId"], second["runId"]])
        self.runner()._check_storage_integrity()

    def test_lock_is_released_after_success_and_after_exceptions(self):
        runner = self.runner()
        runner.run_grounding(VIDEO, "gemini_video", 1, SimulatedApiFixture(GROUNDED))

        class Interrupted(SimulatedApiFixture):
            def invoke(self, kind, **kwargs):
                raise KeyboardInterrupt()

        with self.assertRaises(KeyboardInterrupt):  # escapes the operation, not a recorded failure
            runner.run_grounding("nasa-methane-2020", "gemini_video", 1, Interrupted({}))
        with self.assertRaisesRegex(ValueError, "Unknown Pilot videoId"):
            runner.run_grounding("unknown-video", "gemini_video", 1, SimulatedApiFixture(GROUNDED))
        # No stale lock: the next operation acquires it normally.
        row = self.runner().run_grounding("nasa-methane-2020", "gemini_video", 1, SimulatedApiFixture(GROUNDED))
        self.assertEqual(row["apiStatus"], "success")
        self.assertTrue((self.results / PilotRunner.LOCK_FILE).is_file())  # existing file is not a lock

    def test_only_lock_contention_is_reported_as_busy(self):
        with self.runner()._results_lock():
            with self.assertRaisesRegex(ValueError, BUSY):  # real contention from another handle
                with self.runner()._results_lock():
                    self.fail("the lock must not be granted twice")
        failure = OSError(errno.EIO, "simulated lock I/O failure")
        with mock.patch("msvcrt.locking" if os.name == "nt" else "fcntl.flock", side_effect=failure):
            with self.assertRaises(OSError) as raised:
                with self.runner()._results_lock():
                    self.fail("the lock must not be granted after a lock error")
        self.assertIs(raised.exception, failure)  # not disguised as a concurrent operation
        with self.runner()._results_lock():  # the failed attempt left no lock behind
            pass

    def test_results_lock_holds_across_processes(self):
        child_code = textwrap.dedent("""
            import sys
            from pathlib import Path
            sys.path.insert(0, sys.argv[1])
            from src.pilot_runner import PilotRunner
            with PilotRunner(Path(sys.argv[2]), Path(sys.argv[3]))._results_lock():
                print("locked", flush=True)
                sys.stdin.readline()
            print("released", flush=True)
        """)
        child = subprocess.Popen([sys.executable, "-c", child_code, str(ROOT), str(self.repository), str(self.results)],
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        lines = queue.Queue()
        reader = threading.Thread(target=lambda: [lines.put(line.strip()) for line in child.stdout], daemon=True)
        reader.start()
        try:
            self.assertEqual(lines.get(timeout=30), "locked")  # handshake: the child holds the OS lock
            provider = FixtureProvider({"grounding": {"raw": GROUNDING}})
            with self.assertRaisesRegex(ValueError, BUSY):
                self.runner().run_grounding(VIDEO, "gemini_video", 1, provider)
            self.assertEqual(provider.calls, [])
            self.assertEqual([path.name for path in self.results.rglob("*")], [PilotRunner.LOCK_FILE])
            child.stdin.write("\n")
            child.stdin.flush()
            self.assertEqual(lines.get(timeout=30), "released")
            self.assertEqual(child.wait(timeout=30), 0)
            with self.runner()._results_lock():  # released by the other process: acquired again
                pass
        finally:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=30)
            reader.join(timeout=30)
            child.stdin.close()
            child.stdout.close()

    def test_different_results_directories_do_not_block_each_other(self):
        first_dir, second_dir = self.results / "first", self.results / "second"
        holder = BlockingProvider(GROUNDED)
        future = self.hold(lambda: self.runner(first_dir).run_grounding(VIDEO, "gemini_video", 1, holder), holder)
        row = self.runner(second_dir).run_grounding(VIDEO, "gemini_video", 1, SimulatedApiFixture(GROUNDED))
        holder.release.set()
        self.assertEqual((row["apiStatus"], future.result(timeout=10)["apiStatus"]), ("success", "success"))


if __name__ == "__main__":
    unittest.main()
