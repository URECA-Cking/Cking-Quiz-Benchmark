import copy
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from src.pilot_runner import FixtureProvider, PilotRunner
from src.production_grounding import (PRODUCTION_GROUNDING_VALIDATION_VERSION, ProductionGroundingNotEligible,
                                      load_grounding_artifact, require_production_grounding_eligible,
                                      validate_production_grounding)
from tests.test_pilot_runner import (ALL_PASS, APPROVER, CONTENT, GROUNDING_V3, QUIZ, ROOT, VIDEO,
                                     SimulatedApiFixture)


DURATION = 211  # nasa-water-cycle-2019 durationSeconds in data/videos.jsonl
# Allowlisted Provider metadata of a completed live call, as the real adapter stores it.
COMPLETED_RAW = {"status": "completed", "provider": "gemini",
                 "usage": {"inputTokens": 100, "outputTokens": 50, "thinkingTokens": 0, "toolUseTokens": 0}}
VALIDATED_AT = "2026-10-08T00:00:00+00:00"
APPROVAL_FIELDS = ("contentTextApprovalStatus", "approvedBy", "approvedAt", "groundingReview")


class ProductionGroundingTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repository = Path(self.temp.name)
        (self.repository / "configs").mkdir()
        (self.repository / "data").mkdir()
        shutil.copyfile(ROOT / "configs" / "pilot-v2.yaml", self.repository / "configs" / "pilot-v2.yaml")
        shutil.copyfile(ROOT / "data" / "videos.jsonl", self.repository / "data" / "videos.jsonl")
        self.results = self.repository / "results"
        self.runner = PilotRunner(self.repository, self.results, self.repository / "configs" / "pilot-v2.yaml")

    def tearDown(self):
        self.temp.cleanup()

    def grounding(self, payload=GROUNDING_V3, repetition=1, provider=None):
        provider = provider or SimulatedApiFixture({"grounding": {"normalized": payload, "responseBody": COMPLETED_RAW}})
        return self.runner.run_grounding(VIDEO, "gemini_video", repetition, provider)

    def artifact(self, payload=GROUNDING_V3, repetition=1):
        """A fresh copy of the stored artifact; Pilot v2 allows one successful Grounding per repetition."""
        cache = self.__dict__.setdefault("_artifacts", {})
        if repetition not in cache:
            cache[repetition] = (payload, load_grounding_artifact(self.results, self.grounding(payload, repetition)["runId"]))
        self.assertIs(cache[repetition][0], payload)
        return copy.deepcopy(cache[repetition][1])

    def snapshot(self):
        return {str(path): path.read_bytes() for path in self.results.rglob("*") if path.is_file()}

    def assert_not_eligible(self, reason, artifact, record=None, video_id=VIDEO, duration=DURATION):
        if record is None:
            record = validate_production_grounding(artifact, duration, VALIDATED_AT)
        with self.assertRaises(ProductionGroundingNotEligible) as caught:
            require_production_grounding_eligible(artifact, record, video_id, duration)
        self.assertIn(reason, caught.exception.reasons)

    def assert_validation_fails(self, reason, artifact, duration=DURATION):
        record = validate_production_grounding(artifact, duration, VALIDATED_AT)
        self.assertEqual(record["status"], "fail")
        self.assertIn(reason, record["failureReasons"])
        self.assert_not_eligible(reason, artifact, record, duration=duration)

    # Positive

    def test_a_normal_v3_grounding_passes_and_is_eligible_without_human_approval(self):
        artifact = self.artifact()
        before = self.snapshot()
        record = validate_production_grounding(artifact, DURATION, VALIDATED_AT)
        self.assertEqual(record, {"status": "pass", "version": PRODUCTION_GROUNDING_VALIDATION_VERSION,
                                  "groundingRunId": artifact["row"]["runId"],
                                  "contentTextSha256": hashlib.sha256(CONTENT.encode("utf-8")).hexdigest(),
                                  "validatedAt": VALIDATED_AT, "failureReasons": []})
        self.assertEqual(require_production_grounding_eligible(artifact, record, VIDEO, DURATION), CONTENT)
        # Validation and eligibility only read; the Grounding stays unreviewed and unapproved.
        self.assertEqual(self.snapshot(), before)
        self.assertIsNone(artifact["row"]["contentTextApprovalStatus"])
        self.assertTrue(all(field not in artifact["row"] for field in APPROVAL_FIELDS[1:]))

    def test_unknown_duration_null_timestamps_and_an_explicit_empty_facts_array_are_not_rejected(self):
        # Minimum fact count and null timestamp rules are undecided policy, not this contract.
        nulls = dict(GROUNDING_V3, facts=[dict(GROUNDING_V3["facts"][0], timestampStart=None, timestampEnd=None)])
        for repetition, payload in ((1, nulls), (2, dict(GROUNDING_V3, facts=[]))):
            with self.subTest(facts=payload["facts"]):
                artifact = self.artifact(payload, repetition)
                record = validate_production_grounding(artifact, None, VALIDATED_AT)
                self.assertEqual(record["status"], "pass")
                self.assertEqual(require_production_grounding_eligible(artifact, record, VIDEO, None), CONTENT)

    def test_validated_at_defaults_to_a_utc_timestamp_the_gate_accepts(self):
        artifact = self.artifact()
        record = validate_production_grounding(artifact, DURATION)
        self.assertEqual(require_production_grounding_eligible(artifact, record, VIDEO, DURATION), CONTENT)

    # Semantic boundary: machine validation != factual approval

    def test_machine_validation_pass_is_not_factual_approval(self):
        # Structurally valid but untrue to the video: the contract has no way to notice, by design.
        untrue = "NASA measures rain on Mars every 30 seconds."
        payload = dict(GROUNDING_V3, contentText=untrue,
                       facts=[dict(GROUNDING_V3["facts"][0], fact="NASA measures rain on Mars")])
        artifact = self.artifact(payload)
        record = validate_production_grounding(artifact, DURATION, VALIDATED_AT)
        self.assertEqual(record["status"], "pass")
        self.assertEqual(require_production_grounding_eligible(artifact, record, VIDEO, DURATION), untrue)
        # The record carries no approval meaning and the Pilot Human gate still refuses this source.
        self.assertTrue(set(record).isdisjoint(APPROVAL_FIELDS))
        provider = SimulatedApiFixture({"quiz": {"normalized": dict(QUIZ, promptVersion="pilot-v2"),
                                                 "responseBody": COMPLETED_RAW}})
        with self.assertRaisesRegex(ValueError, "approved"):
            self.runner.run_quiz(VIDEO, "gemini-3.8-flash", 1, untrue, provider,
                                 source_grounding_run_id=artifact["row"]["runId"])
        self.assertEqual(provider.calls, [])

    def test_human_approval_is_not_a_substitute_for_machine_validation(self):
        row = self.grounding()
        self.runner.review_grounding(row["runId"], APPROVER, ALL_PASS)
        artifact = load_grounding_artifact(self.results, row["runId"])
        self.assertEqual(artifact["row"]["contentTextApprovalStatus"], "approved")
        with self.assertRaises(ProductionGroundingNotEligible) as caught:
            require_production_grounding_eligible(artifact, None, VIDEO, DURATION)
        self.assertEqual(caught.exception.reasons, ("missingValidationRecord",))

    # Negative: artifact contract

    def test_missing_facts_is_not_read_as_an_empty_array(self):
        payload = {"contentText": CONTENT}
        row = self.grounding(payload)
        self.assertEqual(row["apiStatus"], "success")  # Pilot behavior is unchanged
        artifact = load_grounding_artifact(self.results, row["runId"])
        self.assertNotIn("facts", artifact["evaluation"])
        self.assert_validation_fails("missingFacts", artifact)

    def test_facts_must_be_an_array(self):
        artifact = self.artifact()
        artifact["evaluation"]["facts"] = {"fact": "not a list"}
        self.assert_validation_fails("factsNotArray", artifact)

    def test_empty_or_missing_content_text_is_rejected(self):
        for content in ("", "   ", None):
            with self.subTest(content=content):
                artifact = self.artifact()
                artifact["evaluation"]["contentText"] = content
                self.assert_validation_fails("missingContentText", artifact)

    def test_malformed_facts_are_rejected(self):
        good = self.artifact()["evaluation"]["facts"][0]
        without = {key: value for key, value in good.items() if key != "timestampEndSeconds"}
        cases = {"not an object": "fact", "missing timestamp key": without,
                 "extra key": dict(good, factExists="pass"), "empty fact": dict(good, fact=" "),
                 "invalid evidenceType": dict(good, evidenceType="narration"),
                 "start after end": dict(good, timestampStartSeconds=80, timestampEndSeconds=70),
                 "negative": dict(good, timestampStartSeconds=-1), "bool": dict(good, timestampEndSeconds=True),
                 "string timestamp": dict(good, timestampStartSeconds="01:01"),
                 "beyond duration": dict(good, timestampEndSeconds=DURATION + 0.1)}
        for name, fact in cases.items():
            with self.subTest(case=name):
                artifact = self.artifact()
                artifact["evaluation"]["facts"] = [fact]
                self.assert_validation_fails("invalidFact", artifact)

    def test_evaluation_facts_must_match_the_result_row(self):
        artifact = self.artifact()
        artifact["row"]["groundingFacts"][0]["fact"] = "changed"
        self.assert_validation_fails("factsMismatch", artifact)

    def test_korean_content_text_passes(self):
        korean = "NASA는 30분마다 전 세계의 비와 눈을 측정합니다."
        artifact = self.artifact(dict(GROUNDING_V3, contentText=korean))
        record = validate_production_grounding(artifact, DURATION, VALIDATED_AT)
        self.assertEqual(record["contentTextSha256"], hashlib.sha256(korean.encode("utf-8")).hexdigest())
        self.assertEqual(require_production_grounding_eligible(artifact, record, VIDEO, DURATION), korean)

    def test_content_text_that_is_not_utf8_encodable_fails_without_leaking_encode_errors(self):
        artifact = self.artifact()
        record = validate_production_grounding(artifact, DURATION, VALIDATED_AT)
        # A JSON "\\ud800" escape loads as a lone surrogate: a str that UTF-8 cannot encode.
        artifact["evaluation"]["contentText"] = json.loads('"NASA \\ud800 rain"')
        failed = validate_production_grounding(artifact, DURATION, VALIDATED_AT)
        self.assertEqual((failed["status"], failed["contentTextSha256"]), ("fail", None))
        self.assertIn("contentTextNotUtf8", failed["failureReasons"])
        for stored in (record, failed):
            with self.subTest(record=stored["status"]):
                with self.assertRaises(ProductionGroundingNotEligible) as caught:
                    require_production_grounding_eligible(artifact, stored, VIDEO, DURATION)
                self.assertIn("contentTextNotUtf8", caught.exception.reasons)

    def test_content_text_must_match_the_row_hash(self):
        artifact = self.artifact()
        artifact["evaluation"]["contentText"] = CONTENT + " changed"
        self.assert_validation_fails("contentTextSha256Mismatch", artifact)
        artifact = self.artifact()
        del artifact["row"]["contentTextSha256"]
        self.assert_validation_fails("missingContentTextSha256", artifact)

    def test_technical_failure_and_incomplete_provider_results_are_rejected(self):
        error_row = self.grounding({"contentText": " "}, repetition=2)  # invalid_grounding_response
        self.assertEqual(error_row["apiStatus"], "error")
        self.assert_validation_fails("providerResultNotCompleted", load_grounding_artifact(self.results, error_row["runId"]))
        artifact = self.artifact()
        artifact["row"].update(apiStatus="error", errorCategory="timeout")
        self.assert_validation_fails("providerResultNotCompleted", artifact)
        artifact = self.artifact()
        artifact["raw"] = dict(COMPLETED_RAW, status="incomplete")
        self.assert_validation_fails("invalidRawMetadata", artifact)  # never a stored raw contract
        artifact = self.artifact()
        artifact["raw"] = None
        self.assert_validation_fails("providerResultNotCompleted", artifact)

    def test_raw_metadata_must_meet_the_canonical_storage_contract(self):
        usage = COMPLETED_RAW["usage"]
        cases = {"usage missing": {"status": "completed", "provider": "gemini"},
                 "usage wrong type": dict(COMPLETED_RAW, usage=[1, 2, 3, 4]),
                 "usage key missing": dict(COMPLETED_RAW, usage={k: v for k, v in usage.items() if k != "toolUseTokens"}),
                 "usage extra key": dict(COMPLETED_RAW, usage=dict(usage, cachedTokens=0)),
                 "usage value wrong type": dict(COMPLETED_RAW, usage=dict(usage, inputTokens="100")),
                 "usage value bool": dict(COMPLETED_RAW, usage=dict(usage, outputTokens=True)),
                 "usage value negative": dict(COMPLETED_RAW, usage=dict(usage, thinkingTokens=-1)),
                 "extra top-level key": dict(COMPLETED_RAW, responseText="..."),
                 "unknown provider": dict(COMPLETED_RAW, provider="azure"),
                 "malformed shape": ["completed", "gemini"]}
        for name, raw in cases.items():
            with self.subTest(case=name):
                # The Pilot storage contract rejects exactly the same metadata.
                with self.assertRaises(ValueError):
                    PilotRunner._validate_raw_metadata(raw)
                artifact = self.artifact()
                artifact["raw"] = raw
                self.assert_validation_fails("invalidRawMetadata", artifact)
        # Canonical but from another Provider: valid storage, not a Gemini Grounding.
        PilotRunner._validate_raw_metadata(dict(COMPLETED_RAW, provider="openai"))
        artifact = self.artifact()
        artifact["raw"] = dict(COMPLETED_RAW, provider="openai")
        self.assert_validation_fails("unsupportedGroundingArtifact", artifact)

    def test_source_model_must_be_present_and_supported(self):
        cases = {"missing": None, "wrong type": 3, "empty": " ", "list": ["gemini-3.8-flash"]}
        for name, model in cases.items():
            with self.subTest(case=name):
                artifact = self.artifact()
                if model is None:
                    del artifact["row"]["model"]
                else:
                    artifact["row"]["model"] = model
                self.assert_validation_fails("missingModel", artifact)
        for model in ("gpt-5.4-mini", "gemini-3.8-pro"):
            with self.subTest(model=model):
                artifact = self.artifact()
                artifact["row"]["model"] = model
                self.assert_validation_fails("unsupportedGroundingArtifact", artifact)

    def test_validated_at_must_be_a_real_canonical_utc_timestamp(self):
        artifact = self.artifact()
        for valid in (VALIDATED_AT, "2026-10-08T12:34:56.123456+00:00", "2028-02-29T23:59:59+00:00"):
            with self.subTest(valid=valid):
                record = validate_production_grounding(artifact, DURATION, valid)
                self.assertEqual(require_production_grounding_eligible(artifact, record, VIDEO, DURATION), CONTENT)
        record = validate_production_grounding(artifact, DURATION, VALIDATED_AT)
        for invalid in ("2026-99-99T25:61:61+00:00", "2026-02-30T00:00:00+00:00", "2027-02-29T00:00:00+00:00",
                        "2026-10-08T24:00:00+00:00", "2026-10-08T00:00:00+09:00", "2026-10-08T00:00:00Z",
                        "2026-10-08T00:00:00", "2026-10-08", 0):
            with self.subTest(invalid=invalid):
                # A caller cannot build a record with it, and a stored record carrying it is refused.
                with self.assertRaises(ValueError):
                    validate_production_grounding(artifact, DURATION, invalid)
                self.assert_not_eligible("invalidValidationRecord", artifact, dict(record, validatedAt=invalid))

    def test_a_fixture_grounding_is_never_a_live_production_artifact(self):
        row = self.grounding(provider=FixtureProvider({"grounding": {"raw": GROUNDING_V3}}))
        self.assertEqual(row["apiStatus"], "not_run")
        self.assert_validation_fails("providerResultNotCompleted", load_grounding_artifact(self.results, row["runId"]))

    def test_unsupported_grounding_artifacts_are_rejected(self):
        for field, value in (("promptVersion", "video-grounding-v2"), ("promptVersion", ["video-grounding-v3"]),
                             ("method", "authorized_transcript"), ("benchmarkType", "quiz_generation")):
            with self.subTest(field=field, value=value):
                artifact = self.artifact()
                artifact["row"][field] = value
                self.assert_validation_fails("unsupportedGroundingArtifact", artifact)

    def test_invalid_duration_metadata_fails_closed(self):
        for duration in (0, -1, True, "211", float("nan"), float("inf")):
            with self.subTest(duration=duration):
                self.assert_validation_fails("invalidDurationMetadata", self.artifact(), duration)

    # Negative: validation record and source binding

    def test_missing_failed_not_run_or_malformed_records_are_not_eligible(self):
        artifact = self.artifact()
        record = validate_production_grounding(artifact, DURATION, VALIDATED_AT)
        with self.assertRaises(ProductionGroundingNotEligible) as caught:
            require_production_grounding_eligible(artifact, None, VIDEO, DURATION)
        self.assertEqual(caught.exception.reasons, ("missingValidationRecord",))
        cases = [("validationFailed", dict(record, status="fail", failureReasons=["invalidFact"])),
                 ("validationNotRun", dict(record, status="not_run"))]
        malformed = [dict(record, status="approved"), dict(record, failureReasons=["invalidFact"]),
                     dict(record, status="fail"), dict(record, validatedAt="2026-10-08"),
                     {key: value for key, value in record.items() if key != "validatedAt"},
                     dict(record, approvedBy=APPROVER), "pass"]
        for reason, bad in cases + [("invalidValidationRecord", bad) for bad in malformed]:
            with self.subTest(reason=reason, record=bad):
                self.assert_not_eligible(reason, artifact, bad)

    def test_unsupported_validation_version_fails_closed(self):
        artifact = self.artifact()
        record = validate_production_grounding(artifact, DURATION, VALIDATED_AT)
        for version in ("production-grounding-validation-v2", "video-grounding-v3", None):
            with self.subTest(version=version):
                self.assert_not_eligible("unsupportedValidationVersion", artifact, dict(record, version=version))

    def test_record_must_be_bound_to_the_exact_content_text_and_run(self):
        artifact = self.artifact()
        record = validate_production_grounding(artifact, DURATION, VALIDATED_AT)
        self.assert_not_eligible("validationRecordContentTextSha256Mismatch", artifact,
                                 dict(record, contentTextSha256=hashlib.sha256(b"other").hexdigest()))
        self.assert_not_eligible("validationRecordRunMismatch", artifact, dict(record, groundingRunId="0" * 32))
        # A pass recorded for another run's identical text does not transfer to this run.
        other = self.artifact(repetition=2)
        self.assertEqual(other["row"]["contentTextSha256"], record["contentTextSha256"])
        self.assert_not_eligible("validationRecordRunMismatch", other, record)

    def test_source_video_must_match(self):
        artifact = self.artifact()
        self.assert_not_eligible("sourceVideoMismatch", artifact, video_id="kari-microgravity-2024")

    def test_a_stored_pass_does_not_override_a_changed_artifact(self):
        artifact = self.artifact()
        record = validate_production_grounding(artifact, DURATION, VALIDATED_AT)
        changed = copy.deepcopy(artifact)
        changed["evaluation"]["facts"][0]["evidenceType"] = "narration"
        self.assert_not_eligible("invalidFact", changed, record)

    # Negative: storage linkage

    def test_missing_ambiguous_or_corrupted_storage_fails_closed(self):
        run_id = self.grounding()["runId"]
        summary = self.results / "video-grounding.jsonl"
        cases = {"invalidRunId": lambda: load_grounding_artifact(self.results, "not-a-run-id"),
                 "missingResultRow": lambda: load_grounding_artifact(self.results, "f" * 32)}
        for reason, load in cases.items():
            with self.subTest(reason=reason), self.assertRaises(ProductionGroundingNotEligible) as caught:
                load()
            self.assertEqual(caught.exception.reasons, (reason,))
        for name, reason in (("evaluation", "missingEvaluation"), ("raw", "missingRawMetadata")):
            path = self.results / name / (run_id + ".json")
            original = path.read_bytes()
            for content, expected in ((None, reason), (b"{", "corruptedArtifact"), (b'{"contentText": NaN}', "corruptedArtifact")):
                with self.subTest(file=name, content=content):
                    path.unlink()
                    if content is not None:
                        path.write_bytes(content)
                    with self.assertRaises(ProductionGroundingNotEligible) as caught:
                        load_grounding_artifact(self.results, run_id)
                    self.assertEqual(caught.exception.reasons, (expected,))
                    path.unlink(missing_ok=True)
                    path.write_bytes(original)
        line = [line for line in summary.read_text(encoding="utf-8").splitlines() if run_id in line][0]
        with summary.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        with self.assertRaises(ProductionGroundingNotEligible) as caught:
            load_grounding_artifact(self.results, run_id)
        self.assertEqual(caught.exception.reasons, ("ambiguousResultRow",))
        summary.write_text("{broken\n", encoding="utf-8")
        with self.assertRaises(ProductionGroundingNotEligible) as caught:
            load_grounding_artifact(self.results, run_id)
        self.assertEqual(caught.exception.reasons, ("corruptedArtifact",))


if __name__ == "__main__":
    unittest.main()
