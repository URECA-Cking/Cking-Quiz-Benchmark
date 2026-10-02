"""Shared approval tracking validation and its schema invariants (no schema library)."""

import json
import re
import unittest
from pathlib import Path

from src.approval_tracking import normalize_approved_by, require_valid_approval_tracking


ROOT = Path(__file__).resolve().parents[1]
VALID_AT = "2026-10-02T05:00:00.123456+00:00"
BAD_AT = ("2026-10-02T05:00:00Z", "2026-10-02T14:00:00+09:00", "2026-10-02T05:00:00.123+00:00",
          "2026-10-02 05:00:00+00:00", "٢٠٢٦-10-02T05:00:00+00:00",
          "2026-02-30T05:00:00+00:00", None, 123)
BAD_BY = (None, 123, "", " padded ", "a\nb", "a​b", "a\ud800b", "x" * 101)


def approved(**fields):
    return {"benchmarkType": "video_grounding", "contentTextApprovalStatus": "approved", **fields}


class ApprovalTrackingTest(unittest.TestCase):
    def test_legacy_and_valid_tracked_approvals_are_accepted(self):
        for row in (approved(), approved(approvedBy="Reviewer Kim", approvedAt=VALID_AT),
                    approved(approvedBy="Kim", approvedAt="2026-10-02T05:00:00+00:00"),
                    {"contentTextApprovalStatus": None}):
            with self.subTest(row=row):
                self.assertIsNone(require_valid_approval_tracking(row))

    def test_invalid_tracking_states_are_rejected(self):
        cases = [approved(approvedBy="Kim"), approved(approvedAt=VALID_AT),
                 {"contentTextApprovalStatus": None, "approvedBy": "Kim", "approvedAt": VALID_AT},
                 {"approvedBy": "Kim", "approvedAt": VALID_AT}]
        cases += [approved(approvedBy=value, approvedAt=VALID_AT) for value in BAD_BY]
        cases += [approved(approvedBy="Kim", approvedAt=value) for value in BAD_AT]
        for row in cases:
            with self.subTest(row=row):
                with self.assertRaisesRegex(ValueError, "approval tracking"):
                    require_valid_approval_tracking(row)

    def test_approved_by_normalization_contract(self):
        self.assertEqual(normalize_approved_by("  Reviewer Kim  "), "Reviewer Kim")
        self.assertEqual(normalize_approved_by("가" * 100), "가" * 100)
        for value in BAD_BY[:3] + ("reviewer\n", "a b", "a﻿b", "a\ud800b", "x" * 101):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "approved_by"):
                    normalize_approved_by(value)


def schema_accepts_tracking(schema, row):
    """Evaluate only the schema keywords that carry the approval tracking invariants."""
    for key, dependencies in schema["dependentRequired"].items():
        if key in row and any(dependency not in row for dependency in dependencies):
            return False
    for rule in schema["allOf"]:
        if all(key in row for key in rule["if"]["required"]):
            then = rule["then"]
            if any(key not in row for key in then["required"]):
                return False
            if any(key in row and row[key] != spec["const"]
                   for key, spec in then["properties"].items()):
                return False
    for field in ("approvedBy", "approvedAt"):
        if field not in row:
            continue
        spec, value = schema["properties"][field], row[field]
        if not isinstance(value, str):
            return False
        if not spec.get("minLength", 0) <= len(value) <= spec.get("maxLength", len(value)):
            return False
        match = re.search(spec["pattern"], value) if "pattern" in spec else True
        if match is None or (match is not True and match.end() != len(value)):
            return False
    return True


class ApprovalTrackingSchemaTest(unittest.TestCase):
    def setUp(self):
        self.schema = json.loads((ROOT / "docs" / "run-result.schema.json").read_text(encoding="utf-8"))

    def test_schema_accepts_legacy_and_canonical_tracked_approvals(self):
        for row in (approved(), approved(approvedBy="Kim", approvedAt="2026-10-02T05:00:00+00:00"),
                    approved(approvedBy="Kim", approvedAt=VALID_AT)):
            with self.subTest(row=row):
                self.assertTrue(schema_accepts_tracking(self.schema, row))
        self.assertNotIn("approvedBy", self.schema["required"])
        self.assertNotIn("approvedAt", self.schema["required"])

    def test_schema_rejects_tracking_invariant_violations(self):
        cases = (approved(approvedBy="Kim"), approved(approvedAt=VALID_AT),
                 {"benchmarkType": "video_grounding", "contentTextApprovalStatus": None,
                  "approvedBy": "Kim", "approvedAt": VALID_AT},
                 {"benchmarkType": "video_grounding", "approvedBy": "Kim", "approvedAt": VALID_AT},
                 approved(benchmarkType="quiz_generation", approvedBy="Kim", approvedAt=VALID_AT),
                 approved(approvedBy="Kim", approvedAt="2026-10-02T05:00:00Z"),
                 approved(approvedBy="Kim", approvedAt="2026-10-02T14:00:00+09:00"),
                 approved(approvedBy="Kim", approvedAt="2026-10-02T05:00:00.123+00:00"),
                 approved(approvedBy="Kim",
                          approvedAt="٢٠٢٦-10-02T05:00:00+00:00"))
        for row in cases:
            with self.subTest(row=row):
                self.assertFalse(schema_accepts_tracking(self.schema, row))


if __name__ == "__main__":
    unittest.main()
