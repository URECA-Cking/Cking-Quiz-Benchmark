"""Shared Human Evaluation value validation and its schema invariants (no schema library)."""

import json
import unittest
from pathlib import Path

from src.human_evaluation import (GROUNDING_FACT_KEYS, QUESTION_REVIEW_KEYS, REVIEW_VALUES,
                                  require_valid_human_evaluation)


ROOT = Path(__file__).resolve().parents[1]
FACT = {"fact": "NASA measures rainfall", "evidenceType": "speech",
        "evidence": "global rain and snow", "timestampStartSeconds": 61,
        "timestampEndSeconds": 73, "factExists": None, "evidenceTypeCorrect": None,
        "timestampAccurate": None, "reviewNote": None}
REVIEW = {"questionIndex": 0, "answerAccuracy": None, "uniqueAnswer": None,
          "evidenceSupportsAnswer": None, "videoGrounding": None, "koreanQuality": None,
          "hallucination": None, "reviewNote": None}
BAD_VALUES = ("Pass", "PASS", "pass ", "fial", "", True, False, 0, 1, 1.0, [], {})


class HumanEvaluationTest(unittest.TestCase):
    def test_allowlists_match_run_result_schema(self):
        schema = json.loads((ROOT / "docs" / "run-result.schema.json").read_text(encoding="utf-8"))
        definitions = schema["$defs"]
        self.assertEqual(set(REVIEW_VALUES), set(definitions["nullableReview"]["enum"]))
        self.assertEqual(GROUNDING_FACT_KEYS, set(definitions["groundingFact"]["properties"]))
        self.assertEqual(QUESTION_REVIEW_KEYS, set(definitions["questionReview"]["properties"]))
        self.assertEqual(definitions["questionReview"]["required"], ["questionIndex"])
        for key in ("omission", "hallucination", "videoGrounding"):
            self.assertEqual(schema["properties"][key], {"$ref": "#/$defs/nullableReview"})

    def test_valid_and_legacy_evaluations_are_accepted(self):
        rows = [{}, {"groundingFacts": []}, {"questionReviews": []},
                {"groundingFacts": [FACT], "omission": None, "hallucination": None},
                {"groundingFacts": [{"fact": "partial legacy fact"}]},
                {"questionReviews": [REVIEW, {**REVIEW, "questionIndex": 7}]},
                {"questionReviews": [{"questionIndex": 0}]},
                {"questionReviews": [{**REVIEW, "reviewNote": "검토 메모"}]},
                {"groundingFacts": [{**FACT, "reviewNote": "note"}]}]
        for value in REVIEW_VALUES:
            rows += [{"omission": value, "hallucination": value, "videoGrounding": value},
                     {"groundingFacts": [{**FACT, "factExists": value,
                                          "evidenceTypeCorrect": value,
                                          "timestampAccurate": value}]},
                     {"questionReviews": [{**REVIEW, **{key: value for key in REVIEW
                                                        if key not in ("questionIndex",
                                                                       "reviewNote")}}]}]
        for row in rows:
            with self.subTest(row=row):
                self.assertIsNone(require_valid_human_evaluation(row))

    def test_invalid_review_values_are_rejected(self):
        rows = []
        for value in BAD_VALUES:
            rows += [{"omission": value}, {"hallucination": value}, {"videoGrounding": value},
                     {"groundingFacts": [{**FACT, "factExists": value}]},
                     {"groundingFacts": [{**FACT, "timestampAccurate": value}]},
                     {"questionReviews": [{**REVIEW, "answerAccuracy": value}]},
                     {"questionReviews": [{**REVIEW, "koreanQuality": value}]}]
        for row in rows:
            with self.subTest(row=row):
                with self.assertRaisesRegex(ValueError, "Human evaluation"):
                    require_valid_human_evaluation(row)

    def test_invalid_review_structure_is_rejected(self):
        rows = [{"groundingFacts": None}, {"questionReviews": None},
                {"groundingFacts": {}}, {"questionReviews": "[]"},
                {"groundingFacts": [None]}, {"questionReviews": [[]]},
                {"groundingFacts": [{**FACT, "factExist": "pass"}]},
                {"questionReviews": [{**REVIEW, "answerAccuraccy": "pass"}]},
                {"groundingFacts": [{**FACT, "reviewNote": 1}]},
                {"questionReviews": [{**REVIEW, "reviewNote": False}]},
                {"questionReviews": [{"answerAccuracy": "pass"}]},
                {"questionReviews": [{**REVIEW, "questionIndex": "0"}]},
                {"questionReviews": [{**REVIEW, "questionIndex": True}]},
                {"questionReviews": [{**REVIEW, "questionIndex": -1}]},
                {"questionReviews": [{**REVIEW, "questionIndex": 0.0}]},
                {"questionReviews": [{**REVIEW, "questionIndex": None}]}]
        for row in rows:
            with self.subTest(row=row):
                with self.assertRaisesRegex(ValueError, "Human evaluation"):
                    require_valid_human_evaluation(row)

    def test_error_message_does_not_echo_stored_values(self):
        with self.assertRaises(ValueError) as caught:
            require_valid_human_evaluation({"questionReviews": [{**REVIEW,
                                                                 "reviewNote": 123456789}]})
        self.assertNotIn("123456789", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
