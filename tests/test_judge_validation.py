import json
import unittest

from src.judge_contract import POINTWISE_ITEMS
from src.judge_validation import JudgeOutputError, validate_pairwise, validate_pointwise
from tests.judge_fixtures import pairwise_output, pointwise_output


SETS = {"SET_1": [0, 1, 2], "SET_2": [0, 1, 2]}


class JudgeValidationTest(unittest.TestCase):
    def assert_category(self, category, function, *args):
        with self.assertRaises(JudgeOutputError) as caught:
            function(*args)
        self.assertEqual(caught.exception.category, category)

    def test_valid_pointwise_output_regardless_of_verdict_values(self):
        for verdict in ("pass", "fail", "uncertain"):
            result = validate_pointwise(pointwise_output([0, 2], verdict), [0, 2])
            self.assertEqual(set(result), {0, 2})
            self.assertEqual(result[2]["koreanQuality"]["verdict"], verdict)

    def test_pointwise_parse_and_schema_errors(self):
        self.assert_category("parse_error", validate_pointwise, "not json", [0])
        self.assert_category("parse_error", validate_pointwise, '{"questions": NaN}', [0])
        data = json.loads(pointwise_output([0]))
        del data["questions"][0]["uniqueAnswer"]
        self.assert_category("schema_error", validate_pointwise, json.dumps(data), [0])
        data = json.loads(pointwise_output([0]))
        data["questions"][0]["extra"] = 1
        self.assert_category("schema_error", validate_pointwise, json.dumps(data), [0])
        data = json.loads(pointwise_output([0]))
        data["questions"][0]["koreanQuality"]["verdict"] = "PASS"
        self.assert_category("schema_error", validate_pointwise, json.dumps(data), [0])
        for index in (True, 0.0):
            data = json.loads(pointwise_output([0]))
            data["questions"][0]["questionIndex"] = index
            self.assert_category("schema_error", validate_pointwise, json.dumps(data), [0])

    def test_pointwise_partial_duplicate_or_extra_questions_are_semantic_errors(self):
        self.assert_category("semantic_error", validate_pointwise, pointwise_output([0, 1]), [0, 1, 2])
        self.assert_category("semantic_error", validate_pointwise, pointwise_output([0, 0, 1]), [0, 1])
        self.assert_category("semantic_error", validate_pointwise, pointwise_output([0, 1, 5]), [0, 1])
        data = json.loads(pointwise_output([0]))
        data["questions"][0][POINTWISE_ITEMS[0]]["reason"] = "  "
        self.assert_category("semantic_error", validate_pointwise, json.dumps(data), [0])

    def test_valid_pairwise_output(self):
        result = validate_pairwise(pairwise_output([1], [], {"coverage": "SET_2"}), SETS)
        self.assertEqual([item["questionIndex"] for item in result["defectiveQuestions"]["SET_1"]], [1])
        self.assertEqual(result["quality"]["coverage"]["winner"], "SET_2")

    def test_pairwise_semantic_errors(self):
        self.assert_category("semantic_error", validate_pairwise, pairwise_output([3]), SETS)
        self.assert_category("semantic_error", validate_pairwise, pairwise_output([1, 1]), SETS)
        for types in ([], ["notUnique", "notUnique"]):
            data = json.loads(pairwise_output([1]))
            data["SET_1"]["defectiveQuestions"][0]["defectTypes"] = types
            self.assert_category("semantic_error", validate_pairwise, json.dumps(data), SETS)
        data = json.loads(pairwise_output())
        data["quality"]["clarity"]["reason"] = ""
        self.assert_category("semantic_error", validate_pairwise, json.dumps(data), SETS)

    def test_duplicate_json_keys_are_rejected_not_last_wins(self):
        text = pointwise_output([0])
        quality_source, defect_source = pairwise_output(), pairwise_output([1])
        cases = (
            ("rubric", text, text.replace('"koreanQuality": {"verdict": "pass", "reason": "근거"}',
                                          '"koreanQuality": {"verdict": "pass", "reason": "근거"}, '
                                          '"koreanQuality": {"verdict": "fail", "reason": "근거"}', 1),
             lambda value: validate_pointwise(value, [0])),
            ("field", text, text.replace('"questionIndex": 0', '"questionIndex": 0, "questionIndex": 0', 1),
             lambda value: validate_pointwise(value, [0])),
            ("quality", quality_source,
             quality_source.replace('"coverage": {"winner": "TIE", "reason": "근거"}',
                                    '"coverage": {"winner": "TIE", "reason": "근거"}, '
                                    '"coverage": {"winner": "SET_1", "reason": "근거"}', 1),
             lambda value: validate_pairwise(value, SETS)),
            ("defect", defect_source,
             defect_source.replace('"defectTypes": ["answerIncorrect"]',
                                   '"defectTypes": ["answerIncorrect"], "defectTypes": ["notUnique"]', 1),
             lambda value: validate_pairwise(value, SETS)))
        for name, source, duplicated, check in cases:
            with self.subTest(case=name):
                self.assertNotEqual(duplicated, source)  # the duplicate key was really injected
                self.assert_category("parse_error", check, duplicated)
        # Ordinary JSON still parses.
        self.assertIn(0, validate_pointwise(text, [0]))

    def test_pairwise_schema_errors(self):
        data = json.loads(pairwise_output())
        del data["quality"]["coverage"]
        self.assert_category("schema_error", validate_pairwise, json.dumps(data), SETS)
        data = json.loads(pairwise_output([1]))
        data["SET_1"]["defectiveQuestions"][0]["defectTypes"] = ["styleIssue"]
        self.assert_category("schema_error", validate_pairwise, json.dumps(data), SETS)
        data = json.loads(pairwise_output())
        data["quality"]["clarity"]["winner"] = "A"
        self.assert_category("schema_error", validate_pairwise, json.dumps(data), SETS)


if __name__ == "__main__":
    unittest.main()
