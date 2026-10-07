import itertools
import unittest

from src import judge_aggregation as agg


def canonical(defects_a=(), defects_b=(), **quality):
    items = {item: quality.get(item, "TIE") for item in ("clarity", "koreanQuality", "distractorQuality", "coverage")}
    return {"defectiveQuestions": {"A": [{"questionIndex": index, "defectTypes": ["answerIncorrect"]}
                                         for index in defects_a],
                                   "B": [{"questionIndex": index, "defectTypes": ["answerIncorrect"]}
                                         for index in defects_b]},
            "quality": items}


def resolved(preference):
    return {"status": agg.RESOLVED, "preference": preference}


class PairwiseCallTest(unittest.TestCase):
    def test_canonicalization_follows_order(self):
        validated = {"defectiveQuestions": {"SET_1": [{"questionIndex": 0}], "SET_2": []},
                     "quality": {"clarity": {"winner": "SET_1"}, "koreanQuality": {"winner": "SET_2"},
                                 "distractorQuality": {"winner": "TIE"}, "coverage": {"winner": "SET_1"}}}
        ab = agg.canonicalize_pairwise("AB", validated)
        ba = agg.canonicalize_pairwise("BA", validated)
        self.assertEqual((len(ab["defectiveQuestions"]["A"]), len(ba["defectiveQuestions"]["B"])), (1, 1))
        self.assertEqual(ab["quality"], {"clarity": "A", "koreanQuality": "B", "distractorQuality": "TIE",
                                         "coverage": "A"})
        self.assertEqual(ba["quality"]["clarity"], "B")

    def test_validity_counts_questions_and_quality_cannot_override(self):
        many_defects_one_question = canonical()
        many_defects_one_question["defectiveQuestions"]["A"] = [
            {"questionIndex": 0, "defectTypes": ["answerIncorrect", "notUnique", "unfaithful"]}]
        self.assertEqual(agg.pairwise_call_final(many_defects_one_question), ("B", "validity"))
        quality_all_a = canonical([0], [], clarity="A", koreanQuality="A", distractorQuality="A", coverage="A")
        self.assertEqual(agg.pairwise_call_final(quality_all_a), ("B", "validity"))
        self.assertEqual(agg.pairwise_call_final(canonical([0], [1], clarity="A")), ("A", "quality"))
        self.assertEqual(agg.pairwise_call_final(canonical(clarity="A", coverage="B")), ("TIE", "tie"))
        self.assertEqual(agg.pairwise_call_final(canonical(koreanQuality="B")), ("B", "quality"))


class OrderAggregationTest(unittest.TestCase):
    def test_all_order_combinations(self):
        expected = {("A", "A"): (agg.RESOLVED, "A", "full"), ("B", "B"): (agg.RESOLVED, "B", "full"),
                    ("TIE", "TIE"): (agg.RESOLVED, "TIE", "full"),
                    ("A", "B"): (agg.INCONSISTENT, None, "contradictory"),
                    ("B", "A"): (agg.INCONSISTENT, None, "contradictory"),
                    ("A", "TIE"): (agg.RESOLVED, "TIE", "partial"), ("TIE", "A"): (agg.RESOLVED, "TIE", "partial"),
                    ("B", "TIE"): (agg.RESOLVED, "TIE", "partial"), ("TIE", "B"): (agg.RESOLVED, "TIE", "partial")}
        for (ab, ba), (status, preference, agreement) in expected.items():
            result = agg.merge_orders(ab, ba)
            self.assertEqual((result["status"], result["preference"], result["orderAgreement"]),
                             (status, preference, agreement), (ab, ba))
        for ab, ba in ((None, "A"), ("TIE", None), (None, None)):
            self.assertEqual(agg.merge_orders(ab, ba),
                             {"status": agg.INCOMPLETE, "preference": None, "orderAgreement": None})


class JudgeAggregationTest(unittest.TestCase):
    STATES = {"A": resolved("A"), "B": resolved("B"), "TIE": resolved("TIE"),
              "INCONSISTENT": {"status": agg.INCONSISTENT, "preference": None},
              "INCOMPLETE": {"status": agg.INCOMPLETE, "preference": None}}

    # Explicit oracle for every ordered Judge pair (Judge 1 result, Judge 2 result).
    R, INC, CON, DIS = "RESOLVED", "INCOMPLETE", "INCONSISTENT", "JUDGE_DISAGREEMENT"
    EXPECTED = {
        ("A", "A"): (R, "A", "full"), ("A", "B"): (DIS, None, "contradictory"),
        ("A", "TIE"): (R, "TIE", "partial"), ("A", "INCONSISTENT"): (CON, None, "notComparable"),
        ("A", "INCOMPLETE"): (INC, None, "notComparable"),
        ("B", "A"): (DIS, None, "contradictory"), ("B", "B"): (R, "B", "full"),
        ("B", "TIE"): (R, "TIE", "partial"), ("B", "INCONSISTENT"): (CON, None, "notComparable"),
        ("B", "INCOMPLETE"): (INC, None, "notComparable"),
        ("TIE", "A"): (R, "TIE", "partial"), ("TIE", "B"): (R, "TIE", "partial"),
        ("TIE", "TIE"): (R, "TIE", "full"), ("TIE", "INCONSISTENT"): (CON, None, "notComparable"),
        ("TIE", "INCOMPLETE"): (INC, None, "notComparable"),
        ("INCONSISTENT", "A"): (CON, None, "notComparable"), ("INCONSISTENT", "B"): (CON, None, "notComparable"),
        ("INCONSISTENT", "TIE"): (CON, None, "notComparable"),
        ("INCONSISTENT", "INCONSISTENT"): (CON, None, "notComparable"),
        ("INCONSISTENT", "INCOMPLETE"): (INC, None, "notComparable"),
        ("INCOMPLETE", "A"): (INC, None, "notComparable"), ("INCOMPLETE", "B"): (INC, None, "notComparable"),
        ("INCOMPLETE", "TIE"): (INC, None, "notComparable"),
        ("INCOMPLETE", "INCONSISTENT"): (INC, None, "notComparable"),
        ("INCOMPLETE", "INCOMPLETE"): (INC, None, "notComparable"),
    }

    def test_all_25_judge_combinations_and_symmetry(self):
        self.assertEqual(set(self.EXPECTED), set(itertools.product(self.STATES, repeat=2)))
        for (left, right), expected in self.EXPECTED.items():
            result = agg.merge_judges(self.STATES[left], self.STATES[right])
            self.assertEqual((result["status"], result["preference"], result["interJudgeAgreement"]),
                             expected, (left, right))
            mirrored = agg.merge_judges(self.STATES[right], self.STATES[left])
            self.assertEqual(result, mirrored)
            if result["status"] != agg.RESOLVED:
                self.assertIsNone(result["preference"])


class HumanComparisonTest(unittest.TestCase):
    def test_uncertain_and_missing_are_never_agreement(self):
        self.assertEqual(agg.compare_verdicts("pass", "pass"), "AGREE")
        self.assertEqual(agg.compare_verdicts("pass", "fail"), "DISAGREE")
        self.assertEqual(agg.compare_verdicts("uncertain", "pass"), "NOT_COMPARABLE_UNCERTAIN")
        self.assertEqual(agg.compare_verdicts("fail", "uncertain"), "NOT_COMPARABLE_UNCERTAIN")
        self.assertEqual(agg.compare_verdicts("pass", None), "MISSING")

    def test_pointwise_mapping(self):
        review = {"answerAccuracy": "fail", "hallucination": "pass", "koreanQuality": "pass"}
        self.assertEqual(agg.compare_pointwise_human("textAnswerCorrect", "fail", review),
                         {"humanField": "answerAccuracy", "relation": "comparableDifferentBasis", "outcome": "AGREE"})
        self.assertEqual(agg.compare_pointwise_human("textFaithfulness", "fail", review)["relation"], "referenceOnly")
        self.assertEqual(agg.compare_pointwise_human("koreanQuality", "pass", review)["relation"], "direct")
        for item in ("questionClarity", "distractorQuality"):
            self.assertIsNone(agg.compare_pointwise_human(item, "pass", review)["humanField"])

    def test_human_derived_validity(self):
        clean = [{"questionIndex": i, "answerAccuracy": "pass", "uniqueAnswer": "pass",
                  "evidenceSupportsAnswer": "pass"} for i in range(3)]
        double_fail = [dict(clean[0], answerAccuracy="fail", uniqueAnswer="fail")] + clean[1:]
        result = agg.human_derived_validity(double_fail, [0, 1, 2], clean, [0, 1, 2])
        self.assertEqual(result, {"result": "B", "defectiveQuestions": {"A": 1, "B": 0}})
        self.assertEqual(agg.human_derived_validity(double_fail, [0, 1, 2], double_fail, [0, 1, 2])["result"],
                         "NO_VALIDITY_DIFFERENCE")
        uncertain = [dict(clean[0], evidenceSupportsAnswer="uncertain")] + clean[1:]
        self.assertEqual(agg.human_derived_validity(uncertain, [0, 1, 2], clean, [0, 1, 2])["result"], "UNDETERMINED")
        # videoGrounding / hallucination / koreanQuality never make a Human-derived defect.
        other = [dict(clean[0], videoGrounding="fail", hallucination="fail", koreanQuality="fail")] + clean[1:]
        self.assertEqual(agg.human_derived_validity(other, [0, 1, 2], clean, [0, 1, 2])["result"],
                         "NO_VALIDITY_DIFFERENCE")

    def test_h2a_comparability(self):
        validity = ["validity"] * 4
        self.assertEqual(agg.h2a_comparison(resolved("A"), validity, "A")["outcome"], "AGREE")
        self.assertEqual(agg.h2a_comparison(resolved("A"), validity, "NO_VALIDITY_DIFFERENCE")["outcome"], "DISAGREE")
        self.assertEqual(agg.h2a_comparison(resolved("A"), ["validity"] * 3 + ["quality"], "A")["comparability"],
                         "NOT_COMPARABLE_DECISION_BASIS")
        self.assertEqual(agg.h2a_comparison(resolved("TIE"), ["tie"] * 4, "NO_VALIDITY_DIFFERENCE")["comparability"],
                         "NOT_COMPARABLE_DECISION_BASIS")
        for status in (agg.JUDGE_DISAGREEMENT, agg.INCONSISTENT, agg.INCOMPLETE):
            result = agg.h2a_comparison({"status": status, "preference": None}, validity, "A")
            self.assertEqual((result["comparability"], result["reason"]), ("NOT_COMPARABLE_STATUS", status))
        self.assertEqual(agg.h2a_comparison(resolved("A"), validity, "UNDETERMINED")["comparability"],
                         "NOT_COMPARABLE_UNDETERMINED")


if __name__ == "__main__":
    unittest.main()
