import json
import tempfile
import unittest
from pathlib import Path

from src.judge_eligibility import SourceIntegrityError, build_plan, eligibility_records
from tests.judge_fixtures import MODELS, build_synthetic_pilot


MARS = "nasa-mars-organics-2025"


class JudgeEligibilityTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.results = build_synthetic_pilot(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def rows(self, name):
        return [json.loads(line) for line in (self.results / name).read_text(encoding="utf-8").splitlines()]

    def write_rows(self, name, rows):
        (self.results / name).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def quiz_evaluation(self, video_id, condition):
        row = next(row for row in self.rows("quiz-generation.jsonl")
                   if row["videoId"] == video_id and row["model"] == MODELS[condition])
        return self.results / "evaluation" / (row["runId"] + ".json")

    def test_current_pilot_shape_selects_8_sets_22_questions_and_3_pairs(self):
        plan = build_plan(self.results, MODELS)
        eligible = [unit for unit in plan["pointwise"] if unit["status"] == "eligible"]
        self.assertEqual(len(eligible), 8)
        self.assertEqual(sum(len(unit["eligibleQuestionIndexes"]) for unit in eligible), 22)
        mars_b = next(unit for unit in eligible if unit["videoId"] == MARS and unit["condition"] == "B")
        self.assertEqual(mars_b["eligibleQuestionIndexes"], [0])
        pairs = {pair["videoId"]: pair for pair in plan["pairwise"]}
        self.assertEqual(sorted(video for video, pair in pairs.items() if pair["status"] == "eligible"),
                         ["kari-microgravity-2024", "nasa-methane-2020", "nasa-water-cycle-2019"])
        self.assertEqual((pairs[MARS]["status"], pairs[MARS]["exclusionReason"], pairs[MARS]["questionCounts"]),
                         ("excluded", "questionCountMismatch", {"A": 3, "B": 1}))

    def test_records_hold_no_content_or_human_values_and_no_direct_quiz(self):
        records = eligibility_records(build_plan(self.results, MODELS))
        text = json.dumps(records, ensure_ascii=False)
        for hidden in ("contentText 본문", "questionReviews", "answerAccuracy", "gemini_direct_quiz"):
            self.assertNotIn(hidden, text)
        mars = next(record for record in records if record.get("pairId") == MARS + ":1")
        self.assertEqual(mars["status"], "excluded")

    def test_structural_exclusion_keeps_original_indexes_and_excludes_pair(self):
        path = self.quiz_evaluation("nasa-water-cycle-2019", "A")
        quiz = json.loads(path.read_text(encoding="utf-8"))
        quiz["questions"][1]["correctOptionIndex"] = 9
        path.write_text(json.dumps(quiz), encoding="utf-8")
        plan = build_plan(self.results, MODELS)
        unit = next(unit for unit in plan["pointwise"]
                    if unit["videoId"] == "nasa-water-cycle-2019" and unit["condition"] == "A")
        self.assertEqual(unit["eligibleQuestionIndexes"], [0, 2])
        self.assertEqual(unit["excludedQuestions"], [{"questionIndex": 1, "reason": "invalidCorrectOptionIndex"}])
        pair = next(pair for pair in plan["pairwise"] if pair["videoId"] == "nasa-water-cycle-2019")
        self.assertEqual((pair["status"], pair["exclusionReason"]), ("excluded", "excludedQuestionPresent"))

    def test_validator_contract_issues_do_not_exclude_questions(self):
        path = self.quiz_evaluation("nasa-water-cycle-2019", "B")
        quiz = json.loads(path.read_text(encoding="utf-8"))
        quiz["questions"][0]["options"] = ["가", "가", "다"]
        quiz["questions"][1]["sourceEvidence"] = "contentText에 없는 문장"
        path.write_text(json.dumps(quiz), encoding="utf-8")
        plan = build_plan(self.results, MODELS)
        unit = next(unit for unit in plan["pointwise"]
                    if unit["videoId"] == "nasa-water-cycle-2019" and unit["condition"] == "B")
        self.assertEqual((unit["eligibleQuestionIndexes"], unit["excludedQuestions"]), ([0, 1, 2], []))

    def test_parse_failure_is_not_pointwise_eligible(self):
        rows = self.rows("quiz-generation.jsonl")
        rows[0]["parseStatus"] = "fail"
        self.write_rows("quiz-generation.jsonl", rows)
        plan = build_plan(self.results, MODELS)
        unit = next(unit for unit in plan["pointwise"] if unit["quizRunId"] == rows[0]["runId"])
        self.assertEqual((unit["status"], unit["exclusionReason"]), ("excluded", "notParsed"))

    def test_grounding_and_quiz_repetition_must_match(self):
        self.assertEqual(len(build_plan(self.results, MODELS)["pointwise"]), 8)
        grounding = self.rows("video-grounding.jsonl")
        self.write_rows("video-grounding.jsonl", [dict(row, repetition=2) if row.get("apiStatus") == "success"
                                                  and row["videoId"] == "nasa-water-cycle-2019" else row
                                                  for row in grounding])
        with self.assertRaisesRegex(SourceIntegrityError, "repetition"):
            build_plan(self.results, MODELS)

    def test_source_integrity_failures_stop_planning(self):
        cases = []
        rows = self.rows("quiz-generation.jsonl")
        cases.append(("quiz-generation.jsonl", [dict(rows[0], contentTextSha256="0" * 64)] + rows[1:]))
        cases.append(("quiz-generation.jsonl", rows + [dict(rows[0], runId="f" * 32, attempt=2)]))
        cases.append(("quiz-generation.jsonl", [dict(rows[0], promptVersion="pilot-v2")] + rows[1:]))
        grounding = self.rows("video-grounding.jsonl")
        cases.append(("video-grounding.jsonl",
                      [dict(row, contentTextApprovalStatus=None) if row.get("apiStatus") == "success" else row
                       for row in grounding]))
        original = {name: (self.results / name).read_text(encoding="utf-8")
                    for name in ("quiz-generation.jsonl", "video-grounding.jsonl")}
        for name, changed in cases:
            with self.subTest(case=len(changed)):
                self.write_rows(name, changed)
                with self.assertRaises(SourceIntegrityError):
                    build_plan(self.results, MODELS)
                (self.results / name).write_text(original[name], encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
