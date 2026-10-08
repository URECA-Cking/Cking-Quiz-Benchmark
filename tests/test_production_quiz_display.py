import contextlib
import copy
import io
import json
import unittest
from unittest import mock

from src.judge_contract import canonical_json
from src.production_creator_review import ProductionCreatorReview
from src.production_quiz import ProductionQuizRunner
from src.production_quiz_display import QuizDisplayError, format_review_questions, main, quiz_review_text
from tests import test_production_judge as judge_tests
from tests.test_production_judge import FakeOpenAI, ok, verdicts


def packet(answers=(0, 1, 2)):
    return {"questions": [
        {"questionIndex": index, "question": "질문 %d: 다음 중 올바른 설명은?" % (index + 1),
         "options": ["보기 가 %d" % index, "보기 나 %d" % index, "보기 다 %d" % index, "보기 라 %d" % index],
         "correctOptionIndex": answer, "explanation": "해설 원문 %d" % index, "sourceEvidence": "근거 원문 %d" % index}
        for index, answer in enumerate(answers)]}


class FormatReviewQuestionsTest(unittest.TestCase):
    def test_the_answer_is_shown_as_its_1_based_option_number(self):
        for answer in range(4):
            with self.subTest(correctOptionIndex=answer):
                text = format_review_questions(packet((answer,)))
                expected = "보기 %s 0" % "가나다라"[answer]
                self.assertIn("정답: %d번 — %s\n" % (answer + 1, expected), text)

    def test_text_keeps_order_and_original_wording(self):
        source = packet((2, 0, 3))
        text = format_review_questions(source)
        self.assertEqual(text.splitlines()[:12], [
            "문제 1. 질문 1: 다음 중 올바른 설명은?", "",
            "1. 보기 가 0", "2. 보기 나 0", "3. 보기 다 0", "4. 보기 라 0", "",
            "정답: 3번 — 보기 다 0", "", "해설: 해설 원문 0", "근거: 근거 원문 0", ""])
        positions = [text.index("문제 %d." % number) for number in (1, 2, 3)]
        self.assertEqual(positions, sorted(positions))
        for question in source["questions"]:
            for value in [question["question"], question["explanation"], question["sourceEvidence"]] + question["options"]:
                self.assertIn(value, text)
        self.assertIn("정답: 4번 — 보기 라 2", text)

    def test_the_packet_is_not_changed(self):
        source = packet((1, 3, 0))
        before_object, before_json = copy.deepcopy(source), canonical_json(source)
        format_review_questions(source)
        self.assertEqual(source, before_object)
        self.assertEqual(canonical_json(source), before_json)
        self.assertEqual([q["correctOptionIndex"] for q in source["questions"]], [1, 3, 0])  # still 0-based
        self.assertEqual([q["questionIndex"] for q in source["questions"]], [0, 1, 2])

    def test_an_invalid_answer_index_is_never_shown_as_an_answer(self):
        for name, value in {"bool": True, "false": False, "float": 1.0, "negative": -1, "out of range": 4,
                            "string": "2", "null": None}.items():
            with self.subTest(case=name):
                bad = packet()
                bad["questions"][1]["correctOptionIndex"] = value
                with self.assertRaises(QuizDisplayError):
                    format_review_questions(bad)

    def test_malformed_questions_are_refused(self):
        cases = {
            "three options": lambda q: q[0].update(options=["a", "b", "c"]),
            "options not a list": lambda q: q[0].update(options="a,b,c,d"),
            "non-text option": lambda q: q[0]["options"].__setitem__(1, 7),
            "empty option": lambda q: q[0]["options"].__setitem__(1, " "),
            "duplicate options": lambda q: q[0]["options"].__setitem__(1, "보기 가 0"),
            "missing evidence": lambda q: q[0].pop("sourceEvidence"),
            "unknown field": lambda q: q[0].update(hint="x"),
            "bool questionIndex": lambda q: q[0].update(questionIndex=False),
            "out-of-order questionIndex": lambda q: q.reverse(),
        }
        for name, change in cases.items():
            with self.subTest(case=name):
                bad = packet()
                change(bad["questions"])
                with self.assertRaises(QuizDisplayError):
                    format_review_questions(bad)
        for empty in ({}, {"questions": []}, None, []):
            with self.subTest(packet=empty), self.assertRaises(QuizDisplayError):
                format_review_questions(empty)


class ProductionQuizDisplayPathTest(unittest.TestCase):
    # The Production Judge fixture: a Judge-ready Quiz in a temporary repository and a fake OpenAI transport.
    setUp = judge_tests.ProductionJudgeTest.setUp
    tearDown = judge_tests.ProductionJudgeTest.tearDown
    quiz = judge_tests.ProductionJudgeTest.quiz
    evaluate = judge_tests.ProductionJudgeTest.evaluate

    def files(self):
        return {str(path): path.read_bytes() for path in self.repository.rglob("*") if path.is_file()}

    def test_the_command_shows_a_stored_quiz_without_writing_or_calling_a_provider(self):
        judge_input = self.runner.quiz.require_judge_ready(self.quiz_id)
        before = self.files()
        out, err = io.StringIO(), io.StringIO()
        with mock.patch("src.production_quiz.ProductionQuizRunner._create_once") as create, \
                mock.patch("src.provider_adapters.urllib_transport", side_effect=AssertionError("Provider call")), \
                mock.patch("src.judge_client.judge_transport", side_effect=AssertionError("Provider call")), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(["--operation-id", self.quiz_id, "--repository", str(self.repository)])
        self.assertEqual((code, err.getvalue()), (0, ""))
        self.assertEqual(create.call_count, 0)
        self.assertEqual(self.files(), before)  # nothing created or changed
        text = out.getvalue()
        for question in judge_input["questions"]:
            self.assertIn("정답: %d번 — %s" % (question["correctOptionIndex"] + 1,
                                              question["options"][question["correctOptionIndex"]]), text)
        self.assertEqual(text.count("정답:"), 3)

    def test_the_view_matches_the_creator_review_packet_and_leaves_it_unchanged(self):
        evaluation_id, _ = self.evaluate(FakeOpenAI(ok(verdicts())))
        review = ProductionCreatorReview(self.repository).review_packet(self.quiz_id, evaluation_id)
        before = canonical_json(review)
        self.assertEqual(format_review_questions(review),
                         quiz_review_text(self.repository, self.quiz_id).split("\n\n", 1)[1])
        self.assertEqual(canonical_json(review), before)

    def test_a_quiz_that_is_not_reviewable_is_refused(self):
        (self.production / "operations" / self.quiz_id / "attempts" / "1" / "output.json").write_text(
            '{"output": {}}', encoding="ascii")
        out, err = io.StringIO(), io.StringIO()
        before = self.files()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(["--operation-id", self.quiz_id, "--repository", str(self.repository)])
        self.assertEqual((code, out.getvalue()), (1, ""))
        self.assertIn("not reviewable", err.getvalue())
        self.assertEqual(self.files(), before)

    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        before = self.files()
        with mock.patch("src.provider_adapters.urllib_transport", side_effect=AssertionError("Provider call")), \
                mock.patch("src.judge_client.judge_transport", side_effect=AssertionError("Provider call")), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(list(argv))
        self.assertEqual(self.files(), before)  # no file created or changed on any path
        return code, out.getvalue(), err.getvalue()

    def test_setup_and_lookup_errors_are_reported_without_a_traceback(self):
        no_config = self.repository / "empty-repository"
        no_config.mkdir()
        broken_config = self.repository / "broken-config"
        (broken_config / "configs").mkdir(parents=True)
        (broken_config / "configs" / "production.yaml").write_text("production: [unclosed", encoding="utf-8")
        cases = {
            "results dir outside results/production": (["--results-dir", str(self.repository / "results" / "pilot-v2-r2")],
                                                       "Invalid --repository or --results-dir"),
            "repository without the production config": (["--repository", str(no_config)], "Production config cannot be read"),
            "unparseable production config": (["--repository", str(broken_config)], "Invalid --repository or --results-dir"),
        }
        before = self.files()
        for name, (extra, message) in cases.items():
            with self.subTest(case=name):
                argv = ["--operation-id", self.quiz_id, "--repository", str(self.repository)] + extra
                code, out, err = self.run_cli(*argv)
                self.assertEqual((code, out), (1, ""))
                self.assertIn(message, err)
                self.assertNotIn("Traceback", err)
        code, out, err = self.run_cli("--operation-id", "f" * 32, "--repository", str(self.repository))
        self.assertEqual((code, out), (1, ""))
        self.assertIn("Quiz is not reviewable", err)
        self.assertNotIn("Traceback", err)
        self.assertEqual(self.files(), before)

    def test_a_production_config_with_the_wrong_structure_is_a_setup_error(self):
        contents = {"empty file": "", "null": "null\n", "list": "[]\n", "top-level string": "production\n",
                    "top-level number": "3\n", "no production section": "other: {}\n",
                    "production is a list": "production: []\n", "production is null": "production:\n"}
        before = self.files()
        for name, text in contents.items():
            with self.subTest(case=name):
                repository = self.repository / ("config-" + name.replace(" ", "-"))
                (repository / "configs").mkdir(parents=True)
                (repository / "configs" / "production.yaml").write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):  # the loader refuses it explicitly, never a TypeError
                    ProductionQuizRunner(repository)
                fixture = self.files()
                code, out, err = self.run_cli("--operation-id", self.quiz_id, "--repository", str(repository))
                self.assertEqual((code, out), (1, ""))
                self.assertIn("Invalid --repository or --results-dir", err)
                self.assertNotIn("Traceback", err)
                self.assertEqual(self.files(), fixture)
        self.assertEqual({key: value for key, value in self.files().items() if key in before}, before)

    def test_a_normal_lookup_still_prints_numbered_options_and_answers(self):
        code, out, err = self.run_cli("--operation-id", self.quiz_id, "--repository", str(self.repository))
        self.assertEqual((code, err), (0, ""))
        for number in (1, 2, 3, 4):
            self.assertIn("\n%d. " % number, out)
        self.assertEqual(out.count("정답: "), 3)

    def test_creator_and_publication_records_are_not_touched(self):
        evaluation_id, _ = self.evaluate(FakeOpenAI(ok(verdicts())))
        ProductionCreatorReview(self.repository).record_decision(self.quiz_id, evaluation_id, "APPROVE", "creator-kim")
        operation = self.production / "operations" / self.quiz_id
        decision = (operation / "creator-decision.json").read_bytes()
        quiz_review_text(self.repository, self.quiz_id)
        self.assertEqual((operation / "creator-decision.json").read_bytes(), decision)
        self.assertFalse((operation / "publication.json").exists())
        self.assertEqual(ProductionCreatorReview(self.repository).decision_status(self.quiz_id)["status"],
                         "CREATOR_APPROVED")


if __name__ == "__main__":
    unittest.main()
