import copy
import hashlib
import json
import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.judge_contract import POINTWISE_ITEMS, canonical_json, sha256_hex
from src import production_judge_aggregation as aggregation_module
from src.production_judge_aggregation import (AGGREGATION_POLICY_VERSION, ProductionJudgeAggregator,
                                              ProductionReviewRefused, aggregate_pointwise, aggregation_policy_sha256)
from src.production_judge import ProductionJudgeUnavailable
from src.production_quiz import new_operation_id
from tests import test_production_judge as judge_tests
from tests.test_production_judge import FakeOpenAI, ok, openai_response, verdicts


def questions(changes=None, indexes=(0, 1, 2)):
    return verdicts(changes, indexes)["questions"]


class AggregatePointwiseTest(unittest.TestCase):
    def test_all_pass(self):
        result = aggregate_pointwise(questions())
        self.assertEqual((result["semanticSummary"], result["reviewRouting"]), ("ALL_PASS", "READY_FOR_CREATOR_REVIEW"))
        self.assertEqual((result["questionCount"], result["evaluationCount"]), (3, 21))
        self.assertEqual(result["counts"], {"pass": 21, "fail": 0, "uncertain": 0})
        self.assertEqual((result["failures"], result["uncertainties"]), ([], []))
        self.assertEqual((result["policyVersion"], result["policySha256"]),
                         (AGGREGATION_POLICY_VERSION, aggregation_policy_sha256()))

    def test_a_single_fail_or_uncertain(self):
        failed = aggregate_pointwise(questions({(1, "uniqueAnswer"): "fail"}))
        self.assertEqual((failed["semanticSummary"], failed["reviewRouting"]), ("HAS_FAIL", "ATTENTION_REQUIRED"))
        self.assertEqual(failed["failures"], [{"questionIndex": 1, "rubric": "uniqueAnswer", "verdict": "fail",
                                               "reason": "uniqueAnswer 문항 1 근거"}])
        uncertain = aggregate_pointwise(questions({(2, "questionClarity"): "uncertain"}))
        self.assertEqual((uncertain["semanticSummary"], uncertain["reviewRouting"]), ("UNCERTAIN_ONLY", "ATTENTION_REQUIRED"))
        self.assertEqual(uncertain["uncertainties"], [{"questionIndex": 2, "rubric": "questionClarity",
                                                       "verdict": "uncertain", "reason": "questionClarity 문항 2 근거"}])
        self.assertEqual(uncertain["failures"], [])

    def test_mixed_verdicts_keep_every_count_and_reason(self):
        changes = {(0, "textAnswerCorrect"): "fail", (0, "koreanQuality"): "uncertain", (2, "textFaithfulness"): "fail",
                   (2, "distractorQuality"): "uncertain", (1, "evidenceSupportsAnswer"): "uncertain"}
        source = questions(changes)
        result = aggregate_pointwise(source)
        self.assertEqual(result["semanticSummary"], "HAS_FAIL")  # uncertain items are still listed
        self.assertEqual(result["counts"], {"pass": 16, "fail": 2, "uncertain": 3})
        self.assertEqual(result["rubricCounts"]["textAnswerCorrect"], {"pass": 2, "fail": 1, "uncertain": 0})
        self.assertEqual(result["rubricCounts"]["evidenceSupportsAnswer"], {"pass": 2, "fail": 0, "uncertain": 1})
        self.assertEqual(set(result["rubricCounts"]), set(POINTWISE_ITEMS))
        self.assertEqual(result["questionCounts"], [{"questionIndex": 0, "pass": 5, "fail": 1, "uncertain": 1},
                                                    {"questionIndex": 1, "pass": 6, "fail": 0, "uncertain": 1},
                                                    {"questionIndex": 2, "pass": 5, "fail": 1, "uncertain": 1}])
        self.assertEqual([(item["questionIndex"], item["rubric"]) for item in result["failures"]],
                         [(0, "textAnswerCorrect"), (2, "textFaithfulness")])
        self.assertEqual([(item["questionIndex"], item["rubric"]) for item in result["uncertainties"]],
                         [(0, "koreanQuality"), (1, "evidenceSupportsAnswer"), (2, "distractorQuality")])
        self.assertEqual(result["questions"], source)  # every original verdict and reason, unchanged

    def test_many_questions(self):
        result = aggregate_pointwise(questions({(4, "questionClarity"): "fail"}, indexes=range(6)))
        self.assertEqual((result["questionCount"], result["evaluationCount"], result["counts"]["fail"]), (6, 42, 1))

    def test_input_order_does_not_change_the_result(self):
        source = questions({(0, "textAnswerCorrect"): "fail", (2, "koreanQuality"): "uncertain"})
        shuffled = [dict(reversed(list(question.items()))) for question in reversed(copy.deepcopy(source))]
        self.assertEqual(canonical_json(aggregate_pointwise(shuffled)), canonical_json(aggregate_pointwise(source)))

    def test_malformed_input_is_rejected_and_nothing_is_invented(self):
        base = questions()
        missing_item = copy.deepcopy(base)
        del missing_item[0]["uniqueAnswer"]
        extra_item = copy.deepcopy(base)
        extra_item[0]["overall"] = {"verdict": "pass", "reason": "x"}
        bad_verdict = copy.deepcopy(base)
        bad_verdict[1]["koreanQuality"]["verdict"] = "partial"
        empty_reason = copy.deepcopy(base)
        empty_reason[2]["textFaithfulness"]["reason"] = " "
        missing_reason = copy.deepcopy(base)
        del missing_reason[2]["textFaithfulness"]["reason"]
        repeated = base + [copy.deepcopy(base[0])]
        bool_index = copy.deepcopy(base)
        bool_index[1]["questionIndex"] = True
        for name, value in {"missing item": missing_item, "extra item": extra_item, "bad verdict": bad_verdict,
                            "empty reason": empty_reason, "missing reason": missing_reason, "repeated index": repeated,
                            "bool index": bool_index, "empty": [], "not a list": {"questions": base}}.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                aggregate_pointwise(value)


class ProductionJudgeAggregatorTest(unittest.TestCase):
    # The Production Judge fixture: a Judge-ready Quiz in a temporary repository and a fake OpenAI transport.
    setUp = judge_tests.ProductionJudgeTest.setUp
    tearDown = judge_tests.ProductionJudgeTest.tearDown
    quiz = judge_tests.ProductionJudgeTest.quiz
    evaluate = judge_tests.ProductionJudgeTest.evaluate
    judge_dir = judge_tests.ProductionJudgeTest.judge_dir
    rewrite = staticmethod(judge_tests.ProductionJudgeTest.rewrite)

    def aggregator(self):
        return ProductionJudgeAggregator(self.repository)

    def aggregation_files(self, evaluation_id):
        folder = self.judge_dir(evaluation_id) / "aggregations"
        return sorted(folder.glob("*.json")) if folder.exists() else []

    def assert_refused(self, evaluation_id, quiz_id=None):
        with self.assertRaises(ProductionReviewRefused):
            self.aggregator().aggregate(quiz_id or self.quiz_id, evaluation_id)
        with self.assertRaises(ProductionReviewRefused):
            self.aggregator().review_status(quiz_id or self.quiz_id, evaluation_id)

    # Completed evaluations

    def test_a_completed_evaluation_is_aggregated_once_and_bound_to_its_provenance(self):
        answer = verdicts({(1, "evidenceSupportsAnswer"): "fail", (2, "questionClarity"): "uncertain"})
        evaluation_id, _ = self.evaluate(FakeOpenAI(ok(answer)))
        status = self.aggregator().review_status(self.quiz_id, evaluation_id)
        self.assertEqual((status["executionStatus"], status["semanticSummary"], status["reviewRouting"],
                          status["manualReviewAllowed"]), ("completed", "HAS_FAIL", "ATTENTION_REQUIRED", True))
        self.assertEqual(status["aggregation"], aggregate_pointwise(answer["questions"]))
        completed = self.runner.require_completed(self.quiz_id, evaluation_id)
        provenance = status["provenance"]
        self.assertEqual((provenance["quiz"], provenance["judge"], provenance["judgeRequestBodySha256"]),
                         (completed["identity"]["quiz"], completed["identity"]["judge"],
                          completed["identity"]["requestBodySha256"]))
        result_file = self.judge_dir(evaluation_id) / "attempts" / "1" / "result.json"
        self.assertEqual(provenance["judgeEvaluation"]["resultSha256"],
                         sha256_hex(json.loads(result_file.read_text(encoding="utf-8"))))
        self.assertEqual(provenance["aggregationPolicy"], {"version": AGGREGATION_POLICY_VERSION,
                                                           "sha256": aggregation_policy_sha256()})
        self.assertEqual(status["aggregationId"], sha256_hex(provenance))  # deterministic, no timestamp
        files = self.aggregation_files(evaluation_id)
        self.assertEqual([path.stem for path in files], [status["aggregationId"]])
        stored = files[0].read_bytes()
        # The same input and policy reuse the stored artifact; nothing is rewritten or added.
        again = self.aggregator().review_status(self.quiz_id, evaluation_id)
        self.assertEqual(again["aggregationId"], status["aggregationId"])
        self.assertEqual(self.aggregation_files(evaluation_id), files)
        self.assertEqual(files[0].read_bytes(), stored)
        # The Judge records are untouched and the evaluation still verifies.
        self.assertEqual(self.runner.evaluation_state(self.quiz_id, evaluation_id), "completed")

    def test_semantic_routing_for_each_summary(self):
        for changes, summary, routing in (({}, "ALL_PASS", "READY_FOR_CREATOR_REVIEW"),
                                          ({(0, "koreanQuality"): "fail"}, "HAS_FAIL", "ATTENTION_REQUIRED"),
                                          ({(0, "koreanQuality"): "uncertain"}, "UNCERTAIN_ONLY", "ATTENTION_REQUIRED")):
            with self.subTest(summary=summary):
                evaluation_id, _ = self.evaluate(FakeOpenAI(ok(verdicts(changes))))
                status = self.aggregator().review_status(self.quiz_id, evaluation_id)
                self.assertEqual((status["executionStatus"], status["semanticSummary"], status["reviewRouting"]),
                                 ("completed", summary, routing))

    def test_a_policy_change_makes_a_separate_artifact(self):
        evaluation_id, _ = self.evaluate(FakeOpenAI(ok()))
        first = self.aggregator().aggregate(self.quiz_id, evaluation_id)
        with mock.patch.object(aggregation_module, "AGGREGATION_POLICY_VERSION", "production-judge-aggregation-v2"):
            second = self.aggregator().aggregate(self.quiz_id, evaluation_id)
        self.assertNotEqual(first["aggregationId"], second["aggregationId"])
        self.assertEqual(len(self.aggregation_files(evaluation_id)), 2)
        self.assertEqual(self.aggregator().aggregate(self.quiz_id, evaluation_id), first)  # v1 is unchanged

    # Unavailable evaluations

    def test_unavailable_evaluations_allow_manual_review_without_semantic_results(self):
        cases = {"not_started": None, "retryable": openai_response("{"), "terminal": (400, {}, None),
                 "uncertain": socket.timeout()}
        for state, response in cases.items():
            with self.subTest(state=state):
                evaluation_id = new_operation_id()
                if response is not None:
                    self.evaluate(FakeOpenAI(response), evaluation_id)
                transport = FakeOpenAI(ok())
                status = self.aggregator().review_status(self.quiz_id, evaluation_id)
                self.assertEqual(status, {"quizOperationId": self.quiz_id, "evaluationId": evaluation_id,
                                          "executionStatus": state, "semanticSummary": "UNAVAILABLE",
                                          "reviewRouting": "JUDGE_UNAVAILABLE", "manualReviewAllowed": True,
                                          "unavailableReason": "evaluationState:" + state})
                self.assertEqual(transport.bodies, [])  # a read never calls the Judge
                with self.assertRaises(ProductionReviewRefused):  # and never creates a semantic aggregation
                    self.aggregator().aggregate(self.quiz_id, evaluation_id)
                self.assertEqual(self.aggregation_files(evaluation_id), [])
        evaluation_id, _ = self.evaluate(FakeOpenAI(openai_response("{")))
        self.evaluate(FakeOpenAI(openai_response("{")), evaluation_id)
        self.assertEqual(self.aggregator().review_status(self.quiz_id, evaluation_id)["executionStatus"], "exhausted")

    def test_execution_uncertain_is_not_semantic_uncertain(self):
        execution, _ = self.evaluate(FakeOpenAI(socket.timeout()))
        semantic, _ = self.evaluate(FakeOpenAI(ok(verdicts({(0, "questionClarity"): "uncertain"}))))
        execution_status = self.aggregator().review_status(self.quiz_id, execution)
        semantic_status = self.aggregator().review_status(self.quiz_id, semantic)
        self.assertEqual((execution_status["executionStatus"], execution_status["semanticSummary"]),
                         ("uncertain", "UNAVAILABLE"))
        self.assertEqual((semantic_status["executionStatus"], semantic_status["semanticSummary"]),
                         ("completed", "UNCERTAIN_ONLY"))

    def test_corrupted_records_and_a_quiz_that_is_not_judge_ready_are_refused(self):
        evaluation_id, _ = self.evaluate(FakeOpenAI(ok()))
        self.rewrite(self.judge_dir(evaluation_id) / "attempts" / "1" / "result.json",
                     lambda r: r["pointwise"][0]["koreanQuality"].update(verdict="fail"))
        self.assert_refused(evaluation_id)
        self.assertEqual(self.aggregation_files(evaluation_id), [])
        quiz_id = self.quiz()
        evaluation_id, _ = self.evaluate(FakeOpenAI(ok()), quiz_id=quiz_id)
        (self.production / "operations" / quiz_id / "attempts" / "1" / "output.json").write_text(
            '{"output": {}}', encoding="ascii")
        self.assert_refused(evaluation_id, quiz_id)

    # Stored aggregation integrity

    def test_an_aggregation_is_never_reused_after_its_quiz_or_judge_output_changes(self):
        for name, tamper in {
                "quiz output": lambda e: (self.production / "operations" / self.quiz_id / "attempts" / "1"
                                          / "output.json").write_text('{"output": {}}', encoding="ascii"),
                "judge output": lambda e: self.rewrite(self.judge_dir(e) / "attempts" / "1" / "output.json",
                                                       lambda r: r.update(text=json.dumps(verdicts({(0, "uniqueAnswer"): "fail"}))))}.items():
            with self.subTest(case=name):
                self.quiz_id = self.quiz()
                evaluation_id, _ = self.evaluate(FakeOpenAI(ok()))
                self.aggregator().aggregate(self.quiz_id, evaluation_id)
                tamper(evaluation_id)
                self.assert_refused(evaluation_id)

    def test_a_tampered_or_partial_stored_aggregation_is_refused_and_never_overwritten(self):
        tampering = {
            "count": lambda a: a["aggregation"]["counts"].update(fail=1),
            "summary": lambda a: a["aggregation"].update(semanticSummary="HAS_FAIL", reviewRouting="ATTENTION_REQUIRED"),
            "reason": lambda a: a["aggregation"]["questions"][0]["textAnswerCorrect"].update(reason="changed"),
            "provenance quiz output": lambda a: a["provenance"]["quiz"].update(quizOutputSha256="f" * 64),
            "provenance judge result": lambda a: a["provenance"]["judgeEvaluation"].update(resultSha256="f" * 64),
            "policy digest": lambda a: a["aggregation"].update(policySha256="f" * 64),
            "policy provenance": lambda a: a["provenance"]["aggregationPolicy"].update(sha256="f" * 64),
            "aggregation id": lambda a: a.update(aggregationId="f" * 64),
            "generatedAt": lambda a: a.update(generatedAt="2026-99-99T25:61:61+00:00"),
            "extra key": lambda a: a.update(approved=True),
            "format": lambda a: a.update(format="other"),
        }
        for name, change in tampering.items():
            with self.subTest(case=name):
                evaluation_id, _ = self.evaluate(FakeOpenAI(ok()))
                self.aggregator().aggregate(self.quiz_id, evaluation_id)
                path = self.aggregation_files(evaluation_id)[0]
                self.rewrite(path, change)
                tampered = path.read_bytes()
                self.assert_refused(evaluation_id)
                self.assertEqual(path.read_bytes(), tampered)  # never overwritten or "repaired"
        for name, data in {"truncated": b'{"format": "production-judge-aggreg', "empty": b"",
                           "not json": b"\xff\xfe"}.items():
            with self.subTest(case=name):
                evaluation_id, _ = self.evaluate(FakeOpenAI(ok()))
                self.aggregator().aggregate(self.quiz_id, evaluation_id)
                path = self.aggregation_files(evaluation_id)[0]
                path.unlink()
                path.write_bytes(data)
                self.assert_refused(evaluation_id)
                self.assertEqual(path.read_bytes(), data)

    def test_a_read_failure_is_refused(self):
        evaluation_id, _ = self.evaluate(FakeOpenAI(ok()))
        aggregator = self.aggregator()
        aggregator.aggregate(self.quiz_id, evaluation_id)
        with mock.patch.object(aggregator.quiz, "_read_json", side_effect=OSError("unreadable")):
            with self.assertRaises(ProductionReviewRefused):
                aggregator.aggregate(self.quiz_id, evaluation_id)


    def test_the_aggregation_is_bound_to_the_exact_result_bytes(self):
        evaluation_id, _ = self.evaluate(FakeOpenAI(ok()))
        first = self.aggregator().aggregate(self.quiz_id, evaluation_id)
        path = self.judge_dir(evaluation_id) / "attempts" / "1" / "result.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        path.unlink()
        path.write_bytes(json.dumps(record, ensure_ascii=False, indent=2).encode("utf-8"))
        second = self.aggregator().aggregate(self.quiz_id, evaluation_id)
        self.assertEqual(second["provenance"]["judgeEvaluation"]["resultSha256"],
                         hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertNotEqual(first["aggregationId"], second["aggregationId"])
        self.assertEqual(first["aggregation"], second["aggregation"])  # same verdicts, different exact bytes
        self.assertEqual(len(self.aggregation_files(evaluation_id)), 2)
        # A result that no longer verifies is never aggregated.
        self.rewrite(path, lambda r: r["pointwise"][0]["uniqueAnswer"].update(verdict="fail"))
        self.assert_refused(evaluation_id)
        self.assertEqual(len(self.aggregation_files(evaluation_id)), 2)

    def test_a_missing_or_damaged_manifest_is_refused_not_routed_to_manual_review(self):
        for name, damage in {"deleted": lambda path: path.unlink(),
                             "directory": lambda path: (path.unlink(), path.mkdir())}.items():
            with self.subTest(case=name):
                evaluation_id, _ = self.evaluate(FakeOpenAI(ok()))
                self.aggregator().aggregate(self.quiz_id, evaluation_id)
                damage(self.judge_dir(evaluation_id) / "evaluation.json")
                with self.assertRaises(ProductionReviewRefused) as caught:
                    self.aggregator().review_status(self.quiz_id, evaluation_id)
                self.assertEqual(caught.exception.reasons, ("evaluationState:corrupted",))
                with self.assertRaises(ProductionReviewRefused):
                    self.aggregator().aggregate(self.quiz_id, evaluation_id)

    def test_an_aggregation_write_failure_is_refused_and_safely_retryable(self):
        evaluation_id, _ = self.evaluate(FakeOpenAI(ok()))
        aggregator = self.aggregator()
        for error in (PermissionError("denied"), OSError("disk full")):
            with self.subTest(error=type(error).__name__):
                with mock.patch.object(aggregator.quiz, "_create_once", side_effect=error):
                    with self.assertRaises(ProductionReviewRefused) as caught:
                        aggregator.aggregate(self.quiz_id, evaluation_id)
                    self.assertEqual(caught.exception.reasons, ("aggregationArtifactWriteFailed",))
                    self.assertIs(caught.exception.__cause__, error)  # the original cause is kept
                    with self.assertRaises(ProductionReviewRefused):  # never reported as JUDGE_UNAVAILABLE
                        aggregator.review_status(self.quiz_id, evaluation_id)
                self.assertEqual(self.aggregation_files(evaluation_id), [])  # nothing partial was left
        stored = aggregator.aggregate(self.quiz_id, evaluation_id)  # a later call simply retries
        files = self.aggregation_files(evaluation_id)
        self.assertEqual([path.stem for path in files], [stored["aggregationId"]])
        data = files[0].read_bytes()
        with mock.patch.object(aggregator.quiz, "_create_once", side_effect=PermissionError("denied")) as create:
            self.assertEqual(aggregator.aggregate(self.quiz_id, evaluation_id), stored)  # the stored artifact is reused
            self.assertEqual(aggregator.review_status(self.quiz_id, evaluation_id)["aggregationId"], stored["aggregationId"])
        create.assert_not_called()
        self.assertEqual(files[0].read_bytes(), data)

    def assert_corrupted_without_side_effects(self, evaluation_id):
        folder = self.judge_dir(evaluation_id)
        before = sorted(str(path.relative_to(folder)) for path in folder.iterdir())
        self.assertEqual(self.runner.evaluation_state(self.quiz_id, evaluation_id), "corrupted")
        with self.assertRaises(ProductionJudgeUnavailable):
            self.runner.require_completed(self.quiz_id, evaluation_id)
        with self.assertRaises(ProductionReviewRefused):  # never manualReviewAllowed
            self.aggregator().review_status(self.quiz_id, evaluation_id)
        transport = FakeOpenAI(ok())
        with self.assertRaisesRegex(ValueError, "corrupted"):
            self.evaluate(transport, evaluation_id)
        self.assertEqual(transport.bodies, [])
        self.assertFalse((folder / "evaluation.json").exists())
        self.assertEqual(sorted(str(path.relative_to(folder)) for path in folder.iterdir()), before)

    def test_a_temp_named_symlink_without_a_manifest_is_corrupted(self):
        evaluation_id = new_operation_id()
        folder = self.judge_dir(evaluation_id)
        folder.mkdir(parents=True)
        target = folder.parent / "elsewhere.tmp"
        target.write_bytes(b"{")
        try:
            os.symlink(target, folder / ".production-abcd1234.tmp")
        except (OSError, NotImplementedError) as exc:
            self.skipTest("creating symlinks is not permitted in this environment: %s" % type(exc).__name__)
        self.assert_corrupted_without_side_effects(evaluation_id)

    def test_a_temp_file_reported_as_a_symlink_is_corrupted(self):
        # Runs where real symlinks cannot be created: the classification only asks Path.is_symlink().
        evaluation_id = new_operation_id()
        folder = self.judge_dir(evaluation_id)
        folder.mkdir(parents=True)
        with tempfile.NamedTemporaryFile("wb", dir=folder, prefix=".production-", suffix=".tmp", delete=False) as stream:
            stream.write(b"{")
        self.assertEqual(self.runner.evaluation_state(self.quiz_id, evaluation_id), "not_started")
        real = Path.is_symlink
        with mock.patch.object(Path, "is_symlink", lambda path: path.name == Path(stream.name).name or real(path)):
            self.assertEqual(self.runner.evaluation_state(self.quiz_id, evaluation_id), "corrupted")
            transport = FakeOpenAI(ok())
            with self.assertRaises(ValueError):
                self.evaluate(transport, evaluation_id)
            self.assertEqual(transport.bodies, [])

    def test_leftover_entries_without_a_manifest_are_corrupted_unless_they_are_create_once_temp_files(self):
        def temp_file(folder):
            with tempfile.NamedTemporaryFile("wb", dir=folder, prefix=".production-", suffix=".tmp", delete=False) as stream:
                stream.write(b"{")
            return Path(stream.name)

        damaged = {
            "unknown hidden file": lambda folder: (folder / ".unexpected-record").write_text("{}", encoding="utf-8"),
            "temp-named directory": lambda folder: (folder / ".production-abcd1234.tmp").mkdir(),
            "orphan temp directory with records": lambda folder: (
                (folder / ".production-orphan.tmp").mkdir(),
                (folder / ".production-orphan.tmp" / "result.json").write_text("{}", encoding="utf-8")),
            "unknown hidden directory": lambda folder: (folder / ".hidden").mkdir(),
            "temp file and unknown hidden file": lambda folder: (
                temp_file(folder), (folder / ".unexpected-record").write_text("{}", encoding="utf-8")),
            "temp file and hidden directory": lambda folder: (temp_file(folder), (folder / ".hidden").mkdir()),
            "temp-like name, wrong random part": lambda folder: (folder / ".production-orphan.tmp").write_bytes(b"{"),
        }
        for name, prepare in damaged.items():
            with self.subTest(case=name):
                evaluation_id = new_operation_id()
                folder = self.judge_dir(evaluation_id)
                folder.mkdir(parents=True)
                prepare(folder)
                self.assert_corrupted_without_side_effects(evaluation_id)

        # No directory, an empty one, or only real create-once temp files: not started, and resumable.
        for name, prepare in {"no directory": None, "empty directory": lambda folder: None,
                              "create-once temp files only": lambda folder: (temp_file(folder), temp_file(folder))}.items():
            with self.subTest(case=name):
                evaluation_id = new_operation_id()
                if prepare is not None:
                    folder = self.judge_dir(evaluation_id)
                    folder.mkdir(parents=True)
                    prepare(folder)
                status = self.aggregator().review_status(self.quiz_id, evaluation_id)
                self.assertEqual((status["executionStatus"], status["manualReviewAllowed"]), ("not_started", True))
                _, result = self.evaluate(FakeOpenAI(ok()), evaluation_id)
                self.assertEqual(result["outcome"], "completed")
                self.assertEqual(self.aggregator().review_status(self.quiz_id, evaluation_id)["executionStatus"], "completed")

if __name__ == "__main__":
    unittest.main()
