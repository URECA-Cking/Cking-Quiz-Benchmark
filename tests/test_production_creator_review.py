import json
import socket
import unittest
from pathlib import Path
from unittest import mock

from src.production_creator_review import (APPROVED, CORRUPTED, PENDING, REJECTED, STALE, CreatorReviewRefused,
                                           ProductionCreatorReview)
from src.production_quiz import new_operation_id
from tests import test_production_judge as judge_tests
from tests.test_production_judge import FakeOpenAI, ok, openai_response, verdicts


REVIEWER = "creator-kim"
FLAGGED = {(0, "textAnswerCorrect"): "fail", (2, "questionClarity"): "uncertain"}
FLAGGED_ITEMS = [{"questionIndex": 0, "rubric": "textAnswerCorrect"}, {"questionIndex": 2, "rubric": "questionClarity"}]


class ProductionCreatorReviewTest(unittest.TestCase):
    # The Production Judge fixture: a Judge-ready Quiz in a temporary repository and a fake OpenAI transport.
    setUp = judge_tests.ProductionJudgeTest.setUp
    tearDown = judge_tests.ProductionJudgeTest.tearDown
    quiz = judge_tests.ProductionJudgeTest.quiz
    evaluate = judge_tests.ProductionJudgeTest.evaluate
    judge_dir = judge_tests.ProductionJudgeTest.judge_dir
    rewrite = staticmethod(judge_tests.ProductionJudgeTest.rewrite)

    def review(self):
        return ProductionCreatorReview(self.repository)

    def decision_path(self, quiz_id=None):
        return self.production / "operations" / (quiz_id or self.quiz_id) / "creator-decision.json"

    def completed(self, changes=None, quiz_id=None):
        evaluation_id, _ = self.evaluate(FakeOpenAI(ok(verdicts(changes))), quiz_id=quiz_id)
        return evaluation_id

    def files(self):
        return {str(path): path.read_bytes() for path in self.production.rglob("*") if path.is_file()}

    def assert_status(self, status, reasons=None, quiz_id=None):
        result = self.review().decision_status(quiz_id or self.quiz_id)
        self.assertEqual(result["status"], status, result["reasons"])
        if reasons is not None:
            self.assertEqual(result["reasons"], reasons)
        return result

    def assert_gate_refused(self, quiz_id=None):
        before = self.files()
        with self.assertRaises(CreatorReviewRefused):
            self.review().require_creator_approved(quiz_id or self.quiz_id)
        self.assertEqual(self.files(), before)  # the gate never writes

    # Review packet

    def test_review_packet_for_a_completed_evaluation(self):
        evaluation_id = self.completed(FLAGGED)
        packet = self.review().review_packet(self.quiz_id, evaluation_id)
        judge_input = self.runner.quiz.require_judge_ready(self.quiz_id)
        self.assertEqual(packet["contentText"], judge_input["contentText"])
        self.assertEqual(packet["questions"], [{key: question[key] for key in (
            "questionIndex", "question", "options", "correctOptionIndex", "explanation", "sourceEvidence")}
            for question in judge_input["questions"]])
        judge = packet["judge"]
        self.assertEqual((judge["executionStatus"], judge["semanticSummary"], judge["reviewRouting"],
                          judge["manualReviewAllowed"], judge["unavailableWarning"]),
                         ("completed", "HAS_FAIL", "ATTENTION_REQUIRED", True, None))
        self.assertEqual(judge["problemItems"], FLAGGED_ITEMS)
        self.assertEqual(judge["aggregation"]["failures"][0]["reason"], "textAnswerCorrect 문항 0 근거")
        self.assertEqual(len(judge["aggregation"]["questions"]), 3)  # every rubric verdict and reason
        self.assertEqual(self.review().review_packet(self.quiz_id, self.completed())["judge"]["reviewRouting"],
                         "READY_FOR_CREATOR_REVIEW")

    def test_review_packet_without_a_judge_result_warns(self):
        evaluation_id, _ = self.evaluate(FakeOpenAI(socket.timeout()))
        packet = self.review().review_packet(self.quiz_id, evaluation_id)
        judge = packet["judge"]
        self.assertEqual((judge["executionStatus"], judge["semanticSummary"], judge["reviewRouting"],
                          judge["aggregation"], judge["problemItems"]),
                         ("uncertain", "UNAVAILABLE", "JUDGE_UNAVAILABLE", None, []))
        self.assertIn("No completed Judge evaluation", judge["unavailableWarning"])
        self.assertEqual(len(packet["questions"]), 3)

    def test_review_packet_refuses_corrupted_judges_and_ineligible_quizzes(self):
        evaluation_id = self.completed()
        self.rewrite(self.judge_dir(evaluation_id) / "attempts" / "1" / "result.json",
                     lambda r: r["pointwise"][0]["koreanQuality"].update(verdict="fail"))
        with self.assertRaises(CreatorReviewRefused):
            self.review().review_packet(self.quiz_id, evaluation_id)
        quiz_id = self.quiz()
        evaluation_id = self.completed(quiz_id=quiz_id)
        (self.production / "operations" / quiz_id / "attempts" / "1" / "output.json").write_text(
            '{"output": {}}', encoding="ascii")
        with self.assertRaises(CreatorReviewRefused) as caught:
            self.review().review_packet(quiz_id, evaluation_id)
        self.assertEqual(caught.exception.reasons[0], "quizNotJudgeReady")

    # Approval and rejection

    def test_approval_kinds_follow_the_reviewed_routing(self):
        cases = [("JUDGE_AGREED", None, None, None), ("JUDGE_OVERRIDE", FLAGGED, "Creator checked the source", FLAGGED_ITEMS)]
        for kind, changes, reason, items in cases:
            with self.subTest(kind=kind):
                quiz_id = self.quiz()
                evaluation_id = self.completed(changes, quiz_id=quiz_id)
                record = self.review().record_decision(quiz_id, evaluation_id, "APPROVE", REVIEWER, reason, items)
                self.assertEqual((record["decision"], record["decisionKind"], record["decidedBy"]),
                                 ("APPROVE", kind, REVIEWER))
                self.assertEqual(record["judge"]["resultSha256"],
                                 self.runner.require_completed(quiz_id, evaluation_id)["resultSha256"])
                self.assertEqual(self.assert_status(APPROVED, quiz_id=quiz_id)["decision"], record)
                approved = self.review().require_creator_approved(quiz_id)
                self.assertEqual((approved["quizOperationId"], approved["decision"]), (quiz_id, record))
                self.assertEqual(len(approved["questions"]), 3)

    def test_manual_approval_without_a_judge_needs_a_reason(self):
        evaluation_id, _ = self.evaluate(FakeOpenAI(openai_response("{")))  # retryable: no Judge result
        with self.assertRaises(CreatorReviewRefused) as caught:
            self.review().record_decision(self.quiz_id, evaluation_id, "APPROVE", REVIEWER)
        self.assertEqual(caught.exception.reasons, ("manualApprovalReasonRequired",))
        self.assertFalse(self.decision_path().exists())
        record = self.review().record_decision(self.quiz_id, evaluation_id, "APPROVE", REVIEWER, "Read it myself")
        self.assertEqual((record["decisionKind"], record["review"]["executionStatus"], record["review"]["reviewRouting"],
                          record["judge"], record["review"]["aggregationId"]),
                         ("MANUAL_WITHOUT_JUDGE", "retryable", "JUDGE_UNAVAILABLE", None, None))
        self.assert_status(APPROVED)
        self.review().require_creator_approved(self.quiz_id)

    def test_every_rejection_needs_a_reason(self):
        evaluation_id = self.completed()  # ALL_PASS
        for reason in (None, "", "  "):
            with self.subTest(reason=reason), self.assertRaises(CreatorReviewRefused):
                self.review().record_decision(self.quiz_id, evaluation_id, "REJECT", REVIEWER, reason)
        self.assertFalse(self.decision_path().exists())
        record = self.review().record_decision(self.quiz_id, evaluation_id, "REJECT", REVIEWER, "Off-topic questions")
        self.assertEqual((record["decision"], record["decisionKind"]), ("REJECT", "CREATOR_REJECTION"))
        self.assert_status(REJECTED)
        self.assert_gate_refused()

    def test_an_override_needs_a_reason_and_exactly_the_flagged_items(self):
        evaluation_id = self.completed(FLAGGED)
        wrong = {"no reason": (None, FLAGGED_ITEMS, "overrideReasonRequired"),
                 "no items": ("ok", None, "acknowledgedItemsMismatch"),
                 "missing item": ("ok", FLAGGED_ITEMS[:1], "acknowledgedItemsMismatch"),
                 "extra item": ("ok", FLAGGED_ITEMS + [{"questionIndex": 1, "rubric": "koreanQuality"}],
                                "acknowledgedItemsMismatch"),
                 "wrong rubric": ("ok", [FLAGGED_ITEMS[0], {"questionIndex": 2, "rubric": "koreanQuality"}],
                                  "acknowledgedItemsMismatch"),
                 "repeated item": ("ok", FLAGGED_ITEMS + FLAGGED_ITEMS[:1], "invalidAcknowledgedItems"),
                 "unknown rubric": ("ok", [{"questionIndex": 0, "rubric": "overall"}], "invalidAcknowledgedItems"),
                 "extra key": ("ok", [dict(FLAGGED_ITEMS[0], note="x"), FLAGGED_ITEMS[1]], "invalidAcknowledgedItems")}
        for name, (reason, items, expected) in wrong.items():
            with self.subTest(case=name):
                with self.assertRaises(CreatorReviewRefused) as caught:
                    self.review().record_decision(self.quiz_id, evaluation_id, "APPROVE", REVIEWER, reason, items)
                self.assertEqual(caught.exception.reasons, (expected,))
        self.assertFalse(self.decision_path().exists())
        reordered = list(reversed(FLAGGED_ITEMS))  # order does not matter, the set does
        record = self.review().record_decision(self.quiz_id, evaluation_id, "APPROVE", REVIEWER, "ok", reordered)
        self.assertEqual((record["decisionKind"], record["acknowledgedItems"]), ("JUDGE_OVERRIDE", FLAGGED_ITEMS))

    def test_items_are_never_acknowledged_where_nothing_is_flagged(self):
        evaluation_id = self.completed()
        with self.assertRaises(CreatorReviewRefused) as caught:
            self.review().record_decision(self.quiz_id, evaluation_id, "APPROVE", REVIEWER, None, FLAGGED_ITEMS)
        self.assertEqual(caught.exception.reasons, ("acknowledgedItemsNotExpected",))

    def test_invalid_requests_are_refused_before_anything_is_written(self):
        evaluation_id = self.completed()
        for decision, decided_by in (("approve", REVIEWER), ("PUBLISH", REVIEWER), ("APPROVE", " "),
                                     ("APPROVE", "a​b"), ("APPROVE", None)):
            with self.subTest(decision=decision, decided_by=decided_by), self.assertRaises(CreatorReviewRefused):
                self.review().record_decision(self.quiz_id, evaluation_id, decision, decided_by)
        self.assertFalse(self.decision_path().exists())

    # One final decision

    def test_one_final_decision_per_quiz(self):
        evaluation_id = self.completed()
        first = self.review().record_decision(self.quiz_id, evaluation_id, "APPROVE", REVIEWER)
        data = self.decision_path().read_bytes()
        again = self.review().record_decision(self.quiz_id, evaluation_id, "APPROVE", REVIEWER)
        self.assertEqual(again, first)  # an identical request returns the stored record
        conflicts = {"other decision": ("REJECT", REVIEWER, "no", None),
                     "other decider": ("APPROVE", "creator-lee", None, None),
                     "other reason": ("APPROVE", REVIEWER, "now with a reason", None)}
        for name, (decision, decided_by, reason, items) in conflicts.items():
            with self.subTest(case=name), self.assertRaises(CreatorReviewRefused) as caught:
                self.review().record_decision(self.quiz_id, evaluation_id, decision, decided_by, reason, items)
            self.assertEqual(caught.exception.reasons, ("decisionAlreadyRecorded",))
        other_evaluation = self.completed()
        with self.assertRaises(CreatorReviewRefused):  # another evaluation is another reviewed provenance
            self.review().record_decision(self.quiz_id, other_evaluation, "APPROVE", REVIEWER)
        self.assertEqual(self.decision_path().read_bytes(), data)  # never overwritten

    def test_concurrent_conflicting_decisions_store_only_one(self):
        evaluation_id = self.completed()
        review = self.review()
        real_create = review.quiz._create_once
        winner = self.review()

        def race(path, data):
            # Another request links its decision between this request's existence check and its write.
            winner.record_decision(self.quiz_id, evaluation_id, "REJECT", "creator-lee", "too easy")
            return real_create(path, data)

        with mock.patch.object(review.quiz, "_create_once", side_effect=race):
            with self.assertRaises(CreatorReviewRefused) as caught:
                review.record_decision(self.quiz_id, evaluation_id, "APPROVE", REVIEWER)
        self.assertEqual(caught.exception.reasons, ("decisionAlreadyRecorded",))
        stored = json.loads(self.decision_path().read_text(encoding="utf-8"))
        self.assertEqual((stored["decision"], stored["decidedBy"]), ("REJECT", "creator-lee"))
        self.assert_status(REJECTED)

    def test_a_write_failure_is_refused_leaves_nothing_and_can_be_retried(self):
        evaluation_id = self.completed()
        review = self.review()
        with mock.patch.object(review.quiz, "_create_once", side_effect=PermissionError("denied")):
            with self.assertRaises(CreatorReviewRefused) as caught:  # the aggregation write fails first
                review.record_decision(self.quiz_id, evaluation_id, "APPROVE", REVIEWER)
        self.assertEqual(caught.exception.reasons, ("aggregationArtifactWriteFailed",))
        review.review_packet(self.quiz_id, evaluation_id)  # the aggregation now exists
        for error in (PermissionError("denied"), OSError("disk full")):
            with self.subTest(error=type(error).__name__):
                with mock.patch.object(review.quiz, "_create_once", side_effect=error):
                    with self.assertRaises(CreatorReviewRefused) as caught:
                        review.record_decision(self.quiz_id, evaluation_id, "APPROVE", REVIEWER)
                self.assertEqual(caught.exception.reasons, ("decisionWriteFailed",))
                self.assertIs(caught.exception.__cause__, error)
                self.assertFalse(self.decision_path().exists())
                self.assert_status(PENDING)
        review.record_decision(self.quiz_id, evaluation_id, "APPROVE", REVIEWER)
        self.assert_status(APPROVED)

    # Status, staleness and the gate

    def test_pending_until_a_decision_is_recorded(self):
        self.assert_status(PENDING)
        self.assert_gate_refused()
        self.assertEqual(self.review().decision_status("not-an-id")["status"], CORRUPTED)

    def test_a_manual_approval_is_stale_once_the_judge_completes(self):
        evaluation_id, _ = self.evaluate(FakeOpenAI(openai_response("{")))
        self.review().record_decision(self.quiz_id, evaluation_id, "APPROVE", REVIEWER, "Judge failed; read it myself")
        self.review().require_creator_approved(self.quiz_id)
        self.evaluate(FakeOpenAI(ok()), evaluation_id)  # explicit attempt 2 completes
        self.assert_status(STALE, ["judgeCompletedAfterManualApproval"])
        self.assert_gate_refused()
        # The decision is final: the same Quiz operation cannot be approved again.
        with self.assertRaises(CreatorReviewRefused) as caught:
            self.review().record_decision(self.quiz_id, evaluation_id, "APPROVE", REVIEWER, "Judge failed; read it myself")
        self.assertEqual(caught.exception.reasons, ("decisionAlreadyRecorded",))

    def test_a_manual_approval_is_stale_when_the_judge_state_changes(self):
        evaluation_id, _ = self.evaluate(FakeOpenAI(openai_response("{")))
        self.review().record_decision(self.quiz_id, evaluation_id, "APPROVE", REVIEWER, "reason")
        self.evaluate(FakeOpenAI(openai_response("{")), evaluation_id)  # now exhausted
        self.assert_status(STALE, ["judgeStateChanged"])
        self.assert_gate_refused()

    def test_approvals_are_stale_after_upstream_changes(self):
        tampering = {
            "quiz output": (lambda quiz_id, evaluation_id: (self.production / "operations" / quiz_id / "attempts" / "1"
                                                            / "output.json").write_text('{"output": {}}', encoding="ascii"),
                            "quizNotJudgeReady"),
            "judge result": (lambda quiz_id, evaluation_id: self.rewrite(
                self.judge_dir(evaluation_id, quiz_id) / "attempts" / "1" / "result.json",
                lambda r: r["pointwise"][0]["koreanQuality"].update(verdict="fail")), "evaluationState:corrupted"),
            "judge result bytes": (lambda quiz_id, evaluation_id: (lambda path: (lambda data: (path.unlink(), path.write_bytes(data)))(
                path.read_bytes() + b"\n"))(self.judge_dir(evaluation_id, quiz_id) / "attempts" / "1" / "result.json"),
                "judgeResultChanged"),
            "aggregation artifact": (lambda quiz_id, evaluation_id: self.rewrite(
                next((self.judge_dir(evaluation_id, quiz_id) / "aggregations").glob("*.json")),
                lambda a: a["aggregation"]["counts"].update(fail=9)), "aggregationArtifactMismatch"),
            "aggregation artifact removed": (lambda quiz_id, evaluation_id: next(
                (self.judge_dir(evaluation_id, quiz_id) / "aggregations").glob("*.json")).unlink(),
                "aggregationArtifactMissing"),
        }
        for name, (tamper, reason) in tampering.items():
            with self.subTest(case=name):
                quiz_id = self.quiz()
                evaluation_id = self.completed(quiz_id=quiz_id)
                self.review().record_decision(quiz_id, evaluation_id, "APPROVE", REVIEWER)
                self.review().require_creator_approved(quiz_id)
                tamper(quiz_id, evaluation_id)
                self.assert_status(STALE, [reason], quiz_id=quiz_id)
                self.assert_gate_refused(quiz_id)

    def test_an_aggregation_policy_change_makes_an_approval_stale(self):
        evaluation_id = self.completed()
        self.review().record_decision(self.quiz_id, evaluation_id, "APPROVE", REVIEWER)
        from src import production_judge_aggregation
        with mock.patch.object(production_judge_aggregation, "AGGREGATION_POLICY_VERSION", "production-judge-aggregation-v2"):
            self.assert_status(STALE, ["aggregationChanged"])
            self.assert_gate_refused()

    def test_damaged_decision_records_are_corrupted_and_never_replaced(self):
        tampering = {
            "kind chosen by hand": lambda r: r.update(decisionKind="JUDGE_OVERRIDE"),
            "rejection without reason": lambda r: r.update(decision="REJECT", decisionKind="CREATOR_REJECTION"),
            "other operation": lambda r: r.update(quizOperationId=new_operation_id()),
            "bad timestamp": lambda r: r.update(decidedAt="2026-99-99T25:61:61+00:00"),
            "unknown field": lambda r: r.update(publishedAt="2026-10-08T00:00:00+00:00"),
            "missing field": lambda r: r.pop("review"),
            "decided by whitespace": lambda r: r.update(decidedBy=" creator-kim "),
            "routing and summary disagree": lambda r: r["review"].update(semanticSummary="HAS_FAIL"),
            "manual approval of a corrupted judge": lambda r: (
                r.update(decisionKind="MANUAL_WITHOUT_JUDGE", reason="by hand", judge=None),
                r["review"].update(executionStatus="corrupted", semanticSummary="UNAVAILABLE",
                                   reviewRouting="JUDGE_UNAVAILABLE", aggregationId=None, aggregationPolicy=None)),
            "manual approval of an unknown state": lambda r: (
                r.update(decisionKind="MANUAL_WITHOUT_JUDGE", reason="by hand", judge=None),
                r["review"].update(executionStatus="mystery", semanticSummary="UNAVAILABLE",
                                   reviewRouting="JUDGE_UNAVAILABLE", aggregationId=None, aggregationPolicy=None)),
        }
        for name, change in tampering.items():
            with self.subTest(case=name):
                quiz_id = self.quiz()
                evaluation_id = self.completed(quiz_id=quiz_id)
                self.review().record_decision(quiz_id, evaluation_id, "APPROVE", REVIEWER)
                self.rewrite(self.decision_path(quiz_id), change)
                data = self.decision_path(quiz_id).read_bytes()
                self.assert_status(CORRUPTED, ["decisionRecordInvalid"], quiz_id=quiz_id)
                self.assert_gate_refused(quiz_id)
                with self.assertRaises(CreatorReviewRefused):
                    self.review().record_decision(quiz_id, evaluation_id, "APPROVE", REVIEWER)
                self.assertEqual(self.decision_path(quiz_id).read_bytes(), data)
        for name, damage in {"not json": lambda path: path.write_bytes(b"{"),
                             "directory": lambda path: path.mkdir()}.items():
            with self.subTest(case=name):
                quiz_id = self.quiz()
                damage(self.decision_path(quiz_id))
                self.assert_status(CORRUPTED, quiz_id=quiz_id)
                self.assert_gate_refused(quiz_id)
                with self.assertRaises(CreatorReviewRefused):
                    self.review().record_decision(quiz_id, self.completed(quiz_id=quiz_id), "APPROVE", REVIEWER)

    def test_a_judge_corrupted_after_approval_is_not_approved(self):
        evaluation_id = self.completed()
        self.review().record_decision(self.quiz_id, evaluation_id, "APPROVE", REVIEWER)
        (self.judge_dir(evaluation_id) / "evaluation.json").unlink()
        self.assert_status(STALE, ["evaluationState:corrupted"])
        self.assert_gate_refused()

    def test_an_aggregation_artifact_removed_during_verification_is_never_recreated(self):
        def removed_on(read, before_read):
            """A _read_json that removes the aggregation artifact as it is about to be (or is being) read."""
            def hooked(path):
                if path.parent.name == "aggregations":
                    path.unlink()
                    if not before_read:
                        raise OSError("file vanished while reading")
                return read(path)
            return hooked

        for name, before_read in (("removed after it was located", True), ("removed while reading", False)):
            with self.subTest(case=name):
                quiz_id = self.quiz()
                evaluation_id = self.completed(quiz_id=quiz_id)
                self.review().record_decision(quiz_id, evaluation_id, "APPROVE", REVIEWER)
                artifact = next((self.judge_dir(evaluation_id, quiz_id) / "aggregations").glob("*.json"))
                decision = self.decision_path(quiz_id).read_bytes()
                review = self.review()
                hooked_read = removed_on(review.quiz._read_json, before_read)
                with mock.patch.object(review.quiz, "_create_once", wraps=review.quiz._create_once) as create, \
                        mock.patch.object(review.quiz, "_read_json", side_effect=hooked_read):
                    status = review.decision_status(quiz_id)
                    self.assertEqual(status["status"], STALE)
                    self.assertEqual(status["reasons"], ["aggregationArtifactMissing" if before_read
                                                         else "aggregationArtifactMismatch"])
                    with self.assertRaises(CreatorReviewRefused):
                        review.require_creator_approved(quiz_id)
                self.assertEqual(create.call_count, 0)  # nothing was recreated
                self.assertFalse(artifact.exists())
                self.assertEqual(list((self.judge_dir(evaluation_id, quiz_id) / "aggregations").iterdir()), [])
                self.assertEqual(self.decision_path(quiz_id).read_bytes(), decision)
                self.assert_status(STALE, ["aggregationArtifactMissing"], quiz_id=quiz_id)

    def test_an_approval_of_result_a_never_passes_through_a_result_b_verified_in_the_same_check(self):
        for entry in ("decision_status", "require_creator_approved"):
            with self.subTest(entry=entry):
                quiz_id = self.quiz()
                evaluation_id = self.completed(quiz_id=quiz_id)
                self.review().record_decision(quiz_id, evaluation_id, "APPROVE", REVIEWER)  # approves result A
                result_path = self.judge_dir(evaluation_id, quiz_id) / "attempts" / "1" / "result.json"
                aggregations = self.judge_dir(evaluation_id, quiz_id) / "aggregations"
                artifact_a = next(aggregations.glob("*.json"))
                artifact_a_bytes = artifact_a.read_bytes()
                decision = self.decision_path(quiz_id).read_bytes()
                review = self.review()
                real_expected = review.aggregator.expected_aggregation
                setup = self.review()  # its own objects: its writes are not counted below
                calls = []

                def changing_after_first(operation_id, evaluation):
                    value = real_expected(operation_id, evaluation)
                    if not calls:
                        # Right after result A was computed and compared, the Judge result becomes B
                        # (same verdicts, other bytes) and a valid aggregation of B appears.
                        data = result_path.read_bytes()
                        result_path.unlink()
                        result_path.write_bytes(data + b"\n")
                        setup.aggregator.aggregate(operation_id, evaluation)
                    calls.append(value)
                    return value

                with mock.patch.object(review.aggregator, "expected_aggregation", side_effect=changing_after_first), \
                        mock.patch.object(review.quiz, "_create_once", wraps=review.quiz._create_once) as create:
                    if entry == "decision_status":
                        status = review.decision_status(quiz_id)
                        self.assertNotEqual(status["status"], APPROVED)
                        self.assertEqual((status["status"], status["reasons"]),
                                         (STALE, ["judgeResultChangedDuringVerification"]))
                    else:
                        with self.assertRaises(CreatorReviewRefused):
                            review.require_creator_approved(quiz_id)
                self.assertEqual(create.call_count, 0)  # status and gate never create or repair
                self.assertNotEqual(calls[0], calls[-1])  # A was seen first, B later
                self.assertEqual(len(list(aggregations.glob("*.json"))), 2)  # A and B only, nothing else
                self.assertEqual(artifact_a.read_bytes(), artifact_a_bytes)
                self.assertEqual(self.decision_path(quiz_id).read_bytes(), decision)
                # Afterwards the approval of A is plainly stale against B.
                self.assert_status(STALE, ["judgeResultChanged"], quiz_id=quiz_id)
                self.assert_gate_refused(quiz_id)

    def test_status_and_gate_never_write(self):
        evaluation_id = self.completed(FLAGGED)
        self.review().record_decision(self.quiz_id, evaluation_id, "APPROVE", REVIEWER, "ok", FLAGGED_ITEMS)
        before = self.files()
        self.assert_status(APPROVED)
        self.review().require_creator_approved(self.quiz_id)
        self.assertEqual(self.files(), before)


if __name__ == "__main__":
    unittest.main()
