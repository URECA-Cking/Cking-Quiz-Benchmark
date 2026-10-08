import hashlib
import json
import os
import unittest
from unittest import mock

from src.judge_contract import sha256_hex
from src.production_creator_review import ProductionCreatorReview
from src.production_publish import (CORRUPTED, CURRENT, NOT_RECORDED, PUBLICATION_POLICY_VERSION, STALE,
                                    LocalPublicationRefused, ProductionLocalPublisher, publish_id)
from tests import test_production_judge as judge_tests
from tests.test_production_judge import FakeOpenAI, ok, verdicts


REVIEWER = "creator-kim"
PUBLISHER = "publisher-park"


class ProductionLocalPublicationTest(unittest.TestCase):
    # The Production Judge fixture: a Judge-ready Quiz in a temporary repository and a fake OpenAI transport.
    setUp = judge_tests.ProductionJudgeTest.setUp
    tearDown = judge_tests.ProductionJudgeTest.tearDown
    quiz = judge_tests.ProductionJudgeTest.quiz
    evaluate = judge_tests.ProductionJudgeTest.evaluate
    judge_dir = judge_tests.ProductionJudgeTest.judge_dir
    rewrite = staticmethod(judge_tests.ProductionJudgeTest.rewrite)

    def publisher(self):
        return ProductionLocalPublisher(self.repository)

    def approved(self, decision="APPROVE", reason=None):
        """A new Quiz operation with a completed all-pass Judge evaluation and a Creator decision."""
        quiz_id = self.quiz()
        evaluation_id, _ = self.evaluate(FakeOpenAI(ok(verdicts())), quiz_id=quiz_id)
        ProductionCreatorReview(self.repository).record_decision(quiz_id, evaluation_id, decision, REVIEWER, reason)
        return quiz_id, evaluation_id

    def operation(self, quiz_id):
        return self.production / "operations" / quiz_id

    def files(self):
        return {str(path): path.read_bytes() for path in self.production.rglob("*") if path.is_file()}

    def assert_status(self, quiz_id, status, reasons=None):
        result = self.publisher().publication_status(quiz_id)
        self.assertEqual(result["status"], status, result["reasons"])
        if reasons is not None:
            self.assertEqual(result["reasons"], reasons)
        return result

    def assert_refused_without_record(self, quiz_id, published_by=PUBLISHER):
        with self.assertRaises(LocalPublicationRefused) as caught:
            self.publisher().record_publication(quiz_id, published_by)
        self.assertFalse((self.operation(quiz_id) / "publication.json").exists())
        return caught.exception

    # Normal flow

    def test_an_approved_quiz_is_recorded_once_and_is_current(self):
        quiz_id, _ = self.approved()
        record = self.publisher().record_publication(quiz_id, PUBLISHER)
        gate = ProductionCreatorReview(self.repository).require_creator_approved(quiz_id)
        decision_bytes = (self.operation(quiz_id) / "creator-decision.json").read_bytes()
        self.assertEqual(record["creatorDecisionSha256"], hashlib.sha256(decision_bytes).hexdigest())
        self.assertEqual(record["creatorDecision"], gate["decision"])
        self.assertEqual(record["payload"], {"contentText": gate["contentText"], "questions": gate["questions"]})
        self.assertEqual(record["payloadSha256"], sha256_hex(record["payload"]))
        self.assertEqual(record["publishId"], publish_id(quiz_id, record["creatorDecisionSha256"], record["payloadSha256"]))
        self.assertEqual((record["policyVersion"], record["publishedBy"]), (PUBLICATION_POLICY_VERSION, PUBLISHER))
        stored = (self.operation(quiz_id) / "publication.json").read_bytes()
        again = self.publisher().record_publication(quiz_id, PUBLISHER)  # identical request
        self.assertEqual(again, record)
        self.assertEqual((self.operation(quiz_id) / "publication.json").read_bytes(), stored)
        status = self.assert_status(quiz_id, CURRENT, [])
        self.assertEqual(status["publication"], record)

    def test_the_decision_sha_is_the_sha_of_the_actual_file_bytes(self):
        quiz_id, _ = self.approved()
        path = self.operation(quiz_id) / "creator-decision.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        path.unlink()
        path.write_bytes(json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"))  # same record, other bytes
        record = self.publisher().record_publication(quiz_id, PUBLISHER)
        self.assertEqual(record["creatorDecisionSha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertNotEqual(record["creatorDecisionSha256"], sha256_hex(data))
        self.assert_status(quiz_id, CURRENT)

    # Refused without an approval

    def test_only_a_current_creator_approval_can_be_recorded(self):
        pending = self.quiz()
        self.assertIn("creatorNotApproved", self.assert_refused_without_record(pending).reasons)
        rejected, _ = self.approved("REJECT", "off-topic")
        self.assertIn("creatorStatus:CREATOR_REJECTED", self.assert_refused_without_record(rejected).reasons)
        changes = {
            "stale: judge result changed": lambda quiz_id, evaluation_id: (lambda path: (lambda data: (
                path.unlink(), path.write_bytes(data + b"\n")))(path.read_bytes()))(
                self.judge_dir(evaluation_id, quiz_id) / "attempts" / "1" / "result.json"),
            "aggregation damaged": lambda quiz_id, evaluation_id: self.rewrite(
                next((self.judge_dir(evaluation_id, quiz_id) / "aggregations").glob("*.json")),
                lambda a: a["aggregation"]["counts"].update(fail=5)),
            "quiz no longer eligible": lambda quiz_id, evaluation_id: (
                self.operation(quiz_id) / "attempts" / "1" / "output.json").write_text('{"output": {}}', encoding="ascii"),
            "decision corrupted": lambda quiz_id, evaluation_id: self.rewrite(
                self.operation(quiz_id) / "creator-decision.json", lambda r: r.update(decisionKind="JUDGE_OVERRIDE")),
            "decision for another operation": lambda quiz_id, evaluation_id: self.rewrite(
                self.operation(quiz_id) / "creator-decision.json", lambda r: r.update(quizOperationId="0" * 32)),
        }
        for name, change in changes.items():
            with self.subTest(case=name):
                quiz_id, evaluation_id = self.approved()
                change(quiz_id, evaluation_id)
                refused = self.assert_refused_without_record(quiz_id)
                self.assertIn("creatorNotApproved", refused.reasons)
        for invalid in (" ", None, "a​b"):
            with self.subTest(published_by=invalid):
                quiz_id, _ = self.approved()
                self.assertEqual(self.assert_refused_without_record(quiz_id, invalid).reasons, ("invalidPublishedBy",))

    # One record per operation

    def test_conflicting_requests_never_replace_the_record(self):
        quiz_id, _ = self.approved()
        self.publisher().record_publication(quiz_id, PUBLISHER)
        path = self.operation(quiz_id) / "publication.json"
        stored = path.read_bytes()
        with self.assertRaises(LocalPublicationRefused) as caught:
            self.publisher().record_publication(quiz_id, "publisher-lee")
        self.assertEqual(caught.exception.reasons, ("publicationAlreadyRecorded",))
        with mock.patch("src.production_publish.PUBLICATION_POLICY_VERSION", "production-local-publication-policy-v2"):
            with self.assertRaises(LocalPublicationRefused):  # another policy is another publication
                self.publisher().record_publication(quiz_id, PUBLISHER)
        self.assertEqual(path.read_bytes(), stored)

    def test_concurrent_requests_store_one_record(self):
        for name, other_publisher, expect_refused in (("same request", PUBLISHER, False),
                                                      ("conflicting request", "publisher-lee", True)):
            with self.subTest(case=name):
                quiz_id, _ = self.approved()
                publisher = self.publisher()
                real_create = publisher.quiz._create_once
                other = self.publisher()

                def race(path, data):
                    other.record_publication(quiz_id, other_publisher)  # links its record first
                    return real_create(path, data)

                with mock.patch.object(publisher.quiz, "_create_once", side_effect=race):
                    if expect_refused:
                        with self.assertRaises(LocalPublicationRefused) as caught:
                            publisher.record_publication(quiz_id, PUBLISHER)
                        self.assertEqual(caught.exception.reasons, ("publicationAlreadyRecorded",))
                    else:
                        record = publisher.record_publication(quiz_id, PUBLISHER)
                        self.assertEqual(record["publishedBy"], PUBLISHER)
                stored = json.loads((self.operation(quiz_id) / "publication.json").read_text(encoding="utf-8"))
                self.assertEqual(stored["publishedBy"], other_publisher)
                self.assertEqual(len(list(self.operation(quiz_id).glob("publication*"))), 1)

    def test_a_write_failure_leaves_no_record_and_can_be_retried(self):
        quiz_id, _ = self.approved()
        publisher = self.publisher()
        for error in (PermissionError("denied"), OSError("disk full")):
            with self.subTest(error=type(error).__name__):
                with mock.patch.object(publisher.quiz, "_create_once", side_effect=error):
                    with self.assertRaises(LocalPublicationRefused) as caught:
                        publisher.record_publication(quiz_id, PUBLISHER)
                self.assertEqual(caught.exception.reasons, ("publicationWriteFailed",))
                self.assertIs(caught.exception.__cause__, error)
                self.assertFalse((self.operation(quiz_id) / "publication.json").exists())
        publisher.record_publication(quiz_id, PUBLISHER)
        self.assert_status(quiz_id, CURRENT)

    # Stability before and after writing

    def test_a_decision_change_just_before_writing_is_refused(self):
        quiz_id, _ = self.approved()
        publisher = self.publisher()
        real_read = publisher._decision_bytes
        reads = []

        def changing(operation_id):
            data = real_read(operation_id)
            reads.append(data)
            if len(reads) == 2:  # after the gate passed: the decision file changes before the last check
                path = self.operation(quiz_id) / "creator-decision.json"
                path.unlink()
                path.write_bytes(data + b"\n")
            return data

        with mock.patch.object(publisher, "_decision_bytes", side_effect=changing):
            with self.assertRaises(LocalPublicationRefused) as caught:
                publisher.record_publication(quiz_id, PUBLISHER)
        self.assertEqual(caught.exception.reasons, ("creatorDecisionChangedBeforeRecording",))
        self.assertFalse((self.operation(quiz_id) / "publication.json").exists())

    def test_a_decision_change_during_the_gate_is_refused(self):
        quiz_id, _ = self.approved()
        publisher = self.publisher()
        real_gate = publisher.review.require_creator_approved

        def changing_gate(operation_id):
            approved = real_gate(operation_id)
            path = self.operation(quiz_id) / "creator-decision.json"
            data = path.read_bytes()
            path.unlink()
            path.write_bytes(data + b"\n")
            return approved

        with mock.patch.object(publisher.review, "require_creator_approved", side_effect=changing_gate):
            with self.assertRaises(LocalPublicationRefused) as caught:
                publisher.record_publication(quiz_id, PUBLISHER)
        self.assertEqual(caught.exception.reasons, ("creatorDecisionChangedDuringVerification",))
        self.assertFalse((self.operation(quiz_id) / "publication.json").exists())

    def test_an_upstream_change_right_after_writing_is_reported_and_the_record_kept(self):
        quiz_id, evaluation_id = self.approved()
        publisher = self.publisher()
        real_create = publisher.quiz._create_once
        result_path = self.judge_dir(evaluation_id, quiz_id) / "attempts" / "1" / "result.json"

        def create_then_change(path, data):
            created = real_create(path, data)
            changed = result_path.read_bytes() + b"\n"
            result_path.unlink()
            result_path.write_bytes(changed)
            return created

        with mock.patch.object(publisher.quiz, "_create_once", side_effect=create_then_change):
            with self.assertRaises(LocalPublicationRefused) as caught:
                publisher.record_publication(quiz_id, PUBLISHER)
        self.assertEqual(caught.exception.reasons[0], "publicationRecordedButNotCurrent")
        self.assertTrue((self.operation(quiz_id) / "publication.json").is_file())  # not hidden or deleted
        self.assert_status(quiz_id, STALE)

    # Status

    def test_status_is_read_only_and_tells_history_from_currency(self):
        quiz_id, evaluation_id = self.approved()
        self.assert_status(quiz_id, NOT_RECORDED, [])
        self.publisher().record_publication(quiz_id, PUBLISHER)
        publisher = self.publisher()
        before = self.files()
        with mock.patch.object(publisher.quiz, "_create_once", wraps=publisher.quiz._create_once) as create:
            self.assertEqual(publisher.publication_status(quiz_id)["status"], CURRENT)
        self.assertEqual(create.call_count, 0)
        self.assertEqual(self.files(), before)
        # Upstream changes after publication: the record stays, only its current validity changes.
        result_path = self.judge_dir(evaluation_id, quiz_id) / "attempts" / "1" / "result.json"
        data = result_path.read_bytes()
        result_path.unlink()
        result_path.write_bytes(data + b"\n")
        record_bytes = (self.operation(quiz_id) / "publication.json").read_bytes()
        before = self.files()
        with mock.patch.object(publisher.quiz, "_create_once", wraps=publisher.quiz._create_once) as create:
            status = publisher.publication_status(quiz_id)
        self.assertEqual(status["status"], STALE)
        self.assertIn("creatorNotApproved", status["reasons"])
        self.assertIsNotNone(status["publication"])
        self.assertEqual(create.call_count, 0)
        self.assertEqual(self.files(), before)
        self.assertEqual((self.operation(quiz_id) / "publication.json").read_bytes(), record_bytes)
        with self.assertRaises(LocalPublicationRefused):  # and it cannot be recorded again
            self.publisher().record_publication(quiz_id, PUBLISHER)

    def test_a_damaged_record_is_corrupted_and_never_replaced(self):
        tampering = {
            "missing field": lambda r: r.pop("publishedBy"),
            "unknown field": lambda r: r.update(servicePublished=True),
            "wrong payload hash": lambda r: r.update(payloadSha256="f" * 64),
            "payload changed with its hash": lambda r: (r["payload"].update(contentText="changed"),
                                                        r.update(payloadSha256=sha256_hex(r["payload"]))),
            "wrong decision sha": lambda r: r.update(creatorDecisionSha256="f" * 64),
            "wrong publish id": lambda r: r.update(publishId="f" * 64),
            "other operation": lambda r: r.update(quizOperationId="0" * 32),
            "bad timestamp": lambda r: r.update(publishedAt="2026-99-99T25:61:61+00:00"),
            "wrong type": lambda r: r["payload"].update(questions="three"),
        }
        for name, change in tampering.items():
            with self.subTest(case=name):
                quiz_id, _ = self.approved()
                self.publisher().record_publication(quiz_id, PUBLISHER)
                path = self.operation(quiz_id) / "publication.json"
                self.rewrite(path, change)
                data = path.read_bytes()
                self.assert_status(quiz_id, CORRUPTED)
                with self.assertRaises(LocalPublicationRefused):
                    self.publisher().record_publication(quiz_id, PUBLISHER)
                self.assertEqual(path.read_bytes(), data)
        quiz_id, _ = self.approved()
        self.publisher().record_publication(quiz_id, PUBLISHER)
        self.rewrite(self.operation(quiz_id) / "publication.json",
                     lambda r: r["creatorDecision"].update(reason="edited"))  # snapshot no longer the decision bytes
        self.assert_status(quiz_id, CORRUPTED, ["creatorDecisionSnapshotMismatch"])
        for name, damage in {"not json": lambda path: path.write_bytes(b"{"), "empty": lambda path: path.write_bytes(b""),
                             "directory": lambda path: path.mkdir()}.items():
            with self.subTest(case=name):
                quiz_id, _ = self.approved()
                damage(self.operation(quiz_id) / "publication.json")
                self.assert_status(quiz_id, CORRUPTED)
                with self.assertRaises(LocalPublicationRefused):
                    self.publisher().record_publication(quiz_id, PUBLISHER)

    # Snapshot structure and exact JSON types

    def published(self):
        quiz_id, evaluation_id = self.approved()
        self.publisher().record_publication(quiz_id, PUBLISHER)
        return quiz_id, evaluation_id

    def tamper_consistently(self, quiz_id, change):
        """Change the stored record and recompute its digests, so only structure and type checks can object."""
        def apply(record):
            change(record)
            record["payloadSha256"] = sha256_hex(record["payload"])
            record["publishId"] = publish_id(quiz_id, record["creatorDecisionSha256"], record["payloadSha256"])
        self.rewrite(self.operation(quiz_id) / "publication.json", apply)

    def assert_corrupted_read_only(self, quiz_id):
        publisher = self.publisher()
        before = self.files()
        with mock.patch.object(publisher.quiz, "_create_once", wraps=publisher.quiz._create_once) as create:
            status = publisher.publication_status(quiz_id)
        self.assertEqual((status["status"], status["reasons"]), (CORRUPTED, ["publicationRecordInvalid"]))
        self.assertEqual(create.call_count, 0)
        self.assertEqual(self.files(), before)  # no file created, changed or repaired
        with self.assertRaises(LocalPublicationRefused):
            self.publisher().record_publication(quiz_id, PUBLISHER)
        self.assertEqual(self.files(), before)

    def test_a_bool_in_place_of_an_integer_is_corrupted_even_with_consistent_digests(self):
        cases = {
            "payload questionIndex 0 -> false": lambda r: r["payload"]["questions"][0].update(questionIndex=False),
            "payload questionIndex 1 -> true": lambda r: r["payload"]["questions"][1].update(questionIndex=True),
            "payload correctOptionIndex -> bool": lambda r: r["payload"]["questions"][0].update(correctOptionIndex=False),
            "payload correctOptionIndex -> float": lambda r: r["payload"]["questions"][0].update(correctOptionIndex=0.0),
            "snapshot quiz attempt -> true": lambda r: r["creatorDecision"]["quiz"].update(attempt=True),
            "snapshot judge attempt -> true": lambda r: r["creatorDecision"]["judge"].update(attempt=True),
        }
        for name, change in cases.items():
            with self.subTest(case=name):
                quiz_id, _ = self.published()
                self.tamper_consistently(quiz_id, change)
                self.assert_corrupted_read_only(quiz_id)

    def test_an_incomplete_decision_snapshot_is_corrupted_even_without_the_upstream_decision(self):
        quiz_id, _ = self.published()
        self.tamper_consistently(quiz_id, lambda r: r.update(
            creatorDecision={"quizOperationId": quiz_id, "decision": "APPROVE"}))
        (self.operation(quiz_id) / "creator-decision.json").unlink()  # upstream gone too
        self.assert_corrupted_read_only(quiz_id)

    def test_nested_missing_or_unknown_fields_are_corrupted(self):
        cases = {
            "payload question missing explanation": lambda r: r["payload"]["questions"][0].pop("explanation"),
            "payload question unknown field": lambda r: r["payload"]["questions"][2].update(hint="look closely"),
            "payload unknown field": lambda r: r["payload"].update(promptVersion="production-quiz-v1"),
            "payload question count": lambda r: r["payload"]["questions"].pop(),
            "payload options count": lambda r: r["payload"]["questions"][0]["options"].pop(),
            "payload evidence not in content": lambda r: r["payload"]["questions"][1].update(sourceEvidence="not there"),
            "payload index order": lambda r: r["payload"]["questions"].reverse(),
            "snapshot missing review": lambda r: r["creatorDecision"].pop("review"),
            "snapshot unknown field": lambda r: r["creatorDecision"].update(publishedAt="2026-10-08T00:00:00+00:00"),
            "snapshot review unknown field": lambda r: r["creatorDecision"]["review"].update(score=1),
            "snapshot judge missing field": lambda r: r["creatorDecision"]["judge"].pop("resultSha256"),
            "snapshot rejection": lambda r: r["creatorDecision"].update(decision="REJECT", decisionKind="CREATOR_REJECTION",
                                                                        reason="no"),
        }
        for name, change in cases.items():
            with self.subTest(case=name):
                quiz_id, _ = self.published()
                self.tamper_consistently(quiz_id, change)
                self.assert_corrupted_read_only(quiz_id)

    def test_valid_records_stay_current_and_upstream_changes_stay_stale(self):
        quiz_id, evaluation_id = self.published()
        self.assert_status(quiz_id, CURRENT, [])
        self.tamper_consistently(quiz_id, lambda r: None)  # a consistent rewrite of the same values
        self.assert_status(quiz_id, CURRENT, [])
        self.assertEqual(self.publisher().record_publication(quiz_id, PUBLISHER)["publishedBy"], PUBLISHER)
        result_path = self.judge_dir(evaluation_id, quiz_id) / "attempts" / "1" / "result.json"
        data = result_path.read_bytes()
        result_path.unlink()
        result_path.write_bytes(data + b"\n")
        self.assert_status(quiz_id, STALE)
        (self.operation(quiz_id) / "creator-decision.json").unlink()  # a valid record whose upstream vanished
        self.assert_status(quiz_id, STALE)

    # Execution status of the decision snapshot

    @staticmethod
    def manual_snapshot(record, execution_status):
        """Turn the stored decision snapshot into a manual approval without a Judge result for this status."""
        decision = record["creatorDecision"]
        decision.update(decisionKind="MANUAL_WITHOUT_JUDGE", reason="checked by hand", acknowledgedItems=[], judge=None)
        decision["review"] = {"executionStatus": execution_status, "semanticSummary": "UNAVAILABLE",
                              "reviewRouting": "JUDGE_UNAVAILABLE", "aggregationId": None, "aggregationPolicy": None}

    def test_a_snapshot_with_a_non_reviewable_execution_status_is_corrupted(self):
        for status in ("corrupted", "quiz_not_judge_ready", "mystery", "COMPLETED", ["uncertain"]):
            for upstream in ("present", "deleted"):
                with self.subTest(executionStatus=status, upstream=upstream):
                    quiz_id, _ = self.published()
                    self.tamper_consistently(quiz_id, lambda r: self.manual_snapshot(r, status))
                    if upstream == "deleted":
                        (self.operation(quiz_id) / "creator-decision.json").unlink()
                    self.assert_corrupted_read_only(quiz_id)

    def test_reviewable_execution_statuses_are_valid_snapshots(self):
        from src.production_judge_aggregation import UNAVAILABLE_STATES
        quiz_id, _ = self.published()
        record = json.loads((self.operation(quiz_id) / "publication.json").read_text(encoding="utf-8"))
        review = ProductionCreatorReview(self.repository)
        self.assertTrue(review._record_valid(quiz_id, record["creatorDecision"]))  # completed
        for status in sorted(UNAVAILABLE_STATES):
            with self.subTest(executionStatus=status):
                manual = json.loads(json.dumps(record))
                self.manual_snapshot(manual, status)
                self.assertTrue(review._record_valid(quiz_id, manual["creatorDecision"]))

    def test_a_real_manual_approval_is_published_current_and_stale_without_its_upstream(self):
        quiz_id = self.quiz()
        evaluation_id, _ = self.evaluate(FakeOpenAI(judge_tests.openai_response("{")), quiz_id=quiz_id)  # retryable
        ProductionCreatorReview(self.repository).record_decision(quiz_id, evaluation_id, "APPROVE", REVIEWER,
                                                                 "Judge failed; checked by hand")
        record = self.publisher().record_publication(quiz_id, PUBLISHER)
        self.assertEqual((record["creatorDecision"]["decisionKind"], record["creatorDecision"]["review"]["executionStatus"]),
                         ("MANUAL_WITHOUT_JUDGE", "retryable"))
        self.assert_status(quiz_id, CURRENT, [])
        (self.operation(quiz_id) / "creator-decision.json").unlink()  # a valid record whose upstream vanished
        self.assert_status(quiz_id, STALE)

    def test_record_comparisons_keep_json_types(self):
        from src.production_publish import _same_json
        self.assertFalse(_same_json(0, False))
        self.assertFalse(_same_json(1, True))
        self.assertFalse(_same_json({"a": [0]}, {"a": [False]}))
        self.assertFalse(_same_json(1, 1.0))
        self.assertTrue(_same_json({"a": 1, "b": [1, "x"]}, {"b": [1, "x"], "a": 1}))

    def test_a_record_that_vanishes_while_being_read_is_not_current(self):
        quiz_id, _ = self.approved()
        self.publisher().record_publication(quiz_id, PUBLISHER)
        publisher = self.publisher()
        real_read = publisher._regular_file_bytes

        def vanishing(path):
            if path.name == "publication.json":
                path.unlink()
            return real_read(path)

        with mock.patch.object(publisher, "_regular_file_bytes", side_effect=vanishing):
            status = publisher.publication_status(quiz_id)
        self.assertEqual((status["status"], status["reasons"]), (CORRUPTED, ["publicationRecordUnreadable"]))
        self.assertFalse((self.operation(quiz_id) / "publication.json").exists())  # nothing was recreated

    def test_a_linked_record_is_corrupted(self):
        quiz_id, _ = self.approved()
        target = self.operation(quiz_id).parent / "elsewhere.json"
        target.write_text("{}", encoding="utf-8")
        try:
            os.symlink(target, self.operation(quiz_id) / "publication.json")
        except (OSError, NotImplementedError) as exc:
            self.skipTest("creating symlinks is not permitted in this environment: %s" % type(exc).__name__)
        self.assert_status(quiz_id, CORRUPTED)

    def test_a_record_reported_as_a_link_is_corrupted(self):
        # Runs where real symlinks cannot be created: the read only asks Path.is_symlink().
        quiz_id, _ = self.approved()
        self.publisher().record_publication(quiz_id, PUBLISHER)
        from pathlib import Path
        real = Path.is_symlink
        with mock.patch.object(Path, "is_symlink", lambda path: path.name == "publication.json" or real(path)):
            self.assert_status(quiz_id, CORRUPTED, ["publicationRecordUnreadable"])


if __name__ == "__main__":
    unittest.main()
