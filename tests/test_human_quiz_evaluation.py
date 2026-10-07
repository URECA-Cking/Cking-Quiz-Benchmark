"""Blind Human Quiz Evaluation for Pilot v2 (temporary repositories and fake Providers only)."""

import copy
import hashlib
import json
import re
import shutil
import tempfile
import threading
import unittest
from datetime import datetime
from pathlib import Path

from src.human_quiz_evaluation import HumanQuizEvaluation
from src.pilot_runner import FixtureProvider, PilotRunner


ROOT = Path(__file__).resolve().parents[1]
CONTENT = "NASA는 30분마다 전 세계의 비와 눈을 측정하고 토양 수분 변화를 관측한다."
WATER, METHANE, MARS, KARI = ("nasa-water-cycle-2019", "nasa-methane-2020", "nasa-mars-organics-2025",
                              "kari-microgravity-2024")
GEMINI, OPENAI = "gemini-3.8-flash", "gpt-5.4-mini"
RAW = {"source": "fixture"}
REVIEW = {item: "pass" for item in ("factualAccuracy", "keyInformationCoverage", "factsConsistency",
                                    "koreanConsistency", "contentTextContractCompliance")}


def quiz(evidence="전 세계의 비와 눈", count=3, index=0):
    return {"promptVersion": "pilot-v2", "questions": [
        {"question": "질문 %d은 무엇인가?" % number, "options": ["가", "나", "다", "라"], "correctOptionIndex": index,
         "explanation": "영상 설명 %d" % number, "sourceEvidence": evidence} for number in range(count)]}


class Simulated(FixtureProvider):
    """Offline double recorded as an actual API run."""

    is_actual_api = True


def answer(kind, payload):
    return Simulated({kind: {"normalized": payload, "responseBody": RAW}})


def failure(kind):
    return Simulated({kind: {"errorCategory": "server_error"}})


def filled(export, verdict="pass"):
    submission = copy.deepcopy(export)
    for video in submission["videos"]:
        for blind in video["sets"]:
            blind["setReview"].update(coverage=verdict, redundancy=verdict, learningValue=verdict)
            for review in blind["questionReviews"]:
                review.update({item: verdict for item in ("answerAccuracy", "uniqueAnswer", "evidenceSupportsAnswer",
                                                          "videoGrounding", "koreanQuality", "hallucination")})
    return submission


class HumanQuizEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repository = Path(self.temp.name)
        (self.repository / "configs").mkdir()
        (self.repository / "data").mkdir()
        for name in ("pilot.yaml", "pilot-v2.yaml"):
            shutil.copyfile(ROOT / "configs" / name, self.repository / "configs" / name)
        shutil.copyfile(ROOT / "data" / "videos.jsonl", self.repository / "data" / "videos.jsonl")
        self.results = self.repository / "results"
        self.v2 = PilotRunner(self.repository, self.results, self.repository / "configs" / "pilot-v2.yaml")
        self.v1 = PilotRunner(self.repository, self.results)
        self.runs = {}
        # WATER: A/B/C eligible. C quotes the video, not contentText (the residual blinding limitation).
        water = self.grounding(WATER)
        self.runs["water-A"] = self.quiz(GEMINI, WATER, water)
        self.runs["water-B"] = self.quiz(OPENAI, WATER, water)
        self.runs["water-C"] = self.direct(WATER, answer("direct", quiz("global rain and snow")))
        # METHANE: A eligible despite a validator failure; B has 2 questions; C fails to parse.
        methane = self.grounding(METHANE)
        self.runs["methane-A"] = self.quiz(GEMINI, METHANE, methane, quiz("영상에 없는 문장"))
        self.runs["methane-B"] = self.quiz(OPENAI, METHANE, methane, quiz(count=2))
        self.runs["methane-C"] = self.direct(METHANE, answer("direct", "{broken"))
        # MARS: A cannot be displayed (answer index out of range); B only failed technically; no C.
        mars = self.grounding(MARS)
        self.runs["mars-A"] = self.quiz(GEMINI, MARS, mars, quiz(index=7))
        self.runs["mars-B"] = self.quiz(OPENAI, MARS, mars, provider=failure("quiz"))
        self.tool = HumanQuizEvaluation(self.repository, self.results)

    def tearDown(self):
        self.temp.cleanup()

    # ----- fixtures ------------------------------------------------------------------------

    def grounding(self, video_id, repetition=1):
        row = self.v2.run_grounding(video_id, "gemini_video", repetition,
                                    answer("grounding", {"contentText": CONTENT, "facts": []}))
        self.v2.review_grounding(row["runId"], "grounding-reviewer", REVIEW)
        return row["runId"]

    def quiz(self, model, video_id, source, payload=None, provider=None, repetition=1):
        return self.v2.run_quiz(video_id, model, repetition, CONTENT, provider or answer("quiz", payload or quiz()),
                                source_grounding_run_id=source)

    def direct(self, video_id, provider, repetition=1):
        return self.v2.run_end_to_end(video_id, "gemini_direct_quiz", repetition, provider)

    def session_dir(self, session_id):
        return self.results / "human" / session_id

    def load(self, session_id, name):
        return json.loads((self.session_dir(session_id) / name).read_text(encoding="utf-8"))

    def pilot_snapshot(self):
        return {path.relative_to(self.results).as_posix(): path.read_bytes() for path in self.results.rglob("*")
                if path.is_file() and "human" not in path.relative_to(self.results).parts
                and path.name != PilotRunner.LOCK_FILE}

    def assert_rejected(self, session_id, submission, message="", reviewer="reviewer-a"):
        with self.assertRaisesRegex(ValueError, message):
            self.tool.import_evaluation(session_id, submission, reviewer)
        self.assertFalse((self.session_dir(session_id) / "evaluation.json").exists())

    # ----- session and blinding -------------------------------------------------------------

    def test_eligibility_and_not_eligible_reasons_are_recorded(self):
        session = self.load(self.tool.create_session("seed-1"), "session.json")
        eligible = {(item["condition"], item["videoId"]): item["runId"] for item in session["items"]}
        self.assertEqual(eligible, {("A", WATER): self.runs["water-A"]["runId"],
                                    ("B", WATER): self.runs["water-B"]["runId"],
                                    ("C", WATER): self.runs["water-C"]["runId"],
                                    ("A", METHANE): self.runs["methane-A"]["runId"]})
        # A validator failure alone (evidence not in contentText) does not exclude a displayable set.
        self.assertEqual(self.runs["methane-A"]["validatorStatus"], "fail")
        reasons = {(record["condition"], record["videoId"], record["repetition"]): record
                   for record in session["notEligible"]}
        self.assertEqual(reasons[("B", METHANE, 1)]["reason"], "questionCountMismatch")
        self.assertEqual(reasons[("C", METHANE, 1)]["reason"], "parseFailed")
        self.assertEqual(reasons[("A", MARS, 1)]["reason"], "invalidQuestionStructure")
        self.assertEqual(reasons[("A", MARS, 1)]["questionProblems"][0]["problem"], "invalidCorrectOptionIndex")
        self.assertEqual(reasons[("B", MARS, 1)]["reason"], "noApiSuccess")
        self.assertEqual([(item["apiStatus"], item["errorCategory"]) for item in reasons[("B", MARS, 1)]["attempts"]],
                         [("error", "server_error")])
        # Never attempted is notRun, not a technical failure; both keep their attempt history.
        self.assertEqual((reasons[("C", MARS, 1)]["reason"], reasons[("C", MARS, 1)]["attempts"]), ("notRun", []))
        self.assertEqual({record["reason"] for key, record in reasons.items() if key[1] == KARI}, {"notRun"})
        self.assertTrue(reasons[("B", MARS, 1)]["attempts"])
        self.assertEqual(reasons[("B", METHANE, 1)]["attempts"][0]["questionCount"], 2)
        # Every planned condition is accounted for: 4 videos x 2 repetitions x A/B/C.
        self.assertEqual(len(session["items"]) + len(session["notEligible"]), 4 * 2 * 3)

    def test_same_seed_reproduces_the_blind_order_and_export(self):
        first, second = self.tool.create_session("fixed-seed"), self.tool.create_session("fixed-seed")
        exports = [self.load(session_id, "blind-export.json") for session_id in (first, second)]
        sessions = [self.load(session_id, "session.json") for session_id in (first, second)]
        for export in exports:
            export.pop("sessionId")
        self.assertEqual(exports[0], exports[1])
        self.assertEqual([(item["blindId"], item["videoRef"], item["runId"]) for item in sessions[0]["items"]],
                         [(item["blindId"], item["videoRef"], item["runId"]) for item in sessions[1]["items"]])
        self.assertEqual((sessions[0]["seed"], sessions[0]["orderingAlgorithm"]), ("fixed-seed", "sha256-rank-v1"))
        # The order is the documented SHA-256 rank, recomputed here independently.
        water = [item for item in sessions[0]["items"] if item["videoId"] == WATER]
        expected = sorted(water, key=lambda item: hashlib.sha256(
            ("fixed-seed|set|" + item["runId"]).encode("utf-8")).hexdigest())
        self.assertEqual(water, expected)

    def test_different_seeds_can_change_the_order(self):
        orders = set()
        for number in range(8):
            session = self.load(self.tool.create_session("seed-%d" % number), "session.json")
            orders.add(tuple(item["runId"] for item in session["items"]))
        self.assertGreater(len(orders), 1)

    def test_omitted_seed_is_generated_once_and_stored(self):
        session = self.load(self.tool.create_session(), "session.json")
        self.assertRegex(session["seed"], r"^[0-9a-f]{32}$")
        for bad in ("", " padded", "a\nb", 1, "x" * 201):
            with self.assertRaises(ValueError):
                self.tool.create_session(bad)

    def test_blind_id_maps_back_to_the_original_quiz_output(self):
        session_id = self.tool.create_session("map")
        session, export = self.load(session_id, "session.json"), self.load(session_id, "blind-export.json")
        exported = {blind["blindId"]: (video, blind) for video in export["videos"] for blind in video["sets"]}
        self.assertEqual(len(exported), len(session["items"]))
        urls = {json.loads(line)["videoId"]: json.loads(line)["youtubeUrl"]
                for line in (self.repository / "data" / "videos.jsonl").read_text(encoding="utf-8").splitlines()}
        for item in session["items"]:
            video, blind = exported[item["blindId"]]
            original = json.loads((self.results / "evaluation" / (item["runId"] + ".json")).read_text(encoding="utf-8"))
            self.assertEqual(video["videoRef"], item["videoRef"])
            self.assertEqual(video["youtubeUrl"], urls[item["videoId"]])
            self.assertEqual([{key: value for key, value in question.items() if key != "questionIndex"}
                              for question in blind["questions"]],
                             [{key: question[key] for key in ("question", "options", "correctOptionIndex",
                                                              "explanation", "sourceEvidence")}
                              for question in original["questions"]])

    def test_blind_export_is_an_exact_allowlist_without_condition_metadata(self):
        session_id = self.tool.create_session("allowlist")
        export = self.load(session_id, "blind-export.json")
        session = self.load(session_id, "session.json")
        self.assertEqual(set(export), {"format", "sessionId", "rubricVersion", "videos"})
        for video in export["videos"]:
            self.assertEqual(set(video), {"videoRef", "youtubeUrl", "sets"})
            self.assertRegex(video["videoRef"], r"^V\d{2}$")
            for blind in video["sets"]:
                self.assertEqual(set(blind), {"blindId", "questions", "setReview", "questionReviews"})
                self.assertRegex(blind["blindId"], r"^S\d{3}$")
                self.assertEqual(blind["setReview"], {"coverage": None, "redundancy": None,
                                                      "learningValue": None, "reviewNote": None})
                for question in blind["questions"]:
                    self.assertEqual(set(question), {"questionIndex", "question", "options", "correctOptionIndex",
                                                     "explanation", "sourceEvidence"})
                    self.assertTrue(question["sourceEvidence"])  # kept for evidenceSupportsAnswer
                for review in blind["questionReviews"]:
                    self.assertEqual(set(review), {"questionIndex", "answerAccuracy", "uniqueAnswer",
                                                   "evidenceSupportsAnswer", "videoGrounding", "koreanQuality",
                                                   "hallucination", "reviewNote"})
                    self.assertTrue(all(value is None for key, value in review.items() if key != "questionIndex"))
        text = (self.session_dir(session_id) / "blind-export.json").read_text(encoding="utf-8")
        hidden = [GEMINI, OPENAI, "gemini", "openai", "fixed_content_text", "gemini_direct_quiz", "pilot-v2",
                  "pilot-v1", "condition", "provider", "model", "method", "runId", "promptVersion",
                  "contentText", "repetition", "attempt", "facts", "evidence_not_in_content", "results", CONTENT,
                  WATER, METHANE, MARS, KARI]
        for item in session["items"]:
            hidden += [item["runId"], item["evaluationSha256"], item.get("sourceGroundingRunId") or "runId",
                       item.get("contentTextSha256") or "runId"]
        for value in hidden:
            self.assertNotIn(value, text)
        for label in ('"A"', '"B"', '"C"'):
            self.assertNotIn(label, text)

    def test_pilot_v1_results_are_never_in_a_session(self):
        source = self.v1.run_grounding(KARI, "gemini_video", 1, answer("grounding", {"contentText": CONTENT,
                                                                                      "facts": []}))
        self.v1.approve_content(source["runId"], "v1-approver")
        v1_quiz = self.v1.run_quiz(KARI, GEMINI, 1, CONTENT, answer("quiz", dict(quiz(), promptVersion="pilot-v1")),
                                   source_grounding_run_id=source["runId"])
        session = self.load(self.tool.create_session("v1"), "session.json")
        text = json.dumps(session)
        self.assertNotIn(v1_quiz["runId"], text)
        self.assertIn({"condition": "A", "videoId": KARI, "repetition": 1, "reason": "notRun",
                       "runId": None, "attempts": []}, session["notEligible"])

    def test_duplicate_api_success_in_one_condition_fails_closed(self):
        rows = (self.results / "quiz-generation.jsonl").read_text(encoding="utf-8").splitlines()
        original = json.loads(rows[0])
        duplicate = dict(original, runId="f" * 32, attempt=2)
        with (self.results / "quiz-generation.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(duplicate) + "\n")
        for directory in ("raw", "evaluation"):
            shutil.copyfile(self.results / directory / (original["runId"] + ".json"),
                            self.results / directory / (duplicate["runId"] + ".json"))
        with self.assertRaisesRegex(ValueError, "More than one API success"):
            self.tool.create_session("dup")
        self.assertFalse((self.results / "human").exists())

    def test_unverifiable_ab_source_fails_closed(self):
        path = self.results / "quiz-generation.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for row in rows:
            if row["runId"] == self.runs["water-B"]["runId"]:
                row["contentTextSha256"] = "0" * 64
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unverifiable source"):
            self.tool.create_session("source")
        self.assertFalse((self.results / "human").exists())

    # ----- import -----------------------------------------------------------------------------

    def session(self):
        session_id = self.tool.create_session("import")
        return session_id, self.load(session_id, "blind-export.json")

    def test_import_records_reviewer_and_maps_back_to_original_runs(self):
        before = self.pilot_snapshot()
        session_id, export = self.session()
        submission = filled(export)
        submission["videos"][0]["sets"][0]["setReview"].update(redundancy="fail", reviewNote="1번과 2번이 겹침")
        submission["videos"][0]["sets"][0]["questionReviews"][2].update(koreanQuality="uncertain", reviewNote="어색함")
        evaluation = self.tool.import_evaluation(session_id, submission, "  Reviewer Kim  ")
        stored = self.load(session_id, "evaluation.json")
        self.assertEqual(stored, evaluation)
        self.assertEqual(stored["reviewedBy"], "Reviewer Kim")
        self.assertRegex(stored["reviewedAt"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{6})?\+00:00$")
        datetime.fromisoformat(stored["reviewedAt"])
        session = self.load(session_id, "session.json")
        self.assertEqual([item["runId"] for item in stored["sets"]], [item["runId"] for item in session["items"]])
        first = stored["sets"][0]
        self.assertEqual(first["blindId"], submission["videos"][0]["sets"][0]["blindId"])
        self.assertEqual(first["setReview"]["redundancy"], "fail")
        self.assertEqual(first["questionReviews"][2]["koreanQuality"], "uncertain")
        # Pilot results, including the v2 summary questionReviews, are never written.
        self.assertEqual(self.pilot_snapshot(), before)
        quiz_rows = [json.loads(line) for line in (self.results / "quiz-generation.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertTrue(all(review["answerAccuracy"] is None for row in quiz_rows for review in row["questionReviews"]))
        self.v2._check_storage_integrity()  # results/human does not disturb Pilot storage integrity

    def test_invalid_submissions_are_rejected_without_writing(self):
        session_id, export = self.session()

        def variant(change):
            submission = filled(export)
            change(submission)
            return submission

        first = lambda submission: submission["videos"][0]["sets"][0]
        cases = [
            ("verdict", lambda s: first(s)["questionReviews"][0].update(answerAccuracy="PASS")),
            ("verdict", lambda s: first(s)["questionReviews"][0].update(hallucination=None)),
            ("verdict", lambda s: first(s)["setReview"].update(coverage=True)),
            ("verdict", lambda s: first(s)["setReview"].update(learningValue=None)),
            ("exactly its exported blind sets", lambda s: s["videos"][0]["sets"].pop()),
            ("exactly its exported blind sets", lambda s: s["videos"][0]["sets"].append(dict(first(s), blindId="S999"))),
            ("exactly its exported blind sets", lambda s: s["videos"][0]["sets"].append(copy.deepcopy(first(s)))),
            ("Unknown video", lambda s: s["videos"].append({"videoRef": "V99", "youtubeUrl": "x", "sets": []})),
            ("Duplicate video", lambda s: s["videos"].append(dict(s["videos"][0], sets=[]))),
            ("missing videos", lambda s: s["videos"].pop()),
            ("video differs", lambda s: s["videos"][0].update(youtubeUrl="https://www.youtube.com/watch?v=other")),
            ("exactly its exported blind sets",
             lambda s: s["videos"][1]["sets"].append(s["videos"][0]["sets"].pop())),
            ("questionIndex", lambda s: first(s)["questionReviews"][2].update(questionIndex=1)),
            ("questionIndex", lambda s: first(s)["questionReviews"].pop()),
            ("Unexpected set entry", lambda s: first(s).update(condition="A")),
            ("questionReviews entries", lambda s: first(s)["questionReviews"][0].update(model=GEMINI)),
            ("setReview", lambda s: first(s)["setReview"].update(overall="pass")),
            ("reviewNote", lambda s: first(s)["setReview"].update(reviewNote=3)),
            ("does not match", lambda s: s.update(sessionId="0" * 32)),
            ("does not match", lambda s: s.update(extra=1)),
            ("differs from the blind export", lambda s: first(s)["questions"][0].update(question="바뀐 질문")),
        ]
        for message, change in cases:
            with self.subTest(message=message):
                self.assert_rejected(session_id, variant(change), message)
        for reviewer in ("", "   ", "a\nb", None, "x" * 101):
            with self.subTest(reviewer=reviewer):
                self.assert_rejected(session_id, filled(export), "reviewed_by", reviewer)

    def test_second_import_is_rejected_and_keeps_the_first_evaluation(self):
        session_id, export = self.session()
        self.tool.import_evaluation(session_id, filled(export), "reviewer-a")
        stored = (self.session_dir(session_id) / "evaluation.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "already has an imported evaluation"):
            self.tool.import_evaluation(session_id, filled(export, "fail"), "reviewer-b")
        self.assertEqual((self.session_dir(session_id) / "evaluation.json").read_bytes(), stored)
        # Re-evaluation uses a new session.
        again, export_again = self.session()
        self.assertNotEqual(again, session_id)
        self.tool.import_evaluation(again, filled(export_again, "fail"), "reviewer-b")

    def test_changed_or_missing_source_runs_fail_closed(self):
        session_id, export = self.session()
        item = self.load(session_id, "session.json")["items"][0]
        output = self.results / "evaluation" / (item["runId"] + ".json")
        original = output.read_bytes()
        output.write_bytes(original + b"\n")
        self.assert_rejected(session_id, filled(export), "changed")
        output.write_bytes(original)
        # Removing the run (and its linked payloads, so Pilot storage stays consistent) is refused too.
        name = "quiz-generation.jsonl" if item["benchmarkType"] == "quiz_generation" else "end-to-end.jsonl"
        path = self.results / name
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if item["runId"] not in line]
        path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
        for directory in ("raw", "evaluation"):
            (self.results / directory / (item["runId"] + ".json")).unlink()
        self.assert_rejected(session_id, filled(export), "missing")

    def test_grounding_rejection_leaves_ab_not_run_without_blocking_the_session(self):
        rejected = self.v2.run_grounding(KARI, "gemini_video", 1, answer("grounding", {"contentText": CONTENT,
                                                                                       "facts": []}))
        self.v2.review_grounding(rejected["runId"], "grounding-reviewer", dict(REVIEW, factualAccuracy="fail"))
        with self.assertRaisesRegex(ValueError, "approved"):  # A/B are never run from a rejected Grounding
            self.quiz(GEMINI, KARI, rejected["runId"])
        before = self.pilot_snapshot()
        session = self.load(self.tool.create_session("rejected"), "session.json")
        reasons = {(record["condition"], record["videoId"], record["repetition"]): record
                   for record in session["notEligible"]}
        for condition in ("A", "B"):
            self.assertEqual((reasons[(condition, KARI, 1)]["reason"], reasons[(condition, KARI, 1)]["attempts"]),
                             ("notRun", []))
        self.assertEqual(self.pilot_snapshot(), before)

    def rewrite_row(self, run_id, **changes):
        for name in ("quiz-generation.jsonl", "end-to-end.jsonl"):
            path = self.results / name
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            if any(row["runId"] == run_id for row in rows):
                path.write_text("".join(json.dumps(dict(row, **changes) if row["runId"] == run_id else row) + "\n"
                                        for row in rows), encoding="utf-8")
                return
        raise AssertionError("row not found")

    def test_identity_changes_with_the_same_quiz_output_fail_closed(self):
        session_id, export = self.session()
        items = {item["condition"]: item for item in self.load(session_id, "session.json")["items"]
                 if item["videoId"] == WATER}
        cases = ((items["C"], {"method": "gemini_grounding_gemini_quiz"}), (items["C"], {"questionCount": 4}),
                 (items["A"], {"method": "other_method"}), (items["B"], {"questionCount": 2}),
                 (items["A"], {"repetition": 2}))
        for item, changes in cases:
            with self.subTest(condition=item["condition"], changes=changes):
                path = self.results / ("quiz-generation.jsonl" if item["benchmarkType"] == "quiz_generation"
                                       else "end-to-end.jsonl")
                original = path.read_bytes()
                output = (self.results / "evaluation" / (item["runId"] + ".json")).read_bytes()
                self.rewrite_row(item["runId"], **changes)
                self.assertEqual(hashlib.sha256(output).hexdigest(), item["evaluationSha256"])  # same output
                self.assert_rejected(session_id, filled(export), "changed|Unexpected")
                path.write_bytes(original)
        self.tool.import_evaluation(session_id, filled(export), "reviewer-a")  # restored sources import

    def test_swapped_private_mapping_is_caught_by_the_question_binding(self):
        session_id, export = self.session()
        session_path = self.session_dir(session_id) / "session.json"
        session = json.loads(session_path.read_text(encoding="utf-8"))
        # Two sets of the same video whose questions differ (C quotes the video, A the contentText).
        a = next(index for index, item in enumerate(session["items"]) if item["videoId"] == WATER and item["condition"] == "A")
        c = next(index for index, item in enumerate(session["items"]) if item["videoId"] == WATER and item["condition"] == "C")
        self.assertEqual(session["items"][a]["videoRef"], session["items"][c]["videoRef"])
        identity = lambda item: {key: value for key, value in item.items() if key not in ("blindId", "videoRef")}
        first, second = session["items"][a], session["items"][c]
        session["items"][a] = dict(identity(second), blindId=first["blindId"], videoRef=first["videoRef"])
        session["items"][c] = dict(identity(first), blindId=second["blindId"], videoRef=second["videoRef"])
        session_path.write_text(json.dumps(session), encoding="utf-8")
        # Everything else stays consistent: same export (SHA still matches), each mapped run and output intact.
        self.assertEqual(hashlib.sha256((self.session_dir(session_id) / "blind-export.json").read_bytes()).hexdigest(),
                         session["blindExportSha256"])
        self.assert_rejected(session_id, filled(export), "does not show the questions of its mapped source run")
        # With the original mapping restored, the same submission imports.
        session["items"][a], session["items"][c] = first, second
        session_path.write_text(json.dumps(session), encoding="utf-8")
        self.tool.import_evaluation(session_id, filled(export), "reviewer-a")

    def test_swapped_blind_ids_with_identical_questions_are_caught_by_mapping_integrity(self):
        session_id, export = self.session()
        session_path = self.session_dir(session_id) / "session.json"
        original = session_path.read_bytes()
        session = json.loads(original)
        a, b = (next(item for item in session["items"] if item["videoId"] == WATER and item["condition"] == condition)
                for condition in ("A", "B"))
        shown = {blind["blindId"]: blind["questions"] for video in export["videos"] for blind in video["sets"]}
        # The two runs show identical questions, so question binding alone cannot tell them apart.
        self.assertNotEqual(a["runId"], b["runId"])
        self.assertEqual(shown[a["blindId"]], shown[b["blindId"]])
        submission = filled(export)
        for video in submission["videos"]:
            for blind in video["sets"]:
                if blind["blindId"] == b["blindId"]:  # different verdicts make a wrong attribution meaningful
                    blind["setReview"].update(coverage="fail", redundancy="fail", learningValue="fail")
        a_id, b_id = a["blindId"], b["blindId"]
        a["blindId"], b["blindId"] = b_id, a_id  # only the blindIds are exchanged
        session_path.write_text(json.dumps(session), encoding="utf-8")
        self.assertEqual(hashlib.sha256((self.session_dir(session_id) / "blind-export.json").read_bytes()).hexdigest(),
                         session["blindExportSha256"])
        self.assert_rejected(session_id, submission, "mapping integrity mismatch")
        # Restored mapping; reordering the item list alone does not change the assignment and is accepted.
        restored = json.loads(original)
        restored["items"].reverse()
        session_path.write_text(json.dumps(restored), encoding="utf-8")
        evaluation = self.tool.import_evaluation(session_id, submission, "reviewer-a")
        verdicts = {item["runId"]: item["setReview"]["coverage"] for item in evaluation["sets"]}
        self.assertEqual((verdicts[a["runId"]], verdicts[b["runId"]]), ("pass", "fail"))

    def test_malformed_unknown_or_incomplete_sessions_fail_closed(self):
        session_id, export = self.session()
        with self.assertRaisesRegex(ValueError, "Invalid Human Evaluation sessionId"):
            self.tool.import_evaluation("../" + session_id, filled(export), "reviewer-a")
        with self.assertRaisesRegex(ValueError, "Unknown Human Evaluation session"):
            self.tool.import_evaluation("a" * 32, filled(export), "reviewer-a")
        session_path = self.session_dir(session_id) / "session.json"
        original = session_path.read_bytes()
        for broken in (b"{", b"[]", json.dumps(dict(json.loads(original), items=[])).encode("utf-8"),
                       json.dumps(dict(json.loads(original), blindExportSha256="0" * 64)).encode("utf-8")):
            with self.subTest(broken=broken[:20]):
                session_path.write_bytes(broken)
                self.assert_rejected(session_id, filled(export), "Malformed")
        session_path.write_bytes(original)
        (self.session_dir(session_id) / "blind-export.json").write_text("{}", encoding="utf-8")
        self.assert_rejected(session_id, filled(export), "Malformed")
        session_path.unlink()  # session.json is written last: without it the session is incomplete
        self.assert_rejected(session_id, filled(export), "Incomplete")

    def test_busy_results_directory_fails_fast(self):
        session_id, export = self.session()
        holder = HumanQuizEvaluation(self.repository, self.results)
        acquired, release, errors = threading.Event(), threading.Event(), []

        def hold():
            try:
                with holder.runner._results_lock():
                    acquired.set()
                    release.wait(timeout=10)
            except Exception as exc:  # pragma: no cover - surfaced below
                errors.append(exc)

        thread = threading.Thread(target=hold)
        thread.start()
        try:
            self.assertTrue(acquired.wait(timeout=10))
            with self.assertRaisesRegex(ValueError, "already in progress"):
                self.tool.import_evaluation(session_id, filled(export), "reviewer-a")
            with self.assertRaisesRegex(ValueError, "already in progress"):
                self.tool.create_session("busy")
        finally:
            release.set()
            thread.join(timeout=10)
        self.assertEqual(errors, [])
        self.assertFalse((self.session_dir(session_id) / "evaluation.json").exists())
        self.assertEqual(len([path for path in (self.results / "human").iterdir()]), 1)
        self.tool.import_evaluation(session_id, filled(export), "reviewer-a")  # lock released

    def test_session_creation_keeps_pilot_results_and_integrity(self):
        before = self.pilot_snapshot()
        session_id = self.tool.create_session("keep")
        self.assertEqual(self.pilot_snapshot(), before)
        self.assertEqual(sorted(path.name for path in self.session_dir(session_id).iterdir()),
                         ["blind-export.json", "session.json"])
        self.v2._check_storage_integrity()
        self.v1._check_storage_integrity()
        tmp_files = [path for path in self.results.rglob("*.tmp")]
        self.assertEqual(tmp_files, [])


if __name__ == "__main__":
    unittest.main()
