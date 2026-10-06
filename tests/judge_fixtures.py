"""Synthetic Pilot results shaped like repetition 1, for Provider-free Judge tests."""

import hashlib
import json
import shutil
from pathlib import Path

from src.judge_contract import POINTWISE_ITEMS, QUALITY_ITEMS


ROOT = Path(__file__).resolve().parents[1]
VIDEOS = ("nasa-water-cycle-2019", "nasa-methane-2020", "nasa-mars-organics-2025", "kari-microgravity-2024")
MODELS = {"A": "gemini-3.8-flash", "B": "gpt-5.4-mini"}
PASS_REVIEW = {"answerAccuracy": "pass", "uniqueAnswer": "pass", "evidenceSupportsAnswer": "pass",
               "videoGrounding": "pass", "koreanQuality": "pass", "hallucination": "pass", "reviewNote": None}
# Human failures mirroring the real Pilot: Methane A/B one defective question each, KARI B one.
HUMAN_FAILS = {("nasa-methane-2020", "A", 1): {"answerAccuracy": "fail", "videoGrounding": "fail"},
               ("nasa-methane-2020", "B", 1): {"answerAccuracy": "fail", "uniqueAnswer": "fail"},
               ("kari-microgravity-2024", "B", 1): {"uniqueAnswer": "fail", "koreanQuality": "fail"}}


def question(video_id, condition, index):
    return {"question": "%s %s 질문 %d?" % (video_id, condition, index),
            "options": ["정답", "오답 하나", "오답 둘", "오답 셋"], "correctOptionIndex": 0,
            "explanation": "본문에 정답이 있습니다.", "sourceEvidence": "근거 문장 %d" % index}


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def build_synthetic_pilot(repository):
    repository = Path(repository)
    for folder in ("configs", "data", "results/raw", "results/evaluation"):
        (repository / folder).mkdir(parents=True, exist_ok=True)
    for name in ("configs/pilot.yaml", "configs/judge.yaml", "data/videos.jsonl"):
        shutil.copyfile(ROOT / name, repository / name)
    results = repository / "results"
    grounding_rows, quiz_rows, e2e_rows = [], [], []
    for number, video_id in enumerate(VIDEOS):
        failed_id = "%032x" % (100 + number)
        grounding_rows.append({"benchmarkType": "video_grounding", "runId": failed_id, "videoId": video_id,
                               "method": "gemini_video", "model": MODELS["A"], "promptVersion": None,
                               "repetition": 1, "attempt": 1, "apiStatus": "error",
                               "errorCategory": "server_error", "httpStatus": 503})
        grounding_id = "%032x" % (200 + number)
        content = "%s 승인된 contentText 본문입니다." % video_id
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        grounding_rows.append({"benchmarkType": "video_grounding", "runId": grounding_id, "videoId": video_id,
                               "method": "gemini_video", "model": MODELS["A"], "promptVersion": None,
                               "repetition": 1, "attempt": 2, "apiStatus": "success", "errorCategory": None,
                               "contentTextSha256": digest, "contentTextApprovalStatus": "approved",
                               "groundingFacts": [], "omission": "pass", "hallucination": "pass"})
        _write_json(results / "evaluation" / (grounding_id + ".json"), {"contentText": content, "facts": []})
        _write_json(results / "raw" / (grounding_id + ".json"), {"source": "fixture"})
        for condition in ("A", "B"):
            quiz_id = "%032x" % (300 + number * 2 + (condition == "B"))
            count = 1 if (video_id == "nasa-mars-organics-2025" and condition == "B") else 3
            reviews = [dict(PASS_REVIEW, questionIndex=index, **HUMAN_FAILS.get((video_id, condition, index), {}))
                       for index in range(count)]
            quiz_rows.append({"benchmarkType": "quiz_generation", "runId": quiz_id, "videoId": video_id,
                              "method": "fixed_content_text", "model": MODELS[condition], "attempt": 1,
                              "promptVersion": "pilot-v1", "repetition": 1, "apiStatus": "success",
                              "errorCategory": "quiz_contract_error" if count != 3 else None,
                              "contentTextSha256": digest, "sourceGroundingRunId": grounding_id,
                              "parseStatus": "pass", "validatorStatus": "pass" if count == 3 else "fail",
                              "beCompatibility": "pass" if count == 3 else "fail", "questionCount": count,
                              "questionReviews": reviews})
            _write_json(results / "evaluation" / (quiz_id + ".json"),
                        {"promptVersion": "pilot-v1",
                         "questions": [question(video_id, condition, index) for index in range(count)]})
            _write_json(results / "raw" / (quiz_id + ".json"), {"source": "fixture"})
        e2e_rows.append({"benchmarkType": "end_to_end", "runId": "%032x" % (500 + number), "videoId": video_id,
                         "method": "gemini_direct_quiz", "model": MODELS["A"], "promptVersion": "pilot-v1",
                         "repetition": 1, "attempt": 1, "apiStatus": "error", "errorCategory": "server_error"})
    for name, rows in (("video-grounding.jsonl", grounding_rows), ("quiz-generation.jsonl", quiz_rows),
                       ("end-to-end.jsonl", e2e_rows)):
        (results / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                                    encoding="utf-8")
    return results


def pointwise_output(indexes, verdict="pass"):
    return json.dumps({"questions": [dict({"questionIndex": index}, **{
        item: {"verdict": verdict, "reason": "근거"} for item in POINTWISE_ITEMS}) for index in indexes]},
        ensure_ascii=False)


def pairwise_output(set_1_defects=(), set_2_defects=(), quality=None):
    quality = quality or {}
    return json.dumps({
        "SET_1": {"defectiveQuestions": [{"questionIndex": index, "defectTypes": ["answerIncorrect"],
                                          "reason": "근거"} for index in set_1_defects]},
        "SET_2": {"defectiveQuestions": [{"questionIndex": index, "defectTypes": ["answerIncorrect"],
                                          "reason": "근거"} for index in set_2_defects]},
        "quality": {item: {"winner": quality.get(item, "TIE"), "reason": "근거"} for item in QUALITY_ITEMS}},
        ensure_ascii=False)


def default_outcome(measurement):
    """Valid output: Pointwise all pass; Pairwise no defects and quality ties (TIE decided by tie)."""
    if measurement["kind"] == "pointwise":
        return {"text": pointwise_output(measurement["expectedQuestionIndexes"]),
                "usage": {"inputTokens": 10, "cachedInputTokens": None, "outputTokens": 5, "reasoningTokens": 2}}
    return {"text": pairwise_output(), "usage": None}
