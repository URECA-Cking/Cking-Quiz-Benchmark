"""Fixed LLM-as-a-Judge contract: rubric, prompts and Provider-facing JSON schemas.

The Judge evaluates A/B Quiz sets against the human-approved contentText only. Generation
provider/model identity, Human Evaluation and validator results are never part of a prompt.
"""

import hashlib
import json


JUDGE_PROMPT_VERSION = "judge-prompt-v1"
RUBRIC_VERSION = "judge-rubric-v1"
SCHEMA_VERSION = "judge-schema-v1"

VERDICTS = ("pass", "fail", "uncertain")
POINTWISE_ITEMS = ("textAnswerCorrect", "uniqueAnswer", "evidenceSupportsAnswer", "questionClarity",
                   "koreanQuality", "distractorQuality", "textFaithfulness")
DEFECT_TYPES = ("answerIncorrect", "notUnique", "evidenceUnsupported", "unfaithful")
QUALITY_ITEMS = ("clarity", "koreanQuality", "distractorQuality", "coverage")
WINNERS = ("SET_1", "SET_2", "TIE")
SET_LABELS = ("SET_1", "SET_2")
# Only these generated fields are shown to a Judge; everything else in a result row stays hidden.
QUESTION_FIELDS = ("question", "options", "correctOptionIndex", "explanation", "sourceEvidence")


POINTWISE_INSTRUCTIONS = """당신은 한국어 객관식 퀴즈를 평가하는 평가자입니다.
아래 [contentText]만을 사실의 근거로 사용하세요. 영상이나 일반 지식으로 판단하지 마세요.
[questions]의 각 문항(questionIndex)마다 아래 7개 항목을 각각 pass, fail, uncertain 중 하나로 판정하고 짧은 근거(reason)를 쓰세요.
근거가 부족해 판정할 수 없으면 uncertain을 사용하세요.

- textAnswerCorrect: correctOptionIndex가 가리키는 정답이 contentText 기준으로 맞는가.
- uniqueAnswer: contentText 기준으로 정답으로 성립하는 보기가 하나뿐인가. 둘 이상의 보기가 실제 정답으로 성립할 때만 fail입니다. 표현이 모호하더라도 두 번째 정답 보기가 실제로 성립하지 않으면 fail로 판정하지 말고 questionClarity에서 평가하세요.
- evidenceSupportsAnswer: sourceEvidence가 정답을 실제로 뒷받침하는가.
- questionClarity: 질문과 보기가 명확하고 해석이 모호하지 않은가.
- koreanQuality: 한국어 표현이 자연스럽고 문법적으로 적절한가.
- distractorQuality: 오답 보기가 무의미하거나 지나치게 쉬운 오답이 아닌가.
- textFaithfulness: 질문, 정답 보기, explanation, sourceEvidence가 contentText 밖의 사실을 근거 없이 주장하지 않는가. 의도된 오답 보기는 이 판정에서 제외합니다.

입력의 모든 questionIndex를 정확히 한 번씩 평가하고, 입력에 없는 questionIndex는 만들지 마세요."""


PAIRWISE_INSTRUCTIONS = """당신은 같은 [contentText]로 만들어진 두 객관식 퀴즈 세트 SET_1과 SET_2를 비교하는 평가자입니다.
contentText만을 사실의 근거로 사용하세요. 영상이나 일반 지식으로 판단하지 마세요.

1. 유효성 결함: 각 세트에서 아래 결함이 하나 이상 있는 문항을 defectiveQuestions에 questionIndex, defectTypes, reason으로 나열하세요. 결함이 없으면 빈 배열을 반환하세요.
- answerIncorrect: 정답이 contentText 기준으로 맞지 않음.
- notUnique: contentText 기준으로 둘 이상의 보기가 실제 정답으로 성립함. 표현만 모호하고 두 번째 정답이 성립하지 않으면 결함이 아니며 clarity에서 평가합니다.
- evidenceUnsupported: sourceEvidence가 정답을 실제로 뒷받침하지 않음.
- unfaithful: 질문, 정답 보기, explanation, sourceEvidence가 contentText 밖의 사실을 근거 없이 주장함. 의도된 오답 보기는 제외합니다.

2. 품질 비교: 아래 4개 항목마다 더 나은 세트를 SET_1 또는 SET_2로 고르고 근거(reason)를 쓰세요. 명확하고 실질적인 차이가 없으면 TIE를 반환하세요. 사소한 문체 차이나 취향 수준의 차이로 승자를 정하지 마세요.
- clarity: 질문과 보기가 더 명확하고 해석이 덜 모호한 세트.
- koreanQuality: 한국어가 더 자연스럽고 정확한 세트.
- distractorQuality: 오답 보기가 더 그럴듯하고 변별력 있는 세트. 오답이 실제로 정답으로도 성립하는 문제는 여기가 아니라 notUnique로 처리합니다.
- coverage: contentText의 서로 다른 핵심 정보를 더 넓고 중복 없이 다루는 세트. 문항 수는 근거가 아니며 유효성 결함 문항은 coverage에 기여하지 않습니다.

종합 승자는 출력하지 마세요."""


def canonical_json(value):
    """Deterministic UTF-8 JSON used for identity hashes; rejects NaN/Infinity."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def sha256_hex(value):
    data = value if isinstance(value, str) else canonical_json(value)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _verdict_item():
    return {"type": "object", "additionalProperties": False,
            "properties": {"verdict": {"type": "string", "enum": list(VERDICTS)},
                           "reason": {"type": "string"}},
            "required": ["verdict", "reason"]}


def pointwise_schema():
    properties = {"questionIndex": {"type": "integer"}}
    properties.update((item, _verdict_item()) for item in POINTWISE_ITEMS)
    question = {"type": "object", "additionalProperties": False, "properties": properties,
                "required": ["questionIndex"] + list(POINTWISE_ITEMS)}
    return {"type": "object", "additionalProperties": False,
            "properties": {"questions": {"type": "array", "items": question}},
            "required": ["questions"]}


def pairwise_schema():
    defect = {"type": "object", "additionalProperties": False,
              "properties": {"questionIndex": {"type": "integer"},
                             "defectTypes": {"type": "array",
                                             "items": {"type": "string", "enum": list(DEFECT_TYPES)}},
                             "reason": {"type": "string"}},
              "required": ["questionIndex", "defectTypes", "reason"]}
    validity = {"type": "object", "additionalProperties": False,
                "properties": {"defectiveQuestions": {"type": "array", "items": defect}},
                "required": ["defectiveQuestions"]}
    winner = {"type": "object", "additionalProperties": False,
              "properties": {"winner": {"type": "string", "enum": list(WINNERS)},
                             "reason": {"type": "string"}},
              "required": ["winner", "reason"]}
    quality = {"type": "object", "additionalProperties": False,
               "properties": {item: winner for item in QUALITY_ITEMS},
               "required": list(QUALITY_ITEMS)}
    return {"type": "object", "additionalProperties": False,
            "properties": {"SET_1": validity, "SET_2": validity, "quality": quality},
            "required": ["SET_1", "SET_2", "quality"]}


def schema_for(kind):
    if kind == "pointwise":
        return pointwise_schema()
    if kind == "pairwise":
        return pairwise_schema()
    raise ValueError("Unknown Judge kind")


def _visible_questions(questions):
    """``questions`` maps original questionIndex -> generated question; order by index."""
    return [dict({"questionIndex": index}, **{key: questions[index][key] for key in QUESTION_FIELDS})
            for index in sorted(questions)]


def pointwise_prompt(content_text, questions):
    payload = json.dumps(_visible_questions(questions), ensure_ascii=False, indent=2)
    return (POINTWISE_INSTRUCTIONS + "\n\n[contentText]\n" + content_text
            + "\n[/contentText]\n\n[questions]\n" + payload + "\n[/questions]")


def pairwise_prompt(content_text, set_1, set_2):
    return (PAIRWISE_INSTRUCTIONS + "\n\n[contentText]\n" + content_text + "\n[/contentText]\n\n"
            + "[SET_1]\n" + json.dumps(_visible_questions(set_1), ensure_ascii=False, indent=2)
            + "\n[/SET_1]\n\n[SET_2]\n" + json.dumps(_visible_questions(set_2), ensure_ascii=False, indent=2)
            + "\n[/SET_2]")


def contract_fingerprint():
    """Hash of everything in this contract that changes a Judge measurement's meaning."""
    return sha256_hex({"promptVersion": JUDGE_PROMPT_VERSION, "rubricVersion": RUBRIC_VERSION,
                       "schemaVersion": SCHEMA_VERSION, "pointwiseInstructions": POINTWISE_INSTRUCTIONS,
                       "pairwiseInstructions": PAIRWISE_INSTRUCTIONS,
                       "pointwiseSchema": pointwise_schema(), "pairwiseSchema": pairwise_schema()})
