"""Runtime validation of Judge output, applied identically to every Provider.

Stages map to error categories: parse_error -> schema_error -> semantic_error. Semantic checks
are structural only (coverage, ranges, duplicates, empty values). They never inspect which
verdict or winner a Judge chose, so a valid but unexpected judgement is never retried away.
"""

import json

from src.judge_contract import DEFECT_TYPES, POINTWISE_ITEMS, QUALITY_ITEMS, SET_LABELS, schema_for


class JudgeOutputError(Exception):
    def __init__(self, category, detail=""):
        super().__init__(category + (": " + detail if detail else ""))
        self.category = category


def _reject_duplicate_keys(pairs):
    # A repeated key makes the output ambiguous; never let the last value win silently.
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def parse_strict(text):
    def reject_constant(_value):
        raise ValueError("Non-finite JSON value")
    if not isinstance(text, str):
        raise JudgeOutputError("parse_error", "output is not text")
    try:
        return json.loads(text, parse_constant=reject_constant, object_pairs_hook=_reject_duplicate_keys)
    except ValueError as exc:
        raise JudgeOutputError("parse_error", str(exc)) from exc


def _check(value, schema, path):
    expected = schema.get("type")
    if expected == "object":
        if not isinstance(value, dict):
            raise JudgeOutputError("schema_error", path + " must be an object")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False and set(value) - set(properties):
            raise JudgeOutputError("schema_error", path + " has unexpected keys")
        for key in schema.get("required", ()):
            if key not in value:
                raise JudgeOutputError("schema_error", path + " is missing " + key)
        for key, child in properties.items():
            if key in value:
                _check(value[key], child, path + "." + key)
    elif expected == "array":
        if not isinstance(value, list):
            raise JudgeOutputError("schema_error", path + " must be an array")
        for index, item in enumerate(value):
            _check(item, schema["items"], "%s[%d]" % (path, index))
    elif expected == "string":
        if not isinstance(value, str):
            raise JudgeOutputError("schema_error", path + " must be a string")
    elif expected == "integer":
        if type(value) is not int:
            raise JudgeOutputError("schema_error", path + " must be an integer")
    else:
        raise ValueError("Unsupported schema type in Judge contract")
    if "enum" in schema and value not in schema["enum"]:
        raise JudgeOutputError("schema_error", path + " is outside the allowed values")


def validate_schema(value, kind):
    _check(value, schema_for(kind), "$")


def _reason(value, path):
    if not value.strip():
        raise JudgeOutputError("semantic_error", path + " reason is empty")


def validate_pointwise(text, expected_indexes):
    """Return {questionIndex: {item: {verdict, reason}}} or raise JudgeOutputError."""
    parsed = parse_strict(text)
    validate_schema(parsed, "pointwise")
    seen = [question["questionIndex"] for question in parsed["questions"]]
    if sorted(seen) != sorted(expected_indexes) or len(set(seen)) != len(seen):
        raise JudgeOutputError("semantic_error", "questionIndex coverage does not match the input")
    result = {}
    for question in parsed["questions"]:
        index = question["questionIndex"]
        for item in POINTWISE_ITEMS:
            _reason(question[item]["reason"], "question %d %s" % (index, item))
        result[index] = {item: {"verdict": question[item]["verdict"], "reason": question[item]["reason"]}
                         for item in POINTWISE_ITEMS}
    return result


def validate_pairwise(text, set_indexes):
    """``set_indexes`` maps SET_1/SET_2 to the questionIndexes shown for that set."""
    parsed = parse_strict(text)
    validate_schema(parsed, "pairwise")
    result = {"defectiveQuestions": {}, "quality": {}}
    for label in SET_LABELS:
        allowed = set(set_indexes[label])
        seen = set()
        defects = []
        for defect in parsed[label]["defectiveQuestions"]:
            index = defect["questionIndex"]
            types = defect["defectTypes"]
            if index not in allowed or index in seen:
                raise JudgeOutputError("semantic_error", label + " defective questionIndex is invalid or repeated")
            if not types or len(set(types)) != len(types) or any(value not in DEFECT_TYPES for value in types):
                raise JudgeOutputError("semantic_error", label + " defectTypes must be non-empty and unique")
            _reason(defect["reason"], "%s question %d" % (label, index))
            seen.add(index)
            defects.append({"questionIndex": index, "defectTypes": list(types), "reason": defect["reason"]})
        result["defectiveQuestions"][label] = sorted(defects, key=lambda item: item["questionIndex"])
    for item in QUALITY_ITEMS:
        _reason(parsed["quality"][item]["reason"], "quality " + item)
        result["quality"][item] = {"winner": parsed["quality"][item]["winner"],
                                   "reason": parsed["quality"][item]["reason"]}
    return result
