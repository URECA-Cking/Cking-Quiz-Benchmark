"""Shared validation of stored Human Evaluation values.

The Pilot runner's storage integrity check and the offline aggregator use the same rules,
matching ``docs/run-result.schema.json``. Only stored values are checked: missing fields and
``null`` (not yet reviewed) are valid, so this is not an evaluation completion check.
"""


REVIEW_VALUES = ("pass", "fail", "uncertain", None)
TOP_LEVEL_REVIEW_KEYS = ("omission", "hallucination", "videoGrounding")
FACT_REVIEW_KEYS = ("factExists", "evidenceTypeCorrect", "timestampAccurate")
QUESTION_REVIEW_VALUE_KEYS = ("answerAccuracy", "uniqueAnswer", "evidenceSupportsAnswer",
                              "videoGrounding", "koreanQuality", "hallucination")
# Fact content keys are allowed but their values are not re-validated here.
GROUNDING_FACT_KEYS = frozenset(("fact", "evidenceType", "evidence", "timestampStartSeconds",
                                 "timestampEndSeconds", "reviewNote") + FACT_REVIEW_KEYS)
QUESTION_REVIEW_KEYS = frozenset(("questionIndex", "reviewNote") + QUESTION_REVIEW_VALUE_KEYS)


def _valid_review(value):
    return value is None or type(value) is str and value in REVIEW_VALUES


def _valid_items(items, allowed_keys, review_keys):
    return isinstance(items, list) and all(
        isinstance(item, dict) and set(item) <= allowed_keys
        and all(_valid_review(item[key]) for key in review_keys if key in item)
        and (item.get("reviewNote") is None or type(item["reviewNote"]) is str)
        for item in items)


def require_valid_human_evaluation(row):
    """Accept valid or legacy (missing) Human Evaluation fields of a result row; else raise."""
    valid = (all(_valid_review(row[key]) for key in TOP_LEVEL_REVIEW_KEYS if key in row)
             and ("groundingFacts" not in row
                  or _valid_items(row["groundingFacts"], GROUNDING_FACT_KEYS, FACT_REVIEW_KEYS))
             and ("questionReviews" not in row
                  or _valid_items(row["questionReviews"], QUESTION_REVIEW_KEYS,
                                  QUESTION_REVIEW_VALUE_KEYS)
                  and all(type(item.get("questionIndex")) is int and item["questionIndex"] >= 0
                          for item in row["questionReviews"])))
    if not valid:
        raise ValueError("Human evaluation is invalid")
