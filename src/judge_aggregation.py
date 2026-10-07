"""Deterministic Judge aggregation and Human comparison. No Provider or file access.

Statuses and preferences are kept in separate fields: a decided result is
``status=RESOLVED`` with ``preference`` A/B/TIE; every other status has ``preference=None``.
"""

from src.judge_contract import QUALITY_ITEMS, SET_LABELS


RESOLVED = "RESOLVED"
INCONSISTENT = "INCONSISTENT"
INCOMPLETE = "INCOMPLETE"
JUDGE_DISAGREEMENT = "JUDGE_DISAGREEMENT"
PREFERENCES = ("A", "B", "TIE")
ORDERS = {"AB": {"SET_1": "A", "SET_2": "B"}, "BA": {"SET_1": "B", "SET_2": "A"}}

# Judge Pointwise item -> (Human field, relation). Items absent here have no Human counterpart.
POINTWISE_HUMAN_MAPPING = {
    "textAnswerCorrect": ("answerAccuracy", "comparableDifferentBasis"),
    "uniqueAnswer": ("uniqueAnswer", "comparableDifferentBasis"),
    "evidenceSupportsAnswer": ("evidenceSupportsAnswer", "comparableDifferentBasis"),
    "koreanQuality": ("koreanQuality", "direct"),
    "textFaithfulness": ("hallucination", "referenceOnly"),
}
H2A_FIELDS = ("answerAccuracy", "uniqueAnswer", "evidenceSupportsAnswer")


def canonicalize_pairwise(order, validated):
    """Convert a validated SET_1/SET_2 Pairwise output into canonical A/B terms."""
    slots = ORDERS[order]
    defective = {slots[label]: validated["defectiveQuestions"][label] for label in SET_LABELS}
    quality = {}
    for item in QUALITY_ITEMS:
        winner = validated["quality"][item]["winner"]
        quality[item] = "TIE" if winner == "TIE" else slots[winner]
    return {"defectiveQuestions": defective, "quality": quality}


def pairwise_call_final(canonical):
    """Validity first by defective QUESTION count, then quality win count; never a full-rubric vote."""
    count_a = len({item["questionIndex"] for item in canonical["defectiveQuestions"]["A"]})
    count_b = len({item["questionIndex"] for item in canonical["defectiveQuestions"]["B"]})
    if count_a != count_b:
        return ("A" if count_a < count_b else "B"), "validity"
    wins_a = sum(1 for winner in canonical["quality"].values() if winner == "A")
    wins_b = sum(1 for winner in canonical["quality"].values() if winner == "B")
    if wins_a != wins_b:
        return ("A" if wins_a > wins_b else "B"), "quality"
    return "TIE", "tie"


def merge_orders(ab, ba):
    """``ab``/``ba`` are canonical call preferences, or None when that order finally failed."""
    if ab is None or ba is None:
        return {"status": INCOMPLETE, "preference": None, "orderAgreement": None}
    if ab not in PREFERENCES or ba not in PREFERENCES:
        raise ValueError("Unknown order preference")
    if ab == ba:
        return {"status": RESOLVED, "preference": ab, "orderAgreement": "full"}
    if "TIE" in (ab, ba):
        return {"status": RESOLVED, "preference": "TIE", "orderAgreement": "partial"}
    return {"status": INCONSISTENT, "preference": None, "orderAgreement": "contradictory"}


def merge_judges(first, second):
    """Combine two Judge-level results symmetrically; no Judge has priority."""
    statuses = (first["status"], second["status"])
    if INCOMPLETE in statuses:
        return {"status": INCOMPLETE, "preference": None, "interJudgeAgreement": "notComparable",
                "ruleId": "anyIncomplete"}
    if INCONSISTENT in statuses:
        return {"status": INCONSISTENT, "preference": None, "interJudgeAgreement": "notComparable",
                "ruleId": "anyInconsistent"}
    if statuses != (RESOLVED, RESOLVED):
        raise ValueError("Unknown Judge-level status")
    left, right = first["preference"], second["preference"]
    if left == right:
        return {"status": RESOLVED, "preference": left, "interJudgeAgreement": "full", "ruleId": "same"}
    if "TIE" in (left, right):
        return {"status": RESOLVED, "preference": "TIE", "interJudgeAgreement": "partial",
                "ruleId": "preferenceWithTie"}
    return {"status": JUDGE_DISAGREEMENT, "preference": None, "interJudgeAgreement": "contradictory",
            "ruleId": "oppositePreferences"}


def compare_verdicts(left, right):
    """Pass/fail agreement; uncertain or missing values are never counted either way."""
    if left is None or right is None:
        return "MISSING"
    if "uncertain" in (left, right):
        return "NOT_COMPARABLE_UNCERTAIN"
    return "AGREE" if left == right else "DISAGREE"


def compare_pointwise_human(item, judge_verdict, human_review):
    mapping = POINTWISE_HUMAN_MAPPING.get(item)
    if mapping is None:
        return {"humanField": None, "relation": "noHumanCounterpart", "outcome": None}
    field, relation = mapping
    human = human_review.get(field) if isinstance(human_review, dict) else None
    return {"humanField": field, "relation": relation, "outcome": compare_verdicts(judge_verdict, human)}


def human_defective_count(reviews, question_indexes):
    """Defective = any H2a field fail; None when a needed value is uncertain or missing."""
    by_index = {review.get("questionIndex"): review for review in reviews if isinstance(review, dict)}
    count = 0
    for index in question_indexes:
        review = by_index.get(index)
        values = [review.get(field) if review else None for field in H2A_FIELDS]
        if any(value not in ("pass", "fail") for value in values):
            return None
        count += "fail" in values
    return count


def human_derived_validity(reviews_a, indexes_a, reviews_b, indexes_b):
    """Exploratory rule-derived comparison; not a Human pairwise judgement."""
    count_a = human_defective_count(reviews_a, indexes_a)
    count_b = human_defective_count(reviews_b, indexes_b)
    if count_a is None or count_b is None:
        result = "UNDETERMINED"
    elif count_a == count_b:
        result = "NO_VALIDITY_DIFFERENCE"
    else:
        result = "A" if count_a < count_b else "B"
    return {"result": result, "defectiveQuestions": {"A": count_a, "B": count_b}}


def h2a_comparison(pair_result, decided_bys, h2a_result):
    """Compare only when every used call was decided at the validity stage."""
    if pair_result["status"] != RESOLVED:
        return {"comparability": "NOT_COMPARABLE_STATUS", "reason": pair_result["status"], "outcome": None}
    if not decided_bys or any(value != "validity" for value in decided_bys):
        return {"comparability": "NOT_COMPARABLE_DECISION_BASIS", "reason": "decidedBy", "outcome": None}
    if h2a_result == "UNDETERMINED":
        return {"comparability": "NOT_COMPARABLE_UNDETERMINED", "reason": "UNDETERMINED", "outcome": None}
    outcome = "AGREE" if pair_result["preference"] == h2a_result else "DISAGREE"
    return {"comparability": "COMPARABLE", "reason": None, "outcome": outcome}


def count_outcomes(outcomes):
    """Descriptive counts only: no rates, kappa or significance."""
    counts = {}
    for outcome in outcomes:
        key = "null" if outcome is None else outcome
        counts[key] = counts.get(key, 0) + 1
    return counts
