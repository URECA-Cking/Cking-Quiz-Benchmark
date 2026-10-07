"""Human Grounding review checklist for Grounding prompt versions that require it (Pilot v2).

The five verdicts are entered by a person; code only checks them and derives the overall
contentText approval status from them. Pilot v1 and legacy Grounding rows never need a review
and keep their existing approval rules.
"""

from datetime import datetime

from src.approval_tracking import APPROVED_AT_PATTERN, normalize_approved_by
from src.provider_adapters import GROUNDING_PROMPTS


# Legacy Grounding rows (promptVersion=null) were recorded before Grounding prompt versioning.
LEGACY_GROUNDING_PROMPT_VERSION = "video-grounding-v1"
REVIEW_REQUIRED_PROMPT_VERSIONS = frozenset({"video-grounding-v2"})
# The Grounding version group each Quiz experiment may take its source contentText from.
QUIZ_GROUNDING_VERSIONS = {"pilot-v1": "video-grounding-v1", "pilot-v2": "video-grounding-v2"}


def is_known_grounding_version(prompt_version):
    """A stored or configured Grounding promptVersion: legacy null or a version with a prompt."""
    return prompt_version is None or (isinstance(prompt_version, str) and prompt_version in GROUNDING_PROMPTS)


def grounding_version_matches_quiz_version(quiz_prompt_version, grounding_prompt_version):
    """Whether a Quiz experiment may use a source Grounding of this version (unknown versions never match).

    pilot-v1 takes legacy null and video-grounding-v1 sources; pilot-v2 takes video-grounding-v2 only.
    """
    expected = (QUIZ_GROUNDING_VERSIONS.get(quiz_prompt_version)
                if isinstance(quiz_prompt_version, str) else None)
    return (expected is not None and is_known_grounding_version(grounding_prompt_version)
            and grounding_attempt_version(grounding_prompt_version) == expected)
REVIEW_ITEMS = ("factualAccuracy", "keyInformationCoverage", "factsConsistency",
                "koreanConsistency", "contentTextContractCompliance")
REVIEW_VERDICTS = ("pass", "fail", "uncertain")
REVIEW_KEYS = frozenset(REVIEW_ITEMS + ("reviewNote", "reviewedBy", "reviewedAt"))


def grounding_attempt_version(prompt_version):
    """Version group for Grounding attempt numbering and history.

    Before versioning, attempt numbering ignored promptVersion, so legacy null rows stay in the
    video-grounding-v1 sequence. Stored rows are never rewritten.
    """
    return LEGACY_GROUNDING_PROMPT_VERSION if prompt_version is None else prompt_version


def grounding_condition(row):
    """Grounding experiment condition: attempt numbering, history and Pilot v2 terminal success."""
    return (tuple(row.get(field) for field in ("videoId", "method", "model", "repetition"))
            + (grounding_attempt_version(row.get("promptVersion")),))


def requires_review(row):
    version = row.get("promptVersion")
    return (row.get("benchmarkType") == "video_grounding" and isinstance(version, str)
            and version in REVIEW_REQUIRED_PROMPT_VERSIONS)


def successful_review_candidates(rows, condition):
    """Successful review-required Groundings of one condition.

    Pilot v2 allows one successful candidate per condition (generation is terminal after a
    success), so the human review always concerns the only candidate. More than one is an
    ambiguous state that is never resolved by picking one automatically.
    """
    return [row for row in rows if requires_review(row) and row.get("apiStatus") == "success"
            and grounding_condition(row) == condition]


def require_single_successful_candidate(rows, row):
    """A successful review-required Grounding must be the only success of its condition; else raise.

    Shared by storage integrity, human review and offline aggregation. Legacy/v1 rows are exempt.
    """
    if requires_review(row) and len(successful_review_candidates(rows, grounding_condition(row))) != 1:
        raise ValueError("This Pilot v2 condition has more than one successful Grounding candidate")


def derive_approval_status(review):
    """approved only when every checklist item is pass; any fail or uncertain rejects."""
    return "approved" if all(review[item] == "pass" for item in REVIEW_ITEMS) else "rejected"


def _valid_reviewed_at(value):
    if not isinstance(value, str) or APPROVED_AT_PATTERN.fullmatch(value) is None:
        return False
    try:
        datetime.fromisoformat(value)
    except ValueError:
        return False
    return True


def build_review(checklist, reviewed_by, reviewed_at):
    """Validate the person's checklist input and return the review to store; else raise."""
    allowed = set(REVIEW_ITEMS) | {"reviewNote"}
    if (not isinstance(checklist, dict) or not set(REVIEW_ITEMS) <= set(checklist)
            or set(checklist) - allowed
            or any(type(checklist[item]) is not str or checklist[item] not in REVIEW_VERDICTS
                   for item in REVIEW_ITEMS)
            or checklist.get("reviewNote") is not None and type(checklist["reviewNote"]) is not str):
        raise ValueError("The human Grounding checklist needs exactly the five pass/fail/uncertain items")
    try:
        reviewer = normalize_approved_by(reviewed_by)
    except ValueError:
        raise ValueError("A valid reviewed_by is required for human Grounding review") from None
    review = {item: checklist[item] for item in REVIEW_ITEMS}
    review.update(reviewNote=checklist.get("reviewNote"), reviewedBy=reviewer, reviewedAt=reviewed_at)
    return review


def require_valid_grounding_review(row):
    """Stored review structure and the approval status derived from it must agree; else raise."""
    status = row.get("contentTextApprovalStatus")
    if "groundingReview" not in row:
        # rejected only exists as a review outcome; a review-required row cannot be approved without one.
        if status == "rejected" or (status == "approved" and requires_review(row)):
            raise ValueError("Grounding approval status does not match its human review")
        return
    review = row["groundingReview"]
    try:
        valid = (requires_review(row) and row.get("apiStatus") == "success"
                 and isinstance(review, dict) and set(review) == REVIEW_KEYS
                 and all(type(review[item]) is str and review[item] in REVIEW_VERDICTS
                         for item in REVIEW_ITEMS)
                 and (review["reviewNote"] is None or type(review["reviewNote"]) is str)
                 and normalize_approved_by(review["reviewedBy"]) == review["reviewedBy"]
                 and _valid_reviewed_at(review["reviewedAt"])
                 and status == derive_approval_status(review))
    except ValueError:
        valid = False
    if valid and status == "approved":
        valid = (row.get("approvedBy") == review["reviewedBy"]
                 and row.get("approvedAt") == review["reviewedAt"])
    elif valid:
        valid = "approvedBy" not in row and "approvedAt" not in row
    if not valid:
        raise ValueError("Grounding approval status does not match its human review")
