"""Shared validation of human contentText approval tracking (approvedBy / approvedAt).

The Pilot runner and the offline aggregator use the same rules so a Grounding row is
judged the same way when it is approved, used as a Quiz source or aggregated offline.
"""

import re
import unicodedata
from datetime import datetime


APPROVED_BY_MAX_LENGTH = 100
# Control, line/paragraph separator, format (zero-width, bidi) and lone surrogate
# characters. Categories follow the running Python's Unicode database.
APPROVED_BY_FORBIDDEN_CATEGORIES = frozenset({"Cc", "Zl", "Zp", "Cf", "Cs"})
# Canonical datetime.now(timezone.utc).isoformat() output; checked before fromisoformat()
# so validity does not depend on the Python version's accepted ISO 8601 variants.
APPROVED_AT_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{6})?\+00:00",
                                 re.ASCII)


def valid_utc_timestamp(value):
    """A canonical UTC timestamp string that is also a real calendar date and time."""
    if not isinstance(value, str) or APPROVED_AT_PATTERN.fullmatch(value) is None:
        return False
    try:
        datetime.fromisoformat(value)
    except ValueError:
        return False
    return True


def normalize_approved_by(approved_by):
    """Return the stripped caller-supplied approver or raise ValueError."""
    if (not isinstance(approved_by, str)
            or any(unicodedata.category(char) in APPROVED_BY_FORBIDDEN_CATEGORIES
                   for char in approved_by)):
        raise ValueError("A valid approved_by is required for human approval")
    normalized = approved_by.strip()
    if not normalized or len(normalized) > APPROVED_BY_MAX_LENGTH:
        raise ValueError("A valid approved_by is required for human approval")
    return normalized


def require_valid_approval_tracking(row):
    """Accept legacy rows without tracking or a fully valid tracked approval; else raise."""
    has_by, has_at = "approvedBy" in row, "approvedAt" in row
    if not has_by and not has_at:
        return
    if has_by != has_at or row.get("contentTextApprovalStatus") != "approved":
        raise ValueError("Grounding approval tracking is invalid")
    approved_by, approved_at = row["approvedBy"], row["approvedAt"]
    try:
        valid = (normalize_approved_by(approved_by) == approved_by
                 and isinstance(approved_at, str)
                 and APPROVED_AT_PATTERN.fullmatch(approved_at) is not None)
        if valid:
            datetime.fromisoformat(approved_at)  # Rejects impossible calendar dates and times.
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("Grounding approval tracking is invalid")
