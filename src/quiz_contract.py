"""Quiz response parsing and structural rules shared by the Pilot runner and production Quiz.

Both checks are structural only: they never judge whether an answer, explanation or evidence is
semantically correct.
"""

import json


def parse_quiz_output(raw):
    """Return the parsed Quiz object, or raise ValueError when it cannot be read as a Quiz.

    Accepts the Provider's normalized object or its unparsed model text. Field presence and types
    are checked; counts, emptiness and promptVersion values are left to the structural rules.
    """
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except (ValueError, TypeError):
        raise ValueError("Quiz output is not JSON") from None
    if not isinstance(parsed, dict) or not isinstance(parsed.get("questions"), list):
        raise ValueError("Quiz output has no questions list")
    if "promptVersion" in parsed and parsed["promptVersion"] is not None and not isinstance(parsed["promptVersion"], str):
        raise ValueError("Quiz promptVersion must be a string")
    for question in parsed["questions"]:
        if not isinstance(question, dict) or any(not isinstance(question.get(key), str)
                                                for key in ("question", "explanation", "sourceEvidence")):
            raise ValueError("Quiz question fields must be strings")
        if not isinstance(question.get("options"), list) or any(
                not isinstance(option, str) for option in question["options"]):
            raise ValueError("Quiz options must be strings")
        index = question.get("correctOptionIndex")
        if type(index) is not int or not -(2 ** 31) <= index <= 2 ** 31 - 1:
            raise ValueError("Quiz correctOptionIndex must be an integer")
    return parsed


def quiz_questions_valid(questions, option_count):
    """Structural rules for parsed questions: non-empty and distinct questions, exactly
    ``option_count`` non-empty distinct options, a 0-based answer index in range, and non-empty
    explanation and sourceEvidence. Distinctness ignores case and surrounding whitespace."""
    seen_questions = set()
    for question in questions:
        options = question.get("options")
        index = question.get("correctOptionIndex")
        normalized_question = question["question"].strip().lower()
        normalized_options = [option.strip().lower() for option in options]
        if (not normalized_question or normalized_question in seen_questions
                or len(options) != option_count
                or any(not option for option in normalized_options)
                or len(set(normalized_options)) != len(options)
                or not 0 <= index < len(options)
                or not question["explanation"].strip()
                or not question["sourceEvidence"].strip()):
            return False
        seen_questions.add(normalized_question)
    return True
