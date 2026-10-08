"""Human-readable view of a production Quiz for Creator review: options and answer numbered from 1.

Stored data keeps the 0-based ``correctOptionIndex`` and ``questionIndex``; only the text shown to a
person is numbered from 1. Nothing here writes a file, calls a Provider, records a decision or publishes.

``format_review_questions`` accepts ``ProductionCreatorReview.review_packet`` output or the read-only Quiz
view used by the command below. The command reads the Quiz with ``require_judge_ready`` only, because
``review_packet`` may create the Judge aggregation artifact.

    python -m src.production_quiz_display --operation-id <quizOperationId>
"""

import argparse
import sys
from pathlib import Path

import yaml

from src.production_quiz import OUTPUT_CONTRACTS, ProductionQuizNotJudgeReady, ProductionQuizRunner
from src.quiz_contract import parse_quiz_output, quiz_questions_valid


DISPLAY_KEYS = ("question", "options", "correctOptionIndex", "explanation", "sourceEvidence")


class QuizDisplayError(ValueError):
    """The questions do not meet the production Quiz output contract, so no answer is shown."""


def _validated(packet, option_count):
    """The packet's questions after the existing output-contract checks; QuizDisplayError otherwise.

    An answer index that is a bool, a float, negative or outside the options is never shown as an answer.
    """
    questions = packet.get("questions") if isinstance(packet, dict) else None
    if not isinstance(questions, list) or not questions:
        raise QuizDisplayError("The packet has no questions")
    if any(not isinstance(question, dict) or set(question) != {"questionIndex", *DISPLAY_KEYS}
           or type(question["questionIndex"]) is not int or question["questionIndex"] != position
           for position, question in enumerate(questions)):
        raise QuizDisplayError("Questions must be numbered 0..n-1 in order with exactly the Quiz fields")
    output = [{key: question[key] for key in DISPLAY_KEYS} for question in questions]
    try:
        parse_quiz_output({"questions": output})
    except ValueError:
        raise QuizDisplayError("A question field has the wrong type") from None
    if not quiz_questions_valid(output, option_count):
        raise QuizDisplayError("A question breaks the output contract (options, answer index or empty text)")
    return questions


def format_review_questions(packet, option_count=OUTPUT_CONTRACTS["production-quiz-output-v1"][1]):
    """The questions as review text: options numbered 1..n and the answer as correctOptionIndex + 1.

    Question, options, explanation and evidence are shown exactly as stored; the packet is not changed.
    """
    lines = []
    for question in _validated(packet, option_count):
        answer = question["correctOptionIndex"]
        lines.append("문제 %d. %s" % (question["questionIndex"] + 1, question["question"]))
        lines.append("")
        lines.extend("%d. %s" % (number, option) for number, option in enumerate(question["options"], start=1))
        lines.append("")
        lines.append("정답: %d번 — %s" % (answer + 1, question["options"][answer]))
        lines.append("")
        lines.append("해설: %s" % question["explanation"])
        lines.append("근거: %s" % question["sourceEvidence"])
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def quiz_review_text(repository, operation_id, results=None):
    """Review text for one Judge-ready production Quiz, read without creating or changing any file."""
    return _review_text(ProductionQuizRunner(repository, results), operation_id)


def _review_text(runner, operation_id):
    quiz = runner.require_judge_ready(operation_id)
    header = "Quiz operation %s (정답 번호는 1부터, 저장된 correctOptionIndex는 0부터)\n\n" % operation_id
    return header + format_review_questions({"questions": quiz["questions"]})


def main(argv=None):
    parser = argparse.ArgumentParser(description="Show a production Quiz with options and answer numbered from 1 (read-only)")
    parser.add_argument("--operation-id", required=True, help="Quiz operation ID (32 hex characters)")
    parser.add_argument("--repository", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--results-dir", default=None, help="Production results directory (default results/production)")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    # Setup errors are reported apart from review errors so no other ValueError is hidden.
    try:
        runner = ProductionQuizRunner(args.repository, args.results_dir)
    except OSError as exc:  # missing or unreadable configs/production.yaml
        sys.stderr.write("Production config cannot be read: %s\n" % (exc.filename or exc))
        return 1
    except (KeyError, ValueError, yaml.YAMLError) as exc:  # results outside results/production, invalid config
        sys.stderr.write("Invalid --repository or --results-dir: %s\n" % exc)
        return 1
    try:
        text = _review_text(runner, args.operation_id)
    except ProductionQuizNotJudgeReady as exc:
        sys.stderr.write("Quiz is not reviewable: %s\n" % ", ".join(exc.reasons))
        return 1
    except QuizDisplayError as exc:
        sys.stderr.write("Quiz cannot be shown: %s\n" % exc)
        return 1
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
