"""Production Quiz generation from a machine-eligible Grounding, and the Judge-ready output gate.

Flow: production Grounding eligibility -> exact contentText -> Gemini Quiz -> output gate -> Judge-ready.
This is separate from the Pilot runner: Pilot Human approval, repetitions and the fixed-content
manifest are never used here, and Pilot results are never written.

Meaning boundaries:
- Grounding machine eligibility is not approval of the video facts (see src/production_grounding.py).
- Passing the output gate (Judge-ready) means the Quiz is structurally usable, its sourceEvidence is
  an exact substring of the contentText and its provenance is intact. It does not mean the answers,
  explanations, Korean or distractors are good; that is the later GPT Pointwise Judge's job.
- Judge-ready is not Creator approval.

Storage (``results/production``), every file written once and never replaced:
- ``sources/<sha256>.json``: evidence snapshot (Grounding row, evaluation, raw metadata and the machine
  validation record actually used), named by the SHA-256 of its bytes.
- ``operations/<operationId>/operation.json``: generation identity, input/config fingerprint, snapshot SHA.
- ``operations/<operationId>/attempts/<n>/started.json``: written before the Provider request.
- ``.../raw.json`` and ``.../output.json``: canonical raw metadata and the model output.
- ``.../result.json``: written last, with the SHA-256 of raw.json and output.json. An attempt without it
  is uncertain (the request may have reached the Provider) and is never replayed automatically.
"""

import hashlib
import json
import os
import re
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.approval_tracking import valid_utc_timestamp
from src.judge_contract import canonical_json, sha256_hex
from src.pilot_runner import PilotRunner
from src.production_grounding import (PRODUCTION_GROUNDING_VALIDATION_VERSION, load_grounding_artifact,
                                      require_production_grounding_eligible)
from src.provider_adapters import FIXTURE_RAW_METADATA, PRODUCTION_QUIZ_PROMPTS, validate_raw_metadata
from src.provider_failure import ProviderFailure
from src.quiz_contract import parse_quiz_output, quiz_questions_valid


SNAPSHOT_FORMAT = "production-grounding-snapshot-v1"
OPERATION_FORMAT = "production-quiz-operation-v1"
STARTED_FORMAT = "production-quiz-attempt-started-v1"
RESULT_FORMAT = "production-quiz-attempt-result-v1"
# Output contract version -> (questions, options per question). 0-based correctOptionIndex.
OUTPUT_CONTRACTS = {"production-quiz-output-v1": (3, 4)}
QUIZ_KEYS = frozenset(("promptVersion", "questions"))
QUESTION_KEYS = frozenset(("question", "options", "correctOptionIndex", "explanation", "sourceEvidence"))
# Settings that change the Quiz request: they are bound to the Provider's effective request settings
# before any call and are part of the input fingerprint. Execution guards (retry, cost, timeout) are not.
GENERATION_SETTING_KEYS = frozenset(("thinking_level", "max_output_tokens"))
IDENTITY_KEYS = frozenset(("videoId", "sourceGroundingRunId", "contentTextSha256", "groundingValidationVersion",
                           "model", "method", "promptVersion", "outputContractVersion", "generationSettings"))
STARTED_KEYS = frozenset(("format", "operationId", "attempt", "inputFingerprint", "sourceSnapshotSha256", "startedAt"))
RESULT_KEYS = frozenset((
    "format", "operationId", "attempt", "inputFingerprint", "sourceSnapshotSha256", "sourceGroundingRunId",
    "contentTextSha256", "model", "promptVersion", "startedAt", "completedAt", "apiStatus", "errorCategory",
    "httpStatus", "providerErrorCode", "retryStopReason", "latencyMs", "inputTokens", "outputTokens",
    "thinkingTokens", "estimatedCostUsd", "pricingReference", "parseStatus", "validatorStatus",
    "evidenceTextContained", "questionCount", "outputGateStatus", "outputGateReasons", "rawSha256", "outputSha256"))
_OPERATION_ID = re.compile(r"[0-9a-f]{32}", re.ASCII)
_ATTEMPT_DIR = re.compile(r"[1-9][0-9]*", re.ASCII)


class ProductionQuizNotJudgeReady(ValueError):
    """The operation has no Quiz that may go to the Judge; ``reasons`` says why."""

    def __init__(self, reasons):
        self.reasons = tuple(reasons)
        super().__init__("Production Quiz is not Judge-ready: " + ", ".join(self.reasons))


def new_operation_id():
    """A new generation operation. A regeneration is a new operation, never a retry of an old one."""
    return uuid.uuid4().hex


def _strict_json(data):
    def reject_constant(_value):
        raise ValueError("Non-finite JSON value")
    return json.loads(data, parse_constant=reject_constant)


def _now():
    return datetime.now(timezone.utc).isoformat()


def validate_production_config(config):
    """The production config contract (configs/production.yaml); raise ValueError otherwise."""
    quiz = config.get("quiz_generation") if isinstance(config, dict) else None
    keys = {"method", "model", "provider", "api_key_environment_variable", "generation_settings",
            "prompt_version", "output_contract_version", "questions_per_quiz", "options_per_question"}
    if (not isinstance(quiz, dict) or set(quiz) != keys
            or config.get("grounding_validation_version") != PRODUCTION_GROUNDING_VALIDATION_VERSION
            or quiz["method"] != "fixed_content_text" or quiz["provider"] != "gemini"
            or any(not isinstance(quiz[key], str) or not quiz[key].strip()
                   for key in ("model", "api_key_environment_variable"))
            or not isinstance(quiz["generation_settings"], dict)
            or set(quiz["generation_settings"]) != GENERATION_SETTING_KEYS
            or not isinstance(quiz["generation_settings"]["thinking_level"], str)
            or not quiz["generation_settings"]["thinking_level"].strip()
            or type(quiz["generation_settings"]["max_output_tokens"]) is not int
            or quiz["generation_settings"]["max_output_tokens"] < 1
            or quiz["prompt_version"] not in PRODUCTION_QUIZ_PROMPTS
            or quiz["output_contract_version"] not in OUTPUT_CONTRACTS
            or (type(quiz["questions_per_quiz"]), type(quiz["options_per_question"])) != (int, int)
            or (quiz["questions_per_quiz"], quiz["options_per_question"]) != OUTPUT_CONTRACTS[quiz["output_contract_version"]]):
        raise ValueError("Invalid production Quiz configuration")
    return config


def router_config(config):
    """The ProviderRouter view of the production config: the Quiz model only, no Grounding or Direct."""
    quiz = validate_production_config(config)["quiz_generation"]
    return {"prompt_version": quiz["prompt_version"], "video_grounding": {"methods": []},
            "quiz_generation": {"models": [{"id": quiz["model"], "provider": quiz["provider"],
                                            "api_key_environment_variable": quiz["api_key_environment_variable"],
                                            "generation_settings": quiz["generation_settings"]}]},
            "end_to_end": {"methods": []}}


def evaluate_quiz_output(output, content_text, quiz_config):
    """Machine output gate for one model output against the exact contentText; structure only.

    Returns parseStatus, validatorStatus, errorCategory, evidenceTextContained, questionCount and the
    gate reasons. sourceEvidence must be an exact substring; it is never trimmed, unquoted or repaired.
    """
    question_count, option_count = OUTPUT_CONTRACTS[quiz_config["output_contract_version"]]
    try:
        parsed = parse_quiz_output(output)
    except (ValueError, TypeError):
        return {"parseStatus": "fail", "validatorStatus": "not_run", "errorCategory": "parse_error",
                "evidenceTextContained": None, "questionCount": None, "outputGateReasons": ["parseFailed"]}
    questions = parsed["questions"]
    reasons = []
    if set(parsed) != QUIZ_KEYS or any(set(question) != QUESTION_KEYS for question in questions):
        reasons.append("unexpectedFields")
    if parsed.get("promptVersion") != quiz_config["prompt_version"]:
        reasons.append("promptVersionMismatch")
    if len(questions) != question_count:
        reasons.append("questionCountMismatch")
    if not quiz_questions_valid(questions, option_count):
        reasons.append("invalidQuestionStructure")
    contained = all(question["sourceEvidence"] in content_text for question in questions)
    category = "quiz_contract_error" if reasons else None
    if not contained:
        reasons.append("evidenceNotInContent")
        category = category or "evidence_not_in_content"
    return {"parseStatus": "pass", "validatorStatus": "fail" if reasons else "pass", "errorCategory": category,
            "evidenceTextContained": contained, "questionCount": len(questions), "outputGateReasons": reasons}


class ProductionQuizRunner:
    RESULTS = Path("results") / "production"

    def __init__(self, repository, results=None, config=None):
        self.repository = Path(repository).resolve()
        root = self.repository / self.RESULTS
        requested = Path(results) if results is not None else root
        self.results = (requested if requested.is_absolute() else self.repository / requested).resolve()
        if self.results != root and root not in self.results.parents:
            raise ValueError("Production results must be inside repository/results/production")
        config_file = Path(config) if config else self.repository / "configs" / "production.yaml"
        self.config = validate_production_config(yaml.safe_load(config_file.read_text(encoding="utf-8"))["production"])
        self.quiz = self.config["quiz_generation"]

    # Storage primitives

    def _path(self, *parts):
        path = self.results.joinpath(*parts)
        if path.resolve() != path:
            raise ValueError("Production result paths must not be links")
        return path

    @staticmethod
    def _create_once(path, data):
        """Atomically create ``path`` with ``data``; False (nothing written) if it already exists."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=".production-", suffix=".tmp",
                                             delete=False) as stream:
                temp_path = Path(stream.name)
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            # Same-directory hard-link creation is atomic and never replaces the first writer.
            try:
                os.link(temp_path, path)
            except FileExistsError:
                return False
            return True
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    def _read_json(self, path):
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path.name)
        return _strict_json(path.read_text(encoding="utf-8"))

    def _attempts(self, operation_id):
        """{attempt: (started record or None, result record or None)}; integrity errors raise ValueError."""
        root = self._path("operations", operation_id, "attempts")
        attempts = {}
        if not root.exists():
            return attempts
        for entry in root.iterdir():
            if entry.name.startswith("."):
                continue
            if not entry.is_dir() or entry.is_symlink() or not _ATTEMPT_DIR.fullmatch(entry.name):
                raise ValueError("Production storage integrity error: unexpected attempt entry")
            started = entry / "started.json"
            result = entry / "result.json"
            if not started.exists() and not result.exists():
                continue  # created before started.json was linked: nothing was sent, so not an attempt
            attempts[int(entry.name)] = (self._read_json(started) if started.exists() else None,
                                         self._read_json(result) if result.exists() else None)
        return attempts

    def _success_output(self, attempt_parts, result):
        """The stored model output of a successful attempt after checking its raw/output hashes; else ValueError."""
        raw_bytes = self._path(*attempt_parts, "raw.json").read_bytes()
        output_bytes = self._path(*attempt_parts, "output.json").read_bytes()
        if (hashlib.sha256(raw_bytes).hexdigest() != result["rawSha256"]
                or hashlib.sha256(output_bytes).hexdigest() != result["outputSha256"]):
            raise ValueError("attempt artifact hash mismatch")
        raw = _strict_json(raw_bytes.decode("utf-8"))
        validate_raw_metadata(raw)
        if raw is None or raw == FIXTURE_RAW_METADATA or raw["provider"] != "gemini":
            raise ValueError("attempt raw metadata is not a completed Gemini call")
        return json.loads(output_bytes.decode("ascii"))["output"]

    def _attempt_problem(self, operation, attempt, started, result):
        """Why one recorded attempt breaks the operation -> started -> artifacts -> result chain, else None.

        Every stored identity field must equal the value re-derived from operation.json (and the
        started record); a success needs hash-matching raw/output files, an error must have none.
        An attempt with a started record and no result is reported as ``uncertain``.
        """
        identity = operation["input"]
        if started is None:
            return "missingStartedRecord"
        expected_started = {"format": STARTED_FORMAT, "operationId": operation["operationId"], "attempt": attempt,
                            "inputFingerprint": operation["inputFingerprint"],
                            "sourceSnapshotSha256": operation["sourceSnapshotSha256"]}
        if (not isinstance(started, dict) or set(started) != STARTED_KEYS
                or any(started[key] != value for key, value in expected_started.items())
                or not valid_utc_timestamp(started["startedAt"])):
            return "startedRecordMismatch"
        if result is None:
            return "uncertain"
        expected_result = dict(expected_started, format=RESULT_FORMAT, startedAt=started["startedAt"],
                               sourceGroundingRunId=identity["sourceGroundingRunId"],
                               contentTextSha256=identity["contentTextSha256"], model=identity["model"],
                               promptVersion=identity["promptVersion"])
        if (not isinstance(result, dict) or set(result) != RESULT_KEYS
                or any(result[key] != value for key, value in expected_result.items())
                or not valid_utc_timestamp(result["completedAt"])):
            return "resultRecordMismatch"
        attempt_parts = ("operations", operation["operationId"], "attempts", str(attempt))
        if result["apiStatus"] == "error":
            if (not isinstance(result["errorCategory"], str) or not result["errorCategory"]
                    or result["rawSha256"] is not None or result["outputSha256"] is not None
                    or any(self._path(*attempt_parts, name).exists() for name in ("raw.json", "output.json"))):
                return "resultRecordMismatch"
            return None
        if result["apiStatus"] != "success":
            return "unknownApiStatus"
        try:
            self._success_output(attempt_parts, result)
        except (OSError, UnicodeError, ValueError, KeyError, TypeError):
            return "attemptArtifactIntegrity"
        return None

    def _verified_attempts(self, operation, attempts):
        """{attempt: problem or None}; attempts must be numbered 1..n."""
        problems = {attempt: self._attempt_problem(operation, attempt, started, result)
                    for attempt, (started, result) in attempts.items()}
        if sorted(attempts) != list(range(1, len(attempts) + 1)):
            problems[0] = "attemptNumbering"
        return problems

    def _provider_settings(self, provider):
        """The generation settings the Provider will actually put in the request; never guessed."""
        resolve = getattr(provider, "request_settings", None)
        if not callable(resolve):
            raise ValueError("The Provider cannot report its request settings, so they cannot be verified")
        try:
            settings = resolve("quiz", self.quiz["model"], self.quiz["prompt_version"])
        except ProviderFailure:
            settings = None
        if settings != self.quiz["generation_settings"]:
            raise ValueError("The Provider's request settings differ from the production config")
        return settings

    # Generation

    def generate(self, operation_id, grounding_results, grounding_run_id, validation_record, video_id,
                 duration_seconds, provider):
        """Run one production Quiz attempt for an operation and return its result record.

        Nothing is called or written unless the Grounding passes production eligibility. The Quiz
        input is exactly the contentText the eligibility gate returned. An operation whose earlier
        attempt reached the Provider (or may have) is not run again; a new Quiz needs a new operation.
        """
        if getattr(provider, "is_actual_api", False) is not True:
            raise ValueError("Production Quiz needs a live Provider; fixture output is never a production result")
        if not isinstance(operation_id, str) or _OPERATION_ID.fullmatch(operation_id) is None:
            raise ValueError("A production operationId is 32 lowercase hex characters")
        # The fingerprint below records the configured settings; refuse before any call or write
        # unless the Provider will send exactly those.
        self._provider_settings(provider)
        artifact = load_grounding_artifact(grounding_results, grounding_run_id)
        content = require_production_grounding_eligible(artifact, validation_record, video_id, duration_seconds)
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        snapshot = {"format": SNAPSHOT_FORMAT, "videoId": video_id, "sourceGroundingRunId": grounding_run_id,
                    "contentTextSha256": content_hash, "durationSeconds": duration_seconds,
                    "validationRecord": validation_record, "grounding": artifact}
        snapshot_bytes = canonical_json(snapshot).encode("utf-8")
        snapshot_hash = hashlib.sha256(snapshot_bytes).hexdigest()
        identity = {"videoId": video_id, "sourceGroundingRunId": grounding_run_id,
                    "contentTextSha256": content_hash,
                    "groundingValidationVersion": validation_record["version"],
                    "model": self.quiz["model"], "method": self.quiz["method"],
                    "promptVersion": self.quiz["prompt_version"],
                    "outputContractVersion": self.quiz["output_contract_version"],
                    "generationSettings": self.quiz["generation_settings"]}
        operation = {"format": OPERATION_FORMAT, "operationId": operation_id,
                     "inputFingerprint": sha256_hex(identity), "input": identity,
                     "sourceSnapshotSha256": snapshot_hash}

        snapshot_path = self._path("sources", snapshot_hash + ".json")
        if not self._create_once(snapshot_path, snapshot_bytes) and snapshot_path.read_bytes() != snapshot_bytes:
            raise ValueError("Production storage integrity error: evidence snapshot does not match its hash")
        operation_path = self._path("operations", operation_id, "operation.json")
        if not self._create_once(operation_path, canonical_json(operation).encode("utf-8")):
            if self._read_json(operation_path) != operation:
                raise ValueError("This operation was created with a different input, config or evidence "
                                 "snapshot; it is never reused. Use a new operation for a new generation")

        attempts = self._attempts(operation_id)
        problems = self._verified_attempts(operation, attempts)
        if "uncertain" in problems.values():
            raise ValueError("An earlier attempt of this operation has no result; it may have reached the "
                             "Provider, so it is not replayed automatically")
        if any(problems.values()):
            raise ValueError("Production storage integrity error: an earlier attempt record is invalid ("
                             + ", ".join(sorted(set(filter(None, problems.values())))) + "); nothing is retried")
        if any(result["apiStatus"] == "success" for _, result in attempts.values()):
            raise ValueError("This operation already has a completed Provider result; "
                             "a new generation needs a new operation")
        # Only verified technical errors (apiStatus=error) remain, so an explicit next attempt is allowed.
        attempt = len(attempts) + 1
        attempt_parts = ("operations", operation_id, "attempts", str(attempt))
        started_at = _now()
        started = {"format": STARTED_FORMAT, "operationId": operation_id, "attempt": attempt,
                   "inputFingerprint": operation["inputFingerprint"], "sourceSnapshotSha256": snapshot_hash,
                   "startedAt": started_at}
        if not self._create_once(self._path(*attempt_parts, "started.json"), canonical_json(started).encode("utf-8")):
            raise ValueError("Another process started this attempt")

        result = {"format": RESULT_FORMAT, "operationId": operation_id, "attempt": attempt,
                  "inputFingerprint": operation["inputFingerprint"], "sourceSnapshotSha256": snapshot_hash,
                  "sourceGroundingRunId": grounding_run_id, "contentTextSha256": content_hash,
                  "model": self.quiz["model"], "promptVersion": self.quiz["prompt_version"],
                  "startedAt": started_at, "completedAt": None, "apiStatus": "error", "errorCategory": None,
                  "httpStatus": None, "providerErrorCode": None, "retryStopReason": None, "latencyMs": None,
                  "inputTokens": None, "outputTokens": None, "thinkingTokens": None, "estimatedCostUsd": None,
                  "pricingReference": None, "parseStatus": "not_run", "validatorStatus": "not_run",
                  "evidenceTextContained": None, "questionCount": None,
                  "outputGateStatus": "fail", "outputGateReasons": ["providerResultNotCompleted"],
                  "rawSha256": None, "outputSha256": None}
        started_clock = time.monotonic()
        response = None
        try:
            response, elapsed = PilotRunner._call(
                provider, "quiz", model=self.quiz["model"], contentText=content,
                promptVersion=self.quiz["prompt_version"], questionCount=self.quiz["questions_per_quiz"],
                optionCount=self.quiz["options_per_question"])
            raw = response.get("responseBody")
            try:
                validate_raw_metadata(raw)
            except ValueError:
                raise ProviderFailure("invalid_provider_response") from None
            if raw is None or raw == FIXTURE_RAW_METADATA or raw["provider"] != self.quiz["provider"]:
                raise ProviderFailure("invalid_provider_response")
            PilotRunner._metrics(result, response, elapsed)
        except ProviderFailure as failure:
            category = failure.category if isinstance(failure.category, str) and failure.category else "provider_error"
            result.update(errorCategory=category, httpStatus=failure.http_status,
                          providerErrorCode=failure.provider_error_code,
                          retryStopReason=failure.retry_stop_reason,
                          latencyMs=round((time.monotonic() - started_clock) * 1000, 3))
            measurements = response if response is not None else failure.measurements
            if measurements is not None:
                try:
                    PilotRunner._metrics(result, measurements, result["latencyMs"])
                except ProviderFailure:
                    pass
            result["completedAt"] = _now()
            self._write_result(attempt_parts, result)
            return result

        output = response.get("normalized")
        result.update(apiStatus="success", **evaluate_quiz_output(output, content, self.quiz))
        result["outputGateStatus"] = "fail" if result["outputGateReasons"] else "pass"
        raw_bytes = canonical_json(raw).encode("utf-8")
        # ASCII escapes keep any model text (even unpaired surrogates) storable; json.loads restores it exactly.
        output_bytes = json.dumps({"output": output}, ensure_ascii=True, sort_keys=True).encode("ascii")
        for name, data in (("raw.json", raw_bytes), ("output.json", output_bytes)):
            if not self._create_once(self._path(*attempt_parts, name), data):
                raise ValueError("Production storage integrity error: attempt artifact already exists")
        result.update(rawSha256=hashlib.sha256(raw_bytes).hexdigest(),
                      outputSha256=hashlib.sha256(output_bytes).hexdigest(), completedAt=_now())
        # The result record is the completion marker, so it is written only after every artifact it names.
        self._write_result(attempt_parts, result)
        return result

    def _write_result(self, attempt_parts, result):
        if not self._create_once(self._path(*attempt_parts, "result.json"), canonical_json(result).encode("utf-8")):
            raise ValueError("Production storage integrity error: attempt result already exists")

    # Reading

    def operation_state(self, operation_id):
        """``not_started``, ``uncertain`` (an attempt without result), ``corrupted`` (a broken record
        chain, e.g. a result without its started record), ``completed`` (a Provider success) or
        ``failed`` (only verified technical errors). Completed says nothing about Judge-readiness."""
        try:
            operation = self._read_json(self._path("operations", operation_id, "operation.json"))
        except FileNotFoundError:
            return "not_started"
        except (UnicodeError, ValueError):
            return "corrupted"
        try:
            attempts = self._attempts(operation_id)
            problems = self._verified_attempts(operation, attempts)
        except (UnicodeError, ValueError, KeyError, TypeError):
            return "corrupted"
        if "uncertain" in problems.values():
            return "uncertain"
        if any(problems.values()):
            return "corrupted"
        if any(result["apiStatus"] == "success" for _, result in attempts.values()):
            return "completed"
        return "failed" if attempts else "not_started"

    def require_judge_ready(self, operation_id):
        """Return the Judge input for this operation's Quiz, or raise ProductionQuizNotJudgeReady.

        Re-derives everything from stored files instead of trusting stored statuses: snapshot and
        fingerprint integrity, the Grounding eligibility of the snapshot, the attempt artifacts'
        hashes and the output gate against the snapshot's exact contentText. The returned dict holds
        only the contentText, the questions and provenance; no generation model identity.
        """
        if not isinstance(operation_id, str) or _OPERATION_ID.fullmatch(operation_id) is None:
            raise ProductionQuizNotJudgeReady(["invalidOperationId"])
        try:
            operation = self._read_json(self._path("operations", operation_id, "operation.json"))
            attempts = self._attempts(operation_id)
        except (FileNotFoundError, UnicodeError, ValueError):
            raise ProductionQuizNotJudgeReady(["missingOrCorruptedOperation"]) from None
        identity = operation.get("input") if isinstance(operation, dict) else None
        if (not isinstance(identity, dict) or set(identity) != IDENTITY_KEYS
                or operation.get("format") != OPERATION_FORMAT
                or operation.get("operationId") != operation_id
                or operation.get("inputFingerprint") != sha256_hex(identity)
                or identity.get("outputContractVersion") not in OUTPUT_CONTRACTS):
            raise ProductionQuizNotJudgeReady(["operationIdentityMismatch"])
        try:
            problems = self._verified_attempts(operation, attempts)
        except (KeyError, TypeError, ValueError):
            raise ProductionQuizNotJudgeReady(["operationIdentityMismatch"]) from None
        if "uncertain" in problems.values():
            raise ProductionQuizNotJudgeReady(["operationUncertain"])
        if any(problems.values()):
            raise ProductionQuizNotJudgeReady(sorted(set(filter(None, problems.values()))))
        completed = [(attempt, result) for attempt, (_, result) in attempts.items()
                     if result["apiStatus"] == "success"]
        if len(completed) != 1:
            raise ProductionQuizNotJudgeReady(["noCompletedGeneration" if not completed else "ambiguousGeneration"])
        attempt, result = completed[0]

        reasons = []
        snapshot_hash = operation.get("sourceSnapshotSha256")
        snapshot = None
        try:
            snapshot_bytes = self._path("sources", str(snapshot_hash) + ".json").read_bytes()
            if hashlib.sha256(snapshot_bytes).hexdigest() != snapshot_hash:
                raise ValueError()
            snapshot = _strict_json(snapshot_bytes.decode("utf-8"))
        except (OSError, UnicodeError, ValueError):
            reasons.append("sourceSnapshotIntegrity")
        content = None
        if snapshot is not None:
            try:
                if (snapshot.get("format") != SNAPSHOT_FORMAT
                        or any(snapshot.get(key) != identity.get(key)
                               for key in ("videoId", "sourceGroundingRunId", "contentTextSha256"))
                        or snapshot["validationRecord"]["version"] != identity.get("groundingValidationVersion")):
                    raise ValueError()
                content = require_production_grounding_eligible(snapshot["grounding"], snapshot["validationRecord"],
                                                                snapshot["videoId"], snapshot["durationSeconds"])
                if hashlib.sha256(content.encode("utf-8")).hexdigest() != identity["contentTextSha256"]:
                    raise ValueError()
            except (KeyError, TypeError, AttributeError, ValueError):  # includes ProductionGroundingNotEligible
                content = None
                reasons.append("sourceProvenanceMismatch")

        attempt_parts = ("operations", operation_id, "attempts", str(attempt))
        output = None
        try:
            # The started -> artifacts -> result chain was verified above; this re-reads the checked output.
            output = self._success_output(attempt_parts, result)
        except (OSError, UnicodeError, ValueError, KeyError, TypeError):
            reasons.append("attemptArtifactIntegrity")
        if content is not None and output is not None:
            quiz_config = {"output_contract_version": identity["outputContractVersion"],
                           "prompt_version": identity["promptVersion"]}
            reasons += evaluate_quiz_output(output, content, quiz_config)["outputGateReasons"]
        if reasons:
            raise ProductionQuizNotJudgeReady(reasons)
        return {
            "operationId": operation_id, "attempt": attempt, "inputFingerprint": operation["inputFingerprint"],
            "sourceGroundingRunId": identity["sourceGroundingRunId"],
            "contentTextSha256": identity["contentTextSha256"],
            "groundingValidationVersion": identity["groundingValidationVersion"],
            "sourceSnapshotSha256": snapshot_hash, "quizOutputSha256": result["outputSha256"],
            "contentText": content,
            "questions": [{"questionIndex": index, **{key: question[key] for key in (
                "question", "options", "correctOptionIndex", "explanation", "sourceEvidence")}}
                for index, question in enumerate(output["questions"])],
        }
