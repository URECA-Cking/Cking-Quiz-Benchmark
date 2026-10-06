"""Post-hoc LLM-as-a-Judge runs over stored A/B Quiz results (Issue #19).

Storage: ``results/judge/<judgeRunId>/``. Raw records (run.json, eligibility, measurements,
attempts, outputs) are written once or appended; ``derived/`` is recomputed from them. Nothing is
written to the Pilot ``results/raw`` or ``results/evaluation`` directories, and Pilot results,
Human Evaluation and ground truth are only read.
"""

import argparse
import hashlib
import json
import os
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src import judge_aggregation as agg
from src.judge_client import (WAIT_CATEGORIES, JudgeCallFailure, JudgeFixtureClient, JudgeHttpClient,
                              JudgePolicy, build_request)
from src.judge_contract import (JUDGE_PROMPT_VERSION, POINTWISE_ITEMS, RUBRIC_VERSION, SCHEMA_VERSION,
                                contract_fingerprint, pairwise_prompt, pointwise_prompt, schema_for,
                                sha256_hex)
from src.judge_eligibility import QUIZ_FILE, build_plan, eligibility_records
from src.judge_validation import JudgeOutputError, validate_pairwise, validate_pointwise


MAX_ATTEMPTS = 3
KINDS = ("pointwise", "pairwise")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _atomic_write(path, text):
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent,
                                         prefix=".judge-", suffix=".tmp", delete=False) as stream:
            temp_path = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path, rows):
    _atomic_write(path, "".join(_dumps(row) + "\n" for row in rows))


def _append_jsonl(path, row):
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    _atomic_write(path, existing + _dumps(row) + "\n")


def load_config(repository, config_path=None, max_output_tokens=None):
    path = Path(config_path) if config_path else Path(repository) / "configs" / "judge.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))["judge"]
    if max_output_tokens:
        config["max_output_tokens"] = dict(config["max_output_tokens"], **max_output_tokens)
    judges = config["judges"]
    if (len(judges) != 2 or {judge["provider"] for judge in judges} != {"openai", "gemini"}
            or len({judge["id"] for judge in judges}) != 2
            or any(judge.get("reasoning") != "medium" for judge in judges)
            or config.get("service_tier") != "standard"):
        raise ValueError("Judge config must define one OpenAI and one Gemini Judge with medium reasoning")
    return config


def identity_config(config):
    """Everything that changes a measurement's meaning; operational guards and prices are excluded."""
    return {"conditions": config["conditions"],
            "judges": [{key: judge[key] for key in ("id", "provider", "model", "reasoning")}
                       for judge in config["judges"]],
            "serviceTier": config["service_tier"], "maxOutputTokens": config["max_output_tokens"],
            "promptVersion": JUDGE_PROMPT_VERSION, "rubricVersion": RUBRIC_VERSION,
            "schemaVersion": SCHEMA_VERSION, "contractFingerprint": contract_fingerprint()}


def plan_measurements(run_id, plan, config, with_bodies=True):
    """Logical measurements in a fixed order: per Judge, Pointwise sets then Pairwise AB/BA."""
    units = {unit["quizRunId"]: unit for unit in plan["pointwise"]}
    measurements = []
    for judge in config["judges"]:
        jobs = []
        for unit in plan["pointwise"]:
            if unit["status"] == "eligible":
                jobs.append(("pointwise", unit["videoId"], unit["repetition"], None, unit["condition"],
                             {unit["condition"]: unit["quizRunId"]},
                             pointwise_prompt(unit["contentText"], unit["questions"]),
                             {"expectedQuestionIndexes": list(unit["eligibleQuestionIndexes"])}))
        for pair in plan["pairwise"]:
            if pair["status"] != "eligible":
                continue
            members = {condition: units[run_id_] for condition, run_id_ in pair["quizRunIds"].items()}
            for order, slots in agg.ORDERS.items():
                set_1, set_2 = members[slots["SET_1"]], members[slots["SET_2"]]
                jobs.append(("pairwise", pair["videoId"], pair["repetition"], order, None, dict(pair["quizRunIds"]),
                             pairwise_prompt(members["A"]["contentText"], set_1["questions"], set_2["questions"]),
                             {"pairId": pair["pairId"],
                              "setQuestionIndexes": {"SET_1": list(set_1["eligibleQuestionIndexes"]),
                                                     "SET_2": list(set_2["eligibleQuestionIndexes"])}}))
        for kind, video_id, repetition, order, condition, quiz_run_ids, prompt, extra in jobs:
            measurement = {"kind": kind, "judgeId": judge["id"], "provider": judge["provider"],
                           "model": judge["model"], "reasoning": judge["reasoning"], "videoId": video_id,
                           "repetition": repetition, "condition": condition, "order": order,
                           "quizRunIds": quiz_run_ids, "promptCharacters": len(prompt),
                           "fixtureKey": ":".join(str(part) for part in (
                               kind, judge["id"], video_id, repetition, condition or order))}
            measurement.update(extra)
            if with_bodies:
                body = build_request(judge["provider"], judge["model"], kind, prompt, schema_for(kind),
                                     judge["reasoning"], config["max_output_tokens"][kind])
                measurement["inputHash"] = sha256_hex(body)
                measurement["logicalMeasurementId"] = sha256_hex({
                    "judgeRunId": run_id, "kind": kind, "judgeId": judge["id"], "quizRunIds": quiz_run_ids,
                    "order": order, "inputHash": measurement["inputHash"]})[:32]
                measurement["_body"] = body
            measurements.append(measurement)
    return measurements


def execution_mode(client):
    return "live" if getattr(client, "is_actual_api", False) is True else "fixture"


def preflight_live_policy(policy, judges):
    """Reject incomplete or invalid live settings before a run directory or attempt is created."""
    for kind in KINDS:
        for judge in judges:
            try:
                policy.authorize(kind, judge["provider"], 0, 0.0)
            except JudgeCallFailure as failure:
                raise ValueError("Invalid live Judge settings for %s/%s" % (kind, judge["provider"])) from failure


def cost_summary(attempts, selected_attempt_ids):
    """Unknown costs are never counted as zero: a total is null when any part is unknown."""
    def summarize(costs):
        known = [value for value in costs if value is not None]
        unknown = len(costs) - len(known)
        return {"usd": None if unknown else sum(known), "knownUsd": sum(known), "unknownCostCount": unknown}
    return {"selectedMeasurements": summarize([item["estimatedCostUsd"] for item in attempts
                                               if item["attemptId"] in selected_attempt_ids]),
            "allAttempts": summarize([item["estimatedCostUsd"] for item in attempts])}


def _persistable(measurement):
    return {key: value for key, value in measurement.items() if not key.startswith("_")}


def measurement_states(measurements, attempts):
    """SELECTED (first success), INCOMPLETE (attempts exhausted) or NOT_RUN; duplicates flagged."""
    states = {}
    for measurement in measurements:
        history = sorted((item for item in attempts
                          if item["logicalMeasurementId"] == measurement["logicalMeasurementId"]),
                         key=lambda item: item["attempt"])
        successes = [item for item in history if item["outcome"] == "success"]
        if successes:
            state = {"status": "SELECTED", "selectedAttemptId": successes[0]["attemptId"],
                     "duplicateSuccessAttemptIds": [item["attemptId"] for item in successes[1:]]}
        elif len(history) >= MAX_ATTEMPTS:
            state = {"status": "INCOMPLETE", "selectedAttemptId": None, "duplicateSuccessAttemptIds": []}
        else:
            # UNFINISHED: a stopped run left this measurement with attempts remaining.
            state = {"status": "NOT_RUN" if not history else "UNFINISHED",
                     "selectedAttemptId": None, "duplicateSuccessAttemptIds": []}
        state["attempts"] = len(history)
        states[measurement["logicalMeasurementId"]] = state
    return states


class JudgeRunner:
    def __init__(self, repository, source_results="results", config_path=None, max_output_tokens=None,
                 sleep=None):
        self.repository = Path(repository).resolve()
        self.results_root = self.repository / "results"
        self.source = self._inside_results(source_results)
        self.judge_root = self.results_root / "judge"
        if self.source == self.judge_root or self.judge_root in self.source.parents:
            raise ValueError("Pilot source results must not be inside results/judge")
        self.config = load_config(self.repository, config_path, max_output_tokens)
        self.config_hash = sha256_hex(identity_config(self.config))
        self.sleep = sleep or time.sleep

    def _inside_results(self, value):
        if self.results_root.exists() and self.results_root.resolve() != self.results_root:
            raise ValueError("results must not be a link")
        path = Path(value)
        resolved = (path if path.is_absolute() else self.repository / path).resolve()
        if resolved != self.results_root and self.results_root not in resolved.parents:
            raise ValueError("Judge source results must be inside repository/results")
        return resolved

    def plan(self, run_id=None, with_bodies=True):
        plan = build_plan(self.source, self.config["conditions"])
        return plan, plan_measurements(run_id, plan, self.config, with_bodies)

    def run_dir(self, run_id):
        if not isinstance(run_id, str) or len(run_id) != 32 or any(c not in "0123456789abcdef" for c in run_id):
            raise ValueError("Invalid judgeRunId")
        path = self.judge_root / run_id
        if self.judge_root.exists() and self.judge_root.resolve() != self.judge_root:
            raise ValueError("results/judge must not be a link")
        return path

    def start(self, client, operational=None):
        run_id = uuid.uuid4().hex
        plan, measurements = self.plan(run_id)
        run_dir = self.run_dir(run_id)
        (run_dir / "outputs").mkdir(parents=True)
        _atomic_write(run_dir / "run.json", _dumps({
            "judgeRunId": run_id, "createdAt": _now(), "status": "running", "stopReason": None,
            "configHash": self.config_hash, "config": identity_config(self.config),
            "modelDocsCheckedAt": {judge["id"]: judge.get("model_docs_checked_at")
                                   for judge in self.config["judges"]},
            "operational": operational or {}, "executionMode": execution_mode(client),
            "sourceResults": self.source.relative_to(self.repository).as_posix(),
            "sourceSnapshot": plan["sourceSnapshot"]}))
        _write_jsonl(run_dir / "eligibility.jsonl", eligibility_records(plan))
        _write_jsonl(run_dir / "measurements.jsonl", [_persistable(item) for item in measurements])
        return self._execute(run_dir, measurements, client)

    def resume(self, run_id, client):
        run_dir = self.run_dir(run_id)
        manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        # Checked before planning or any state change: one run never mixes fixture and live results.
        if manifest.get("executionMode") != execution_mode(client):
            raise ValueError("Execution mode differs from the stored run: fixture and live runs cannot resume each other")
        plan, measurements = self.plan(run_id)
        if manifest["configHash"] != self.config_hash or manifest["sourceSnapshot"] != plan["sourceSnapshot"]:
            raise ValueError("Configuration or source changed: start a new Judge run instead of resuming")
        stored = [item["logicalMeasurementId"] for item in _read_jsonl(run_dir / "measurements.jsonl")]
        if stored != [item["logicalMeasurementId"] for item in measurements]:
            raise ValueError("Planned measurements differ from the stored run: start a new Judge run")
        self._set_status(run_dir, "running", None)
        return self._execute(run_dir, measurements, client)

    def _set_status(self, run_dir, status, reason):
        manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        manifest.update(status=status, stopReason=reason, updatedAt=_now())
        _atomic_write(run_dir / "run.json", _dumps(manifest))
        return manifest

    def _execute(self, run_dir, measurements, client):
        attempts = _read_jsonl(run_dir / "attempts.jsonl")
        for measurement in measurements:
            history = [item for item in attempts if item["logicalMeasurementId"] == measurement["logicalMeasurementId"]]
            if any(item["outcome"] == "success" for item in history):
                continue  # A successful measurement is never run again.
            while len(history) < MAX_ATTEMPTS:
                if history and history[-1]["errorCategory"] in WAIT_CATEGORIES:
                    # Never retry a 429/5xx immediately; wait for the official hint recorded on the attempt.
                    if history[-1]["retryAfterSeconds"] is None:
                        return self._set_status(run_dir, "stopped", "retryWaitUnresolved")
                    self.sleep(history[-1]["retryAfterSeconds"])
                record = self._attempt(run_dir, measurement, len(history) + 1, client)
                history.append(record)
                attempts.append(record)
                if record["outcome"] == "success":
                    break
                if record["runStopReason"]:
                    return self._set_status(run_dir, "stopped", record["runStopReason"])
            if (len(history) == MAX_ATTEMPTS and all(item["outcome"] == "failure" for item in history)
                    and all(item["errorCategory"] == "incomplete" for item in history)):
                return self._set_status(run_dir, "stopped", "repeatedIncomplete")
        return self._set_status(run_dir, "completed", None)

    def _attempt(self, run_dir, measurement, number, client):
        body = measurement["_body"]
        if sha256_hex(body) != measurement["inputHash"]:
            raise ValueError("Retry input differs from the logical measurement")
        attempt_id = "%s-%d" % (measurement["logicalMeasurementId"], number)
        price = (getattr(getattr(client, "policy", None), "prices", {}) or {}).get(measurement["provider"]) or {}
        record = {"attemptId": attempt_id, "logicalMeasurementId": measurement["logicalMeasurementId"],
                  "attempt": number, "startedAt": _now(), "finishedAt": None, "outcome": "failure",
                  "errorCategory": None, "httpStatus": None, "providerErrorCode": None, "httpRequests": [],
                  "retryAfterSeconds": None, "runStopReason": None, "requestedModel": measurement["model"],
                  "reportedModel": None, "reasoning": measurement["reasoning"], "inputHash": measurement["inputHash"],
                  "usage": None, "estimatedCostUsd": None, "pricingReference": price.get("reference"),
                  "pricingCheckedAt": price.get("checkedAt"), "outputRef": None}
        output = {"rawText": None, "parsed": None, "validationStage": None}
        try:
            result = client.call_measurement(measurement, body)
            record.update(reportedModel=result.get("reportedModel"), usage=result.get("usage"),
                          estimatedCostUsd=result.get("estimatedCostUsd"),
                          httpRequests=result.get("httpRequests", []))
            output["rawText"] = result["text"]
            try:
                if measurement["kind"] == "pointwise":
                    parsed = validate_pointwise(result["text"], measurement["expectedQuestionIndexes"])
                    output["parsed"] = {str(index): items for index, items in parsed.items()}
                else:
                    output["parsed"] = validate_pairwise(result["text"], measurement["setQuestionIndexes"])
                output["validationStage"] = "passed"
                record["outcome"] = "success"
            except JudgeOutputError as error:
                output["validationStage"] = record["errorCategory"] = error.category
        except JudgeCallFailure as failure:
            record.update(errorCategory=failure.category, httpStatus=failure.http_status,
                          providerErrorCode=failure.provider_error_code, httpRequests=failure.http_requests,
                          retryAfterSeconds=failure.retry_after, reportedModel=failure.reported_model,
                          usage=failure.usage, estimatedCostUsd=failure.estimated_cost_usd)
            output["rawText"] = failure.raw_text
            if failure.stop_run:
                record["runStopReason"] = failure.category
            elif failure.retry_wait_unresolved:
                # No official retry hint and no decided fallback wait: stop instead of retrying blindly.
                record["runStopReason"] = "retryWaitUnresolved"
        record["finishedAt"] = _now()
        if output["rawText"] is not None or output["parsed"] is not None:
            record["outputRef"] = "outputs/%s.json" % attempt_id
            _atomic_write(run_dir / record["outputRef"], _dumps(output))
        _append_jsonl(run_dir / "attempts.jsonl", record)
        return record

    def derive(self, run_id):
        """Recompute every derived file from the run's raw records; Pilot files are only read."""
        run_dir = self.run_dir(run_id)
        manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        source = self._inside_results(manifest["sourceResults"])
        quiz_path = source / QUIZ_FILE
        if hashlib.sha256(quiz_path.read_bytes()).hexdigest() != manifest["sourceSnapshot"][QUIZ_FILE]:
            raise ValueError("Pilot Quiz results changed after the Judge run")
        reviews = {row.get("runId"): row.get("questionReviews", [])
                   for row in _read_jsonl(quiz_path) if isinstance(row, dict)}
        eligibility = _read_jsonl(run_dir / "eligibility.jsonl")
        measurements = _read_jsonl(run_dir / "measurements.jsonl")
        attempts = _read_jsonl(run_dir / "attempts.jsonl")
        states = measurement_states(measurements, attempts)
        attempts_by_id = {item["attemptId"]: item for item in attempts}

        def parsed(measurement):
            state = states[measurement["logicalMeasurementId"]]
            if state["status"] != "SELECTED":
                return None
            ref = attempts_by_id[state["selectedAttemptId"]]["outputRef"]
            return json.loads((run_dir / ref).read_text(encoding="utf-8"))["parsed"]

        status_rows = [dict(states[item["logicalMeasurementId"]], logicalMeasurementId=item["logicalMeasurementId"],
                            kind=item["kind"], judgeId=item["judgeId"]) for item in measurements]
        verdicts, human_rows = [], []
        for measurement in (item for item in measurements if item["kind"] == "pointwise"):
            result = parsed(measurement)
            if result is None:
                continue
            quiz_run_id = next(iter(measurement["quizRunIds"].values()))
            by_index = {review.get("questionIndex"): review for review in reviews.get(quiz_run_id, [])
                        if isinstance(review, dict)}
            for index, items in sorted(result.items(), key=lambda item: int(item[0])):
                for item in POINTWISE_ITEMS:
                    base = {"judgeId": measurement["judgeId"], "quizRunId": quiz_run_id,
                            "videoId": measurement["videoId"], "condition": measurement["condition"],
                            "questionIndex": int(index), "item": item}
                    verdicts.append(dict(base, verdict=items[item]["verdict"], reason=items[item]["reason"]))
                    comparison = agg.compare_pointwise_human(item, items[item]["verdict"], by_index.get(int(index)))
                    if comparison["humanField"] is not None:
                        human_rows.append(dict(base, **comparison))

        calls = []
        for measurement in (item for item in measurements if item["kind"] == "pairwise"):
            result = parsed(measurement)
            if result is None:
                continue
            canonical = agg.canonicalize_pairwise(measurement["order"], result)
            preference, decided_by = agg.pairwise_call_final(canonical)
            calls.append({"judgeId": measurement["judgeId"], "pairId": measurement["pairId"],
                          "order": measurement["order"], "logicalMeasurementId": measurement["logicalMeasurementId"],
                          "defectiveQuestions": canonical["defectiveQuestions"], "quality": canonical["quality"],
                          "preference": preference, "decidedBy": decided_by})

        judge_ids = [judge["id"] for judge in manifest["config"]["judges"]]
        pairs = [row for row in eligibility if row["unitType"] == "pairwise_pair" and row["status"] == "eligible"]
        units = {row["quizRunId"]: row for row in eligibility if row["unitType"] == "pointwise_set"}
        judge_rows, pair_rows = [], []
        for pair in pairs:
            per_judge = {}
            for judge_id in judge_ids:
                by_order = {call["order"]: call for call in calls
                            if call["pairId"] == pair["pairId"] and call["judgeId"] == judge_id}
                merged = agg.merge_orders(*(by_order[order]["preference"] if order in by_order else None
                                            for order in ("AB", "BA")))
                merged.update(judgeId=judge_id, pairId=pair["pairId"],
                              decidedBy={order: call["decidedBy"] for order, call in by_order.items()})
                per_judge[judge_id] = merged
                judge_rows.append(merged)
            aggregate = agg.merge_judges(per_judge[judge_ids[0]], per_judge[judge_ids[1]])
            decided_bys = [value for row in per_judge.values() for value in row["decidedBy"].values()]
            a_id, b_id = pair["quizRunIds"]["A"], pair["quizRunIds"]["B"]
            h2a = agg.human_derived_validity(reviews.get(a_id, []), units[a_id]["eligibleQuestionIndexes"],
                                             reviews.get(b_id, []), units[b_id]["eligibleQuestionIndexes"])
            pair_rows.append(dict(aggregate, pairId=pair["pairId"], videoId=pair["videoId"],
                                  judgeResults={judge_id: {"status": row["status"], "preference": row["preference"],
                                                           "orderAgreement": row["orderAgreement"]}
                                                for judge_id, row in per_judge.items()},
                                  humanDerivedValidity=h2a,
                                  h2aComparison=agg.h2a_comparison(aggregate, decided_bys, h2a["result"])))

        report = self._report(manifest, eligibility, status_rows, verdicts, human_rows, judge_rows,
                              pair_rows, attempts, states, judge_ids)
        derived = run_dir / "derived"
        derived.mkdir(exist_ok=True)
        for name, rows in (("measurement-status.jsonl", status_rows), ("pointwise-verdicts.jsonl", verdicts),
                           ("pairwise-calls.jsonl", calls), ("pairwise-judge.jsonl", judge_rows),
                           ("pairwise-pairs.jsonl", pair_rows), ("human-pointwise.jsonl", human_rows)):
            _write_jsonl(derived / name, rows)
        _atomic_write(derived / "report-counts.json", json.dumps(report, ensure_ascii=False, indent=2,
                                                                 allow_nan=False))
        return report

    @staticmethod
    def _report(manifest, eligibility, status_rows, verdicts, human_rows, judge_rows, pair_rows,
                attempts, states, judge_ids):
        selected = {state["selectedAttemptId"] for state in states.values() if state["selectedAttemptId"]}
        pointwise_units = [row for row in eligibility if row["unitType"] == "pointwise_set"]
        inter_judge = {}
        keyed = {}
        for row in verdicts:
            keyed.setdefault((row["quizRunId"], row["questionIndex"], row["item"]), {})[row["judgeId"]] = row["verdict"]
        for (quiz, index, item), by_judge in keyed.items():
            inter_judge.setdefault(item, []).append(
                agg.compare_verdicts(by_judge.get(judge_ids[0]), by_judge.get(judge_ids[1])))
        return {
            "note": "Descriptive counts only. Not accuracy, kappa or significance; samples are not independent.",
            "judgeRunId": manifest["judgeRunId"], "runStatus": manifest["status"],
            "stopReason": manifest["stopReason"],
            "eligibility": {
                "pointwiseEligibleSets": sum(row["status"] == "eligible" for row in pointwise_units),
                "pointwiseEligibleQuestions": sum(len(row["eligibleQuestionIndexes"]) for row in pointwise_units),
                "pairwiseEligiblePairs": sum(row["status"] == "eligible" for row in eligibility
                                             if row["unitType"] == "pairwise_pair"),
                "excluded": [row for row in eligibility if row["status"] != "eligible"],
                "excludedQuestions": [{"quizRunId": row["quizRunId"], "excludedQuestions": row["excludedQuestions"]}
                                      for row in pointwise_units if row["excludedQuestions"]]},
            "measurementStatus": agg.count_outcomes(row["status"] for row in status_rows),
            "pointwiseVerdicts": {judge: {item: agg.count_outcomes(
                row["verdict"] for row in verdicts if row["judgeId"] == judge and row["item"] == item)
                for item in POINTWISE_ITEMS} for judge in judge_ids},
            "pointwiseHumanComparison": {judge: {item: agg.count_outcomes(
                row["outcome"] for row in human_rows if row["judgeId"] == judge and row["item"] == item)
                for item in agg.POINTWISE_HUMAN_MAPPING} for judge in judge_ids},
            "pointwiseInterJudge": {item: agg.count_outcomes(values) for item, values in inter_judge.items()},
            "pairwiseJudge": {judge: agg.count_outcomes(
                row["status"] + (":" + row["preference"] if row["preference"] else "")
                for row in judge_rows if row["judgeId"] == judge) for judge in judge_ids},
            "pairwisePairs": agg.count_outcomes(
                row["status"] + (":" + row["preference"] if row["preference"] else "") for row in pair_rows),
            "h2aComparison": agg.count_outcomes(
                row["h2aComparison"]["comparability"] + (":" + row["h2aComparison"]["outcome"]
                                                         if row["h2aComparison"]["outcome"] else "")
                for row in pair_rows),
            "cost": cost_summary(attempts, selected),
        }


def _price_args(parser):
    for provider in ("openai", "gemini"):
        parser.add_argument("--%s-input-price-per-million" % provider, type=float)
        parser.add_argument("--%s-output-price-per-million" % provider, type=float)
        parser.add_argument("--%s-cached-input-price-per-million" % provider, type=float)
        parser.add_argument("--%s-pricing-reference" % provider)
        parser.add_argument("--%s-pricing-checked-at" % provider)


def main():
    parser = argparse.ArgumentParser(description="Post-hoc LLM-as-a-Judge over stored A/B Quiz results")
    parser.add_argument("command", choices=("plan", "run", "derive"))
    parser.add_argument("--results-dir", default="results", help="Pilot results to judge (read-only)")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--max-output-tokens-pointwise", type=int)
    parser.add_argument("--max-output-tokens-pairwise", type=int)
    parser.add_argument("--run-id", help="derive, or resume an interrupted run with an unchanged configuration")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--fixture", type=Path)
    source.add_argument("--live", action="store_true")
    parser.add_argument("--http-request-limit", type=int)
    parser.add_argument("--per-call-cost-limit", type=float)
    parser.add_argument("--total-cost-limit", type=float)
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--estimated-input-tokens-pointwise", type=int)
    parser.add_argument("--estimated-input-tokens-pairwise", type=int)
    _price_args(parser)
    args = parser.parse_args()
    overrides = {kind: value for kind, value in (("pointwise", args.max_output_tokens_pointwise),
                                                 ("pairwise", args.max_output_tokens_pairwise)) if value}
    runner = JudgeRunner(Path(__file__).resolve().parents[1], args.results_dir, args.config, overrides)
    if args.command == "plan":
        # Read-only dry run: no Provider call and no file written.
        plan, measurements = runner.plan(with_bodies=False)
        sizes = {kind: [item["promptCharacters"] for item in measurements if item["kind"] == kind] for kind in KINDS}
        print(json.dumps({
            "pointwiseEligibleSets": sum(unit["status"] == "eligible" for unit in plan["pointwise"]),
            "pointwiseEligibleQuestions": sum(len(unit["eligibleQuestionIndexes"]) for unit in plan["pointwise"]),
            "pairwiseEligiblePairs": sum(pair["status"] == "eligible" for pair in plan["pairwise"]),
            "excluded": [record for record in eligibility_records(plan) if record["status"] != "eligible"],
            "plannedMeasurements": {kind: len(sizes[kind]) for kind in KINDS},
            "promptCharacters": {kind: {"min": min(values), "max": max(values)} if values else None
                                 for kind, values in sizes.items()}}, ensure_ascii=False, indent=2))
        return
    if args.command == "derive":
        print(json.dumps(runner.derive(args.run_id), ensure_ascii=False, indent=2))
        return
    if args.fixture:
        script = json.loads(args.fixture.read_text(encoding="utf-8"))
        client = JudgeFixtureClient(script.get("script", {}))
        operational = {"mode": "fixture"}
    elif args.live:
        prices = {}
        for provider in ("openai", "gemini"):
            prices[provider] = {
                "input": getattr(args, provider + "_input_price_per_million"),
                "output": getattr(args, provider + "_output_price_per_million"),
                "cachedInput": getattr(args, provider + "_cached_input_price_per_million"),
                "reference": getattr(args, provider + "_pricing_reference"),
                "checkedAt": getattr(args, provider + "_pricing_checked_at")}
        policy = JudgePolicy(live=True, http_request_limit=args.http_request_limit,
                             per_call_cost_limit=args.per_call_cost_limit, total_cost_limit=args.total_cost_limit,
                             timeout_seconds=args.timeout_seconds,
                             estimated_input_tokens={"pointwise": args.estimated_input_tokens_pointwise,
                                                     "pairwise": args.estimated_input_tokens_pairwise},
                             max_output_tokens=dict(runner.config["max_output_tokens"]), prices=prices)
        try:
            preflight_live_policy(policy, runner.config["judges"])
        except ValueError as error:
            parser.error("--live requires decided output limits, guards, token estimates and prices: %s" % error)
        client = JudgeHttpClient(policy, {judge["provider"]: judge["api_key_environment_variable"]
                                          for judge in runner.config["judges"]})
        operational = {"mode": "live", "httpRequestLimit": args.http_request_limit,
                       "perCallCostLimit": args.per_call_cost_limit, "totalCostLimit": args.total_cost_limit,
                       "timeoutSeconds": args.timeout_seconds, "prices": prices}
    else:
        parser.error("run requires --fixture or --live")
    manifest = runner.resume(args.run_id, client) if args.run_id else runner.start(client, operational)
    print(json.dumps({"judgeRunId": manifest["judgeRunId"], "status": manifest["status"],
                      "stopReason": manifest["stopReason"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
