"""Validate a promoted parity bundle and emit its scientific-consumer sidecar."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from research_math import actionbench_parity as parity
from research_math.control_scoring import (
    ARMS, contract_scorer_command, file_ref, resolve_ref, verify_request,
)


def object_digest(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def require_canonical_path(actual: Path, expected: Path, label: str) -> None:
    if Path(actual).resolve() != Path(expected).resolve():
        raise ValueError("Canonical " + label + " path required")


def verify_complete_promotion(root: Path, job: dict, attempt: dict) -> None:
    """Require every receipt-declared output at its stable project target."""
    by_path = {ref["path"]: ref for ref in attempt.get("output_refs", [])}
    for relative in job.get("output_paths", []):
        original_path = f"{attempt['attempt_path']}/workspace/{relative}"
        original_ref = by_path.get(original_path)
        if original_ref is None:
            raise ValueError("Harness-declared output missing from native receipt")
        promoted_path = root / relative
        if file_ref(root, promoted_path)["sha256"] != original_ref["sha256"]:
            raise ValueError("Promoted output differs from harness receipt: " + relative)


def verify_harness_origin(root: Path, harness_plan_path: Path,
                          harness_report_path: Path, native_plan_path: Path,
                          native_receipt_path: Path, approved_plan_digest: str,
                          promoted_attestation_ref: dict) -> tuple[dict, dict, list[dict]]:
    harness_plan = json.loads(harness_plan_path.read_text())
    report = json.loads(harness_report_path.read_text())
    native_plan = json.loads(native_plan_path.read_text())
    receipt = json.loads(native_receipt_path.read_text())
    batch_root = (root / harness_plan.get("output_root", "") /
                  harness_plan.get("batch_id", "")).resolve()
    run_root = (root / native_plan.get("output_root", "") /
                native_plan.get("run_id", "")).resolve()
    require_canonical_path(harness_report_path, batch_root / "report.json",
                           "harness report")
    require_canonical_path(native_receipt_path, run_root / "receipt.json",
                           "native receipt")
    if (json.loads((batch_root / "plan.json").read_text()) != harness_plan or
            json.loads((run_root / "plan.json").read_text()) != native_plan):
        raise ValueError("Supplied plan differs from canonical executed plan copy")
    if (harness_plan.get("plan_digest") != approved_plan_digest or
            object_digest({key: value for key, value in harness_plan.items()
                           if key != "plan_digest"}) != approved_plan_digest or
            report.get("plan_digest") != approved_plan_digest or
            report.get("batch_id") != harness_plan.get("batch_id") or
            report.get("status") != "completed"):
        raise ValueError("Approved completed harness identity required")
    tasks = harness_plan.get("tasks", [])
    if len(tasks) != 1 or tasks[0].get("task_id") != "actionbench-official-faithful-parity":
        raise ValueError("Exact one-task parity harness required")
    if tasks[0].get("plan_ref") != file_ref(root, native_plan_path):
        raise ValueError("Harness does not bind the supplied native plan")
    if (native_plan.get("plan_digest") != object_digest({
            key: value for key, value in native_plan.items() if key != "plan_digest"}) or
            native_plan.get("purpose") != "engineering" or
            native_plan.get("evidence_mode") != "developmental" or
            native_plan.get("protocol_ref") is not None or
            native_plan.get("protocol_digest") is not None or
            receipt.get("plan_digest") != native_plan.get("plan_digest") or
            receipt.get("run_id") != native_plan.get("run_id") or
            receipt.get("purpose") != "engineering" or
            receipt.get("evidence_mode") != "developmental" or
            receipt.get("status") != "completed" or
            receipt.get("provenance") != native_plan.get("provenance")):
        raise ValueError("Completed engineering native receipt required")
    jobs = native_plan.get("jobs", [])
    attempts = receipt.get("attempts", [])
    if (len(jobs) != 1 or len(attempts) != 1 or
            jobs[0].get("arm_role") != "scorer-parity" or
            jobs[0].get("group") != "engineering" or
            attempts[0].get("status") != "completed" or
            attempts[0].get("exit_code") != 0 or
            attempts[0].get("input_refs") != jobs[0].get("input_refs") or
            attempts[0].get("code_refs") != jobs[0].get("code_refs") or
            any(attempts[0].get(key) != jobs[0].get(key)
                for key in ("trial_id", "seed", "group", "arm_role"))):
        raise ValueError("Exact completed scorer-parity attempt required")
    attempt_root = run_root / attempts[0]["attempt_id"]
    require_canonical_path(root / attempts[0]["attempt_path"], attempt_root,
                           "native attempt")
    if json.loads((attempt_root / "attempt.json").read_text()) != attempts[0]:
        raise ValueError("Canonical attempt record differs from native receipt")
    expected_outputs = []
    by_path = {ref["path"]: ref for ref in attempts[0].get("output_refs", [])}
    for relative in jobs[0].get("output_paths", []):
        path = f"{attempts[0]['attempt_path']}/workspace/{relative}"
        ref = by_path.get(path)
        if ref is None:
            raise ValueError("Harness-declared output missing from native receipt")
        resolve_ref(root, ref)
        expected_outputs.append(ref)
    task_result = report.get("tasks", {}).get(tasks[0]["task_id"], {})
    task_root = batch_root / "tasks" / tasks[0]["task_id"]
    if json.loads((task_root / "task.json").read_text()) != tasks[0]:
        raise ValueError("Canonical harness task differs from approved plan")
    raw_task_result = json.loads((task_root / "result.json").read_text())
    if any(task_result.get(key) != value for key, value in raw_task_result.items()):
        raise ValueError("Harness report differs from canonical task result")
    state = json.loads((batch_root / "state.json").read_text())
    if (state.get("plan_digest") != approved_plan_digest or
            state.get("status") != "completed" or
            state.get("tasks") != report.get("tasks")):
        raise ValueError("Canonical completed harness state required")
    if (task_result.get("status") != "completed" or
            task_result.get("receipt_ref") != file_ref(root, native_receipt_path) or
            sorted(task_result.get("output_refs", []), key=lambda value: value["path"]) !=
            sorted(expected_outputs, key=lambda value: value["path"])):
        raise ValueError("Harness report output inventory differs from native receipt")
    original_attestation_path = (
        f"{attempts[0]['attempt_path']}/workspace/"
        "actionmesh/actionbench-parity-output/parity-bundle-attestation.json")
    original_attestation_ref = by_path.get(original_attestation_path)
    if (original_attestation_ref is None or
            original_attestation_ref["sha256"] != promoted_attestation_ref["sha256"]):
        raise ValueError("Promoted attestation is not the harness-recorded output")
    origin_refs = [
        file_ref(root, batch_root / "plan.json"),
        file_ref(root, batch_root / "state.json"),
        file_ref(root, task_root / "task.json"),
        file_ref(root, task_root / "result.json"),
        file_ref(root, run_root / "plan.json"),
        file_ref(root, attempt_root / "attempt.json"),
    ]
    return jobs[0], attempts[0], origin_refs


def finalize(root: Path, request_path: Path, contract_path: Path, output: Path,
             harness_plan_path: Path, harness_report_path: Path,
             native_plan_path: Path, native_receipt_path: Path,
             approved_plan_digest: str) -> dict:
    (root, request_path, contract_path, output, harness_plan_path,
     harness_report_path, native_plan_path, native_receipt_path) = map(
        lambda value: Path(value).resolve(),
        (root, request_path, contract_path, output, harness_plan_path,
         harness_report_path, native_plan_path, native_receipt_path))
    expected_output = root / "actionmesh" / "actionbench-parity-output"
    if output != expected_output:
        raise ValueError("Promoted bundle must use the contract's exact project target")
    request = json.loads(request_path.read_text())
    verify_request(root, request)
    contract = parity.validate_parity_contract(
        root, request, request_path, contract_path)
    parity.verify_frozen_contract_refs(root, contract)
    attestation_path = output / "parity-bundle-attestation.json"
    record_path = output / "record.json"
    evidence_path = output / "parity-evidence.json"
    attestation = json.loads(attestation_path.read_text())
    record = json.loads(record_path.read_text())
    evidence = json.loads(evidence_path.read_text())
    job, attempt, origin_refs = verify_harness_origin(
        root, harness_plan_path, harness_report_path, native_plan_path,
        native_receipt_path, approved_plan_digest, file_ref(root, attestation_path))
    verify_complete_promotion(root, job, attempt)
    if (attestation.get("kind") != "actionbench-parity-bundle-attestation" or
            attestation.get("bundle_target") != output.relative_to(root).as_posix() or
            attestation.get("bundle_promotion_required") is not True or
            record.get("status") != "passed" or evidence.get("status") != "passed" or
            record.get("frozen_ref_recheck", {}).get("status") != "passed" or
            attestation.get("scientific_effect_qualification") is not False or
            attestation.get("native_contract_qualified") is not False or
            attestation.get("metric_tolerances") != contract["metric_tolerances"] or
            attestation.get("source_refs") != contract["source_refs"]):
        raise ValueError("Promoted parity bundle is not a complete passed attestation")
    expected_record_ref = file_ref(root, record_path)
    expected_evidence_ref = file_ref(root, evidence_path)
    if (attestation.get("record_ref") != expected_record_ref or
            attestation.get("parity_evidence_ref") != expected_evidence_ref or
            attestation.get("request_ref") != file_ref(root, request_path) or
            attestation.get("population_ref") != request["population_ref"] or
            attestation.get("ground_truth_ref") != request["ground_truth_ref"] or
            attestation.get("environment_ref") != contract["environment_ref"] or
            attestation.get("sample_manifest_ref") != contract["sample_manifest_ref"] or
            attestation.get("official_scorer") != contract["official_scorer"] or
            attestation.get("harness_scorer") != contract["harness_scorer"]):
        raise ValueError("Promoted parity attestation identity mismatch")
    for promoted_ref in (expected_record_ref, expected_evidence_ref):
        original_path = (f"{attempt['attempt_path']}/workspace/" +
                         promoted_ref["path"])
        original = next((ref for ref in attempt["output_refs"]
                         if ref["path"] == original_path), None)
        if original is None or original["sha256"] != promoted_ref["sha256"]:
            raise ValueError("Promoted core evidence is not harness-recorded")
    for ref in (expected_record_ref, expected_evidence_ref,
                attestation["request_ref"], attestation["population_ref"],
                attestation["ground_truth_ref"], attestation["environment_ref"],
                attestation["sample_manifest_ref"]):
        resolve_ref(root, ref)
    for arm in ARMS:
        arm_evidence = attestation.get("arm_evidence", {}).get(arm, {})
        expected_prediction = next(
            row["sequence_ref"] for row in request["arms"] if row["arm"] == arm)
        if arm_evidence.get("prediction_ref") != expected_prediction:
            raise ValueError("Promoted prediction identity mismatch: " + arm)
        resolve_ref(root, arm_evidence["prediction_ref"])
        if arm_evidence["prediction_ref"] not in attempt["input_refs"]:
            raise ValueError("Prediction is not immutable attempt input: " + arm)
        for key in ("official_output_ref", "faithful_output_ref"):
            resolve_ref(root, arm_evidence[key])
            expected_path = (f"{attempt['attempt_path']}/workspace/" +
                             arm_evidence[key]["path"])
            original = next((ref for ref in attempt["output_refs"]
                             if ref["path"] == expected_path), None)
            if original is None or original["sha256"] != arm_evidence[key]["sha256"]:
                raise ValueError("Arm evidence is not harness-recorded: " + arm)
    harness = dict(contract["harness_scorer"])
    harness["command"] = contract_scorer_command(root, request)
    finalizer_ref = file_ref(root, Path(__file__).resolve())
    verification = {
        "official_scorer": contract["official_scorer"],
        "harness_scorer": harness,
        "canonical_parity_harness_scorer": contract["harness_scorer"],
        "sample_manifest_ref": contract["sample_manifest_ref"],
        "source_refs": contract["source_refs"] + [finalizer_ref],
        "metric_tolerances": contract["metric_tolerances"],
        "bundle_attestation_ref": file_ref(root, attestation_path),
        "record_ref": expected_record_ref,
        "parity_evidence_ref": expected_evidence_ref,
        "harness_plan_ref": file_ref(root, harness_plan_path),
        "harness_report_ref": file_ref(root, harness_report_path),
        "native_plan_ref": file_ref(root, native_plan_path),
        "native_receipt_ref": file_ref(root, native_receipt_path),
        "approved_plan_digest": approved_plan_digest,
        "trial_id": job["trial_id"],
        "attempt_id": attempt["attempt_id"],
        "execution_origin_refs": origin_refs,
        "scientific_effect_qualification": False,
        "native_contract_qualified": False,
    }
    path = output / "faithful-harness-verification.json"
    if path.exists():
        raise FileExistsError("Scientific-consumer sidecar is single-use")
    parity.census.write_json(path, verification)
    return verification


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "request", "contract", "output", "harness-plan",
                 "harness-report", "native-plan", "native-receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--approved-plan-digest", required=True)
    args = parser.parse_args()
    verification = finalize(
        args.root, args.request, args.contract, args.output, args.harness_plan,
        args.harness_report, args.native_plan, args.native_receipt,
        args.approved_plan_digest)
    print(json.dumps({"verification_ref": file_ref(
        args.root.resolve(), args.output.resolve() / "faithful-harness-verification.json"),
        "bundle_attestation_ref": verification["bundle_attestation_ref"],
        "scientific_effect_qualification": False,
        "native_contract_qualified": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
