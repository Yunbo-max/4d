"""Fail-closed admission of one completed three-arm calibration unit.

This module only verifies retained evidence and emits a measurement sidecar.  It
does not run inference or scoring, qualify a scientific effect, or build a queue.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

from research_math.complete_unit_contract import (
    ARMS,
    METRICS,
    complete_unit_output_paths,
    require_three_scores,
    validate_generation_profile,
)
from research_math.control_scoring import file_ref, resolve_ref


REQUIRED_STAGES = ("generation", "export", "controls", "official_scoring")
INCIDENTAL_PYTHON_CACHE_PATHS = tuple(
    "official-scores.json.official/official-source/__pycache__/" + name +
    ".cpython-312.pyc"
    for name in ("benchmark", "chamfer", "icp", "sample_mesh", "sample_point_cloud")
)


def object_digest(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_admission_contract(contract: dict) -> None:
    source = contract.get("source_unit")
    if (contract.get("kind") != "actionbench-complete-unit-admission-contract" or
            contract.get("version") != "1.1.0" or not isinstance(source, dict) or
            source.get("run_id") != "complete-lowram-r7" or
            source.get("approved_plan_digest") !=
            "a94aa69f8605266587f56f0977740001bbd21971463bb5b78d4e20eaf2bf574b" or
            source.get("task_id") != "complete-fp16-lowram-v1-three-arm-unit" or
            source.get("trial_id") != "complete-fp16-lowram-v1-three-arm-unit" or
            source.get("runtime_profile") != "fp16-lowram-v1" or
            source.get("generation_seed") != 42 or
            source.get("official_scoring_seed") != 44 or
            source.get("successful_output_file_count") != 117 or
            source.get("nested_pre_result_output_count") != 121 or
            source.get("incidental_python_cache_paths") !=
            list(INCIDENTAL_PYTHON_CACHE_PATHS) or
            contract.get("required_status") != "admitted_engineering_complete_unit" or
            contract.get("scientific_effect_qualification") is not False or
            contract.get("native_scientific_qualification") is not False or
            contract.get("candidate_methods_tested") is not False or
            contract.get("queue_approved") is not False or
            contract.get("queue_generated") is not False):
        raise ValueError("Frozen first complete-unit admission contract required")
    uid = source.get("uid")
    if not isinstance(uid, str) or not uid or Path(uid).name != uid:
        raise ValueError("Frozen calibration UID required")


def _positive_number(value, label: str, *, upper: float | None = None) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or value <= 0 or
            (upper is not None and value > upper)):
        raise ValueError("Invalid " + label)
    return float(value)


def _validate_scores(scores: dict, uid: str) -> dict:
    if not isinstance(scores, dict) or set(scores) != set(ARMS):
        raise ValueError("Exactly three completed score arms required")
    checked = {}
    for arm in ARMS:
        row = scores[arm]
        if not isinstance(row, dict) or set(row) != set(METRICS):
            raise ValueError("Exact official metric set required: " + arm)
        checked[arm] = {}
        for metric in METRICS:
            value = row[metric]
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(value) or value < 0):
                raise ValueError("Invalid official metric: " + arm + "/" + metric)
            checked[arm][metric] = float(value)
    return checked


def validate_official_report(report: dict, uid: str) -> dict:
    if (report.get("seed") != 44 or report.get("device") != "cuda:0" or
            report.get("denominator", {}).get("frozen") is not True or
            report.get("denominator", {}).get("n_declared") != 3 or
            report.get("summary") != {
                "n_total": 3, "n_success": 3, "n_failed": 0,
                "success_rate": 1.0,
            }):
        raise ValueError("Frozen successful official three-arm report required")
    return require_three_scores(report, uid)


def validate_completed_result(result: dict, uid: str, gpu_total_mib: int) -> dict:
    """Return only measured fields after validating a completed runner result."""
    if (result.get("status") != "completed" or
            result.get("scientific_effect_qualification") is not False or
            result.get("candidate_methods_tested") is not False):
        raise ValueError("Completed non-scientific calibration result required")
    if not isinstance(gpu_total_mib, int) or gpu_total_mib <= 0:
        raise ValueError("Positive physical GPU memory required")
    if not isinstance(uid, str) or not uid:
        raise ValueError("Calibration UID required")
    profile = result.get("runtime_profile")
    if profile not in ("default", "fp16-lowram-v1"):
        raise ValueError("Reviewed runtime profile required")
    stages = result.get("stages")
    if not isinstance(stages, dict) or any(name not in stages for name in REQUIRED_STAGES):
        raise ValueError("All generation/export/control/scoring stages required")
    stage_seconds = {}
    for name in REQUIRED_STAGES:
        stage = stages[name]
        if not isinstance(stage, dict) or stage.get("status") != "completed":
            raise ValueError("Completed stage required: " + name)
        stage_seconds[name] = _positive_number(
            stage.get("elapsed_seconds"), name + " elapsed seconds", upper=27000)
    elapsed = _positive_number(result.get("elapsed_seconds"),
                               "complete-unit elapsed seconds", upper=27000)
    device = result.get("device_memory")
    if (not isinstance(device, dict) or device.get("sample_interval_seconds") != 1 or
            not isinstance(device.get("samples"), int) or device["samples"] <= 0 or
            device.get("exact_peak") is not False or device.get("errors") != []):
        raise ValueError("Complete one-second GPU telemetry required")
    peak = _positive_number(device.get("observed_peak_mib"),
                            "observed GPU peak", upper=gpu_total_mib)
    host = result.get("host_resources")
    if (not isinstance(host, dict) or host.get("sample_interval_seconds") != 1 or
            not isinstance(host.get("samples"), int) or host["samples"] <= 0 or
            host.get("exact_peak") is not False or host.get("errors") != []):
        raise ValueError("Complete one-second host telemetry required")
    host_measurements = {}
    for key in ("observed_peak_rss_bytes", "observed_peak_output_bytes",
                "minimum_free_disk_bytes"):
        host_measurements[key] = int(_positive_number(host.get(key), key))
    return {
        "uid": uid,
        "runtime_profile": profile,
        "gpu_uuid": result.get("gpu_uuid"),
        "gpu_total_mib": gpu_total_mib,
        "observed_peak_mib": peak,
        "device_sample_interval_seconds": 1,
        "device_samples": device["samples"],
        "elapsed_seconds": elapsed,
        "stage_elapsed_seconds": stage_seconds,
        "host_resources": host_measurements,
        "scores": _validate_scores(result.get("scores"), uid),
        "scientific_effect_qualification": False,
        "candidate_methods_tested": False,
        "queue_approved": False,
        "queue_generated": False,
    }


def _jsonl_rows(path: Path, label: str) -> list[dict]:
    rows = []
    for index, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line:
            raise ValueError("Blank raw telemetry row: " + label)
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError("Invalid raw telemetry JSON: " + label) from error
        if not isinstance(row, dict):
            raise ValueError("Raw telemetry object required: " + label)
        rows.append(row)
    if not rows:
        raise ValueError("Raw telemetry rows required: " + label)
    return rows


def _finite_nonnegative(value, label: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or value < 0):
        raise ValueError("Invalid " + label)
    return float(value)


def validate_telemetry(result: dict, device_path: Path, host_path: Path,
                       gpu_uuid: str, gpu_total_mib: int) -> None:
    """Recompute the retained resource summaries from their raw JSONL rows."""
    if not isinstance(gpu_uuid, str) or not gpu_uuid.startswith("GPU-"):
        raise ValueError("Physical GPU UUID required for telemetry admission")
    device_summary = result.get("device_memory", {})
    host_summary = result.get("host_resources", {})
    device_rows = _jsonl_rows(device_path, "device")
    host_rows = _jsonl_rows(host_path, "host")
    if (len(device_rows) != device_summary.get("samples") or
            len(host_rows) != host_summary.get("samples")):
        raise ValueError("Raw telemetry count differs from retained summary")

    device_used = []
    observed_at = []
    for row in device_rows:
        if set(row) != {"observed_at", "uuid", "name", "total_mib", "used_mib",
                       "utilization_percent"}:
            raise ValueError("Error or malformed device telemetry row")
        timestamp = row["observed_at"]
        try:
            parsed_at = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except (AttributeError, ValueError) as error:
            raise ValueError("Invalid device telemetry timestamp") from error
        epoch = parsed_at.timestamp()
        if (not math.isfinite(epoch) or epoch in observed_at or
                row["uuid"] != gpu_uuid or
                not isinstance(row["name"], str) or not row["name"]):
            raise ValueError("Device telemetry identity or timestamp mismatch")
        observed_at.append(epoch)
        total = _finite_nonnegative(row["total_mib"], "device total memory")
        used = _finite_nonnegative(row["used_mib"], "device used memory")
        utilization = _finite_nonnegative(row["utilization_percent"],
                                          "device utilization")
        if total != float(gpu_total_mib) or used > total or utilization > 100:
            raise ValueError("Device telemetry exceeds physical bounds")
        device_used.append(used)
    if max(device_used) != float(device_summary.get("observed_peak_mib")):
        raise ValueError("Raw device peak differs from retained summary")
    if any(not 0 < right - left <= 12 for left, right in
           zip(observed_at, observed_at[1:])):
        raise ValueError("Device telemetry cadence differs from one-second sampler")
    device_span = observed_at[-1] - observed_at[0]

    host_rss, host_output, host_free, epochs = [], [], [], []
    for row in host_rows:
        if set(row) != {"epoch", "process_tree_rss_bytes", "output_bytes",
                       "free_disk_bytes"}:
            raise ValueError("Error or malformed host telemetry row")
        epochs.append(_finite_nonnegative(row["epoch"], "host telemetry epoch"))
        host_rss.append(_finite_nonnegative(row["process_tree_rss_bytes"],
                                            "host RSS"))
        host_output.append(_finite_nonnegative(row["output_bytes"],
                                               "host output bytes"))
        host_free.append(_finite_nonnegative(row["free_disk_bytes"],
                                             "host free disk bytes"))
    if any(right <= left for left, right in zip(epochs, epochs[1:])):
        raise ValueError("Host telemetry epochs must increase")
    if any(right - left > 2.5 for left, right in zip(epochs, epochs[1:])):
        raise ValueError("Host telemetry cadence differs from one-second sampler")
    elapsed = _positive_number(result.get("elapsed_seconds"),
                               "complete-unit elapsed seconds", upper=27000)
    if epochs[-1] - epochs[0] < max(0, elapsed - 5):
        raise ValueError("Host telemetry does not cover the complete unit")
    stage_span = sum(_positive_number(
        result.get("stages", {}).get(name, {}).get("elapsed_seconds"),
        name + " elapsed seconds", upper=elapsed) for name in REQUIRED_STAGES)
    if (device_span < max(0, stage_span - 5) or
            observed_at[0] < epochs[0] - 5 or observed_at[-1] > epochs[-1] + 5):
        raise ValueError("Device telemetry does not cover the staged GPU workload")
    if (max(host_rss) != float(host_summary.get("observed_peak_rss_bytes")) or
            max(host_output) != float(host_summary.get("observed_peak_output_bytes")) or
            min(host_free) != float(host_summary.get("minimum_free_disk_bytes"))):
        raise ValueError("Raw host resource summary mismatch")


def _remapped_command(root: Path, attempt_root: Path, command) -> list[str]:
    if not isinstance(command, list) or any(not isinstance(value, str) for value in command):
        raise ValueError("Native job command list required")
    remapped = []
    for value in command:
        path = Path(value)
        if path.is_absolute():
            try:
                relative = path.resolve().relative_to(root)
            except ValueError:
                pass
            else:
                value = str(attempt_root / "workspace" / relative)
        remapped.append(value)
    return remapped


def validate_execution_binding(root: Path, origin_root: Path, harness_plan: dict,
                               native_plan: dict,
                               task: dict, job: dict, attempt: dict,
                               task_result: dict, result: dict, task_root: Path,
                               attempt_root: Path) -> Path:
    """Bind task allocation, remapped command, guard and stage records."""
    root, origin_root, task_root, attempt_root = map(lambda path: Path(path).resolve(),
                                                     (root, origin_root, task_root,
                                                      attempt_root))
    origin_attempt_root = origin_root / attempt.get("attempt_path", "")
    context_path = resolve_ref(root, task_result.get("execution_context_ref"))
    _canonical_path(context_path, task_root / "execution-context.json",
                    "harness execution context")
    context = json.loads(context_path.read_text())
    devices = context.get("devices")
    gpu_uuid = result.get("gpu_uuid")
    if (context.get("batch_plan_digest") != harness_plan.get("plan_digest") or
            context.get("native_plan_digest") != native_plan.get("plan_digest") or
            context.get("task_id") != task.get("task_id") or
            context.get("declared_resources") != task.get("resources") or
            devices != [gpu_uuid] or context.get("CUDA_VISIBLE_DEVICES") != gpu_uuid or
            context.get("gate_advanced") is not False or
            not isinstance(context.get("process"), dict) or
            task_result.get("devices") != devices or
            task_result.get("gate_advanced") is not False or
            task_result.get("process") != context.get("process")):
        raise ValueError("Harness execution context differs from admitted unit")
    if (attempt.get("command") != _remapped_command(origin_root, origin_attempt_root,
                                                     job.get("command")) or
            attempt.get("retry_index") != 0 or
            attempt.get("evidence_mode") != native_plan.get("evidence_mode") or
            attempt.get("provenance") != native_plan.get("provenance")):
        raise ValueError("Native attempt execution metadata mismatch")
    if job.get("cwd") != "actionmesh":
        raise ValueError("Frozen complete-unit job cwd required")
    _canonical_path(Path(attempt.get("cwd", "")),
                    origin_attempt_root / "workspace/actionmesh", "attempt cwd")
    guard_path = resolve_ref(root, attempt.get("process_guard_ref"))
    stdout_path = resolve_ref(root, attempt.get("stdout_ref"))
    stderr_path = resolve_ref(root, attempt.get("stderr_ref"))
    guard = json.loads(guard_path.read_text())
    if (guard.get("command") != attempt.get("command") or
            guard.get("cwd") != attempt.get("cwd") or
            guard.get("status") != "completed" or guard.get("exit_code") != 0 or
            guard.get("reason_code") is not None or
            guard.get("guard_process", {}).get("parent_pid") !=
            context["process"].get("pid")):
        raise ValueError("Completed process guard differs from native attempt")
    attempt_seconds = _positive_number(
        attempt.get("seconds"), "native attempt seconds",
        upper=native_plan.get("limits", {}).get("wall_time_seconds"))
    guard_seconds = _positive_number(guard.get("seconds"), "process guard seconds",
                                     upper=attempt_seconds)
    elapsed = _positive_number(result.get("elapsed_seconds"),
                               "complete-unit elapsed seconds", upper=attempt_seconds)
    if guard_seconds > attempt_seconds or elapsed > attempt_seconds:
        raise ValueError("Execution timing exceeds native attempt")
    for path, label in ((stdout_path, "attempt stdout"), (stderr_path, "attempt stderr")):
        if not path.is_file():
            raise ValueError("Missing " + label)
    stage_total = 0.0
    output = attempt_root / "workspace/actionmesh/unit-output"
    for stage_name, filename in (("generation", "generation.execution.json"),
                                 ("official_scoring",
                                  "official-scoring.execution.json")):
        stage = result.get("stages", {}).get(stage_name, {})
        retained = json.loads((output / filename).read_text())
        if (retained.get("status") != "completed" or retained.get("exit_code") != 0 or
                retained.get("elapsed_seconds") != stage.get("elapsed_seconds") or
                stage.get("status") != "completed"):
            raise ValueError("Retained stage execution mismatch: " + stage_name)
    for name in REQUIRED_STAGES:
        stage_total += _positive_number(
            result.get("stages", {}).get(name, {}).get("elapsed_seconds"),
            name + " elapsed seconds", upper=elapsed)
    if stage_total > elapsed + 1e-6:
        raise ValueError("Stage timings exceed complete-unit elapsed time")
    return context_path


def verify_output_closure(root: Path, job: dict, attempt: dict,
                          uid: str) -> list[dict]:
    """Rehash the exact 117 receipt outputs and the runner's nested inventory."""
    root = Path(root).resolve()
    expected = complete_unit_output_paths(uid)
    if job.get("output_paths") != expected or len(expected) != 117:
        raise ValueError("Exact 117-file complete-unit plan closure required")
    attempt_path = attempt.get("attempt_path")
    if not isinstance(attempt_path, str) or not attempt_path:
        raise ValueError("Canonical attempt path required")
    refs = attempt.get("output_refs")
    if not isinstance(refs, list):
        raise ValueError("Native receipt output refs required")
    by_path = {}
    for ref in refs:
        path = ref.get("path") if isinstance(ref, dict) else None
        if not isinstance(path, str) or path in by_path:
            raise ValueError("Unique native receipt output refs required")
        by_path[path] = ref
    verified = []
    for relative in expected:
        receipt_path = attempt_path + "/workspace/" + relative
        ref = by_path.get(receipt_path)
        if ref is None:
            raise ValueError("Harness-declared output missing from receipt: " + relative)
        resolve_ref(root, ref)
        verified.append(ref)
    expected_receipt_paths = {
        attempt_path + "/workspace/" + relative for relative in expected
    }
    if set(by_path) != expected_receipt_paths:
        raise ValueError("Receipt contains undeclared or duplicate complete-unit outputs")

    result_path = root / attempt_path / "workspace/actionmesh/unit-output/result.json"
    result = json.loads(result_path.read_text())
    nested = result.get("outputs")
    if not isinstance(nested, list):
        raise ValueError("Runner nested output inventory required")
    nested_by_path = {}
    for row in nested:
        path = row.get("path") if isinstance(row, dict) else None
        if not isinstance(path, str) or path in nested_by_path:
            raise ValueError("Unique runner nested output paths required")
        nested_by_path[path] = row
    expected_nested = {
        Path(relative).relative_to("actionmesh/unit-output").as_posix()
        for relative in expected if not relative.endswith("/result.json")
    }
    expected_nested.update(INCIDENTAL_PYTHON_CACHE_PATHS)
    if set(nested_by_path) != expected_nested:
        raise ValueError("Runner nested inventory differs from successful closure")
    unit_root = result_path.parent
    for relative, row in nested_by_path.items():
        path = unit_root / relative
        ref = file_ref(root, path)
        if row.get("sha256") != ref["sha256"] or row.get("bytes") != path.stat().st_size:
            raise ValueError("Runner nested output hash or size mismatch: " + relative)
    return verified


def _canonical_path(actual: Path, expected: Path, label: str) -> None:
    if Path(actual).resolve() != Path(expected).resolve():
        raise ValueError("Canonical " + label + " path required")


def verify_origin(root: Path, harness_plan_path: Path, harness_report_path: Path,
                  native_plan_path: Path, native_receipt_path: Path,
                  approved_plan_digest: str, contract: dict) -> tuple[dict, list[dict]]:
    """Bind the successful harness/native records and return an admission body."""
    root = Path(root).resolve()
    validate_admission_contract(contract)
    paths = [Path(value).resolve() for value in (
        harness_plan_path, harness_report_path, native_plan_path, native_receipt_path)]
    harness_plan, report, native_plan, receipt = [
        json.loads(path.read_text()) for path in paths]
    source_contract = contract["source_unit"]
    if (harness_plan.get("batch_id") != source_contract["run_id"] or
            native_plan.get("run_id") != source_contract["run_id"] or
            approved_plan_digest != source_contract["approved_plan_digest"]):
        raise ValueError("Runtime evidence differs from prospective admission contract")
    batch_root = root / harness_plan.get("output_root", "") / harness_plan.get("batch_id", "")
    run_root = root / native_plan.get("output_root", "") / native_plan.get("run_id", "")
    _canonical_path(harness_report_path, batch_root / "report.json", "harness report")
    _canonical_path(native_receipt_path, run_root / "receipt.json", "native receipt")
    if (json.loads((batch_root / "plan.json").read_text()) != harness_plan or
            json.loads((run_root / "plan.json").read_text()) != native_plan):
        raise ValueError("Supplied plan differs from canonical executed plan")
    if (harness_plan.get("plan_digest") != approved_plan_digest or
            object_digest({k: v for k, v in harness_plan.items() if k != "plan_digest"}) !=
            approved_plan_digest or report.get("plan_digest") != approved_plan_digest or
            report.get("status") != "completed"):
        raise ValueError("Approved completed complete-unit harness required")
    tasks = harness_plan.get("tasks", [])
    jobs = native_plan.get("jobs", [])
    attempts = receipt.get("attempts", [])
    if len(tasks) != 1 or len(jobs) != 1 or len(attempts) != 1:
        raise ValueError("Exact one-task, one-job, one-attempt unit required")
    task, job, attempt = tasks[0], jobs[0], attempts[0]
    if (task.get("task_id") != source_contract["task_id"] or
            job.get("trial_id") != source_contract["trial_id"]):
        raise ValueError("Task or trial differs from prospective contract")
    if task.get("plan_ref") != file_ref(root, native_plan_path):
        raise ValueError("Harness does not bind supplied native plan")
    if (native_plan.get("plan_digest") != object_digest({
            k: v for k, v in native_plan.items() if k != "plan_digest"}) or
            native_plan.get("purpose") != "engineering" or
            native_plan.get("evidence_mode") != "developmental" or
            receipt.get("plan_digest") != native_plan.get("plan_digest") or
            receipt.get("status") != "completed" or
            receipt.get("provenance") != native_plan.get("provenance")):
        raise ValueError("Completed engineering native receipt required")
    if (job.get("arm_role") != "complete-current-release-calibration" or
            job.get("group") != "engineering" or job.get("seed") != 42 or
            attempt.get("status") != "completed" or attempt.get("exit_code") != 0 or
            attempt.get("input_refs") != job.get("input_refs") or
            attempt.get("code_refs") != job.get("code_refs") or
            any(attempt.get(key) != job.get(key)
                for key in ("trial_id", "seed", "group", "arm_role"))):
        raise ValueError("Exact completed calibration attempt required")
    for ref in job.get("input_refs", []) + job.get("code_refs", []):
        resolve_ref(root, ref)
    attempt_root = run_root / attempt.get("attempt_id", "")
    _canonical_path(root / attempt.get("attempt_path", ""), attempt_root,
                    "native attempt")
    if json.loads((attempt_root / "attempt.json").read_text()) != attempt:
        raise ValueError("Canonical attempt differs from native receipt")

    manifest_refs = [ref for ref in job.get("input_refs", [])
                     if ref.get("path", "").endswith("/unit-manifest.json")]
    if len(manifest_refs) != 1:
        raise ValueError("One frozen unit manifest input required")
    manifest_path = resolve_ref(root, manifest_refs[0])
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "frozen_engineering_current_release_unit":
        raise ValueError("Frozen engineering unit manifest required")
    uid = manifest.get("calibration_unit", {}).get("uid")
    profile = validate_generation_profile(manifest.get("generation", {}))
    if uid != source_contract["uid"] or profile != source_contract["runtime_profile"]:
        raise ValueError("Frozen unit identity differs from admission contract")
    outputs = verify_output_closure(root, job, attempt, uid)

    task_id = task.get("task_id")
    task_root = batch_root / "tasks" / str(task_id)
    task_result = report.get("tasks", {}).get(task_id, {})
    raw_result = json.loads((task_root / "result.json").read_text())
    state = json.loads((batch_root / "state.json").read_text())
    if (json.loads((task_root / "task.json").read_text()) != task or
            any(task_result.get(k) != v for k, v in raw_result.items()) or
            state.get("plan_digest") != approved_plan_digest or
            state.get("status") != "completed" or
            state.get("tasks") != report.get("tasks") or
            task_result.get("status") != "completed" or
            task_result.get("receipt_ref") != file_ref(root, native_receipt_path) or
            sorted(task_result.get("output_refs", []), key=lambda x: x["path"]) !=
            sorted(outputs, key=lambda x: x["path"])):
        raise ValueError("Canonical completed harness task evidence required")

    snapshots = report.get("gpu_snapshot")
    if not isinstance(snapshots, list) or len(snapshots) != 1:
        raise ValueError("Exactly one physical GPU snapshot required")
    snapshot = snapshots[0]
    result_path = attempt_root / "workspace/actionmesh/unit-output/result.json"
    result = json.loads(result_path.read_text())
    if result.get("gpu_uuid") != snapshot.get("uuid"):
        raise ValueError("Result GPU differs from harness allocation")
    measurement = validate_completed_result(result, uid, snapshot.get("total_mib"))
    origin_root_value = report.get("root")
    if not isinstance(origin_root_value, str) or not Path(origin_root_value).is_absolute():
        raise ValueError("Absolute recorded source harness root required")
    execution_context_path = validate_execution_binding(
        root, Path(origin_root_value), harness_plan, native_plan, task, job, attempt,
        task_result, result, task_root, attempt_root)
    validate_telemetry(
        result,
        attempt_root / "workspace/actionmesh/unit-output/device-samples.jsonl",
        attempt_root / "workspace/actionmesh/unit-output/host-samples.jsonl",
        snapshot.get("uuid"), snapshot.get("total_mib"),
    )
    official_path = attempt_root / "workspace/actionmesh/unit-output/official-scores.json"
    official_scores = validate_official_report(json.loads(official_path.read_text()), uid)
    if measurement["scores"] != official_scores:
        raise ValueError("Result score summary differs from retained official report")
    if measurement["runtime_profile"] != profile:
        raise ValueError("Result runtime profile differs from frozen manifest")
    process_guard_path = resolve_ref(root, attempt["process_guard_ref"])
    stdout_path = resolve_ref(root, attempt["stdout_ref"])
    stderr_path = resolve_ref(root, attempt["stderr_ref"])
    origin_refs = [file_ref(root, path) for path in (
        batch_root / "plan.json", batch_root / "state.json", batch_root / "report.json",
        task_root / "task.json", task_root / "result.json", run_root / "plan.json",
        execution_context_path, run_root / "receipt.json", attempt_root / "attempt.json",
        process_guard_path, stdout_path, stderr_path,
        attempt_root / "workspace/actionmesh/unit-output/generation.execution.json",
        attempt_root / "workspace/actionmesh/unit-output/official-scoring.execution.json")]
    admission = {
        "kind": "actionbench-complete-unit-admission",
        "version": "1.0.0",
        "status": "admitted_engineering_complete_unit",
        "run_id": native_plan.get("run_id"),
        "trial_id": job.get("trial_id"),
        "attempt_id": attempt.get("attempt_id"),
        "approved_plan_digest": approved_plan_digest,
        "unit_manifest_ref": file_ref(root, manifest_path),
        "result_ref": file_ref(root, result_path),
        "origin_refs": origin_refs,
        "output_refs": outputs,
        "successful_output_file_count": len(outputs),
        "measurement": measurement,
        "eligible_for_queue_pricing": True,
        "scientific_effect_qualification": False,
        "native_scientific_qualification": False,
        "candidate_methods_tested": False,
        "queue_approved": False,
        "queue_generated": False,
    }
    return admission, origin_refs


def admit(root: Path, contract_path: Path, harness_plan: Path, harness_report: Path,
          native_plan: Path, native_receipt: Path, approved_plan_digest: str,
          output: Path) -> dict:
    output = Path(output).resolve()
    root = Path(root).resolve()
    output.relative_to(root)
    if output.exists():
        raise FileExistsError("Complete-unit admission is single-use")
    contract_path = Path(contract_path).resolve()
    contract_path.relative_to(root)
    contract = json.loads(contract_path.read_text())
    admission, _ = verify_origin(root, harness_plan, harness_report, native_plan,
                                 native_receipt, approved_plan_digest, contract)
    admission["contract_ref"] = file_ref(root, contract_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(admission, indent=2, allow_nan=False) + "\n")
    return admission


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "contract", "harness-plan", "harness-report", "native-plan",
                 "native-receipt", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--approved-plan-digest", required=True)
    args = parser.parse_args()
    admission = admit(args.root, args.contract, args.harness_plan, args.harness_report,
                      args.native_plan, args.native_receipt,
                      args.approved_plan_digest, args.output)
    print(json.dumps({
        "admission_ref": file_ref(args.root.resolve(), args.output.resolve()),
        "status": admission["status"],
        "elapsed_seconds": admission["measurement"]["elapsed_seconds"],
        "observed_peak_mib": admission["measurement"]["observed_peak_mib"],
        "scientific_effect_qualification": False,
        "queue_approved": False,
        "queue_generated": False,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
