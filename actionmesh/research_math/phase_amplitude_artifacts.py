"""Materialize C20's four real method/control arms from one frozen target.

The target must be produced by :mod:`c20_consensus_target` from an exact retained
ActionMesh decoder context.  All arms share that target and the same B0 sequence.
This module never loads a model, GT, camera, scorer or confirmation outcome.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import tarfile
import time
import traceback
import sys

import numpy as np

from research_math.phase_amplitude_candidate import (
    PhaseAmplitudeError, build_endpoint_sine_basis, solve_phase_amplitude)


CANDIDATE_ID = "4d-math-20261006-c20"
ARMS = ("phase_only", "amplitude_only", "simple_lag",
        "joint_monotone_phase_amplitude")
CANDIDATE_ARM = "joint_monotone_phase_amplitude"
METHOD_IDS = {
    "phase_only": "c20-full-phase-only-control",
    "amplitude_only": "c20-action-amplitude-only-control",
    "simple_lag": "c20-rank-one-endpoint-lag-control",
    CANDIDATE_ARM: CANDIDATE_ID,
}
ERROR_MESSAGE_LIMIT = 4096
TRACEBACK_LIMIT = 16384
PRIMARY_PARAMETER_PROFILE = {
    "phase_rank": 3, "lambda_value": 0.01, "min_slope": 0.25,
    "identifiability_floor": 0.1, "max_abs_warp": 0.25,
    "max_relative_linearization_remainder": 0.25,
    "max_gamma_noise_amplification": 10.0,
    "max_relative_stationarity_residual": 1e-6,
    "min_action_residual_fraction": 0.05, "max_abs_gamma": 0.25,
    "max_amplitude_displacement": 0.10,
    "min_relative_target_improvement": 0.01,
    "min_face_area_ratio": 0.10, "max_artifact_bytes": 1073741824,
}
PRIMARY_PARAMETER_PROFILE_ID = "c20-primary-v1"


def validate_primary_parameters(parameters: dict) -> None:
    if parameters != PRIMARY_PARAMETER_PROFILE:
        raise ValueError("Exact frozen C20 primary-v1 parameters required")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def canonical_digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def validate_producer_lineage(root: Path, *, producer_plan: Path,
                              producer_receipt: Path,
                              producer_harness_plan: Path,
                              producer_harness_report: Path,
                              producer_consumption: Path,
                              target_report: Path, target_path: Path) -> list[dict]:
    """Verify native+harness+authorization evidence without trusting a builder."""
    from datetime import datetime
    from research_math import c20_consensus_target as target_module

    root = Path(root).resolve()
    paths = [Path(path).resolve() for path in (
        producer_plan, producer_receipt, producer_harness_plan,
        producer_harness_report, producer_consumption, target_report, target_path)]
    for path in paths:
        path.relative_to(root)
        if path.is_symlink() or not path.is_file():
            raise ValueError("Physical C20 producer lineage file required")
    (producer_plan, producer_receipt, producer_harness_plan,
     producer_harness_report, producer_consumption,
     target_report, target_path) = paths
    plan, receipt, outer, outer_report, consumed = map(
        read_json, (producer_plan, producer_receipt, producer_harness_plan,
                    producer_harness_report, producer_consumption))
    if (plan.get("plan_digest") != canonical_digest({
            key: value for key, value in plan.items() if key != "plan_digest"})
            or outer.get("plan_digest") != canonical_digest({
                key: value for key, value in outer.items() if key != "plan_digest"})):
        raise ValueError("C20 producer plan digest mismatch")
    run_root = (root / plan.get("output_root", "") / plan.get("run_id", "")).resolve()
    batch_root = (root / outer.get("output_root", "") / outer.get("batch_id", "")).resolve()
    run_root.relative_to(root); batch_root.relative_to(root)
    if (producer_receipt != run_root / "receipt.json"
            or producer_harness_report != batch_root / "report.json"
            or read_json(run_root / "plan.json") != plan
            or read_json(batch_root / "plan.json") != outer
            or receipt.get("status") != "completed"
            or receipt.get("plan_digest") != plan.get("plan_digest")
            or outer_report.get("status") != "completed"
            or outer_report.get("plan_digest") != outer.get("plan_digest")):
        raise ValueError("Canonical completed C20 producer runtime evidence required")
    jobs, attempts, tasks = plan.get("jobs"), receipt.get("attempts"), outer.get("tasks")
    if (not isinstance(jobs, list) or len(jobs) != 1
            or not isinstance(attempts, list) or len(attempts) != 1
            or not isinstance(tasks, list) or len(tasks) != 1
            or plan.get("limits", {}).get("max_attempts") != 1
            or plan.get("limits", {}).get("max_retries_per_trial") != 0):
        raise ValueError("Exact one-attempt C20 producer lineage required")
    job, attempt, task = jobs[0], attempts[0], tasks[0]
    attempt_root = (root / attempt.get("attempt_path", "")).resolve()
    task_root = batch_root / "tasks" / task.get("task_id", "")
    state_path = batch_root / "state.json"
    task_path = task_root / "task.json"
    result_path = task_root / "result.json"
    attempt_path = attempt_root / "attempt.json"
    state, task_record, task_result, attempt_record = map(
        read_json, (state_path, task_path, result_path, attempt_path))
    report_tasks = outer_report.get("tasks")
    reported_task = (report_tasks.get(task.get("task_id"), {})
                     if isinstance(report_tasks, dict) else {})
    if (job.get("trial_id") != "c20-decoder-consensus-target"
            or attempt.get("trial_id") != job["trial_id"]
            or attempt.get("status") != "completed" or attempt.get("exit_code") != 0
            or attempt.get("retry_index") != 0
            or any(attempt.get(key) != job.get(key)
                   for key in ("input_refs", "code_refs", "seed", "group", "arm_role"))
            or task.get("task_id") != "c20-decoder-consensus-target"
            or task.get("plan_ref") != file_ref(root, producer_plan)
            or attempt_record != attempt or task_record != task
            or any(reported_task.get(key) != value
                   for key, value in task_result.items())
            or state.get("plan_digest") != outer.get("plan_digest")
            or state.get("status") != "completed"
            or state.get("tasks") != outer_report.get("tasks")
            or reported_task.get("status") != "completed"
            or reported_task.get("receipt_ref") != file_ref(root, producer_receipt)):
        raise ValueError("C20 producer task/attempt identity mismatch")
    output_refs = attempt.get("output_refs", [])
    if (file_ref(root, target_report) not in output_refs
            or file_ref(root, target_path) not in output_refs
            or sorted(reported_task.get("output_refs", []), key=lambda ref: ref["path"]) !=
               sorted(output_refs, key=lambda ref: ref["path"])):
        raise ValueError("C20 producer receipt does not acknowledge target outputs")
    auth_ref = consumed.get("authorization_ref")
    auth_path = resolve_ref(root, auth_ref)
    auth = read_json(auth_path)
    target_module.validate_target_plan(root, plan, auth, auth_path)
    target_record = read_json(target_report)
    target_source = next((ref for ref in job.get("code_refs", [])
                          if ref.get("path") ==
                          "actionmesh/research_math/c20_consensus_target.py"), None)
    if (target_source is None
            or target_record.get("implementation_sha256") != target_source["sha256"]):
        raise ValueError("C20 target report implementation differs from exact plan")
    expected_consumption = (root / "inputs/c20-target/authorization-consumption" /
                            (digest(auth_path) + ".json")).resolve()
    try:
        issued = datetime.fromisoformat(auth["issued_at"].replace("Z", "+00:00"))
        expires = datetime.fromisoformat(auth["expires_at"].replace("Z", "+00:00"))
        consumed_at = datetime.fromisoformat(
            consumed["consumed_at"].replace("Z", "+00:00"))
    except (KeyError, AttributeError, ValueError) as error:
        raise ValueError("C20 producer authorization window is invalid") from error
    expected = {
        "kind": "c20-target-authorization-consumption", "version": 1,
        "state": "consumed_for_exact_plan", "authorization_ref": auth_ref,
        "run_id": auth.get("run_id"), "gpu_uuid": auth.get("gpu_uuid"),
        "native_plan_ref": file_ref(root, producer_plan),
        "native_plan_digest": plan.get("plan_digest"),
        "harness_plan_ref": file_ref(root, producer_harness_plan),
        "harness_plan_digest": outer.get("plan_digest"),
        "no_scientific_retry": True,
    }
    if (producer_consumption != expected_consumption
            or any(consumed.get(key) != value for key, value in expected.items())
            or consumed.get("consumption_digest") != canonical_digest({
                key: value for key, value in consumed.items()
                if key != "consumption_digest"})
            or auth.get("authorization_digest") != canonical_digest({
                key: value for key, value in auth.items()
                if key != "authorization_digest"})
            or auth.get("kind") != "c20-target-gpu-resume-authorization"
            or auth.get("version") != 1
            or auth.get("candidate_id") != CANDIDATE_ID
            or auth.get("scope") != "single_c20_target_attempt"
            or auth.get("explicit_resume_for_exact_attempt") is not True
            or auth.get("no_scientific_retry") is not True
            or auth.get("run_id") != plan.get("run_id")
            or auth.get("generation_seed") != job.get("seed")
            or auth.get("input_freeze_ref") not in job.get("input_refs", [])
            or auth.get("environment_ref") not in job.get("input_refs", [])
            or outer.get("gpus", {}).get("uuids") != [auth.get("gpu_uuid")]
            or plan.get("limits", {}).get("wall_time_seconds") !=
               auth.get("wall_seconds")
            or issued.tzinfo is None or expires.tzinfo is None
            or consumed_at.tzinfo is None or not issued <= consumed_at < expires
            or auth_ref not in job.get("input_refs", [])):
        raise ValueError("C20 producer lacks exact in-window STOP authorization consumption")
    return [file_ref(root, path) for path in (
        state_path, task_path, result_path, attempt_path, auth_path)]


def validate_candidate_execution(root: Path, *, plan_path: Path,
                                 receipt_path: Path,
                                 artifact_path: Path) -> None:
    """Validate the exact retained zero-retry candidate attempt and inventory."""
    root = Path(root).resolve()
    plan_path, receipt_path, artifact_path = map(
        lambda path: Path(path).resolve(), (plan_path, receipt_path, artifact_path))
    for path in (plan_path, receipt_path, artifact_path):
        path.relative_to(root)
        if path.is_symlink() or not path.is_file():
            raise ValueError("Physical C20 candidate execution evidence required")
    plan, receipt = map(read_json, (plan_path, receipt_path))
    if plan.get("plan_digest") != canonical_digest({
            key: value for key, value in plan.items() if key != "plan_digest"}):
        raise ValueError("C20 candidate plan digest mismatch")
    run_root = (root / plan.get("output_root", "") / plan.get("run_id", "")).resolve()
    run_root.relative_to(root)
    if (receipt_path != run_root / "receipt.json"
            or read_json(run_root / "plan.json") != plan
            or receipt.get("status") != "completed"
            or receipt.get("run_id") != plan.get("run_id")
            or receipt.get("plan_digest") != plan.get("plan_digest")
            or receipt.get("purpose") != plan.get("purpose")
            or receipt.get("evidence_mode") != plan.get("evidence_mode")
            or receipt.get("provenance") != plan.get("provenance")):
        raise ValueError("Canonical completed C20 candidate receipt required")
    expected_environment_digest = canonical_digest({
        "python": sys.version, "python_executable": sys.executable,
        "numpy": np.__version__,
        "parameter_profile": PRIMARY_PARAMETER_PROFILE_ID})
    if plan.get("provenance", {}).get("environment_digest") != expected_environment_digest:
        raise ValueError("C20 candidate execution environment differs from frozen plan")
    jobs, attempts = plan.get("jobs"), receipt.get("attempts")
    if (not isinstance(jobs, list) or len(jobs) != 1
            or not isinstance(attempts, list) or len(attempts) != 1
            or plan.get("limits", {}).get("max_attempts") != 1
            or plan.get("limits", {}).get("max_retries_per_trial") != 0):
        raise ValueError("Exact zero-retry C20 candidate attempt required")
    job, attempt = jobs[0], attempts[0]
    artifact = read_json(artifact_path)
    command = job.get("command")
    if (not isinstance(command, list) or len(command) < 5
            or command[0] != sys.executable
            or command[1:3] != ["-m", "research_math.phase_amplitude_artifacts"]
            or len(command[3:]) % 2):
        raise ValueError("Exact C20 candidate materializer command required")
    pairs = list(zip(command[3::2], command[4::2]))
    arguments = {key: value for key, value in pairs}
    required_flags = {
        "--root", "--source-sequence", "--source-report", "--target",
        "--target-report", "--producer-plan", "--producer-receipt",
        "--producer-harness-plan", "--producer-harness-report",
        "--producer-consumption", "--output", "--uid", "--seed",
        "--phase-rank", "--lambda-value", "--min-slope",
        "--identifiability-floor", "--max-abs-warp",
        "--max-relative-linearization-remainder",
        "--max-gamma-noise-amplification", "--max-relative-stationarity-residual",
        "--min-action-residual-fraction", "--max-abs-gamma",
        "--max-amplitude-displacement", "--min-relative-target-improvement",
        "--min-face-area-ratio", "--max-artifact-bytes",
    }
    if artifact.get("application_stage") is not None:
        required_flags.add("--application-stage")
    if (len(arguments) != len(pairs) or set(arguments) != required_flags
            or arguments.get("--root") != ".."
            or arguments.get("--uid") != artifact.get("uid")
            or arguments.get("--seed") != str(artifact.get("seed"))
            or arguments.get("--application-stage") != artifact.get("application_stage")
            or job.get("cwd") != "actionmesh"
            or job.get("group") != "c20-artifact"
            or job.get("arm_role") != "four-role-phase-amplitude-materialization"
            or job.get("seed") != artifact.get("seed")
            or plan.get("purpose") != "engineering"
            or plan.get("evidence_mode") != "developmental"
            or plan.get("protocol_ref") is not None
            or plan.get("protocol_digest") is not None):
        raise ValueError("C20 candidate plan is not the exact materializer task")
    parameter_flags = {
        "phase_rank": "--phase-rank", "lambda_value": "--lambda-value",
        "min_slope": "--min-slope", "identifiability_floor": "--identifiability-floor",
        "max_abs_warp": "--max-abs-warp",
        "max_relative_linearization_remainder": "--max-relative-linearization-remainder",
        "max_gamma_noise_amplification": "--max-gamma-noise-amplification",
        "max_relative_stationarity_residual": "--max-relative-stationarity-residual",
        "min_action_residual_fraction": "--min-action-residual-fraction",
        "max_abs_gamma": "--max-abs-gamma",
        "max_amplitude_displacement": "--max-amplitude-displacement",
        "min_relative_target_improvement": "--min-relative-target-improvement",
        "min_face_area_ratio": "--min-face-area-ratio",
        "max_artifact_bytes": "--max-artifact-bytes",
    }
    if any(arguments[flag] != str(artifact.get("parameters", {}).get(name))
           for name, flag in parameter_flags.items()):
        raise ValueError("C20 candidate parameters differ from retained artifact")
    validate_primary_parameters(artifact.get("parameters"))
    ref_flags = {
        "--source-sequence": artifact["source_refs"]["sequence"],
        "--source-report": artifact["source_refs"]["report"],
        "--target": artifact["target_ref"],
        "--target-report": artifact["producer_provenance_ref"],
        "--producer-plan": artifact["producer_plan_ref"],
        "--producer-receipt": artifact["producer_receipt_ref"],
        "--producer-harness-plan": artifact["producer_harness_plan_ref"],
        "--producer-harness-report": artifact["producer_harness_report_ref"],
        "--producer-consumption": artifact["producer_consumption_ref"],
    }
    for flag, ref in ref_flags.items():
        argument_parts, ref_parts = Path(arguments[flag]).parts, Path(ref["path"]).parts
        if len(argument_parts) < len(ref_parts) or argument_parts[-len(ref_parts):] != ref_parts:
            raise ValueError("C20 candidate command input differs: " + flag)
    code_paths = [root / "actionmesh/research_math" / name for name in (
        "__init__.py", "phase_amplitude_candidate.py", "phase_amplitude_artifacts.py")]
    code_paths.append(root / "actionmesh/prepare_phase_amplitude_candidate.py")
    if job.get("code_refs") != [file_ref(root, path) for path in code_paths]:
        raise ValueError("Exact C20 candidate implementation closure required")
    producer_plan_value = read_json(resolve_ref(root, artifact["producer_plan_ref"]))
    producer_receipt_value = read_json(resolve_ref(root, artifact["producer_receipt_ref"]))
    outer_value = read_json(resolve_ref(root, artifact["producer_harness_plan_ref"]))
    producer_report_value = read_json(resolve_ref(root, artifact["producer_provenance_ref"]))
    producer_job = producer_plan_value["jobs"][0]
    producer_attempt = producer_receipt_value["attempts"][0]
    batch_root = root / outer_value["output_root"] / outer_value["batch_id"]
    run_root_producer = root / producer_plan_value["output_root"] / producer_plan_value["run_id"]
    task_root = batch_root / "tasks/c20-decoder-consensus-target"
    origin_paths = [
        resolve_ref(root, artifact[key]) for key in (
            "producer_plan_ref", "producer_receipt_ref", "producer_harness_plan_ref",
            "producer_harness_report_ref", "producer_consumption_ref", "target_ref")]
    origin_paths += [resolve_ref(root, artifact["producer_provenance_ref"]),
        resolve_ref(root, artifact["source_refs"]["sequence"]),
        resolve_ref(root, artifact["source_refs"]["report"]),
        resolve_ref(root, artifact["producer_provenance_ref"]).with_name(
            "producer-manifest.json"),
        run_root_producer / "plan.json", batch_root / "plan.json",
        batch_root / "state.json", task_root / "task.json", task_root / "result.json",
        root / producer_attempt["attempt_path"] / "attempt.json"]
    nested_refs = [producer_report_value["input_freeze_ref"],
                   producer_report_value["environment_ref"],
                   *producer_report_value["input_refs"],
                   *producer_job["input_refs"], *producer_job["code_refs"]]
    environment = read_json(resolve_ref(root, producer_report_value["environment_ref"]))
    nested_refs += environment["dependency_lock_refs"]
    expected_input_refs = [file_ref(root, path) for path in origin_paths]
    expected_input_refs += [file_ref(root, resolve_ref(root, ref)) for ref in nested_refs]
    expected_by_path = {}
    for ref in expected_input_refs:
        if ref["path"] in expected_by_path and expected_by_path[ref["path"]] != ref:
            raise ValueError("Conflicting exact C20 candidate input closure")
        expected_by_path[ref["path"]] = ref
    actual_by_path = {ref.get("path"): ref for ref in job.get("input_refs", [])
                      if isinstance(ref, dict)}
    if (len(job.get("input_refs", [])) != len(actual_by_path)
            or actual_by_path != expected_by_path):
        raise ValueError("C20 candidate plan has missing or extra staged inputs")
    attempt_relative = Path(attempt.get("attempt_path", ""))
    if attempt_relative.is_absolute() or ".." in attempt_relative.parts:
        raise ValueError("Canonical C20 candidate attempt path required")
    workspace = (root / attempt_relative / "workspace").resolve()
    attempt_record = (root / attempt_relative / "attempt.json").resolve()
    expected_cwd = workspace if job.get("cwd") == "." else workspace / job.get("cwd", "")
    if (attempt_relative.parent != run_root.relative_to(root)
            or attempt.get("attempt_id") != attempt_relative.name
            or attempt_record.is_symlink() or not attempt_record.is_file()
            or read_json(attempt_record) != attempt
            or not workspace.is_dir()
            or job.get("trial_id") != "c20-phase-amplitude-artifact"
            or attempt.get("trial_id") != job.get("trial_id")
            or attempt.get("status") != "completed" or attempt.get("exit_code") != 0
            or attempt.get("retry_index") != 0
            or any(attempt.get(key) != job.get(key)
                   for key in ("input_refs", "code_refs", "seed", "group", "arm_role"))):
        raise ValueError("C20 candidate job/attempt identity mismatch")
    replacements = {}
    for ref in job.get("input_refs", []) + job.get("code_refs", []):
        source = resolve_ref(root, ref)
        staged = (workspace / ref["path"]).resolve(); staged.relative_to(workspace)
        if staged.is_symlink() or not staged.is_file() or digest(staged) != ref["sha256"]:
            raise ValueError("C20 candidate staged closure differs")
        replacements[str(source)] = str(staged)
    expected_command = []
    for value in job.get("command", []):
        for source, staged in sorted(replacements.items(), key=lambda pair: -len(pair[0])):
            value = value.replace(source, staged)
        expected_command.append(value)
    if (attempt.get("command") != expected_command
            or Path(attempt.get("cwd", "")) != expected_cwd):
        raise ValueError("C20 candidate staged command/cwd mismatch")
    expected_outputs = []
    for relative in job.get("output_paths", []):
        path = (workspace / relative).resolve(); path.relative_to(workspace)
        expected_outputs.append(file_ref(root, path))
    if attempt.get("output_refs") != expected_outputs:
        raise ValueError("C20 candidate output receipt inventory mismatch")
    try:
        output_arg = Path(job["command"][job["command"].index("--output") + 1])
    except (KeyError, ValueError, IndexError) as error:
        raise ValueError("C20 candidate command lacks output") from error
    if output_arg.is_absolute() or ".." in output_arg.parts:
        raise ValueError("C20 candidate output must be cwd-relative")
    output_prefix = (Path("actionmesh") / output_arg).as_posix() + "/"
    expected_declared = [output_prefix + name for name in (
        "candidate.json", "manifest.json", "common-target.npz",
        "artifact.tar", "artifact-archive.json",
        *(role + "/report.json" for role in ARMS))]
    if job.get("output_paths") != expected_declared:
        raise ValueError("Exact C20 candidate terminal output inventory required")
    expected_output = ((workspace / output_arg.relative_to(root)) if
        output_arg.is_absolute() else (expected_cwd / output_arg)).resolve()
    expected_output.relative_to(workspace)
    if artifact_path != expected_output / "candidate.json":
        raise ValueError("C20 candidate artifact is outside retained attempt output")


def read_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object: " + str(path))
    return value


def file_ref(root: Path, path: Path) -> dict:
    root, path = Path(root).resolve(), Path(path).resolve()
    return {"path": path.relative_to(root).as_posix(), "sha256": digest(path)}


def resolve_ref(root: Path, ref: dict) -> Path:
    if (not isinstance(ref, dict) or set(ref) != {"path", "sha256"}
            or not isinstance(ref.get("path"), str)
            or not isinstance(ref.get("sha256"), str)
            or len(ref["sha256"]) != 64):
        raise ValueError("Exact path/sha256 reference required")
    relative = Path(ref["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Project-relative nonescaping reference required")
    root = Path(root).resolve(); unresolved = root / relative
    if unresolved.is_symlink():
        raise ValueError("Symlinked evidence is not accepted")
    path = unresolved.resolve(); path.relative_to(root)
    if not path.is_file() or digest(path) != ref["sha256"]:
        raise ValueError("Missing/stale C20 reference: " + ref["path"])
    return path


def _arrays(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as saved:
        return {name: saved[name].copy() for name in saved.files}


def validate_native_arrays(arrays: dict) -> None:
    required = {"vertices", "faces", "frame_indices", "timesteps"}
    if not required.issubset(arrays):
        raise ValueError("Complete native sequence arrays required")
    vertices, faces = arrays["vertices"], arrays["faces"]
    if (vertices.ndim != 3 or vertices.shape[0] != 16 or vertices.shape[2] != 3
            or not np.issubdtype(vertices.dtype, np.floating)
            or not np.isfinite(vertices).all()
            or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or not np.issubdtype(faces.dtype, np.integer)
            or faces.min() < 0 or faces.max() >= vertices.shape[1]
            or not np.array_equal(arrays["frame_indices"], np.arange(16))
            or arrays["timesteps"].shape != (16,)
            or not np.all(np.diff(arrays["timesteps"]) > 0)):
        raise ValueError("Finite identity-preserving 16-frame native sequence required")


def _positive(name: str, value: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not math.isfinite(value) or value <= 0:
        raise ValueError(name + " must be finite and positive")
    return float(value)


def build_action_amplitude_basis(vertices: np.ndarray, times: np.ndarray,
                                 phase_basis: np.ndarray, weights: np.ndarray,
                                 action_seed: np.ndarray,
                                 min_residual_fraction: float) -> tuple[np.ndarray, dict]:
    """W-residualize one declared action direction against the phase design."""
    threshold = _positive("min_action_residual_fraction", min_residual_fraction)
    if threshold > 1:
        raise ValueError("min_action_residual_fraction must be <=1")
    derivative = np.gradient(vertices.astype(np.float64), times, axis=0, edge_order=2)
    phase = np.stack([
        derivative * phase_basis[:, index, None, None]
        for index in range(phase_basis.shape[1])
    ], axis=-1).reshape(-1, phase_basis.shape[1])
    seed = np.asarray(action_seed, dtype=np.float64).reshape(-1)
    sqrt_weight = np.sqrt(np.asarray(weights, dtype=np.float64).reshape(-1))
    weighted_seed = sqrt_weight * seed
    raw_norm = float(np.linalg.norm(weighted_seed))
    if not math.isfinite(raw_norm) or raw_norm <= np.finfo(float).tiny:
        raise PhaseAmplitudeError("action amplitude seed has zero weighted energy")
    if phase.shape[1]:
        weighted_phase = phase * sqrt_weight[:, None]
        q, singular, _ = np.linalg.svd(weighted_phase, full_matrices=False)
        rank = int(np.count_nonzero(singular > 1e-10 * singular[0])) if singular.size else 0
        q = q[:, :rank]
        residual = weighted_seed - q @ (q.T @ weighted_seed)
        raw_overlap = float(np.linalg.norm(q.T @ weighted_seed) / raw_norm)
    else:
        rank = 0; residual = weighted_seed; raw_overlap = 0.0
    residual_norm = float(np.linalg.norm(residual))
    fraction = residual_norm / raw_norm
    if fraction < threshold:
        raise PhaseAmplitudeError("action amplitude collapses after phase-gauge removal")
    basis = (residual / sqrt_weight).reshape(vertices.shape) / residual_norm
    basis[0] = 0.0
    renorm = float(np.linalg.norm(sqrt_weight * basis.reshape(-1)))
    basis /= renorm
    return basis[None], {
        "gauge": "W-orthogonal action direction chosen prospectively; kappa is not empirical proof",
        "phase_rank": rank, "raw_weighted_norm": raw_norm,
        "residual_weighted_norm": residual_norm,
        "residual_fraction": fraction, "raw_phase_overlap_fraction": raw_overlap,
        "normalized_weighted_norm": float(np.linalg.norm(
            sqrt_weight * basis.reshape(-1))),
    }


def _mesh_certificate(source: np.ndarray, output: np.ndarray, faces: np.ndarray,
                      min_face_area_ratio: float) -> dict:
    ratio = _positive("min_face_area_ratio", min_face_area_ratio)
    if ratio > 1:
        raise ValueError("min_face_area_ratio must be <=1")
    if not np.isfinite(output).all() or float(np.max(np.abs(output))) > 1.0:
        raise PhaseAmplitudeError("C20 output leaves native direct-coordinate bounds")
    triangles0 = source[:, faces]
    triangles1 = output[:, faces]
    area0 = 0.5 * np.linalg.norm(np.cross(
        triangles0[:, :, 1] - triangles0[:, :, 0],
        triangles0[:, :, 2] - triangles0[:, :, 0]), axis=-1)
    area1 = 0.5 * np.linalg.norm(np.cross(
        triangles1[:, :, 1] - triangles1[:, :, 0],
        triangles1[:, :, 2] - triangles1[:, :, 0]), axis=-1)
    scale = float(np.linalg.norm(np.ptp(source[0], axis=0)))
    floor = max(np.finfo(float).tiny, scale * scale * 1e-14)
    eligible = area0 > floor
    if np.any(area1[eligible] < ratio * area0[eligible]):
        raise PhaseAmplitudeError("C20 output collapses a source-nondegenerate face")
    return {
        "native_coordinate_bound": 1.0, "finite": True,
        "source_nondegenerate_faces": int(np.count_nonzero(eligible)),
        "minimum_source_relative_face_area": float(np.min(
            area1[eligible] / area0[eligible], initial=np.inf)),
        "minimum_face_area_ratio": ratio,
    }


def _result_certificate(result, action_diagnostics: dict,
                        mesh: dict, target_norm: float,
                        min_improvement: float, max_abs_gamma: float,
                        max_amplitude_displacement: float, *,
                        enforce_improvement: bool) -> dict:
    gamma = result.amplitude_coefficients
    if gamma.size and float(np.max(np.abs(gamma))) > max_abs_gamma:
        raise PhaseAmplitudeError("amplitude coefficient exceeds frozen bound")
    if float(np.max(np.abs(result.amplitude_delta), initial=0.0)) > max_amplitude_displacement:
        raise PhaseAmplitudeError("amplitude displacement exceeds frozen bound")
    improvement = 1.0 - result.weighted_nonlinear_residual / max(
        target_norm, np.finfo(float).tiny)
    if enforce_improvement and improvement < min_improvement:
        raise PhaseAmplitudeError("nonlinear C20 output misses frozen target-improvement gate")
    return {
        "solver_regime": result.solver_regime,
        "phase_coefficients": result.phase_coefficients.tolist(),
        "amplitude_coefficients": result.amplitude_coefficients.tolist(),
        "warp_offsets": result.warp_offsets.tolist(),
        "kappa": result.kappa,
        "cross_subspace_singular_value": result.cross_subspace_singular_value,
        "minimum_warp_slope": result.minimum_warp_slope,
        "max_constraint_violation": result.max_constraint_violation,
        "active_constraints": list(result.active_constraints),
        "active_multipliers": list(result.active_multipliers),
        "weighted_linear_residual": result.weighted_linear_residual,
        "weighted_nonlinear_residual": result.weighted_nonlinear_residual,
        "weighted_linearization_remainder": result.weighted_linearization_remainder,
        "relative_linearization_remainder": result.relative_linearization_remainder,
        "relative_stationarity_residual": result.relative_stationarity_residual,
        "target_relative_improvement": improvement,
        "minimum_target_relative_improvement": min_improvement,
        "target_improvement_gate_enforced": enforce_improvement,
        "action_basis": action_diagnostics, "mesh": mesh,
    }


def _member_paths(output: Path, reports: dict) -> list[Path]:
    paths = [output / "candidate.json", output / "manifest.json",
             output / "common-target.npz"]
    for role in ARMS:
        paths.append(output / role / "report.json")
        if reports[role]["status"] == "completed":
            paths.extend((output / role / "sequence.npz",
                          output / role / "certificate.npz"))
    return sorted(paths, key=lambda item: item.relative_to(output).as_posix())


def _write_archive(output: Path, reports: dict, maximum: int) -> dict:
    if type(maximum) is not int or not 10240 <= maximum <= 1024 ** 3:
        raise ValueError("max_artifact_bytes must be within [10 KiB,1 GiB]")
    members = _member_paths(output, reports)
    refs = [{"path": path.relative_to(output).as_posix(),
             "sha256": digest(path), "size_bytes": path.stat().st_size}
            for path in members]
    temporary, archive = output / "artifact.tar.tmp", output / "artifact.tar"
    with tarfile.open(temporary, "w", format=tarfile.USTAR_FORMAT) as bundle:
        for path, ref in zip(members, refs):
            info = tarfile.TarInfo(ref["path"]); info.size = ref["size_bytes"]
            info.mode = 0o644; info.uid = info.gid = 0
            info.uname = info.gname = ""; info.mtime = 0
            with path.open("rb") as stream:
                bundle.addfile(info, stream)
    if temporary.stat().st_size > maximum:
        raise ValueError("C20 terminal archive exceeds frozen byte limit")
    temporary.replace(archive)
    record = {"kind": "c20-terminal-artifact-archive", "version": 1,
              "members": refs, "max_artifact_bytes": maximum,
              "archive": {"path": "artifact.tar", "sha256": digest(archive),
                          "size_bytes": archive.stat().st_size}}
    _write_json(output / "artifact-archive.json", record)
    return record


def materialize(root: Path, source_sequence: Path, source_report: Path,
                target_path: Path, target_report: Path,
                producer_plan: Path, producer_receipt: Path,
                producer_harness_plan: Path, producer_harness_report: Path,
                producer_consumption: Path,
                output: Path, *,
                uid: str, seed: int, phase_rank: int, lambda_value: float,
                min_slope: float, identifiability_floor: float,
                max_abs_warp: float, max_relative_linearization_remainder: float,
                max_gamma_noise_amplification: float,
                max_relative_stationarity_residual: float,
                min_action_residual_fraction: float,
                max_abs_gamma: float, max_amplitude_displacement: float,
                min_relative_target_improvement: float,
                min_face_area_ratio: float, max_artifact_bytes: int,
                application_stage: str | None = None) -> dict:
    root, output = Path(root).resolve(), Path(output)
    resolved = tuple(map(lambda path: Path(path).resolve(),
        (source_sequence, source_report, target_path, target_report,
         producer_plan, producer_receipt, producer_harness_plan,
         producer_harness_report, producer_consumption)))
    (source_sequence, source_report, target_path, target_report,
     producer_plan, producer_receipt, producer_harness_plan,
     producer_harness_report, producer_consumption) = resolved
    if output.exists():
        raise FileExistsError("Preserve prior C20 artifact")
    if application_stage not in (None, "d1", "d2", "confirmation"):
        raise ValueError("Invalid C20 application stage")
    validate_primary_parameters({
        "phase_rank": phase_rank, "lambda_value": lambda_value,
        "min_slope": min_slope, "identifiability_floor": identifiability_floor,
        "max_abs_warp": max_abs_warp,
        "max_relative_linearization_remainder": max_relative_linearization_remainder,
        "max_gamma_noise_amplification": max_gamma_noise_amplification,
        "max_relative_stationarity_residual": max_relative_stationarity_residual,
        "min_action_residual_fraction": min_action_residual_fraction,
        "max_abs_gamma": max_abs_gamma,
        "max_amplitude_displacement": max_amplitude_displacement,
        "min_relative_target_improvement": min_relative_target_improvement,
        "min_face_area_ratio": min_face_area_ratio,
        "max_artifact_bytes": max_artifact_bytes,
    })
    if isinstance(phase_rank, bool) or not isinstance(phase_rank, int) or not 1 <= phase_rank <= 14:
        raise ValueError("phase_rank must be in [1,14]")
    for name, value in (("max_abs_gamma", max_abs_gamma),
                        ("max_amplitude_displacement", max_amplitude_displacement),
                        ("min_relative_target_improvement", min_relative_target_improvement)):
        _positive(name, value)
    if min_relative_target_improvement >= 1:
        raise ValueError("min_relative_target_improvement must be <1")
    validate_producer_lineage(
        root, producer_plan=producer_plan, producer_receipt=producer_receipt,
        producer_harness_plan=producer_harness_plan,
        producer_harness_report=producer_harness_report,
        producer_consumption=producer_consumption,
        target_report=target_report, target_path=target_path)
    source = _arrays(source_sequence); validate_native_arrays(source)
    source_meta, target_meta = map(lambda path: json.loads(path.read_text()),
                                   (source_report, target_report))
    target = _arrays(target_path)
    if (source_meta.get("status") != "completed" or source_meta.get("uid") != uid
            or source_meta.get("seed") != seed
            or source_meta.get("sha256", {}).get("sequence.npz") != digest(source_sequence)
            or target_meta.get("kind") != "c20-decoder-consensus-target"
            or target_meta.get("status") != "completed"
            or target_meta.get("uid") != uid or target_meta.get("seed") != seed
            or target_meta.get("source_sequence_sha256") != digest(source_sequence)
            or target_meta.get("target_sha256") != digest(target_path)):
        raise ValueError("Exact same-identity C20 source/target pair required")
    vertices = source["vertices"].astype(np.float64)
    times = source["timesteps"].astype(np.float64)
    for name, expected in (("vertices", source["vertices"]), ("faces", source["faces"]),
                           ("frame_indices", source["frame_indices"]),
                           ("timesteps", source["timesteps"])):
        if name not in target or not np.array_equal(target[name], expected):
            raise ValueError("C20 target changed native identity: " + name)
    desired = np.asarray(target["target"], dtype=np.float64)
    weights = np.asarray(target["weights"], dtype=np.float64)
    if (desired.shape != vertices.shape or weights.shape != vertices.shape
            or not np.isfinite(desired).all() or not np.isfinite(weights).all()
            or np.any(weights <= 0) or not np.array_equal(desired[0], np.zeros_like(desired[0]))):
        raise ValueError("Finite anchor-preserving C20 target/weights required")
    full_phase = build_endpoint_sine_basis(times, phase_rank)
    amplitude, action_diagnostics = build_action_amplitude_basis(
        vertices, times, full_phase, weights,
        target["action_amplitude_seed"], min_action_residual_fraction)
    output.mkdir(parents=True)
    common = output / "common-target.npz"
    np.savez_compressed(common, target=desired, weights=weights,
                        phase_basis=full_phase, amplitude_basis=amplitude,
                        times=times, faces=source["faces"],
                        action_amplitude_seed=target["action_amplitude_seed"])
    reports = {}
    target_norm = float(np.linalg.norm(np.sqrt(weights.reshape(-1)) * desired.reshape(-1)))
    implementation = digest(Path(__file__))
    configurations = {
        "phase_only": (full_phase, np.zeros((0,) + vertices.shape)),
        "amplitude_only": (np.zeros((16, 0)), amplitude),
        "simple_lag": (full_phase[:, :1], np.zeros((0,) + vertices.shape)),
        CANDIDATE_ARM: (full_phase, amplitude),
    }
    for role in ARMS:
        directory = output / role; directory.mkdir()
        report = {"kind": "c20-phase-amplitude-arm", "version": 1,
                  "candidate_id": CANDIDATE_ID, "candidate_arm": role,
                  "method_id": METHOD_IDS[role], "status": "error",
                  "uid": uid, "seed": seed, "implementation_sha256": implementation,
                  "source_sequence_sha256": digest(source_sequence),
                  "source_report_sha256": digest(source_report),
                  "target_sha256": digest(target_path), "common_target_sha256": digest(common)}
        try:
            phase, basis = configurations[role]
            result = solve_phase_amplitude(
                vertices, desired, times, phase, basis, weights=weights,
                lambda_value=lambda_value, min_slope=min_slope,
                identifiability_floor=identifiability_floor,
                max_abs_warp=max_abs_warp,
                max_relative_linearization_remainder=max_relative_linearization_remainder,
                max_gamma_noise_amplification=max_gamma_noise_amplification,
                max_relative_stationarity_residual=max_relative_stationarity_residual)
            exported = result.output_vertices.astype(source["vertices"].dtype)
            if not np.array_equal(exported[0], source["vertices"][0]):
                raise PhaseAmplitudeError("float32 export changed exact native anchor")
            mesh = _mesh_certificate(vertices, exported.astype(np.float64),
                                     source["faces"], min_face_area_ratio)
            certificate = _result_certificate(
                result, action_diagnostics, mesh, target_norm,
                min_relative_target_improvement, max_abs_gamma,
                max_amplitude_displacement,
                enforce_improvement=(role == CANDIDATE_ARM))
            sequence_arrays = {name: value.copy() for name, value in source.items()}
            sequence_arrays["vertices"] = exported
            sequence_path = directory / "sequence.npz"
            np.savez_compressed(sequence_path, **sequence_arrays)
            certificate_path = directory / "certificate.npz"
            np.savez_compressed(
                certificate_path, phase_coefficients=result.phase_coefficients,
                amplitude_coefficients=result.amplitude_coefficients,
                warp_offsets=result.warp_offsets,
                amplitude_delta=result.amplitude_delta,
                linearized_repair=result.linearized_repair)
            report.update(status="completed", certificate=certificate,
                          sha256={"sequence.npz": digest(sequence_path),
                                  "certificate.npz": digest(certificate_path)})
        except Exception as error:
            report.update(exception_type=type(error).__name__[:256],
                          error=(str(error) or "<empty>")[:ERROR_MESSAGE_LIMIT],
                          traceback=traceback.format_exc()[-TRACEBACK_LIMIT:])
        report["report_digest"] = canonical_digest(report)
        _write_json(directory / "report.json", report); reports[role] = report
    candidate = {
        "kind": "c20-phase-amplitude-candidate", "version": 1,
        "candidate_id": CANDIDATE_ID, "status": "completed" if
        reports[CANDIDATE_ARM]["status"] == "completed" else "incomplete",
        "uid": uid, "seed": seed, "application_stage": application_stage,
        "native_qualified": False, "scientific_admission": False,
        "generated_unexecuted_at_authoring": True,
        "source_sequence_sha256": digest(source_sequence),
        "source_report_sha256": digest(source_report),
        "source_refs": {"sequence": file_ref(root, source_sequence),
                        "report": file_ref(root, source_report)},
        "producer_provenance_ref": file_ref(root, target_report),
        "producer_plan_ref": file_ref(root, producer_plan),
        "producer_receipt_ref": file_ref(root, producer_receipt),
        "producer_harness_plan_ref": file_ref(root, producer_harness_plan),
        "producer_harness_report_ref": file_ref(root, producer_harness_report),
        "producer_consumption_ref": file_ref(root, producer_consumption),
        "target_ref": file_ref(root, target_path),
        "common_target": {"path": "common-target.npz", "sha256": digest(common)},
        "implementation_refs": [file_ref(root, Path(__file__)),
                                file_ref(root, Path(__file__).with_name(
                                    "phase_amplitude_candidate.py"))],
        "arms": [reports[role] for role in ARMS],
        "parameters": {
            "phase_rank": phase_rank, "lambda_value": lambda_value,
            "min_slope": min_slope, "identifiability_floor": identifiability_floor,
            "max_abs_warp": max_abs_warp,
            "max_relative_linearization_remainder": max_relative_linearization_remainder,
            "max_gamma_noise_amplification": max_gamma_noise_amplification,
            "max_relative_stationarity_residual": max_relative_stationarity_residual,
            "min_action_residual_fraction": min_action_residual_fraction,
            "max_abs_gamma": max_abs_gamma,
            "max_amplitude_displacement": max_amplitude_displacement,
            "min_relative_target_improvement": min_relative_target_improvement,
            "min_face_area_ratio": min_face_area_ratio,
            "max_artifact_bytes": max_artifact_bytes,
        },
    }
    _write_json(output / "candidate.json", candidate)
    cases = [{"case_id": uid + "-" + role, "uid": uid, "case_dir": role,
              "arm_role": role} for role in ARMS if reports[role]["status"] == "completed"]
    manifest = {"kind": "c20-candidate-manifest", "version": 1,
                "candidate_id": CANDIDATE_ID, "expected_roles": list(ARMS),
                "cases": cases, "common_target": candidate["common_target"],
                "terminal_status": candidate["status"]}
    _write_json(output / "manifest.json", manifest)
    _write_archive(output, reports, max_artifact_bytes)
    return candidate


def validate_candidate_artifact(root: Path, artifact_path: Path) -> dict:
    root, artifact_path = Path(root).resolve(), Path(artifact_path).resolve()
    artifact_path.relative_to(root)
    artifact = json.loads(artifact_path.read_text())
    validate_primary_parameters(artifact.get("parameters"))
    if (artifact_path.name != "candidate.json"
            or artifact.get("kind") != "c20-phase-amplitude-candidate"
            or artifact.get("candidate_id") != CANDIDATE_ID
            or artifact.get("status") not in ("completed", "incomplete")
            or artifact.get("native_qualified") is not False
            or artifact.get("scientific_admission") is not False
            or [row.get("candidate_arm") for row in artifact.get("arms", [])] != list(ARMS)):
        raise ValueError("Terminal unqualified C20 candidate required")
    directory = artifact_path.parent
    source = _arrays(resolve_ref(root, artifact["source_refs"]["sequence"]))
    validate_native_arrays(source)
    manifest = json.loads((directory / "manifest.json").read_text())
    common = directory / "common-target.npz"
    if artifact.get("common_target") != {
            "path": "common-target.npz", "sha256": digest(common)}:
        raise ValueError("C20 common-target binding mismatch")
    treatment = next(row for row in artifact["arms"]
                     if row.get("candidate_arm") == CANDIDATE_ARM)
    expected_status = "completed" if treatment.get("status") == "completed" else "incomplete"
    if artifact.get("status") != expected_status:
        raise ValueError("C20 candidate status differs from treatment terminal state")
    implementation_hashes = {
        ref["sha256"] for ref in artifact.get("implementation_refs", [])}
    producer_report = read_json(resolve_ref(root, artifact["producer_provenance_ref"]))
    if (producer_report.get("kind") != "c20-decoder-consensus-target"
            or producer_report.get("status") != "completed"
            or producer_report.get("report_digest") != canonical_digest({
                key: value for key, value in producer_report.items()
                if key != "report_digest"})):
        raise ValueError("C20 producer provenance is not a completed digest-bound report")
    nested_producer_refs = [producer_report.get("input_freeze_ref"),
                            producer_report.get("environment_ref"),
                            producer_report.get("runtime_identity", {}).get(
                                "dependency_lock_ref"),
                            *producer_report.get("input_refs", [])]
    if any(not isinstance(ref, dict) for ref in nested_producer_refs):
        raise ValueError("C20 producer nested evidence closure missing")
    for ref in nested_producer_refs:
        resolve_ref(root, ref)
    for key in ("producer_plan_ref", "producer_receipt_ref",
                "producer_harness_plan_ref", "producer_harness_report_ref",
                "producer_consumption_ref", "target_ref"):
        resolve_ref(root, artifact[key])
    producer_runtime_refs = validate_producer_lineage(
        root,
        producer_plan=resolve_ref(root, artifact["producer_plan_ref"]),
        producer_receipt=resolve_ref(root, artifact["producer_receipt_ref"]),
        producer_harness_plan=resolve_ref(
            root, artifact["producer_harness_plan_ref"]),
        producer_harness_report=resolve_ref(
            root, artifact["producer_harness_report_ref"]),
        producer_consumption=resolve_ref(
            root, artifact["producer_consumption_ref"]),
        target_report=resolve_ref(root, artifact["producer_provenance_ref"]),
        target_path=resolve_ref(root, artifact["target_ref"]))
    reports = {}
    for role in ARMS:
        report_path = directory / role / "report.json"
        report = json.loads(report_path.read_text()); reports[role] = report
        if (report.get("candidate_arm") != role or report.get("uid") != artifact["uid"]
                or report.get("seed") != artifact["seed"]
                or report.get("method_id") != METHOD_IDS[role]
                or report.get("implementation_sha256") not in implementation_hashes
                or report.get("source_sequence_sha256") != artifact["source_sequence_sha256"]
                or report.get("source_report_sha256") != artifact["source_report_sha256"]
                or report.get("target_sha256") != artifact["target_ref"]["sha256"]
                or report.get("common_target_sha256") != artifact["common_target"]["sha256"]
                or report.get("status") not in ("completed", "error")
                or report.get("report_digest") != canonical_digest({
                    key: value for key, value in report.items() if key != "report_digest"})):
            raise ValueError("Invalid C20 terminal role: " + role)
        if report["status"] == "completed":
            arrays = _arrays(directory / role / "sequence.npz")
            validate_native_arrays(arrays)
            if set(arrays) != set(source):
                raise ValueError("C20 role changed native array inventory")
            for name in source:
                if name == "vertices":
                    if (arrays[name].shape != source[name].shape
                            or arrays[name].dtype != source[name].dtype
                            or not np.array_equal(arrays[name][0], source[name][0])):
                        raise ValueError("C20 role changed vertex identity/anchor")
                elif not np.array_equal(arrays[name], source[name]):
                    raise ValueError("C20 role changed native identity: " + name)
            if report.get("sha256") != {
                    "sequence.npz": digest(directory / role / "sequence.npz"),
                    "certificate.npz": digest(directory / role / "certificate.npz")}:
                raise ValueError("C20 role hash mismatch")
            certificate = _arrays(directory / role / "certificate.npz")
            if (set(certificate) != {"phase_coefficients", "amplitude_coefficients",
                    "warp_offsets", "amplitude_delta", "linearized_repair"}
                    or any(not np.isfinite(value).all()
                           for value in certificate.values())):
                raise ValueError("C20 role numerical certificate is incomplete")
        elif not isinstance(report.get("error"), str):
            raise ValueError("Failed C20 role lacks bounded terminal error")
    if artifact.get("arms") != [reports[role] for role in ARMS]:
        raise ValueError("C20 candidate differs from terminal role reports")
    expected_cases = [{"case_id": artifact["uid"] + "-" + role,
                       "uid": artifact["uid"], "case_dir": role,
                       "arm_role": role} for role in ARMS
                      if reports[role]["status"] == "completed"]
    if (manifest.get("expected_roles") != list(ARMS)
            or manifest.get("cases") != expected_cases
            or manifest.get("common_target") != artifact.get("common_target")
            or manifest.get("terminal_status") != artifact.get("status")):
        raise ValueError("C20 manifest mismatch")
    archive_record = directory / "artifact-archive.json"
    archive = directory / "artifact.tar"
    record = json.loads(archive_record.read_text())
    expected_members = [{"path": path.relative_to(directory).as_posix(),
                         "sha256": digest(path), "size_bytes": path.stat().st_size}
                        for path in _member_paths(directory, reports)]
    if (record.get("kind") != "c20-terminal-artifact-archive"
            or record.get("members") != expected_members
            or record.get("max_artifact_bytes") !=
               artifact.get("parameters", {}).get("max_artifact_bytes")
            or record.get("archive") != {"path": "artifact.tar",
                "sha256": digest(archive), "size_bytes": archive.stat().st_size}):
        raise ValueError("C20 bounded artifact archive mismatch")
    return {
        "candidate_id": CANDIDATE_ID, "uid": artifact["uid"],
        "seed": artifact["seed"], "status": artifact["status"],
        "artifact_archive": {"record": file_ref(root, archive_record),
                             "archive": file_ref(root, archive)},
        "producer_provenance_ref": artifact["producer_provenance_ref"],
        "implementation_refs": artifact["implementation_refs"],
        "upstream_refs": [artifact["producer_plan_ref"],
                          artifact["producer_receipt_ref"],
                          artifact["producer_harness_plan_ref"],
                          artifact["producer_harness_report_ref"],
                          artifact["producer_consumption_ref"],
                          artifact["target_ref"],
                          *artifact["source_refs"].values(), *nested_producer_refs,
                          *producer_runtime_refs],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "source-sequence", "source-report", "target",
                 "target-report", "producer-plan", "producer-receipt",
                 "producer-harness-plan", "producer-harness-report",
                 "producer-consumption", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--uid", required=True); parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--phase-rank", type=int, required=True)
    for name in ("lambda-value", "min-slope", "identifiability-floor",
                 "max-abs-warp", "max-relative-linearization-remainder",
                 "max-gamma-noise-amplification", "max-relative-stationarity-residual",
                 "min-action-residual-fraction", "max-abs-gamma",
                 "max-amplitude-displacement", "min-relative-target-improvement",
                 "min-face-area-ratio"):
        parser.add_argument("--" + name, type=float, required=True)
    parser.add_argument("--max-artifact-bytes", type=int, required=True)
    parser.add_argument("--application-stage", choices=("d1", "d2", "confirmation"))
    args = parser.parse_args(argv)
    result = materialize(
        args.root, args.source_sequence, args.source_report, args.target,
        args.target_report, args.producer_plan, args.producer_receipt,
        args.producer_harness_plan, args.producer_harness_report,
        args.producer_consumption,
        args.output, uid=args.uid, seed=args.seed,
        phase_rank=args.phase_rank, lambda_value=args.lambda_value,
        min_slope=args.min_slope, identifiability_floor=args.identifiability_floor,
        max_abs_warp=args.max_abs_warp,
        max_relative_linearization_remainder=args.max_relative_linearization_remainder,
        max_gamma_noise_amplification=args.max_gamma_noise_amplification,
        max_relative_stationarity_residual=args.max_relative_stationarity_residual,
        min_action_residual_fraction=args.min_action_residual_fraction,
        max_abs_gamma=args.max_abs_gamma,
        max_amplitude_displacement=args.max_amplitude_displacement,
        min_relative_target_improvement=args.min_relative_target_improvement,
        min_face_area_ratio=args.min_face_area_ratio,
        max_artifact_bytes=args.max_artifact_bytes,
        application_stage=args.application_stage)
    print(json.dumps({"status": result["status"], "native_qualified": False,
                      "scientific_admission": False}))
    # A terminal incomplete artifact is a successful materialization receipt:
    # the failed treatment remains in the frozen denominator downstream.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
