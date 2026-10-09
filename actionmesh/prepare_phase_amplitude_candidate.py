"""Build C20's receipt-bound, zero-GPU four-arm artifact plan; never run it."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from research_math import c20_consensus_target as target_module
from research_math import phase_amplitude_artifacts as candidate


def _physical(root: Path, value) -> Path:
    root = Path(root).resolve(); path = Path(value)
    path = path if path.is_absolute() else root / path
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise ValueError("Physical nonsymlinked C20 path required")
    path = path.resolve(); path.relative_to(root)
    if not path.is_file():
        raise ValueError("Missing C20 input: " + str(path))
    return path


def _producer_execution(root: Path, plan_path: Path, receipt_path: Path, native):
    root = Path(root).resolve(); plan_path = _physical(root, plan_path)
    receipt_path = _physical(root, receipt_path)
    plan, receipt = map(lambda path: json.loads(path.read_text()),
                        (plan_path, receipt_path))
    native.validate_plan(root, plan)
    run_root = (root / plan.get("output_root", "") / plan.get("run_id", "")).resolve()
    run_root.relative_to(root)
    jobs, attempts = plan.get("jobs"), receipt.get("attempts")
    if (receipt_path != run_root / "receipt.json"
            or receipt.get("status") != "completed"
            or receipt.get("run_id") != plan.get("run_id")
            or receipt.get("plan_digest") != plan.get("plan_digest")
            or receipt.get("purpose") != plan.get("purpose")
            or receipt.get("evidence_mode") != plan.get("evidence_mode")
            or receipt.get("provenance") != plan.get("provenance")
            or not isinstance(jobs, list) or len(jobs) != 1
            or not isinstance(attempts, list) or len(attempts) != 1
            or plan.get("limits", {}).get("max_attempts") != 1
            or plan.get("limits", {}).get("max_retries_per_trial") != 0):
        raise ValueError("Exact completed one-attempt C20 producer receipt required")
    job, attempt = jobs[0], attempts[0]
    if (job.get("trial_id") != "c20-decoder-consensus-target"
            or attempt.get("trial_id") != job["trial_id"]
            or attempt.get("status") != "completed" or attempt.get("exit_code") != 0
            or attempt.get("retry_index") != 0
            or any(attempt.get(name) != job.get(name)
                   for name in ("input_refs", "code_refs", "seed", "group", "arm_role"))):
        raise ValueError("C20 producer job/attempt identity mismatch")
    attempt_path = Path(attempt.get("attempt_path", ""))
    if attempt_path.is_absolute() or ".." in attempt_path.parts:
        raise ValueError("Canonical C20 attempt path required")
    workspace = (root / attempt_path / "workspace").resolve(); workspace.relative_to(root)
    if (attempt_path.parent != run_root.relative_to(root)
            or attempt.get("attempt_id") != attempt_path.name or not workspace.is_dir()):
        raise ValueError("Retained C20 producer workspace missing")
    replacements = {}
    for ref in job["input_refs"] + job["code_refs"]:
        source = _physical(root, ref["path"])
        staged = _physical(root, workspace / ref["path"])
        if target_module.digest(source) != ref["sha256"] or target_module.digest(staged) != ref["sha256"]:
            raise ValueError("C20 producer staged closure differs")
        replacements[str(source)] = str(staged)
    expected_command = []
    for argument in job["command"]:
        for source, staged in sorted(replacements.items(), key=lambda pair: -len(pair[0])):
            argument = argument.replace(source, staged)
        expected_command.append(argument)
    expected_cwd = workspace if job["cwd"] == "." else workspace / job["cwd"]
    if attempt.get("command") != expected_command or Path(attempt.get("cwd", "")) != expected_cwd:
        raise ValueError("C20 producer staged command/cwd mismatch")
    expected_outputs = [candidate.file_ref(root, _physical(root, workspace / relative))
                        for relative in job["output_paths"]]
    if attempt.get("output_refs") != expected_outputs:
        raise ValueError("C20 producer output receipt inventory mismatch")
    output = workspace / "actionmesh/c20-consensus-target-output"
    report = _physical(root, output / "producer-report.json")
    manifest = _physical(root, output / "producer-manifest.json")
    record, inventory = map(lambda path: json.loads(path.read_text()), (report, manifest))
    target = _physical(root, output / "consensus-target.npz")
    if (record.get("kind") != target_module.TARGET_KIND or record.get("status") != "completed"
            or record.get("target_sha256") != target_module.digest(target)
            or inventory.get("status") != "completed"
            or inventory.get("report", {}).get("sha256") != target_module.digest(report)
            or inventory.get("conditional_outputs") != [{
                "path": "consensus-target.npz", "sha256": target_module.digest(target)}]
            or inventory.get("manifest_digest") != target_module.canonical_digest({
                key: value for key, value in inventory.items() if key != "manifest_digest"})):
        raise ValueError("Completed exact C20 target artifact required")
    return plan_path, receipt_path, workspace, target, report, manifest, record


def _producer_supervision(root: Path, *, native_plan: Path,
                          native_receipt: Path,
                          harness_plan: Path, harness_report: Path,
                          consumption: Path, harness) -> tuple[Path, ...]:
    """Require proof that the target ran through the STOP-aware outer launcher."""
    root = Path(root).resolve()
    native_plan = _physical(root, native_plan)
    harness_plan = _physical(root, harness_plan)
    harness_report = _physical(root, harness_report)
    consumption = _physical(root, consumption)
    native_receipt = _physical(root, native_receipt)
    native_value, receipt, harness_value, report, consumed = map(
        lambda path: json.loads(path.read_text()),
        (native_plan, native_receipt, harness_plan, harness_report, consumption))
    harness.validate_plan(root, harness_value)
    batch_root = (root / harness_value.get("output_root", "") /
                  harness_value.get("batch_id", "")).resolve()
    run_root = (root / native_value.get("output_root", "") /
                native_value.get("run_id", "")).resolve()
    batch_root.relative_to(root); run_root.relative_to(root)
    canonical_native_plan = _physical(root, run_root / "plan.json")
    canonical_harness_plan = _physical(root, batch_root / "plan.json")
    tasks = harness_value.get("tasks")
    attempts = receipt.get("attempts")
    if not isinstance(attempts, list) or len(attempts) != 1:
        raise ValueError("Exact C20 native attempt required for harness closure")
    attempt_root = (root / attempts[0].get("attempt_path", "")).resolve()
    attempt_record = _physical(root, attempt_root / "attempt.json")
    task_root = batch_root / "tasks/c20-decoder-consensus-target"
    state_path = _physical(root, batch_root / "state.json")
    task_path = _physical(root, task_root / "task.json")
    result_path = _physical(root, task_root / "result.json")
    state, task_record, task_result = map(
        lambda path: json.loads(path.read_text()),
        (state_path, task_path, result_path))
    report_tasks = report.get("tasks")
    reported_task = (report_tasks.get("c20-decoder-consensus-target", {})
                     if isinstance(report_tasks, dict) else {})
    expected_outputs = attempts[0].get("output_refs", [])
    if (harness_report != batch_root / "report.json"
            or json.loads(canonical_harness_plan.read_text()) != harness_value
            or json.loads(canonical_native_plan.read_text()) != native_value
            or report.get("status") != "completed"
            or report.get("plan_digest") != harness_value.get("plan_digest")
            or not isinstance(tasks, list) or len(tasks) != 1
            or tasks[0].get("task_id") != "c20-decoder-consensus-target"
            or tasks[0].get("plan_ref") != candidate.file_ref(root, native_plan)
            or task_record != tasks[0]
            or json.loads(attempt_record.read_text()) != attempts[0]
            or any(reported_task.get(key) != value
                   for key, value in task_result.items())
            or state.get("plan_digest") != harness_value.get("plan_digest")
            or state.get("status") != "completed"
            or state.get("tasks") != report.get("tasks")
            or reported_task.get("status") != "completed"
            or reported_task.get("receipt_ref") !=
               candidate.file_ref(root, native_receipt)
            or sorted(reported_task.get("output_refs", []), key=lambda ref: ref["path"]) !=
               sorted(expected_outputs, key=lambda ref: ref["path"])):
        raise ValueError("Completed exact C20 target harness evidence required")
    auth_ref = consumed.get("authorization_ref")
    auth_path = target_module.resolve_ref(root, auth_ref)
    expected_consumption = (root / "inputs/c20-target/authorization-consumption" /
                            (target_module.digest(auth_path) + ".json")).resolve()
    if consumption != expected_consumption:
        raise ValueError("Canonical C20 target authorization consumption required")
    auth = target_module.read_json(auth_path)
    try:
        from datetime import datetime
        issued = datetime.fromisoformat(auth["issued_at"].replace("Z", "+00:00"))
        expires = datetime.fromisoformat(auth["expires_at"].replace("Z", "+00:00"))
        consumed_at = datetime.fromisoformat(
            consumed["consumed_at"].replace("Z", "+00:00"))
    except (KeyError, AttributeError, ValueError) as error:
        raise ValueError("C20 target consumption window is invalid") from error
    expected = {
        "kind": "c20-target-authorization-consumption", "version": 1,
        "state": "consumed_for_exact_plan", "authorization_ref": auth_ref,
        "run_id": auth.get("run_id"), "gpu_uuid": auth.get("gpu_uuid"),
        "native_plan_ref": candidate.file_ref(root, native_plan),
        "native_plan_digest": native_value.get("plan_digest"),
        "harness_plan_ref": candidate.file_ref(root, harness_plan),
        "harness_plan_digest": harness_value.get("plan_digest"),
        "no_scientific_retry": True,
    }
    jobs = native_value.get("jobs")
    if (any(consumed.get(key) != value for key, value in expected.items())
            or consumed.get("consumption_digest") != target_module.canonical_digest({
                key: value for key, value in consumed.items()
                if key != "consumption_digest"})
            or issued.tzinfo is None or expires.tzinfo is None
            or consumed_at.tzinfo is None or not issued <= consumed_at < expires
            or not isinstance(jobs, list) or len(jobs) != 1
            or auth_ref not in jobs[0].get("input_refs", [])):
        raise ValueError("C20 target authorization consumption identity differs")
    return (harness_plan, harness_report, consumption, canonical_native_plan,
            canonical_harness_plan, state_path, task_path, result_path,
            attempt_record, auth_path)


def build_plans(root: Path, *, skill_dir: Path, producer_plan: Path,
                producer_receipt: Path, producer_harness_plan: Path,
                producer_harness_report: Path, producer_consumption: Path,
                source_sequence: Path,
                source_report: Path, output: Path, plan_dir: Path,
                run_id: str, uid: str, seed: int, phase_rank: int,
                lambda_value: float, min_slope: float,
                identifiability_floor: float, max_abs_warp: float,
                max_relative_linearization_remainder: float,
                max_gamma_noise_amplification: float,
                max_relative_stationarity_residual: float,
                min_action_residual_fraction: float, max_abs_gamma: float,
                max_amplitude_displacement: float,
                min_relative_target_improvement: float,
                min_face_area_ratio: float, max_artifact_bytes: int,
                wall_seconds: int, ram_mib: int,
                application_stage: str | None = None):
    root, skill_dir, output, plan_dir = map(lambda item: Path(item).resolve(),
                                             (root, skill_dir, output, plan_dir))
    output.relative_to(root); plan_dir.relative_to(root)
    actionmesh_root = root / "actionmesh"
    output_relative_to_cwd = output.relative_to(actionmesh_root)
    if output.exists() or plan_dir.exists():
        raise FileExistsError("Preserve prior C20 artifact/plan")
    if type(wall_seconds) is not int or not 1 <= wall_seconds <= 26940:
        raise ValueError("Bounded C20 CPU wall budget required")
    if type(ram_mib) is not int or ram_mib < 1:
        raise ValueError("Positive C20 RAM budget required")
    candidate.validate_primary_parameters({
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
    scripts = skill_dir / "scripts"
    if not (scripts / "run_harness.py").is_file():
        raise ValueError("Complete installed research-autopilot required")
    sys.path.insert(0, str(scripts))
    import run_experiments as native
    import run_harness as harness
    plan_path, receipt_path, workspace, target, target_report, manifest, record = \
        _producer_execution(root, producer_plan, producer_receipt, native)
    (harness_plan_path, harness_report_path, consumption_path,
     canonical_native_plan, canonical_harness_plan, harness_state,
     harness_task, harness_result, native_attempt_record,
     authorization_path) = \
        _producer_supervision(
            root, native_plan=plan_path, native_receipt=receipt_path,
            harness_plan=producer_harness_plan,
            harness_report=producer_harness_report,
            consumption=producer_consumption, harness=harness)
    source_sequence, source_report = map(lambda path: _physical(root, path),
                                         (source_sequence, source_report))
    if (record.get("uid") != uid or record.get("seed") != seed
            or record.get("source_sequence_sha256") != target_module.digest(source_sequence)
            or record.get("source_report_sha256") != target_module.digest(source_report)):
        raise ValueError("C20 target producer and B0 identities differ")
    inputs = [candidate.file_ref(root, path) for path in
              (plan_path, receipt_path, harness_plan_path, harness_report_path,
               consumption_path, canonical_native_plan, canonical_harness_plan,
               harness_state, harness_task, harness_result,
               native_attempt_record, authorization_path,
               target, target_report, manifest,
               source_sequence, source_report)]
    nested_refs = [record.get("input_freeze_ref"),
                   record.get("environment_ref"), *record.get("input_refs", [])]
    if any(not isinstance(ref, dict) for ref in nested_refs):
        raise ValueError("C20 producer report lacks complete nested input closure")
    for ref in nested_refs:
        path = target_module.resolve_ref(root, ref)
        inputs.append(candidate.file_ref(root, path))
    producer_plan_value = target_module.read_json(plan_path)
    producer_jobs = producer_plan_value.get("jobs")
    if not isinstance(producer_jobs, list) or len(producer_jobs) != 1:
        raise ValueError("Exact C20 producer job closure required")
    for ref in (producer_jobs[0].get("input_refs", []) +
                producer_jobs[0].get("code_refs", [])):
        inputs.append(candidate.file_ref(root, target_module.resolve_ref(root, ref)))
    environment_value = target_module.read_json(
        target_module.resolve_ref(root, record["environment_ref"]))
    dependency_refs = environment_value.get("dependency_lock_refs")
    if not isinstance(dependency_refs, list) or len(dependency_refs) != 1:
        raise ValueError("C20 producer environment lacks dependency closure")
    inputs.append(candidate.file_ref(
        root, target_module.resolve_ref(root, dependency_refs[0])))
    unique = {}
    for ref in inputs:
        if ref["path"] in unique and unique[ref["path"]] != ref:
            raise ValueError("Conflicting C20 candidate input identity")
        unique[ref["path"]] = ref
    code_paths = [root / "actionmesh/research_math" / name for name in (
        "__init__.py", "phase_amplitude_candidate.py", "phase_amplitude_artifacts.py")]
    code_paths.append(root / "actionmesh/prepare_phase_amplitude_candidate.py")
    command = [sys.executable, "-m", "research_math.phase_amplitude_artifacts",
        "--root", "..", "--source-sequence", str(source_sequence),
        "--source-report", str(source_report), "--target", str(target),
        "--target-report", str(target_report),
        "--producer-plan", str(plan_path), "--producer-receipt", str(receipt_path),
        "--producer-harness-plan", str(harness_plan_path),
        "--producer-harness-report", str(harness_report_path),
        "--producer-consumption", str(consumption_path),
        "--output", str(output_relative_to_cwd),
        "--uid", uid, "--seed", str(seed), "--phase-rank", str(phase_rank),
        "--lambda-value", str(lambda_value), "--min-slope", str(min_slope),
        "--identifiability-floor", str(identifiability_floor),
        "--max-abs-warp", str(max_abs_warp),
        "--max-relative-linearization-remainder", str(max_relative_linearization_remainder),
        "--max-gamma-noise-amplification", str(max_gamma_noise_amplification),
        "--max-relative-stationarity-residual", str(max_relative_stationarity_residual),
        "--min-action-residual-fraction", str(min_action_residual_fraction),
        "--max-abs-gamma", str(max_abs_gamma),
        "--max-amplitude-displacement", str(max_amplitude_displacement),
        "--min-relative-target-improvement", str(min_relative_target_improvement),
        "--min-face-area-ratio", str(min_face_area_ratio),
        "--max-artifact-bytes", str(max_artifact_bytes)]
    if application_stage is not None:
        command += ["--application-stage", application_stage]
    relative_output = output.relative_to(root).as_posix()
    always = ["candidate.json", "manifest.json", "common-target.npz",
              "artifact.tar", "artifact-archive.json",
              *(role + "/report.json" for role in candidate.ARMS)]
    plan = native.make_plan(root, run_id=run_id, purpose="engineering",
        evidence_mode="developmental", jobs=[{
            "trial_id": "c20-phase-amplitude-artifact", "command": command,
            "cwd": "actionmesh", "input_refs": list(unique.values()),
            "code_refs": [candidate.file_ref(root, path) for path in code_paths],
            "output_paths": [relative_output + "/" + name for name in always],
            "seed": seed, "group": "c20-artifact",
            "arm_role": "four-role-phase-amplitude-materialization"}],
        provenance={"git_revision": "exact code refs; no clean-tree claim",
            "model_revision": "none; target producer receipt is frozen input",
            "data_revision": target_module.digest(target),
            "environment_digest": target_module.canonical_digest({
                "python": sys.version, "python_executable": sys.executable,
                "numpy": candidate.np.__version__,
                "parameter_profile": candidate.PRIMARY_PARAMETER_PROFILE_ID})},
        limits={"max_attempts": 1, "max_development_trials": 1,
            "max_confirmation_trials": 0, "max_retries_per_trial": 0,
            "wall_time_seconds": wall_seconds,
            "attempt_timeout_seconds": wall_seconds})
    plan_dir.mkdir(parents=True)
    native_path = plan_dir / "native.json"
    native_path.write_text(json.dumps(plan, indent=2) + "\n")
    outer = harness.make_plan(root, batch_id=run_id, tasks=[{
        "task_id": "c20-phase-amplitude-artifact", "idea_id": candidate.CANDIDATE_ID,
        "depends_on": [], "priority": 1, "plan_ref": candidate.file_ref(root, native_path),
        "resources": {"cpu_cores": 1, "ram_mib": ram_mib, "gpu_count": 0,
            "gpu_peak_mib": None, "allow_gpu_share": False,
            "memory_profile_ref": None, "exclusive_keys": ["c20-artifact"]}}],
        limits={"total_wall_seconds": wall_seconds + 60,
            "window_seconds": wall_seconds + 60, "max_parallel_tasks": 1,
            "cpu_cores": 1, "ram_mib": ram_mib, "max_gpu_task_seconds": 0})
    (plan_dir / "harness.json").write_text(json.dumps(outer, indent=2) + "\n")
    return plan, outer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "skill-dir", "producer-plan", "producer-receipt",
                 "producer-harness-plan", "producer-harness-report",
                 "producer-consumption",
                 "source-sequence", "source-report", "output", "plan-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--run-id", required=True); parser.add_argument("--uid", required=True)
    parser.add_argument("--seed", type=int, required=True); parser.add_argument("--phase-rank", type=int, required=True)
    for name in ("lambda-value", "min-slope", "identifiability-floor", "max-abs-warp",
                 "max-relative-linearization-remainder", "max-gamma-noise-amplification",
                 "max-relative-stationarity-residual", "min-action-residual-fraction",
                 "max-abs-gamma", "max-amplitude-displacement",
                 "min-relative-target-improvement", "min-face-area-ratio"):
        parser.add_argument("--" + name, type=float, required=True)
    parser.add_argument("--max-artifact-bytes", type=int, required=True)
    parser.add_argument("--wall-seconds", type=int, required=True)
    parser.add_argument("--ram-mib", type=int, required=True)
    parser.add_argument("--application-stage", choices=("d1", "d2", "confirmation"))
    args = parser.parse_args(argv)
    _, outer = build_plans(args.root, skill_dir=args.skill_dir,
        producer_plan=args.producer_plan, producer_receipt=args.producer_receipt,
        producer_harness_plan=args.producer_harness_plan,
        producer_harness_report=args.producer_harness_report,
        producer_consumption=args.producer_consumption,
        source_sequence=args.source_sequence, source_report=args.source_report,
        output=args.output, plan_dir=args.plan_dir, run_id=args.run_id,
        uid=args.uid, seed=args.seed, phase_rank=args.phase_rank,
        lambda_value=args.lambda_value, min_slope=args.min_slope,
        identifiability_floor=args.identifiability_floor,
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
        wall_seconds=args.wall_seconds, ram_mib=args.ram_mib,
        application_stage=args.application_stage)
    print(json.dumps({"plan": str(args.plan_dir.resolve() / "harness.json"),
        "approved_plan_digest": outer["plan_digest"], "execution_started": False,
        "source_delivery_status": "generated_unexecuted", "gpu_count": 0,
        "native_qualified": False, "scientific_admission": False}))


if __name__ == "__main__":
    main()
