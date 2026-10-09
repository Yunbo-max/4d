"""Receipt-aware zero-GPU C20 retained-artifact acceptance/request plan."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from prepare_control_scoring_checks import acceptance_sources
from prepare_c06_native_acceptance import official_source_names
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


def _candidate_execution(root: Path, plan_path: Path, receipt_path: Path, native):
    root = Path(root).resolve(); plan_path = _physical(root, plan_path)
    receipt_path = _physical(root, receipt_path)
    plan, receipt = map(lambda path: json.loads(path.read_text()),
                        (plan_path, receipt_path))
    native.validate_plan(root, plan)
    run_root = (root / plan.get("output_root", "") / plan.get("run_id", "")).resolve()
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
        raise ValueError("Exact completed C20 artifact receipt required")
    job, attempt = jobs[0], attempts[0]
    if (job.get("trial_id") != "c20-phase-amplitude-artifact"
            or attempt.get("trial_id") != job["trial_id"]
            or attempt.get("status") != "completed" or attempt.get("exit_code") != 0
            or attempt.get("retry_index") != 0
            or any(attempt.get(name) != job.get(name)
                   for name in ("input_refs", "code_refs", "seed", "group", "arm_role"))):
        raise ValueError("C20 artifact job/attempt identity mismatch")
    attempt_path = Path(attempt.get("attempt_path", ""))
    if attempt_path.is_absolute() or ".." in attempt_path.parts:
        raise ValueError("Canonical C20 artifact attempt path required")
    workspace = (root / attempt_path / "workspace").resolve(); workspace.relative_to(root)
    if (attempt_path.parent != run_root.relative_to(root)
            or attempt.get("attempt_id") != attempt_path.name or not workspace.is_dir()):
        raise ValueError("Retained C20 artifact workspace missing")
    replacements = {}
    for ref in job["input_refs"] + job["code_refs"]:
        source = _physical(root, ref["path"]); staged = _physical(root, workspace / ref["path"])
        if candidate.digest(source) != ref["sha256"] or candidate.digest(staged) != ref["sha256"]:
            raise ValueError("C20 artifact staged closure differs")
        replacements[str(source)] = str(staged)
    expected_command = []
    for argument in job["command"]:
        for source, staged in sorted(replacements.items(), key=lambda pair: -len(pair[0])):
            argument = argument.replace(source, staged)
        expected_command.append(argument)
    expected_cwd = workspace if job["cwd"] == "." else workspace / job["cwd"]
    if attempt.get("command") != expected_command or Path(attempt.get("cwd", "")) != expected_cwd:
        raise ValueError("C20 artifact staged command/cwd mismatch")
    expected_outputs = [candidate.file_ref(root, _physical(root, workspace / relative))
                        for relative in job["output_paths"]]
    if attempt.get("output_refs") != expected_outputs:
        raise ValueError("C20 artifact output receipt inventory mismatch")
    try:
        output_arg = job["command"][job["command"].index("--output") + 1]
    except (ValueError, IndexError) as error:
        raise ValueError("C20 artifact command lacks output") from error
    output = Path(output_arg)
    if output.is_absolute():
        output = workspace / output.relative_to(root)
    else:
        output = expected_cwd / output
    output = output.resolve(); output.relative_to(workspace)
    artifact = _physical(root, output / "candidate.json")
    candidate.validate_candidate_execution(
        root, plan_path=plan_path, receipt_path=receipt_path,
        artifact_path=artifact)
    candidate.validate_candidate_artifact(root, artifact)
    return plan_path, receipt_path, workspace, artifact


def build_plans(root: Path, *, skill_dir: Path, candidate_plan: Path,
                candidate_receipt: Path, artifact_candidate: Path,
                plan_dir: Path, run_id: str, wall_seconds: int, ram_mib: int,
                freeze: Path | None = None, ground_truth: Path | None = None,
                population: Path | None = None, dataset_admission: Path | None = None,
                dataset_semantics: Path | None = None, repo_root: Path | None = None,
                timeout_seconds: int | None = None):
    root, skill_dir, plan_dir = map(lambda item: Path(item).resolve(),
                                    (root, skill_dir, plan_dir))
    plan_dir.relative_to(root)
    if plan_dir.exists():
        raise FileExistsError("Preserve prior C20 acceptance plan")
    if type(wall_seconds) is not int or not 1 <= wall_seconds <= 26940:
        raise ValueError("Bounded C20 acceptance wall budget required")
    if type(ram_mib) is not int or ram_mib < 1:
        raise ValueError("Positive C20 acceptance RAM required")
    scripts = skill_dir / "scripts"
    if not (scripts / "run_harness.py").is_file():
        raise ValueError("Complete installed research-autopilot required")
    sys.path.insert(0, str(scripts))
    import run_experiments as native
    import run_harness as harness
    plan_path, receipt_path, workspace, derived_artifact = _candidate_execution(
        root, candidate_plan, candidate_receipt, native)
    artifact_candidate = Path(artifact_candidate)
    artifact_candidate = (artifact_candidate if artifact_candidate.is_absolute()
                          else root / artifact_candidate).resolve()
    if artifact_candidate != derived_artifact:
        raise ValueError("C20 artifact path differs from exact retained attempt")
    verified = candidate.validate_candidate_artifact(root, artifact_candidate)
    artifact = json.loads(artifact_candidate.read_text())
    candidate_plan_value = json.loads(plan_path.read_text())
    candidate_receipt_value = json.loads(receipt_path.read_text())
    candidate_job = candidate_plan_value["jobs"][0]
    candidate_attempt = candidate_receipt_value["attempts"][0]
    candidate_run_root = (root / candidate_plan_value["output_root"]
                          / candidate_plan_value["run_id"]).resolve()
    candidate_attempt_root = (root / candidate_attempt["attempt_path"]).resolve()
    candidate_workspace = candidate_attempt_root / "workspace"
    directory = artifact_candidate.parent
    paths = [plan_path, receipt_path, candidate_run_root / "plan.json",
             candidate_attempt_root / "attempt.json",
             *(candidate.resolve_ref(root, ref) for ref in
               candidate_job["input_refs"] + candidate_job["code_refs"]),
             *(candidate_workspace / ref["path"] for ref in
               candidate_job["input_refs"] + candidate_job["code_refs"]),
             artifact_candidate,
             directory / "manifest.json", directory / "common-target.npz",
             candidate.resolve_ref(root, verified["artifact_archive"]["record"]),
             candidate.resolve_ref(root, verified["artifact_archive"]["archive"]),
             candidate.resolve_ref(root, verified["producer_provenance_ref"]),
             *(candidate.resolve_ref(root, ref) for ref in verified["implementation_refs"]),
             *(candidate.resolve_ref(root, ref) for ref in verified["upstream_refs"])]
    for role in candidate.ARMS:
        report = json.loads((directory / role / "report.json").read_text())
        paths.append(directory / role / "report.json")
        if report["status"] == "completed":
            paths.extend((directory / role / "sequence.npz",
                          directory / role / "certificate.npz"))
    inputs = [candidate.file_ref(root, path) for path in paths]
    request_paths = (ground_truth, population, dataset_admission,
                     dataset_semantics, repo_root)
    wants_scoring = any(value is not None for value in (*request_paths, timeout_seconds))
    if wants_scoring and (freeze is None or any(value is None for value in request_paths)
                          or type(timeout_seconds) is not int or timeout_seconds < 1):
        raise ValueError("C20 scoring preparation requires freeze and all scoring inputs")
    if freeze is not None:
        freeze = _physical(root, freeze); inputs.append(candidate.file_ref(root, freeze))
    scoring_paths, official_paths = [], []
    if wants_scoring:
        scoring_paths = [_physical(root, item) for item in request_paths[:4]]
        repository = Path(repo_root)
        repository = repository if repository.is_absolute() else root / repository
        official_paths = [_physical(root, repository / "actionbench" / name)
                          for name in official_source_names(root)]
        inputs.extend(candidate.file_ref(root, path) for path in [*scoring_paths, *official_paths])
    unique = {}
    for ref in inputs:
        if ref["path"] in unique and unique[ref["path"]] != ref:
            raise ValueError("Conflicting C20 acceptance input identity")
        unique[ref["path"]] = ref
    code = [candidate.file_ref(root, path) for path in acceptance_sources(root)]
    code.extend(candidate.file_ref(root, root / "actionmesh/research_math" / name)
                for name in ("phase_amplitude_candidate.py", "phase_amplitude_artifacts.py",
                              "c20_native_comparison.py", "c20_native_scoring.py"))
    program = (
        "import json,sys\nfrom pathlib import Path\n"
        "from research_math import phase_amplitude_artifacts as candidate\n"
        "root=Path(candidate.__file__).resolve().parents[2]\n"
        "plan=Path(sys.argv[1]); receipt=Path(sys.argv[2]); artifact=Path(sys.argv[3])\n"
        "candidate.validate_candidate_execution(root,plan_path=plan,receipt_path=receipt,artifact_path=artifact)\n"
        "verified=candidate.validate_candidate_artifact(root,artifact)\n"
        "print(json.dumps({'mode':'native_method_acceptance','status':verified['status'],"
        "'local_method_verified':False}))\n")
    command = [sys.executable, "-c", program, str(plan_path), str(receipt_path),
               str(artifact_candidate)]
    outputs = []
    if freeze is not None:
        program += (
            "from research_math import c20_native_comparison as comparison\n"
            "request=comparison.make_request(root,freeze_path=Path(sys.argv[4]))\n"
            "comparison_path=Path('c20-comparison-request.json')\n"
            "comparison_path.write_text(json.dumps(request,indent=2,allow_nan=False)+'\\n')\n")
        command.append(str(freeze)); outputs.append("actionmesh/c20-comparison-request.json")
    if wants_scoring:
        program += (
            "from research_math import c20_native_scoring as scoring\n"
            "request=scoring.make_scoring_request(root,comparison_path=comparison_path,"
            "ground_truth=Path(sys.argv[5]),population=Path(sys.argv[6]),"
            "dataset_admission=Path(sys.argv[7]),dataset_semantics=Path(sys.argv[8]),"
            "repo_root=Path(sys.argv[9]).parent.parent,timeout_seconds=int(sys.argv[10]))\n"
            "Path('c20-scoring-request.json').write_text(json.dumps(request,indent=2,allow_nan=False)+'\\n')\n")
        command.extend(str(path) for path in scoring_paths)
        command.extend((str(official_paths[0]), str(timeout_seconds)))
        outputs.append("actionmesh/c20-scoring-request.json")
    command[2] = program
    mode = "request_preparation" if freeze is not None else "native_method_acceptance"
    plan = native.make_plan(root, run_id=run_id, jobs=[{
        "trial_id": "c20-retained-native-acceptance", "command": command,
        "cwd": "actionmesh", "input_refs": list(unique.values()),
        "code_refs": code, "output_paths": outputs, "seed": 42,
        "group": "engineering", "arm_role": mode}],
        provenance={"git_revision": "Exact current C20 acceptance code_refs",
            "model_revision": "none; no generation or scoring",
            "data_revision": candidate.digest(artifact_candidate),
            "environment_digest": hashlib.sha256(json.dumps({
                "python": sys.version, "skill": str(skill_dir)},
                sort_keys=True).encode()).hexdigest()},
        limits={"max_attempts": 1, "max_development_trials": 1,
            "max_confirmation_trials": 0, "max_retries_per_trial": 0,
            "wall_time_seconds": wall_seconds,
            "attempt_timeout_seconds": wall_seconds})
    plan_dir.mkdir(parents=True)
    native_path = plan_dir / "native.json"
    native_path.write_text(json.dumps(plan, indent=2) + "\n")
    outer = harness.make_plan(root, batch_id=run_id, tasks=[{
        "task_id": "c20-retained-native-acceptance", "idea_id": candidate.CANDIDATE_ID,
        "depends_on": [], "priority": 1, "plan_ref": candidate.file_ref(root, native_path),
        "resources": {"cpu_cores": 1, "ram_mib": ram_mib, "gpu_count": 0,
            "gpu_peak_mib": None, "allow_gpu_share": False,
            "memory_profile_ref": None, "exclusive_keys": []}}],
        limits={"total_wall_seconds": wall_seconds + 60,
            "window_seconds": wall_seconds + 60, "max_parallel_tasks": 1,
            "cpu_cores": 1, "ram_mib": ram_mib, "max_gpu_task_seconds": 0})
    (plan_dir / "harness.json").write_text(json.dumps(outer, indent=2) + "\n")
    return plan, outer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "skill-dir", "candidate-plan", "candidate-receipt",
                 "artifact-candidate", "plan-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--wall-seconds", type=int, required=True)
    parser.add_argument("--ram-mib", type=int, required=True)
    parser.add_argument("--freeze", type=Path)
    for name in ("ground-truth", "population", "dataset-admission",
                 "dataset-semantics", "repo-root"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--timeout-seconds", type=int)
    args = parser.parse_args(argv)
    _, outer = build_plans(args.root, skill_dir=args.skill_dir,
        candidate_plan=args.candidate_plan, candidate_receipt=args.candidate_receipt,
        artifact_candidate=args.artifact_candidate, plan_dir=args.plan_dir,
        run_id=args.run_id, wall_seconds=args.wall_seconds, ram_mib=args.ram_mib,
        freeze=args.freeze, ground_truth=args.ground_truth,
        population=args.population, dataset_admission=args.dataset_admission,
        dataset_semantics=args.dataset_semantics, repo_root=args.repo_root,
        timeout_seconds=args.timeout_seconds)
    print(json.dumps({"plan": str(args.plan_dir.resolve() / "harness.json"),
        "approved_plan_digest": outer["plan_digest"], "execution_started": False,
        "source_delivery_status": "generated_unexecuted", "gpu_count": 0,
        "local_method_verified": False, "native_qualified": False}))


if __name__ == "__main__":
    main()
