"""Plan-only, zero-GPU acceptance and request preparation for retained C05 artifacts.

This builder hashes the complete retained artifact and recursively pins its
source evidence.  The emitted Local task re-runs the artifact validator and may
optionally create prospective comparison/scoring *requests*.  It never invokes
ActionMesh, a scorer, or a GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from prepare_control_scoring_checks import acceptance_sources
from prepare_c05_mode_bank import _validate_weights_manifest_schema
from prepare_c06_native_acceptance import (
    _checked_ref, _digest, _load, _nested_refs, _physical, _ref,
    official_source_names,
)


CANDIDATE_ID = "4d-math-20261006-c05"
ROLES = (
    "localized_mean", "temperature_matched_mean", "surface_projected_mean",
    "independent_top1", "joint_spatial_labels",
)
SCOPE = {
    "native_scientific_qualification": False,
    "scientific_effect_qualification": False,
    "local_method_verified": False,
    "dispatch_ready": False,
}


def _iter_refs(value):
    """Yield exact file-reference objects, including byte-sized C05 refs."""
    if isinstance(value, dict):
        keys = set(value)
        if {"path", "sha256"}.issubset(keys):
            yield value
        else:
            for child in value.values():
                yield from _iter_refs(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_refs(child)


def _candidate_directory(root, artifact_candidate) -> tuple[Path, Path, dict]:
    candidate_path = _physical(root, artifact_candidate)
    if candidate_path.name != "candidate.json":
        raise ValueError("Retained C05 candidate.json path required")
    candidate = _load(candidate_path)
    if (candidate.get("kind") != "c05-spatial-mode-candidate"
            or candidate.get("version") != 1
            or candidate.get("candidate_id") != CANDIDATE_ID
            or candidate.get("status") not in
                ("completed_unqualified", "incomplete_natural_gate")
            or any(candidate.get(name) is not value
                   for name, value in SCOPE.items())):
        raise ValueError("Terminal unqualified C05 candidate required")
    return candidate_path.parent, candidate_path, candidate


def _resolve_nested_ref(root: Path, document: Path, ref: dict, *,
                        evidence_root: Path, producer_root: Path) -> Path:
    if (not isinstance(ref, dict) or not {"path", "sha256"}.issubset(ref)
            or not isinstance(ref["path"], str)):
        raise ValueError("Exact path/hash reference required")
    relative = Path(ref["path"])
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != ref["path"]:
        raise ValueError("Canonical relative C05 evidence path required")
    namespace = producer_root if document.is_relative_to(producer_root) else evidence_root
    size = ref.get("bytes", ref.get("size_bytes"))
    matches = []
    for path in (namespace / relative, document.parent / relative):
        try:
            physical = _physical(root, path)
        except (ValueError, FileNotFoundError):
            continue
        if (_digest(physical) == ref["sha256"] and (size is None
                or (type(size) is int and physical.stat().st_size == size))):
            matches.append(physical)
    matches = list(dict.fromkeys(matches))
    if len(matches) != 1:
        raise ValueError("C05 evidence ref missing or ambiguous: " + ref["path"])
    return matches[0]


def _walk_source_closure(root: Path, document: Path, initial, *,
                         evidence_root: Path, producer_root: Path,
                         model_inventory: Path | None = None) -> list[Path]:
    """Follow exact refs through JSON evidence with a strict finite bound."""
    pending: list[tuple[Path, dict]] = []
    for ref in _iter_refs(initial):
        pending.append((document, ref))
    files: list[Path] = []
    seen: set[Path] = set()
    while pending:
        if len(seen) + len(pending) > 512:
            raise ValueError("C05 recursive evidence closure exceeds 512 files")
        document, ref = pending.pop()
        path = _resolve_nested_ref(
            root, document, ref, evidence_root=evidence_root,
            producer_root=producer_root)
        if path in seen:
            continue
        seen.add(path); files.append(path)
        if path == model_inventory:
            # This exact producer-bound document inventories external weights.
            # CPU artifact acceptance retains the inventory, not model assets.
            _validate_weights_manifest_schema(path)
        elif path.suffix == ".json":
            value = _load(path)
            pending.extend((path, child) for child in _iter_refs(value))
    return files


def _producer_root_from_candidate(root: Path, evidence_root: Path,
                                  candidate: dict) -> Path:
    mode_ref = candidate.get("input_refs", {}).get("mode_bank")
    if not isinstance(mode_ref, dict):
        raise ValueError("C05 candidate mode-bank ref required")
    mode_path = _physical(root, evidence_root / mode_ref["path"])
    if _digest(mode_path) != mode_ref.get("sha256"):
        raise ValueError("C05 candidate mode-bank bytes changed")
    mode_root = mode_path.parent
    manifest = _load(mode_root / "raw-manifest.json")
    requests = [ref for ref in manifest.get("producer_input_refs", [])
                if isinstance(ref, dict)
                and Path(str(ref.get("path", ""))).name == "request.json"]
    if len(requests) != 1:
        raise ValueError("One C05 producer request ref required")
    request_ref = requests[0]
    relative = Path(request_ref["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Canonical C05 producer request ref required")
    matches = []
    ancestor = mode_root
    while ancestor.is_relative_to(evidence_root):
        path = ancestor / relative
        if (path.is_file() and not path.is_symlink()
                and _digest(path) == request_ref.get("sha256")):
            request = _load(path)
            output = Path(str(request.get("output_relative", "")))
            if (not output.is_absolute() and ".." not in output.parts
                    and (ancestor / output).resolve() == mode_root):
                matches.append(ancestor)
        if ancestor == evidence_root:
            break
        ancestor = ancestor.parent
    matches = list(dict.fromkeys(matches))
    if len(matches) != 1:
        raise ValueError("Unique C05 producer workspace required")
    return matches[0]


def artifact_refs(root, artifact_candidate, *, evidence_root):
    """Pin every retained artifact byte plus its recursive source closure."""
    root = Path(root).resolve()
    artifact, candidate_path, candidate = _candidate_directory(root, artifact_candidate)
    inventory = list(artifact.rglob("*"))
    if any(path.is_symlink() for path in inventory):
        raise ValueError("Symlinks are forbidden in retained C05 artifacts")
    files = [path.resolve() for path in inventory if path.is_file()]
    if candidate_path not in files or len(files) < 5 or len(files) > 32:
        raise ValueError("Bounded complete C05 terminal artifact required")
    required = {"candidate.json", "result.json", "raw-manifest.json",
                "certificate.json", "raw-evidence.tar"}
    if not required.issubset({path.relative_to(artifact).as_posix() for path in files}):
        raise ValueError("C05 terminal artifact lacks required receipt/archive files")
    method_ids = candidate.get("method_ids")
    if not isinstance(method_ids, dict) or set(method_ids) != set(ROLES):
        raise ValueError("Exact five-role C05 method map required")
    if candidate["status"] == "completed_unqualified":
        expected = {f"roles/{role}/{name}" for role in ROLES
                    for name in ("sequence.npz", "report.json")}
        if not expected.issubset(
                {path.relative_to(artifact).as_posix() for path in files}):
            raise ValueError("Completed C05 artifact lacks a full five-role inventory")
    inputs = candidate.get("input_refs")
    if not isinstance(inputs, (dict, list)):
        raise ValueError("C05 candidate must retain exact input_refs")
    evidence_root = Path(evidence_root).resolve()
    candidate_path.relative_to(evidence_root)
    producer_root = _producer_root_from_candidate(root, evidence_root, candidate)
    mode_path = _physical(root, evidence_root / inputs["mode_bank"]["path"])
    producer_manifest_path = mode_path.parent / "raw-manifest.json"
    producer_manifest = _load(producer_manifest_path)
    model_ref = producer_manifest.get("provenance", {}).get("model_ref")
    model_inventory = None
    if model_ref is not None:
        model_inventory = _resolve_nested_ref(
            root, producer_manifest_path, model_ref, evidence_root=evidence_root,
            producer_root=producer_root)
    files.extend(_walk_source_closure(
        root, candidate_path, inputs, evidence_root=evidence_root,
        producer_root=producer_root, model_inventory=model_inventory))
    # De-duplicate only after exact physical resolution, then normalize every
    # source/artifact reference into the project namespace used by the harness.
    return [_ref(root, path) for path in sorted(set(files))]


def _candidate_execution(root: Path, plan_path: Path, receipt_path: Path, native):
    """Bind acceptance to one exact successful candidate-materialization attempt."""
    plan_path = _physical(root, plan_path)
    receipt_path = _physical(root, receipt_path)
    plan, receipt = _load(plan_path), _load(receipt_path)
    native.validate_plan(root, plan)
    run_root = (root / plan.get("output_root", "") / plan.get("run_id", "")).resolve()
    run_root.relative_to(root)
    if receipt_path != run_root / "receipt.json":
        raise ValueError("Canonical C05 candidate receipt path required")
    jobs, attempts = plan.get("jobs"), receipt.get("attempts")
    if (receipt.get("status") != "completed"
            or receipt.get("run_id") != plan.get("run_id")
            or receipt.get("plan_digest") != plan.get("plan_digest")
            or receipt.get("purpose") != plan.get("purpose")
            or receipt.get("evidence_mode") != plan.get("evidence_mode")
            or receipt.get("provenance") != plan.get("provenance")
            or not isinstance(jobs, list) or len(jobs) != 1
            or not isinstance(attempts, list) or len(attempts) != 1
            or plan.get("limits", {}).get("max_attempts") != 1
            or plan.get("limits", {}).get("max_retries_per_trial") != 0):
        raise ValueError("Exact completed single-attempt C05 candidate receipt required")
    job, attempt = jobs[0], attempts[0]
    if (job.get("trial_id") != "prepare-c05-spatial-mode-candidate"
            or attempt.get("trial_id") != job["trial_id"]
            or attempt.get("status") != "completed"
            or attempt.get("exit_code") != 0
            or attempt.get("retry_index") != 0
            or any(attempt.get(name) != job.get(name)
                   for name in ("input_refs", "code_refs", "seed", "group", "arm_role"))):
        raise ValueError("C05 candidate job/attempt identity mismatch")
    attempt_relative = Path(attempt.get("attempt_path", ""))
    if attempt_relative.is_absolute() or ".." in attempt_relative.parts:
        raise ValueError("Canonical C05 candidate attempt path required")
    workspace = (root / attempt_relative / "workspace").resolve()
    workspace.relative_to(root)
    if (attempt_relative.parent != run_root.relative_to(root)
            or attempt.get("attempt_id") != attempt_relative.name
            or not workspace.is_dir()):
        raise ValueError("Retained C05 candidate workspace is missing")
    replacements = {}
    for ref in job["input_refs"] + job["code_refs"]:
        source = _physical(root, ref["path"])
        staged = _physical(root, workspace / ref["path"])
        if _digest(source) != ref["sha256"] or _digest(staged) != ref["sha256"]:
            raise ValueError("C05 candidate source/staged identity mismatch")
        replacements[str(source)] = str(staged)
    expected_command = []
    for argument in job["command"]:
        for source, staged in sorted(replacements.items(), key=lambda row: -len(row[0])):
            argument = argument.replace(source, staged)
        expected_command.append(argument)
    expected_cwd = workspace if job["cwd"] == "." else workspace / job["cwd"]
    if attempt.get("command") != expected_command or Path(attempt.get("cwd", "")) != expected_cwd:
        raise ValueError("C05 candidate staged command/cwd differs from frozen plan")
    expected_outputs = [_ref(root, _physical(root, workspace / relative))
                        for relative in job["output_paths"]]
    if attempt.get("output_refs") != expected_outputs:
        raise ValueError("C05 candidate output receipt inventory mismatch")
    try:
        index = job["command"].index("--output-relative")
        output_relative = Path(job["command"][index + 1])
    except (ValueError, IndexError) as error:
        raise ValueError("Frozen C05 candidate output argv required") from error
    if (output_relative.is_absolute() or ".." in output_relative.parts
            or output_relative.as_posix() != job["command"][index + 1]):
        raise ValueError("Canonical C05 candidate output path required")
    output = (workspace / output_relative).resolve()
    output.relative_to(workspace)
    always = {"candidate.json", "result.json", "raw-manifest.json",
              "certificate.json", "solver-certificate.json", "raw-evidence.tar"}
    always.update(f"roles/{role}/report.json" for role in ROLES)
    declared = {Path(path).relative_to(output_relative).as_posix()
                for path in job["output_paths"]
                if Path(path).is_relative_to(output_relative)}
    if declared != always or len(declared) != len(job["output_paths"]):
        raise ValueError("C05 candidate plan must declare exact always-present outputs")
    return plan_path, receipt_path, workspace, output / "candidate.json"


def prospective_freeze_refs(root, artifact_candidate, freeze_path):
    freeze_path = _physical(root, freeze_path)
    freeze = _load(freeze_path)
    if freeze.get("candidate_artifact_ref") != _ref(root, artifact_candidate):
        raise ValueError("Freeze must bind the supplied C05 candidate")
    refs = [_ref(root, freeze_path), *_nested_refs(freeze)]
    decision = _load(_checked_ref(root, freeze["b_star_decision_ref"]))
    basis = decision.get("selection_basis_refs")
    if not isinstance(basis, list) or not basis:
        raise ValueError("Prospective C05 B* selection basis required")
    refs.extend(basis)
    if freeze.get("semantic_review_ref") is not None:
        review = _load(_checked_ref(root, freeze["semantic_review_ref"]))
        refs.extend(_nested_refs(review))
        for ref in [review.get("method_spec_ref"), *review.get("evidence_refs", [])]:
            if ref is not None:
                refs.extend(_nested_refs(_load(_checked_ref(root, ref))))
    for ref in refs:
        _checked_ref(root, ref)
    return refs


def build_plans(root, *, skill_dir, artifact_candidate, candidate_plan,
                candidate_receipt, plan_dir, run_id,
                wall_seconds, ram_mib, freeze=None, ground_truth=None,
                population=None, dataset_admission=None, dataset_semantics=None,
                repo_root=None, timeout_seconds=None):
    root, skill_dir, plan_dir = (Path(item).resolve()
                                 for item in (root, skill_dir, plan_dir))
    artifact_candidate = _physical(root, artifact_candidate)
    plan_dir.relative_to(root)
    if type(wall_seconds) is not int or not 1 <= wall_seconds <= 26940:
        raise ValueError("Explicit remaining CPU budget must be 1..26940 seconds")
    if type(ram_mib) is not int or ram_mib < 1:
        raise ValueError("Explicit positive RAM budget required")
    scripts = skill_dir / "scripts"
    if not (scripts / "run_harness.py").is_file():
        raise ValueError("Complete installed research-autopilot required")
    sys.path.insert(0, str(scripts))
    import run_experiments as native
    import run_harness as harness
    candidate_plan_path, candidate_receipt_path, candidate_workspace, \
        derived_candidate = _candidate_execution(
            root, candidate_plan, candidate_receipt, native)
    if artifact_candidate != derived_candidate:
        raise ValueError("Candidate path differs from exact materialization receipt")
    inputs = artifact_refs(
        root, artifact_candidate, evidence_root=candidate_workspace)
    inputs.extend((_ref(root, candidate_plan_path),
                   _ref(root, candidate_receipt_path)))
    _, _, candidate_document = _candidate_directory(root, artifact_candidate)
    implementation_ref = candidate_document.get("input_refs", {}).get(
        "artifact_implementation")
    if not isinstance(implementation_ref, dict):
        raise ValueError("C05 candidate implementation ref required")
    evidence_sentinel = _physical(candidate_workspace, implementation_ref["path"])
    if (_digest(evidence_sentinel) != implementation_ref.get("sha256")
            or _digest(root / "actionmesh/research_math/c05_candidate_artifacts.py")
               != implementation_ref.get("sha256")):
        raise ValueError("C05 retained/current artifact implementation differs")
    request_paths = (ground_truth, population, dataset_admission,
                     dataset_semantics, repo_root)
    wants_scoring = any(value is not None for value in (*request_paths, timeout_seconds))
    if wants_scoring and (freeze is None or any(value is None for value in request_paths)
                          or type(timeout_seconds) is not int or timeout_seconds < 1):
        raise ValueError("Scoring preparation requires freeze and all scoring inputs")
    if freeze is not None:
        freeze = _physical(root, freeze)
        inputs.extend(prospective_freeze_refs(root, artifact_candidate, freeze))
    scoring_paths, official_paths = [], []
    if wants_scoring:
        scoring_paths = [_physical(root, path) for path in request_paths[:4]]
        repository = Path(repo_root)
        repository = repository if repository.is_absolute() else root / repository
        official_paths = [_physical(root, repository / "actionbench" / name)
                          for name in official_source_names(root)]
        inputs.extend(_ref(root, path) for path in [*scoring_paths, *official_paths])
    unique = {}
    for ref in inputs:
        if ref["path"] in unique and unique[ref["path"]] != ref:
            raise ValueError("Conflicting C05 acceptance input identity")
        unique[ref["path"]] = ref
    code = [_ref(root, path) for path in acceptance_sources(root)]
    mode = "request_preparation" if freeze is not None else "native_method_acceptance"
    program = (
        "import json,sys\nfrom pathlib import Path\n"
        "from research_math import c05_candidate_artifacts as candidate\n"
        "root=Path(candidate.__file__).resolve().parents[2]\n"
        "sentinel=Path(sys.argv[2]).resolve();relative=Path(sys.argv[3])\n"
        "evidence_root=sentinel\n"
        "for _ in relative.parts:evidence_root=evidence_root.parent\n"
        "if (evidence_root/relative).resolve()!=sentinel: "
        "raise ValueError('C05 candidate workspace sentinel mismatch')\n"
        "terminal=candidate.validate_candidate_artifact(root,Path(sys.argv[1]).resolve().parent,"
        "evidence_root=evidence_root)\n"
        "if terminal['status']!='completed_unqualified': "
        "raise ValueError('Complete C05 artifact required for acceptance/request preparation')\n"
        "if set(terminal['method_ids'])!=set(candidate.METHOD_IDS): "
        "raise ValueError('Exact five C05 methods required')\n"
        "print(json.dumps({'mode':" + repr(mode) + ","
        "'terminal_status':terminal['status'],'candidate_id':terminal['candidate_id'],"
        "'native_qualified':False,'scientific_admission':False}))\n")
    command = [sys.executable, "-c", program, str(artifact_candidate),
               str(evidence_sentinel), implementation_ref["path"]]
    outputs = []
    if freeze is not None:
        program += (
            "from research_math import c05_native_comparison as comparison\n"
            "request=comparison.make_request(root,freeze_path=Path(sys.argv[4]),"
            "artifact_evidence_root=evidence_root)\n"
            "comparison_path=Path('c05-comparison-request.json')\n"
            "with comparison_path.open('x') as stream: "
            "json.dump(request,stream,indent=2,allow_nan=False)\n")
        command.append(str(freeze)); outputs.append("actionmesh/c05-comparison-request.json")
    if wants_scoring:
        program += (
            "from research_math import c05_native_scoring as scoring\n"
            "request=scoring.make_scoring_request(root,comparison_path=comparison_path,"
            "ground_truth=Path(sys.argv[5]),population=Path(sys.argv[6]),"
            "dataset_admission=Path(sys.argv[7]),dataset_semantics=Path(sys.argv[8]),"
            "repo_root=Path(sys.argv[9]).parent.parent,timeout_seconds=int(sys.argv[10]))\n"
            "with Path('c05-scoring-request.json').open('x') as stream: "
            "json.dump(request,stream,indent=2,allow_nan=False)\n")
        command.extend(str(path) for path in scoring_paths)
        command.extend((str(official_paths[0]), str(timeout_seconds)))
        outputs.append("actionmesh/c05-scoring-request.json")
    command[2] = program
    plan = native.make_plan(root, run_id=run_id, jobs=[{
        "trial_id": "c05-retained-native-acceptance", "command": command,
        "cwd": "actionmesh", "input_refs": list(unique.values()),
        "code_refs": code, "output_paths": outputs, "seed": 42,
        "group": "engineering", "arm_role": mode,
    }], provenance={
        "git_revision": "exact current C05 acceptance code_refs",
        "model_revision": "none; no generation or scoring",
        "data_revision": _digest(artifact_candidate),
        "environment_digest": hashlib.sha256(json.dumps({
            "python": sys.version, "skill": str(skill_dir),
            "native": _digest(scripts / "run_experiments.py"),
            "harness": _digest(scripts / "run_harness.py"),
        }, sort_keys=True).encode()).hexdigest(),
    }, limits={"max_attempts": 1, "max_development_trials": 1,
        "max_confirmation_trials": 0, "max_retries_per_trial": 0,
        "wall_time_seconds": wall_seconds, "attempt_timeout_seconds": wall_seconds})
    plan_dir.mkdir(parents=True, exist_ok=False)
    native_path = plan_dir / "native.json"
    native_path.write_text(json.dumps(plan, indent=2) + "\n")
    outer = harness.make_plan(root, batch_id=run_id, tasks=[{
        "task_id": "c05-retained-native-acceptance", "idea_id": CANDIDATE_ID,
        "depends_on": [], "priority": 1, "plan_ref": _ref(root, native_path),
        "resources": {"cpu_cores": 1, "ram_mib": ram_mib, "gpu_count": 0,
            "gpu_peak_mib": None, "allow_gpu_share": False,
            "memory_profile_ref": None, "exclusive_keys": []},
    }], limits={"total_wall_seconds": wall_seconds + 60,
        "window_seconds": wall_seconds + 60, "max_parallel_tasks": 1,
        "cpu_cores": 1, "ram_mib": ram_mib, "max_gpu_task_seconds": 0})
    (plan_dir / "harness.json").write_text(json.dumps(outer, indent=2) + "\n")
    return plan, outer


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "skill-dir", "artifact-candidate", "candidate-plan",
                 "candidate-receipt", "plan-dir"):
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
        artifact_candidate=args.artifact_candidate,
        candidate_plan=args.candidate_plan,
        candidate_receipt=args.candidate_receipt, plan_dir=args.plan_dir,
        run_id=args.run_id, wall_seconds=args.wall_seconds, ram_mib=args.ram_mib,
        freeze=args.freeze, ground_truth=args.ground_truth,
        population=args.population, dataset_admission=args.dataset_admission,
        dataset_semantics=args.dataset_semantics, repo_root=args.repo_root,
        timeout_seconds=args.timeout_seconds)
    print(json.dumps({"plan": str(args.plan_dir.resolve() / "harness.json"),
        "approved_plan_digest": outer["plan_digest"], "execution_started": False,
        "source_delivery_status": "generated_unexecuted", "gpu_count": 0,
        "mode": "request_preparation" if args.freeze is not None
        else "native_method_acceptance", "native_qualified": False,
        "scientific_admission": False, "local_method_verified": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
