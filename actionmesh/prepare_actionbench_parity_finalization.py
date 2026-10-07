"""Build, but never execute, the CPU-only parity finalization harness plan."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from research_math.control_scoring import file_ref, resolve_ref


FINALIZATION_SOURCE_NAMES = (
    "research_math/__init__.py",
    "research_math/control_scoring.py",
    "research_math/actionbench_parity.py",
    "research_census_eval.py",
    "official_actionbench_adapter.py",
    "finalize_actionbench_parity.py",
    "prepare_actionbench_parity_finalization.py",
)


def finalization_code_sources(root: Path) -> list[Path]:
    root = Path(root).resolve()
    return [root / "actionmesh" / name for name in FINALIZATION_SOURCE_NAMES]


def _nested_refs(value):
    if isinstance(value, dict):
        if set(value) == {"path", "sha256"}:
            yield value
            return
        for nested in value.values():
            yield from _nested_refs(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _nested_refs(nested)


def reference_closure(root: Path, seed_paths: list[Path]) -> list[dict]:
    """Pin seed files and every project ref reachable from their JSON bodies."""
    root = Path(root).resolve()
    queue = [Path(path).resolve() for path in seed_paths]
    refs: dict[str, dict] = {}
    while queue:
        path = queue.pop()
        ref = file_ref(root, path)
        previous = refs.get(ref["path"])
        if previous is not None:
            if previous != ref:
                raise ValueError("Conflicting finalization reference: " + ref["path"])
            continue
        refs[ref["path"]] = ref
        if path.suffix == ".json":
            document = json.loads(path.read_text())
            for nested_ref in _nested_refs(document):
                # Raw scorer provenance records observed absolute paths. Admit
                # them only if they resolve inside this exact controller root,
                # then retain a project-relative ref without editing raw evidence.
                if Path(nested_ref['path']).is_absolute():
                    absolute = Path(nested_ref['path']).resolve()
                    try:
                        relative = absolute.relative_to(root).as_posix()
                    except ValueError as exc:
                        raise ValueError('Absolute evidence ref escapes project root') from exc
                    nested_ref = {'path': relative, 'sha256': nested_ref['sha256']}
                queue.append(resolve_ref(root, nested_ref))
    return [refs[path] for path in sorted(refs)]


def canonical_origin_paths(root: Path, harness_plan: dict, native_plan: dict,
                           native_receipt: dict) -> list[Path]:
    """Return the canonical runtime records consumed by the finalizer."""
    root = Path(root).resolve()
    tasks = harness_plan.get("tasks", [])
    attempts = native_receipt.get("attempts", [])
    if len(tasks) != 1 or len(attempts) != 1:
        raise ValueError("Exact one-task, one-attempt parity origin required")
    batch_root = root / harness_plan.get("output_root", "") / harness_plan.get("batch_id", "")
    task_root = batch_root / "tasks" / tasks[0].get("task_id", "")
    run_root = root / native_plan.get("output_root", "") / native_plan.get("run_id", "")
    attempt_root = run_root / attempts[0].get("attempt_id", "")
    return [
        batch_root / "plan.json",
        batch_root / "state.json",
        batch_root / "report.json",
        task_root / "task.json",
        task_root / "result.json",
        run_root / "plan.json",
        run_root / "receipt.json",
        attempt_root / "attempt.json",
    ]


def build_plans(root: Path, *, request_path: Path, contract_path: Path,
                parity_harness_plan_path: Path, parity_harness_report_path: Path,
                parity_native_plan_path: Path, parity_native_receipt_path: Path,
                approved_parity_plan_digest: str, output: Path, skill_dir: Path,
                plan_dir: Path, run_id: str, wall_seconds: int, ram_mib: int,
                cpu_cores: int):
    root = Path(root).resolve()
    paths = [request_path, contract_path, parity_harness_plan_path,
             parity_harness_report_path, parity_native_plan_path,
             parity_native_receipt_path, output, plan_dir]
    (request_path, contract_path, parity_harness_plan_path,
     parity_harness_report_path, parity_native_plan_path,
     parity_native_receipt_path, output, plan_dir) = [
        Path(path).resolve() for path in paths]
    output.relative_to(root)
    plan_dir.relative_to(root)
    if output != root / "actionmesh" / "actionbench-parity-output":
        raise ValueError("Exact promoted parity bundle target required")
    if not 1 <= wall_seconds <= 900:
        raise ValueError("Finite CPU finalization limit must be <=900 seconds")
    if ram_mib < 1 or cpu_cores < 1:
        raise ValueError("Positive admitted RAM and CPU limits required")
    sidecar = output / "faithful-harness-verification.json"
    if sidecar.exists():
        raise FileExistsError("Parity finalization sidecar is single-use")
    scripts = Path(skill_dir).resolve() / "scripts"
    if not (scripts / "run_harness.py").is_file():
        raise ValueError("Complete installed research-autopilot skill required")
    sys.path.insert(0, str(scripts))
    import run_experiments as native
    import run_harness as harness

    request = json.loads(request_path.read_text())
    contract = json.loads(contract_path.read_text())
    harness_plan = json.loads(parity_harness_plan_path.read_text())
    harness_report = json.loads(parity_harness_report_path.read_text())
    native_plan = json.loads(parity_native_plan_path.read_text())
    native_receipt = json.loads(parity_native_receipt_path.read_text())
    if (harness_plan.get("plan_digest") != approved_parity_plan_digest or
            harness_report.get("plan_digest") != approved_parity_plan_digest or
            harness_report.get("status") != "completed"):
        raise ValueError("Approved completed parity harness required")
    if (native_receipt.get("plan_digest") != native_plan.get("plan_digest") or
            native_receipt.get("status") != "completed"):
        raise ValueError("Completed parity native receipt required")
    jobs = native_plan.get("jobs", [])
    if len(jobs) != 1:
        raise ValueError("Exact one-job parity native plan required")

    seed_paths = [
        request_path,
        contract_path,
        output / "record.json",
        output / "parity-evidence.json",
        output / "parity-bundle-attestation.json",
        parity_harness_plan_path,
        parity_harness_report_path,
        parity_native_plan_path,
        parity_native_receipt_path,
    ]
    seed_paths += canonical_origin_paths(
        root, harness_plan, native_plan, native_receipt)
    seed_paths += [root / relative for relative in jobs[0].get("output_paths", [])]
    input_refs = reference_closure(root, seed_paths)

    code_paths = finalization_code_sources(root)
    missing = [path for path in code_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing finalization source: " + str(missing[0]))
    code_refs = [file_ref(root, path) for path in code_paths]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(
        ["git", "diff", "HEAD", "--binary"], cwd=root, check=True,
        capture_output=True).stdout
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"], cwd=root,
        check=True, capture_output=True, text=True).stdout.splitlines()
    if any(path.endswith(".py") for path in untracked):
        raise ValueError("Commit all executable Python sources before Local acceptance")

    plan_dir.mkdir(parents=True, exist_ok=False)
    dirty_path = plan_dir / "dirty.patch"
    dirty_path.write_bytes(dirty)
    input_refs = {ref["path"]: ref for ref in input_refs}
    dirty_ref = file_ref(root, dirty_path)
    input_refs[dirty_ref["path"]] = dirty_ref
    finalizer_path = root / "actionmesh" / "finalize_actionbench_parity.py"
    command = [
        sys.executable,
        str(finalizer_path),
        "--root", "..",
        "--request", str(request_path),
        "--contract", str(contract_path),
        "--output", "actionbench-parity-output",
        "--harness-plan", str(parity_harness_plan_path),
        "--harness-report", str(parity_harness_report_path),
        "--native-plan", str(parity_native_plan_path),
        "--native-receipt", str(parity_native_receipt_path),
        "--approved-plan-digest", approved_parity_plan_digest,
    ]
    plan = native.make_plan(
        root, run_id=run_id, purpose="engineering", evidence_mode="developmental",
        jobs=[{
            "trial_id": "finalize-promoted-actionbench-parity",
            "command": command,
            "cwd": "actionmesh",
            "input_refs": list(input_refs.values()),
            "code_refs": code_refs,
            "output_paths": [
                "actionmesh/actionbench-parity-output/"
                "faithful-harness-verification.json"
            ],
            "seed": request.get("scoring_seed", 0),
            "group": "engineering",
            "arm_role": "parity-finalization",
        }],
        provenance={
            "git_revision": revision,
            "git_refs": [dirty_ref],
            "model_revision": "none; deterministic evidence finalization only",
            "data_revision": request.get("ground_truth_ref", {}).get("sha256", "none"),
            "environment_digest": hashlib.sha256(sys.version.encode()).hexdigest(),
        },
        limits={
            "max_attempts": 1,
            "max_development_trials": 1,
            "max_confirmation_trials": 0,
            "max_retries_per_trial": 0,
            "wall_time_seconds": wall_seconds,
            "attempt_timeout_seconds": wall_seconds,
        },
    )
    native_path = plan_dir / "native.json"
    native_path.write_text(json.dumps(plan, indent=2) + "\n")
    outer = harness.make_plan(
        root, batch_id=run_id,
        tasks=[{
            "task_id": "actionbench-parity-finalization",
            "idea_id": "evaluator-implementation-equivalence",
            "depends_on": [],
            "priority": 1,
            "plan_ref": file_ref(root, native_path),
            "resources": {
                "cpu_cores": cpu_cores,
                "ram_mib": ram_mib,
                "gpu_count": 0,
                "gpu_peak_mib": None,
                "allow_gpu_share": False,
                "memory_profile_ref": None,
                "exclusive_keys": ["actionbench-parity-finalization"],
            },
        }],
        limits={
            "total_wall_seconds": wall_seconds,
            "window_seconds": wall_seconds,
            "max_parallel_tasks": 1,
            "cpu_cores": cpu_cores,
            "ram_mib": ram_mib,
            "max_gpu_task_seconds": 0,
        },
    )
    (plan_dir / "harness.json").write_text(json.dumps(outer, indent=2) + "\n")
    return outer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "root", "request", "contract", "parity-harness-plan",
        "parity-harness-report", "parity-native-plan", "parity-native-receipt",
        "output", "skill-dir", "plan-dir",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--approved-parity-plan-digest", required=True)
    parser.add_argument("--run-id", required=True)
    for name in ("wall-seconds", "ram-mib", "cpu-cores"):
        parser.add_argument("--" + name, type=int, required=True)
    args = parser.parse_args()
    plan = build_plans(
        args.root,
        request_path=args.request,
        contract_path=args.contract,
        parity_harness_plan_path=args.parity_harness_plan,
        parity_harness_report_path=args.parity_harness_report,
        parity_native_plan_path=args.parity_native_plan,
        parity_native_receipt_path=args.parity_native_receipt,
        approved_parity_plan_digest=args.approved_parity_plan_digest,
        output=args.output,
        skill_dir=args.skill_dir,
        plan_dir=args.plan_dir,
        run_id=args.run_id,
        wall_seconds=args.wall_seconds,
        ram_mib=args.ram_mib,
        cpu_cores=args.cpu_cores,
    )
    print(json.dumps({
        "plan": str(args.plan_dir / "harness.json"),
        "approved_plan_digest": plan["plan_digest"],
        "execution_started": False,
        "scope": "promoted parity evidence finalization only",
        "gpu_count": 0,
        "native_contract_qualified": False,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
