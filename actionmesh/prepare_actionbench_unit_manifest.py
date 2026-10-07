"""Emit, but never execute, the current-release unit-manifest plan."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def file_ref(root: Path, path: Path):
    path = Path(path).resolve()
    return {"path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def build_plan(root: Path, *, skill_dir: Path, contract_path: Path,
               population_path: Path, snapshot_admission_path: Path,
               dataset_semantics_path: Path, source_root: Path,
               dataset_root: Path, plan_dir: Path, run_id: str,
               wall_seconds: int, ram_mib: int, cpu_cores: int):
    root = Path(root).resolve()
    skill_dir = Path(skill_dir).resolve()
    plan_dir = Path(plan_dir).resolve()
    paths = [Path(value).resolve() for value in
             (contract_path, population_path, snapshot_admission_path,
              dataset_semantics_path)]
    for path in [*paths, plan_dir]:
        path.relative_to(root)
    source_root, dataset_root = Path(source_root), Path(dataset_root)
    if (not source_root.is_dir() or source_root.is_symlink() or
            not dataset_root.is_dir() or dataset_root.is_symlink()):
        raise ValueError("Physical source and dataset roots are required")
    source_root, dataset_root = source_root.resolve(), dataset_root.resolve()
    if not 60 <= wall_seconds <= 3600:
        raise ValueError("Unit-manifest wall limit must be 60..3600 seconds")
    if ram_mib < 256 or cpu_cores < 1:
        raise ValueError("Positive bounded CPU/RAM admission required")
    scripts = skill_dir / "scripts"
    if not (scripts / "run_harness.py").is_file():
        raise ValueError("Complete installed research-autopilot skill required")
    sys.path.insert(0, str(scripts))
    import run_experiments as native
    import run_harness as harness

    contract, population, snapshot, semantics = [json.loads(path.read_text())
                                                  for path in paths]
    if snapshot.get("status") != "admitted_engineering_snapshot":
        raise ValueError("Successful snapshot admission required")
    if semantics.get("status") != "admitted_engineering_dataset_semantics":
        raise ValueError("Successful dataset semantic admission required")
    if contract.get("population", {}).get("revision") != population.get("revision"):
        raise ValueError("Contract and population revisions differ")
    input_refs = [file_ref(root, path) for path in paths]
    code_paths = [
        root / "actionmesh/research_math/__init__.py",
        root / "actionmesh/research_math/actionbench_unit_manifest.py",
        root / "actionmesh/prepare_actionbench_unit_manifest.py",
    ]
    code_refs = [file_ref(root, path) for path in code_paths]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                              check=True, capture_output=True,
                              text=True).stdout.strip()
    output_relative = "inputs/actionbench-full128-snapshots/unit-manifest.json"
    command = [
        sys.executable, "-m", "research_math.actionbench_unit_manifest",
        "--contract", str(paths[0]), "--population", str(paths[1]),
        "--snapshot-admission", str(paths[2]),
        "--dataset-semantics", str(paths[3]),
        "--source-root", str(source_root), "--dataset-root", str(dataset_root),
        "--output", "../" + output_relative,
    ]
    plan = native.make_plan(
        root, run_id=run_id, purpose="engineering", evidence_mode="developmental",
        jobs=[{
            "trial_id": "actionbench-current-release-unit-manifest",
            "command": command, "cwd": "actionmesh",
            "input_refs": input_refs, "code_refs": code_refs,
            "output_paths": [output_relative], "seed": 0,
            "group": "engineering", "arm_role": "unit-manifest-freeze",
        }],
        provenance={
            "git_revision": revision, "model_revision": "bound by snapshot admission",
            "data_revision": population["revision"],
            "environment_digest": hashlib.sha256(sys.version.encode()).hexdigest(),
            "data_refs": input_refs,
        },
        limits={"max_attempts": 1, "max_development_trials": 1,
                "max_confirmation_trials": 0, "max_retries_per_trial": 0,
                "wall_time_seconds": wall_seconds,
                "attempt_timeout_seconds": wall_seconds},
    )
    plan_dir.mkdir(parents=True, exist_ok=False)
    native_path = plan_dir / "native.json"
    native_path.write_text(json.dumps(plan, indent=2) + "\n")
    outer = harness.make_plan(
        root, batch_id=run_id,
        tasks=[{
            "task_id": "actionbench-current-release-unit-manifest",
            "idea_id": "baseline-qualification", "depends_on": [], "priority": 1,
            "plan_ref": file_ref(root, native_path),
            "resources": {"cpu_cores": cpu_cores, "ram_mib": ram_mib,
                          "gpu_count": 0, "gpu_peak_mib": None,
                          "allow_gpu_share": False,
                          "memory_profile_ref": None,
                          "exclusive_keys": ["actionbench-current-release-unit-manifest"]},
        }],
        limits={"total_wall_seconds": wall_seconds + 60,
                "window_seconds": wall_seconds + 60, "max_parallel_tasks": 1,
                "cpu_cores": cpu_cores, "ram_mib": ram_mib,
                "max_gpu_task_seconds": 0},
    )
    (plan_dir / "harness.json").write_text(json.dumps(outer, indent=2) + "\n")
    return outer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "skill-dir", "contract", "population",
                 "snapshot-admission", "dataset-semantics", "source-root",
                 "dataset-root", "plan-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--wall-seconds", type=int, default=900)
    parser.add_argument("--ram-mib", type=int, default=1024)
    parser.add_argument("--cpu-cores", type=int, default=1)
    args = parser.parse_args()
    outer = build_plan(
        args.root, skill_dir=args.skill_dir, contract_path=args.contract,
        population_path=args.population,
        snapshot_admission_path=args.snapshot_admission,
        dataset_semantics_path=args.dataset_semantics,
        source_root=args.source_root, dataset_root=args.dataset_root,
        plan_dir=args.plan_dir, run_id=args.run_id,
        wall_seconds=args.wall_seconds, ram_mib=args.ram_mib,
        cpu_cores=args.cpu_cores)
    print(json.dumps({"plan": str(args.plan_dir.resolve() / "harness.json"),
                      "approved_plan_digest": outer["plan_digest"],
                      "execution_started": False, "gpu_count": 0,
                      "scientific_effect_qualification": False,
                      "dispatch_ready": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
