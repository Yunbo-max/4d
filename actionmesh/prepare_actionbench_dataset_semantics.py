"""Emit, but never execute, the ActionBench dataset-semantics admission plan."""
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
               dataset_root: Path, plan_dir: Path, run_id: str,
               wall_seconds: int, ram_mib: int, cpu_cores: int):
    root = Path(root).resolve()
    skill_dir = Path(skill_dir).resolve()
    contract_path = Path(contract_path).resolve()
    population_path = Path(population_path).resolve()
    snapshot_admission_path = Path(snapshot_admission_path).resolve()
    dataset_root = Path(dataset_root)
    plan_dir = Path(plan_dir).resolve()
    for path in (contract_path, population_path, snapshot_admission_path, plan_dir):
        path.relative_to(root)
    if not dataset_root.is_dir() or dataset_root.is_symlink():
        raise ValueError("A physical admitted ActionBench dataset root is required")
    dataset_root = dataset_root.resolve()
    if not 300 <= wall_seconds <= 7200:
        raise ValueError("Dataset semantics wall limit must be 300..7200 seconds")
    if ram_mib < 512 or cpu_cores < 1:
        raise ValueError("Positive bounded CPU/RAM admission required")
    scripts = skill_dir / "scripts"
    if not (scripts / "run_harness.py").is_file():
        raise ValueError("Complete installed research-autopilot skill required")
    sys.path.insert(0, str(scripts))
    import run_experiments as native
    import run_harness as harness

    contract = json.loads(contract_path.read_text())
    population = json.loads(population_path.read_text())
    admission = json.loads(snapshot_admission_path.read_text())
    if (admission.get("kind") != "actionbench-full128-snapshot-admission" or
            admission.get("status") != "admitted_engineering_snapshot"):
        raise ValueError("A successful snapshot admission is required")
    if (contract.get("dataset_revision") != population.get("revision") or
            admission.get("snapshots", {}).get("dataset", {}).get("revision") !=
            population.get("revision")):
        raise ValueError("Contract, population and snapshot admission differ")
    population_ref = file_ref(root, population_path)
    snapshot_ref = file_ref(root, snapshot_admission_path)
    if contract.get("population_ref") != population_ref:
        raise ValueError("Contract does not bind the exact population file")
    code_paths = [
        root / "actionmesh/research_math/__init__.py",
        root / "actionmesh/research_math/actionbench_dataset_semantics.py",
        root / "actionmesh/prepare_actionbench_dataset_semantics.py",
    ]
    input_refs = [file_ref(root, contract_path), population_ref, snapshot_ref]
    code_refs = [file_ref(root, path) for path in code_paths]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True).stdout.strip()
    output_relative = "inputs/actionbench-full128-snapshots/dataset-semantics.json"
    command = [
        sys.executable, "-m", "research_math.actionbench_dataset_semantics",
        "--contract", str(contract_path),
        "--population", str(population_path),
        "--snapshot-admission", str(snapshot_admission_path),
        "--dataset-root", str(dataset_root),
        "--output", "../" + output_relative,
    ]
    plan = native.make_plan(
        root, run_id=run_id, purpose="engineering", evidence_mode="developmental",
        jobs=[{
            "trial_id": "actionbench-full128-dataset-semantics",
            "command": command,
            "cwd": "actionmesh",
            "input_refs": input_refs,
            "code_refs": code_refs,
            "output_paths": [output_relative],
            "seed": 0,
            "group": "engineering",
            "arm_role": "dataset-semantics-admission",
        }],
        provenance={
            "git_revision": revision,
            "model_revision": "none",
            "data_revision": population["revision"],
            "environment_digest": hashlib.sha256(sys.version.encode()).hexdigest(),
            "data_refs": input_refs,
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
    plan_dir.mkdir(parents=True, exist_ok=False)
    native_path = plan_dir / "native.json"
    native_path.write_text(json.dumps(plan, indent=2) + "\n")
    outer = harness.make_plan(
        root, batch_id=run_id,
        tasks=[{
            "task_id": "actionbench-full128-dataset-semantics",
            "idea_id": "baseline-qualification",
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
                "exclusive_keys": ["actionbench-full128-dataset-semantics"],
            },
        }],
        limits={
            "total_wall_seconds": wall_seconds + 60,
            "window_seconds": wall_seconds + 60,
            "max_parallel_tasks": 1,
            "cpu_cores": cpu_cores,
            "ram_mib": ram_mib,
            "max_gpu_task_seconds": 0,
        },
    )
    outer_path = plan_dir / "harness.json"
    outer_path.write_text(json.dumps(outer, indent=2) + "\n")
    return outer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "skill-dir", "contract", "population",
                 "snapshot-admission", "dataset-root", "plan-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--wall-seconds", type=int, default=3600)
    parser.add_argument("--ram-mib", type=int, default=2048)
    parser.add_argument("--cpu-cores", type=int, default=1)
    args = parser.parse_args()
    outer = build_plan(
        args.root, skill_dir=args.skill_dir, contract_path=args.contract,
        population_path=args.population,
        snapshot_admission_path=args.snapshot_admission,
        dataset_root=args.dataset_root, plan_dir=args.plan_dir,
        run_id=args.run_id, wall_seconds=args.wall_seconds,
        ram_mib=args.ram_mib, cpu_cores=args.cpu_cores)
    print(json.dumps({
        "plan": str(args.plan_dir.resolve() / "harness.json"),
        "approved_plan_digest": outer["plan_digest"],
        "execution_started": False,
        "scope": "CPU-only byte-bound ActionBench tensor/camera/RGBA semantics",
        "scientific_effect_qualification": False,
        "dispatch_ready": False,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
