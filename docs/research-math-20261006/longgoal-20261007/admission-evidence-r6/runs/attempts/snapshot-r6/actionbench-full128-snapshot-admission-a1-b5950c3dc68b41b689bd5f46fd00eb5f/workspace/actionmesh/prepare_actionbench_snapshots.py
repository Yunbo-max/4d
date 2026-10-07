"""Emit, but never execute, the CPU-only ActionBench snapshot admission plan."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from research_math.snapshot_admission import SNAPSHOT_KEYS


def file_ref(root: Path, path: Path):
    path = Path(path).resolve()
    return {"path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def build_plan(root: Path, *, skill_dir: Path, contract_path: Path,
               population_path: Path, source_root: Path, snapshot_roots: dict,
               plan_dir: Path, run_id: str, wall_seconds: int,
               ram_mib: int, cpu_cores: int):
    root = Path(root).resolve()
    skill_dir = Path(skill_dir).resolve()
    contract_path = Path(contract_path).resolve()
    population_path = Path(population_path).resolve()
    plan_dir = Path(plan_dir).resolve()
    for path in (contract_path, population_path, plan_dir):
        path.relative_to(root)
    source_root = Path(source_root).resolve()
    snapshot_roots = {key: Path(value).resolve()
                      for key, value in snapshot_roots.items()}
    if set(snapshot_roots) != set(SNAPSHOT_KEYS):
        raise ValueError("Exactly five snapshot roots are required")
    if not source_root.is_dir():
        raise FileNotFoundError("Official ActionMesh source root missing")
    for key, path in snapshot_roots.items():
        if not path.is_dir():
            raise FileNotFoundError("Snapshot root missing for " + key)
    if not 60 <= wall_seconds <= 3600:
        raise ValueError("Snapshot admission wall limit must be 60..3600 seconds")
    if ram_mib < 256 or cpu_cores < 1:
        raise ValueError("Positive bounded CPU/RAM admission required")
    scripts = skill_dir / "scripts"
    if not (scripts / "run_harness.py").is_file():
        raise ValueError("Complete installed research-autopilot skill required")
    sys.path.insert(0, str(scripts))
    import run_experiments as native
    import run_harness as harness

    contract = json.loads(contract_path.read_text())
    population = json.loads(population_path.read_text())
    if (contract.get("snapshots", {}).get("dataset", {}).get("revision") !=
            population.get("revision")):
        raise ValueError("Contract and population revisions differ")
    population_ref = file_ref(root, population_path)
    if (contract.get("dataset_layout", {}).get("uid_source_ref") != population_ref):
        raise ValueError("Contract does not bind the exact population file")
    code_paths = [
        root / "actionmesh/research_math/__init__.py",
        root / "actionmesh/research_math/snapshot_admission.py",
        root / "actionmesh/prepare_actionbench_snapshots.py",
    ]
    input_refs = [file_ref(root, contract_path), population_ref]
    code_refs = [file_ref(root, path) for path in code_paths]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True).stdout.strip()
    output_relative = "inputs/actionbench-full128-snapshots/admission.json"
    command = [
        sys.executable, "-m", "research_math.snapshot_admission",
        "--contract", str(contract_path),
        "--population", str(population_path),
        "--source-root", str(source_root),
    ]
    for key in SNAPSHOT_KEYS:
        command += ["--" + key + "-root", str(snapshot_roots[key])]
    command += ["--output", "../" + output_relative]
    plan = native.make_plan(
        root, run_id=run_id, purpose="engineering", evidence_mode="developmental",
        jobs=[{
            "trial_id": "actionbench-full128-snapshot-admission",
            "command": command,
            "cwd": "actionmesh",
            "input_refs": input_refs,
            "code_refs": code_refs,
            "output_paths": [output_relative],
            "seed": 0,
            "group": "engineering",
            "arm_role": "snapshot-admission",
        }],
        provenance={
            "git_revision": revision,
            "model_revision": "five immutable Hub revisions declared by contract",
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
            "task_id": "actionbench-full128-snapshot-admission",
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
                "exclusive_keys": ["actionbench-full128-snapshot-admission"],
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
    for name in ("root", "skill-dir", "contract", "population", "source-root",
                 "plan-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    for key in SNAPSHOT_KEYS:
        parser.add_argument("--" + key + "-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--wall-seconds", type=int, default=1800)
    parser.add_argument("--ram-mib", type=int, default=2048)
    parser.add_argument("--cpu-cores", type=int, default=1)
    args = parser.parse_args()
    roots = {key: getattr(args, key + "_root") for key in SNAPSHOT_KEYS}
    outer = build_plan(
        args.root, skill_dir=args.skill_dir, contract_path=args.contract,
        population_path=args.population, source_root=args.source_root,
        snapshot_roots=roots, plan_dir=args.plan_dir, run_id=args.run_id,
        wall_seconds=args.wall_seconds, ram_mib=args.ram_mib,
        cpu_cores=args.cpu_cores)
    print(json.dumps({
        "plan": str(args.plan_dir.resolve() / "harness.json"),
        "approved_plan_digest": outer["plan_digest"],
        "execution_started": False,
        "scope": "CPU-only immutable snapshot byte/revision/file-closure admission",
        "scientific_effect_qualification": False,
        "dispatch_ready": False,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
