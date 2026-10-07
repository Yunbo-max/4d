"""Build, but never execute, a CPU-only full128 queue-pricing plan."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from prepare_complete_unit_admission import reference_closure
from research_math.actionbench_queue_pricing import (
    build_pricing_manifest,
    runner_inventory_refs,
)
from research_math.control_scoring import file_ref, resolve_ref


SOURCE_NAMES = (
    "research_math/__init__.py",
    "research_math/control_scoring.py",
    "research_math/complete_unit_contract.py",
    "research_math/complete_unit_admission.py",
    "research_math/actionbench_queue_pricing.py",
    "prepare_complete_unit_admission.py",
    "prepare_actionbench_queue_pricing.py",
)


def pricing_output_path(root: Path) -> Path:
    return Path(root).resolve() / "inputs/actionbench-full128-queue/pricing.json"


def validate_source_records(admission: dict) -> None:
    if (admission.get("kind") != "actionbench-complete-unit-admission" or
            admission.get("version") != "1.0.0" or
            admission.get("status") != "admitted_engineering_complete_unit" or
            admission.get("eligible_for_queue_pricing") is not True or
            admission.get("scientific_effect_qualification") is not False or
            admission.get("native_scientific_qualification") is not False or
            admission.get("candidate_methods_tested") is not False or
            admission.get("queue_approved") is not False or
            admission.get("queue_generated") is not False):
        raise ValueError("Completed non-scientific complete-unit admission required")


def build_plans(root: Path, *, contract_path: Path, admission_path: Path,
                population_path: Path, output: Path, skill_dir: Path,
                plan_dir: Path, run_id: str, wall_seconds: int,
                ram_mib: int, cpu_cores: int):
    root = Path(root).resolve()
    contract_path = Path(contract_path).resolve()
    admission_path = Path(admission_path).resolve()
    population_path = Path(population_path).resolve()
    output = Path(output).resolve()
    plan_dir = Path(plan_dir).resolve()
    for path in (contract_path, admission_path, population_path, output, plan_dir):
        path.relative_to(root)
    if contract_path != root / (
            "docs/research-math-20261006/"
            "actionbench-full128-queue-pricing-contract.json"):
        raise ValueError("Canonical full128 queue-pricing contract required")
    if admission_path != root / (
            "inputs/complete-unit-admissions/complete-lowram-r7.json"):
        raise ValueError("Canonical complete-unit admission required")
    if population_path != root / (
            "actionmesh/research_overnight/assets/actionbench_population.json"):
        raise ValueError("Canonical ActionBench population required")
    if output != pricing_output_path(root):
        raise ValueError("Canonical full128 queue-pricing output required")
    if output.exists():
        raise FileExistsError("Queue pricing output is single-use")
    if not 1 <= wall_seconds <= 900 or ram_mib < 1 or cpu_cores < 1:
        raise ValueError("Bounded positive CPU pricing resources required")
    scripts = Path(skill_dir).resolve() / "scripts"
    if not (scripts / "run_harness.py").is_file():
        raise ValueError("Complete installed research-autopilot skill required")
    sys.path.insert(0, str(scripts))
    import run_experiments as native
    import run_harness as harness

    contract = json.loads(contract_path.read_text())
    admission = json.loads(admission_path.read_text())
    population = json.loads(population_path.read_text())
    validate_source_records(admission)
    # Fail before plan creation if the admitted measurement, population or any
    # nested evidence reference cannot produce the deterministic frozen price.
    build_pricing_manifest(root, contract, admission, population)

    reproduction_ref = contract.get("full128_reproduction_contract_ref")
    admission_contract_ref = contract.get("source_admission", {}).get("contract_ref")
    if not isinstance(reproduction_ref, dict) or not isinstance(admission_contract_ref, dict):
        raise ValueError("Pricing contract reference closure required")
    seed_paths = [
        contract_path,
        admission_path,
        population_path,
        root / "plans/complete-lowram-r7/harness.json",
        root / "plans/complete-lowram-r7/native.json",
        resolve_ref(root, reproduction_ref),
        resolve_ref(root, admission_contract_ref),
    ]
    seed_paths.extend(resolve_ref(root, ref) for ref in
                      runner_inventory_refs(root, admission))
    input_refs = reference_closure(root, seed_paths)

    sources = [root / "actionmesh" / name for name in SOURCE_NAMES]
    missing = [path for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing queue-pricing source: " + str(missing[0]))
    code_refs = [file_ref(root, path) for path in sources]
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
    refs = {ref["path"]: ref for ref in input_refs}
    dirty_ref = file_ref(root, dirty_path)
    refs[dirty_ref["path"]] = dirty_ref
    command = [
        sys.executable, "-m", "research_math.actionbench_queue_pricing",
        "--root", "..",
        "--contract", str(contract_path),
        "--admission", str(admission_path),
        "--population", str(population_path),
        "--output", str(output),
    ]
    plan = native.make_plan(
        root, run_id=run_id, purpose="engineering", evidence_mode="developmental",
        jobs=[{
            "trial_id": "price-full128-queue-from-admitted-unit",
            "command": command,
            "cwd": "actionmesh",
            "input_refs": list(refs.values()),
            "code_refs": code_refs,
            "output_paths": [output.relative_to(root).as_posix()],
            "seed": 42,
            "group": "engineering",
            "arm_role": "queue-pricing-only",
        }],
        provenance={
            "git_revision": revision,
            "git_refs": [dirty_ref],
            "model_revision": "bound by admitted complete-unit evidence",
            "data_revision": contract["population"]["revision"],
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
            "task_id": "full128-queue-pricing",
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
                "exclusive_keys": ["full128-queue-pricing"],
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
    for name in ("root", "contract", "admission", "population", "output",
                 "skill-dir", "plan-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    for name in ("wall-seconds", "ram-mib", "cpu-cores"):
        parser.add_argument("--" + name, type=int, required=True)
    args = parser.parse_args()
    plan = build_plans(
        args.root,
        contract_path=args.contract,
        admission_path=args.admission,
        population_path=args.population,
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
        "gpu_count": 0,
        "queue_priced": False,
        "queue_approved": False,
        "queue_generated": False,
        "dispatch_ready": False,
        "scope": "queue pricing only; no GPU queue or scientific claim",
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
