"""Emit the one-attempt C03 development tracked-query plan; never run it."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from research_math import c03_label_bank as producer


def file_ref(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    return {"path": path.relative_to(root).as_posix(), "sha256": producer.digest(path)}


def build_plans(root, *, skill_dir, inventory, source_root, weights_root,
                environment, plan_dir, run_id, gpu_uuid, points_per_uid,
                selection_seed, wall_seconds, cpu_cores, ram_mib):
    root, skill_dir, inventory, environment, plan_dir = map(
        lambda value: Path(value).resolve(),
        (root, skill_dir, inventory, environment, plan_dir))
    plan_dir.relative_to(root)
    if plan_dir.exists():
        raise FileExistsError("Preserve prior C03 label plan")
    if (not isinstance(gpu_uuid, str) or not gpu_uuid.startswith("GPU-")
            or any(character.isspace() or character == "," for character in gpu_uuid)):
        raise ValueError("One exact physical GPU UUID required")
    if type(wall_seconds) is not int or not 60 <= wall_seconds <= 26940:
        raise ValueError("Explicit bounded C03 development-probe budget required")
    if any(type(value) is not int or value < 1
           for value in (points_per_uid, cpu_cores, ram_mib)):
        raise ValueError("Positive integer resources/point count required")
    if not Path(source_root).is_absolute() or not Path(source_root).is_dir():
        raise ValueError("Existing absolute pinned ActionMesh source root required")
    if not Path(weights_root).is_absolute() or not Path(weights_root).is_dir():
        raise ValueError("Existing absolute pinned ActionMesh weights root required")
    scripts = skill_dir / "scripts"
    if not (scripts / "run_harness.py").is_file():
        raise ValueError("Complete installed research-autopilot required")
    record = producer.read_json(inventory)
    _, refs = producer.validate_inventory(root, record)
    env = producer.read_json(environment)
    if (env.get("gpu_uuid") != gpu_uuid or env.get("status") not in
            ("captured_native_environment", "accepted_native_environment")):
        raise ValueError("Current exact-GPU native environment record required")
    inputs = [file_ref(root, inventory), file_ref(root, environment)]
    inputs += refs
    unique = {}
    for ref in inputs:
        if ref["path"] in unique and unique[ref["path"]] != ref:
            raise ValueError("Conflicting C03 label input reference")
        unique[ref["path"]] = ref
    code_paths = [root / "actionmesh/research_math" / name for name in (
        "__init__.py", "c03_label_bank.py", "decoder_observer.py",
        "pipeline_decoder_observer.py", "native_context_runner.py")]
    code_paths += [root / "actionmesh/prepare_c03_label_bank.py"]
    command = [sys.executable, "-m", "research_math.c03_label_bank",
        "--root", str(root), "--inventory", str(inventory),
        "--source-root", str(Path(source_root)),
        "--weights-root", str(Path(weights_root)),
        "--output", "c03-development-label-output",
        "--points-per-uid", str(points_per_uid),
        "--selection-seed", str(selection_seed)]
    sys.path.insert(0, str(scripts))
    import run_experiments as native
    import run_harness as harness
    plan = native.make_plan(root, run_id=run_id, purpose="engineering",
        evidence_mode="developmental", jobs=[{
            "trial_id": "c03-tracked-query-label-bank", "command": command,
            "cwd": "actionmesh", "input_refs": list(unique.values()),
            "code_refs": [file_ref(root, path) for path in code_paths],
            "output_paths": [
                "actionmesh/c03-development-label-output/development-bank.npz",
                "actionmesh/c03-development-label-output/producer-report.json"],
            "seed": 42, "group": "development-label-producer",
            "arm_role": "tracked-gt-query-label-bank-no-confirmation"}],
        provenance={"git_revision": "exact code refs; no clean-tree claim",
            "model_revision": "environment/inventory-bound ActionMesh autoencoder",
            "data_revision": producer.digest(inventory),
            "environment_digest": producer.digest(environment)},
        limits={"max_attempts": 1, "max_development_trials": 1,
            "max_confirmation_trials": 0, "max_retries_per_trial": 0,
            "wall_time_seconds": wall_seconds,
            "attempt_timeout_seconds": wall_seconds})
    plan_dir.mkdir(parents=True)
    native_path = plan_dir / "native.json"
    native_path.write_text(json.dumps(plan, indent=2) + "\n")
    outer = harness.make_plan(root, batch_id=run_id, tasks=[{
        "task_id": "c03-tracked-query-label-bank",
        "idea_id": producer.CANDIDATE_ID, "depends_on": [], "priority": 1,
        "plan_ref": file_ref(root, native_path),
        "resources": {"cpu_cores": cpu_cores, "ram_mib": ram_mib,
            "gpu_count": 1, "gpu_peak_mib": None, "allow_gpu_share": False,
            "memory_profile_ref": None,
            "exclusive_keys": ["c03-development-label-producer"]}}],
        limits={"total_wall_seconds": wall_seconds,
            "window_seconds": wall_seconds + 1800, "max_parallel_tasks": 1,
            "cpu_cores": cpu_cores, "ram_mib": ram_mib,
            "max_gpu_task_seconds": wall_seconds},
        gpus={"uuids": [gpu_uuid], "safety_margin_mib": 1024,
               "max_tasks_per_gpu": 1})
    (plan_dir / "harness.json").write_text(json.dumps(outer, indent=2) + "\n")
    return plan, outer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "skill-dir", "inventory", "source-root", "weights-root",
                 "environment", "plan-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    for name in ("points-per-uid", "selection-seed", "wall-seconds",
                 "cpu-cores", "ram-mib"):
        parser.add_argument("--" + name, type=int, required=True)
    args = parser.parse_args(argv)
    _, outer = build_plans(args.root, skill_dir=args.skill_dir,
        inventory=args.inventory, source_root=args.source_root,
        weights_root=args.weights_root, environment=args.environment,
        plan_dir=args.plan_dir, run_id=args.run_id, gpu_uuid=args.gpu_uuid,
        points_per_uid=args.points_per_uid, selection_seed=args.selection_seed,
        wall_seconds=args.wall_seconds, cpu_cores=args.cpu_cores, ram_mib=args.ram_mib)
    print(json.dumps({"plan": str(args.plan_dir.resolve() / "harness.json"),
        "approved_plan_digest": outer["plan_digest"], "execution_started": False,
        "dispatch_ready": False, "scientific_admission": False,
        "source_delivery_status": "generated_unexecuted",
        "gpu_stop_remains_effective": True}))


if __name__ == "__main__":
    main()
