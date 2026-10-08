"""Build C04's CPU-only three-arm artifact plan.

The builder only emits plans for the existing research-autopilot harness.  It
does not execute the target generator, candidate, tests, model or scorer and it
does not authorize or resume GPU work.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np


def file_ref(root: Path, path: Path) -> dict:
    root, path = Path(root).resolve(), Path(path).resolve()
    relative = path.relative_to(root).as_posix()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": relative, "sha256": digest.hexdigest()}


def _positive(name: str, value: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _unit_interval(name: str, value: float) -> float:
    value = _positive(name, value)
    if value > 1.0:
        raise ValueError(f"{name} must be at most one")
    return value


def build_plans(root: Path, *, source_sequence: Path, run_id: str,
                parameters: dict, coordinate_lower: float,
                coordinate_upper: float, bounds_policy: str,
                max_artifact_bytes: int, plan_dir: Path, wall_seconds: int):
    import run_experiments as native
    import run_harness as harness

    root = Path(root).resolve()
    source_sequence = Path(source_sequence).resolve()
    plan_dir = Path(plan_dir).resolve()
    plan_dir.relative_to(root)
    if source_sequence.name != "sequence.npz":
        raise ValueError("Expected receipt-bound sequence.npz")
    from research_math.robust_motion_candidate import validate_parameters
    parameters = validate_parameters(parameters)
    if (not math.isfinite(coordinate_lower) or not math.isfinite(coordinate_upper)
            or coordinate_lower >= coordinate_upper):
        raise ValueError("Finite ordered coordinate bounds required")
    if bounds_policy not in ("preserve_and_report", "reject"):
        raise ValueError("Invalid bounds_policy")
    if (isinstance(max_artifact_bytes, bool) or not isinstance(max_artifact_bytes, int)
            or max_artifact_bytes < 10240):
        raise ValueError("max_artifact_bytes must be an integer >= 10240")
    if isinstance(wall_seconds, bool) or not 1 <= wall_seconds <= 26940:
        raise ValueError("CPU budget must be 1..26940 seconds so outer total remains <=27000")

    source_report = source_sequence.with_name("report.json")
    inputs = [file_ref(root, source_sequence), file_ref(root, source_report)]
    source = json.loads(source_report.read_text())
    if (source.get("status") != "completed"
            or source.get("sha256", {}).get("sequence.npz") != inputs[0]["sha256"]):
        raise ValueError("Completed source report and current sequence hash required")
    uid, seed = source.get("uid"), source.get("seed")
    if not isinstance(uid, str) or not uid or Path(uid).name != uid or uid in (".", ".."):
        raise ValueError("Filesystem-safe native UID required")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Integer inference seed required")

    command = [
        sys.executable, "-m", "research_math.robust_motion_candidate",
        "--source-sequence", str(source_sequence),
        "--output", "c04-robust-output", "--uid", uid,
        "--expected-sequence-sha256", inputs[0]["sha256"],
        "--source-sequence-ref", inputs[0]["path"],
        "--source-report-ref", inputs[1]["path"],
        "--coordinate-bounds", str(coordinate_lower), str(coordinate_upper),
        "--bounds-policy", bounds_policy,
        "--max-artifact-bytes", str(max_artifact_bytes),
    ]
    for key, value in parameters.items():
        command.extend(("--" + key.replace("_", "-"), str(value)))
    code_paths = (
        "actionmesh/research_math/__init__.py",
        "actionmesh/research_math/robust_motion_candidate.py",
        "actionmesh/research_math/protected_geometry_candidate.py",
        "actionmesh/research_ten/__init__.py",
        "actionmesh/research_ten/m01_elasticity.py",
    )
    code = [file_ref(root, root / path) for path in code_paths]
    environment = {
        "python_executable": sys.executable,
        "python": platform.python_version(),
        "numpy": importlib.metadata.version("numpy"),
        "scope": "CPU C04 common-target three-arm artifact only; no model/scorer/GPU",
        "sparse_contract": "matrix-free ARAP plus diagonal conic dual; no dense vertex matrix",
    }
    environment_digest = hashlib.sha256(json.dumps(
        environment, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    outputs = [
        "actionmesh/c04-robust-output/candidate.json",
        "actionmesh/c04-robust-output/artifact-archive.json",
        "actionmesh/c04-robust-output/artifact.tar",
    ]
    plan = native.make_plan(
        root, run_id=run_id,
        jobs=[{
            "trial_id": "prepare-c04-robust-motion-candidate",
            "command": command, "cwd": "actionmesh",
            "input_refs": inputs, "code_refs": code, "output_paths": outputs,
            "seed": seed, "group": "candidate-artifacts-only",
            "arm_role": "three-common-target-artifacts-no-scorer-no-admission",
        }],
        provenance={
            "git_revision": "source files pinned in code_refs; no implicit clean-tree claim",
            "model_revision": "no model loaded; receipt-bound native sequence only",
            "data_revision": inputs[0]["sha256"],
            "environment_digest": environment_digest,
        },
        limits={
            "max_attempts": 1, "max_development_trials": 1,
            "max_confirmation_trials": 0, "max_retries_per_trial": 0,
            "wall_time_seconds": wall_seconds,
            "attempt_timeout_seconds": wall_seconds,
        })
    plan_dir.mkdir(parents=True, exist_ok=False)
    native_path = plan_dir / "native.json"
    native_path.write_text(json.dumps(plan, indent=2) + "\n")
    (plan_dir / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    outer = harness.make_plan(
        root, batch_id=run_id,
        tasks=[{
            "task_id": "prepare-c04-robust-motion-candidate",
            "idea_id": "4d-math-20261006-c04", "depends_on": [], "priority": 1,
            "plan_ref": file_ref(root, native_path),
            "resources": {
                "cpu_cores": 1, "ram_mib": 8192, "gpu_count": 0,
                "gpu_peak_mib": None, "allow_gpu_share": False,
                "memory_profile_ref": None, "exclusive_keys": [],
            },
        }],
        limits={
            "total_wall_seconds": wall_seconds + 60,
            "window_seconds": wall_seconds + 60,
            "max_parallel_tasks": 1, "cpu_cores": 1, "ram_mib": 8192,
            "max_gpu_task_seconds": 0,
        })
    (plan_dir / "harness.json").write_text(json.dumps(outer, indent=2) + "\n")
    return plan, outer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "skill-dir", "source-sequence", "plan-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    from research_math.robust_motion_candidate import FLOAT_PARAMETERS, INTEGER_PARAMETERS
    for name in FLOAT_PARAMETERS:
        parser.add_argument("--"+name.replace("_","-"),type=float,required=True)
    for name in INTEGER_PARAMETERS:
        parser.add_argument("--"+name.replace("_","-"),type=int,required=True)
    parser.add_argument("--coordinate-bounds", nargs=2, type=float, required=True,
                        metavar=("LOWER", "UPPER"))
    parser.add_argument("--bounds-policy", required=True,
                        choices=("preserve_and_report", "reject"))
    parser.add_argument("--max-artifact-bytes", type=int, required=True)
    parser.add_argument("--wall-seconds", type=int, required=True)
    args = parser.parse_args()
    scripts = args.skill_dir.resolve() / "scripts"
    if not (scripts / "run_harness.py").is_file():
        parser.error("Full installed research-autopilot skill required")
    sys.path.insert(0, str(scripts))
    _, outer = build_plans(
        args.root, source_sequence=args.source_sequence, run_id=args.run_id,
        parameters={key:getattr(args,key) for key in FLOAT_PARAMETERS+INTEGER_PARAMETERS},
        coordinate_lower=args.coordinate_bounds[0],
        coordinate_upper=args.coordinate_bounds[1],
        bounds_policy=args.bounds_policy,
        max_artifact_bytes=args.max_artifact_bytes, plan_dir=args.plan_dir,
        wall_seconds=args.wall_seconds)
    print(json.dumps({
        "plan": str(args.plan_dir.resolve() / "harness.json"),
        "approved_plan_digest": outer["plan_digest"],
        "execution_started": False, "candidate_id": "4d-math-20261006-c04",
        "native_qualified": False, "scientific_admission": False,
        "local_method_verified": False,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
