"""Build C11's CPU-only three-role retained artifact plan.

The builder emits one zero-retry plan for the existing research-autopilot
harness.  It never runs the candidate, model, scorer, tests or GPU workload.
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


def file_ref(root: Path, path: Path) -> dict:
    root, path = Path(root).resolve(), Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": path.relative_to(root).as_posix(),
            "sha256": digest.hexdigest()}


def _positive(name: str, value) -> float:
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def build_plans(root: Path, *, source_sequence: Path, run_id: str,
                stretch_lower: float, stretch_upper: float,
                elastic_weight: float, degeneracy_epsilon: float,
                absolute_tolerance: float, relative_tolerance: float,
                max_iterations: int, coordinate_lower: float,
                coordinate_upper: float, bounds_policy: str,
                max_artifact_bytes: int, plan_dir: Path,
                wall_seconds: int):
    import run_experiments as native
    import run_harness as harness

    root, source_sequence = Path(root).resolve(), Path(source_sequence).resolve()
    plan_dir = Path(plan_dir).resolve()
    source_sequence.relative_to(root)
    plan_dir.relative_to(root)
    if source_sequence.name != "sequence.npz":
        raise ValueError("Expected receipt-bound sequence.npz")
    stretch_lower = _positive("stretch_lower", stretch_lower)
    stretch_upper = _positive("stretch_upper", stretch_upper)
    if not stretch_lower <= 1.0 <= stretch_upper or stretch_lower >= stretch_upper:
        raise ValueError("Stretch interval must contain one")
    elastic_weight = _positive("elastic_weight", elastic_weight)
    degeneracy_epsilon = _positive("degeneracy_epsilon", degeneracy_epsilon)
    absolute_tolerance = _positive("absolute_tolerance", absolute_tolerance)
    relative_tolerance = _positive("relative_tolerance", relative_tolerance)
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if (not math.isfinite(coordinate_lower) or not math.isfinite(coordinate_upper)
            or coordinate_lower >= coordinate_upper):
        raise ValueError("Finite ordered coordinate bounds required")
    if bounds_policy not in ("preserve_and_report", "reject"):
        raise ValueError("Invalid bounds_policy")
    if (isinstance(max_artifact_bytes, bool) or not isinstance(max_artifact_bytes, int)
            or max_artifact_bytes < 10240):
        raise ValueError("max_artifact_bytes must be an integer >= 10240")
    if isinstance(wall_seconds, bool) or not 1 <= wall_seconds <= 26940:
        raise ValueError("CPU budget must be 1..26940 seconds")

    source_report = source_sequence.with_name("report.json")
    inputs = [file_ref(root, source_sequence), file_ref(root, source_report)]
    report = json.loads(source_report.read_text())
    if (report.get("status") != "completed"
            or report.get("sha256", {}).get("sequence.npz") != inputs[0]["sha256"]):
        raise ValueError("Completed source report and current sequence hash required")
    uid, seed = report.get("uid"), report.get("seed")
    if not isinstance(uid, str) or not uid or Path(uid).name != uid or uid in (".", ".."):
        raise ValueError("Filesystem-safe native UID required")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Integer inference seed required")

    command = [
        sys.executable, "-m", "research_math.strain_projection_candidate",
        "--source-sequence", str(source_sequence),
        "--output", "c11-strain-output", "--uid", uid,
        "--expected-sequence-sha256", inputs[0]["sha256"],
        "--source-sequence-ref", inputs[0]["path"],
        "--source-report-ref", inputs[1]["path"],
        "--stretch-bounds", str(stretch_lower), str(stretch_upper),
        "--elastic-weight", str(elastic_weight),
        "--degeneracy-epsilon", str(degeneracy_epsilon),
        "--absolute-tolerance", str(absolute_tolerance),
        "--relative-tolerance", str(relative_tolerance),
        "--max-iterations", str(max_iterations),
        "--coordinate-bounds", str(coordinate_lower), str(coordinate_upper),
        "--bounds-policy", bounds_policy,
        "--max-artifact-bytes", str(max_artifact_bytes),
    ]
    code = [
        file_ref(root, root / "actionmesh/research_math/__init__.py"),
        file_ref(root, root / "actionmesh/research_math/strain_projection_candidate.py"),
        file_ref(root, root / "actionmesh/research_math/integrable_gradient_candidate.py"),
        file_ref(root, root / "actionmesh/research_math/corotational_residual_candidate.py"),
    ]
    environment = {
        "python_executable": sys.executable,
        "python": platform.python_version(),
        "numpy": importlib.metadata.version("numpy"),
        "scope": "CPU C11 square-map/polar projection artifacts only; no model/scorer/GPU",
        "lift": "shared matrix-free pinned weighted Poisson solve",
    }
    environment_digest = hashlib.sha256(json.dumps(
        environment, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    outputs = [
        "actionmesh/c11-strain-output/candidate.json",
        "actionmesh/c11-strain-output/artifact-archive.json",
        "actionmesh/c11-strain-output/artifact.tar",
    ]
    plan = native.make_plan(
        root, run_id=run_id,
        jobs=[{
            "trial_id": "prepare-c11-strain-projection-candidate",
            "command": command, "cwd": "actionmesh",
            "input_refs": inputs, "code_refs": code,
            "output_paths": outputs, "seed": seed,
            "group": "candidate-artifacts-only",
            "arm_role": "three-shared-map-lift-artifacts-no-scorer-no-admission",
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
            "task_id": "prepare-c11-strain-projection-candidate",
            "idea_id": "4d-math-20261006-c11", "depends_on": [], "priority": 1,
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
            "max_parallel_tasks": 1, "cpu_cores": 1,
            "ram_mib": 8192, "max_gpu_task_seconds": 0,
        })
    (plan_dir / "harness.json").write_text(json.dumps(outer, indent=2) + "\n")
    return plan, outer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "skill-dir", "source-sequence", "plan-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stretch-bounds", nargs=2, type=float, required=True,
                        metavar=("LOWER", "UPPER"))
    parser.add_argument("--elastic-weight", type=float, required=True)
    parser.add_argument("--degeneracy-epsilon", type=float, required=True)
    parser.add_argument("--absolute-tolerance", type=float, required=True)
    parser.add_argument("--relative-tolerance", type=float, required=True)
    parser.add_argument("--max-iterations", type=int, required=True)
    parser.add_argument("--coordinate-bounds", nargs=2, type=float, required=True,
                        metavar=("LOWER", "UPPER"))
    parser.add_argument("--bounds-policy", choices=("preserve_and_report", "reject"),
                        required=True)
    parser.add_argument("--max-artifact-bytes", type=int, required=True)
    parser.add_argument("--wall-seconds", type=int, required=True)
    args = parser.parse_args()
    scripts = args.skill_dir.resolve() / "scripts"
    if not (scripts / "run_harness.py").is_file():
        parser.error("Full installed research-autopilot skill required")
    sys.path.insert(0, str(scripts))
    _, outer = build_plans(
        args.root, source_sequence=args.source_sequence,
        run_id=args.run_id, stretch_lower=args.stretch_bounds[0],
        stretch_upper=args.stretch_bounds[1],
        elastic_weight=args.elastic_weight,
        degeneracy_epsilon=args.degeneracy_epsilon,
        absolute_tolerance=args.absolute_tolerance,
        relative_tolerance=args.relative_tolerance,
        max_iterations=args.max_iterations,
        coordinate_lower=args.coordinate_bounds[0],
        coordinate_upper=args.coordinate_bounds[1],
        bounds_policy=args.bounds_policy,
        max_artifact_bytes=args.max_artifact_bytes,
        plan_dir=args.plan_dir, wall_seconds=args.wall_seconds)
    print(json.dumps({
        "plan": str(args.plan_dir.resolve() / "harness.json"),
        "approved_plan_digest": outer["plan_digest"],
        "execution_started": False, "candidate_id": "4d-math-20261006-c11",
        "native_qualified": False, "scientific_admission": False,
        "local_method_verified": False,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
