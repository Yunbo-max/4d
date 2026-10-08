"""Build C12's CPU-only retained three-role artifact plan; never execute it."""
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
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return {"path": path.relative_to(root).as_posix(), "sha256": value.hexdigest()}


def _finite(name: str, value: float, *, positive: bool = True) -> float:
    if isinstance(value, bool) or not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    return float(value)


def build_plans(root: Path, *, source_sequence: Path, run_id: str,
                beta: float, fixed_alpha: float, backtracking_factor: float,
                backtracking_max_steps: int, degeneracy_epsilon: float,
                root_tolerance: float, absolute_margin: float,
                relative_margin: float, arap_weight: float,
                temporal_weight: float, arap_iterations: int,
                cg_tolerance: float, cg_max_iterations: int,
                coordinate_lower: float, coordinate_upper: float,
                bounds_policy: str, max_artifact_bytes: int,
                plan_dir: Path, wall_seconds: int):
    import run_experiments as native
    import run_harness as harness

    root, source_sequence, plan_dir = (Path(root).resolve(),
                                       Path(source_sequence).resolve(),
                                       Path(plan_dir).resolve())
    source_sequence.relative_to(root); plan_dir.relative_to(root)
    if source_sequence.name != "sequence.npz":
        raise ValueError("Expected receipt-bound sequence.npz")
    for name, value in (("beta", beta), ("fixed_alpha", fixed_alpha),
                        ("backtracking_factor", backtracking_factor)):
        _finite(name, value)
    if not 0 < beta < 1 or not 0 < fixed_alpha <= 1 or not 0 < backtracking_factor < 1:
        raise ValueError("beta/backtracking must be in (0,1), fixed_alpha in (0,1]")
    for name, value in (("degeneracy_epsilon", degeneracy_epsilon),
                        ("root_tolerance", root_tolerance),
                        ("absolute_margin", absolute_margin),
                        ("relative_margin", relative_margin),
                        ("arap_weight", arap_weight),
                        ("temporal_weight", temporal_weight),
                        ("cg_tolerance", cg_tolerance)):
        _finite(name, value)
    for name, value in (("backtracking_max_steps", backtracking_max_steps),
                        ("arap_iterations", arap_iterations),
                        ("cg_max_iterations", cg_max_iterations)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if (not math.isfinite(coordinate_lower) or not math.isfinite(coordinate_upper)
            or coordinate_lower >= coordinate_upper):
        raise ValueError("Finite ordered coordinate bounds required")
    if bounds_policy not in ("preserve_and_report", "reject"):
        raise ValueError("Invalid bounds policy")
    if (isinstance(max_artifact_bytes, bool) or not isinstance(max_artifact_bytes, int)
            or max_artifact_bytes < 10240):
        raise ValueError("max_artifact_bytes must be an integer >=10240")
    if isinstance(wall_seconds, bool) or not isinstance(wall_seconds, int) or not 1 <= wall_seconds <= 26940:
        raise ValueError("CPU wall_seconds must be 1..26940")

    source_report = source_sequence.with_name("report.json")
    inputs = [file_ref(root, source_sequence), file_ref(root, source_report)]
    report = json.loads(source_report.read_text())
    if (report.get("status") != "completed"
            or report.get("sha256", {}).get("sequence.npz") != inputs[0]["sha256"]):
        raise ValueError("Completed matching source report required")
    uid, seed = report.get("uid"), report.get("seed")
    if not isinstance(uid, str) or not uid or Path(uid).name != uid or uid in (".", ".."):
        raise ValueError("Filesystem-safe native UID required")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Integer inference seed required")

    command = [
        sys.executable, "-m", "research_math.area_admission_candidate",
        "--source-sequence", str(source_sequence), "--output", "c12-area-output",
        "--uid", uid, "--expected-sequence-sha256", inputs[0]["sha256"],
        "--source-sequence-ref", inputs[0]["path"],
        "--source-report-ref", inputs[1]["path"],
        "--beta", str(beta), "--fixed-alpha", str(fixed_alpha),
        "--backtracking-factor", str(backtracking_factor),
        "--backtracking-max-steps", str(backtracking_max_steps),
        "--degeneracy-epsilon", str(degeneracy_epsilon),
        "--root-tolerance", str(root_tolerance),
        "--absolute-margin", str(absolute_margin),
        "--relative-margin", str(relative_margin),
        "--arap-weight", str(arap_weight), "--temporal-weight", str(temporal_weight),
        "--arap-iterations", str(arap_iterations),
        "--cg-tolerance", str(cg_tolerance),
        "--cg-max-iterations", str(cg_max_iterations),
        "--coordinate-bounds", str(coordinate_lower), str(coordinate_upper),
        "--bounds-policy", bounds_policy,
        "--max-artifact-bytes", str(max_artifact_bytes),
    ]
    code = [
        file_ref(root, root / "actionmesh/research_math/__init__.py"),
        file_ref(root, root / "actionmesh/research_math/area_admission_candidate.py"),
        file_ref(root, root / "actionmesh/research_math/protected_geometry_candidate.py"),
        file_ref(root, root / "actionmesh/research_ten/m01_elasticity.py"),
    ]
    environment = {
        "python_executable": sys.executable, "python": platform.python_version(),
        "numpy": importlib.metadata.version("numpy"),
        "scope": "CPU C12 common ARAP plus three scalar admission artifacts; no model/scorer/GPU",
        "scientific_limit": "projected oriented area is not global injectivity or collision safety",
    }
    environment_digest = hashlib.sha256(json.dumps(
        environment, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    plan = native.make_plan(
        root, run_id=run_id, jobs=[{
            "trial_id": "prepare-c12-area-admission-candidate",
            "command": command, "cwd": "actionmesh", "input_refs": inputs,
            "code_refs": code,
            "output_paths": ["actionmesh/c12-area-output/candidate.json",
                             "actionmesh/c12-area-output/artifact-archive.json",
                             "actionmesh/c12-area-output/artifact.tar"],
            "seed": seed, "group": "candidate-artifacts-only",
            "arm_role": "three-shared-update-admission-artifacts-no-scorer-no-admission",
        }],
        provenance={
            "git_revision": "source files pinned in code_refs; no implicit clean-tree claim",
            "model_revision": "no model loaded; receipt-bound native sequence only",
            "data_revision": inputs[0]["sha256"],
            "environment_digest": environment_digest,
        },
        limits={"max_attempts": 1, "max_development_trials": 1,
                "max_confirmation_trials": 0, "max_retries_per_trial": 0,
                "wall_time_seconds": wall_seconds,
                "attempt_timeout_seconds": wall_seconds})
    plan_dir.mkdir(parents=True, exist_ok=False)
    native_path = plan_dir / "native.json"
    native_path.write_text(json.dumps(plan, indent=2) + "\n")
    (plan_dir / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    outer = harness.make_plan(
        root, batch_id=run_id, tasks=[{
            "task_id": "prepare-c12-area-admission-candidate",
            "idea_id": "4d-math-20261006-c12", "depends_on": [], "priority": 1,
            "plan_ref": file_ref(root, native_path),
            "resources": {"cpu_cores": 1, "ram_mib": 8192, "gpu_count": 0,
                          "gpu_peak_mib": None, "allow_gpu_share": False,
                          "memory_profile_ref": None, "exclusive_keys": []},
        }],
        limits={"total_wall_seconds": wall_seconds + 60,
                "window_seconds": wall_seconds + 60, "max_parallel_tasks": 1,
                "cpu_cores": 1, "ram_mib": 8192, "max_gpu_task_seconds": 0})
    (plan_dir / "harness.json").write_text(json.dumps(outer, indent=2) + "\n")
    return plan, outer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "skill-dir", "source-sequence", "plan-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--fixed-alpha", type=float, required=True)
    parser.add_argument("--backtracking-factor", type=float, required=True)
    parser.add_argument("--backtracking-max-steps", type=int, required=True)
    parser.add_argument("--degeneracy-epsilon", type=float, required=True)
    parser.add_argument("--root-tolerance", type=float, required=True)
    parser.add_argument("--absolute-margin", type=float, required=True)
    parser.add_argument("--relative-margin", type=float, required=True)
    parser.add_argument("--arap-weight", type=float, required=True)
    parser.add_argument("--temporal-weight", type=float, required=True)
    parser.add_argument("--arap-iterations", type=int, required=True)
    parser.add_argument("--cg-tolerance", type=float, required=True)
    parser.add_argument("--cg-max-iterations", type=int, required=True)
    parser.add_argument("--coordinate-bounds", nargs=2, type=float, required=True)
    parser.add_argument("--bounds-policy", choices=("preserve_and_report", "reject"), required=True)
    parser.add_argument("--max-artifact-bytes", type=int, required=True)
    parser.add_argument("--wall-seconds", type=int, required=True)
    args = parser.parse_args()
    scripts = args.skill_dir.resolve() / "scripts"
    if not (scripts / "run_harness.py").is_file():
        parser.error("Full installed research-autopilot skill required")
    sys.path.insert(0, str(scripts))
    _, outer = build_plans(
        args.root, source_sequence=args.source_sequence, run_id=args.run_id,
        beta=args.beta, fixed_alpha=args.fixed_alpha,
        backtracking_factor=args.backtracking_factor,
        backtracking_max_steps=args.backtracking_max_steps,
        degeneracy_epsilon=args.degeneracy_epsilon,
        root_tolerance=args.root_tolerance,
        absolute_margin=args.absolute_margin,
        relative_margin=args.relative_margin,
        arap_weight=args.arap_weight, temporal_weight=args.temporal_weight,
        arap_iterations=args.arap_iterations, cg_tolerance=args.cg_tolerance,
        cg_max_iterations=args.cg_max_iterations,
        coordinate_lower=args.coordinate_bounds[0],
        coordinate_upper=args.coordinate_bounds[1],
        bounds_policy=args.bounds_policy,
        max_artifact_bytes=args.max_artifact_bytes,
        plan_dir=args.plan_dir, wall_seconds=args.wall_seconds)
    print(json.dumps({"plan": str(args.plan_dir.resolve() / "harness.json"),
                      "approved_plan_digest": outer["plan_digest"],
                      "execution_started": False,
                      "candidate_id": "4d-math-20261006-c12",
                      "native_qualified": False, "scientific_admission": False,
                      "local_method_verified": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
