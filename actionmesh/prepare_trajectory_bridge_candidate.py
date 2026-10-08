"""Build C08's CPU-only four-role geometry trajectory-bridge artifact plan."""
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
    return {"path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def build_plans(root: Path, *, source_sequence: Path, run_id: str,
                epsilon: float, neighbors: int, tolerance: float,
                max_iterations: int, smoothing_strength: float,
                coordinate_lower: float, coordinate_upper: float,
                bounds_policy: str, max_artifact_bytes: int,
                plan_dir: Path, wall_seconds: int, ram_mib: int):
    import run_experiments as native
    import run_harness as harness

    root, source_sequence, plan_dir = (Path(root).resolve(),
                                       Path(source_sequence).resolve(),
                                       Path(plan_dir).resolve())
    source_sequence.relative_to(root); plan_dir.relative_to(root)
    if source_sequence.name != "sequence.npz":
        raise ValueError("Expected receipt-bound sequence.npz")
    for name, value in (("epsilon", epsilon), ("tolerance", tolerance),
                        ("smoothing_strength", smoothing_strength)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(name + " must be finite and positive")
    for name, value in (("neighbors", neighbors), ("max_iterations", max_iterations),
                        ("max_artifact_bytes", max_artifact_bytes),
                        ("wall_seconds", wall_seconds), ("ram_mib", ram_mib)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(name + " must be a positive integer")
    if (wall_seconds > 26940 or ram_mib > 65536 or max_artifact_bytes < 10240
            or not math.isfinite(coordinate_lower)
            or not math.isfinite(coordinate_upper) or coordinate_lower >= coordinate_upper
            or bounds_policy not in ("preserve_and_report", "reject")):
        raise ValueError("C08 bounds/resources outside frozen source envelope")
    source_report = source_sequence.with_name("report.json")
    inputs = [file_ref(root, source_sequence), file_ref(root, source_report)]
    report = json.loads(source_report.read_text())
    if (report.get("status") != "completed"
            or report.get("sha256", {}).get("sequence.npz") != inputs[0]["sha256"]
            or not isinstance(report.get("uid"), str)
            or isinstance(report.get("seed"), bool) or not isinstance(report.get("seed"), int)):
        raise ValueError("Completed matching native source pair required")
    uid, seed = report["uid"], report["seed"]
    if not uid or Path(uid).name != uid or uid in (".", ".."):
        raise ValueError("Filesystem-safe UID required")
    code_paths = (
        "actionmesh/research_math/__init__.py",
        "actionmesh/research_math/trajectory_bridge_candidate.py",
        "actionmesh/research_math/area_transport_candidate.py",
        "actionmesh/research_math/protected_geometry_candidate.py",
    )
    code = [file_ref(root, root / path) for path in code_paths]
    environment = {
        "python_executable": sys.executable, "python": platform.python_version(),
        "numpy": importlib.metadata.version("numpy"),
        "scipy": importlib.metadata.version("scipy"),
        "scope": "CPU geometry-only C08 artifacts; no model, GT, scorer or GPU",
        "feature_source": "predicted_mesh_geometry_only_no_gt_attention_or_external_correspondence",
    }
    plan_dir.mkdir(parents=True, exist_ok=False)
    (plan_dir / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    producer_path = plan_dir / "producer-provenance.json"
    producer_path.write_text(json.dumps({
        "kind": "c08-candidate-producer-provenance", "version": 1,
        "code_refs": code, "environment": environment, "source_refs": inputs,
        "execution_contract": {"gpu_count": 0, "max_attempts": 1,
                               "max_retries_per_trial": 0},
    }, indent=2) + "\n")
    producer_ref = file_ref(root, producer_path); inputs.append(producer_ref)
    command = [
        sys.executable, "-m", "research_math.trajectory_bridge_candidate",
        "--source-sequence", str(source_sequence), "--output", "c08-trajectory-bridge-output",
        "--uid", uid, "--expected-sequence-sha256", inputs[0]["sha256"],
        "--source-sequence-ref", inputs[0]["path"],
        "--source-report-ref", inputs[1]["path"],
        "--producer-provenance", str(producer_path),
        "--producer-provenance-ref", producer_ref["path"],
        "--expected-producer-provenance-sha256", producer_ref["sha256"],
        "--epsilon", str(epsilon), "--neighbors", str(neighbors),
        "--tolerance", str(tolerance), "--max-iterations", str(max_iterations),
        "--smoothing-strength", str(smoothing_strength),
        "--coordinate-bounds", str(coordinate_lower), str(coordinate_upper),
        "--bounds-policy", bounds_policy, "--max-artifact-bytes", str(max_artifact_bytes),
    ]
    environment_digest = hashlib.sha256(json.dumps(
        environment, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    plan = native.make_plan(
        root, run_id=run_id, jobs=[{
            "trial_id": "prepare-c08-trajectory-bridge-candidate", "command": command,
            "cwd": "actionmesh", "input_refs": inputs, "code_refs": code,
            "output_paths": ["actionmesh/c08-trajectory-bridge-output/candidate.json",
                             "actionmesh/c08-trajectory-bridge-output/artifact-archive.json",
                             "actionmesh/c08-trajectory-bridge-output/artifact.tar"],
            "seed": seed, "group": "candidate-artifacts-only",
            "arm_role": "four-shared-geometry-path-artifacts-no-scorer",
        }], provenance={
            "git_revision": "source files pinned in code_refs; no implicit clean-tree claim",
            "model_revision": "no model loaded; receipt-bound ActionMesh sequence only",
            "data_revision": inputs[0]["sha256"], "environment_digest": environment_digest,
        }, limits={"max_attempts": 1, "max_development_trials": 1,
                   "max_confirmation_trials": 0, "max_retries_per_trial": 0,
                   "wall_time_seconds": wall_seconds,
                   "attempt_timeout_seconds": wall_seconds})
    native_path = plan_dir / "native.json"; native_path.write_text(json.dumps(plan, indent=2) + "\n")
    outer = harness.make_plan(
        root, batch_id=run_id, tasks=[{
            "task_id": "prepare-c08-trajectory-bridge-candidate",
            "idea_id": "4d-math-20261006-c08", "depends_on": [], "priority": 1,
            "plan_ref": file_ref(root, native_path),
            "resources": {"cpu_cores": 1, "ram_mib": ram_mib, "gpu_count": 0,
                          "gpu_peak_mib": None, "allow_gpu_share": False,
                          "memory_profile_ref": None, "exclusive_keys": []},
        }], limits={"total_wall_seconds": wall_seconds + 60,
                    "window_seconds": wall_seconds + 60, "max_parallel_tasks": 1,
                    "cpu_cores": 1, "ram_mib": ram_mib, "max_gpu_task_seconds": 0})
    (plan_dir / "harness.json").write_text(json.dumps(outer, indent=2) + "\n")
    return plan, outer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "skill-dir", "source-sequence", "plan-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--neighbors", type=int, required=True)
    parser.add_argument("--tolerance", type=float, required=True)
    parser.add_argument("--max-iterations", type=int, required=True)
    parser.add_argument("--smoothing-strength", type=float, required=True)
    parser.add_argument("--coordinate-bounds", nargs=2, type=float, required=True)
    parser.add_argument("--bounds-policy", choices=("preserve_and_report", "reject"), required=True)
    parser.add_argument("--max-artifact-bytes", type=int, required=True)
    parser.add_argument("--wall-seconds", type=int, required=True)
    parser.add_argument("--ram-mib", type=int, required=True)
    args = parser.parse_args()
    scripts = args.skill_dir.resolve() / "scripts"
    if not (scripts / "run_harness.py").is_file():
        parser.error("Complete installed research-autopilot skill required")
    sys.path.insert(0, str(scripts))
    _, outer = build_plans(
        args.root, source_sequence=args.source_sequence, run_id=args.run_id,
        epsilon=args.epsilon, neighbors=args.neighbors, tolerance=args.tolerance,
        max_iterations=args.max_iterations, smoothing_strength=args.smoothing_strength,
        coordinate_lower=args.coordinate_bounds[0], coordinate_upper=args.coordinate_bounds[1],
        bounds_policy=args.bounds_policy, max_artifact_bytes=args.max_artifact_bytes,
        plan_dir=args.plan_dir, wall_seconds=args.wall_seconds, ram_mib=args.ram_mib)
    print(json.dumps({"plan": str(args.plan_dir.resolve() / "harness.json"),
                      "approved_plan_digest": outer["plan_digest"],
                      "execution_started": False, "candidate_id": "4d-math-20261006-c08",
                      "native_qualified": False, "local_method_verified": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
