"""Build C02's CPU-only complete candidate-artifact plan; never run it."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys


CANDIDATE_ID = "4d-math-20261006-c02"


def file_ref(root: Path, path: Path) -> dict:
    root, path = Path(root).resolve(), Path(path).resolve()
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return {"path": path.relative_to(root).as_posix(), "sha256": value.hexdigest()}


def _positive(name: str, value) -> float:
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _positive_integer(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def build_plans(root: Path, *, source_sequence: Path, run_id: str,
                patch_count: int, arap_weight: float, temporal_weight: float,
                iterations: int, cg_tolerance: float,
                cg_max_iterations: int, plan_dir: Path,
                wall_seconds: int):
    import run_experiments as native
    import run_harness as harness

    root = Path(root).resolve()
    source_sequence = Path(source_sequence).resolve()
    plan_dir = Path(plan_dir).resolve(); plan_dir.relative_to(root)
    if source_sequence.name != "sequence.npz":
        raise ValueError("Expected sequence.npz")
    patch_count = _positive_integer("patch_count", patch_count)
    iterations = _positive_integer("iterations", iterations)
    cg_max_iterations = _positive_integer("cg_max_iterations", cg_max_iterations)
    for name, value in (("arap_weight", arap_weight),
                        ("temporal_weight", temporal_weight),
                        ("cg_tolerance", cg_tolerance)):
        _positive(name, value)
    if isinstance(wall_seconds, bool) or not 1 <= wall_seconds <= 26940:
        raise ValueError("CPU budget must be 1..26940 seconds")
    report_path = source_sequence.with_name("report.json")
    inputs = [file_ref(root, source_sequence), file_ref(root, report_path)]
    source = json.loads(report_path.read_text())
    if (source.get("status") != "completed"
            or source.get("sha256", {}).get("sequence.npz") != inputs[0]["sha256"]):
        raise ValueError("Completed source report and current sequence hash required")
    uid, seed = source.get("uid"), source.get("seed")
    if not isinstance(uid, str) or not uid or Path(uid).name != uid or uid in (".", ".."):
        raise ValueError("Filesystem-safe native UID required")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Integer inference seed required")
    modules = [
        root / "actionmesh/research_math/protected_geometry_candidate.py",
        root / "actionmesh/research_ten/m01_elasticity.py",
    ]
    command = [
        sys.executable, "-m", "research_math.protected_geometry_candidate",
        "--source-sequence", str(source_sequence),
        "--output", "c02-protected-geometry-output", "--uid", uid,
        "--expected-sequence-sha256", inputs[0]["sha256"],
        "--patch-count", str(patch_count), "--arap-weight", str(arap_weight),
        "--temporal-weight", str(temporal_weight),
        "--iterations", str(iterations), "--cg-tolerance", str(cg_tolerance),
        "--cg-max-iterations", str(cg_max_iterations),
    ]
    code = [file_ref(root, root / "actionmesh/research_math/__init__.py"),
            file_ref(root, root / "actionmesh/research_ten/__init__.py"),
            *(file_ref(root, module) for module in modules)]
    environment = {
        "python_executable": sys.executable,
        "python": platform.python_version(),
        "numpy": importlib.metadata.version("numpy"),
        "scope": "CPU C02 candidate artifact preparation only; no model or scorer",
    }
    environment_digest = hashlib.sha256(json.dumps(
        environment, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    prefix = "actionmesh/c02-protected-geometry-output/"
    outputs = [prefix + "candidate.json", prefix + "manifest.json",
               prefix + "certificate.npz"]
    for role in ("geometry_only", "strength_matched_blend", "protected_step"):
        outputs.extend([prefix + role + "/sequence.npz", prefix + role + "/report.json"])
    plan = native.make_plan(
        root, run_id=run_id,
        jobs=[{
            "trial_id": "prepare-c02-protected-geometry-candidate",
            "command": command, "cwd": "actionmesh", "input_refs": inputs,
            "code_refs": code, "output_paths": outputs, "seed": seed,
            "group": "candidate-artifacts-only",
            "arm_role": "candidate-and-operation-controls-no-scorer-no-admission",
        }],
        provenance={
            "git_revision": "source files pinned in code_refs; no implicit clean-tree claim",
            "model_revision": "no model loaded; cached generated sequence identity only",
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
            "task_id": "prepare-c02-protected-geometry-candidate",
            "idea_id": CANDIDATE_ID, "depends_on": [], "priority": 1,
            "plan_ref": file_ref(root, native_path),
            "resources": {
                "cpu_cores": 1, "ram_mib": 8192, "gpu_count": 0,
                "gpu_peak_mib": None, "allow_gpu_share": False,
                "memory_profile_ref": None, "exclusive_keys": [],
            },
        }],
        limits={
            "total_wall_seconds": wall_seconds + 60,
            "window_seconds": wall_seconds + 60, "max_parallel_tasks": 1,
            "cpu_cores": 1, "ram_mib": 8192, "max_gpu_task_seconds": 0,
        })
    (plan_dir / "harness.json").write_text(json.dumps(outer, indent=2) + "\n")
    return plan, outer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "skill-dir", "source-sequence", "plan-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--patch-count", type=int, required=True)
    parser.add_argument("--arap-weight", type=float, required=True)
    parser.add_argument("--temporal-weight", type=float, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--cg-tolerance", type=float, required=True)
    parser.add_argument("--cg-max-iterations", type=int, required=True)
    parser.add_argument("--wall-seconds", type=int, required=True)
    args = parser.parse_args()
    scripts = args.skill_dir.resolve() / "scripts"
    if not (scripts / "run_harness.py").is_file():
        parser.error("Full installed research-autopilot skill required")
    sys.path.insert(0, str(scripts))
    _, outer = build_plans(
        args.root, source_sequence=args.source_sequence, run_id=args.run_id,
        patch_count=args.patch_count, arap_weight=args.arap_weight,
        temporal_weight=args.temporal_weight, iterations=args.iterations,
        cg_tolerance=args.cg_tolerance,
        cg_max_iterations=args.cg_max_iterations,
        plan_dir=args.plan_dir, wall_seconds=args.wall_seconds)
    print(json.dumps({
        "plan": str(args.plan_dir.resolve() / "harness.json"),
        "approved_plan_digest": outer["plan_digest"],
        "execution_started": False, "candidate_id": CANDIDATE_ID,
        "native_qualified": False, "scientific_admission": False,
        "local_method_verified": False,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
