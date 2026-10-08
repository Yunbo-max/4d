"""Build C15's CPU-only three-role artifact plan; never run or score it."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys


CANDIDATE_ID = "4d-math-20261006-c15"
RANK_POLICY = "candidate_export_numeric_rank"
ALLOWED_BASIS_POLICIES = {
    "analytic_anchored_dct",
    "development_only_frozen",
    "input_only_geometry_predeclared",
}
ARMS = (
    "protected_residual_svt",
    "unprotected_svt",
    "rank_matched_tsvd",
)


def file_ref(root: Path, path: Path) -> dict:
    root, path = Path(root).resolve(), Path(path).resolve()
    relative = path.relative_to(root).as_posix()
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return {"path": relative, "sha256": value.hexdigest()}


def _positive(name: str, value) -> float:
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def build_plans(root: Path, *, source_sequence: Path, basis: Path,
                basis_source_policy: str, lambda_value: float,
                residual_rank_policy: str, basis_orthogonality_tolerance: float,
                protected_coefficient_tolerance: float,
                numeric_rank_tolerance: float, max_artifact_bytes: int,
                run_id: str, plan_dir: Path, wall_seconds: int,
                basis_evidence: Path):
    import run_experiments as native
    import run_harness as harness

    root = Path(root).resolve()
    source_sequence = Path(source_sequence).resolve()
    basis = Path(basis).resolve()
    plan_dir = Path(plan_dir).resolve()
    plan_dir.relative_to(root)
    if source_sequence.name != "sequence.npz":
        raise ValueError("Expected retained sequence.npz")
    if basis.suffix != ".npy":
        raise ValueError("Expected explicit temporal basis .npy")
    if basis_source_policy not in ALLOWED_BASIS_POLICIES:
        raise ValueError("Unsupported basis source policy")
    if residual_rank_policy != RANK_POLICY:
        raise ValueError(f"residual_rank_policy must be {RANK_POLICY}")
    _positive("lambda_value", lambda_value)
    _positive("basis_orthogonality_tolerance", basis_orthogonality_tolerance)
    if basis_orthogonality_tolerance > 1e-8:
        raise ValueError("basis_orthogonality_tolerance must be <=1e-8")
    _positive("protected_coefficient_tolerance", protected_coefficient_tolerance)
    if protected_coefficient_tolerance > 1e-4:
        raise ValueError("protected_coefficient_tolerance must be <=1e-4")
    _positive("numeric_rank_tolerance", numeric_rank_tolerance)
    if numeric_rank_tolerance > 1e-3:
        raise ValueError("numeric_rank_tolerance is relative and must be <=1e-3")
    if (isinstance(max_artifact_bytes, bool) or not isinstance(max_artifact_bytes, int)
            or not 10240 <= max_artifact_bytes <= 1024 * 1024 * 1024):
        raise ValueError("max_artifact_bytes must be an integer in [10240, 1 GiB]")
    if isinstance(wall_seconds, bool) or not 1 <= wall_seconds <= 26940:
        raise ValueError("CPU budget must be 1..26940 seconds so outer total stays <=27000")
    report_path = source_sequence.with_name("report.json")
    inputs = [
        file_ref(root, source_sequence), file_ref(root, report_path),
        file_ref(root, basis),
    ]
    basis_evidence = Path(basis_evidence).resolve()
    evidence_ref = file_ref(root, basis_evidence)
    inputs.append(evidence_ref)
    evidence = json.loads(basis_evidence.read_text())
    if (evidence.get("kind") != "c15-protection-basis-evidence"
            or evidence.get("version") != "1.0.0"
            or evidence.get("status") != "completed"
            or evidence.get("basis_sha256") != inputs[2]["sha256"]
            or evidence.get("basis_source_policy") != basis_source_policy
            or evidence.get("confirmation_outcomes_used") is not False
            or evidence.get("prospective_freeze") is not True
            or not isinstance(evidence.get("frozen_at"), str)
            or not isinstance(evidence.get("source_refs"), list)
            or not evidence["source_refs"]):
        raise ValueError("Basis evidence does not bind the declared prospective legal source policy")
    for ref in evidence["source_refs"]:
        if (not isinstance(ref, dict) or set(ref) != {"path", "sha256"}
                or not isinstance(ref["path"], str) or not ref["path"]
                or not isinstance(ref["sha256"], str) or len(ref["sha256"]) != 64):
            raise ValueError("Basis evidence source_refs must be exact path/SHA-256 rows")
        relative_ref = Path(ref["path"])
        if relative_ref.is_absolute() or ".." in relative_ref.parts:
            raise ValueError("Basis evidence source ref must be project-relative")
        cursor = root
        for part in relative_ref.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise ValueError("Basis evidence source ref path must not contain symlinks")
        current = file_ref(root, root / relative_ref)
        if current != ref:
            raise ValueError("Basis evidence source ref is missing or stale: " + ref["path"])
        if current not in inputs:
            inputs.append(current)
    source = json.loads(report_path.read_text())
    if (source.get("status") != "completed"
            or source.get("sha256", {}).get("sequence.npz") != inputs[0]["sha256"]):
        raise ValueError("Completed source report and current sequence hash required")
    uid, seed = source.get("uid"), source.get("seed")
    if not isinstance(uid, str) or not uid or Path(uid).name != uid or uid in (".", ".."):
        raise ValueError("Filesystem-safe native UID required")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Integer inference seed required")
    module = root / "actionmesh/research_math/protected_lowrank_candidate.py"
    command = [
        sys.executable, "-m", "research_math.protected_lowrank_candidate",
        "--project-root", "..",
        "--source-sequence", str(source_sequence),
        "--basis", str(basis),
        "--output", "c15-protected-lowrank-output",
        "--uid", uid,
        "--expected-sequence-sha256", inputs[0]["sha256"],
        "--expected-basis-sha256", inputs[2]["sha256"],
        "--basis-source-policy", basis_source_policy,
        "--lambda-value", str(lambda_value),
        "--residual-rank-policy", residual_rank_policy,
        "--basis-orthogonality-tolerance", str(basis_orthogonality_tolerance),
        "--protected-coefficient-tolerance", str(protected_coefficient_tolerance),
        "--numeric-rank-tolerance", str(numeric_rank_tolerance),
        "--max-artifact-bytes", str(max_artifact_bytes),
        "--basis-evidence", str(basis_evidence),
        "--expected-basis-evidence-sha256", evidence_ref["sha256"],
    ]
    code = [
        file_ref(root, root / "actionmesh/research_math/__init__.py"),
        file_ref(root, module),
    ]
    environment = {
        "python_executable": sys.executable,
        "python": platform.python_version(),
        "numpy": importlib.metadata.version("numpy"),
        "scope": "CPU C15 three-role artifact preparation only; no model or scorer",
        "svd_memory_policy": "economy SVD with native temporal dimension T=16",
    }
    environment_digest = hashlib.sha256(json.dumps(
        environment, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    outputs = [
        "actionmesh/c15-protected-lowrank-output/candidate.json",
        "actionmesh/c15-protected-lowrank-output/artifact-archive.json",
        "actionmesh/c15-protected-lowrank-output/artifact.tar",
    ]
    plan = native.make_plan(
        root, run_id=run_id,
        jobs=[{
            "trial_id": "prepare-c15-protected-lowrank-candidate",
            "command": command, "cwd": "actionmesh",
            "input_refs": inputs, "code_refs": code, "output_paths": outputs,
            "seed": seed, "group": "candidate-artifacts-only",
            "arm_role": "candidate-plus-two-controls-no-scorer-no-admission",
        }],
        provenance={
            "git_revision": "source files pinned in code_refs; no implicit clean-tree claim",
            "model_revision": "no model loaded; retained generated sequence identity only",
            "data_revision": inputs[0]["sha256"],
            "basis_revision": inputs[2]["sha256"],
            "basis_evidence_revision": evidence_ref["sha256"],
            "environment_digest": environment_digest,
            "parameter_freeze": {
                "lambda": float(lambda_value),
                "residual_rank_policy": residual_rank_policy,
                "basis_source_policy": basis_source_policy,
                "basis_orthogonality_tolerance": float(basis_orthogonality_tolerance),
                "protected_coefficient_tolerance": float(protected_coefficient_tolerance),
                "numeric_rank_relative_tolerance": float(numeric_rank_tolerance),
                "numeric_rank_threshold_formula": "s_i > relative_tolerance * s_0",
                "max_artifact_bytes": max_artifact_bytes,
                "confirmation_outcomes_used": False,
            },
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
            "task_id": "prepare-c15-protected-lowrank-candidate",
            "idea_id": CANDIDATE_ID, "depends_on": [], "priority": 1,
            "plan_ref": file_ref(root, native_path),
            "resources": {
                "cpu_cores": 1, "ram_mib": 4096, "gpu_count": 0,
                "gpu_peak_mib": None, "allow_gpu_share": False,
                "memory_profile_ref": None, "exclusive_keys": [],
            },
        }],
        limits={
            "total_wall_seconds": wall_seconds + 60,
            "window_seconds": wall_seconds + 60,
            "max_parallel_tasks": 1, "cpu_cores": 1, "ram_mib": 4096,
            "max_gpu_task_seconds": 0,
        })
    (plan_dir / "harness.json").write_text(json.dumps(outer, indent=2) + "\n")
    return plan, outer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "skill-dir", "source-sequence", "basis", "plan-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--basis-evidence", type=Path, required=True)
    parser.add_argument("--basis-source-policy", required=True,
                        choices=sorted(ALLOWED_BASIS_POLICIES))
    parser.add_argument("--lambda-value", type=float, required=True)
    parser.add_argument("--residual-rank-policy", required=True, choices=[RANK_POLICY])
    parser.add_argument("--basis-orthogonality-tolerance", type=float, required=True)
    parser.add_argument("--protected-coefficient-tolerance", type=float, required=True)
    parser.add_argument("--numeric-rank-tolerance", type=float, required=True)
    parser.add_argument("--max-artifact-bytes", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--wall-seconds", type=int, required=True)
    args = parser.parse_args()
    scripts = args.skill_dir.resolve() / "scripts"
    if not (scripts / "run_harness.py").is_file():
        parser.error("Full installed research-autopilot skill required")
    sys.path.insert(0, str(scripts))
    _, outer = build_plans(
        args.root, source_sequence=args.source_sequence, basis=args.basis,
        basis_evidence=args.basis_evidence,
        basis_source_policy=args.basis_source_policy,
        lambda_value=args.lambda_value,
        residual_rank_policy=args.residual_rank_policy,
        basis_orthogonality_tolerance=args.basis_orthogonality_tolerance,
        protected_coefficient_tolerance=args.protected_coefficient_tolerance,
        numeric_rank_tolerance=args.numeric_rank_tolerance,
        max_artifact_bytes=args.max_artifact_bytes,
        run_id=args.run_id, plan_dir=args.plan_dir,
        wall_seconds=args.wall_seconds)
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
