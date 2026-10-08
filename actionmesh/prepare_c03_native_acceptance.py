"""Prepare zero-GPU Local acceptance for one retained C03 artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from research_math import correlated_calibration_candidate as candidate


SOURCE_NAMES = (
    "research_math/__init__.py",
    "research_math/correlated_calibration.py",
    "research_math/c03_calibration_artifacts.py",
    "research_math/correlated_calibration_candidate.py",
    "research_math/self_map_candidate.py",
    "research_math/native_context_delivery.py",
    "research_math/native_context_runner.py",
    "research_math/pipeline_decoder_observer.py",
    "research_math/decoder_observer.py",
    "research_math/complete_unit_export.py",
    "research_math/tests/__init__.py",
    "research_math/tests/test_correlated_calibration_candidate.py",
    "prepare_c03_native_acceptance.py",
)


def _ref(root: Path, path: Path) -> dict:
    return candidate.file_ref(root, candidate.c01.physical(path))


def artifact_refs(root: Path, candidate_path: Path) -> list[dict]:
    """Pin the complete declared artifact plus upstream retained inputs."""
    root, candidate_path = Path(root).resolve(), Path(candidate_path).resolve()
    candidate_path.relative_to(root)
    record = candidate.read_json(candidate_path)
    if (candidate_path.name != "candidate.json"
            or record.get("kind") != "c03-correlated-calibration-candidate"
            or record.get("status") not in ("completed", "incomplete")
            or record.get("native_qualified") is not False):
        raise ValueError("Current terminal unqualified C03 candidate required")
    verified = candidate.validate_candidate_artifact(root, candidate_path)
    output = candidate_path.parent
    archive_record = candidate.read_json(output / "artifact-archive.json")
    archive = candidate.c01.physical(output / "artifact.tar")
    if (archive_record.get("kind") != "c03-terminal-artifact-archive"
            or archive_record.get("archive") != {"path": "artifact.tar",
                "sha256": candidate.digest(archive),
                "size_bytes": archive.stat().st_size}):
        raise ValueError("C03 terminal archive identity differs")
    files = [archive, output / "artifact-archive.json"]
    for row in archive_record.get("members", []):
        if (not isinstance(row, dict)
                or set(row) != {"path", "sha256", "size_bytes"}):
            raise ValueError("Exact C03 archive member required")
        path = candidate.c01.physical(output / row["path"])
        if (candidate.digest(path) != row["sha256"]
                or path.stat().st_size != row["size_bytes"]):
            raise ValueError("C03 archive member changed")
        files.append(path)
    for key in ("c01_candidate_ref", "fit_bundle_ref", "producer_provenance_ref"):
        ref = record[key]
        path = candidate.c01.physical(root / ref["path"])
        if candidate.digest(path) != ref["sha256"]:
            raise ValueError("C03 upstream reference changed: " + key)
        files.append(path)
    for ref in record["implementation_refs"]:
        path = candidate.c01.physical(root / ref["path"])
        if candidate.digest(path) != ref["sha256"]:
            raise ValueError("C03 implementation reference changed")
        files.append(path)
    for ref in verified.get("upstream_refs", []):
        path = candidate.resolve_ref(root, ref)
        files.append(path)
    actual = {item.resolve() for item in output.rglob("*")
              if item.is_file() or item.is_symlink()}
    if any(item.is_symlink() for item in output.rglob("*")):
        raise ValueError("Symlinks are forbidden in retained C03 artifacts")
    if actual != {Path(path).resolve() for path in files
                  if Path(path).resolve().is_relative_to(output)}:
        raise ValueError("Undeclared file in retained C03 artifact")
    return [_ref(root, path) for path in sorted(set(files))]


def build_plans(root, *, skill_dir, artifact_candidate, plan_dir, run_id,
                wall_seconds, ram_mib):
    root, skill_dir, artifact_candidate, plan_dir = (
        Path(value).resolve() for value in
        (root, skill_dir, artifact_candidate, plan_dir))
    plan_dir.relative_to(root)
    if type(wall_seconds) is not int or not 1 <= wall_seconds <= 26940:
        raise ValueError("Explicit CPU acceptance budget must be 1..26940 seconds")
    if type(ram_mib) is not int or ram_mib < 1:
        raise ValueError("Explicit positive CPU RAM budget required")
    scripts = skill_dir / "scripts"
    if not (scripts / "run_harness.py").is_file():
        raise ValueError("Complete installed research-autopilot required")
    inputs = artifact_refs(root, artifact_candidate)
    artifact_seed = candidate.read_json(artifact_candidate).get("seed")
    if (type(artifact_seed) is not int
            or artifact_seed not in candidate.c01.G01_GENERATION_SEEDS):
        raise ValueError("Exact G01 generation seed required")
    sources = [root / "actionmesh" / name for name in SOURCE_NAMES]
    code = [_ref(root, path) for path in sources]
    program = (
        "import os,sys,unittest\nfrom pathlib import Path\n"
        "sys.path.insert(0," + repr(str(scripts)) + ")\n"
        "os.environ['RESEARCH_AUTOPILOT_SKILL_DIR']=" + repr(str(skill_dir)) + "\n"
        "root=Path(sys.argv[1]).resolve()\n"
        "os.environ['C03_NATIVE_ROOT']=str(root)\n"
        "os.environ['C03_NATIVE_ARTIFACT']=str(Path(sys.argv[2]).resolve())\n"
        "suite=unittest.defaultTestLoader.loadTestsFromName("
        "'research_math.tests.test_correlated_calibration_candidate.C03RetainedNativeAcceptance')\n"
        "result=unittest.TextTestRunner(verbosity=2).run(suite)\n"
        "if not result.wasSuccessful() or result.skipped or result.testsRun!=1: raise SystemExit(1)\n")
    command = [sys.executable, "-c", program, str(root), str(artifact_candidate)]
    sys.path.insert(0, str(scripts))
    import run_experiments as native
    import run_harness as harness
    plan = native.make_plan(root, run_id=run_id, jobs=[{
        "trial_id": "c03-retained-native-acceptance", "command": command,
        "cwd": "actionmesh", "input_refs": inputs, "code_refs": code,
        "output_paths": [], "seed": artifact_seed, "group": "engineering",
        "arm_role": "retained-artifact-recomputation-no-scorer"}],
        provenance={"git_revision": "Exact C03 acceptance code refs",
            "model_revision": "none; retained CPU artifact only",
            "data_revision": candidate.digest(artifact_candidate),
            "environment_digest": hashlib.sha256(json.dumps({
                "python": sys.version,
                "harness": candidate.digest(scripts / "run_harness.py"),
                "native": candidate.digest(scripts / "run_experiments.py")},
                sort_keys=True).encode()).hexdigest()},
        limits={"max_attempts": 1, "max_development_trials": 1,
            "max_confirmation_trials": 0, "max_retries_per_trial": 0,
            "wall_time_seconds": wall_seconds,
            "attempt_timeout_seconds": wall_seconds})
    plan_dir.mkdir(parents=True, exist_ok=False)
    native_path = plan_dir / "native.json"
    native_path.write_text(json.dumps(plan, indent=2) + "\n")
    outer = harness.make_plan(root, batch_id=run_id, tasks=[{
        "task_id": "c03-retained-native-acceptance",
        "idea_id": candidate.CANDIDATE_ID, "depends_on": [], "priority": 1,
        "plan_ref": _ref(root, native_path), "resources": {"cpu_cores": 1,
            "ram_mib": ram_mib, "gpu_count": 0, "gpu_peak_mib": None,
            "allow_gpu_share": False, "memory_profile_ref": None,
            "exclusive_keys": []}}], limits={"total_wall_seconds": wall_seconds + 60,
        "window_seconds": wall_seconds + 60, "max_parallel_tasks": 1,
        "cpu_cores": 1, "ram_mib": ram_mib, "max_gpu_task_seconds": 0})
    (plan_dir / "harness.json").write_text(json.dumps(outer, indent=2) + "\n")
    return plan, outer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "skill-dir", "artifact-candidate", "plan-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--wall-seconds", type=int, required=True)
    parser.add_argument("--ram-mib", type=int, required=True)
    args = parser.parse_args(argv)
    _, outer = build_plans(args.root, skill_dir=args.skill_dir,
        artifact_candidate=args.artifact_candidate, plan_dir=args.plan_dir,
        run_id=args.run_id, wall_seconds=args.wall_seconds, ram_mib=args.ram_mib)
    print(json.dumps({"plan": str(args.plan_dir.resolve() / "harness.json"),
        "approved_plan_digest": outer["plan_digest"], "execution_started": False,
        "source_delivery_status": "generated_unexecuted",
        "local_method_verified": False, "native_qualified": False}))


if __name__ == "__main__":
    main()
