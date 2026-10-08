"""Emit the zero-GPU C03 confirmation-artifact plan; never execute it."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from research_math import correlated_calibration_candidate as candidate
from research_math import c03_calibration_artifacts as fitting
from research_math import self_map_candidate as c01


def ref(root, path):
    return candidate.file_ref(root, path)


def build_plans(root, *, skill_dir, c01_candidate, fit_bundle, plan_dir,
                run_id, confirmation_family, coordinate_bounds, bounds_policy,
                application_stage,
                max_artifact_bytes, wall_seconds, ram_mib):
    root, skill_dir, c01_candidate, fit_bundle, plan_dir = map(
        lambda value: Path(value).resolve(),
        (root, skill_dir, c01_candidate, fit_bundle, plan_dir))
    plan_dir.relative_to(root)
    if type(wall_seconds) is not int or not 1 <= wall_seconds <= 26940:
        raise ValueError("Explicit bounded CPU wall time required")
    if type(ram_mib) is not int or ram_mib < 1:
        raise ValueError("Positive RAM admission required")
    if type(max_artifact_bytes) is not int or max_artifact_bytes < 1:
        raise ValueError("Positive artifact byte ceiling required")
    scripts = skill_dir / "scripts"
    if not (scripts / "run_harness.py").is_file():
        raise ValueError("Complete installed research-autopilot required")
    source = c01.verify_candidate_artifacts(c01_candidate.parent)
    bundle = fitting.validate_bundle(candidate.read_json(fit_bundle))
    uid = candidate.read_json(source["source_report"])["uid"]
    generation_seed = candidate.read_json(source["source_report"])["seed"]
    if generation_seed not in c01.G01_GENERATION_SEEDS:
        raise ValueError("Exact G01 generation seed required")
    stage_uids = {"d1": bundle["policy"]["development_uids"],
                  "d2": bundle["policy"]["d2_uids"],
                  "confirmation": bundle["policy"]["confirmation_uids"]}
    if (application_stage not in stage_uids or uid not in stage_uids[application_stage]
            or bundle["policy"]["uid_to_family"].get(uid) != confirmation_family):
        raise ValueError("Exact frozen C03 application stage/UID/family required")
    inputs = [ref(root, c01_candidate), ref(root, fit_bundle)]
    fit_refs = bundle["input_refs"]
    inputs += [fit_refs["data"], fit_refs["policy"],
               *[fit_refs["evidence"][name] for name in sorted(fit_refs["evidence"])]]
    inputs += fit_refs["producer_inputs"]
    inputs += [ref(root, path) for path in source["context_files"]]
    inputs += [ref(root, source["certificate"]), ref(root, source["source_sequence"]),
               ref(root, source["source_report"])]
    unique = {item["path"]: item for item in inputs}
    code_paths = [root / "actionmesh/research_math" / name for name in (
        "__init__.py", "correlated_calibration.py", "c03_calibration_artifacts.py",
        "correlated_calibration_candidate.py", "self_map_candidate.py",
        "native_context_delivery.py", "native_context_runner.py",
        "pipeline_decoder_observer.py", "decoder_observer.py", "complete_unit_export.py")]
    code_paths.append(root / "actionmesh/prepare_correlated_calibration_candidate.py")
    command = [sys.executable, "-m", "research_math.correlated_calibration_candidate",
        "--root", str(root), "--c01-candidate", str(c01_candidate),
        "--fit-bundle", str(fit_bundle), "--output", "c03-candidate-output",
        "--application-stage", application_stage,
        "--application-family", confirmation_family,
        "--coordinate-bounds", *(str(value) for value in coordinate_bounds),
        "--bounds-policy", bounds_policy,
        "--max-artifact-bytes", str(max_artifact_bytes)]
    sys.path.insert(0, str(scripts))
    import run_experiments as native
    import run_harness as harness
    outputs = ["actionmesh/c03-candidate-output/" + name for name in
               ("candidate.json", "manifest.json", "common-target.npz",
                "artifact.tar", "artifact-archive.json")]
    for role in candidate.ROLES:
        # Failed arms are part of the frozen denominator and intentionally have
        # no sequence/certificate.  Only terminal reports are unconditional.
        outputs.append(f"actionmesh/c03-candidate-output/{role}/report.json")
    plan = native.make_plan(root, run_id=run_id, jobs=[{
        "trial_id": "prepare-c03-correlated-calibration-candidate",
        "command": command, "cwd": "actionmesh", "input_refs": list(unique.values()),
        "code_refs": [ref(root, path) for path in code_paths],
        "output_paths": outputs, "seed": generation_seed,
        "group": "candidate-artifacts-only",
        "arm_role": "confirmation-artifacts-no-labels-no-scorer-no-admission"}],
        provenance={"git_revision": "exact code refs; no clean-tree claim",
            "model_revision": "none; retained C01 context and frozen fits only",
            "data_revision": candidate.digest(fit_bundle),
            "environment_digest": hashlib.sha256(json.dumps({
                "python": sys.version, "ram_mib": ram_mib,
                "scope": "CPU frozen-fit application only"},
                sort_keys=True).encode()).hexdigest()},
        limits={"max_attempts": 1, "max_development_trials": 1,
            "max_confirmation_trials": 0, "max_retries_per_trial": 0,
            "wall_time_seconds": wall_seconds,
            "attempt_timeout_seconds": wall_seconds})
    plan_dir.mkdir(parents=True, exist_ok=False)
    native_path = plan_dir / "native.json"
    native_path.write_text(json.dumps(plan, indent=2) + "\n")
    outer = harness.make_plan(root, batch_id=run_id, tasks=[{
        "task_id": "prepare-c03-correlated-calibration-candidate",
        "idea_id": candidate.CANDIDATE_ID, "depends_on": [], "priority": 1,
        "plan_ref": ref(root, native_path), "resources": {"cpu_cores": 1,
            "ram_mib": ram_mib, "gpu_count": 0, "gpu_peak_mib": None,
            "allow_gpu_share": False, "memory_profile_ref": None,
            "exclusive_keys": []}}], limits={"total_wall_seconds": wall_seconds + 60,
        "window_seconds": wall_seconds + 60, "max_parallel_tasks": 1,
        "cpu_cores": 1, "ram_mib": ram_mib, "max_gpu_task_seconds": 0})
    (plan_dir / "harness.json").write_text(json.dumps(outer, indent=2) + "\n")
    return plan, outer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "skill-dir", "c01-candidate", "fit-bundle", "plan-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--confirmation-family", required=True)
    parser.add_argument("--application-stage", choices=("d1", "d2", "confirmation"),
                        required=True)
    parser.add_argument("--coordinate-bounds", type=float, nargs=2, required=True)
    parser.add_argument("--bounds-policy", choices=("preserve_and_report", "reject"), required=True)
    for name in ("max-artifact-bytes", "wall-seconds", "ram-mib"):
        parser.add_argument("--" + name, type=int, required=True)
    args = parser.parse_args(argv)
    _, outer = build_plans(args.root, skill_dir=args.skill_dir,
        c01_candidate=args.c01_candidate, fit_bundle=args.fit_bundle,
        plan_dir=args.plan_dir, run_id=args.run_id,
        application_stage=args.application_stage,
        confirmation_family=args.confirmation_family,
        coordinate_bounds=args.coordinate_bounds, bounds_policy=args.bounds_policy,
        max_artifact_bytes=args.max_artifact_bytes,
        wall_seconds=args.wall_seconds, ram_mib=args.ram_mib)
    print(json.dumps({"plan": str(args.plan_dir.resolve() / "harness.json"),
        "approved_plan_digest": outer["plan_digest"], "execution_started": False,
        "source_delivery_status": "generated_unexecuted", "native_qualified": False,
        "scientific_admission": False}))


if __name__ == "__main__":
    main()
