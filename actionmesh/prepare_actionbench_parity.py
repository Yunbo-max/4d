"""Build, but never execute, the official-vs-faithful ActionBench parity plan."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

from research_math.control_scoring import file_ref, resolve_ref, verify_request


def build_plans(root: Path, *, request_path: Path, environment_path: Path,
                skill_dir: Path, plan_dir: Path, run_id: str, gpu_uuid: str,
                wall_seconds: int, ram_mib: int, cpu_cores: int):
    root, request_path, environment_path, plan_dir = map(
        lambda value: Path(value).resolve(),
        (root, request_path, environment_path, plan_dir))
    plan_dir.relative_to(root)
    if not 1 <= wall_seconds <= 27000:
        raise ValueError("Finite parity limit <=27000; retain the 1800-second collection reserve")
    if ram_mib < 1 or cpu_cores < 1:
        raise ValueError("Positive admitted RAM and CPU limits required")
    if not gpu_uuid.startswith("GPU-") or any(char in gpu_uuid for char in "\n\r, "):
        raise ValueError("One actual physical GPU UUID required")
    scripts = Path(skill_dir).resolve() / "scripts"
    if not (scripts / "run_harness.py").is_file():
        raise ValueError("Complete installed research-autopilot skill required")
    sys.path.insert(0, str(scripts))
    import run_experiments as native
    import run_harness as harness

    request = json.loads(request_path.read_text())
    verify_request(root, request)
    if any(arm["preparation_status"] != "completed" for arm in request["arms"]):
        raise ValueError("All three completed control arms are required")
    environment = json.loads(environment_path.read_text())
    environment_ref = file_ref(root, environment_path)
    if (environment.get("python_executable") != sys.executable or
            environment.get("gpu_uuid") != gpu_uuid or
            environment.get("execution_mode") != "native_host"):
        raise ValueError("Current native interpreter/runtime/GPU identity required")
    for name in ("numpy", "torch", "trimesh", "scipy", "pytorch3d"):
        if environment.get("packages", {}).get(name) != importlib.metadata.version(name):
            raise ValueError("Installed package differs from runtime capture: " + name)
    dependency_refs = environment.get("dependency_lock_refs")
    if not isinstance(dependency_refs, list) or not dependency_refs:
        raise ValueError("Retained dependency lock references required")
    for ref in dependency_refs:
        resolve_ref(root, ref)
    from research_math.actionbench_parity import (
        parity_contract, parity_sample_manifest, scorer_descriptors,
        validate_parity_contract, verify_source_evidence,
    )
    descriptors = scorer_descriptors(root, request)
    population = json.loads(resolve_ref(root, request["population_ref"]).read_text())
    if (population.get("dataset"), population.get("revision")) != (
            "facebook/actionbench", descriptors["official_scorer"]["revision"]):
        raise ValueError("Released population and official scorer revision differ")
    source_evidence = (root / "docs" / "research-math-20261006" /
                       "actionbench-official-source-evidence.json")
    if not source_evidence.is_file():
        raise ValueError("Committed ActionBench source evidence is required")
    verify_source_evidence(root, request, source_evidence)
    source_evidence_ref = file_ref(root, source_evidence)

    new_sources = [root / "actionmesh" / "official_actionbench_adapter.py",
                   root / "actionmesh" / "research_math" / "actionbench_parity.py",
                   Path(__file__).resolve()]
    code_refs = {ref["path"]: ref for ref in request["code_refs"]}
    for path in new_sources:
        ref = file_ref(root, path)
        code_refs[ref["path"]] = ref
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                              capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "diff", "HEAD", "--binary"], cwd=root, check=True,
                           capture_output=True).stdout
    untracked = subprocess.run(["git", "ls-files", "--others", "--exclude-standard"],
                               cwd=root, check=True, capture_output=True,
                               text=True).stdout.splitlines()
    if any(path.endswith(".py") for path in untracked):
        raise ValueError("Commit all executable Python sources before Local acceptance")
    plan_dir.mkdir(parents=True, exist_ok=False)
    dirty_path = plan_dir / "dirty.patch"
    dirty_path.write_bytes(dirty)
    sample_manifest_path = plan_dir / "actionbench-parity-sample-manifest.json"
    request_ref = file_ref(root, request_path)
    sample_manifest_path.write_text(json.dumps(parity_sample_manifest(
        request, request_ref, descriptors["official_scorer"]["revision"]), indent=2) + "\n")
    sample_manifest_ref = file_ref(root, sample_manifest_path)
    contract_path = plan_dir / "actionbench-scorer-equivalence-contract.json"
    contract = parity_contract(root, request, request_ref, sample_manifest_ref,
                               source_evidence_ref, environment_ref)
    contract_path.write_text(json.dumps(contract, indent=2) + "\n")
    validate_parity_contract(root, request, request_path, contract_path)
    published_refs = [
        file_ref(root, root / "actionmesh" / "repo" / "actionbench" / "README.md"),
        file_ref(root, source_evidence),
    ]
    if contract.get("source_evidence_ref") != source_evidence_ref:
        raise ValueError("Parity contract must cite the verified ActionBench source evidence")
    inputs = request["input_refs"] + [file_ref(root, request_path),
                                      file_ref(root, contract_path),
                                      sample_manifest_ref, environment_ref,
                                      file_ref(root, dirty_path)] + published_refs + dependency_refs
    unique_inputs = {ref["path"]: ref for ref in inputs}
    command = [sys.executable, "-m", "research_math.actionbench_parity",
               "--root", "..", "--request", str(request_path),
               "--contract", str(contract_path),
               "--environment", str(environment_path),
               "--output", "actionbench-parity-output", "--device", "cuda:0",
               "--gpu-uuid", gpu_uuid,
               "--timeout-seconds", str(wall_seconds)]
    outputs = ["actionmesh/actionbench-parity-output/record.json",
               "actionmesh/actionbench-parity-output/parity-evidence.json",
               "actionmesh/actionbench-parity-output/parity-bundle-attestation.json",
               "actionmesh/actionbench-parity-output/device-samples.jsonl"]
    for arm in ("native", "world_gaussian", "body_gaussian"):
        official_case = f"{request['uid']}-{arm}"
        official_root = f"actionmesh/actionbench-parity-output/{arm}/official.json.official"
        outputs += [f"actionmesh/actionbench-parity-output/{arm}/manifest.json",
                    f"actionmesh/actionbench-parity-output/{arm}/official.json",
                    f"actionmesh/actionbench-parity-output/{arm}/faithful.json",
                    f"actionmesh/actionbench-parity-output/{arm}/official.stdout.log",
                    f"actionmesh/actionbench-parity-output/{arm}/official.stderr.log",
                    f"actionmesh/actionbench-parity-output/{arm}/faithful.stdout.log",
                    f"actionmesh/actionbench-parity-output/{arm}/faithful.stderr.log",
                    f"{official_root}/{official_case}/official.csv",
                    f"{official_root}/{official_case}/official.summary.json",
                    f"{official_root}/{official_case}/execution.json",
                    f"{official_root}/{official_case}/export-manifest.json",
                    f"{official_root}/{official_case}/stdout.log",
                    f"{official_root}/{official_case}/stderr.log"]
        outputs += [f"{official_root}/{official_case}/predictions/{request['uid']}/mesh_{index:05d}.glb"
                    for index in range(16)]
        outputs += [f"{official_root}/official-source/{name}"
                    for name in ("benchmark.py", "chamfer.py", "icp.py", "sample_mesh.py",
                                 "sample_point_cloud.py", "evaluate_dataset.py")]
    plan = native.make_plan(
        root, run_id=run_id, purpose="engineering", evidence_mode="developmental",
        jobs=[{"trial_id": "official-faithful-three-arm-parity", "command": command,
               "cwd": "actionmesh", "input_refs": list(unique_inputs.values()),
               "code_refs": list(code_refs.values()), "output_paths": outputs,
               "seed": request["scoring_seed"], "group": "engineering",
               "arm_role": "scorer-parity"}],
        provenance={"git_revision": revision, "git_refs": [file_ref(root, dirty_path)],
                    "model_revision": "none; scoring existing frozen predictions only",
                    "data_revision": request["ground_truth_ref"]["sha256"],
                    "environment_digest": environment_ref["sha256"],
                    "environment_refs": [environment_ref] + dependency_refs},
        limits={"max_attempts": 1, "max_development_trials": 1,
                "max_confirmation_trials": 0, "max_retries_per_trial": 0,
                "wall_time_seconds": wall_seconds,
                "attempt_timeout_seconds": wall_seconds})
    plan["plan_digest"] = native.plan_digest(plan)
    native.validate_plan(root, plan)
    native_path = plan_dir / "native.json"
    native_path.write_text(json.dumps(plan, indent=2) + "\n")
    outer = harness.make_plan(
        root, batch_id=run_id,
        tasks=[{"task_id": "actionbench-official-faithful-parity",
                "idea_id": "evaluator-implementation-equivalence", "depends_on": [], "priority": 1,
                "plan_ref": file_ref(root, native_path),
                "resources": {"cpu_cores": cpu_cores, "ram_mib": ram_mib,
                              "gpu_count": 1, "gpu_peak_mib": None,
                              "allow_gpu_share": False, "memory_profile_ref": None,
                              "exclusive_keys": ["actionbench-native-scorer"]}}],
        limits={"total_wall_seconds": wall_seconds, "window_seconds": wall_seconds,
                "max_parallel_tasks": 1, "cpu_cores": cpu_cores,
                "ram_mib": ram_mib, "max_gpu_task_seconds": wall_seconds},
        gpus={"uuids": [gpu_uuid], "safety_margin_mib": 1024,
              "max_tasks_per_gpu": 1})
    (plan_dir / "harness.json").write_text(json.dumps(outer, indent=2) + "\n")
    return outer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "request", "environment", "skill-dir", "plan-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("run-id", "gpu-uuid"):
        parser.add_argument("--" + name, required=True)
    for name in ("wall-seconds", "ram-mib", "cpu-cores"):
        parser.add_argument("--" + name, type=int, required=True)
    args = parser.parse_args()
    plan = build_plans(args.root, request_path=args.request,
        environment_path=args.environment, skill_dir=args.skill_dir,
        plan_dir=args.plan_dir, run_id=args.run_id, gpu_uuid=args.gpu_uuid,
        wall_seconds=args.wall_seconds, ram_mib=args.ram_mib,
        cpu_cores=args.cpu_cores)
    print(json.dumps({"plan": str(args.plan_dir / "harness.json"),
                      "approved_plan_digest": plan["plan_digest"],
                      "execution_started": False,
                      "scope": "official-vs-faithful scorer parity only",
                      "native_contract_qualified": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
