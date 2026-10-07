"""Replay official ActionBench CLI and faithful harness on identical arms."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import official_actionbench_adapter as official
import research_census_eval as census
from research_math.control_scoring import (
    ARMS, METRICS, DeviceSamples, file_ref,
    resolve_ref, validate_native_output, verify_request,
)


ACTIONBENCH_FRAMES = 16


def one_case(output: dict, uid: str) -> dict:
    rows = output.get("cases")
    if (output.get("denominator", {}).get("n_declared") != 1 or
            not isinstance(rows, list) or len(rows) != 1):
        raise ValueError("Scorer output must retain exactly one declared prediction")
    row = rows[0]
    if row.get("uid") != uid or row.get("status") != "success":
        raise ValueError("Scorer output is not a successful requested UID")
    values = {}
    for metric in METRICS:
        value = row.get(metric)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Missing numeric scorer metric: " + metric)
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError("Invalid scorer metric: " + metric)
        values[metric] = value
    return values


def exact_comparison(official: dict, faithful: dict) -> dict:
    differences = {metric: abs(official[metric] - faithful[metric]) for metric in METRICS}
    return {"metric_tolerances": {metric: 0.0 for metric in METRICS},
            "absolute_differences": differences,
            "passed": all(value == 0.0 for value in differences.values())}


def project_path_arg(root: Path, path: Path) -> str:
    """Return a scorer argv path relative to the staged project root."""
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def parity_harness_scorer_command(root: Path, request: dict, cpu_knn_backward=False) -> list[str]:
    """Describe the faithful scorer independently of checkout location."""
    root = Path(root).resolve()
    entry = (root/'actionmesh/deterministic_actionbench_entry.py' if cpu_knn_backward
             else Path(census.__file__))
    return [sys.executable, project_path_arg(root, entry),
            "--case-dir", project_path_arg(
                root, resolve_ref(root, request["manifest_ref"]).parent),
            "--gt-dir", project_path_arg(
                root, resolve_ref(root, request["ground_truth_ref"]).parent.parent),
            "--output", "{output}", "--manifest", "{predictions}",
            "--repo-root", project_path_arg(root, root / request["repo_root"]),
            "--device", "cuda:0", "--seed", "44"]


def validate_contract_boundary(contract: dict) -> None:
    """Keep scorer-equivalence evidence outside scientific effect qualification."""
    forbidden = {
        "baseline_qualification", "control_qualifications", "criteria",
        "contrasts", "effect_threshold", "qualification_rules",
    }
    if (not isinstance(contract, dict) or
            contract.get("purpose") != "scorer-equivalence-only" or
            contract.get("scientific_effect_qualification") is not False or
            contract.get("native_contract_qualified") is not False or
            forbidden.intersection(contract)):
        raise ValueError(
            "Scorer-equivalence contract cannot contain scientific qualification")


def validate_runtime_identity(environment: dict, *, gpu_uuid: str,
                              visible_gpu: str | None,
                              python_executable: str,
                              package_versions: dict) -> None:
    required = ("numpy", "torch", "trimesh", "scipy", "pytorch3d")
    if (not isinstance(environment, dict) or
            environment.get("python_executable") != python_executable or
            environment.get("gpu_uuid") != gpu_uuid or
            environment.get("execution_mode") != "native_host" or
            visible_gpu != gpu_uuid or
            any(environment.get("packages", {}).get(name) != package_versions.get(name)
                for name in required)):
        raise ValueError("Current native runtime/GPU identity required")


def parity_arm_inputs(request: dict) -> list[dict]:
    """Return the ordered prediction/report identity frozen by the request."""
    by_name = {row["arm"]: row for row in request["arms"]}
    return [{"arm": arm,
             "preparation_status": by_name[arm]["preparation_status"],
             "report_ref": by_name[arm]["report_ref"],
             "prediction_ref": by_name[arm]["sequence_ref"]}
            for arm in ARMS]


def parity_sample_manifest(request: dict, request_ref: dict, revision: str) -> dict:
    return {
        "kind": "actionbench-parity-sample-manifest",
        "version": "1.0.0",
        "benchmark_id": "facebook/actionbench",
        "benchmark_revision": revision,
        "split": "released-single-asset-parity-only",
        "sample_ids": [request["uid"]],
        "request_ref": request_ref,
        "request_digest": request["request_digest"],
        "population_ref": request["population_ref"],
        "denominator": 1,
        "predictions_per_sample": 1,
        "arm_inputs": parity_arm_inputs(request),
        "labels_or_tests_ref": request["ground_truth_ref"],
        "sampling": {"policy": "official-actionbench-full-sequence",
                     "parameters": request["native_protocol"]},
        "budget": {"frames": ACTIONBENCH_FRAMES, **request["native_protocol"]},
    }


def parity_contract(root: Path, request: dict, request_ref: dict,
                    sample_manifest_ref: dict,
                    source_evidence_ref: dict, environment_ref: dict,
                    cpu_knn_backward=False) -> dict:
    descriptors = scorer_descriptors(root, request, cpu_knn_backward)
    revision = descriptors["official_scorer"]["revision"]
    refs = [source_evidence_ref]
    for scorer in descriptors.values():
        refs.extend(scorer["source_refs"])
        refs.extend(scorer["code_refs"])
    source_refs = {ref["path"]: ref for ref in refs}
    contract = {
        "kind": "actionbench-scorer-equivalence-contract",
        "version": "1.0.0",
        "purpose": "scorer-equivalence-only",
        "scientific_effect_qualification": False,
        "native_contract_qualified": False,
        "benchmark_id": "facebook/actionbench",
        "benchmark_revision": revision,
        "split": "released-single-asset-parity-only",
        "uid": request["uid"],
        "request_ref": request_ref,
        "request_digest": request["request_digest"],
        "population_ref": request["population_ref"],
        "arm_inputs": parity_arm_inputs(request),
        "source_evidence_ref": source_evidence_ref,
        "source_refs": list(source_refs.values()),
        "environment_ref": environment_ref,
        "sample_manifest_ref": sample_manifest_ref,
        "labels_or_tests_ref": request["ground_truth_ref"],
        "metrics": list(METRICS),
        "sampling": {"policy": "official-actionbench-full-sequence",
                     "parameters": request["native_protocol"]},
        "budget": {"frames": ACTIONBENCH_FRAMES, **request["native_protocol"]},
        "arms": list(ARMS),
        "official_scorer": descriptors["official_scorer"],
        "harness_scorer": descriptors["harness_scorer"],
        "metric_tolerances": {metric: 0.0 for metric in METRICS},
    }
    if cpu_knn_backward:
        contract['runtime_policy'] = 'strict-cuda-forward-upstream-cpu-knn-backward-v1'
    return contract


def uses_cpu_backward(contract):
    policy = contract.get('runtime_policy')
    if policy not in (None, 'strict-cuda-forward-upstream-cpu-knn-backward-v1'):
        raise ValueError('Unknown scorer runtime policy')
    return policy is not None


def validate_parity_contract(root: Path, request: dict, request_path: Path,
                             contract_path: Path) -> dict:
    root, request_path, contract_path = map(Path.resolve,
                                            (Path(root), Path(request_path),
                                             Path(contract_path)))
    contract = census.read_json(contract_path)
    validate_contract_boundary(contract)
    evidence_path = (root / "docs" / "research-math-20261006" /
                     "actionbench-official-source-evidence.json")
    verify_source_evidence(root, request, evidence_path)
    evidence_ref = file_ref(root, evidence_path)
    sample_ref = contract.get("sample_manifest_ref")
    sample_path = resolve_ref(root, sample_ref)
    environment_ref = contract.get("environment_ref")
    resolve_ref(root, environment_ref)
    population = census.read_json(resolve_ref(root, request["population_ref"]))
    request_ref = file_ref(root, request_path)
    expected_sample = parity_sample_manifest(request, request_ref,
                                             population["revision"])
    if census.read_json(sample_path) != expected_sample:
        raise ValueError("Parity sample manifest differs from the released request")
    expected = parity_contract(root, request, request_ref, sample_ref, evidence_ref,
                               environment_ref, uses_cpu_backward(contract))
    if contract != expected:
        raise ValueError("Parity equivalence contract identity mismatch")
    return contract


def verify_frozen_contract_refs(root: Path, contract: dict) -> None:
    """Re-resolve every frozen file immediately before parity attestation."""
    refs = [contract[key] for key in (
        "request_ref", "population_ref", "source_evidence_ref", "environment_ref",
        "sample_manifest_ref", "labels_or_tests_ref")]
    for arm in contract["arm_inputs"]:
        refs.extend((arm["report_ref"], arm["prediction_ref"]))
    refs.extend(contract["source_refs"])
    for descriptor in (contract["official_scorer"], contract["harness_scorer"]):
        refs.extend(descriptor["source_refs"])
        refs.extend(descriptor["code_refs"])
    for ref in {(item["path"], item["sha256"]): item for item in refs}.values():
        resolve_ref(root, ref)


def verify_source_evidence(root: Path, request: dict, evidence_path: Path) -> dict:
    """Bind the retained web-source review to this exact request and checkout."""
    root, evidence_path = Path(root).resolve(), Path(evidence_path).resolve()
    evidence = census.read_json(evidence_path)
    population = census.read_json(resolve_ref(root, request["population_ref"]))
    benchmark = evidence.get("benchmark", {})
    selected = evidence.get("selected_development_asset", {})
    published = evidence.get("published_definition", {})
    threshold = evidence.get("threshold_review", {})
    if ((benchmark.get("id"), benchmark.get("revision")) !=
            ("facebook/actionbench", population.get("revision")) or
            benchmark.get("population_ref") != request["population_ref"]):
        raise ValueError("Source evidence benchmark/population differs from request")
    ground_truth = resolve_ref(root, request["ground_truth_ref"])
    if (selected.get("uid") != request["uid"] or
            selected.get("surfaces_lfs_sha256") != request["ground_truth_ref"]["sha256"] or
            selected.get("surfaces_bytes") != ground_truth.stat().st_size or
            selected.get("returned_gt_sha256_matches_released_lfs_oid") is not True):
        raise ValueError("Source evidence selected GT differs from request")
    if (published.get("metrics") != list(METRICS) or
            published.get("frames_per_sample") != ACTIONBENCH_FRAMES or
            published.get("tracked_points_per_frame") !=
            request["native_protocol"].get("n_pts_chamfer") or
            published.get("direction") != "lower_is_better"):
        raise ValueError("Source evidence metric/sampling definition differs from request")
    metric_domain = evidence.get("metric_domain_review", {})
    if (metric_domain.get("range") != "finite values >= 0" or
            metric_domain.get("supports_performance_qualification") is not False):
        raise ValueError("Source evidence lacks the scorer-validity domain boundary")
    if (threshold.get("per_asset_qualification_threshold_published") is not False or
            threshold.get("aggregate_values_may_be_applied_to_one_asset") is not False):
        raise ValueError("Source evidence threshold boundary changed")
    repo_root = (root / request["repo_root"] / "actionbench").resolve()
    expected_sources = [file_ref(root, repo_root / "README.md")] + [
        file_ref(root, repo_root / name) for name in census.OFFICIAL_FILES]
    if evidence.get("official_local_sources") != expected_sources:
        raise ValueError("Source evidence no longer matches pinned official files")
    return evidence


def scorer_descriptors(root: Path, request: dict, cpu_knn_backward=False) -> dict:
    """Return the exact official-adapter and faithful-harness identities."""
    root = Path(root).resolve()
    official_adapter = root / "actionmesh" / "official_actionbench_adapter.py"
    faithful_adapter = Path(census.__file__).resolve()
    repo_root = (root / request["repo_root"]).resolve()
    control_root = resolve_ref(root, request["manifest_ref"]).parent
    gt_root = resolve_ref(root, request["ground_truth_ref"]).parent.parent
    population = census.read_json(resolve_ref(root, request["population_ref"]))
    official_refs = [file_ref(root, repo_root / "actionbench" / name)
                     for name in census.OFFICIAL_FILES]
    official = {
        "kind": "official",
        "identity": "facebook/actionbench evaluate_dataset.py via format-only NPZ-to-GLB adapter",
        "revision": population["revision"],
        "source_refs": official_refs,
        "code_refs": official_refs + [file_ref(root, official_adapter)],
        "command": [sys.executable, project_path_arg(root, official_adapter),
                    "--case-dir", project_path_arg(root, control_root),
                    "--gt-dir", project_path_arg(root, gt_root),
                    "--output", "{output}", "--manifest", "{predictions}",
                    "--repo-root", project_path_arg(root, repo_root), "--device", "cuda:0",
                    "--seed", "{seed}"],
        "cwd": ".",
        "output": {"format": "json", "source": "file", "path": "{output}"},
        "denominator_path": ["denominator", "n_declared"],
    }
    faithful_refs = [file_ref(root, faithful_adapter)] + official_refs
    faithful = {
        "kind": "faithful_harness",
        "identity": "4d/research_census_eval.py ActionBench faithful harness",
        "revision": population["revision"],
        "source_refs": official_refs,
        "code_refs": faithful_refs,
        "command": parity_harness_scorer_command(root, request, cpu_knn_backward),
        "cwd": ".",
        "output": {"format": "json", "source": "file", "path": "{output}"},
        "denominator_path": ["denominator", "n_declared"],
    }
    if cpu_knn_backward:
        backend_refs = [file_ref(root, root/'actionmesh'/name) for name in (
            'deterministic_actionbench_entry.py', 'research_math/deterministic_knn.py',
            'research_math/knn_backend_checks.py', 'research_math/__init__.py')]
        official['command'].append('--cpu-knn-backward')
        for descriptor in (official, faithful):
            descriptor['code_refs'] += backend_refs
            descriptor['identity'] += '; strict CUDA forward / upstream CPU KNN backward v1'
    return {"official_scorer": official, "harness_scorer": faithful}


def run_scorer(command: list[str], cwd: Path, project_root: Path, timeout: int,
               stdout_path: Path, stderr_path: Path) -> dict:
    started = time.monotonic()
    try:
        completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True,
                                   timeout=timeout)
        stdout_path.write_text(completed.stdout)
        stderr_path.write_text(completed.stderr)
        return {"status": "completed" if completed.returncode == 0 else "error",
                "command": command, "cwd": str(cwd), "exit_code": completed.returncode,
                "elapsed_seconds": time.monotonic() - started,
                "stdout_ref": file_ref(project_root, stdout_path),
                "stderr_ref": file_ref(project_root, stderr_path)}
    except Exception as exc:
        stdout_path.write_text("")
        stderr_path.write_text(traceback.format_exc())
        return {"status": "error", "command": command, "cwd": str(cwd),
                "exit_code": None, "elapsed_seconds": time.monotonic() - started,
                "error": f"{type(exc).__name__}: {exc}",
                "stdout_ref": file_ref(project_root, stdout_path),
                "stderr_ref": file_ref(project_root, stderr_path)}


def validate_execution(root: Path, record: dict, command: list[str], cwd: Path) -> None:
    if (record.get("status") != "completed" or record.get("exit_code") != 0 or
            record.get("command") != command or record.get("cwd") != str(cwd)):
        raise ValueError("Scorer process execution binding mismatch")
    resolve_ref(root, record.get("stdout_ref"))
    resolve_ref(root, record.get("stderr_ref"))


def _absolute_ref(root: Path, ref: dict) -> dict:
    path = resolve_ref(root, ref)
    return {"path": str(path), "sha256": ref["sha256"]}


def validate_official_output(root: Path, request: dict, arm: str, stage: Path,
                             device: str, raw: dict, exit_code: int,
                             cpu_knn_backward=False) -> dict:
    """Require complete official adapter/source/input/output provenance."""
    root, stage = Path(root).resolve(), Path(stage).resolve()
    manifest_path = stage / "manifest.json"
    output_path = stage / "official.json"
    work = output_path.with_name(output_path.name + ".official")
    source_root = work / "official-source"
    control_root = resolve_ref(root, request["manifest_ref"]).parent
    repo_root = (root / request["repo_root"] / "actionbench").resolve()
    entry = next((row for row in request["arms"] if row["arm"] == arm), None)
    if not entry or entry.get("preparation_status") != "completed":
        raise ValueError("Official output arm was not a completed frozen input")
    if (not isinstance(raw, dict) or raw.get("schema_version") != 1 or
            raw.get("adapter_role") !=
            "format-only wrapper around official evaluate_dataset.py" or
            raw.get("metric_implementation") !=
            "none; metrics come from the official CLI subprocess" or
            raw.get("device") != device or raw.get("seed") != request["scoring_seed"] or
            raw.get("adapter_sha256") != official.digest(Path(official.__file__).resolve())):
        raise ValueError("Official output adapter/device/seed identity mismatch")
    denominator = raw.get("denominator")
    if (not isinstance(denominator, dict) or denominator.get("frozen") is not True or
            denominator.get("n_declared") != 1 or
            denominator.get("manifest") != str(manifest_path) or
            denominator.get("manifest_sha256") != official.digest(manifest_path)):
        raise ValueError("Official output frozen denominator mismatch")
    expected_original = {name: official.digest(repo_root / name)
                         for name in official.OFFICIAL_FILES}
    source = raw.get("official_source")
    expected_patch = {
        "file": "sample_mesh.py",
        "function": "get_baryc_sampling_mesh",
        "old": "devices=[verts.device], enabled=True",
        "new": "devices=([verts.device] if verts.is_cuda else []), enabled=True",
        "reason": "CPU surface sampler must not query CUDA RNG state for a CPU device",
        "metric_or_draw_change": False,
    }
    expected_patched = {name: official.digest(source_root / name)
                        for name in official.OFFICIAL_FILES}
    if (not isinstance(source, dict) or
            source.get("original_sha256") != expected_original or
            source.get("patched_source_sha256") != expected_patched or
            source.get("compatibility_patch") != expected_patch):
        raise ValueError("Official executed source provenance mismatch")
    rows = raw.get("cases")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise ValueError("Official output must retain exactly one case")
    row = rows[0]
    case_id = request["uid"] + "-" + arm
    case_work = work / case_id
    if (row.get("case_id") != case_id or row.get("uid") != request["uid"] or
            row.get("case_dir") != str(control_root / arm) or
            row.get("n_frames") != ACTIONBENCH_FRAMES or
            row.get("status") != "success" or exit_code != 0):
        raise ValueError("Official output case/UID/frame identity mismatch")
    if raw.get("summary") != {
            "n_total": 1, "n_success": 1, "n_failed": 0, "success_rate": 1.0}:
        raise ValueError("Official output summary/denominator mismatch")
    expected_inputs = {
        "sequence": _absolute_ref(root, entry["sequence_ref"]),
        "ground_truth": _absolute_ref(root, request["ground_truth_ref"]),
    }
    if row.get("inputs") != expected_inputs:
        raise ValueError("Official output input provenance mismatch")
    expected_output_paths = {
        "csv": case_work / "official.csv",
        "summary": case_work / "official.summary.json",
        "export_manifest": case_work / "export-manifest.json",
        "execution": case_work / "execution.json",
    }
    expected_outputs = {name: {"path": str(path), "sha256": official.digest(path)}
                        for name, path in expected_output_paths.items()}
    if row.get("official_outputs") != expected_outputs:
        raise ValueError("Official retained output provenance mismatch")
    export = official.read_json(expected_output_paths["export_manifest"])
    files = export.get("files")
    if (export.get("source_sequence") != expected_inputs["sequence"]["path"] or
            export.get("source_sequence_sha256") != expected_inputs["sequence"]["sha256"] or
            export.get("n_frames") != ACTIONBENCH_FRAMES or
            export.get("round_trip") != "exact vertices and faces for every frame" or
            not isinstance(files, list) or len(files) != ACTIONBENCH_FRAMES):
        raise ValueError("Official GLB export provenance mismatch")
    for index, item in enumerate(files):
        path = case_work / "predictions" / request["uid"] / f"mesh_{index:05d}.glb"
        if (item.get("frame_index") != index or item.get("path") != str(path) or
                item.get("sha256") != official.digest(path) or
                item.get("bytes") != path.stat().st_size):
            raise ValueError("Official GLB frame provenance mismatch")
    execution = official.read_json(expected_output_paths["execution"])
    expected_command = official.official_command(
        source_root / "evaluate_dataset.py", resolve_ref(root, request["ground_truth_ref"]).parent.parent,
        case_work / "predictions", case_work / "official.csv", device,
        request["scoring_seed"], cpu_knn_backward)
    if (execution.get("command") != expected_command or
            execution.get("cwd") != str(source_root) or execution.get("exit_code") != 0 or
            execution.get("stdout_sha256") != official.digest(case_work / "stdout.log") or
            execution.get("stderr_sha256") != official.digest(case_work / "stderr.log")):
        raise ValueError("Official CLI execution provenance mismatch")
    return {**row, **one_case(raw, request["uid"])}


def evaluate(root: Path, request_path: Path, contract_path: Path,
             environment_path: Path, output: Path,
             device: str, gpu_uuid: str,
             timeout_seconds: int) -> int:
    root, request_path, contract_path, environment_path, output = map(
        lambda value: Path(value).resolve(),
        (root, request_path, contract_path, environment_path, output))
    output.relative_to(root)
    if output.exists():
        raise FileExistsError("Parity output is single-use")
    request = json.loads(request_path.read_text())
    verify_request(root, request)
    contract = validate_parity_contract(root, request, request_path, contract_path)
    cpu_knn_backward = uses_cpu_backward(contract)
    if file_ref(root, environment_path) != contract["environment_ref"]:
        raise ValueError("Parity runtime environment differs from frozen contract")
    environment = census.read_json(environment_path)
    packages = {name: importlib.metadata.version(name)
                for name in ("numpy", "torch", "trimesh", "scipy", "pytorch3d")}
    validate_runtime_identity(
        environment, gpu_uuid=gpu_uuid,
        visible_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
        python_executable=sys.executable, package_versions=packages)
    descriptors = {"official_scorer": contract["official_scorer"],
                   "harness_scorer": contract["harness_scorer"]}
    sample_manifest_ref = contract.get("sample_manifest_ref")
    manifest = census.read_json(resolve_ref(root, sample_manifest_ref))
    if (manifest.get("sample_ids") != [request["uid"]] or
            manifest.get("denominator") != 1 or
            manifest.get("predictions_per_sample") != 1 or
            manifest.get("labels_or_tests_ref") != request["ground_truth_ref"]):
        raise ValueError("Parity contract must bind the one exact released sample")
    if any(arm["preparation_status"] != "completed" for arm in request["arms"]):
        raise ValueError("All three frozen prediction arms are required for parity")
    output.mkdir(parents=True, exist_ok=False)
    official_adapter = root / "actionmesh" / "official_actionbench_adapter.py"
    faithful_adapter = Path(census.__file__).resolve()
    if cpu_knn_backward:
        faithful_adapter = root/'actionmesh/deterministic_actionbench_entry.py'
    control_root = resolve_ref(root, request["manifest_ref"]).parent
    gt_root = resolve_ref(root, request["ground_truth_ref"]).parent.parent
    repo_root = (root / request["repo_root"]).resolve()
    record = {
        "kind": "actionbench-official-faithful-parity",
        "version": "1.0.0", "status": "running",
        "scope": "one released UID and three frozen prediction artifacts; scorer implementation equivalence only",
        "request_ref": file_ref(root, request_path),
        "parity_contract_ref": file_ref(root, contract_path),
        "sample_manifest_ref": sample_manifest_ref,
        "uid": request["uid"], "device": device, "gpu_uuid": gpu_uuid,
        "metric_tolerances": {metric: 0.0 for metric in METRICS},
        "environment_ref": contract["environment_ref"],
        "source_refs": contract["source_refs"],
        "arms": {}, "native_contract_qualified": False,
        "scorer_descriptors": descriptors,
        "limitations": [
            "It is not the nonce-bound trusted replay required for native run acceptance.",
            "It does not score or qualify a candidate method.",
            "It does not qualify baseline/control performance or satisfy a scientific protocol.",
        ],
    }
    census.write_json(output / "record.json", record)
    telemetry = DeviceSamples(gpu_uuid, output / "device-samples.jsonl")
    try:
        telemetry.start()
        for arm in ARMS:
            arm_root = output / arm
            arm_root.mkdir()
            manifest = {"cases": [{"case_id": request["uid"] + "-" + arm,
                                    "uid": request["uid"], "case_dir": arm}],
                        "scope": "same single prediction for official/faithful parity"}
            manifest_path = arm_root / "manifest.json"
            census.write_json(manifest_path, manifest)
            official_output = arm_root / "official.json"
            faithful_output = arm_root / "faithful.json"
            official_command = [sys.executable, str(official_adapter),
                "--case-dir", str(control_root), "--gt-dir", str(gt_root),
                "--output", str(official_output), "--manifest", str(manifest_path),
                "--repo-root", str(repo_root), "--device", device,
                "--seed", str(request["scoring_seed"])]
            faithful_command = [sys.executable, str(faithful_adapter),
                "--case-dir", str(control_root), "--gt-dir", str(gt_root),
                "--output", str(faithful_output), "--manifest", str(manifest_path),
                "--repo-root", str(repo_root), "--device", device,
                "--seed", str(request["scoring_seed"])]
            if cpu_knn_backward:
                official_command.append('--cpu-knn-backward')
            executions = {}
            executions["official"] = run_scorer(
                official_command, root, root, timeout_seconds,
                arm_root / "official.stdout.log", arm_root / "official.stderr.log")
            executions["faithful"] = run_scorer(
                faithful_command, root, root, timeout_seconds,
                arm_root / "faithful.stdout.log", arm_root / "faithful.stderr.log")
            arm_record = {"executions": executions, "status": "error"}
            try:
                validate_execution(root, executions["official"], official_command,
                                   root)
                validate_execution(root, executions["faithful"], faithful_command,
                                   root)
                official_raw = census.read_json(official_output)
                faithful_raw = census.read_json(faithful_output)
                official_row = validate_official_output(
                    root, request, arm, arm_root, device, official_raw,
                    executions["official"]["exit_code"], cpu_knn_backward)
                faithful_row = validate_native_output(
                    root, request, arm, arm_root, device, faithful_raw,
                    executions["faithful"]["exit_code"])
                if cpu_knn_backward:
                    lhs = official_raw['cases'][0].get('additional_runtime_compatibility', {})
                    rhs = faithful_raw.get('additional_runtime_compatibility', {})
                    source_hash = file_ref(root, root/'actionmesh/research_math/deterministic_knn.py')['sha256']
                    for meta in (lhs, rhs):
                        if (meta.get('source_sha256') != source_hash or
                                meta.get('cpu_backward_calls') != 6800 or
                                meta.get('kind') != 'experimental-knn-backward-cpu-compatibility'):
                            raise ValueError('Missing or inconsistent executed KNN backend provenance')
                    if lhs.get('extension_sha256') != rhs.get('extension_sha256'):
                        raise ValueError('Different installed PyTorch3D extensions')
                official_metrics = {metric: official_row[metric] for metric in METRICS}
                faithful_metrics = {metric: faithful_row[metric] for metric in METRICS}
                comparison = exact_comparison(official_metrics, faithful_metrics)
                arm_record.update(official_metrics=official_metrics,
                                  faithful_metrics=faithful_metrics,
                                  comparison=comparison,
                                  official_output_ref=file_ref(root, official_output),
                                  faithful_output_ref=file_ref(root, faithful_output),
                                  prediction_ref=next(row["sequence_ref"] for row in request["arms"]
                                                      if row["arm"] == arm),
                                  status="passed" if comparison["passed"] else "failed")
            except Exception as exc:
                arm_record.update(error=f"{type(exc).__name__}: {exc}",
                                  traceback=traceback.format_exc())
            record["arms"][arm] = arm_record
            census.write_json(output / "record.json", record)
    finally:
        telemetry.close()
    record["device_observations"] = {
        "samples": len(telemetry.memory),
        "max_observed_used_mib": max(telemetry.memory) if telemetry.memory else None,
        "telemetry_errors": telemetry.errors,
        "not_exact_peak": True,
    }
    frozen_ref_error = None
    try:
        validate_parity_contract(root, request, request_path, contract_path)
        verify_frozen_contract_refs(root, contract)
    except Exception as exc:
        frozen_ref_error = f"{type(exc).__name__}: {exc}"
    record["frozen_ref_recheck"] = {
        "status": "passed" if frozen_ref_error is None else "failed",
        "error": frozen_ref_error,
    }
    record["status"] = ("passed" if not telemetry.errors and
                        frozen_ref_error is None and
                        all(record["arms"].get(arm, {}).get("status") == "passed"
                            for arm in ARMS) else "failed")
    record["recorded_at"] = datetime.now(timezone.utc).isoformat()
    census.write_json(output / "record.json", record)
    evidence = {
        "kind": "faithful-harness-parity-evidence",
        "status": record["status"], "request_ref": record["request_ref"],
        "parity_contract_ref": record["parity_contract_ref"],
        "sample_uid": request["uid"], "source_refs": record["source_refs"],
        "metric_tolerances": record["metric_tolerances"],
        "official_scorer": descriptors["official_scorer"],
        "harness_scorer": descriptors["harness_scorer"],
        "arm_results": {arm: {key: record["arms"][arm].get(key)
                              for key in ("status", "official_metrics", "faithful_metrics", "comparison")}
                        for arm in ARMS},
        "bundle_attestation_status": (
            "written" if record["status"] == "passed" else "not_written_failed_parity"),
        "consumer_verification_status": (
            "pending_bundle_promotion_and_finalization" if record["status"] == "passed"
            else "not_eligible_failed_parity"),
        "native_contract_qualified": False,
    }
    census.write_json(output / "parity-evidence.json", evidence)
    if record["status"] == "passed":
        record_ref = file_ref(root, output / "record.json")
        evidence_ref = file_ref(root, output / "parity-evidence.json")
        verification = {
            "kind": "actionbench-parity-bundle-attestation",
            "bundle_target": "actionmesh/actionbench-parity-output",
            "bundle_promotion_required": True,
            "official_scorer": descriptors["official_scorer"],
            "harness_scorer": descriptors["harness_scorer"],
            "sample_manifest_ref": sample_manifest_ref,
            "parity_contract_ref": record["parity_contract_ref"],
            "request_ref": record["request_ref"],
            "population_ref": request["population_ref"],
            "ground_truth_ref": request["ground_truth_ref"],
            "environment_ref": record["environment_ref"],
            "record_ref": record_ref,
            "parity_evidence_ref": evidence_ref,
            "arm_evidence": {
                arm: {key: record["arms"][arm][key] for key in (
                    "prediction_ref", "official_output_ref", "faithful_output_ref")}
                for arm in ARMS
            },
            "source_refs": record["source_refs"],
            "metric_tolerances": record["metric_tolerances"],
            "scientific_effect_qualification": False,
            "native_contract_qualified": False,
        }
        census.write_json(output / "parity-bundle-attestation.json", verification)
    return 0 if record["status"] == "passed" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "request", "contract", "environment", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cuda:0"), default="cuda:0")
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--timeout-seconds", type=int, required=True)
    args = parser.parse_args()
    if not args.gpu_uuid.startswith("GPU-") or not 1 <= args.timeout_seconds <= 27000:
        parser.error("Physical GPU UUID and finite timeout <=27000 required")
    return evaluate(args.root, args.request, args.contract, args.environment, args.output,
                    args.device, args.gpu_uuid, args.timeout_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
