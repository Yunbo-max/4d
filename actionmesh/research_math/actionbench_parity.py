"""Replay official ActionBench CLI and faithful harness on identical arms."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback

import research_census_eval as census
from research_math.control_scoring import (
    ARMS, METRICS, DeviceSamples, contract_scorer_command, file_ref, resolve_ref,
    verify_request,
)


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


def scorer_descriptors(root: Path, request: dict) -> dict:
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
        "command": [sys.executable, str(official_adapter),
                    "--case-dir", str(control_root), "--gt-dir", str(gt_root),
                    "--output", "{output}", "--manifest", "{predictions}",
                    "--repo-root", str(repo_root), "--device", "cuda:0",
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
        "command": contract_scorer_command(root, request),
        "cwd": ".",
        "output": {"format": "json", "source": "file", "path": "{output}"},
        "denominator_path": ["denominator", "n_declared"],
    }
    return {"official_scorer": official, "harness_scorer": faithful}


def run_scorer(command: list[str], cwd: Path, timeout: int,
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
                "stdout_ref": file_ref(cwd.parent, stdout_path),
                "stderr_ref": file_ref(cwd.parent, stderr_path)}
    except Exception as exc:
        stdout_path.write_text("")
        stderr_path.write_text(traceback.format_exc())
        return {"status": "error", "command": command, "cwd": str(cwd),
                "exit_code": None, "elapsed_seconds": time.monotonic() - started,
                "error": f"{type(exc).__name__}: {exc}",
                "stdout_ref": file_ref(cwd.parent, stdout_path),
                "stderr_ref": file_ref(cwd.parent, stderr_path)}


def evaluate(root: Path, request_path: Path, protocol_path: Path, output: Path, device: str,
             gpu_uuid: str, group: str, timeout_seconds: int) -> int:
    root, request_path, protocol_path, output = map(lambda value: Path(value).resolve(),
                                     (root, request_path, protocol_path, output))
    output.relative_to(root)
    if output.exists():
        raise FileExistsError("Parity output is single-use")
    request = json.loads(request_path.read_text())
    verify_request(root, request)
    protocol = json.loads(protocol_path.read_text())
    descriptors = scorer_descriptors(root, request)
    contracts = protocol.get("native_eval_contracts")
    contract = contracts.get(group) if isinstance(contracts, dict) else protocol.get("native_eval_contract")
    if not isinstance(contract, dict) or contract.get("scorer") != descriptors["official_scorer"]:
        raise ValueError("Parity starts from a frozen official-scorer native contract")
    if any(arm["preparation_status"] != "completed" for arm in request["arms"]):
        raise ValueError("All three frozen prediction arms are required for parity")
    output.mkdir(parents=True, exist_ok=False)
    official_adapter = root / "actionmesh" / "official_actionbench_adapter.py"
    faithful_adapter = Path(census.__file__).resolve()
    control_root = resolve_ref(root, request["manifest_ref"]).parent
    gt_root = resolve_ref(root, request["ground_truth_ref"]).parent.parent
    repo_root = (root / request["repo_root"]).resolve()
    record = {
        "kind": "actionbench-official-faithful-parity",
        "version": "1.0.0", "status": "running",
        "scope": "one released development UID, three frozen baseline/control arms",
        "request_ref": file_ref(root, request_path),
        "protocol_ref": file_ref(root, protocol_path),
        "uid": request["uid"], "group": group, "device": device, "gpu_uuid": gpu_uuid,
        "metric_tolerances": {metric: 0.0 for metric in METRICS},
        "source_refs": [file_ref(root, official_adapter), file_ref(root, faithful_adapter)] +
                       [file_ref(root, repo_root / "actionbench" / name)
                        for name in census.OFFICIAL_FILES],
        "arms": {}, "native_contract_qualified": False,
        "scorer_descriptors": descriptors,
        "limitations": [
            "This parity unit does not supply source-backed baseline/control thresholds.",
            "It is not the nonce-bound trusted replay required for native run acceptance.",
            "It does not score or qualify a candidate method.",
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
                "--repo-root", str(repo_root), "--device", device, "--seed", "44"]
            faithful_command = [sys.executable, str(faithful_adapter),
                "--case-dir", str(control_root), "--gt-dir", str(gt_root),
                "--output", str(faithful_output), "--manifest", str(manifest_path),
                "--repo-root", str(repo_root), "--device", device, "--seed", "44"]
            executions = {}
            executions["official"] = run_scorer(
                official_command, root / "actionmesh", timeout_seconds,
                arm_root / "official.stdout.log", arm_root / "official.stderr.log")
            executions["faithful"] = run_scorer(
                faithful_command, root / "actionmesh", timeout_seconds,
                arm_root / "faithful.stdout.log", arm_root / "faithful.stderr.log")
            arm_record = {"executions": executions, "status": "error"}
            try:
                if any(value["status"] != "completed" for value in executions.values()):
                    raise RuntimeError("Both scorer processes must exit successfully")
                official_metrics = one_case(census.read_json(official_output), request["uid"])
                faithful_metrics = one_case(census.read_json(faithful_output), request["uid"])
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
    record["status"] = ("passed" if not telemetry.errors and
                        all(record["arms"].get(arm, {}).get("status") == "passed"
                            for arm in ARMS) else "failed")
    record["recorded_at"] = datetime.now(timezone.utc).isoformat()
    census.write_json(output / "record.json", record)
    evidence = {
        "kind": "faithful-harness-parity-evidence",
        "status": record["status"], "request_ref": record["request_ref"],
        "protocol_ref": record["protocol_ref"],
        "sample_uid": request["uid"], "source_refs": record["source_refs"],
        "metric_tolerances": record["metric_tolerances"],
        "official_scorer": descriptors["official_scorer"],
        "harness_scorer": descriptors["harness_scorer"],
        "arm_results": {arm: {key: record["arms"][arm].get(key)
                              for key in ("status", "official_metrics", "faithful_metrics", "comparison")}
                        for arm in ARMS},
        "sidecar_status": "written" if record["status"] == "passed" else "not_written_failed_parity",
        "native_contract_qualified": False,
    }
    census.write_json(output / "parity-evidence.json", evidence)
    if record["status"] == "passed":
        verification = {
            "official_scorer": descriptors["official_scorer"],
            "harness_scorer": descriptors["harness_scorer"],
            "sample_manifest_ref": contract["sample_manifest_ref"],
            "source_refs": record["source_refs"],
            "metric_tolerances": record["metric_tolerances"],
        }
        census.write_json(output / "faithful-harness-verification.json", verification)
    return 0 if record["status"] == "passed" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "request", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cuda:0"), default="cuda:0")
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--timeout-seconds", type=int, required=True)
    args = parser.parse_args()
    if not args.gpu_uuid.startswith("GPU-") or not 1 <= args.timeout_seconds <= 27000:
        parser.error("Physical GPU UUID and finite timeout <=27000 required")
    return evaluate(args.root, args.request, args.protocol, args.output, args.device,
                    args.gpu_uuid, args.group, args.timeout_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
