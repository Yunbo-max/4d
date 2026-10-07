"""Price, but never dispatch, the full-128 ActionBench rolling queue.

The only admissible timing input is the separately finalized complete three-arm
engineering unit.  This module rehashes that sidecar's retained evidence and
freezes deterministic 128-UID window partitions.  It does not create executable
GPU plans, qualify the official baseline, or test a candidate method.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from research_math.complete_unit_contract import ARMS, METRICS
from research_math.complete_unit_admission import (
    INCIDENTAL_PYTHON_CACHE_PATHS,
    validate_completed_result,
    verify_origin,
)
from research_math.control_scoring import file_ref, resolve_ref


ADMISSION_CONTRACT_PATH = (
    "docs/research-math-20261006/actionbench-complete-unit-admission-contract.json")
REPRODUCTION_CONTRACT_PATH = (
    "docs/research-math-20261006/actionbench-full128-reproduction-contract.json")
POPULATION_PATH = "actionmesh/research_overnight/assets/actionbench_population.json"


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _is_pinned_ref(value, expected_path: str) -> bool:
    if (not isinstance(value, dict) or set(value) != {"path", "sha256"} or
            value.get("path") != expected_path):
        return False
    digest = value.get("sha256")
    return (isinstance(digest, str) and len(digest) == 64 and
            all(character in "0123456789abcdef" for character in digest))


def validate_pricing_contract(contract: dict) -> None:
    source = contract.get("source_admission", {})
    population = contract.get("population", {})
    budget = contract.get("budget", {})
    expected_population = {
        "dataset": "facebook/actionbench",
        "revision": "2796071cbe6248422fcbeab3101fa9f9886cb7b9",
        "size": 128,
    }
    expected_budget = {
        "hard_window_seconds": 28800,
        "collection_reserve_seconds": 1800,
        "workload_budget_seconds": 27000,
        "unit_runtime_headroom_numerator": 5,
        "unit_runtime_headroom_denominator": 4,
    }
    if (contract.get("kind") != "actionbench-full128-queue-pricing-contract" or
            contract.get("version") != "1.0.0" or
            source.get("kind") != "actionbench-complete-unit-admission" or
            source.get("version") != "1.0.0" or
            source.get("status") != "admitted_engineering_complete_unit" or
            source.get("run_id") != "complete-lowram-r7" or
            source.get("runtime_profile") != "fp16-lowram-v1" or
            source.get("successful_output_file_count") != 117 or
            not _is_pinned_ref(source.get("contract_ref"),
                               ADMISSION_CONTRACT_PATH) or
            population != expected_population or
            budget != expected_budget or
            not _is_pinned_ref(contract.get("full128_reproduction_contract_ref"),
                               REPRODUCTION_CONTRACT_PATH) or
            not _is_pinned_ref(contract.get("population_ref"), POPULATION_PATH) or
            contract.get("output_target") !=
            "inputs/actionbench-full128-queue/pricing.json" or
            contract.get("scientific_effect_qualification") is not False or
            contract.get("native_scientific_qualification") is not False or
            contract.get("candidate_methods_tested") is not False or
            contract.get("queue_approved") is not False or
            contract.get("queue_generated") is not False or
            contract.get("dispatch_ready") is not False):
        raise ValueError("Frozen full128 queue-pricing contract required")


def _positive_finite(value, label: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or value <= 0):
        raise ValueError("Positive finite " + label + " required")
    return float(value)


def _validate_scores(scores: dict) -> None:
    if not isinstance(scores, dict) or set(scores) != set(ARMS):
        raise ValueError("Exact complete-unit score arms required")
    for arm in ARMS:
        row = scores[arm]
        if not isinstance(row, dict) or set(row) != set(METRICS):
            raise ValueError("Exact complete-unit score metrics required")
        for metric in METRICS:
            value = row[metric]
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(value) or value < 0):
                raise ValueError("Finite nonnegative complete-unit score required")


def _canonical_origin_paths(root: Path, harness_plan: dict, native_plan: dict,
                            native_receipt: dict) -> list[Path]:
    tasks = harness_plan.get("tasks", [])
    attempts = native_receipt.get("attempts", [])
    if len(tasks) != 1 or len(attempts) != 1:
        raise ValueError("Exact one-task, one-attempt source unit required")
    batch = root / harness_plan.get("output_root", "") / harness_plan.get("batch_id", "")
    task = batch / "tasks" / tasks[0].get("task_id", "")
    run = root / native_plan.get("output_root", "") / native_plan.get("run_id", "")
    attempt = root / attempts[0].get("attempt_path", "")
    return [
        batch / "plan.json", batch / "state.json", batch / "report.json",
        task / "task.json", task / "result.json", run / "plan.json",
        task / "execution-context.json", run / "receipt.json",
        attempt / "attempt.json",
        resolve_ref(root, attempts[0].get("process_guard_ref")),
        resolve_ref(root, attempts[0].get("stdout_ref")),
        resolve_ref(root, attempts[0].get("stderr_ref")),
        attempt / "workspace/actionmesh/unit-output/generation.execution.json",
        attempt / "workspace/actionmesh/unit-output/official-scoring.execution.json",
    ]


def canonical_harness_plan_path(root: Path, admission: dict) -> Path:
    """Resolve the retained harness plan pinned by the admission sidecar.

    Completed harness batches retain their executed plan under
    ``runs/harness/<run_id>/plan.json``.  The original builder input under
    ``plans/<run_id>/harness.json`` is not part of this admission's canonical
    closure, so pricing must validate and use the retained executed plan.
    """
    run_id = "complete-lowram-r7"
    path = Path(root) / "runs" / "harness" / run_id / "plan.json"
    expected = file_ref(root, path)
    refs = admission.get("origin_refs")
    if (not isinstance(refs, list) or
            [ref for ref in refs if isinstance(ref, dict) and
             ref.get("path") == expected["path"]] != [expected]):
        raise ValueError("Exact canonical harness plan ref required")
    return path


def _validate_origin(root: Path, admission: dict) -> None:
    run_id = "complete-lowram-r7"
    harness_plan_path = canonical_harness_plan_path(root, admission)
    harness_report_path = root / "runs/harness" / run_id / "report.json"
    native_plan_path = root / "plans" / run_id / "native.json"
    native_receipt_path = root / "runs/attempts" / run_id / "receipt.json"
    harness_plan = json.loads(harness_plan_path.read_text())
    harness_report = json.loads(harness_report_path.read_text())
    native_plan = json.loads(native_plan_path.read_text())
    native_receipt = json.loads(native_receipt_path.read_text())
    digest = admission["approved_plan_digest"]
    if (harness_plan.get("plan_digest") != digest or
            harness_report.get("plan_digest") != digest or
            harness_report.get("status") != "completed" or
            native_plan.get("run_id") != run_id or
            native_receipt.get("plan_digest") != native_plan.get("plan_digest") or
            native_receipt.get("status") != "completed"):
        raise ValueError("Canonical completed source records required")
    expected = [file_ref(root, path) for path in _canonical_origin_paths(
        root, harness_plan, native_plan, native_receipt)]
    if admission.get("origin_refs") != expected:
        raise ValueError("Exact canonical 14-record origin closure required")
    contract_path = resolve_ref(root, admission.get("contract_ref"))
    contract = json.loads(contract_path.read_text())
    recomputed, _ = verify_origin(
        root, harness_plan_path, harness_report_path, native_plan_path,
        native_receipt_path, digest, contract)
    recomputed["contract_ref"] = file_ref(root, contract_path)
    if recomputed != admission:
        raise ValueError("Admission differs from canonical source recomputation")


def runner_inventory_refs(root: Path, admission: dict) -> list[dict]:
    """Rehash the 121 pre-result files and bind measurement to raw result."""
    root = Path(root).resolve()
    result_path = resolve_ref(root, admission.get("result_ref"))
    result = json.loads(result_path.read_text())
    rows = result.get("outputs")
    if not isinstance(rows, list) or len(rows) != 121:
        raise ValueError("Exact 121-file runner inventory required")
    refs = []
    seen = set()
    for row in rows:
        relative = Path(row.get("path", "")) if isinstance(row, dict) else Path()
        if (not isinstance(row, dict) or
                set(row) != {"path", "bytes", "sha256"} or relative.is_absolute() or
                not relative.parts or ".." in relative.parts or
                relative.as_posix() in seen):
            raise ValueError("Unique relative runner inventory row required")
        path = (result_path.parent / relative).resolve()
        path.relative_to(result_path.parent.resolve())
        ref = file_ref(root, path)
        if (row["path"] != relative.as_posix() or
                row["bytes"] != path.stat().st_size or
                row["sha256"] != ref["sha256"]):
            raise ValueError("Runner inventory hash or size mismatch")
        seen.add(relative.as_posix())
        refs.append(ref)
    cache_paths = {
        (result_path.parent / relative).resolve()
        for relative in INCIDENTAL_PYTHON_CACHE_PATHS
    }
    inventory_paths = {resolve_ref(root, ref) for ref in refs}
    output_refs = admission.get("output_refs", [])
    output_paths = {resolve_ref(root, ref) for ref in output_refs}
    if (inventory_paths - (output_paths - {result_path}) != cache_paths or
            (output_paths - {result_path}) - inventory_paths or
            result_path not in output_paths):
        raise ValueError("117 receipt outputs and five-cache inventory closure required")
    measurement = admission.get("measurement", {})
    recomputed = validate_completed_result(
        result, measurement.get("uid"), measurement.get("gpu_total_mib"))
    if recomputed != measurement:
        raise ValueError("Admission measurement differs from retained result")
    return refs


def _validate_admission(root: Path, contract: dict, admission: dict) -> dict:
    source = contract["source_admission"]
    measurement = admission.get("measurement", {})
    if (admission.get("kind") != source["kind"] or
            admission.get("version") != source["version"] or
            admission.get("status") != source["status"] or
            admission.get("run_id") != source["run_id"] or
            admission.get("approved_plan_digest") !=
            "a94aa69f8605266587f56f0977740001bbd21971463bb5b78d4e20eaf2bf574b" or
            admission.get("successful_output_file_count") !=
            source["successful_output_file_count"] or
            admission.get("contract_ref") != source["contract_ref"] or
            admission.get("eligible_for_queue_pricing") is not True or
            admission.get("scientific_effect_qualification") is not False or
            admission.get("native_scientific_qualification") is not False or
            admission.get("candidate_methods_tested") is not False or
            admission.get("queue_approved") is not False or
            admission.get("queue_generated") is not False or
            measurement.get("runtime_profile") != source["runtime_profile"] or
            measurement.get("scientific_effect_qualification") is not False or
            measurement.get("candidate_methods_tested") is not False or
            measurement.get("queue_approved") is not False or
            measurement.get("queue_generated") is not False):
        raise ValueError("Exact admitted complete-unit pricing source required")
    output_refs = admission.get("output_refs")
    origin_refs = admission.get("origin_refs")
    if (not isinstance(output_refs, list) or len(output_refs) != 117 or
            len({json.dumps(ref, sort_keys=True) for ref in output_refs}) != 117 or
            not isinstance(origin_refs, list) or len(origin_refs) != 14 or
            len({json.dumps(ref, sort_keys=True) for ref in origin_refs}) != 14 or
            not isinstance(admission.get("unit_manifest_ref"), dict) or
            not isinstance(admission.get("result_ref"), dict) or
            admission.get("result_ref") not in output_refs):
        raise ValueError("Complete admission evidence closure required")
    seen = {}
    for ref in [source["contract_ref"], admission["unit_manifest_ref"],
                admission["result_ref"], *origin_refs, *output_refs]:
        path = resolve_ref(root, ref)
        if path in seen and seen[path] != ref:
            raise ValueError("Conflicting admission evidence reference")
        seen[path] = ref
    _validate_origin(root, admission)
    runner_inventory_refs(root, admission)
    elapsed = _positive_finite(measurement.get("elapsed_seconds"),
                               "admitted elapsed seconds")
    peak = _positive_finite(measurement.get("observed_peak_mib"),
                            "admitted observed GPU peak")
    total = _positive_finite(measurement.get("gpu_total_mib"),
                             "admitted GPU total memory")
    if peak > total:
        raise ValueError("Admitted GPU peak exceeds physical memory")
    uid = measurement.get("uid")
    if not isinstance(uid, str) or not uid or Path(uid).name != uid:
        raise ValueError("Canonical admitted UID required")
    _validate_scores(measurement.get("scores"))
    return {"elapsed_seconds": elapsed, "observed_peak_mib": peak,
            "gpu_total_mib": total, "uid": uid}


def _validate_reproduction_contract(root: Path, contract: dict) -> dict:
    path = resolve_ref(root, contract["full128_reproduction_contract_ref"])
    reproduction = json.loads(path.read_text())
    queue = reproduction.get("queue_policy", {})
    if (reproduction.get("kind") !=
            "actionbench-full-population-reproduction-contract" or
            reproduction.get("version") != "1.0.0" or
            reproduction.get("purpose") !=
            "official-full-population-baseline-reproduction" or
            queue.get("hard_window_seconds") != 28800 or
            queue.get("collection_reserve_seconds") != 1800 or
            reproduction.get("scientific_effect_qualification") is not False or
            reproduction.get("candidate_methods_tested") is not False or
            reproduction.get("dispatch_ready") is not False):
        raise ValueError("Frozen conditional full128 reproduction contract required")
    return reproduction


def _validate_population(root: Path, contract: dict, population: dict) -> list[str]:
    expected = contract["population"]
    canonical = json.loads(resolve_ref(root, contract["population_ref"]).read_text())
    uids = population.get("uids")
    if (population != canonical or
            population.get("dataset") != expected["dataset"] or
            population.get("revision") != expected["revision"] or
            not isinstance(uids, list) or len(uids) != expected["size"] or
            len(uids) != len(set(uids)) or uids != sorted(uids) or
            any(not isinstance(uid, str) or not uid or Path(uid).name != uid
                for uid in uids)):
        raise ValueError("Exact canonical 128-UID population required")
    return uids


def build_pricing_manifest(root: Path, contract: dict, admission: dict,
                           population: dict) -> dict:
    root = Path(root).resolve()
    validate_pricing_contract(contract)
    measurement = _validate_admission(root, contract, admission)
    _validate_reproduction_contract(root, contract)
    uids = _validate_population(root, contract, population)
    budget = contract["budget"]
    unit_timeout = math.ceil(
        measurement["elapsed_seconds"] *
        budget["unit_runtime_headroom_numerator"] /
        budget["unit_runtime_headroom_denominator"])
    units_per_window = budget["workload_budget_seconds"] // unit_timeout
    if units_per_window < 1:
        raise ValueError("Headroom-adjusted complete unit does not fit workload budget")
    windows = []
    for start in range(0, len(uids), units_per_window):
        selected = uids[start:start + units_per_window]
        windows.append({
            "window_id": f"full128-window-{len(windows) + 1:02d}",
            "population_start_index": start,
            "population_stop_index_exclusive": start + len(selected),
            "uids": selected,
            "unit_count": len(selected),
            "planned_workload_seconds": len(selected) * unit_timeout,
        })
    full_window = units_per_window * unit_timeout
    return {
        "kind": "actionbench-full128-queue-pricing",
        "version": "1.0.0",
        "status": "priced_engineering_only",
        "scope": ("Deterministic rolling-window pricing for the conditional full128 "
                  "current-public-release baseline reproduction; not an executable "
                  "queue or candidate experiment."),
        "population": contract["population"],
        "population_uid_sha256": hashlib.sha256(_canonical(uids)).hexdigest(),
        "source_admission_run_id": admission["run_id"],
        "source_admission_plan_digest": admission["approved_plan_digest"],
        "source_measurement": measurement,
        "hard_window_seconds": budget["hard_window_seconds"],
        "collection_reserve_seconds": budget["collection_reserve_seconds"],
        "workload_budget_seconds": budget["workload_budget_seconds"],
        "unit_runtime_headroom_numerator":
            budget["unit_runtime_headroom_numerator"],
        "unit_runtime_headroom_denominator":
            budget["unit_runtime_headroom_denominator"],
        "unit_timeout_seconds": unit_timeout,
        "units_per_window": units_per_window,
        "window_count": len(windows),
        "planned_workload_seconds_per_full_window": full_window,
        "unallocated_workload_seconds_per_full_window":
            budget["workload_budget_seconds"] - full_window,
        "windows": windows,
        "queue_priced": True,
        "queue_approved": False,
        "queue_generated": False,
        "dispatch_ready": False,
        "scientific_effect_qualification": False,
        "native_scientific_qualification": False,
        "candidate_methods_tested": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "contract", "admission", "population", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    output.relative_to(root)
    if args.contract.resolve() != root / (
            "docs/research-math-20261006/"
            "actionbench-full128-queue-pricing-contract.json"):
        raise ValueError("Canonical queue-pricing contract path required")
    if args.admission.resolve() != root / (
            "inputs/complete-unit-admissions/complete-lowram-r7.json"):
        raise ValueError("Canonical complete-unit admission path required")
    if args.population.resolve() != root / POPULATION_PATH:
        raise ValueError("Canonical ActionBench population path required")
    if output != root / "inputs/actionbench-full128-queue/pricing.json":
        raise ValueError("Canonical queue-pricing output target required")
    if output.exists():
        raise FileExistsError("Queue pricing is single-use")
    contract = json.loads(args.contract.read_text())
    admission = json.loads(args.admission.read_text())
    population = json.loads(args.population.read_text())
    result = build_pricing_manifest(root, contract, admission, population)
    result["contract_ref"] = file_ref(root, args.contract)
    result["admission_ref"] = file_ref(root, args.admission)
    result["population_ref"] = file_ref(root, args.population)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False,
                                 allow_nan=False) + "\n")
    print(json.dumps({
        "pricing_ref": file_ref(root, output),
        "status": result["status"],
        "units_per_window": result["units_per_window"],
        "window_count": result["window_count"],
        "queue_approved": False,
        "queue_generated": False,
        "dispatch_ready": False,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
