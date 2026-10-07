"""Derive the retained r9 Full128 reconciliation from exact harness state.

This builder performs no model execution, scoring, queue approval, or dispatch.
It refuses aggregate-only STATUS evidence: every disposition is copied from a
hash-bound research-harness state snapshot and then revalidated by the window
compiler before the sidecar is written.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from prepare_actionbench_full128_window import (
    R9_CAMPAIGN_PLAN_PATH,
    R9_RECONCILIATION_PROFILE,
    R9_STATE_SNAPSHOT_PATH,
    R9_STATUS_SNAPSHOT_PATH,
    RETAINED_DISPOSITIONS,
    validate_active_batch_reconciliation,
)
from research_math.control_scoring import file_ref


PRICING_PATH = "inputs/actionbench-full128-queue/pricing.json"
OUTPUT_PATH = (
    "inputs/actionbench-full128-queue/active-batch-reconciliation.json")


def build_reconciliation(*, root: Path, pricing_path: Path,
                         campaign_path: Path, state_path: Path,
                         status_path: Path, output_path: Path,
                         required_profile: dict | None = None) -> dict:
    """Write one compiler-admissible sidecar derived from harness task state."""
    root = Path(root).resolve()
    profile = (R9_RECONCILIATION_PROFILE if required_profile is None
               else required_profile)
    expected = {
        "pricing": root / PRICING_PATH,
        "campaign": root / profile["campaign_plan_path"],
        "state": root / profile["state_snapshot_path"],
        "status": root / profile["status_snapshot_path"],
        "output": root / OUTPUT_PATH,
    }
    actual = {
        "pricing": Path(pricing_path).resolve(),
        "campaign": Path(campaign_path).resolve(),
        "state": Path(state_path).resolve(),
        "status": Path(status_path).resolve(),
        "output": Path(output_path).resolve(),
    }
    for name, expected_path in expected.items():
        if actual[name] != expected_path:
            raise ValueError("Canonical reconciliation path required: " + name)
    if actual["output"].exists():
        raise FileExistsError("Preserve existing active-batch reconciliation")

    pricing = json.loads(actual["pricing"].read_text())
    state = json.loads(actual["state"].read_text())
    status = json.loads(actual["status"].read_text())
    start, stop = profile["population_range"]
    task_states = state.get("tasks")
    expected_task_ids = [f"population-{index:03d}"
                         for index in range(start, stop)]
    if (state.get("format") != "research-harness-state-v1" or
            state.get("batch_id") != profile["run_id"] or
            state.get("plan_digest") != profile["plan_digest"] or
            not isinstance(task_states, dict) or
            set(task_states) != set(expected_task_ids)):
        raise ValueError("Exact complete retained harness state required")

    flattened = [uid for window in pricing.get("windows", [])
                 for uid in window.get("uids", [])]
    dispositions = []
    for index, task_id in zip(range(start, stop), expected_task_ids):
        task = task_states[task_id]
        task_status = task.get("status") if isinstance(task, dict) else None
        if task_status not in RETAINED_DISPOSITIONS:
            raise ValueError("Unsupported retained harness task status")
        if index >= len(flattened):
            raise ValueError("Retained task lies outside priced population")
        dispositions.append({
            "population_index": index,
            "uid": flattened[index],
            "status": task_status,
        })

    population_window = status.get("population_window")
    if not isinstance(population_window, dict):
        raise ValueError("Exact population status snapshot required")
    result = {
        "kind": "actionbench-full128-active-batch-reconciliation",
        "version": "1.0.0",
        "status": "reconciled_for_plan_generation",
        "pricing_ref": file_ref(root, actual["pricing"]),
        "population_uid_sha256": pricing.get("population_uid_sha256"),
        "observed_at": population_window.get("observed_at"),
        "source_runs": [{
            "run_id": profile["run_id"],
            "plan_digest": profile["plan_digest"],
            "population_start_index": start,
            "population_stop_index_exclusive": stop,
            "campaign_plan_ref": file_ref(root, actual["campaign"]),
            "status_snapshot_ref": file_ref(root, actual["status"]),
            "state_snapshot_ref": file_ref(root, actual["state"]),
            "dispositions": dispositions,
        }],
        "queue_approved": False,
        "dispatch_ready": False,
        "scientific_effect_qualification": False,
        "candidate_methods_tested": False,
    }
    windows = pricing.get("windows", [])
    nonoverlapping = [window for window in windows
                      if window.get("population_start_index", -1) >= stop]
    if not nonoverlapping:
        raise ValueError("No nonoverlapping priced window available")
    validate_active_batch_reconciliation(
        root, pricing, result, nonoverlapping[0], profile)
    actual["output"].parent.mkdir(parents=True, exist_ok=True)
    actual["output"].write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    result = build_reconciliation(
        root=root,
        pricing_path=root / PRICING_PATH,
        campaign_path=root / R9_CAMPAIGN_PLAN_PATH,
        state_path=root / R9_STATE_SNAPSHOT_PATH,
        status_path=root / R9_STATUS_SNAPSHOT_PATH,
        output_path=root / OUTPUT_PATH,
    )
    print(json.dumps({
        "output": OUTPUT_PATH,
        "source_run_id": result["source_runs"][0]["run_id"],
        "disposition_count": len(result["source_runs"][0]["dispositions"]),
        "execution_started": False,
        "queue_approved": False,
        "dispatch_ready": False,
        "scientific_effect_qualification": False,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
