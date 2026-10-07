"""Compile one priced Full128 window into a single-GPU harness plan.

This builder performs no model execution, official scoring, queue approval, or
dispatch.  It accepts only the canonical, recomputed pricing receipt and emits
one no-retry native plan per UID under one serial outer harness.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from prepare_complete_unit_admission import reference_closure
from research_math.actionbench_full128_unit import (
    validate_pricing_shape,
    verify_pricing_receipt,
)
from research_math.complete_unit_contract import complete_unit_output_paths
from research_math.control_scoring import file_ref, resolve_ref


HARD_WINDOW_SECONDS = 28800
COLLECTION_RESERVE_SECONDS = 1800
WORKLOAD_BUDGET_SECONDS = 27000
CURRENT_UNIT_TIMEOUT_SECONDS = 1664
CURRENT_UNITS_PER_WINDOW = 16
RETAINED_DISPOSITIONS = {"completed", "running", "pending", "failed"}
R9_RETAINED_RANGE = (1, 10)
R9_PLAN_DIGEST = "6bfec342ca9b2eb5b3f8174e2cf9be597d4b540da6615f62cd13bb48a0011ac1"
R9_CAMPAIGN_PLAN_PATH = (
    "docs/research-math-20261006/longgoal-20261007/resumed-evidence-r9/"
    "4d-longgoal-r9/plans/population-gpu-current-r9/harness.json")
R9_STATUS_SNAPSHOT_PATH = (
    "docs/research-math-20261006/longgoal-20261007/resumed-evidence-r9/"
    "4d-longgoal-r9/runs/harness/population-gpu-current-r9/"
    "status-snapshot.json")
R9_STATE_SNAPSHOT_PATH = (
    "docs/research-math-20261006/longgoal-20261007/resumed-evidence-r9/"
    "4d-longgoal-r9/runs/harness/population-gpu-current-r9/state.json")
R9_RECONCILIATION_PROFILE = {
    "run_id": "population-gpu-current-r9",
    "population_range": R9_RETAINED_RANGE,
    "plan_digest": R9_PLAN_DIGEST,
    "campaign_plan_path": R9_CAMPAIGN_PLAN_PATH,
    "status_snapshot_path": R9_STATUS_SNAPSHOT_PATH,
    "state_snapshot_path": R9_STATE_SNAPSHOT_PATH,
}


def _plan_digest(plan: dict) -> str:
    """Match research-autopilot's canonical plan-digest algorithm."""
    payload = {key: value for key, value in plan.items()
               if key != "plan_digest"}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def retained_observation_window(status: dict, state: dict,
                                profile: dict) -> dict:
    """Validate an observer output and expose its bound population summary.

    Older retained evidence stores a ``population_window`` wrapper. The
    installed harness's read-only ``--status`` interface instead returns the
    exact harness state with an observation timestamp and a provenance flag.
    Accept both forms while preserving the official status bytes unchanged.
    """
    window = status.get("population_window")
    if isinstance(window, dict):
        return window

    if (status.get("format") != "research-harness-state-v1" or
            status.get("status_is_retained_observation") is not True):
        raise ValueError("Exact retained observer status required")
    observed_state = {
        key: value for key, value in status.items()
        if key not in {"observed_at", "status_is_retained_observation"}
    }
    if observed_state != state:
        raise ValueError("Official observer status differs from harness state")
    start, stop = profile["population_range"]
    expected_task_ids = [f"population-{index:03d}"
                         for index in range(start, stop)]
    tasks = state.get("tasks")
    if not isinstance(tasks, dict) or set(tasks) != set(expected_task_ids):
        raise ValueError("Official observer status lacks the exact task set")
    counts = {name: 0 for name in RETAINED_DISPOSITIONS}
    for task_id in expected_task_ids:
        task = tasks[task_id]
        disposition = task.get("status") if isinstance(task, dict) else None
        if disposition not in counts:
            raise ValueError("Official observer status has an unsupported task status")
        counts[disposition] += 1
    return {
        "run_id": status.get("batch_id"),
        "plan_digest": status.get("plan_digest"),
        "indices": list(range(start, stop)),
        "observed_at": status.get("observed_at"),
        **counts,
    }


def extend_with_opaque_refs(root: Path, refs: list[dict],
                            paths: list[Path]) -> list[dict]:
    """Add already-validated archived evidence without rebasing nested refs.

    Historical campaign/native JSON keeps paths relative to its original run
    root.  The dedicated reconciliation validator resolves and verifies that
    campaign-to-native edge against the retained archive root.  Rewalking those
    immutable bytes with the current project root would bind a different path.
    """
    merged = {ref["path"]: ref for ref in refs}
    for path in paths:
        ref = file_ref(root, Path(path).resolve())
        if ref["path"] in merged and merged[ref["path"]] != ref:
            raise ValueError("Conflicting archived reconciliation reference")
        merged[ref["path"]] = ref
    return [merged[path] for path in sorted(merged)]


def build_input_ref_closure(root: Path, traversed_paths: list[Path],
                            opaque_paths: list[Path]) -> list[dict]:
    """Build the live closure plus the separately validated archive closure."""
    refs = reference_closure(
        root, [Path(path).resolve() for path in traversed_paths])
    return extend_with_opaque_refs(root, refs, opaque_paths)


def select_priced_window(pricing: dict, window_id: str) -> dict:
    """Return one exact current priced window or fail before plan creation."""
    validate_pricing_shape(pricing)
    if (pricing.get("hard_window_seconds") != HARD_WINDOW_SECONDS or
            pricing.get("collection_reserve_seconds") !=
            COLLECTION_RESERVE_SECONDS or
            pricing.get("workload_budget_seconds") != WORKLOAD_BUDGET_SECONDS or
            pricing.get("unit_timeout_seconds") != CURRENT_UNIT_TIMEOUT_SECONDS or
            pricing.get("units_per_window") != CURRENT_UNITS_PER_WINDOW or
            pricing.get("planned_workload_seconds_per_full_window") != 26624 or
            pricing.get("unallocated_workload_seconds_per_full_window") != 376):
        raise ValueError("Exact admitted Full128 queue price required")
    flattened = [uid for window in pricing["windows"] for uid in window["uids"]]
    if hashlib.sha256(json.dumps(
            flattened, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False).encode()).hexdigest() != \
            pricing.get("population_uid_sha256"):
        raise ValueError("Priced UID order differs from canonical population")
    matches = [window for window in pricing["windows"]
               if window.get("window_id") == window_id]
    if len(matches) != 1:
        raise ValueError("Exactly one priced window ID required")
    window = matches[0]
    start = window["population_start_index"]
    expected = pricing["windows"].index(window) * CURRENT_UNITS_PER_WINDOW
    if (start != expected or window["unit_count"] != CURRENT_UNITS_PER_WINDOW or
            window["population_stop_index_exclusive"] !=
            start + CURRENT_UNITS_PER_WINDOW or
            window["planned_workload_seconds"] !=
            CURRENT_UNITS_PER_WINDOW * CURRENT_UNIT_TIMEOUT_SECONDS or
            window["planned_workload_seconds"] > WORKLOAD_BUDGET_SECONDS):
        raise ValueError("Exact complete priced Full128 window required")
    return window


def validate_active_batch_reconciliation(root: Path, pricing: dict,
                                         reconciliation: dict,
                                         target_window: dict,
                                         required_profile: dict | None = None
                                         ) -> list[Path]:
    """Bind known earlier work and reject any target-window duplication."""
    profile = (R9_RECONCILIATION_PROFILE if required_profile is None
               else required_profile)
    profile_keys = {
        "run_id", "population_range", "plan_digest", "campaign_plan_path",
        "status_snapshot_path", "state_snapshot_path",
    }
    if set(profile) != profile_keys:
        raise ValueError("Exact retained-run validation profile required")
    required = {
        "kind", "version", "status", "pricing_ref",
        "population_uid_sha256", "observed_at", "source_runs",
        "queue_approved", "dispatch_ready", "scientific_effect_qualification",
        "candidate_methods_tested",
    }
    if (set(reconciliation) != required or
            reconciliation.get("kind") !=
            "actionbench-full128-active-batch-reconciliation" or
            reconciliation.get("version") != "1.0.0" or
            reconciliation.get("status") != "reconciled_for_plan_generation" or
            reconciliation.get("population_uid_sha256") !=
            pricing.get("population_uid_sha256") or
            any(reconciliation.get(key) is not False for key in (
                "queue_approved", "dispatch_ready",
                "scientific_effect_qualification", "candidate_methods_tested"))):
        raise ValueError("Exact non-scientific active-batch reconciliation required")
    try:
        observed_at = reconciliation["observed_at"]
        if not isinstance(observed_at, str) or not observed_at.endswith("Z"):
            raise ValueError
        datetime.fromisoformat(observed_at.removesuffix("Z") + "+00:00")
    except (KeyError, TypeError, ValueError):
        raise ValueError("UTC reconciliation observation time required") from None

    pricing_path = resolve_ref(root, reconciliation["pricing_ref"])
    if json.loads(pricing_path.read_text()) != pricing:
        raise ValueError("Reconciliation pricing identity mismatch")
    flattened = [uid for window in pricing["windows"] for uid in window["uids"]]
    source_runs = reconciliation["source_runs"]
    if not isinstance(source_runs, list) or len(source_runs) != 1:
        raise ValueError(
            "This compiler revision accepts exactly the retained r9 handoff; "
            "additional retained runs require a reviewed compiler revision")
    run_ids = [run.get("run_id") for run in source_runs
               if isinstance(run, dict)]
    if run_ids != [profile["run_id"]]:
        raise ValueError("Known active batch is absent from reconciliation")

    resolved = [pricing_path.resolve()]
    occupied_uids: set[str] = set()
    run_keys = {
        "run_id", "plan_digest", "population_start_index",
        "population_stop_index_exclusive", "campaign_plan_ref",
        "status_snapshot_ref", "state_snapshot_ref", "dispositions",
    }
    disposition_keys = {"population_index", "uid", "status"}
    for run in source_runs:
        if set(run) != run_keys:
            raise ValueError("Exact source-run reconciliation schema required")
        digest = run["plan_digest"]
        start = run["population_start_index"]
        stop = run["population_stop_index_exclusive"]
        if (not isinstance(digest, str) or len(digest) != 64 or
                any(character not in "0123456789abcdef" for character in digest) or
                not isinstance(start, int) or not isinstance(stop, int) or
                not 0 <= start < stop <= len(flattened)):
            raise ValueError("Valid source-run identity and population range required")
        if ((start, stop) != tuple(profile["population_range"]) or
                digest != profile["plan_digest"]):
            raise ValueError("Exact retained r9 range and plan digest required")
        campaign_ref = run["campaign_plan_ref"]
        status_ref = run["status_snapshot_ref"]
        state_ref = run["state_snapshot_ref"]
        if (not isinstance(campaign_ref, dict) or
                campaign_ref.get("path") != profile["campaign_plan_path"] or
                not isinstance(status_ref, dict) or
                status_ref.get("path") != profile["status_snapshot_path"] or
                not isinstance(state_ref, dict) or
                state_ref.get("path") != profile["state_snapshot_path"]):
            raise ValueError(
                "Canonical campaign plan, status, and harness-state evidence required")
        campaign_path = resolve_ref(root, campaign_ref)
        status_path = resolve_ref(root, status_ref)
        state_path = resolve_ref(root, state_ref)
        campaign = json.loads(campaign_path.read_text())
        expected_task_ids = [f"population-{index:03d}"
                             for index in range(start, stop)]
        tasks = campaign.get("tasks")
        if (campaign.get("schema_id") != "harness-plan" or
                campaign.get("schema_version") != "1.0.0" or
                campaign.get("batch_id") != run["run_id"] or
                campaign.get("plan_digest") != digest or
                _plan_digest(campaign) != digest or
                not isinstance(tasks, list) or
                [task.get("task_id") for task in tasks
                 if isinstance(task, dict)] != expected_task_ids):
            raise ValueError("Campaign plan does not bind the retained run")
        archive_root = campaign_path.parent.parent.parent
        for index, task in zip(range(start, stop), tasks):
            expected_path = f"plans/{run['run_id']}/{index:03d}/native.json"
            plan_ref = task.get("plan_ref")
            if (not isinstance(plan_ref, dict) or
                    set(plan_ref) != {"path", "sha256"} or
                    plan_ref.get("path") != expected_path):
                raise ValueError("Exact hash-bound retained native plan required")
            native_path = resolve_ref(archive_root, plan_ref)
            native_plan = json.loads(native_path.read_text())
            if (native_plan.get("schema_id") != "experiment-run-plan" or
                    native_plan.get("schema_version") != "1.0.0" or
                    native_plan.get("run_id") !=
                    f"{run['run_id']}-unit-{index:03d}" or
                    native_plan.get("purpose") != "engineering" or
                    native_plan.get("evidence_mode") != "developmental" or
                    native_plan.get("plan_digest") != _plan_digest(native_plan)):
                raise ValueError("Retained native plan identity or digest mismatch")
            resolved.append(native_path.resolve())
        status = json.loads(status_path.read_text())
        state = json.loads(state_path.read_text())
        population_window = retained_observation_window(
            status, state, profile)
        if (not isinstance(population_window, dict) or
                population_window.get("run_id") != run["run_id"] or
                population_window.get("plan_digest") != digest or
                population_window.get("indices") != list(range(start, stop)) or
                population_window.get("observed_at") != observed_at):
            raise ValueError("Status snapshot does not bind the retained run")
        task_states = state.get("tasks")
        if (state.get("format") != "research-harness-state-v1" or
                state.get("batch_id") != run["run_id"] or
                state.get("plan_digest") != digest or
                not isinstance(task_states, dict) or
                set(task_states) != set(expected_task_ids)):
            raise ValueError("Harness state does not bind every retained task")
        resolved.extend((campaign_path.resolve(), status_path.resolve(),
                         state_path.resolve()))
        dispositions = run["dispositions"]
        expected_indices = list(range(start, stop))
        if (not isinstance(dispositions, list) or
                len(dispositions) != len(expected_indices)):
            raise ValueError("Complete source-run dispositions required")
        actual_indices = []
        for disposition in dispositions:
            if (not isinstance(disposition, dict) or
                    set(disposition) != disposition_keys):
                raise ValueError("Exact source-run disposition schema required")
            index = disposition["population_index"]
            uid = disposition["uid"]
            status = disposition["status"]
            if (not isinstance(index, int) or index < 0 or
                    index >= len(flattened) or uid != flattened[index] or
                    status not in RETAINED_DISPOSITIONS):
                raise ValueError("Disposition differs from canonical population")
            actual_indices.append(index)
            task_id = f"population-{index:03d}"
            task_state = task_states.get(task_id)
            if (not isinstance(task_state, dict) or
                    task_state.get("status") != status):
                raise ValueError(
                    "Disposition differs from exact retained harness task state")
            if uid in occupied_uids:
                raise ValueError("Duplicate retained UID disposition")
            occupied_uids.add(uid)
        if actual_indices != expected_indices:
            raise ValueError("Ordered complete source-run range required")
        counts = {status: 0 for status in RETAINED_DISPOSITIONS}
        for disposition in dispositions:
            counts[disposition["status"]] += 1
        if any(population_window.get(status) != count
               for status, count in counts.items()):
            raise ValueError("Disposition counts differ from observed status")

    if occupied_uids.intersection(target_window["uids"]):
        raise ValueError("Target Full128 window overlaps retained work")
    return sorted(set(resolved))


def window_limits(pricing: dict, window: dict) -> dict:
    if window not in pricing.get("windows", []):
        raise ValueError("Window must belong to the verified pricing receipt")
    planned = window.get("planned_workload_seconds")
    if (not isinstance(planned, int) or planned < 1 or
            planned > pricing.get("workload_budget_seconds", -1) or
            pricing.get("hard_window_seconds") - planned <
            pricing.get("collection_reserve_seconds", HARD_WINDOW_SECONDS + 1)):
        raise ValueError("Full128 window does not preserve collection reserve")
    return {
        # Stop workload scheduling at the 27,000-second boundary.  The outer
        # 28,800-second cadence is retained only as the reporting window, so
        # the final 1,800 seconds cannot be consumed by task-start overhead.
        "total_wall_seconds": pricing["workload_budget_seconds"],
        "window_seconds": pricing["hard_window_seconds"],
        "max_parallel_tasks": 1,
        "cpu_cores": 8,
        "ram_mib": 32768,
        "max_gpu_task_seconds": planned,
    }


def native_limits(unit_timeout_seconds: int) -> dict:
    return {
        "max_attempts": 1,
        "max_development_trials": 1,
        "max_confirmation_trials": 0,
        "max_retries_per_trial": 0,
        "wall_time_seconds": unit_timeout_seconds,
        "attempt_timeout_seconds": unit_timeout_seconds,
    }


def validate_native_plan_set(window: dict, plans: list[dict],
                             unit_timeout_seconds: int) -> None:
    """Reject incomplete, reordered, retried, or fallback-bearing units."""
    if len(plans) != CURRENT_UNITS_PER_WINDOW or len(window["uids"]) != len(plans):
        raise ValueError("Exact sixteen-plan Full128 window required")
    limits = native_limits(unit_timeout_seconds)
    for offset, (uid, plan) in enumerate(zip(window["uids"], plans)):
        jobs = plan.get("jobs", [])
        command = jobs[0].get("command", []) if len(jobs) == 1 else []
        expected_index = window["population_start_index"] + offset
        if (plan.get("limits") != limits or len(jobs) != 1 or
                jobs[0].get("trial_id") != f"full128-{expected_index:03d}-{uid}" or
                jobs[0].get("cwd") != "actionmesh" or
                command.count("--uid") != 1 or
                command[command.index("--uid") + 1] != uid or
                any(flag in command for flag in
                    ("--fast", "--low-ram", "--dtype", "--retry"))):
            raise ValueError("Exact ordered no-retry Full128 native plans required")


def build_native_command(args, uid: str, window_id: str,
                         unit_timeout_seconds: int) -> list[str]:
    command = [sys.executable, "-m", "research_math.complete_unit_runner"]
    paths = (
        ("contract", args.contract),
        ("population", args.population),
        ("snapshot-contract", args.snapshot_contract),
        ("snapshot-admission", args.snapshot_admission),
        ("dataset-semantics", args.dataset_semantics),
        ("unit-manifest", args.unit_manifest),
    )
    for flag, path in paths:
        staged = Path("..") / Path(path).resolve().relative_to(args.root.resolve())
        command.extend(("--" + flag, staged.as_posix()))
    for flag, path in (("source-root", args.source_root),
                       ("dataset-root", args.dataset_root),
                       ("weights-root", args.weights_root)):
        command.extend(("--" + flag, str(Path(path).resolve())))
    command.extend((
        "--root", "..",
        "--pricing", (Path("..") / args.pricing.resolve().relative_to(
            args.root.resolve())).as_posix(),
        "--uid", uid,
        "--window-id", window_id,
        "--gpu-uuid", args.gpu_uuid,
        "--wall-seconds", str(unit_timeout_seconds),
        "--output", "unit-output",
    ))
    return command


def _canonical_paths(root: Path, args) -> None:
    expected = {
        "pricing": root / "inputs/actionbench-full128-queue/pricing.json",
        "contract": root /
            "docs/research-math-20261006/actionbench-current-release-unit-contract.json",
        "population": root /
            "actionmesh/research_overnight/assets/actionbench_population.json",
        "snapshot_contract": root /
            "docs/research-math-20261006/actionbench-full128-snapshot-contract.json",
        "snapshot_admission": root /
            "inputs/actionbench-full128-snapshots/admission.json",
        "dataset_semantics": root /
            "inputs/actionbench-full128-snapshots/dataset-semantics.json",
        "unit_manifest": root /
            "inputs/actionbench-full128-snapshots/unit-manifest.json",
        "environment": root / "inputs/native-runtime/environment.json",
        "active_batch_reconciliation": root /
            "inputs/actionbench-full128-queue/active-batch-reconciliation.json",
    }
    for name, path in expected.items():
        if Path(getattr(args, name)).resolve() != path:
            raise ValueError("Canonical Full128 path required: " + name)


def validate_environment_closure(root: Path, environment_path: Path,
                                 environment: dict, gpu_uuid: str) -> list[Path]:
    """Validate the exact current native manifest and its dependency lock."""
    required = {
        "execution_mode", "python_executable", "python_version",
        "python_prefix", "conda_prefix", "gpu_uuid", "packages",
        "dependency_lock_refs", "captured_at", "gpu_identity_source",
        "gpu_identity_verified", "native_contract_qualified", "scope",
    }
    if set(environment) != required:
        raise ValueError("Exact current native environment schema required")
    if (environment["execution_mode"] != "native_host" or
            environment["python_executable"] != sys.executable or
            environment["gpu_uuid"] != gpu_uuid):
        raise ValueError("Fresh current interpreter and physical GPU required")
    packages = environment["packages"]
    if (not isinstance(packages, dict) or
            any(not isinstance(packages.get(name), str) or not packages[name]
                for name in ("numpy", "torch", "trimesh", "scipy", "pytorch3d"))):
        raise ValueError("Required native package identities missing")
    refs = environment["dependency_lock_refs"]
    expected_path = "inputs/native-runtime/dependencies.json"
    if (not isinstance(refs, list) or len(refs) != 1 or
            set(refs[0]) != {"path", "sha256"} or
            refs[0]["path"] != expected_path):
        raise ValueError("Exact native dependency lock reference required")
    dependency_path = resolve_ref(root, refs[0])
    dependency = json.loads(dependency_path.read_text())
    dependency_keys = {
        "conda_packages", "conda_prefix", "kind", "packages",
        "python_executable", "python_prefix", "python_version", "scope",
        "version",
    }
    if (set(dependency) != dependency_keys or
            dependency["kind"] != "installed-native-dependency-inventory" or
            dependency["version"] != "1.0.0" or
            dependency["python_executable"] != environment["python_executable"] or
            dependency["python_prefix"] != environment["python_prefix"] or
            dependency["python_version"] != environment["python_version"] or
            any(dependency["packages"].get(name) != version
                for name, version in packages.items()) or
            dependency["conda_prefix"] != environment["conda_prefix"] or
            not isinstance(dependency["conda_packages"], list) or
            not dependency["conda_packages"]):
        raise ValueError("Native dependency inventory differs from environment")
    return [environment_path.resolve(), dependency_path.resolve()]


def build_plan(args) -> dict:
    root = args.root.resolve()
    plan_dir = args.plan_dir.resolve()
    plan_dir.relative_to(root)
    _canonical_paths(root, args)
    if plan_dir.exists():
        raise FileExistsError("Preserve existing Full128 window plan")
    if not args.gpu_uuid.startswith("GPU-"):
        raise ValueError("Physical GPU UUID required")
    for path in (args.source_root, args.dataset_root, args.weights_root):
        if not path.resolve().is_dir():
            raise FileNotFoundError(path)

    pricing = json.loads(args.pricing.read_text())
    verified, admission = verify_pricing_receipt(root, pricing)
    window = select_priced_window(verified, args.window_id)
    reconciliation = json.loads(args.active_batch_reconciliation.read_text())
    reconciliation_paths = validate_active_batch_reconciliation(
        root, verified, reconciliation, window)
    environment = json.loads(args.environment.read_text())
    environment_paths = validate_environment_closure(
        root, args.environment, environment, args.gpu_uuid)

    scripts = args.skill_dir.resolve() / "scripts"
    if not (scripts / "run_harness.py").is_file():
        raise ValueError("Complete installed research-autopilot skill required")
    sys.path.insert(0, str(scripts))
    import run_experiments as native
    import run_harness as harness

    seed_paths = [
        args.pricing, args.contract, args.population, args.snapshot_contract,
        args.snapshot_admission, args.dataset_semantics, args.unit_manifest,
    ]
    seed_paths.extend(environment_paths)
    input_refs = build_input_ref_closure(
        root, seed_paths,
        [args.active_batch_reconciliation, *reconciliation_paths])

    sources = sorted((root / "actionmesh/research_math").rglob("*.py"))
    sources.extend(root / "actionmesh" / name for name in (
        "official_actionbench_adapter.py",
        "deterministic_actionbench_entry.py",
        "research_census_eval.py",
        "prepare_actionbench_full128_window.py",
    ))
    missing = [path for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Full128 window source: " + str(missing[0]))
    code_refs = [file_ref(root, path) for path in sorted(set(sources))]

    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(
        ["git", "diff", "HEAD", "--binary"], cwd=root, check=True,
        capture_output=True).stdout
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"], cwd=root,
        check=True, capture_output=True, text=True).stdout.splitlines()
    if any(path.endswith(".py") for path in untracked):
        raise ValueError("Commit all executable Python sources before Local acceptance")

    plan_dir.mkdir(parents=True, exist_ok=False)
    dirty_path = plan_dir / "dirty.patch"
    dirty_path.write_bytes(dirty)
    dirty_ref = file_ref(root, dirty_path)
    refs = {ref["path"]: ref for ref in input_refs}
    refs[dirty_ref["path"]] = dirty_ref
    tasks = []
    unit_plans = []
    unit_timeout = verified["unit_timeout_seconds"]
    for offset, uid in enumerate(window["uids"]):
        population_index = window["population_start_index"] + offset
        unit_run_id = f"{args.run_id}-unit-{population_index:03d}"
        command = build_native_command(args, uid, args.window_id, unit_timeout)
        plan = native.make_plan(
            root,
            run_id=unit_run_id,
            purpose="engineering",
            evidence_mode="developmental",
            jobs=[{
                "trial_id": f"full128-{population_index:03d}-{uid}",
                "command": command,
                "cwd": "actionmesh",
                "input_refs": list(refs.values()),
                "code_refs": code_refs,
                "output_paths": complete_unit_output_paths(uid),
                "seed": 42,
                "group": "engineering",
                "arm_role": "full128-current-release-three-arm-unit",
            }],
            provenance={
                "git_revision": revision,
                "git_refs": [dirty_ref],
                "model_revision": "bound by admitted complete-unit template",
                "data_revision": verified["population"]["revision"],
                "environment_digest": file_ref(root, args.environment)["sha256"],
            },
            limits=native_limits(unit_timeout),
        )
        unit_plans.append(plan)

    validate_native_plan_set(window, unit_plans, unit_timeout)
    for offset, (uid, plan) in enumerate(zip(window["uids"], unit_plans)):
        population_index = window["population_start_index"] + offset
        unit_dir = plan_dir / f"{population_index:03d}"
        unit_dir.mkdir()
        native_path = unit_dir / "native.json"
        native_path.write_text(json.dumps(plan, indent=2) + "\n")
        tasks.append({
            "task_id": f"full128-{population_index:03d}",
            "idea_id": "baseline-qualification",
            "depends_on": [],
            "priority": 128 - population_index,
            "plan_ref": file_ref(root, native_path),
            "resources": {
                "cpu_cores": 8,
                "ram_mib": 32768,
                "gpu_count": 1,
                "gpu_peak_mib": None,
                "allow_gpu_share": False,
                "memory_profile_ref": None,
                "exclusive_keys": ["actionbench-full128-baseline"],
            },
        })
    outer = harness.make_plan(
        root,
        batch_id=args.run_id,
        tasks=tasks,
        limits=window_limits(verified, window),
        gpus={
            "uuids": [args.gpu_uuid],
            "safety_margin_mib": 1024,
            "max_tasks_per_gpu": 1,
        },
    )
    target = plan_dir / "harness.json"
    target.write_text(json.dumps(outer, indent=2) + "\n")
    return {
        "plan": str(target),
        "plan_digest": outer["plan_digest"],
        "execution_started": False,
        "window_id": args.window_id,
        "unit_count": len(window["uids"]),
        "queue_priced": True,
        "queue_generated": True,
        "queue_approved": False,
        "dispatch_ready": False,
        "scientific_effect_qualification": False,
        "native_scientific_qualification": False,
        "candidate_methods_tested": False,
        "source_admission_run_id": admission["run_id"],
        "reconciled_source_run_ids": [
            run["run_id"] for run in reconciliation["source_runs"]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
            "root", "plan-dir", "skill-dir", "pricing", "contract",
            "population", "snapshot-contract", "snapshot-admission",
            "dataset-semantics", "unit-manifest", "environment",
            "active-batch-reconciliation",
            "source-root", "dataset-root", "weights-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--window-id", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    print(json.dumps(build_plan(parser.parse_args())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
