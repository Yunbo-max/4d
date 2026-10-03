"""Audit and aggregate cached census results; never import models or run metrics.

Primary statistics use seed 42 once per asset. Seed 43 is a two-asset repeated
seed control, never an extra independent asset. By default a metadata checkpoint
may omit sequence.npz: the matching evaluator/generator hash declarations remain
auditable, but this is explicitly weaker than verifying artifact bytes. Use
--require-artifacts on the complete remote result tree to require byte checks.
Existing output is immutable unless --allow-update is explicitly supplied.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile


METRICS = ("cd_3d", "cd_4d", "cd_motion")
EXPECTED_PROTOCOL = {"n_pts_chamfer": 100000, "n_pts_icp": 10000,
                     "icp_initial_rotations": 24, "icp_iterations": 200,
                     "icp_learning_rate": .01, "sampling_seed": 44,
                     "cd_query_points_per_direction": 10000,
                     "cd_query_subsampling_seeds": {"predicted": 44, "ground_truth": 45}}
FAILED = {"failed", "error", "timeout", "timed_out", "dependency_failed", "launch_failed",
          "missing_output", "refused_existing_output", "cancelled", "terminated"}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def read(path, inventory):
    path = Path(path)
    if not path.is_file():
        return None
    content = path.read_bytes()
    inventory[str(path)] = hashlib.sha256(content).hexdigest()
    return json.loads(content)


def require(condition, errors, code, detail=None):
    if not condition:
        errors.append({"code": code, "detail": detail})


def same_path_tail(path, *parts):
    return Path(str(path)).parts[-len(parts):] == tuple(parts)


def command_arg(argv, flag):
    try:
        return argv[argv.index(flag) + 1]
    except (ValueError, IndexError):
        return None


def queue_data(root, inventory):
    jobs, queues = [], []
    for path in sorted(root.glob("queue*/queue.json")):
        data = read(path, inventory)
        queues.append({"path": str(path), "status": data.get("status"),
                       "started_utc": data.get("started_utc"), "finished_utc": data.get("finished_utc"),
                       "max_concurrency": data.get("max_concurrency"),
                       "elapsed_seconds": data.get("elapsed_seconds")})
        for job in data.get("jobs", []):
            jobs.append(dict(job, queue_source=str(path)))
    return jobs, queues


def matching_jobs(jobs, entry, kind):
    chosen = []
    for job in jobs:
        argv = job.get("argv", [])
        path = command_arg(argv, "--case-dir") if kind == "evaluation" else job.get("output")
        if kind == "generation" and command_arg(argv, "--case-dir"):
            continue
        if path and same_path_tail(path, "cases", entry["case_dir"]):
            chosen.append(job)
    return chosen


def stage_status(matched):
    if not matched:
        return None
    # A later retry of the same selected directory wins; all queue records remain.
    return sorted(matched, key=lambda row: row.get("started_utc", ""))[-1].get("status")


def audit_case(entry, root, manifest, inventory, jobs, require_artifacts=False):
    uid, case_id, seed, directory = entry["uid"], entry["case_id"], entry["seed"], entry["case_dir"]
    base = root / "cases" / directory
    errors, warnings = [], []
    row = {"case_id": case_id, "uid": uid, "seed": seed, "case_dir": directory,
           "status": "pending", "audit_errors": errors, "audit_warnings": warnings,
           "artifact_bytes_verified": False, "failed_attempts": []}
    generation_jobs = matching_jobs(jobs, entry, "generation")
    evaluation_jobs = matching_jobs(jobs, entry, "evaluation")
    row["queue_generation_status"] = stage_status(generation_jobs)
    row["queue_evaluation_status"] = stage_status(evaluation_jobs)
    for failed_dir in entry.get("failed_attempt_dirs", []):
        failed_path = root / "cases" / failed_dir / "report.json"
        failed = read(failed_path, inventory)
        row["failed_attempts"].append({"directory": failed_dir, "report_present": failed is not None,
            "report_sha256": inventory.get(str(failed_path)),
            **({key: failed.get(key) for key in ("status", "uid", "seed", "error_type", "error", "resources")}
               if failed else {})})
        if failed:
            require(failed.get("uid") == uid and failed.get("seed") == seed, errors,
                    "failed_attempt_identity_mismatch", failed_dir)
            require(failed.get("status") in FAILED, errors, "declared_failed_attempt_not_failed", failed_dir)
    generation = read(base / "report.json", inventory)
    command = read(base / "command.json", inventory)
    progress = read(base / "progress.json", inventory)
    row["last_progress"] = progress[-1] if isinstance(progress, list) and progress else None
    if command:
        require(command.get("uid") == uid and command.get("seed") == seed,
                errors, "command_identity_mismatch")
        argv = command.get("argv", [])
        require(command_arg(argv, "--uid") == uid and command_arg(argv, "--seed") == str(seed),
                errors, "command_argv_identity_mismatch")
        require(same_path_tail(command_arg(argv, "--output"), "cases", directory),
                errors, "command_output_directory_mismatch")
    if generation:
        require(generation.get("uid") == uid and generation.get("seed") == seed,
                errors, "generation_identity_mismatch")
        row["generation_status"] = generation.get("status")
        row["resources"] = generation.get("resources", {})
        row["stage_seconds"] = generation.get("stage_seconds", {})
        row["generation_started_utc"] = generation.get("started_utc")
        row["generation_ended_utc"] = generation.get("ended_utc")
        if generation.get("status") in FAILED:
            row.update(status="generation_failed", generation_error=generation.get("error"))
        elif generation.get("status") == "completed":
            row["status"] = "awaiting_evaluation"
            require(same_path_tail(generation.get("output"), "cases", directory),
                    errors, "generation_output_directory_mismatch")
            expected = manifest.get("inference", {})
            for key, expected_value in {"frames": expected.get("frames", 16),
                "stage0_steps": expected.get("stage0_steps", 100),
                "stage1_steps": expected.get("stageI_steps", 30),
                "guidance_scale": expected.get("cfg", 7.5), "training": False, "steering": False}.items():
                require(generation.get(key) == expected_value, errors, "generation_protocol_mismatch", key)
            input_path = base / "inputs.json"
            inputs = read(input_path, inventory)
            require(inputs is not None, errors, "missing_input_hash_provenance")
            if inputs:
                require(inputs.get("uid") == uid, errors, "input_uid_mismatch")
                require(generation.get("sha256", {}).get("inputs.json") == inventory[str(input_path)],
                        errors, "input_provenance_hash_mismatch")
                records = inputs.get("records", [])
                require([item.get("index") for item in records] == list(range(16)),
                        errors, "input_frame_indices_mismatch")
                pinned = {item["path"]: item.get("lfs", {}).get("oid") for item in manifest.get("files", [])}
                for item in records:
                    name = f"{item.get('index', -1):02d}.png"
                    key = f"data/{uid}/imgs/{name}"
                    require(bool(pinned.get(key)) and item.get("source_sha256") == pinned[key],
                            errors, "input_pinned_hash_mismatch", key)
                    require(same_path_tail(item.get("source"), uid, "imgs", name),
                            errors, "input_source_path_mismatch", name)
    elif row["queue_generation_status"] in FAILED:
        row["status"] = "generation_failed"
    elif row["queue_generation_status"] in ("running", "starting"):
        row["status"] = "generation_running"
    elif command or progress:
        row["status"] = "generation_started_snapshot_no_final_status"
    elif row["queue_generation_status"] == "completed":
        require(False, errors, "queue_completed_without_generation_report")
    elif row["queue_generation_status"] == "not_started":
        row["status"] = "not_started"
    evaluation_path = root / "evaluations" / (case_id + ".json")
    evaluation = read(evaluation_path, inventory)
    if evaluation:
        row["evaluation_sha256"] = inventory[str(evaluation_path)]
        cases = evaluation.get("cases", [])
        require(len(cases) == 1, errors, "evaluation_requires_exactly_one_case")
        if len(cases) == 1:
            evaluated = cases[0]
            require(evaluated.get("uid") == uid, errors, "evaluation_uid_mismatch")
            require(evaluated.get("case_id") in (case_id, directory), errors, "evaluation_case_id_mismatch")
            require(same_path_tail(evaluated.get("case_dir"), "cases", directory),
                    errors, "evaluation_case_directory_mismatch")
            row["evaluation_status"] = evaluated.get("status")
            row["evaluation_seconds"] = evaluated.get("elapsed_seconds")
            if evaluated.get("status") == "success":
                require(generation is not None and generation.get("status") == "completed",
                        errors, "evaluation_without_completed_generation")
                require(row["queue_generation_status"] not in FAILED,
                        errors, "completed_generation_but_queue_failure")
                require(row["queue_evaluation_status"] not in FAILED,
                        errors, "successful_evaluation_but_queue_failure")
                for key, value in EXPECTED_PROTOCOL.items():
                    require(evaluation.get("protocol", {}).get(key) == value,
                            errors, "evaluation_protocol_mismatch", key)
                provenance = evaluated.get("inputs", {})
                seq_hash = provenance.get("sequence", {}).get("sha256")
                require(bool(seq_hash) and seq_hash == (generation or {}).get("sha256", {}).get("sequence.npz"),
                        errors, "sequence_hash_chain_mismatch")
                require(provenance.get("generation_report", {}).get("sha256") == inventory.get(str(base / "report.json")),
                        errors, "generation_report_bytes_hash_mismatch")
                for label, filename in (("sequence", "sequence.npz"), ("generation_report", "report.json")):
                    require(same_path_tail(provenance.get(label, {}).get("path"), "cases", directory, filename),
                            errors, "evaluation_provenance_path_mismatch", label)
                gt = provenance.get("ground_truth", {})
                pinned_gt = next((item.get("lfs", {}).get("oid") for item in manifest.get("files", [])
                                  if item["path"] == f"data/{uid}/surfaces.npy"), None)
                require(bool(pinned_gt) and gt.get("sha256") == pinned_gt, errors, "ground_truth_pinned_hash_mismatch")
                require(same_path_tail(gt.get("path"), uid, "surfaces.npy"), errors, "ground_truth_uid_path_mismatch")
                sequence = base / "sequence.npz"
                if sequence.is_file():
                    actual = sha(sequence)
                    inventory[str(sequence)] = actual
                    require(actual == seq_hash, errors, "sequence_bytes_hash_mismatch")
                    row["artifact_bytes_verified"] = actual == seq_hash
                elif require_artifacts:
                    require(False, errors, "sequence_bytes_missing_required")
                else:
                    warnings.append("sequence.npz absent from checkpoint; only matching generation/evaluation hash attestations verified")
                evaluator_source = root / "source" / "research_census_eval.py"
                if evaluator_source.is_file():
                    inventory[str(evaluator_source)] = sha(evaluator_source)
                    require(inventory[str(evaluator_source)] == evaluation.get("evaluator_sha256"),
                            errors, "evaluator_source_hash_mismatch")
                else:
                    warnings.append("evaluator source not included in checkpoint")
                row["evaluation_contract_hash"] = canonical_hash({"protocol": evaluation.get("protocol"),
                    "official_source_hashes": evaluation.get("official_source", {}).get("sha256"),
                    "compatibility_patch": evaluation.get("official_source", {}).get("runtime_compatibility_patch"),
                    "evaluator_sha256": evaluation.get("evaluator_sha256"), "packages": evaluation.get("packages")})
                for metric in METRICS:
                    value = evaluated.get(metric)
                    require(isinstance(value, (int, float)) and not isinstance(value, bool)
                            and math.isfinite(value) and value >= 0, errors, "invalid_metric", metric)
                    row[metric] = value
                row["status"] = "success"
            elif evaluated.get("status") in FAILED:
                row.update(status="evaluation_failed", evaluation_error=evaluated.get("error"))
    elif row["status"] == "awaiting_evaluation":
        state = row["queue_evaluation_status"]
        if state in ("running", "starting"):
            row["status"] = "evaluation_running"
        elif state in FAILED:
            row["status"] = "evaluation_failed"
    if errors:
        row["status"] = "audit_error"
    row["audit_status"] = "failed" if errors else (
        "artifact_and_metadata_verified" if row["artifact_bytes_verified"] else
        "metadata_chain_verified" if row["status"] == "success" else "incomplete_or_failed_execution")
    return row


def metric_summary(rows):
    good = [row for row in rows if row["status"] == "success"]
    return {"n_planned": len(rows), "n_success": len(good),
            "n_assets": len({r["uid"] for r in rows}),
            "status_counts": dict(Counter(r["status"] for r in rows)),
            "success_case_ids": [r["case_id"] for r in good],
            "means_success_only": {key: statistics.mean(r[key] for r in good) if good else None for key in METRICS},
            "missing_or_failed_are_not_zero": True}


def seed_comparison(rows):
    lookup = {(row["uid"], row["seed"]): row for row in rows}
    planned = sorted({row["uid"] for row in rows if row["seed"] == 43})
    pairs = []
    for uid in planned:
        a, b = lookup.get((uid, 42)), lookup.get((uid, 43))
        if not a or not b or a["status"] != "success" or b["status"] != "success":
            continue
        pairs.append({"uid": uid, "seed42": {k: a[k] for k in METRICS},
                      "seed43": {k: b[k] for k in METRICS},
                      "delta_seed43_minus_seed42": {k: b[k] - a[k] for k in METRICS}})
    return {"n_planned_assets": len(planned), "n_complete_pairs": len(pairs), "pairs": pairs,
            "interpretation": "Two-asset paired seed sensitivity only; no population significance or best-of-N claim"}


def telemetry(root, inventory):
    samples, malformed, source_count = {}, [], 0
    for path in sorted(root.glob("queue*/gpu-samples.jsonl")):
        content = path.read_bytes()
        inventory[str(path)] = hashlib.sha256(content).hexdigest()
        for index, line in enumerate(content.decode().splitlines()):
            try:
                item = json.loads(line)
                key = (item["utc"], item.get("uuid", item.get("gpu_index")))
                timestamp(item["utc"])
                if item.get("error"):
                    continue
                source_count += 1
                merged = samples.setdefault(key, dict(item, processes={}))
                merged["used_mib"] = max(merged.get("used_mib", 0), item.get("used_mib", 0))
                for process in item.get("processes", []):
                    if process.get("owner_job_id") and process.get("memory_mib", 0) > 0:
                        merged["processes"][(process.get("pid"), process["owner_job_id"])] = process
            except (ValueError, KeyError, TypeError):
                malformed.append({"path": str(path), "line": index + 1})
    records = []
    for (_, gpu), item in sorted(samples.items()):
        owners = {process["owner_job_id"] for process in item["processes"].values()}
        records.append({"utc": item["utc"], "gpu": gpu, "used_mib": item.get("used_mib", 0),
                        "owned": len(owners), "generation": sum(s.startswith("gen-") for s in owners),
                        "evaluation": sum(s.startswith("eval-") for s in owners)})
    intervals = []
    by_gpu = defaultdict(list)
    for record in records:
        by_gpu[record["gpu"]].append(record)
    observed_seconds, concurrent_seconds = 0., 0.
    for gpu, group in by_gpu.items():
        group.sort(key=lambda x: x["utc"])
        for start, end in zip(group, group[1:]):
            seconds = timestamp(end["utc"]) - timestamp(start["utc"])
            if not 0 < seconds <= 10:
                continue
            observed_seconds += seconds
            # Both interval endpoints must observe >=2 owned resident processes.
            if min(start["owned"], end["owned"]) >= 2:
                concurrent_seconds += seconds
                signature = (min(start["generation"], end["generation"]),
                             min(start["evaluation"], end["evaluation"]))
                if intervals and intervals[-1]["end_utc"] == start["utc"] and intervals[-1]["gpu"] == gpu and tuple(intervals[-1]["minimum_generation_evaluation_counts"]) == signature:
                    intervals[-1]["end_utc"] = end["utc"]
                    intervals[-1]["seconds"] += seconds
                else:
                    intervals.append({"gpu": gpu, "start_utc": start["utc"], "end_utc": end["utc"],
                                      "seconds": seconds, "minimum_generation_evaluation_counts": list(signature)})
    return {"raw_samples": source_count, "deduplicated_timestamp_gpu_samples": len(records),
            "sampled_physical_peak_mib": max((r["used_mib"] for r in records), default=None),
            "max_observed_owned_gpu_jobs": max((r["owned"] for r in records), default=0),
            "observed_bracket_seconds": observed_seconds, "concurrent_bracket_seconds": concurrent_seconds,
            "concurrent_intervals": intervals, "malformed_lines": malformed,
            "measurement_scope": "Distinct owned job IDs with GPU-resident processes at telemetry samples; bracket gaps >10s excluded; not proof of simultaneous kernels or throughput improvement"}


def aggregate(manifest_path, root, require_artifacts=False):
    inventory = {}
    manifest = read(manifest_path, inventory)
    entries = manifest.get("cases", [])
    if not entries or len({r["case_id"] for r in entries}) != len(entries):
        raise ValueError("Manifest cases must be nonempty with unique case_id")
    if len({(r["uid"], r["seed"]) for r in entries}) != len(entries):
        raise ValueError("Retries must be attempt metadata, not duplicate UID/seed planned cases")
    for entry in entries:
        if any(Path(str(entry[key])).name != str(entry[key]) for key in ("uid", "case_id", "case_dir")):
            raise ValueError("Manifest identities and case directories must be single path components")
    jobs, queues = queue_data(root, inventory)
    rows = [audit_case(entry, root, manifest, inventory, jobs, require_artifacts) for entry in entries]
    contracts = {r["evaluation_contract_hash"] for r in rows if r["status"] == "success"}
    cohort_errors = []
    if len(contracts) > 1:
        cohort_errors.append("Inconsistent evaluator/protocol/package contracts across successful runs")
        for row in rows:
            if row["status"] == "success":
                row["status"] = "audit_error"
                row["audit_status"] = "failed"
                row["audit_errors"].append({"code": "cohort_evaluation_contract_mismatch", "detail": None})
    primary = [r for r in rows if r["seed"] == 42]
    selected = set(manifest.get("uids", []))
    if selected and {r["uid"] for r in primary} != selected:
        raise ValueError("Primary seed42 cohort differs from frozen UID list")
    physical = telemetry(root, inventory)
    generation_resources = [r.get("resources", {}) for r in rows if r.get("generation_status") == "completed"]
    good_times = [r["elapsed_seconds"] for r in generation_resources if r.get("elapsed_seconds") is not None]
    failed_attempts = [dict(item, case_id=row["case_id"]) for row in rows for item in row["failed_attempts"]]
    report = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path), "manifest_sha256": inventory[str(manifest_path)],
        "results_root": str(root), "aggregator_sha256": sha(Path(__file__)),
        "audit": {"status": "failed" if cohort_errors or any(r["audit_errors"] for r in rows) else "passed_available_evidence",
                  "errors": cohort_errors, "n_case_audit_errors": sum(bool(r["audit_errors"]) for r in rows),
                  "n_verified_artifact_bytes": sum(r["artifact_bytes_verified"] for r in rows),
                  "require_artifacts": require_artifacts,
                  "scope": "Independent cached JSON/identity/hash/denominator audit; no numerical metric recomputation"},
        "n_planned_cases": len(rows), "n_independent_assets": len({r["uid"] for r in rows}),
        "all_planned_run_status_counts": dict(Counter(r["status"] for r in rows)),
        "primary_baseline_seed42": metric_summary(primary), "seed_control": seed_comparison(rows),
        "case_results": rows, "failed_attempts": failed_attempts,
        "failed_attempt_count_not_extra_assets": len(failed_attempts),
        "resources": {"n_completed_generation_reports": len(generation_resources),
            "sum_generation_process_seconds_completed": sum(good_times),
            "mean_generation_process_seconds_completed": statistics.mean(good_times) if good_times else None,
            "sum_evaluation_function_seconds_completed": sum(r.get("evaluation_seconds", 0) or 0 for r in rows if r["status"] == "success"),
            "max_torch_allocated_bytes": max((r.get("peak_allocated_bytes", 0) for r in generation_resources), default=None),
            "max_torch_reserved_bytes": max((r.get("peak_reserved_bytes", 0) for r in generation_resources), default=None),
            "process_seconds_not_wall_time": True},
        "queues": queues, "telemetry": physical,
        "limitations": ["Development census: eight selected assets, no population significance",
                        "Primary seed42 metrics condition on audited successful cases; denominator and missing cases remain visible",
                        "Repeated seeds and failed retries do not increase independent asset count",
                        "No method quality claim, intervention result, or downstream causal attribution follows from baseline metrics"]}
    report["source_files_sha256"] = inventory
    report["source_snapshot_sha256"] = canonical_hash(inventory)
    return report


def save(path, report, allow_update=False):
    if path.exists() and not allow_update:
        raise FileExistsError(f"Refusing to replace snapshot without --allow-update: {path}")
    if path.exists():
        report["replaces_summary_sha256"] = sha(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def self_test():
    # Scientific aggregation tests use tiny fixture bytes, not a fake geometry evaluation.
    with tempfile.TemporaryDirectory(prefix="census-aggregate-test-") as temporary:
        root = Path(temporary)
        uid = "asset-a"
        entries = [{"case_id": f"{uid}__seed{s}", "uid": uid, "seed": s, "case_dir": f"{uid}__seed{s}"} for s in (42, 43)]
        entries.append({"case_id": "asset-b__seed42", "uid": "asset-b", "seed": 42, "case_dir": "asset-b__seed42"})
        files = [{"path": f"data/{uid}/imgs/{i:02d}.png", "lfs": {"oid": f"image-{i}"}} for i in range(16)]
        files.append({"path": f"data/{uid}/surfaces.npy", "lfs": {"oid": "gt-hash"}})
        entries[0]["failed_attempt_dirs"] = ["asset-a-failed-attempt"]
        manifest = {"uids": [uid, "asset-b"], "files": files, "cases": entries}
        def put(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
        manifest_path = root / "manifest.json"
        put(manifest_path, manifest)
        put(root / "cases" / "asset-a-failed-attempt" / "report.json",
            {"uid": uid, "seed": 42, "status": "failed", "error": "fixture failure"})
        for entry in entries[:2]:
            directory = root / "cases" / entry["case_dir"]
            directory.mkdir(parents=True)
            (directory / "sequence.npz").write_bytes(b"fixture-opaque-output")
            inputs = {"uid": uid, "records": [{"index": i, "source": f"/data/{uid}/imgs/{i:02d}.png", "source_sha256": f"image-{i}"} for i in range(16)]}
            put(directory / "inputs.json", inputs)
            generated = {"status": "completed", "uid": uid, "seed": entry["seed"], "output": str(directory),
                         "frames": 16, "stage0_steps": 100, "stage1_steps": 30, "guidance_scale": 7.5,
                         "training": False, "steering": False,
                         "sha256": {"sequence.npz": sha(directory / "sequence.npz"), "inputs.json": sha(directory / "inputs.json")}}
            put(directory / "report.json", generated)
            evaluated = {"case_id": entry["case_id"], "uid": uid, "case_dir": str(directory), "status": "success",
                         "inputs": {"sequence": {"path": str(directory / "sequence.npz"), "sha256": generated["sha256"]["sequence.npz"]},
                                    "generation_report": {"path": str(directory / "report.json"), "sha256": sha(directory / "report.json")},
                                    "ground_truth": {"path": f"/data/{uid}/surfaces.npy", "sha256": "gt-hash"}},
                         **{key: 2. if entry["seed"] == 42 else 100. for key in METRICS}}
            put(root / "evaluations" / (entry["case_id"] + ".json"), {"protocol": EXPECTED_PROTOCOL, "cases": [evaluated]})
        result = aggregate(manifest_path, root, True)
        assert result["n_planned_cases"] == 3 and result["n_independent_assets"] == 2
        assert result["primary_baseline_seed42"]["means_success_only"]["cd_motion"] == 2.
        assert result["primary_baseline_seed42"]["n_planned"] == 2
        assert result["primary_baseline_seed42"]["n_success"] == 1
        assert result["seed_control"]["n_complete_pairs"] == 1
        assert result["failed_attempt_count_not_extra_assets"] == 1
        telemetry_dir = root / "queue-test"
        telemetry_dir.mkdir()
        samples = [{"utc": f"2026-10-02T00:00:{seconds:02d}Z", "uuid": "gpu0", "used_mib": 12000,
                    "processes": [{"pid": 1, "memory_mib": 10000, "owner_job_id": "gen-a"},
                                  {"pid": 2, "memory_mib": 500, "owner_job_id": "eval-b"}]}
                   for seconds in (0, 2, 4)]
        (telemetry_dir / "gpu-samples.jsonl").write_text("\n".join(json.dumps(item) for item in samples))
        measured = telemetry(root, {})
        assert measured["max_observed_owned_gpu_jobs"] == 2
        assert measured["concurrent_bracket_seconds"] == 4
        path = root / "evaluations" / (entries[0]["case_id"] + ".json")
        broken = json.loads(path.read_text()); broken["cases"][0]["uid"] = "wrong-asset"; put(path, broken)
        bad = aggregate(manifest_path, root, True)
        assert bad["primary_baseline_seed42"]["n_success"] == 0
        assert bad["case_results"][0]["status"] == "audit_error"
        broken["cases"][0]["uid"] = uid; put(path, broken)
        (root / "cases" / entries[0]["case_dir"] / "sequence.npz").write_bytes(b"tampered")
        tampered = aggregate(manifest_path, root, True)
        assert any(e["code"] == "sequence_bytes_hash_mismatch" for e in tampered["case_results"][0]["audit_errors"])
        out = root / "summary.json"; save(out, result)
        try:
            save(out, result)
        except FileExistsError:
            pass
        else:
            raise AssertionError("Immutable snapshot overwritten")
    return {"status": "pass", "checks": ["repeated seeds do not create assets", "primary means exclude seed43",
            "missing predictions remain denominator and never zero", "UID mismatch excludes metric",
            "failed attempt retained without extra asset", "concurrency brackets recovered from telemetry",
            "tampered output bytes detected", "snapshot replacement requires explicit flag"],
            "scope": "cached aggregation and provenance only; no metric reruns"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-update", action="store_true")
    parser.add_argument("--require-artifacts", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        report = self_test()
        if args.output:
            save(args.output, report, args.allow_update)
        print(json.dumps(report, indent=2))
        return 0
    if any(getattr(args, name) is None for name in ("manifest", "results_root", "output")):
        parser.error("--manifest, --results-root and --output required")
    report = aggregate(args.manifest, args.results_root, args.require_artifacts)
    save(args.output, report, args.allow_update)
    print(json.dumps({"audit": report["audit"], "counts": report["all_planned_run_status_counts"],
                      "primary_baseline_seed42": report["primary_baseline_seed42"], "output": str(args.output)}, indent=2))
    return int(report["audit"]["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
