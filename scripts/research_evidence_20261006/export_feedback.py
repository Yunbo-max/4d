"""Export final overnight receipts and their complete raw closure. Stdlib only.

This checks file integrity and consistency of recorded native reports. It never
runs inference or a scorer, and cannot grant a scientific gate or method verdict.
The protocol fingerprint must come from a previously trusted record.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import random
import shutil
import socket
import stat
import statistics
import tarfile

HISTORICAL_FINGERPRINT = "3c3fe5a7f9b0ec7b75ce32d2ab08a21bcd36cfc64ae7e9a39b88517d08072d0d"
METRICS = ("cd_3d", "cd_4d", "cd_motion")
PHASES = ("generation", "stationary", "native_score", "stationary_score")


class EvidenceError(ValueError):
    """Evidence cannot be safely certified or exported."""


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def read_json(path, expected_sha256=None):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, f"Duplicate JSON key in {path}: {key}")
            value[key] = item
        return value

    def bad_constant(value):
        raise EvidenceError(f"Nonfinite JSON value in {path}: {value}")

    data = path.read_bytes()
    if expected_sha256 is not None:
        require(hashlib.sha256(data).hexdigest() == expected_sha256, f"Parsed bytes differ from snapshot: {path}")
    value = json.loads(data, object_pairs_hook=pairs, parse_constant=bad_constant)
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def state(path):
    s = path.stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def regular(root, path):
    path = Path(os.path.abspath(path))
    root = Path(os.path.abspath(root))
    require(path.is_relative_to(root), f"Path outside allowed root: {path}")
    for item in [path, *path.parents]:
        require(not item.is_symlink(), f"Symlink is not supported in an immutable export: {item}")
    require(path.is_file() and stat.S_ISREG(path.stat().st_mode), f"Missing regular file: {path}")
    return path


def relative_file(root, name):
    require(isinstance(name, str) and bool(name), "Empty artifact path")
    p = PurePosixPath(name)
    require(not p.is_absolute() and ".." not in p.parts and p.as_posix() == name and "\\" not in name,
            f"Unsafe receipt path: {name}")
    return regular(root, root / name)


def idle(identity, label):
    if identity is None:
        return
    require(isinstance(identity, dict) and identity.get("host") == socket.gethostname(),
            f"Unreconciled process on another host: {label}; run on the original host")
    pid = identity.get("pid")
    require(type(pid) is int and pid > 0, f"Invalid process identity: {label}")
    boot = Path("/proc/sys/kernel/random/boot_id")
    if identity.get("boot_id") and boot.exists() and identity["boot_id"] != boot.read_text().strip():
        return
    self_stat = Path("/proc/self/stat")
    if self_stat.exists() and int(self_stat.read_text().split(" ", 1)[0]) == os.getpid():
        try:
            text = (Path("/proc") / str(pid) / "stat").read_text()
        except FileNotFoundError:
            return
        fields = text[text.rfind(")") + 2:].split()
        if fields[0] == "Z":
            return
        if identity.get("start_ticks") is not None and str(identity["start_ticks"]) != fields[19]:
            return
        raise EvidenceError(f"Process is still alive: {label} pid={pid}")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return
    except PermissionError:
        raise EvidenceError(f"Process liveness cannot be established: {label}") from None
    raise EvidenceError(f"Process is still alive or cannot be reconciled: {label} pid={pid}")


def paired_metadata_summary(values):
    """Audit the historical descriptive aggregation; never compute a mesh score."""
    result = {"n_assets": len(values), "mean": statistics.fmean(values), "bootstrap_95_ci": None,
              "scope": "conditional on this development cohort and inference seed; no method verdict"}
    if len(values) >= 8:
        rng = random.Random(271828)
        means = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(4000))
        result["bootstrap_95_ci"] = [means[100], means[3899]]
    return result


def audit_feedback(project_root, run_dir, expected_fingerprint):
    root, run = Path(project_root).absolute(), Path(run_dir).absolute()
    regular(root, run / "protocol.json")
    seen = {}
    members = {}

    def remember(path, expected=None):
        path = regular(root, path)
        before = state(path)
        record = {"sha256": digest(path), "bytes": before[2]}
        require(state(path) == before, f"File changed while hashing: {path}")
        if expected is not None:
            expected = {"sha256": expected} if isinstance(expected, str) else expected
            require(isinstance(expected, dict) and re.fullmatch(r"[0-9a-f]{64}", str(expected.get("sha256", ""))) is not None,
                    f"Invalid expected digest: {path}")
            require(record["sha256"] == expected["sha256"], f"Hash mismatch: {path}")
            if "bytes" in expected:
                require(type(expected["bytes"]) is int and record["bytes"] == expected["bytes"], f"Size mismatch: {path}")
        if path in seen:
            require(seen[path]["record"] == record and seen[path]["state"] == before, f"File changed during audit: {path}")
        else:
            seen[path] = {"record": record, "state": before}
        return record

    def bound_json(path):
        require(path in seen and state(path) == seen[path]["state"], f"JSON changed before parsing: {path}")
        value = read_json(path, seen[path]["record"]["sha256"])
        require(state(path) == seen[path]["state"], f"JSON changed while parsing: {path}")
        return value

    for path in sorted(run.rglob("*")):
        require(not path.is_symlink(), f"Symlink in run tree: {path}")
        if path.is_dir():
            continue
        remember(path)
        members["run/" + path.relative_to(run).as_posix()] = path
    protocol = bound_json(run / "protocol.json")
    signature = hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    require(signature == expected_fingerprint, "Protocol differs from trusted expected fingerprint")
    require(protocol.get("schema_version") == 1 and protocol.get("suite") == "generation", "Unsupported frozen protocol")
    old_root = Path(protocol["root"])
    require(old_root.is_absolute(), "Protocol root must be absolute")

    def remap(name):
        p = Path(name)
        require(p.is_absolute() and p.is_relative_to(old_root) and ".." not in p.parts, f"Source path outside original project: {p}")
        return regular(root, root / p.relative_to(old_root))

    record_sets = {
        "runner_sources": protocol["runner_sources"],
        "native_sources": protocol["native_source"]["native_files_sha256"],
        "data": protocol["generation"]["data_records"],
        "weights": protocol["generation"]["weight_records"],
    }
    for category, records in record_sets.items():
        require(isinstance(records, dict) and bool(records), f"Missing frozen {category} identities")
        for name, expected in records.items():
            path = remap(name)
            remember(path, expected)
            if category in ("runner_sources", "native_sources"):
                members["source/" + path.relative_to(root).as_posix()] = path
    native_hashes = {Path(p).name: (v if isinstance(v, str) else v["sha256"])
                     for p, v in record_sets["native_sources"].items()
                     if Path(p).parent.name == "actionbench" and Path(p).suffix == ".py"}
    evaluator_hashes = {v if isinstance(v, str) else v["sha256"] for p, v in record_sets["runner_sources"].items()
                        if Path(p).name == "research_census_eval.py"}
    require(bool(native_hashes) and bool(evaluator_hashes), "Frozen scorer source identities are absent")
    evaluator_paths = [remap(p) for p in record_sets["runner_sources"] if Path(p).name == "research_census_eval.py"]
    require(len(evaluator_paths) == 1, "Ambiguous frozen census evaluator source")
    evaluator_path = evaluator_paths[0]
    source_bytes = evaluator_path.read_bytes()
    require(hashlib.sha256(source_bytes).hexdigest() == seen[evaluator_path]["record"]["sha256"]
            and state(evaluator_path) == seen[evaluator_path]["state"], "Scorer source changed before literal inspection")
    try:
        assignments = [node.value for node in ast.parse(source_bytes).body if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == "PROTOCOL" for target in node.targets)]
        require(len(assignments) == 1, "Missing/ambiguous scorer PROTOCOL constant")
        expected_score_protocol = dict(ast.literal_eval(assignments[0]), sampling_seed=44)
    except (SyntaxError, ValueError, TypeError) as exc:
        raise EvidenceError(f"Unsupported frozen scorer PROTOCOL literal: {exc}") from exc
    window = bound_json(run / "window.json")
    require(window["fingerprint"] == signature, "Window fingerprint mismatch")
    queue = bound_json(run / "queue.json")
    require(queue.get("status") == "inventory_complete" and queue.get("active_unit") is None, "Queue is not final/idle")
    idle(queue.get("active_process"), "queue worker")
    guardian_path = run / "completion-status.json"
    if guardian_path.exists():
        guardian = bound_json(guardian_path)
        require(guardian.get("fingerprint") == signature and guardian.get("status") == "complete", "Completion supervisor is not final")
        idle(guardian.get("process"), "completion supervisor")

    def validate_attempts(attempts, label, require_last_success=True):
        require(isinstance(attempts, list) and bool(attempts), f"Missing attempt trail: {label}")
        if require_last_success:
            require(attempts[-1].get("status") == "completed" and attempts[-1].get("returncode") == 0,
                    f"Last attempt is not complete: {label}")
        for attempt in attempts:
            require(attempt.get("status") not in (None, "running", "starting", "queued"), f"Nonterminal attempt: {label}")
            require(remap(attempt["log"]).is_relative_to(run), f"Attempt log outside run: {label}")
            ended = datetime.fromisoformat(attempt["ended_utc"])
            require(ended.tzinfo is not None and ended <= datetime.now(timezone.utc), f"Invalid ended timestamp: {label}")
            idle(attempt.get("process"), "attempt " + label)

    if "preflight_attempts" in queue:
        validate_attempts(queue["preflight_attempts"], "preflight")
    cohort = protocol["generation"]["cohort"]
    uids = cohort["uids"]
    require(isinstance(uids, list) and bool(uids) and all(isinstance(u, str) and re.fullmatch(r"[A-Za-z0-9_-]+", u) for u in uids)
            and len(set(uids)) == len(uids), "Invalid/duplicate frozen asset inventory")
    unit_ids = ["gen-" + uid for uid in uids]
    require(queue.get("unit_ids") == unit_ids and set(queue["units"]) == set(unit_ids), "Queue inventory differs from frozen cohort")
    inferred = cohort["native_inference"]
    evaluated = cohort["native_evaluation"]
    for key, expected in {"frames": 16, "stage0_steps": 100, "stageI_steps": 30, "cfg": 7.5, "seed": 42}.items():
        require(inferred.get(key) == expected, f"Unsupported native inference setting: {key}")
    budget = {"sampling_seed": 44, "n_pts_chamfer": 100000, "n_pts_icp": 10000, "icp_initial_rotations": 24, "icp_iterations": 200}
    for key, expected in {"frames": 16, "sampling_seed": 44, "surface_samples": 100000, "icp_samples": 10000,
                          "icp_rotations": 24, "icp_iterations": 200}.items():
        require(evaluated.get(key) == expected, f"Unsupported native evaluation setting: {key}")

    def receipt(unit, name, expected):
        path = unit / name
        remember(path)
        value = bound_json(path)
        require(value.get("status") == "complete" and value.get("fingerprint") == expected, f"Receipt is not complete/bound: {path}")
        require(isinstance(value.get("artifacts"), dict) and bool(value["artifacts"]), f"Empty receipt: {path}")
        for name, record in value["artifacts"].items():
            remember(relative_file(unit, name), record)
        return value

    pairs = []
    for uid, unit_id in zip(uids, unit_ids):
        unit = run / "units" / unit_id
        item = queue["units"][unit_id]
        require(item.get("status") == "complete" and item.get("family") == "generation", f"Incomplete unit: {unit_id}")
        validate_attempts(item.get("attempts"), unit_id)
        final = receipt(unit, "receipt.json", signature)
        require(final.get("uid") == uid and final.get("family") == "generation", f"Final receipt UID/family mismatch: {uid}")
        require("pair.json" in final["artifacts"], f"Final receipt omits pair: {uid}")
        phases = {}
        for name in PHASES:
            filename = name + ".receipt.json"
            require(filename in final["artifacts"], f"Final receipt omits phase: {filename}")
            value = receipt(unit, filename, signature + ":" + name)
            require(all(final["artifacts"].get(k) == v for k, v in value["artifacts"].items()), f"Final receipt omits/changes nested raw product: {name}")
            phases[name] = value
        pair = bound_json(unit / "pair.json")
        require(pair.get("uid") == uid, f"Pair UID mismatch: {uid}")
        native_dir = remap(phases["generation"]["result"]["directory"] + "/report.json").parent
        static_dir = remap(phases["stationary"]["result"]["directory"] + "/report.json").parent
        for directory in (native_dir, static_dir):
            require(directory.is_relative_to(unit), f"Generation directory outside unit: {directory}")
        for phase, directory in [("generation", native_dir), ("stationary", static_dir)]:
            require(all(str((directory / filename).relative_to(unit)) in phases[phase]["artifacts"]
                        for filename in ("report.json", "sequence.npz")), f"Generation report/sequence is not phase-bound: {uid} {phase}")
        generated = bound_json(native_dir / "report.json")
        for key, expected in {"status": "completed", "uid": uid, "seed": 42, "frames": 16,
                              "stage0_steps": 100, "stage1_steps": 30, "guidance_scale": 7.5}.items():
            require(generated.get(key) == expected, f"Recorded inference differs: {uid} {key}")
        for name, expected in generated["sha256"].items():
            path = relative_file(native_dir, name)
            require(str(path.relative_to(unit)) in phases["generation"]["artifacts"], f"Generation receipt omits output: {path}")
            remember(path, expected)
        if "report" in phases["generation"]["result"]:
            require(phases["generation"]["result"]["report"] == generated, f"Embedded generation result differs: {uid}")
        require({"inputs.json", "preprocessing.json", "sequence.npz"}.issubset(generated["sha256"]),
                f"Generation nested evidence records are omitted: {uid}")
        input_info = bound_json(native_dir / "inputs.json")
        preprocessing_info = bound_json(native_dir / "preprocessing.json")
        require(input_info.get("uid") == uid and input_info.get("frame_indices") == list(range(16))
                and input_info.get("timesteps") == list(range(16)), f"Saved input timeline differs: {uid}")
        for value, keys in [(input_info, [("saved_file", "saved_sha256")]),
                            (preprocessing_info, [("masked_file", "masked_sha256"), ("processed_file", "processed_sha256")])]:
            records = value["records"]
            require(isinstance(records, list) and [r.get("index") for r in records] == list(range(16)),
                    f"Nested frame inventory differs: {uid}")
            for record in records:
                for path_key, hash_key in keys:
                    remember(relative_file(native_dir, record[path_key]), record[hash_key])
                if value is input_info:
                    source = record["source"]
                    expected_source = Path(protocol["generation"]["data_root"]) / "data" / uid / "imgs" / f"{record['index']:02d}.png"
                    require(Path(source) == expected_source and source in record_sets["data"], f"Nested input source differs: {uid}")
                    remember(remap(source), record["source_sha256"])
        static = bound_json(static_dir / "report.json")
        require(static.get("status") == "completed" and static.get("uid") == uid and static.get("frames") == 16
                and static.get("source_sequence_sha256") == remember(native_dir / "sequence.npz")["sha256"], f"Stationary provenance mismatch: {uid}")
        for phase, arm, directory in [("native_score", "native", native_dir), ("stationary_score", "stationary", static_dir)]:
            result = phases[phase]["result"]
            path = remap(result["native_report"])
            require(path.is_relative_to(unit) and str(path.relative_to(unit)) in phases[phase]["artifacts"], f"Scorer report not receipt-bound: {uid}")
            scored = bound_json(path)
            require(all(scored["protocol"].get(k) == v for k, v in budget.items()), f"Native scorer budget differs: {uid} {arm}")
            require(scored["protocol"] == expected_score_protocol, f"Full native scorer protocol differs: {uid} {arm}")
            require(scored.get("evaluator_sha256") in evaluator_hashes and isinstance(scored.get("official_source"), dict)
                    and scored["official_source"].get("sha256") == native_hashes,
                    f"Scorer source differs: {uid} {arm}")
            cases = scored["cases"]
            require(len(cases) == 1 and cases[0].get("status") == "success" and cases[0].get("uid") == uid, f"Scorer case incomplete: {uid} {arm}")
            case = cases[0]
            manifest_path = path.parent / "manifest.json"
            log_path = path.parent / "score.log"
            require(all(str(p.relative_to(unit)) in phases[phase]["artifacts"] for p in (manifest_path, log_path)),
                    f"Scorer manifest/log is not phase-bound: {uid} {arm}")
            denominator = scored["denominator"]
            require(denominator.get("frozen") is True and denominator.get("n_declared") == 1
                    and remap(denominator["manifest"]) == manifest_path
                    and denominator["manifest_sha256"] == seen[manifest_path]["record"]["sha256"], f"Scorer denominator differs: {uid} {arm}")
            manifest = bound_json(manifest_path)
            require(isinstance(manifest.get("cases"), list) and len(manifest["cases"]) == 1, f"Scorer manifest case inventory differs: {uid}")
            entry = manifest["cases"][0]
            require(entry.get("case_id") == phase and entry.get("uid") == uid
                    and remap(entry["case_dir"] + "/sequence.npz") == directory / "sequence.npz"
                    and case.get("manifest_entry") == entry and case.get("case_id") == phase
                    and case.get("case_dir") == entry["case_dir"] and case.get("generation_status") == "completed", f"Scorer case provenance differs: {uid} {arm}")
            shapes = case["shapes"]
            expected_shape = {"vertices": [16, generated["vertices_per_frame"], 3],
                              "faces": [generated["faces"], 3], "gt_positions": [16, 100000, 3]}
            require(type(generated["vertices_per_frame"]) is int and generated["vertices_per_frame"] > 0
                    and type(generated["faces"]) is int and generated["faces"] > 0
                    and shapes == expected_shape, f"Recorded full-timeline shapes differ: {uid} {arm}")
            for name, identity in case["inputs"].items():
                input_path = remap(identity["path"])
                remember(input_path, identity["sha256"])
                if name == "sequence":
                    require(input_path == directory / "sequence.npz", f"Score consumed wrong sequence: {uid} {arm}")
                if name == "ground_truth":
                    expected_gt = Path(protocol["generation"]["data_root"]) / "data" / uid / "surfaces.npy"
                    require(Path(identity["path"]) == expected_gt and identity["path"] in record_sets["data"],
                            f"Scorer GT differs from frozen asset data: {uid}")
                if name == "generation_report":
                    require(input_path == directory / "report.json", f"Scorer consumed wrong generation report: {uid} {arm}")
            require({"sequence", "ground_truth", "generation_report"}.issubset(case["inputs"]), f"Scorer inputs omitted: {uid} {arm}")
            for key in METRICS:
                value = case[key]
                require(type(value) in (int, float) and math.isfinite(value) and value >= 0, f"Invalid recorded metric: {uid} {key}")
                require(pair[arm][key] == value and result["metrics"][key] == value, f"Pair/receipt/scorer disagreement: {uid} {arm} {key}")
            metric_values = {key: case[key] for key in METRICS}
            expected_summary = {"n_total": 1, "n_success": 1, "n_failed": 0, "n_pending": 0,
                "n_validated_only": 0, "success_rate": 1., "n_unique_assets": 1, "n_assets_with_success": 1,
                "successful_case_ids": [phase], "means": metric_values, "asset_balanced_means": metric_values,
                "asset_results": [{"uid": uid, "n_runs": 1, "n_success": 1, "means": metric_values}]}
            require(all(scored["summary"].get(k) == v for k, v in expected_summary.items()), f"Scorer summary disagrees with its case: {uid} {arm}")
        require(all(pair["native_minus_stationary"][k] == pair["native"][k] - pair["stationary"][k] for k in METRICS), f"Pair difference mismatch: {uid}")
        pairs.append(pair)
    full_summary = bound_json(run / "summary.json")
    require(full_summary.get("queue_status") == queue["status"] and full_summary.get("scientific_gate_pass") is False
            and full_summary.get("candidate_methods_tested") is False, "Top-level summary status/claim differs")
    summary = full_summary["generation"]
    require(summary.get("declared_assets") == len(uids) and summary.get("complete_pairs") == len(uids)
            and summary.get("pairs") == pairs, "Summary differs from frozen complete pairs")
    require(summary.get("pending_or_failed") == [] and summary.get("failure_labels_inferred") is False
            and summary.get("inspection_by_native_cd_motion") == [p["uid"] for p in sorted(pairs, key=lambda p: p["native"]["cd_motion"])],
            "Summary pending/inspection state differs")
    expected_aggregates = {key: paired_metadata_summary([p["native_minus_stationary"][key] for p in pairs]) for key in METRICS}
    require(summary.get("native_minus_stationary") == expected_aggregates, "Summary aggregates differ from verified pairs")
    report = {"schema_version": 1, "checked_utc": datetime.now(timezone.utc).isoformat(), "fingerprint": signature,
              "declared_assets": len(uids), "verified_complete_pairs": len(pairs),
              "scope": "Local receipt/file integrity and recorded native-report consistency; not a scientific gate or replay",
              "scientific_gate_pass": False, "candidate_methods_tested": False, "native_replay_performed": False,
              "receipt_raw_closure_complete": True, "known_nested_raw_files_complete": True, "tool_sha256": digest(Path(__file__)),
              "telemetry_scope": "All existing run files retained; continuous coverage and true peak memory are not qualified",
              "input_weights_included": False, "external_record_counts": {k: len(v) for k, v in record_sets.items()},
              "original_project_root": str(old_root), "observed_project_root": str(root), "observed_run_dir": str(run),
              "missing_scientific_obligations": ["Trusted full native replay", "Natural Gate 0/strongest simple baseline", "Functional collision audit/IPCG", "Fresh family/exposure split"],
              "members": {name: seen[path]["record"] for name, path in sorted(members.items())}}
    return report, members, seen


class SplitWriter:
    """Bounded compressed-stream parts concatenate to one ordinary tar.gz."""
    def __init__(self, destination, part_bytes):
        self.destination, self.limit = destination, part_bytes
        self.stream = None
        self.size = 0
        self.parts = []
        self.total_hash = hashlib.sha256()

    def write(self, data):
        count = len(data)
        self.total_hash.update(data)
        while data:
            if self.stream is None:
                self.path = self.destination / f"feedback.tar.gz.part{len(self.parts) + 1:04d}"
                self.stream = self.path.open("xb")
                self.part_hash, self.size = hashlib.sha256(), 0
            block, data = data[:self.limit - self.size], data[self.limit - self.size:]
            self.stream.write(block)
            self.part_hash.update(block)
            self.size += len(block)
            if self.size == self.limit:
                self.finish_part()
        return count

    def finish_part(self):
        if self.stream is not None:
            self.stream.flush()
            os.fsync(self.stream.fileno())
            self.stream.close()
            self.parts.append({"name": self.path.name, "bytes": self.size, "sha256": self.part_hash.hexdigest()})
            self.stream = None


class HashReader:
    def __init__(self, stream):
        self.stream = stream
        self.value = hashlib.sha256()
        self.bytes = 0

    def read(self, length):
        block = self.stream.read(length)
        self.value.update(block)
        self.bytes += len(block)
        return block


@contextmanager
def readonly_runner_locks(run_dir):
    """Hold existing lock files without rewriting their historical owner record."""
    with ExitStack() as stack:
        for name in ("runner.lock", "completion.lock"):
            path = Path(run_dir).absolute() / name
            if path.exists() or path.is_symlink():
                regular(Path(run_dir).absolute(), path)
                stream = stack.enter_context(path.open("rb"))
                try:
                    fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise EvidenceError(f"Original runner/supervisor still holds {path}") from None
        yield


def export_feedback(project_root, run_dir, destination, expected_fingerprint, part_bytes=24 * 1024 * 1024):
    try:
        with readonly_runner_locks(run_dir):
            return _export_feedback(project_root, run_dir, destination, expected_fingerprint, part_bytes)
    except OSError as exc:
        raise EvidenceError(f"Cannot lock/read evidence: {exc}") from exc


def _export_feedback(project_root, run_dir, destination, expected_fingerprint, part_bytes):
    destination = Path(destination).absolute()
    run_dir = Path(run_dir).absolute()
    require(not destination.exists() and not destination.is_symlink(), f"Destination already exists: {destination}")
    require(all(not path.is_symlink() for path in destination.parents), "Destination parent is a symlink")
    require(not destination.is_relative_to(run_dir) and not run_dir.is_relative_to(destination), "Destination overlaps original run")
    require(type(part_bytes) is int and 32768 <= part_bytes <= 32 * 1024 * 1024, "Part size must be between 32 KiB and 32 MiB")
    created = False
    writer = None
    try:
        report, members, seen = audit_feedback(project_root, run_dir, expected_fingerprint)
        destination.mkdir(parents=True, exist_ok=False)
        created = True
        writer = SplitWriter(destination, part_bytes)
        with tarfile.open(fileobj=writer, mode="w|gz", format=tarfile.PAX_FORMAT) as archive:
            for name, path in sorted(members.items()):
                regular(Path(project_root), path)
                require(state(path) == seen[path]["state"], f"File changed before packing: {path}")
                info = archive.gettarinfo(str(path), arcname=name)
                require(info.isfile(), f"Nonregular archive member: {path}")
                with path.open("rb") as stream:
                    reader = HashReader(stream)
                    archive.addfile(info, reader)
                require(reader.bytes == seen[path]["record"]["bytes"] and reader.value.hexdigest() == seen[path]["record"]["sha256"], f"File changed while packing: {path}")
        writer.finish_part()
        require(set("run/" + p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*") if not p.is_dir())
                == {n for n in members if n.startswith("run/")}, "Run inventory changed during export")
        for path, value in seen.items():
            regular(Path(project_root), path)
            require(state(path) == value["state"], f"Source/input changed during export: {path}")
        for part in writer.parts:
            require(digest(destination / part["name"]) == part["sha256"], f"Packed part failed readback: {part['name']}")
        report.update(archives=writer.parts, archive_sha256=writer.total_hash.hexdigest(), archive_format="concatenate parts in manifest order to tar.gz")
        (destination / "manifest.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        return report
    except (OSError, KeyError, TypeError, ValueError, IndexError) as exc:
        if writer is not None and writer.stream is not None:
            writer.stream.close()
        if created:
            shutil.rmtree(destination)
        if isinstance(exc, EvidenceError):
            raise
        raise EvidenceError(f"Incomplete/invalid evidence: {type(exc).__name__}: {exc}") from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--expected-fingerprint", default=HISTORICAL_FINGERPRINT)
    parser.add_argument("--part-mib", type=int, default=24)
    args = parser.parse_args()
    try:
        report = export_feedback(args.project_root, args.run_dir, args.destination,
                                 args.expected_fingerprint, args.part_mib * 1024 * 1024)
    except EvidenceError as exc:
        print(json.dumps({"status": "blocked", "error": str(exc), "scientific_gate_pass": False}))
        return 2
    print(json.dumps({k: report[k] for k in ("checked_utc", "fingerprint", "declared_assets", "verified_complete_pairs",
                                           "archives", "archive_sha256", "scientific_gate_pass", "native_replay_performed")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
