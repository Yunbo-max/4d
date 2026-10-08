#!/usr/bin/env python3
"""Finite Local campaign controller for the installed Research Autopilot harness.

Generated/unexecuted source. GPU dispatch is stopped by default. This
controller never executes an experiment command, repairs source, resets an
attempt, grants a scientific gate, or changes the native runner's retry limits.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib
import importlib.abc
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import tempfile
import time


_LOADED_SKILL = None


class CampaignError(ValueError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CampaignError("duplicate JSON key: " + key)
        result[key] = value
    return result


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_pairs,
                          parse_constant=lambda value: (_ for _ in ()).throw(
                              CampaignError("nonfinite JSON: " + value)))
    except (OSError, ValueError) as error:
        raise CampaignError("unreadable JSON: " + str(path)) from error


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def campaign_digest(value):
    return hashlib.sha256(canonical({key: item for key, item in value.items()
                                    if key != "campaign_digest"}).encode()).hexdigest()


def _digest(value):
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
        raise CampaignError("invalid SHA-256")
    return value


def _fields(value, names):
    if not isinstance(value, dict) or set(value) != set(names.split()):
        raise CampaignError("unexpected contract fields")


def _number(value, low, high):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or not low <= value <= high):
        raise CampaignError("finite numeric bound required")
    return value


def _absolute(value, *, directory=True):
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise CampaignError("absolute path required")
    path = Path(value)
    if path != path.resolve(strict=True):
        raise CampaignError("canonical absolute path required; no symlink")
    if directory and not path.is_dir():
        raise CampaignError("directory required")
    return path


def safe_path(root, relative):
    if (not isinstance(relative, str) or not relative or Path(relative).is_absolute() or
            any(part in ("", ".", "..") for part in relative.split("/"))):
        raise CampaignError("unsafe relative path")
    path = root
    for part in relative.split("/"):
        path = path / part
        if path.is_symlink():
            raise CampaignError("symlink artifact rejected")
    path.resolve().relative_to(root)
    return path


def verify_ref(root, ref):
    _fields(ref, "path sha256")
    path = safe_path(root, ref["path"])
    if not path.is_file() or file_digest(path) != _digest(ref["sha256"]):
        raise CampaignError("pinned bytes changed: " + ref["path"])
    return path


def skill_digest(root):
    """Same sorted inventory algorithm as research_autopilot.artifacts.digest_tree."""
    root = Path(root).resolve(strict=True)
    excluded = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
                ".venv", "venv", ".runtime", "runtime-state", "node_modules"}
    suffixes = (".pyc", ".pyo", ".db", ".sqlite", ".sqlite3", "-wal", "-shm")
    files = []
    def on_error(error):
        raise error
    for directory, dirs, names in os.walk(root, followlinks=False, onerror=on_error):
        dirs[:] = sorted(name for name in dirs if name not in excluded)
        if any((Path(directory) / name).is_symlink() for name in dirs):
            raise CampaignError("skill source symlink")
        for name in sorted(names):
            if name.endswith(suffixes):
                continue
            path = Path(directory) / name
            if path.is_symlink() or not path.is_file():
                raise CampaignError("skill source must be regular files")
            files.append({"path": path.relative_to(root).as_posix(), "sha256": file_digest(path)})
    return hashlib.sha256(canonical(files).encode()).hexdigest()


class _SourceLoader(importlib.abc.Loader):
    def __init__(self, path, data):
        self.path, self.data = path, data

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        module.__file__ = str(self.path)
        exec(compile(self.data, str(self.path), "exec"), module.__dict__)
        module.__supervisor_source_sha256__ = hashlib.sha256(self.data).hexdigest()


class _SourceFinder(importlib.abc.MetaPathFinder):
    def __init__(self, sources):
        self.sources = sources

    def find_spec(self, fullname, path=None, target=None):
        if fullname in self.sources:
            source, data = self.sources[fullname]
            return importlib.util.spec_from_loader(fullname, _SourceLoader(source, data))
        return None


def load_harness(root, expected_digest):
    """Load actual verified canonical source, bypassing unpinned Python caches."""
    global _LOADED_SKILL
    root = Path(root).resolve(strict=True)
    if skill_digest(root) != _digest(expected_digest):
        raise CampaignError("installed skill digest changed")
    if _LOADED_SKILL is not None and _LOADED_SKILL != (root, expected_digest):
        raise CampaignError("different skill revision requires a fresh supervisor process")
    sources = {path.stem: (path, path.read_bytes()) for path in (root / "scripts").glob("*.py")}
    if "run_harness" not in sources:
        raise CampaignError("complete installed harness required")
    if skill_digest(root) != expected_digest:
        raise CampaignError("skill changed during source capture")
    for name, (path, data) in sources.items():
        existing = sys.modules.get(name)
        if existing is not None and Path(getattr(existing, "__file__", "")).resolve() != path:
            raise CampaignError("conflicting canonical module: " + name)
    # The acceptance suite may already have imported the same installed modules.
    # Recompile every supervisor dependency from captured bytes rather than trust
    # those cached objects or their .pyc. Existing callers keep their own references.
    for name in sources:
        sys.modules.pop(name, None)
    sys.meta_path[:] = [finder for finder in sys.meta_path if not isinstance(finder, _SourceFinder)]
    finder = _SourceFinder(sources)
    sys.meta_path.insert(0, finder)
    _LOADED_SKILL = (root, expected_digest)
    # Keep the finder for canonical modules imported lazily by the validators.
    return importlib.import_module("run_harness")


class Campaign:
    def __init__(self, path):
        self.path = Path(path).resolve(strict=True)
        self.data = read_json(self.path)
        m = self.data
        _fields(m, "kind version campaign_id root python skill_root skill_digest pool_dir "
                   "total_wall_seconds collection_reserve_seconds plans campaign_digest")
        if m["kind"] != "research-harness-campaign" or type(m["version"]) is not int or m["version"] not in (1, 2):
            raise CampaignError("unsupported campaign")
        if not isinstance(m["campaign_id"], str) or not re.fullmatch("[A-Za-z0-9][A-Za-z0-9_-]{0,95}", m["campaign_id"]):
            raise CampaignError("invalid campaign identifier")
        if campaign_digest(m) != _digest(m["campaign_digest"]):
            raise CampaignError("campaign digest mismatch")
        self.root = _absolute(m["root"])
        if (self.root / ".pending-commit.json").exists():
            raise CampaignError("explicit project recovery required before inspection")
        self.skill = _absolute(m["skill_root"])
        self.pool = _absolute(m["pool_dir"])
        interpreter = _absolute(m["python"], directory=False)
        if interpreter != Path(sys.executable).resolve() or not interpreter.is_file():
            raise CampaignError("campaign must pin the current Local Python interpreter")
        _number(m["total_wall_seconds"], 1801, 28800)
        _number(m["collection_reserve_seconds"], 1800, m["total_wall_seconds"] - 1)
        if not isinstance(m["plans"], list) or not 1 <= len(m["plans"]) <= 256:
            raise CampaignError("finite nonempty plan inventory required")
        self.H = load_harness(self.skill, m["skill_digest"])
        self.entries, self.plans, self.native, self.missing = {}, {}, {}, {}
        self.directory = safe_path(self.root, "runs/supervisor/" + m["campaign_id"])
        native_ids, batch_ids, trial_ids, repairs = set(), set(), set(), set()
        reserved = 0
        for entry in m["plans"]:
            _fields(entry, "id plan_ref plan_digest dependencies on_failure_of" +
                    (" required_inputs" if m["version"] == 2 else ""))
            name = entry["id"]
            if not isinstance(name, str) or not re.fullmatch("[A-Za-z0-9][A-Za-z0-9_-]{0,95}", name) or name in self.entries:
                raise CampaignError("invalid or repeated plan id")
            deps = entry["dependencies"]
            if not isinstance(deps, list) or any(not isinstance(x, str) for x in deps) or len(deps) != len(set(deps)):
                raise CampaignError("invalid dependency inventory")
            trigger = entry["on_failure_of"]
            if trigger is not None:
                if not isinstance(trigger, str) or not re.fullmatch("[A-Za-z0-9][A-Za-z0-9_-]{0,95}", trigger):
                    raise CampaignError("invalid repair trigger identifier")
                if trigger in deps or trigger in repairs:
                    raise CampaignError("one distinct bounded repair per failed parent")
                repairs.add(trigger)
            plan = read_json(verify_ref(self.root, entry["plan_ref"]))
            # Missing *explicitly declared* immutable inputs can delay one plan.
            # Schemas, digests, executable code and plan identities are still read
            # now; the full canonical validator remains mandatory before dispatch.
            self.H.C.validate(plan, "harness-plan")
            if self.H.plan_digest(plan) != plan["plan_digest"]:
                raise CampaignError("harness digest mismatch")
            required = entry.get("required_inputs", [])
            if not isinstance(required, list):
                raise CampaignError("required_inputs must be a list")
            required_keys = set()
            for ref in required:
                _fields(ref, "path sha256")
                safe_path(self.root, ref["path"])
                required_keys.add((ref["path"], _digest(ref["sha256"])))
            if len(required_keys) != len(required):
                raise CampaignError("duplicate required input")
            if plan["plan_digest"] != _digest(entry["plan_digest"]) or plan["pool_dir"] != str(self.pool):
                raise CampaignError("harness plan identity or host pool mismatch")
            batch = safe_path(self.root, plan["output_root"] + "/" + plan["batch_id"])
            if batch in batch_ids or batch == self.directory or batch in self.directory.parents or self.directory in batch.parents:
                raise CampaignError("overlapping retained batch identity")
            batch_ids.add(batch)
            reserved += plan["limits"]["total_wall_seconds"]
            natives = {}
            for task in plan["tasks"]:
                native = read_json(verify_ref(self.root, task["plan_ref"]))
                self.H.C.validate(native, "experiment-run-plan")
                if self.H.R.plan_digest(native) != native["plan_digest"]:
                    raise CampaignError("native digest mismatch")
                # Input availability cannot excuse changed or missing program bytes.
                for job in native["jobs"]:
                    for ref in job["code_refs"]:
                        verify_ref(self.root, ref)
                identity = safe_path(self.root, native["output_root"] + "/" + native["run_id"])
                if identity in native_ids or identity == self.directory or identity in self.directory.parents or self.directory in identity.parents:
                    raise CampaignError("repeated or overlapping native identity")
                native_ids.add(identity)
                for job in native["jobs"]:
                    if job["trial_id"] in trial_ids:
                        raise CampaignError("repeated trial identity; child plans cannot retry existing experiments")
                    trial_ids.add(job["trial_id"])
                natives[task["task_id"]] = native
            native_inputs = {(ref["path"], ref["sha256"])
                             for native in natives.values() for job in native["jobs"]
                             for ref in job["input_refs"]}
            if not required_keys.issubset(native_inputs):
                raise CampaignError("required input must be pinned in a native job")
            self.entries[name], self.plans[name], self.native[name] = entry, plan, natives
            self.validate_ready(name)
        if reserved > m["total_wall_seconds"] - m["collection_reserve_seconds"]:
            raise CampaignError("all plans including repairs must fit the finite workload budget")
        retained = list(native_ids | batch_ids)
        for index, first in enumerate(retained):
            for second in retained[index + 1:]:
                if first in second.parents or second in first.parents:
                    raise CampaignError("overlapping retained output directories")
        if native_ids & batch_ids:
            raise CampaignError("native and harness output identities overlap")
        seen, visiting = set(), set()
        def visit(name):
            if name not in self.entries:
                raise CampaignError("unknown dependency")
            if name in visiting:
                raise CampaignError("campaign dependency cycle")
            if name in seen:
                return
            visiting.add(name)
            entry = self.entries[name]
            for dep in entry["dependencies"] + ([entry["on_failure_of"]] if entry["on_failure_of"] is not None else []):
                visit(dep)
            visiting.remove(name)
            seen.add(name)
        for name in self.entries:
            visit(name)

    def validate_ready(self, name):
        missing = []
        for ref in self.entries[name].get("required_inputs", []):
            path = safe_path(self.root, ref["path"])
            if not path.exists():
                missing.append(ref["path"])
            else:
                verify_ref(self.root, ref)  # Wrong bytes are never a readiness wait.
        self.missing[name] = missing
        if not missing:
            try:
                self.H.validate_plan(self.root, self.plans[name])
            except Exception as error:
                raise CampaignError("canonical harness validation denied: " + name) from error

    def gpu(self, name):
        return any(task["resources"]["gpu_count"] for task in self.plans[name]["tasks"])

    def recheck(self):
        if (self.root / ".pending-commit.json").exists():
            raise CampaignError("explicit project recovery required")
        if read_json(self.path) != self.data or skill_digest(self.skill) != self.data["skill_digest"]:
            raise CampaignError("approved campaign or installed source changed")
        for name, entry in self.entries.items():
            if read_json(verify_ref(self.root, entry["plan_ref"])) != self.plans[name]:
                raise CampaignError("approved plan changed")
            for task in self.plans[name]["tasks"]:
                if read_json(verify_ref(self.root, task["plan_ref"])) != self.native[name][task["task_id"]]:
                    raise CampaignError("approved native plan changed")
                for job in self.native[name][task["task_id"]]["jobs"]:
                    for ref in job["code_refs"]:
                        verify_ref(self.root, ref)
            self.validate_ready(name)

    def observe(self, name):
        """Classify real retained artifacts; saved supervisor status is never proof."""
        plan = self.plans[name]
        batch = safe_path(self.root, plan["output_root"] + "/" + plan["batch_id"])
        natives = self.native[name]
        native_roots = {key: safe_path(self.root, n["output_root"] + "/" + n["run_id"])
                        for key, n in natives.items()}
        try:
            if not (batch / "state.json").is_file():
                retained = (batch.exists() and any(batch.iterdir())) or any(p.exists() for p in native_roots.values())
                if not retained and self.missing[name]:
                    return {"status": "waiting_inputs", "repairable": False,
                            "missing_inputs": self.missing[name]}
                return {"status": "unknown" if retained else "absent", "repairable": False}
            state = read_json(batch / "state.json")
            if (state.get("format") != "research-harness-state-v1" or state.get("root") != str(self.root) or
                    state.get("batch_id") != plan["batch_id"] or state.get("plan_digest") != plan["plan_digest"] or
                    read_json(batch / "plan.json") != plan or
                    state.get("boot_id") != self.H._self_identity()["boot_id"] or
                    set(state["tasks"]) != set(natives)):
                raise CampaignError("retained harness identity mismatch")
            outcomes, deferred = {}, []
            repairable = False
            for task in plan["tasks"]:
                key = task["task_id"]
                record = state["tasks"][key]
                task_dir = batch / "tasks" / key
                result_path = task_dir / "result.json"
                identities = [record.get("process")]
                if (task_dir / "process.json").is_file():
                    identities.append(read_json(task_dir / "process.json"))
                for identity in identities:
                    if identity is not None:
                        _fields(identity, "pid start_ticks boot_id")
                        if type(identity["pid"]) is not int or identity["pid"] <= 0 or not isinstance(identity["start_ticks"], str) or not identity["start_ticks"].isdigit():
                            raise CampaignError("invalid retained process")
                if any(self.H._alive(identity) for identity in identities if identity):
                    outcomes[key] = "running"
                elif result_path.is_file():
                    result = read_json(result_path)
                    if read_json(task_dir / "task.json") != task:
                        raise CampaignError("retained task identity mismatch")
                    self.H._verify_result(self.root, task, result)
                    outcome = result["status"]
                    if outcome not in self.H.TERMINAL:
                        raise CampaignError("nonterminal task result")
                    if result.get("receipt_ref"):
                        receipt = read_json(verify_ref(self.root, result["receipt_ref"]))
                        if receipt["status"] == "interrupted":
                            raise CampaignError("interrupted native attempt requires reconciliation")
                        repairable = repairable or outcome == "failed" and receipt["status"] == "failed"
                    elif native_roots[key].exists() or outcome == "completed":
                        raise CampaignError("unreceipted retained native execution")
                    outcomes[key] = outcome
                elif record.get("status") == "pending" and not task_dir.exists() and not native_roots[key].exists():
                    outcomes[key] = "pending"
                elif record.get("status") in ("blocked", "budget_exhausted") and not task_dir.exists() and not native_roots[key].exists():
                    deferred.append(task)
                else:
                    raise CampaignError("unproven retained execution")
            while deferred:
                progressed = False
                for task in list(deferred):
                    record = state["tasks"][task["task_id"]]
                    status = record["status"]
                    if status == "blocked" and record.get("reason_code") == "HARNESS_DEPENDENCY_FAILED" and any(outcomes.get(dep) in self.H.TERMINAL - {"completed"} for dep in task["depends_on"]):
                        outcomes[task["task_id"]] = status
                    elif status == "budget_exhausted" and record.get("reason_code") == "HARNESS_HARD_DEADLINE" and state.get("hard_deadline_epoch") == state.get("started_epoch", -1) + plan["limits"]["total_wall_seconds"] and time.time() >= state["hard_deadline_epoch"]:
                        outcomes[task["task_id"]] = status
                    else:
                        continue
                    deferred.remove(task)
                    progressed = True
                if not progressed:
                    raise CampaignError("unproven blocked task")
            if any(status in ("running", "pending") for status in outcomes.values()):
                status = "running"
            else:
                status = "completed" if all(status == "completed" for status in outcomes.values()) else "failed"
            return {"status": status, "repairable": status == "failed" and repairable,
                    "tasks": outcomes}
        except Exception as error:
            return {"status": "unknown", "repairable": False,
                    "reason": type(error).__name__ + ": " + str(error)}


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(canonical(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        _sync_directory(path.parent)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class HarnessTransport:
    """Only the fixed installed harness argv reaches a child process."""
    def start(self, argv, directory, environment):
        with (directory / "driver.stdout.log").open("ab") as stdout, (directory / "driver.stderr.log").open("ab") as stderr:
            return subprocess.Popen(argv, stdout=stdout, stderr=stderr, env=environment,
                                    start_new_session=True, shell=False)


def _driver_busy(pool):
    return _lock_busy(pool / "driver.lock")


def _lock_busy(path):
    if not path.exists():
        return False
    with path.open("rb") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(stream, fcntl.LOCK_UN)
        return False


def harness_argv(campaign, name):
    entry = campaign.entries[name]
    return [campaign.data["python"], "-B", str(campaign.skill / "scripts/run_harness.py"),
            str(verify_ref(campaign.root, entry["plan_ref"])), "--root", str(campaign.root),
            "--execute", "--approved-plan-digest", entry["plan_digest"]]


def reconcile_dispatch(campaign, state, observed):
    """Reconcile the exact launch identity; never infer absence from a lost ACK."""
    outcomes = {name: item["status"] for name, item in observed.items()}
    driver_live = []
    for name in campaign.entries:
        driver_path = safe_path(campaign.root,
            str((campaign.directory / name / "driver.json").relative_to(campaign.root)))
        if driver_path.exists():
            driver = read_json(driver_path)
            _fields(driver, "campaign_digest plan_digest process argv")
            if (driver["campaign_digest"] != campaign.data["campaign_digest"] or
                    driver["plan_digest"] != campaign.entries[name]["plan_digest"] or
                    driver["argv"] != harness_argv(campaign, name)):
                raise CampaignError("retained driver identity mismatch")
            _fields(driver["process"], "pid start_ticks boot_id")
            identity = driver["process"]
            if (type(identity["pid"]) is not int or identity["pid"] <= 0 or
                    not isinstance(identity["start_ticks"], str) or not identity["start_ticks"].isdigit() or
                    not isinstance(identity["boot_id"], str)):
                raise CampaignError("malformed retained driver process identity")
            if campaign.H._alive(identity):
                driver_live.append(name)
        if state["starts"][name] and outcomes[name] in ("absent", "waiting_inputs"):
            outcomes[name] = "running" if name in driver_live else "unknown"
            observed[name] = {"status": outcomes[name], "repairable": False,
                              "reason": "dispatch intent requires exact process/receipt reconciliation"}
    return outcomes, driver_live


REPAIR_AGENT_STATUSES = {
    "not_connected", "configured_no_worker_receipt",
    "worker_returned_unreviewed", "reconcile_required", "failed",
}


def _repair_invocation(value):
    """Validate the immutable result envelope emitted by repair_bridge._invoke."""
    _fields(value, "argv returncode transport_error termination started_epoch "
            "finished_epoch stdout stdout_sha256 stdout_bytes stdout_truncated "
            "stderr stderr_sha256 stderr_bytes stderr_truncated")
    _fields(value["termination"],
            "process_group term_sent kill_sent direct_child_reaped")
    if (not isinstance(value["argv"], list) or
            any(not isinstance(part, str) for part in value["argv"]) or
            (value["returncode"] is not None and
             (isinstance(value["returncode"], bool) or
              not isinstance(value["returncode"], int))) or
            (value["transport_error"] is not None and
             not isinstance(value["transport_error"], str)) or
            any(type(value["termination"][key]) is not bool for key in
                ("term_sent", "kill_sent", "direct_child_reaped")) or
            (value["termination"]["process_group"] is not None and
             (isinstance(value["termination"]["process_group"], bool) or
              not isinstance(value["termination"]["process_group"], int))) or
            any(not isinstance(value[key], str) for key in ("stdout", "stderr")) or
            any(not isinstance(value[key], int) or value[key] < 0 for key in
                ("stdout_bytes", "stderr_bytes")) or
            any(type(value[key]) is not bool for key in
                ("stdout_truncated", "stderr_truncated"))):
        raise CampaignError("invalid repair invocation receipt")
    _number(value["started_epoch"], 0, 1e20)
    _number(value["finished_epoch"], value["started_epoch"], 1e20)
    _digest(value["stdout_sha256"])
    _digest(value["stderr_sha256"])
    return value


def _repair_outcome(status, invocation):
    suffix = status.rsplit("_", 1)[-1]
    if suffix == "returned":
        valid = invocation["returncode"] == 0 and invocation["transport_error"] is None
    elif suffix == "unknown":
        valid = invocation["returncode"] is None and isinstance(
            invocation["transport_error"], str)
    elif suffix == "failed":
        valid = (isinstance(invocation["returncode"], int) and
                 invocation["returncode"] != 0 and
                 invocation["transport_error"] is None)
    else:
        valid = False
    if not valid:
        raise CampaignError("repair invocation outcome mismatch")


def _repair_campaign_digests(campaign):
    """Return the current digest plus exactly retained extension ancestors."""
    allowed = {campaign.data["campaign_digest"]}
    history = campaign.directory / "manifest-history"
    if history.is_symlink():
        raise CampaignError("repair lineage history symlink rejected")
    if not history.exists():
        return allowed
    if not history.is_dir():
        raise CampaignError("repair lineage history must be a physical directory")
    manifests = {campaign.data["campaign_digest"]: campaign.data}
    receipts = []
    for path in history.iterdir():
        if path.is_symlink() or not path.is_file():
            raise CampaignError("repair lineage entries must be regular files")
        if re.fullmatch(r"[0-9a-f]{64}\.json", path.name):
            value = read_json(path)
            if (campaign_digest(value) != value.get("campaign_digest") or
                    path.stem != value["campaign_digest"] or
                    _fixed_campaign_envelope(value) != _fixed_campaign_envelope(campaign.data)):
                raise CampaignError("retained repair ancestor manifest mismatch")
            manifests[value["campaign_digest"]] = value
        elif path.name.endswith("-extension.json"):
            receipt = read_json(path)
            _fields(receipt, "kind previous_campaign_digest campaign_digest added_plans "
                     "started_epoch deadline_epoch")
            if (receipt["kind"] != "research-supervisor-extension-v1" or
                    path.name != receipt["campaign_digest"] + "-extension.json"):
                raise CampaignError("retained repair extension receipt mismatch")
            receipts.append(receipt)
        # State/driver/heartbeat history is validated by extension, not lineage.
    changed = True
    while changed:
        changed = False
        for receipt in receipts:
            if (receipt["campaign_digest"] in allowed and
                    receipt["previous_campaign_digest"] not in allowed):
                previous = manifests.get(receipt["previous_campaign_digest"])
                successor = manifests.get(receipt["campaign_digest"])
                if previous is None or successor is None:
                    raise CampaignError("repair extension ancestor manifest missing")
                old = [dict(entry, required_inputs=entry.get("required_inputs", []))
                       for entry in previous["plans"]]
                new = [dict(entry, required_inputs=entry.get("required_inputs", []))
                       for entry in successor["plans"]]
                if new[:len(old)] != old or len(new) <= len(old):
                    raise CampaignError("repair extension lineage is not append-only")
                allowed.add(receipt["previous_campaign_digest"])
                changed = True
    return allowed


def repair_bridge_status(campaign):
    """Read the bridge's append-only receipts without contacting a provider.

    The bridge and supervisor intentionally have separate owners.  This view is
    therefore observation only: it neither dispatches a repair worker nor treats
    a successful worker process as a reviewed patch or executable child plan.
    Malformed/cross-campaign state fails closed instead of being summarized as a
    connected agent.
    """
    parent = campaign.directory / "repair-requests"
    if parent.is_symlink():
        raise CampaignError("repair request root symlink rejected")
    if not parent.exists():
        return {"status": "not_connected", "requests": {}}
    if not parent.is_dir():
        raise CampaignError("repair request root must be a physical directory")
    allowed_campaigns = _repair_campaign_digests(campaign)
    requests = {}
    ranks = {"configured_no_worker_receipt": 1, "worker_returned_unreviewed": 2,
             "failed": 3, "reconcile_required": 4}
    aggregate = "configured_no_worker_receipt"
    directories = sorted(parent.iterdir(), key=lambda path: path.name)
    if not directories:
        return {"status": "not_connected", "requests": {}}
    for directory in directories:
        if (directory.is_symlink() or not directory.is_dir() or
                not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", directory.name)):
            raise CampaignError("invalid retained repair request directory")
        lock_path = directory / "owner.lock"
        if lock_path.is_symlink():
            raise CampaignError("repair request owner lock symlink rejected")
        if not lock_path.exists():
            item = {"status": "reconcile_required",
                    "reason": "incomplete_unlocked_request"}
        elif not lock_path.is_file():
            raise CampaignError("repair request owner lock must be a regular file")
        else:
            lock = lock_path.open("r")
            try:
                try:
                    fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
                except BlockingIOError:
                    item = {"status": "reconcile_required",
                            "reason": "bridge_owner_active"}
                else:
                    request_path = directory / "request.json"
                    if request_path.is_symlink():
                        raise CampaignError("repair request identity symlink rejected")
                    if not request_path.is_file():
                        item = {"status": "reconcile_required",
                                "reason": "incomplete_unlocked_request"}
                    else:
                        request = read_json(request_path)
                        _fields(request, "kind version request_id root runtime_executable "
                                "runtime_sha256 db db_device db_inode project_id "
                                "dedicated_project owner adapter adapters_config "
                                "adapters_sha256 project_ref admission_ref inputs "
                                "instructions output_paths max_cost max_seconds "
                                "capabilities request_digest")
                        actual_digest = hashlib.sha256(canonical({
                            key: value for key, value in request.items()
                            if key != "request_digest"}).encode()).hexdigest()
                        if (request.get("kind") != "research-runtime-repair-request" or
                                request.get("version") != 1 or
                                request.get("request_id") != directory.name or
                                request.get("project_id") != directory.name or
                                request.get("dedicated_project") is not True or
                                request.get("adapter") == "native" or
                                request.get("capabilities") !=
                                ["filesystem_read", "filesystem_write"] or
                                request.get("root") != str(campaign.root) or
                                actual_digest != request.get("request_digest")):
                            raise CampaignError("retained repair request identity mismatch")
                        identifier = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}")
                        if (any(not isinstance(request[key], str) or
                                not identifier.fullmatch(request[key]) for key in
                                ("request_id", "project_id", "owner", "adapter")) or
                                isinstance(request["max_cost"], bool) or
                                not isinstance(request["max_cost"], int) or
                                not 1 <= request["max_cost"] <= 1000000000 or
                                isinstance(request["max_seconds"], bool) or
                                not isinstance(request["max_seconds"], int) or
                                not 1 <= request["max_seconds"] <= 86400):
                            raise CampaignError("retained repair request bounds mismatch")
                        runtime = _absolute(request["runtime_executable"], directory=False)
                        database = _absolute(request["db"], directory=False)
                        adapters_path = _absolute(request["adapters_config"], directory=False)
                        if (not runtime.is_file() or not os.access(runtime, os.X_OK) or
                                runtime.is_symlink() or
                                file_digest(runtime) != _digest(request["runtime_sha256"]) or
                                not database.is_file() or database.is_symlink() or
                                type(request["db_device"]) is not int or
                                type(request["db_inode"]) is not int or
                                (database.stat().st_dev, database.stat().st_ino) !=
                                (request["db_device"], request["db_inode"]) or
                                not adapters_path.is_file() or adapters_path.is_symlink() or
                                file_digest(adapters_path) !=
                                _digest(request["adapters_sha256"])):
                            raise CampaignError("retained repair runtime identity mismatch")
                        project_path = verify_ref(campaign.root, request["project_ref"])
                        project = read_json(project_path)
                        role = project.get("assigned_role", project.get("role"))
                        adapters = project.get("adapter_allowlist",
                            project.get("allowed_adapters", project.get("adapters")))
                        if (project.get("project_id") != request["project_id"] or
                                project.get("root") != str(campaign.root) or
                                project.get("workflow_class") != "scientific_method" or
                                role != "web_supervisor" or
                                not isinstance(adapters, list) or
                                request["adapter"] not in adapters or
                                project.get("max_workers") != 1 or
                                project.get("max_cost") != request["max_cost"] or
                                project.get("max_seconds") != request["max_seconds"] or
                                not isinstance(project.get("skill_root"), str) or
                                not isinstance(project.get("skill_digest"), str) or
                                not re.fullmatch(r"[0-9a-f]{64}",
                                                 project["skill_digest"])):
                            raise CampaignError("retained repair project mismatch")
                        admission = read_json(verify_ref(
                            campaign.root, request["admission_ref"]))
                        _fields(admission, "kind version campaign_id campaign_digest "
                                "failed_plan_id status classification "
                                "scientific_retry_allowed gpu_allowed reviewer reviewed_at "
                                "repair_scope failure_ref failed_plan_ref "
                                "supervisor_status_ref budget_ref campaign_ref")
                        if (admission.get("kind") != "research-repair-admission" or
                                admission.get("version") != 1 or
                                admission.get("campaign_id") != campaign.data["campaign_id"] or
                                admission.get("campaign_digest") not in allowed_campaigns or
                                admission.get("failed_plan_id") not in campaign.entries or
                                admission.get("status") != "terminal_failed" or
                                admission.get("classification") != "code_error" or
                                admission.get("gpu_allowed") is not False or
                                admission.get("scientific_retry_allowed") is not False or
                                any(not isinstance(admission[key], str) or
                                    not admission[key] for key in
                                    ("reviewer", "reviewed_at", "repair_scope"))):
                            raise CampaignError("repair admission is not bound to this campaign")
                        for name in ("inputs", "instructions"):
                            refs = request[name]
                            if not isinstance(refs, list) or not refs:
                                raise CampaignError("repair request refs must be nonempty")
                            keys = []
                            for ref in refs:
                                verify_ref(campaign.root, ref)
                                keys.append((ref["path"], ref["sha256"]))
                            if len(keys) != len(set(keys)):
                                raise CampaignError("duplicate repair request ref")
                        input_keys = {(ref["path"], ref["sha256"])
                                      for ref in request["inputs"]}
                        required_refs = (admission["failure_ref"],
                                         admission["failed_plan_ref"],
                                         admission["supervisor_status_ref"],
                                         admission["budget_ref"],
                                         admission["campaign_ref"])
                        if any((ref["path"], ref["sha256"]) not in input_keys
                               for ref in required_refs):
                            raise CampaignError("repair admission evidence not in task inputs")
                        failure = read_json(verify_ref(
                            campaign.root, admission["failure_ref"]))
                        failed_campaign = read_json(verify_ref(
                            campaign.root, admission["campaign_ref"]))
                        if (failure.get("status") != "failed" or
                                failed_campaign.get("kind") !=
                                "research-harness-campaign" or
                                failed_campaign.get("campaign_id") !=
                                admission["campaign_id"] or
                                campaign_digest(failed_campaign) !=
                                admission["campaign_digest"] or
                                failed_campaign.get("campaign_digest") !=
                                admission["campaign_digest"]):
                            raise CampaignError("repair failed campaign evidence mismatch")
                        selected = next((entry for entry in failed_campaign.get("plans", [])
                                         if entry.get("id") ==
                                         admission["failed_plan_id"]), None)
                        if (selected is None or
                                selected.get("plan_ref") !=
                                admission["failed_plan_ref"] or
                                failure.get("plan_digest") !=
                                selected.get("plan_digest") or
                                campaign.entries[admission["failed_plan_id"]]["plan_ref"] !=
                                admission["failed_plan_ref"]):
                            raise CampaignError("repair failed plan evidence mismatch")
                        supervisor_status = read_json(verify_ref(
                            campaign.root, admission["supervisor_status_ref"]))
                        observed = supervisor_status.get("plans", {}).get(
                            admission["failed_plan_id"])
                        if (supervisor_status.get("status") != "inspection" or
                                supervisor_status.get("campaign_digest") !=
                                admission["campaign_digest"] or
                                not isinstance(observed, dict) or
                                observed.get("status") != "failed" or
                                observed.get("repairable") is not True):
                            raise CampaignError("repair supervisor evidence mismatch")
                        budget = read_json(verify_ref(
                            campaign.root, admission["budget_ref"]))
                        _fields(budget, "kind version campaign_id campaign_digest "
                                "max_cost max_seconds")
                        if (budget.get("kind") != "research-repair-campaign-budget" or
                                budget.get("version") != 1 or
                                budget.get("campaign_id") != admission["campaign_id"] or
                                budget.get("campaign_digest") !=
                                admission["campaign_digest"] or
                                isinstance(budget.get("max_cost"), bool) or
                                not isinstance(budget.get("max_cost"), int) or
                                not 0 <= budget["max_cost"] <= 1000000000 or
                                isinstance(budget.get("max_seconds"), bool) or
                                not isinstance(budget.get("max_seconds"), int) or
                                not 0 <= budget["max_seconds"] <= 86400 or
                                request["max_cost"] > budget["max_cost"] or
                                request["max_seconds"] > budget["max_seconds"]):
                            raise CampaignError("repair budget evidence mismatch")
                        outputs = request["output_paths"]
                        output_prefix = ("repairs/" + admission["campaign_id"] + "/" +
                                         request["request_id"] + "/")
                        if (not isinstance(outputs, list) or not outputs or
                                len(outputs) != len(set(outputs))):
                            raise CampaignError("invalid repair output inventory")
                        existing_outputs = []
                        for relative in outputs:
                            target = safe_path(campaign.root, relative)
                            if (not relative.startswith(output_prefix) or
                                    target.exists() and
                                    (target.is_symlink() or not target.is_file())):
                                raise CampaignError("invalid repair output namespace")
                            if target.exists():
                                existing_outputs.append(relative)
                        item = {"request_digest": request["request_digest"],
                                "status": "configured_no_worker_receipt"}
                        event_statuses = []
                        event_values = []
                        last_path = directory / "last-status.json"
                        if last_path.is_symlink():
                            raise CampaignError("repair last status symlink rejected")
                        if last_path.exists():
                            if not last_path.is_file():
                                raise CampaignError("repair last status must be a regular file")
                            last = read_json(last_path)
                            _fields(last, "event status request_digest")
                            if last["request_digest"] != request["request_digest"]:
                                raise CampaignError("repair status request identity mismatch")
                            events = directory / "events"
                            if events.is_symlink() or not events.is_dir():
                                raise CampaignError("repair event root must be a physical directory")
                            entries = list(events.iterdir())
                            if any(path.is_symlink() or not path.is_file() for path in entries):
                                raise CampaignError("repair events must be regular files")
                            names = [path.name for path in entries]
                            if any(not re.fullmatch(r"[0-9]{20}-[0-9a-f]{64}\.json", name)
                                   for name in names):
                                raise CampaignError("invalid repair event inventory")
                            event_hashes = []
                            event_by_name = {}
                            previous = None
                            ordered = sorted(entries, key=lambda path: path.name)
                            for sequence, path in enumerate(ordered, 1):
                                value = read_json(path)
                                digest = hashlib.sha256(canonical(value).encode()).hexdigest()
                                if (path.name.split("-", 1)[1] != digest + ".json" or
                                        path.name.split("-", 1)[0] != f"{sequence:020d}" or
                                        value.get("sequence") != sequence or
                                        value.get("previous_event_sha256") != previous or
                                        value.get("request_digest") !=
                                        request["request_digest"]):
                                    raise CampaignError(
                                        "repair event inventory identity mismatch")
                                event_hashes.append(digest)
                                event_by_name[path.name] = value
                                event_statuses.append(value.get("status"))
                                event_values.append(value)
                                previous = digest
                            event_path = events / last["event"]
                            if last["event"] not in names:
                                raise CampaignError("repair status event identity missing")
                            event = event_by_name[event_path.name]
                            expected_hash = hashlib.sha256(canonical(event).encode()).hexdigest()
                            if (last["event"].split("-", 1)[1] != expected_hash + ".json" or
                                    event.get("status") != last["status"] or
                                    event.get("request_digest") != request["request_digest"]):
                                raise CampaignError("repair status event identity mismatch")
                            journal = directory / "journal.jsonl"
                            if journal.is_symlink() or not journal.is_file():
                                raise CampaignError("repair journal must be a regular file")
                            lines = journal.read_text(encoding="utf-8").splitlines()
                            try:
                                journal_hashes = [hashlib.sha256(canonical(json.loads(
                                    line, object_pairs_hook=_pairs)).encode()).hexdigest()
                                    for line in lines]
                            except (ValueError, TypeError) as error:
                                raise CampaignError("invalid repair journal") from error
                            raw = last["status"]
                            if (journal_hashes != event_hashes or
                                    not journal_hashes or journal_hashes[-1] != expected_hash):
                                status, raw = "reconcile_required", "unindexed_event"
                            elif raw == "worker_returned":
                                status = "worker_returned_unreviewed"
                            elif raw in {"worker_unknown", "reconcile_required",
                                         "enqueue_unknown", "project_registration_unknown"}:
                                status = "reconcile_required"
                            elif raw in {"worker_failed", "enqueue_failed",
                                         "project_registration_failed",
                                         "budget_reservation_failed"}:
                                status = "failed"
                            elif raw in {"project_registration_returned", "budget_reserved",
                                         "enqueue_returned"}:
                                status = "reconcile_required"
                            else:
                                status = "reconcile_required"
                            item.update(status=status, bridge_status=raw,
                                        event=last["event"])
                        # Validate the exact bridge-emitted result envelopes and
                        # the only legal side-effect progression.  Hashes prove
                        # bytes; this proves those bytes are a bridge receipt.
                        for value in event_values:
                            status = value.get("status")
                            if status in {"project_registration_returned",
                                          "project_registration_unknown",
                                          "project_registration_failed"}:
                                _fields(value, "status stage request_digest registration "
                                        "sequence previous_event_sha256")
                                if value["stage"] != "register-project":
                                    raise CampaignError("repair registration stage mismatch")
                                _repair_invocation(value["registration"])
                                _repair_outcome(status, value["registration"])
                            elif status == "budget_reserved":
                                _fields(value, "status stage request_digest reservation_ref "
                                        "sequence previous_event_sha256")
                                if value["stage"] != "reserve-campaign-budget":
                                    raise CampaignError("repair reservation stage mismatch")
                                reservation = read_json(verify_ref(
                                    campaign.root, value["reservation_ref"]))
                                _fields(reservation, "kind version campaign_digest request_id "
                                        "request_digest max_cost max_seconds")
                                if reservation != {
                                        "kind": "research-repair-reservation",
                                        "version": 1,
                                        "campaign_digest": admission["campaign_digest"],
                                        "request_id": request["request_id"],
                                        "request_digest": request["request_digest"],
                                        "max_cost": request["max_cost"],
                                        "max_seconds": request["max_seconds"]}:
                                    raise CampaignError(
                                        "repair reservation receipt identity mismatch")
                            elif status == "budget_reservation_failed":
                                _fields(value, "status stage request_digest error sequence "
                                        "previous_event_sha256")
                                if (value["stage"] != "reserve-campaign-budget" or
                                        not isinstance(value["error"], str) or
                                        not value["error"]):
                                    raise CampaignError("repair reservation failure mismatch")
                            elif status in {"enqueue_returned", "enqueue_unknown",
                                            "enqueue_failed"}:
                                _fields(value, "status stage request_digest enqueue sequence "
                                        "previous_event_sha256")
                                if value["stage"] != "enqueue":
                                    raise CampaignError("repair enqueue stage mismatch")
                                _repair_invocation(value["enqueue"])
                                _repair_outcome(status, value["enqueue"])
                            elif status in {"worker_returned", "worker_unknown",
                                            "worker_failed"}:
                                _fields(value, "status request_digest registration enqueue "
                                        "worker sequence previous_event_sha256")
                                for key in ("registration", "enqueue", "worker"):
                                    _repair_invocation(value[key])
                                _repair_outcome(status, value["worker"])
                        core_statuses = list(event_statuses)
                        if "reconcile_required" in core_statuses:
                            cut = core_statuses.index("reconcile_required")
                            if any(status != "reconcile_required"
                                   for status in core_statuses[cut:]):
                                core_statuses = ["invalid_progression"]
                            else:
                                core_statuses = core_statuses[:cut]
                        legal_progressions = {
                            (),
                            ("project_registration_returned",),
                            ("project_registration_unknown",),
                            ("project_registration_failed",),
                            ("project_registration_returned", "budget_reserved"),
                            ("project_registration_returned",
                             "budget_reservation_failed"),
                            ("project_registration_returned", "budget_reserved",
                             "enqueue_returned"),
                            ("project_registration_returned", "budget_reserved",
                             "enqueue_unknown"),
                            ("project_registration_returned", "budget_reserved",
                             "enqueue_failed"),
                            ("project_registration_returned", "budget_reserved",
                             "enqueue_returned", "worker_returned"),
                            ("project_registration_returned", "budget_reserved",
                             "enqueue_returned", "worker_unknown"),
                            ("project_registration_returned", "budget_reserved",
                             "enqueue_returned", "worker_failed"),
                        }
                        progression_valid = tuple(core_statuses) in legal_progressions
                        if len(core_statuses) == 4 and progression_valid:
                            if (event_values[3]["registration"] !=
                                    event_values[0]["registration"] or
                                    event_values[3]["enqueue"] !=
                                    event_values[2]["enqueue"]):
                                raise CampaignError("repair worker predecessor mismatch")
                        prefix = [request["runtime_executable"], "--db", request["db"]]
                        task_path = directory / "task.json"
                        expected_task = {
                            "task_id": request["request_id"],
                            "action": "candidate_code",
                            "inputs": list(request["inputs"]),
                            "instructions": [request["admission_ref"],
                                             *request["instructions"]],
                            "dependencies": [],
                            "output_paths": list(request["output_paths"]),
                            "completion": {"kind": "outputs_verified"},
                            "max_cost": request["max_cost"],
                            "max_seconds": request["max_seconds"],
                            "capabilities": list(request["capabilities"]),
                            "adapter": request["adapter"],
                            "parameters": {
                                "campaign_digest": admission["campaign_digest"],
                                "failed_plan_id": admission["failed_plan_id"],
                                "gpu_allowed": False,
                                "scientific_retry_allowed": False,
                                "requires_reviewed_child_version": True}}
                        if task_path.is_symlink():
                            raise CampaignError("repair task identity symlink rejected")
                        if not task_path.exists():
                            item.update(status="reconcile_required",
                                        bridge_status="retained_task_missing")
                        elif (not task_path.is_file() or
                              read_json(task_path) != expected_task):
                            raise CampaignError("repair task identity mismatch")
                        stage_specs = {
                            "registration-intent.json": ({
                                "kind": "research-repair-stage-intent-v1",
                                "stage": "register-project",
                                "request_digest": request["request_digest"],
                                "argv": prefix + ["register-project", str(project_path)]},
                                {"project_registration_returned",
                                 "project_registration_unknown",
                                 "project_registration_failed"}),
                            "budget-reservation-intent.json": ({
                                "kind": "research-repair-stage-intent-v1",
                                "stage": "reserve-campaign-budget",
                                "request_digest": request["request_digest"],
                                "max_cost": request["max_cost"],
                                "max_seconds": request["max_seconds"]},
                                {"budget_reserved", "budget_reservation_failed"}),
                            "enqueue-intent.json": ({
                                "kind": "research-repair-stage-intent-v1",
                                "stage": "enqueue",
                                "request_digest": request["request_digest"],
                                "argv": prefix + ["enqueue", request["project_id"],
                                                  str(task_path)]},
                                {"enqueue_returned", "enqueue_unknown",
                                 "enqueue_failed"}),
                        }
                        expected_worker = {
                            "kind": "research-repair-worker-intent-v1",
                            "request_digest": request["request_digest"],
                            "project_id": request["project_id"],
                            "task_id": request["request_id"],
                            "owner": request["owner"],
                            "adapter": request["adapter"],
                            "argv": prefix + ["worker", request["project_id"],
                                              "--owner", request["owner"],
                                              "--adapter", request["adapter"],
                                              "--adapters-config",
                                              request["adapters_config"]],
                            "timeout_seconds": request["max_seconds"],
                        }
                        for event_value in event_values:
                            status = event_value.get("status")
                            if (status in {"project_registration_returned",
                                           "project_registration_unknown",
                                           "project_registration_failed"} and
                                    event_value["registration"]["argv"] !=
                                    stage_specs["registration-intent.json"][0]["argv"]):
                                raise CampaignError(
                                    "repair registration invocation identity mismatch")
                            if (status in {"enqueue_returned", "enqueue_unknown",
                                          "enqueue_failed"} and
                                    event_value["enqueue"]["argv"] !=
                                    stage_specs["enqueue-intent.json"][0]["argv"]):
                                raise CampaignError(
                                    "repair enqueue invocation identity mismatch")
                            if (status in {"worker_returned", "worker_unknown",
                                          "worker_failed"} and
                                    event_value["worker"]["argv"] !=
                                    expected_worker["argv"]):
                                raise CampaignError(
                                    "repair worker invocation identity mismatch")
                        unmatched_stage = []
                        present_stage = set()
                        for name, (expected, terminal_statuses) in stage_specs.items():
                            stage_path = directory / name
                            if stage_path.is_symlink():
                                raise CampaignError("repair stage intent symlink rejected")
                            if stage_path.exists():
                                if not stage_path.is_file():
                                    raise CampaignError(
                                        "repair stage intent must be a regular file")
                                value = read_json(stage_path)
                                if value != expected:
                                    raise CampaignError(
                                        "repair stage intent identity mismatch")
                                present_stage.add(expected["stage"])
                                if not terminal_statuses.intersection(event_statuses):
                                    unmatched_stage.append(expected["stage"])
                        if unmatched_stage:
                            item.update(status="reconcile_required",
                                bridge_status="stage_intent_without_result",
                                unmatched_stage_intents=unmatched_stage)
                        intent = directory / "worker-intent.json"
                        worker_intent_present = False
                        if intent.is_symlink():
                            raise CampaignError("repair worker intent symlink rejected")
                        if intent.exists():
                            if not intent.is_file():
                                raise CampaignError(
                                    "repair worker intent must be a regular file")
                            value = read_json(intent)
                            if value != expected_worker:
                                raise CampaignError("repair worker intent identity mismatch")
                            worker_intent_present = True
                            if item.get("bridge_status") not in {
                                    "worker_returned", "worker_failed", "worker_unknown",
                                    "reconcile_required"}:
                                item.update(status="reconcile_required",
                                    bridge_status="worker_intent_without_terminal_event")
                        worker_terminal = bool(core_statuses and core_statuses[-1] in {
                            "worker_returned", "worker_failed", "worker_unknown"})
                        required_stage = set()
                        if core_statuses:
                            required_stage.add("register-project")
                        if len(core_statuses) >= 2:
                            required_stage.add("reserve-campaign-budget")
                        if len(core_statuses) >= 3:
                            required_stage.add("enqueue")
                        if (not progression_valid or
                                present_stage != required_stage or
                                worker_terminal and
                                not worker_intent_present):
                            item.update(status="reconcile_required",
                                        bridge_status="invalid_bridge_progression")
                        if existing_outputs and not worker_intent_present:
                            item.update(status="reconcile_required",
                                        bridge_status="unowned_repair_output",
                                        existing_outputs=existing_outputs)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
                lock.close()
        requests[directory.name] = item
        if ranks[item["status"]] > ranks[aggregate]:
            aggregate = item["status"]
    return {"status": aggregate, "requests": requests}


def retained_status(campaign):
    """Observation only: retained heartbeat is labelled stale, never a receipt."""
    directory = campaign.directory
    repair = repair_bridge_status(campaign)
    result = {"stop_requested": (directory / "STOP").exists(),
              "online_repair_agent": repair["status"],
              "repair_bridge": repair, "owner_live": False,
              "supervisor_lock_held": _lock_busy(directory / "campaign.lock")}
    heartbeat_path = directory / "heartbeat.json"
    if heartbeat_path.is_file():
        heartbeat = read_json(heartbeat_path)
        if heartbeat.get("campaign_digest") != campaign.data["campaign_digest"]:
            raise CampaignError("heartbeat identity mismatch")
        result.update(heartbeat=heartbeat,
                      heartbeat_age_seconds=max(0, time.time() - heartbeat["observed_epoch"]),
                      owner_live=campaign.H._alive(heartbeat.get("owner")))
    return result


@contextmanager
def control_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "control.lock"
    if path.is_symlink():
        raise CampaignError("control lock symlink rejected")
    with path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def stop_target(path, approved_digest):
    manifest = read_json(path)
    if (manifest.get("kind") != "research-harness-campaign" or
            manifest.get("version") not in (1, 2) or
            campaign_digest(manifest) != manifest.get("campaign_digest") or
            approved_digest != manifest.get("campaign_digest")):
        raise CampaignError("exact campaign identity required for STOP")
    name = manifest.get("campaign_id")
    if not isinstance(name, str) or not re.fullmatch("[A-Za-z0-9][A-Za-z0-9_-]{0,95}", name):
        raise CampaignError("invalid campaign identity for STOP")
    directory = safe_path(_absolute(manifest["root"]), "runs/supervisor/" + name)
    return directory


def stop_identity(directory):
    path = directory / "STOP"
    if path.is_symlink():
        raise CampaignError("STOP symlink rejected")
    if not path.exists():
        return None
    with path.open("rb") as stream:
        stat = os.fstat(stream.fileno())
        return (stat.st_dev, stat.st_ino, stat.st_mtime_ns,
                hashlib.sha256(stream.read()).hexdigest())


def request_stop(path, approved_digest):
    """Stop remains available when missing inputs or a broken skill block loading.

    This writes only the canonical campaign sentinel; it never signals a PID,
    rewrites state, resumes an experiment or declares any workload terminated.
    """
    directory = stop_target(path, approved_digest)
    with control_lock(directory):
        _atomic(directory / "STOP", {"campaign_digest": approved_digest,
                                    "requested_epoch": time.time(),
                                    "generation": os.urandom(16).hex()})
    return {"status": "stop_requested", "campaign_digest": approved_digest,
            "workers_terminated": False, "gpu_dispatch_enabled": False}


def validate_retained_state(state, campaign, observed_now):
    """Apply one state contract to inspection, execution and extension."""
    try:
        _fields(state, "kind campaign_digest started_epoch deadline_epoch last_epoch active starts")
        for key in ("started_epoch", "deadline_epoch", "last_epoch"):
            _number(state[key], 0, 1e20)
        starts = state["starts"]
        valid = (
            state["kind"] == "research-supervisor-state-v1" and
            state["campaign_digest"] == campaign.data["campaign_digest"] and
            state["deadline_epoch"] == state["started_epoch"] + campaign.data["total_wall_seconds"] and
            state["last_epoch"] >= state["started_epoch"] and
            observed_now >= state["last_epoch"] and
            (state["active"] is None or state["active"] in campaign.entries) and
            isinstance(starts, dict) and set(starts) == set(campaign.entries) and
            all(type(value) is int and 0 <= value <= 32 for value in starts.values()))
    except (CampaignError, KeyError, TypeError, ValueError) as error:
        raise CampaignError("invalid retained campaign state") from error
    if not valid:
        raise CampaignError("invalid retained campaign state")
    return state


def validate_heartbeat(value, campaign, state, observed_now):
    """Validate archived heartbeat identity; status is evidence, never a receipt."""
    try:
        common = ("kind campaign_digest observed_epoch owner deadline_epoch "
                  "remaining_seconds plans gpu_observation gpu_observation_available "
                  "gpu_no_compute_process_observed gpu_idle_while_retained_running "
                  "gpu_dispatch_enabled online_repair_agent scientific_result_verified")
        if value.get("kind") == "research-supervisor-heartbeat-v2":
            _fields(value, common + " repair_bridge_status")
        else:
            _fields(value, common)
        _number(value["observed_epoch"], 0, 1e20)
        _number(value["deadline_epoch"], 0, 1e20)
        _number(value["remaining_seconds"], 0, 28800)
        _fields(value["owner"], "pid start_ticks boot_id")
        valid = (
            value["kind"] in {"research-supervisor-heartbeat-v1",
                              "research-supervisor-heartbeat-v2"} and
            value["campaign_digest"] == campaign.data["campaign_digest"] and
            value["deadline_epoch"] == state["deadline_epoch"] and
            observed_now >= value["observed_epoch"] and
            type(value["owner"]["pid"]) is int and value["owner"]["pid"] > 0 and
            isinstance(value["owner"]["start_ticks"], str) and value["owner"]["start_ticks"].isdigit() and
            isinstance(value["owner"]["boot_id"], str) and
            isinstance(value["plans"], dict) and set(value["plans"]) == set(campaign.entries) and
            isinstance(value["gpu_observation"], list) and
            all(type(value[key]) is bool for key in (
                "gpu_observation_available", "gpu_no_compute_process_observed",
                "gpu_idle_while_retained_running", "gpu_dispatch_enabled")) and
            value["online_repair_agent"] in REPAIR_AGENT_STATUSES and
            (value["kind"] == "research-supervisor-heartbeat-v1" and
             value["online_repair_agent"] == "not_connected" or
             value["kind"] == "research-supervisor-heartbeat-v2" and
             value.get("repair_bridge_status") == value["online_repair_agent"]) and
            value["scientific_result_verified"] is False)
    except (CampaignError, KeyError, TypeError, ValueError) as error:
        raise CampaignError("invalid retained heartbeat identity") from error
    if not valid:
        raise CampaignError("invalid retained heartbeat identity")
    return value


def regular_identity(path):
    record = os.lstat(path)
    if not stat.S_ISREG(record.st_mode):
        raise CampaignError("nonregular retained artifact rejected")
    return (record.st_dev, record.st_ino, record.st_size, record.st_mtime_ns)


def _normalized_entry(entry):
    """Compare v1/v2 entries without treating an empty readiness list as a change."""
    value = dict(entry)
    value.setdefault("required_inputs", [])
    return value


def _fixed_campaign_envelope(manifest):
    return {key: manifest[key] for key in (
        "kind", "campaign_id", "root", "python", "skill_root", "skill_digest",
        "pool_dir", "total_wall_seconds", "collection_reserve_seconds")}


def extend_campaign(current_path, next_path, *, approved_current_digest,
                    approved_next_digest, now=time.time):
    """Activate one exact append-only campaign revision without resetting state.

    Both arguments are complete reviewed manifests.  The next revision may append
    immutable plans only; it cannot rewrite the campaign envelope or any retained
    plan.  Retained execution records are migrated under one exclusive owner and
    the original started/deadline epochs survive unchanged.
    """
    current_data = read_json(current_path)
    next_data = read_json(next_path)
    if (campaign_digest(current_data) != current_data.get("campaign_digest") or
            approved_current_digest != current_data.get("campaign_digest")):
        raise CampaignError("exact current campaign approval required")
    if (campaign_digest(next_data) != next_data.get("campaign_digest") or
            approved_next_digest != next_data.get("campaign_digest")):
        raise CampaignError("exact next campaign approval required")
    if next_data.get("version") != 2:
        raise CampaignError("append-only extension requires manifest version 2")
    if _fixed_campaign_envelope(current_data) != _fixed_campaign_envelope(next_data):
        raise CampaignError("fixed campaign envelope cannot change during extension")
    old_plans = current_data.get("plans")
    new_plans = next_data.get("plans")
    if (not isinstance(old_plans, list) or not isinstance(new_plans, list) or
            len(new_plans) <= len(old_plans) or
            [_normalized_entry(entry) for entry in new_plans[:len(old_plans)]] !=
            [_normalized_entry(entry) for entry in old_plans]):
        raise CampaignError("extension must be a strict append-only plan inventory")

    # Full construction validates every old/new plan, native identity, exact code
    # ref, cumulative reservation, dependency and repair edge before state changes.
    current = Campaign(current_path)
    proposed = Campaign(next_path)
    if current.data != current_data or proposed.data != next_data:
        raise CampaignError("approved manifest changed before validation")
    if current.directory != proposed.directory:
        raise CampaignError("fixed campaign owner changed during extension")
    directory = current.directory
    state_path, manifest_path = directory / "state.json", directory / "manifest.json"
    lock_path = safe_path(current.root, str((directory / "campaign.lock").relative_to(current.root)))
    directory.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open("a+")
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise CampaignError("extension requires a settled campaign with no active owner") from error
        with control_lock(directory):
            if read_json(current.path) != current_data or read_json(proposed.path) != next_data:
                raise CampaignError("approved manifest changed before retained-state mutation")
            if not state_path.is_file() or not manifest_path.is_file():
                raise CampaignError("extension requires retained campaign state; start the reviewed base first")
            state = read_json(state_path)
            retained_manifest = read_json(manifest_path)
            history = safe_path(current.root,
                                str((directory / "manifest-history").relative_to(current.root)))
            if history.is_symlink():
                raise CampaignError("manifest history symlink rejected")
            observed_now = now()
            if (state.get("campaign_digest") == proposed.data["campaign_digest"] and
                    retained_manifest == proposed.data):
                validate_retained_state(state, proposed, observed_now)
                if state["active"] is not None:
                    raise CampaignError("extension requires a settled campaign; active owner retained")
                receipt_path = safe_path(current.root, str((history / (
                    proposed.data["campaign_digest"] + "-extension.json")).relative_to(current.root)))
                if not receipt_path.is_file():
                    raise CampaignError("retained extension predecessor receipt missing")
                receipt = read_json(receipt_path)
                expected_receipt = {
                    "kind": "research-supervisor-extension-v1",
                    "previous_campaign_digest": current.data["campaign_digest"],
                    "campaign_digest": proposed.data["campaign_digest"],
                    "added_plans": list(proposed.entries)[len(current.entries):],
                    "started_epoch": state["started_epoch"],
                    "deadline_epoch": state["deadline_epoch"]}
                if receipt != expected_receipt:
                    raise CampaignError("retained extension predecessor does not match this transition")
                return {
                    "status": "already_extended",
                    "previous_campaign_digest": current.data["campaign_digest"],
                    "campaign_digest": proposed.data["campaign_digest"],
                    "added_plans": list(proposed.entries)[len(current.entries):],
                    "started_epoch": state["started_epoch"],
                    "deadline_epoch": state["deadline_epoch"],
                    "stop_preserved": (directory / "STOP").exists(),
                    "gpu_dispatch_enabled": False,
                    "online_repair_agent": "readback_required"}
            if (state.get("campaign_digest") != current.data["campaign_digest"] or
                    retained_manifest not in (current.data, proposed.data)):
                raise CampaignError("extension requires exact settled campaign identity")
            validate_retained_state(state, current, observed_now)
            if state["active"] is not None:
                raise CampaignError("extension requires a settled campaign; active owner retained")
            observed = {name: current.observe(name) for name in current.entries}
            outcomes, live = reconcile_dispatch(current, state, observed)
            if live or any(status in ("running", "unknown") for status in outcomes.values()):
                raise CampaignError("extension requires a settled campaign; reconcile active or unknown work first")

            heartbeat_path = safe_path(current.root, str((directory / "heartbeat.json").relative_to(current.root)))
            heartbeat_identity = heartbeat_value = None
            if heartbeat_path.exists():
                heartbeat_identity = regular_identity(heartbeat_path)
                heartbeat_value = validate_heartbeat(
                    read_json(heartbeat_path), current, state, observed_now)
                if regular_identity(heartbeat_path) != heartbeat_identity:
                    raise CampaignError("retained heartbeat changed during extension")

            history.mkdir(exist_ok=True)
            previous = safe_path(current.root, str((history / (
                current.data["campaign_digest"] + ".json")).relative_to(current.root)))
            if previous.exists() and read_json(previous) != current.data:
                raise CampaignError("conflicting retained campaign history")
            if not previous.exists():
                _atomic(previous, current.data)
            archived_state = safe_path(current.root, str((history / (
                current.data["campaign_digest"] + "-state.json")).relative_to(current.root)))
            if archived_state.exists() and read_json(archived_state) != state:
                raise CampaignError("conflicting retained campaign state history")
            if not archived_state.exists():
                _atomic(archived_state, state)

            # Completed driver records retain their original campaign identity in
            # history; only live/uncertain records were rejected above.
            for name in current.entries:
                driver = directory / name / "driver.json"
                if driver.is_file():
                    archived = safe_path(current.root, str((history / (
                        current.data["campaign_digest"] + "-" + name + "-driver.json")).relative_to(current.root)))
                    if archived.exists() and read_json(archived) != read_json(driver):
                        raise CampaignError("conflicting retained driver history")
                    if not archived.exists():
                        os.replace(driver, archived)
                        _sync_directory(history)
                        _sync_directory(driver.parent)
                    else:
                        driver.unlink()
                        _sync_directory(driver.parent)
            if heartbeat_identity is not None:
                archived = safe_path(current.root, str((history / (
                    current.data["campaign_digest"] + "-heartbeat.json")).relative_to(current.root)))
                if archived.exists() and read_json(archived) != heartbeat_value:
                    raise CampaignError("conflicting retained heartbeat history")
                if not archived.exists():
                    if regular_identity(heartbeat_path) != heartbeat_identity:
                        raise CampaignError("retained heartbeat changed before archival")
                    os.replace(heartbeat_path, archived)
                    _sync_directory(history)
                    _sync_directory(directory)
                else:
                    heartbeat_path.unlink()
                    _sync_directory(directory)

            migrated = dict(state)
            migrated["campaign_digest"] = proposed.data["campaign_digest"]
            migrated["last_epoch"] = max(state["last_epoch"], observed_now)
            migrated["starts"] = {name: state["starts"].get(name, 0)
                                  for name in proposed.entries}
            # The deadline is intentionally copied, not recomputed: extension is
            # new reviewed work inside the same cumulative campaign, not a reset.
            receipt_path = safe_path(current.root, str((history / (
                proposed.data["campaign_digest"] + "-extension.json")).relative_to(current.root)))
            receipt = {
                "kind": "research-supervisor-extension-v1",
                "previous_campaign_digest": current.data["campaign_digest"],
                "campaign_digest": proposed.data["campaign_digest"],
                "added_plans": list(proposed.entries)[len(current.entries):],
                "started_epoch": migrated["started_epoch"],
                "deadline_epoch": migrated["deadline_epoch"]}
            if receipt_path.exists() and read_json(receipt_path) != receipt:
                raise CampaignError("conflicting retained extension predecessor receipt")
            if not receipt_path.exists():
                _atomic(receipt_path, receipt)
            _atomic(manifest_path, proposed.data)
            _atomic(state_path, migrated)
            with (directory / "journal.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(canonical({
                    "event": "campaign_extended", "observed_epoch": now(),
                    "campaign_digest": proposed.data["campaign_digest"],
                    "previous_campaign_digest": current.data["campaign_digest"],
                    "added_plans": list(proposed.entries)[len(current.entries):],
                    "deadline_epoch": migrated["deadline_epoch"]}) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            _sync_directory(directory)
            return {
                "status": "extended",
                "previous_campaign_digest": current.data["campaign_digest"],
                "campaign_digest": proposed.data["campaign_digest"],
                "added_plans": list(proposed.entries)[len(current.entries):],
                "started_epoch": migrated["started_epoch"],
                "deadline_epoch": migrated["deadline_epoch"],
                "stop_preserved": (directory / "STOP").exists(),
                "gpu_dispatch_enabled": False,
                "online_repair_agent": "readback_required"}
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def run_campaign(path, *, execute=False, approved_digest=None, poll_seconds=1.0,
                 now=time.time, transport=None, allow_gpu_after_user_resume=False,
                 watch_ready=False, heartbeat_seconds=30.0, resume=False):
    # This flag acknowledges a separate explicit human resume; it cannot prove or
    # manufacture that permission. It is deliberately absent from the manifest.
    if type(allow_gpu_after_user_resume) is not bool:
        raise CampaignError("explicit Local resume acknowledgment must be boolean")
    _number(poll_seconds, .05, 30)
    _number(heartbeat_seconds, .05, 60)
    if type(watch_ready) is not bool or type(resume) is not bool:
        raise CampaignError("watch_ready/resume must be boolean")
    if resume and not execute:
        raise CampaignError("resume requires execution approval")
    resume_stop = None
    if resume:
        resume_directory = stop_target(path, approved_digest)
        with control_lock(resume_directory):
            resume_stop = stop_identity(resume_directory)
    campaign = Campaign(path)
    m, directory = campaign.data, campaign.directory
    observations = {name: campaign.observe(name) for name in campaign.entries}
    if not execute:
        state_path = directory / "state.json"
        if state_path.exists():
            retained = read_json(state_path)
            if read_json(directory / "manifest.json") != m:
                raise CampaignError("invalid retained campaign inspection identity")
            try:
                validate_retained_state(retained, campaign, time.time())
            except CampaignError as error:
                raise CampaignError("invalid retained campaign inspection identity") from error
            reconcile_dispatch(campaign, retained, observations)
        return {"status": "inspection", "campaign_digest": m["campaign_digest"],
                "gpu_dispatch_enabled": allow_gpu_after_user_resume, "plans": observations,
                "retained_status": retained_status(campaign)}
    if approved_digest != m["campaign_digest"]:
        raise CampaignError("exact explicit campaign approval required")
    if not Path("/proc/sys/kernel/random/boot_id").is_file():
        raise CampaignError("Local Linux execution host required")
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = safe_path(campaign.root, str((directory / "campaign.lock").relative_to(campaign.root)))
    lock = lock_path.open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock.close()
        raise CampaignError("campaign supervisor already active") from error
    stop_requested = [False]
    old_signals = {}
    child = None
    def stop(signum, frame):
        stop_requested[0] = True
    state_path, manifest_path = directory / "state.json", directory / "manifest.json"
    state = None
    def checkpoint(event, **details):
        state["last_epoch"] = max(state["last_epoch"], now())
        _atomic(state_path, state)
        with (directory / "journal.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(canonical({"event": event, "observed_epoch": now(),
                                    "campaign_digest": m["campaign_digest"], **details}) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    try:
        if state_path.exists() or manifest_path.exists():
            if not state_path.is_file() or not manifest_path.is_file() or read_json(manifest_path) != m:
                raise CampaignError("incomplete or conflicting retained campaign; reconciliation required")
            state = read_json(state_path)
            try:
                validate_retained_state(state, campaign, now())
            except CampaignError as error:
                raise CampaignError("invalid retained deadline or identity; reconciliation required")
        else:
            if resume:
                raise CampaignError("resume requires original retained state; no fresh budget")
            if (directory / "journal.jsonl").exists():
                raise CampaignError("journal without state; no budget reset")
            start = now()
            state = {"kind": "research-supervisor-state-v1", "campaign_digest": m["campaign_digest"],
                     "started_epoch": start, "deadline_epoch": start + m["total_wall_seconds"],
                     "last_epoch": start, "active": None, "starts": {name: 0 for name in campaign.entries}}
            _atomic(manifest_path, m)
            checkpoint("campaign_started")
        if resume:
            with control_lock(directory):
                if stop_identity(directory) != resume_stop:
                    raise CampaignError("newer STOP arrived during resume; request retained")
                stop_path = safe_path(campaign.root, str((directory / "STOP").relative_to(campaign.root)))
                if stop_path.exists():
                    stop_path.unlink()
                    _sync_directory(directory)
                checkpoint("explicit_resume", gpu_dispatch_enabled=allow_gpu_after_user_resume)
        for signum in (signal.SIGINT, signal.SIGTERM):
            old_signals[signum] = signal.signal(signum, stop)
        monotonic_deadline = time.monotonic() + max(0, state["deadline_epoch"] - now())
        def remaining():
            return min(state["deadline_epoch"] - now(), monotonic_deadline - time.monotonic())
        def stopped():
            return stop_requested[0] or (directory / "STOP").exists()
        def report(status, outcomes, **details):
            heartbeat(outcomes, force=True)
            checkpoint("campaign_" + status, plans=outcomes, **details)
            repair = repair_bridge_status(campaign)
            return {"status": status, "campaign_digest": m["campaign_digest"],
                    "deadline_epoch": state["deadline_epoch"], "plans": outcomes,
                    "gpu_dispatch_enabled": allow_gpu_after_user_resume,
                    "scientific_result_verified": False,
                    "online_repair_agent": repair["status"],
                    "repair_bridge": repair, **details}
        waiting_for = None
        last_heartbeat = [-float("inf")]
        def heartbeat(outcomes, *, force=False):
            if not force and time.monotonic() - last_heartbeat[0] < heartbeat_seconds:
                return
            last_heartbeat[0] = time.monotonic()
            gpu_ids = {uuid for name in campaign.entries if campaign.gpu(name)
                       for uuid in campaign.plans[name]["gpus"]["uuids"]}
            snapshots = campaign.H.gpu_snapshot() if gpu_ids else []
            gpu = [item for item in snapshots if item["uuid"] in gpu_ids]
            running_gpu = any(campaign.gpu(name) and value == "running"
                              for name, value in outcomes.items())
            # Memory use is not utilization. Idle here means no observed compute
            # PID on every expected physical device, never permission to launch.
            all_devices = {item["uuid"] for item in gpu} == gpu_ids
            idle = bool(gpu_ids and all_devices and all(not item["foreign_pids"] for item in gpu))
            repair = repair_bridge_status(campaign)
            _atomic(directory / "heartbeat.json", {
                "kind": "research-supervisor-heartbeat-v2",
                "campaign_digest": m["campaign_digest"], "observed_epoch": now(),
                "owner": campaign.H._self_identity(), "deadline_epoch": state["deadline_epoch"],
                "remaining_seconds": max(0, remaining()), "plans": outcomes,
                "gpu_observation": gpu, "gpu_observation_available": bool(gpu_ids and all_devices),
                "gpu_no_compute_process_observed": idle,
                "gpu_idle_while_retained_running": idle and running_gpu,
                "gpu_dispatch_enabled": allow_gpu_after_user_resume,
                "online_repair_agent": repair["status"],
                "repair_bridge_status": repair["status"],
                "scientific_result_verified": False})
            checkpoint("heartbeat", plans=outcomes)
        while True:
            campaign.recheck()
            observed = {name: campaign.observe(name) for name in campaign.entries}
            outcomes = {name: value["status"] for name, value in observed.items()}
            outcomes, driver_live = reconcile_dispatch(campaign, state, observed)
            heartbeat(outcomes)
            if any(value == "unknown" for value in outcomes.values()):
                return report("reconciliation_required", outcomes, observations=observed)
            eligible = []
            unresolved = set(campaign.entries)
            while unresolved:
                progressed = False
                for name in list(unresolved):
                    entry = campaign.entries[name]
                    prerequisites = entry["dependencies"] + ([entry["on_failure_of"]] if entry["on_failure_of"] is not None else [])
                    if any(dep in unresolved for dep in prerequisites):
                        continue
                    unresolved.remove(name)
                    progressed = True
                    if outcomes[name] in ("completed", "failed", "running"):
                        continue
                    if any(outcomes[dep] in ("failed", "blocked", "skipped", "blocked_gpu_stop") for dep in entry["dependencies"]):
                        outcomes[name] = "blocked"
                    elif any(outcomes[dep] != "completed" for dep in entry["dependencies"]):
                        pass
                    elif entry["on_failure_of"] is not None and outcomes[entry["on_failure_of"]] == "completed":
                        outcomes[name] = "skipped"
                    elif entry["on_failure_of"] is not None and outcomes[entry["on_failure_of"]] != "failed":
                        if outcomes[entry["on_failure_of"]] in ("blocked", "skipped", "blocked_gpu_stop"):
                            outcomes[name] = "blocked"
                    elif entry["on_failure_of"] is not None and not observed[entry["on_failure_of"]]["repairable"]:
                        outcomes[name] = "blocked"
                    elif outcomes[name] == "waiting_inputs":
                        pass
                    elif campaign.gpu(name) and not allow_gpu_after_user_resume:
                        outcomes[name] = "blocked_gpu_stop"
                    else:
                        eligible.append(name)
                if not progressed:
                    raise CampaignError("unresolvable campaign graph")
            active = [name for name, status in outcomes.items() if status == "running"]
            if any(campaign.gpu(name) for name in active) and not allow_gpu_after_user_resume:
                return report("blocked_gpu_stop", outcomes, reason="GPU STOP: observation only, no driver resume")
            if len(active) > 1:
                return report("reconciliation_required", outcomes, reason="multiple retained batches in one pool")
            target = active[0] if active else next((name for name in campaign.entries if name in eligible), None)
            if target is None and "waiting_inputs" in outcomes.values():
                if stopped():
                    return report("paused", outcomes)
                if remaining() <= m["collection_reserve_seconds"]:
                    return report("budget_exhausted", outcomes)
                if watch_ready:
                    time.sleep(poll_seconds)
                    continue
                return report("waiting_inputs", outcomes, observations=observed)
            if target is None:
                status = "blocked_gpu_stop" if "blocked_gpu_stop" in outcomes.values() else "failed" if any(s in ("failed", "blocked") for s in outcomes.values()) else "completed"
                return report(status, outcomes)
            if stopped():
                return report("paused", outcomes, reason="STOP or signal; retained workers keep original deadlines")
            if remaining() <= m["collection_reserve_seconds"]:
                return report("budget_exhausted", outcomes)
            if driver_live or _driver_busy(campaign.pool):
                if waiting_for != target:
                    checkpoint("host_driver_wait", plan_id=target)
                    waiting_for = target
                time.sleep(poll_seconds)
                continue
            if waiting_for is not None:
                checkpoint("host_driver_released", plan_id=target)
                waiting_for = None
            if not active and campaign.plans[target]["limits"]["total_wall_seconds"] > remaining() - m["collection_reserve_seconds"]:
                return report("budget_exhausted", outcomes)
            if state["starts"][target] >= 32:
                return report("reconciliation_required", outcomes, reason="bounded driver recovery allowance exhausted")
            campaign.recheck()
            if stopped():
                return report("paused", outcomes)
            if remaining() <= m["collection_reserve_seconds"]:
                return report("budget_exhausted", outcomes)
            if campaign.missing[target]:
                if active:
                    return report("reconciliation_required", outcomes,
                                  reason="retained active plan lost pinned inputs; no driver resume")
                # Readiness can disappear during final validation; no process
                # creation was attempted, so keep this item eligible to wait.
                continue
            entry = campaign.entries[target]
            argv = harness_argv(campaign, target)
            logs = directory / target
            logs.mkdir(exist_ok=True)
            # A private empty cache prefix prevents stale .pyc from bypassing source pins.
            with tempfile.TemporaryDirectory(prefix="harness-cache-", dir=directory) as cache:
                environment = {**os.environ, "PYTHONPYCACHEPREFIX": cache, "PYTHONDONTWRITEBYTECODE": "1"}
                # Revalidation and filesystem preparation consume the original
                # campaign budget. Check again at the process-creation boundary.
                if stopped():
                    return report("paused", outcomes)
                workload_remaining = remaining() - m["collection_reserve_seconds"]
                if workload_remaining <= 0 or (
                        not active and campaign.plans[target]["limits"]["total_wall_seconds"] > workload_remaining):
                    return report("budget_exhausted", outcomes,
                                  reason="revalidation consumed the remaining dispatch budget")
                # Serialize CLI STOP acceptance with the process-creation boundary.
                # Do not hold this lock while waiting for the child.
                with control_lock(directory):
                    if stopped():
                        return report("paused", outcomes)
                    previous_active = state["active"]
                    state["active"] = target
                    state["starts"][target] += 1
                    checkpoint("dispatch_intent", plan_id=target,
                               plan_digest=campaign.entries[target]["plan_digest"])
                    # fsync itself consumes time; a *known* pre-Popen cancellation
                    # is not an uncertain launch and must not poison future resume.
                    # A crash instead leaves the durable intent unreconciled.
                    workload_remaining = remaining() - m["collection_reserve_seconds"]
                    cancelled = "paused" if stopped() else "budget_exhausted" if (
                        workload_remaining <= 0 or not active and
                        campaign.plans[target]["limits"]["total_wall_seconds"] > workload_remaining) else None
                    if cancelled:
                        state["starts"][target] -= 1
                        state["active"] = previous_active
                        checkpoint("dispatch_cancelled_before_process_creation", plan_id=target,
                                   reason=cancelled)
                        return report(cancelled, outcomes)
                    child = (transport or HarnessTransport()).start(argv, logs, environment)
                identity = campaign.H._identity(child.pid)
                if identity is not None:
                    _atomic(logs / "driver.json", {"campaign_digest": m["campaign_digest"],
                        "plan_digest": entry["plan_digest"], "process": identity, "argv": argv})
                checkpoint("driver_started", plan_id=target, process=identity, argv=argv)
                while child.poll() is None:
                    live_outcomes = dict(outcomes)
                    live_outcomes[target] = "running"
                    heartbeat(live_outcomes)
                    if stopped() or remaining() <= m["collection_reserve_seconds"]:
                        child.send_signal(signal.SIGINT)
                        try:
                            child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            pass  # Never kill a driver/worker to manufacture a terminal outcome.
                        retained = campaign.observe(target)
                        outcomes[target] = retained["status"]
                        return report("paused" if stopped() else "budget_exhausted", outcomes,
                                      reason="requested canonical driver handoff; retained identity unchanged",
                                      retained_observation=retained)
                    time.sleep(poll_seconds)
                code = child.returncode
                child = None
            after = campaign.observe(target)
            checkpoint("driver_returned", plan_id=target, exit_code=code, observation=after)
            if after["status"] not in ("completed", "failed"):
                outcomes[target] = after["status"]
                return report("reconciliation_required", outcomes, reason="driver return is not terminal receipt proof")
            state["active"] = None
            checkpoint("plan_settled", plan_id=target, observation=after)
    finally:
        if child is not None and child.poll() is None:
            # Same documented handoff on validation/I/O exceptions; never terminate workers.
            child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        for signum, handler in old_signals.items():
            signal.signal(signum, handler)
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--status", action="store_true", help="Read-only exact retained process/receipt inspection")
    mode.add_argument("--stop", action="store_true", help="Request cooperative driver handoff; never kill workers")
    mode.add_argument("--resume", action="store_true", help="Clear campaign STOP with original state/budget; GPU STOP remains")
    mode.add_argument("--extend", type=Path,
                      help="Activate an exact reviewed append-only manifest revision")
    parser.add_argument("--approved-campaign-digest")
    parser.add_argument("--approved-next-campaign-digest")
    parser.add_argument("--print-digest", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--watch-ready", action="store_true",
                        help="Wait for declared exact input bytes within the original campaign deadline")
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--allow-gpu-after-user-resume", action="store_true",
                        help="Local acknowledgment ONLY after explicit user GPU resume and gate acceptance")
    args = parser.parse_args(argv)
    try:
        if args.print_digest:
            if args.execute or args.stop or args.resume or args.extend:
                raise CampaignError("digest printing cannot mutate")
            result = {"campaign_digest": campaign_digest(read_json(args.campaign)), "execution_started": False}
        elif args.stop:
            result = request_stop(args.campaign, args.approved_campaign_digest)
        elif args.extend:
            result = extend_campaign(
                args.campaign, args.extend,
                approved_current_digest=args.approved_campaign_digest,
                approved_next_digest=args.approved_next_campaign_digest)
        else:
            result = run_campaign(args.campaign, execute=args.execute or args.resume,
                                  approved_digest=args.approved_campaign_digest, poll_seconds=args.poll_seconds,
                                  allow_gpu_after_user_resume=args.allow_gpu_after_user_resume,
                                  watch_ready=args.watch_ready, heartbeat_seconds=args.heartbeat_seconds,
                                  resume=args.resume)
        print(canonical(result))
        return 0 if result.get("status") in (
            None, "inspection", "completed", "stop_requested", "extended", "already_extended") else 2
    except Exception as error:
        print(canonical({"status": "blocked", "error": type(error).__name__, "reason": str(error),
                         "gpu_dispatch_enabled": args.allow_gpu_after_user_resume,
                         "execution_outcome": "not_established"}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
