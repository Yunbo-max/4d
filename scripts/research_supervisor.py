#!/usr/bin/env python3
"""Finite Local campaign controller for the installed Research Autopilot harness.

Generated/unexecuted source. GPU dispatch is stopped by default. This
controller never executes an experiment command, repairs source, resets an
attempt, grants a scientific gate, or changes the native runner's retry limits.
"""
from __future__ import annotations

import argparse
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
        if m["kind"] != "research-harness-campaign" or type(m["version"]) is not int or m["version"] != 1:
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
        self.entries, self.plans, self.native = {}, {}, {}
        self.directory = safe_path(self.root, "runs/supervisor/" + m["campaign_id"])
        native_ids, batch_ids, trial_ids, repairs = set(), set(), set(), set()
        reserved = 0
        for entry in m["plans"]:
            _fields(entry, "id plan_ref plan_digest dependencies on_failure_of")
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
            try:
                self.H.validate_plan(self.root, plan)
            except Exception as error:
                raise CampaignError("canonical harness validation denied") from error
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
                identity = safe_path(self.root, native["output_root"] + "/" + native["run_id"])
                if identity in native_ids or identity == self.directory or identity in self.directory.parents or self.directory in identity.parents:
                    raise CampaignError("repeated or overlapping native identity")
                native_ids.add(identity)
                for job in native["jobs"]:
                    if job["trial_id"] in trial_ids:
                        raise CampaignError("repeated trial identity; child plans cannot retry existing experiments")
                    trial_ids.add(job["trial_id"])
                natives[task["task_id"]] = native
            self.entries[name], self.plans[name], self.native[name] = entry, plan, natives
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
            self.H.validate_plan(self.root, self.plans[name])

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
    path = pool / "driver.lock"
    if not path.exists():
        return False
    with path.open("rb") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(stream, fcntl.LOCK_UN)
        return False


def run_campaign(path, *, execute=False, approved_digest=None, poll_seconds=1.0,
                 now=time.time, transport=None, allow_gpu_after_user_resume=False):
    # This flag acknowledges a separate explicit human resume; it cannot prove or
    # manufacture that permission. It is deliberately absent from the manifest.
    if type(allow_gpu_after_user_resume) is not bool:
        raise CampaignError("explicit Local resume acknowledgment must be boolean")
    _number(poll_seconds, .05, 30)
    campaign = Campaign(path)
    m, directory = campaign.data, campaign.directory
    observations = {name: campaign.observe(name) for name in campaign.entries}
    if not execute:
        return {"status": "inspection", "campaign_digest": m["campaign_digest"],
                "gpu_dispatch_enabled": allow_gpu_after_user_resume, "plans": observations}
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
            _fields(state, "kind campaign_digest started_epoch deadline_epoch last_epoch active starts")
            for key in ("started_epoch", "deadline_epoch", "last_epoch"):
                _number(state[key], 0, 1e20)
            if (state["kind"] != "research-supervisor-state-v1" or state["campaign_digest"] != m["campaign_digest"] or
                    state["deadline_epoch"] != state["started_epoch"] + m["total_wall_seconds"] or
                    state["last_epoch"] < state["started_epoch"] or now() < state["last_epoch"] or
                    state["active"] is not None and state["active"] not in campaign.entries or
                    not isinstance(state["starts"], dict) or set(state["starts"]) != set(campaign.entries) or
                    any(type(value) is not int or not 0 <= value <= 32 for value in state["starts"].values())):
                raise CampaignError("invalid retained deadline or identity; reconciliation required")
        else:
            if (directory / "journal.jsonl").exists():
                raise CampaignError("journal without state; no budget reset")
            start = now()
            state = {"kind": "research-supervisor-state-v1", "campaign_digest": m["campaign_digest"],
                     "started_epoch": start, "deadline_epoch": start + m["total_wall_seconds"],
                     "last_epoch": start, "active": None, "starts": {name: 0 for name in campaign.entries}}
            _atomic(manifest_path, m)
            checkpoint("campaign_started")
        for signum in (signal.SIGINT, signal.SIGTERM):
            old_signals[signum] = signal.signal(signum, stop)
        monotonic_deadline = time.monotonic() + max(0, state["deadline_epoch"] - now())
        def remaining():
            return min(state["deadline_epoch"] - now(), monotonic_deadline - time.monotonic())
        def stopped():
            return stop_requested[0] or (directory / "STOP").exists()
        def report(status, outcomes, **details):
            checkpoint("campaign_" + status, plans=outcomes, **details)
            return {"status": status, "campaign_digest": m["campaign_digest"],
                    "deadline_epoch": state["deadline_epoch"], "plans": outcomes,
                    "gpu_dispatch_enabled": allow_gpu_after_user_resume,
                    "scientific_result_verified": False, **details}
        waiting_for = None
        while True:
            campaign.recheck()
            observed = {name: campaign.observe(name) for name in campaign.entries}
            outcomes = {name: value["status"] for name, value in observed.items()}
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
            if target is None:
                status = "blocked_gpu_stop" if "blocked_gpu_stop" in outcomes.values() else "failed" if any(s in ("failed", "blocked") for s in outcomes.values()) else "completed"
                return report(status, outcomes)
            if stopped():
                return report("paused", outcomes, reason="STOP or signal; retained workers keep original deadlines")
            if remaining() <= m["collection_reserve_seconds"]:
                return report("budget_exhausted", outcomes)
            if _driver_busy(campaign.pool):
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
            state["active"] = target
            state["starts"][target] += 1
            checkpoint("dispatch_intent", plan_id=target, plan_digest=campaign.entries[target]["plan_digest"])
            campaign.recheck()
            if stopped():
                return report("paused", outcomes)
            entry = campaign.entries[target]
            argv = [m["python"], "-B", str(campaign.skill / "scripts/run_harness.py"),
                    str(verify_ref(campaign.root, entry["plan_ref"])), "--root", str(campaign.root),
                    "--execute", "--approved-plan-digest", entry["plan_digest"]]
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
                child = (transport or HarnessTransport()).start(argv, logs, environment)
                checkpoint("driver_started", plan_id=target, pid=child.pid, argv=argv)
                while child.poll() is None:
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
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approved-campaign-digest")
    parser.add_argument("--print-digest", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--allow-gpu-after-user-resume", action="store_true",
                        help="Local acknowledgment ONLY after explicit user GPU resume and gate acceptance")
    args = parser.parse_args(argv)
    try:
        if args.print_digest:
            if args.execute:
                raise CampaignError("digest printing cannot execute")
            result = {"campaign_digest": campaign_digest(read_json(args.campaign)), "execution_started": False}
        else:
            result = run_campaign(args.campaign, execute=args.execute,
                                  approved_digest=args.approved_campaign_digest, poll_seconds=args.poll_seconds,
                                  allow_gpu_after_user_resume=args.allow_gpu_after_user_resume)
        print(canonical(result))
        return 0 if result.get("status") in (None, "inspection", "completed") else 2
    except Exception as error:
        print(canonical({"status": "blocked", "error": type(error).__name__, "reason": str(error),
                         "gpu_dispatch_enabled": args.allow_gpu_after_user_resume,
                         "execution_outcome": "not_established"}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
