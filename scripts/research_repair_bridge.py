#!/usr/bin/env python3
"""Bind one reviewed terminal code failure to the existing runtime controller.

Generated/unexecuted Web source.  This bridge neither applies a patch nor
relaunches a scientific attempt.  It enqueues one immutable ``candidate_code``
task in an already installed/registered ``research-autopilot`` controller and
runs at most one configured ACP worker turn.  Local must review/version any
outputs and append a distinct plan before a later experiment can be considered.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import tempfile
import time


class RepairBridgeError(ValueError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise RepairBridgeError("duplicate JSON key: " + key)
        value[key] = item
    return value


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"),
                          object_pairs_hook=_pairs,
                          parse_constant=lambda item: (_ for _ in ()).throw(
                              RepairBridgeError("nonfinite JSON: " + item)))
    except (OSError, ValueError) as error:
        raise RepairBridgeError("unreadable JSON: " + str(path)) from error


def read_retained_json(path):
    path = Path(path)
    try:
        record = os.lstat(path)
    except OSError as error:
        raise RepairBridgeError("unreadable retained JSON: " + str(path)) from error
    if not stat.S_ISREG(record.st_mode) or path.is_symlink():
        raise RepairBridgeError("regular retained JSON required: " + str(path))
    return read_json(path)


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def request_digest(value):
    body = {key: item for key, item in value.items() if key != "request_digest"}
    return hashlib.sha256(canonical(body).encode()).hexdigest()


def campaign_manifest_digest(value):
    body = {key: item for key, item in value.items() if key != "campaign_digest"}
    return hashlib.sha256(canonical(body).encode()).hexdigest()


def _digest(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise RepairBridgeError("invalid SHA-256")
    return value


def _fields(value, names):
    if not isinstance(value, dict) or set(value) != set(names.split()):
        raise RepairBridgeError("unexpected contract fields")


def _identifier(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", value):
        raise RepairBridgeError("invalid " + label)
    return value


def _integer(value, low, high, label):
    if type(value) is not int or not low <= value <= high:
        raise RepairBridgeError("invalid " + label)
    return value


def _absolute_file(value, label, *, executable=False):
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise RepairBridgeError(label + " must be an absolute path")
    path = Path(value)
    if path != path.resolve(strict=True) or path.is_symlink() or not path.is_file():
        raise RepairBridgeError(label + " must be a canonical regular file")
    mode = path.stat().st_mode
    if not stat.S_ISREG(mode) or executable and not mode & stat.S_IXUSR:
        raise RepairBridgeError(label + " is not executable")
    return path


def _root(value):
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise RepairBridgeError("root must be absolute")
    path = Path(value)
    if path != path.resolve(strict=True) or not path.is_dir():
        raise RepairBridgeError("root must be a canonical directory")
    return path


def safe_path(root, relative):
    if (not isinstance(relative, str) or not relative or Path(relative).is_absolute() or
            any(part in ("", ".", "..") for part in relative.split("/"))):
        raise RepairBridgeError("unsafe relative path")
    path = root
    for part in relative.split("/"):
        path = path / part
        if path.is_symlink():
            raise RepairBridgeError("symlink path rejected")
    path.resolve().relative_to(root)
    return path


def verify_ref(root, ref):
    _fields(ref, "path sha256")
    path = safe_path(root, ref["path"])
    if not path.is_file() or path.is_symlink() or file_digest(path) != _digest(ref["sha256"]):
        raise RepairBridgeError("pinned bytes changed: " + ref["path"])
    return path


def _atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(canonical(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _record_run(directory, value):
    """Preserve every external-call/reconciliation record append-only."""
    events = directory / "events"
    if events.is_symlink():
        raise RepairBridgeError("repair event directory symlink rejected")
    events.mkdir(parents=True, exist_ok=True)
    journal = directory / "journal.jsonl"
    if journal.is_symlink():
        raise RepairBridgeError("repair journal symlink rejected")
    retained = []
    if journal.exists():
        if not journal.is_file():
            raise RepairBridgeError("repair journal must be a regular file")
        try:
            retained = [json.loads(line, object_pairs_hook=_pairs)
                        for line in journal.read_text(encoding="utf-8").splitlines()]
        except (OSError, UnicodeError, ValueError, TypeError) as error:
            raise RepairBridgeError("invalid retained repair journal") from error
    previous = None
    for sequence, event in enumerate(retained, 1):
        digest = hashlib.sha256(canonical(event).encode()).hexdigest()
        expected = events / f"{sequence:020d}-{digest}.json"
        if (event.get("sequence") != sequence or
                event.get("previous_event_sha256") != previous or
                not expected.is_file() or expected.is_symlink() or
                read_retained_json(expected) != event):
            raise RepairBridgeError("retained repair event chain mismatch")
        previous = digest
    inventory = list(events.iterdir())
    if len(inventory) != len(retained):
        raise RepairBridgeError("unindexed retained repair event")
    sequence = len(retained) + 1
    value = dict(value, sequence=sequence, previous_event_sha256=previous)
    encoded = canonical(value).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    name = f"{sequence:020d}-{digest}.json"
    target = events / name
    if target.exists():
        raise RepairBridgeError("repair event identity collision")
    _atomic(target, value)
    with journal.open("ab") as stream:
        stream.write(encoded + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    _atomic(directory / "last-status.json",
            {"event": str(target.name), "status": value["status"],
             "request_digest": value["request_digest"]})


def _retain_exact(path, value, label):
    if path.exists():
        if read_retained_json(path) != value:
            raise RepairBridgeError("retained " + label + " identity mismatch")
    else:
        _atomic(path, value)


def _validate_campaign_contract(root, campaign):
    """Validate the literal supervisor manifest/entry envelope without running it."""
    _fields(campaign, "kind version campaign_id root python skill_root skill_digest pool_dir "
             "total_wall_seconds collection_reserve_seconds plans campaign_digest")
    if (campaign["kind"] != "research-harness-campaign" or
            type(campaign["version"]) is not int or campaign["version"] not in (1, 2) or
            campaign["root"] != str(root)):
        raise RepairBridgeError("failed campaign contract mismatch")
    _identifier(campaign["campaign_id"], "campaign id")
    for name in ("python", "skill_root", "pool_dir"):
        if not isinstance(campaign[name], str) or not Path(campaign[name]).is_absolute():
            raise RepairBridgeError("campaign " + name + " must be absolute")
    _digest(campaign["skill_digest"])
    _integer(campaign["total_wall_seconds"], 1801, 28800, "campaign wall time")
    _integer(campaign["collection_reserve_seconds"], 1800,
             campaign["total_wall_seconds"] - 1, "campaign collection reserve")
    if not isinstance(campaign["plans"], list) or not 1 <= len(campaign["plans"]) <= 256:
        raise RepairBridgeError("finite campaign plan inventory required")
    entries = {}
    for entry in campaign["plans"]:
        names = "id plan_ref plan_digest dependencies on_failure_of"
        if campaign["version"] == 2:
            names += " required_inputs"
        _fields(entry, names)
        identifier = _identifier(entry["id"], "campaign plan id")
        if identifier in entries:
            raise RepairBridgeError("repeated campaign plan id")
        _digest(entry["plan_digest"])
        if (not isinstance(entry["dependencies"], list) or
                len(entry["dependencies"]) != len(set(entry["dependencies"])) or
                any(not isinstance(item, str) for item in entry["dependencies"])):
            raise RepairBridgeError("invalid campaign dependencies")
        if entry["on_failure_of"] is not None:
            _identifier(entry["on_failure_of"], "campaign failure trigger")
        required = entry.get("required_inputs", [])
        if not isinstance(required, list):
            raise RepairBridgeError("invalid campaign required inputs")
        for ref in required:
            _fields(ref, "path sha256")
            safe_path(root, ref["path"])
            _digest(ref["sha256"])
        plan = read_json(verify_ref(root, entry["plan_ref"]))
        if plan.get("plan_digest") != entry["plan_digest"]:
            raise RepairBridgeError("campaign harness plan digest mismatch")
        entries[identifier] = entry
    return entries


class Request:
    def __init__(self, path):
        raw = Path(path)
        if raw.is_symlink() or not raw.is_file():
            raise RepairBridgeError("request must be a regular file")
        self.path = raw.resolve(strict=True)
        self.data = read_json(self.path)
        m = self.data
        _fields(m, "kind version request_id root runtime_executable runtime_sha256 db db_device db_inode "
                   "project_id dedicated_project owner adapter adapters_config adapters_sha256 "
                   "project_ref admission_ref inputs instructions output_paths "
                   "max_cost max_seconds capabilities request_digest")
        if m["kind"] != "research-runtime-repair-request" or m["version"] != 1:
            raise RepairBridgeError("unsupported repair request")
        if request_digest(m) != _digest(m["request_digest"]):
            raise RepairBridgeError("request digest mismatch")
        self.root = _root(m["root"])
        self.runtime = _absolute_file(m["runtime_executable"], "runtime executable", executable=True)
        self.db = _absolute_file(m["db"], "runtime database")
        self.adapters = _absolute_file(m["adapters_config"], "adapter registry")
        if file_digest(self.runtime) != _digest(m["runtime_sha256"]):
            raise RepairBridgeError("runtime executable bytes changed")
        if file_digest(self.adapters) != _digest(m["adapters_sha256"]):
            raise RepairBridgeError("adapter registry bytes changed")
        if (type(m["db_device"]) is not int or type(m["db_inode"]) is not int or
                (self.db.stat().st_dev, self.db.stat().st_ino) !=
                (m["db_device"], m["db_inode"])):
            raise RepairBridgeError("runtime controller identity changed")
        _identifier(m["request_id"], "request id")
        _identifier(m["project_id"], "project id")
        if m["dedicated_project"] is not True or m["project_id"] != m["request_id"]:
            raise RepairBridgeError("request-scoped dedicated runtime project required")
        _identifier(m["owner"], "owner")
        _identifier(m["adapter"], "adapter")
        if m["adapter"] == "native":
            raise RepairBridgeError("repair requires a configured ACP adapter, not native")
        _integer(m["max_cost"], 1, 1000000000, "cost bound")
        _integer(m["max_seconds"], 1, 86400, "time bound")
        if m["capabilities"] != ["filesystem_read", "filesystem_write"]:
            raise RepairBridgeError("repair capability inventory is fixed")
        self.project_path = verify_ref(self.root, m["project_ref"])
        project = read_json(self.project_path)
        self.project = project
        if (project.get("project_id") != m["project_id"] or
                project.get("root") != str(self.root)):
            raise RepairBridgeError("dedicated runtime project identity mismatch")
        role = project.get("assigned_role", project.get("role"))
        adapters = project.get("adapter_allowlist",
                               project.get("allowed_adapters", project.get("adapters")))
        if (project.get("workflow_class") != "scientific_method" or
                role != "web_supervisor" or not isinstance(adapters, list) or
                m["adapter"] not in adapters or project.get("max_workers") != 1):
            raise RepairBridgeError("dedicated scientific repair project required")
        if (not isinstance(project.get("skill_root"), str) or
                not isinstance(project.get("skill_digest"), str) or
                not re.fullmatch(r"[0-9a-f]{64}", project["skill_digest"])):
            raise RepairBridgeError("project must pin the complete research skill")
        self.admission = read_json(verify_ref(self.root, m["admission_ref"]))
        _fields(self.admission, "kind version campaign_id campaign_digest failed_plan_id "
                "status classification scientific_retry_allowed gpu_allowed reviewer "
                "reviewed_at repair_scope failure_ref failed_plan_ref supervisor_status_ref "
                "budget_ref campaign_ref")
        a = self.admission
        if (a["kind"] != "research-repair-admission" or a["version"] != 1 or
                a["status"] != "terminal_failed" or a["classification"] != "code_error"):
            raise RepairBridgeError("reviewed terminal code failure required")
        _identifier(a["campaign_id"], "campaign id")
        _identifier(a["failed_plan_id"], "failed plan id")
        _digest(a["campaign_digest"])
        if a["gpu_allowed"] is not False:
            raise RepairBridgeError("repair bridge cannot authorize GPU work")
        if a["scientific_retry_allowed"] is not False:
            raise RepairBridgeError("repair bridge cannot retry a scientific attempt")
        if any(not isinstance(a[key], str) or not a[key] for key in ("reviewer", "reviewed_at", "repair_scope")):
            raise RepairBridgeError("incomplete repair admission")
        for name in ("inputs", "instructions"):
            refs = m[name]
            if not isinstance(refs, list) or not refs:
                raise RepairBridgeError(name + " must be nonempty")
            keys = []
            for ref in refs:
                verify_ref(self.root, ref)
                keys.append((ref["path"], ref["sha256"]))
            if len(keys) != len(set(keys)):
                raise RepairBridgeError("duplicate " + name)
        failure_path = verify_ref(self.root, a["failure_ref"])
        verify_ref(self.root, a["failed_plan_ref"])
        input_keys = {(ref["path"], ref["sha256"]) for ref in m["inputs"]}
        for required in (a["failure_ref"], a["failed_plan_ref"],
                         a["supervisor_status_ref"], a["budget_ref"], a["campaign_ref"]):
            if (required["path"], required["sha256"]) not in input_keys:
                raise RepairBridgeError("admission evidence must be a task input")
        failure = read_json(failure_path)
        if failure.get("status") != "failed":
            raise RepairBridgeError("actual terminal failed receipt required")
        campaign = read_json(verify_ref(self.root, a["campaign_ref"]))
        if (campaign.get("kind") != "research-harness-campaign" or
                campaign.get("campaign_id") != a["campaign_id"] or
                campaign.get("campaign_digest") != a["campaign_digest"] or
                campaign_manifest_digest(campaign) != a["campaign_digest"]):
            raise RepairBridgeError("failed campaign identity mismatch")
        campaign_entries = _validate_campaign_contract(self.root, campaign)
        selected = campaign_entries.get(a["failed_plan_id"])
        if selected is None or selected["plan_ref"] != a["failed_plan_ref"]:
            raise RepairBridgeError("failed plan reference is not campaign-bound")
        if failure.get("plan_digest") != selected["plan_digest"]:
            raise RepairBridgeError("failed receipt is not bound to the campaign plan")
        supervisor_status = read_json(verify_ref(self.root, a["supervisor_status_ref"]))
        observed = supervisor_status.get("plans", {}).get(a["failed_plan_id"])
        if (supervisor_status.get("status") != "inspection" or
                supervisor_status.get("campaign_digest") != a["campaign_digest"] or
                not isinstance(observed, dict) or observed.get("status") != "failed" or
                observed.get("repairable") is not True):
            raise RepairBridgeError("repairable failed supervisor observation required")
        self.budget = read_json(verify_ref(self.root, a["budget_ref"]))
        _fields(self.budget, "kind version campaign_id campaign_digest max_cost max_seconds")
        if (self.budget["kind"] != "research-repair-campaign-budget" or
                self.budget["version"] != 1 or
                self.budget["campaign_id"] != a["campaign_id"] or
                self.budget["campaign_digest"] != a["campaign_digest"]):
            raise RepairBridgeError("campaign repair budget identity mismatch")
        _integer(self.budget["max_cost"], 0, 1000000000, "campaign repair cost")
        _integer(self.budget["max_seconds"], 0, 86400, "campaign repair time")
        if m["max_cost"] > self.budget["max_cost"] or m["max_seconds"] > self.budget["max_seconds"]:
            raise RepairBridgeError("campaign repair budget exceeded")
        if (project.get("max_cost") != m["max_cost"] or
                project.get("max_seconds") != m["max_seconds"]):
            raise RepairBridgeError("dedicated project limits must equal the request reservation")
        if (not isinstance(m["output_paths"], list) or not m["output_paths"] or
                len(m["output_paths"]) != len(set(m["output_paths"]))):
            raise RepairBridgeError("invalid output inventory")
        output_prefix = "repairs/" + a["campaign_id"] + "/" + m["request_id"] + "/"
        for relative in m["output_paths"]:
            target = safe_path(self.root, relative)
            if not relative.startswith(output_prefix):
                raise RepairBridgeError("repair output must use its dedicated namespace")
            if target.exists() and (target.is_symlink() or not target.is_file()):
                raise RepairBridgeError("retained repair output must be a regular file")
        self.directory = safe_path(
            self.root, "runs/supervisor/" + a["campaign_id"] + "/repair-requests/" + m["request_id"])
        self.task = {
            "task_id": m["request_id"], "action": "candidate_code",
            "inputs": list(m["inputs"]),
            "instructions": [m["admission_ref"], *m["instructions"]],
            "dependencies": [], "output_paths": list(m["output_paths"]),
            "completion": {"kind": "outputs_verified"},
            "max_cost": m["max_cost"], "max_seconds": m["max_seconds"],
            "capabilities": list(m["capabilities"]), "adapter": m["adapter"],
            "parameters": {
                "campaign_digest": a["campaign_digest"],
                "failed_plan_id": a["failed_plan_id"],
                "gpu_allowed": False, "scientific_retry_allowed": False,
                "requires_reviewed_child_version": True}}


def _invoke(argv, *, cwd, environment, timeout):
    started = time.time()
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        process, returncode, transport_error = None, None, None
        termination = {"process_group": None, "term_sent": False,
                       "kill_sent": False, "direct_child_reaped": False}
        try:
            process = subprocess.Popen(argv, cwd=cwd, env=environment, shell=False,
                                       stdout=stdout, stderr=stderr, start_new_session=True)
            termination["process_group"] = process.pid
            try:
                returncode = process.wait(timeout=timeout)
                termination["direct_child_reaped"] = True
            except subprocess.TimeoutExpired as error:
                transport_error = type(error).__name__ + ": " + str(error)
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    termination["term_sent"] = True
                except ProcessLookupError:
                    pass
                try:
                    returncode = process.wait(timeout=1)
                    termination["direct_child_reaped"] = True
                except subprocess.TimeoutExpired:
                    pass
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                    termination["kill_sent"] = True
                except ProcessLookupError:
                    pass
                if process.poll() is None:
                    try:
                        returncode = process.wait(timeout=1)
                        termination["direct_child_reaped"] = True
                    except subprocess.TimeoutExpired:
                        pass
        except OSError as error:
            transport_error = type(error).__name__ + ": " + str(error)
        def capture(stream):
            stream.seek(0)
            digest = hashlib.sha256()
            kept, size = bytearray(), 0
            while True:
                block = stream.read(65536)
                if not block:
                    break
                digest.update(block)
                size += len(block)
                if len(kept) < 1048576:
                    kept.extend(block[:1048576 - len(kept)])
            return {"text": bytes(kept).decode("utf-8", "replace"),
                    "sha256": digest.hexdigest(), "bytes": size,
                    "truncated": size > len(kept)}
        out, err = capture(stdout), capture(stderr)
    return {"argv": argv, "returncode": None if transport_error else returncode,
            "transport_error": transport_error,
            "termination": termination,
            "started_epoch": started, "finished_epoch": time.time(),
            "stdout": out["text"], "stdout_sha256": out["sha256"],
            "stdout_bytes": out["bytes"], "stdout_truncated": out["truncated"],
            "stderr": err["text"], "stderr_sha256": err["sha256"],
            "stderr_bytes": err["bytes"], "stderr_truncated": err["truncated"]}


def _ref_from_absolute(root, path):
    path = Path(path)
    if path != path.resolve(strict=True) or path.is_symlink() or not path.is_file():
        raise RepairBridgeError("request input must be a canonical regular file")
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as error:
        raise RepairBridgeError("request input must be inside project root") from error
    safe_path(root, relative)
    return {"path": relative, "sha256": file_digest(path)}


def build_request(*, root, runtime_executable, db, project_id, owner, adapter,
                  adapters_config, project, admission, inputs, instructions, output_paths,
                  max_cost, max_seconds, request_id, out):
    """Build immutable refs; validation remains identical to execution."""
    root = _root(str(root))
    out = Path(out)
    if not out.is_absolute():
        raise RepairBridgeError("output request path must be absolute")
    try:
        out.resolve().relative_to(root)
    except ValueError as error:
        raise RepairBridgeError("output request must be inside project root") from error
    if out.exists() or out.is_symlink():
        raise RepairBridgeError("output request path must be new")
    for relative in output_paths:
        target = safe_path(root, relative)
        if target.exists():
            raise RepairBridgeError("initial repair output path must be new")
    value = {
        "kind": "research-runtime-repair-request", "version": 1,
        "request_id": request_id, "root": str(root),
        "runtime_executable": str(Path(runtime_executable)),
        "runtime_sha256": file_digest(runtime_executable), "db": str(Path(db)),
        "db_device": Path(db).stat().st_dev, "db_inode": Path(db).stat().st_ino,
        "project_id": project_id, "dedicated_project": True,
        "owner": owner, "adapter": adapter,
        "adapters_config": str(Path(adapters_config)),
        "adapters_sha256": file_digest(adapters_config),
        "project_ref": _ref_from_absolute(root, project),
        "admission_ref": _ref_from_absolute(root, admission),
        "inputs": [_ref_from_absolute(root, path) for path in inputs],
        "instructions": [_ref_from_absolute(root, path) for path in instructions],
        "output_paths": list(output_paths), "max_cost": max_cost,
        "max_seconds": max_seconds,
        "capabilities": ["filesystem_read", "filesystem_write"]}
    value["request_digest"] = request_digest(value)
    out.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".repair-request-", dir=out.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(canonical(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Constructing Request checks every semantic and filesystem constraint
        # before the requested path becomes visible.
        checked = Request(temporary)
        os.replace(temporary, out)
        directory = os.open(out.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return checked.data
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _reserve_budget(request):
    """Conservatively reserve from one campaign-wide repair allocation."""
    m, a = request.data, request.admission
    directory = safe_path(request.root, "runs/supervisor/" + a["campaign_id"])
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / "repair-budget.lock"
    if lock_path.is_symlink():
        raise RepairBridgeError("repair budget lock symlink rejected")
    with lock_path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        verify_ref(request.root, a["budget_ref"])
        identity_path = directory / "repair-budget.json"
        if identity_path.exists():
            if read_retained_json(identity_path) != request.budget:
                raise RepairBridgeError("retained campaign repair budget mismatch")
        else:
            _atomic(identity_path, request.budget)
        receipts = directory / "repair-reservations"
        if receipts.is_symlink():
            raise RepairBridgeError("repair reservation directory symlink rejected")
        receipts.mkdir(parents=True, exist_ok=True)
        retained = {}
        for path in receipts.iterdir():
            try:
                if path.is_symlink() or not path.is_file() or path.suffix != ".json":
                    raise RepairBridgeError("regular reservation receipt required")
                item = read_retained_json(path)
                _fields(item, "kind version campaign_digest request_id request_digest "
                        "max_cost max_seconds")
                if (item["kind"] != "research-repair-reservation" or item["version"] != 1 or
                        item["campaign_digest"] != a["campaign_digest"]):
                    raise RepairBridgeError("reservation campaign mismatch")
                identifier = _identifier(item["request_id"], "retained reservation id")
                _digest(item["request_digest"])
                _integer(item["max_cost"], 0, 1000000000, "retained repair cost")
                _integer(item["max_seconds"], 0, 86400, "retained repair time")
                expected_name = identifier + "-" + item["request_digest"] + ".json"
                if path.name != expected_name or identifier in retained:
                    raise RepairBridgeError("reservation receipt identity mismatch")
                retained[identifier] = item
            except RepairBridgeError as error:
                raise RepairBridgeError("invalid retained reservation") from error
        reservation = {"kind": "research-repair-reservation", "version": 1,
                       "campaign_digest": a["campaign_digest"],
                       "request_id": m["request_id"], "request_digest": m["request_digest"],
                       "max_cost": m["max_cost"], "max_seconds": m["max_seconds"]}
        existing = retained.get(m["request_id"])
        if existing is not None and existing != reservation:
            raise RepairBridgeError("repair request reservation identity mismatch")
        proposed = dict(retained)
        proposed[m["request_id"]] = reservation
        if (sum(item["max_cost"] for item in proposed.values()) > request.budget["max_cost"] or
                sum(item["max_seconds"] for item in proposed.values()) > request.budget["max_seconds"]):
            raise RepairBridgeError("campaign repair budget exceeded")
        receipt_path = receipts / (m["request_id"] + "-" + m["request_digest"] + ".json")
        _retain_exact(receipt_path, reservation, "repair reservation")
        return {"path": receipt_path.relative_to(request.root).as_posix(),
                "sha256": file_digest(receipt_path)}


def run_request(path, *, execute=False, approved_digest=None, environment=None):
    request = Request(path)
    m = request.data
    result = {"status": "inspection", "request_digest": m["request_digest"],
              "task": request.task, "gpu_dispatch_enabled": False,
              "scientific_retry_allowed": False,
              "source_delivery_status": "generated_unexecuted",
              "bridge_execution_attempted": bool(execute)}
    if not execute:
        return result
    if approved_digest != m["request_digest"]:
        raise RepairBridgeError("exact repair request approval required")
    request.directory.mkdir(parents=True, exist_ok=True)
    lock = request.directory / "owner.lock"
    if lock.is_symlink():
        raise RepairBridgeError("owner lock symlink rejected")
    with lock.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        if (read_json(request.path) != m or
                file_digest(request.runtime) != m["runtime_sha256"] or
                file_digest(request.adapters) != m["adapters_sha256"] or
                (request.db.stat().st_dev, request.db.stat().st_ino) !=
                (m["db_device"], m["db_inode"])):
            raise RepairBridgeError("approved request authority changed before dispatch")
        verify_ref(request.root, m["project_ref"])
        verify_ref(request.root, m["admission_ref"])
        for ref in [*m["inputs"], *m["instructions"]]:
            verify_ref(request.root, ref)
        request_path = request.directory / "request.json"
        if request_path.exists():
            if read_retained_json(request_path) != m:
                raise RepairBridgeError("retained full request identity mismatch")
        else:
            _atomic(request_path, m)
        task_path = request.directory / "task.json"
        intent_path = request.directory / "worker-intent.json"
        if (not intent_path.exists() and
                any(safe_path(request.root, relative).exists()
                    for relative in m["output_paths"])):
            raise RepairBridgeError("unowned repair output exists before first dispatch")
        if task_path.exists():
            if read_retained_json(task_path) != request.task:
                raise RepairBridgeError("retained repair task identity mismatch")
        else:
            _atomic(task_path, request.task)
        env = dict(os.environ)
        if environment:
            env.update(environment)
        prefix = [str(request.runtime), "--db", str(request.db)]
        worker_argv = prefix + ["worker", m["project_id"], "--owner", m["owner"],
                                "--adapter", m["adapter"],
                                "--adapters-config", str(request.adapters)]
        intent = {"kind": "research-repair-worker-intent-v1",
                  "request_digest": m["request_digest"], "project_id": m["project_id"],
                  "task_id": request.task["task_id"], "owner": m["owner"],
                  "adapter": m["adapter"], "argv": worker_argv,
                  "timeout_seconds": m["max_seconds"]}
        if intent_path.exists():
            if read_retained_json(intent_path) != intent:
                raise RepairBridgeError("retained worker intent identity mismatch")
            status_result = _invoke(prefix + ["status", m["project_id"]],
                                    cwd=request.root, environment=env, timeout=60)
            record = {"status": "reconcile_required", "request_digest": m["request_digest"],
                      "registration": None, "enqueue": None, "worker": None,
                      "runtime_status": status_result,
                      "reason": "worker intent already retained; automatic redispatch forbidden"}
            _record_run(request.directory, record)
            return {**result, **record, "task_path": str(task_path)}
        stage_intents = {
            request.directory / "registration-intent.json": {
                "kind": "research-repair-stage-intent-v1",
                "stage": "register-project", "request_digest": m["request_digest"],
                "argv": prefix + ["register-project", str(request.project_path)]},
            request.directory / "budget-reservation-intent.json": {
                "kind": "research-repair-stage-intent-v1",
                "stage": "reserve-campaign-budget",
                "request_digest": m["request_digest"],
                "max_cost": m["max_cost"], "max_seconds": m["max_seconds"]},
            request.directory / "enqueue-intent.json": {
                "kind": "research-repair-stage-intent-v1", "stage": "enqueue",
                "request_digest": m["request_digest"],
                "argv": prefix + ["enqueue", m["project_id"], str(task_path)]},
        }
        retained_stage = []
        for stage_path, expected in stage_intents.items():
            if stage_path.is_symlink():
                raise RepairBridgeError("repair stage intent symlink rejected")
            if stage_path.exists():
                if not stage_path.is_file():
                    raise RepairBridgeError("repair stage intent must be a regular file")
                stage_value = read_retained_json(stage_path)
                if stage_value != expected:
                    raise RepairBridgeError("retained repair stage intent mismatch")
                retained_stage.append(expected["stage"])
        if retained_stage:
            status_result = _invoke(prefix + ["status", m["project_id"]],
                                    cwd=request.root, environment=env, timeout=60)
            record = {"status": "reconcile_required",
                      "request_digest": m["request_digest"],
                      "registration": None, "enqueue": None, "worker": None,
                      "runtime_status": status_result,
                      "retained_stage_intents": retained_stage,
                      "reason": "stage intent already retained; automatic replay forbidden"}
            _record_run(request.directory, record)
            return {**result, **record, "task_path": str(task_path)}
        registration_argv = prefix + ["register-project", str(request.project_path)]
        _retain_exact(request.directory / "registration-intent.json",
                      {"kind": "research-repair-stage-intent-v1", "stage": "register-project",
                       "request_digest": m["request_digest"], "argv": registration_argv},
                      "registration intent")
        registration = _invoke(registration_argv,
                               cwd=request.root, environment=env, timeout=60)
        registration_status = ("project_registration_returned" if registration["returncode"] == 0 else
                               "project_registration_unknown" if registration["returncode"] is None else
                               "project_registration_failed")
        _record_run(request.directory,
                    {"status": registration_status, "stage": "register-project",
                     "request_digest": m["request_digest"], "registration": registration})
        if registration["returncode"] != 0:
            record = {"status": registration_status,
                      "request_digest": m["request_digest"],
                      "registration": registration, "enqueue": None, "worker": None}
            return {**result, **record, "task_path": str(task_path)}
        # Registration is provider-free.  Reserve before exposing the task to
        # the runtime queue, so a crash or rejected cumulative allocation can
        # never leave executable work outside the campaign ledger.  The
        # reservation is request-identity idempotent and is conservatively
        # retained if enqueue later fails or has an unknown acknowledgement.
        _retain_exact(request.directory / "budget-reservation-intent.json",
                      {"kind": "research-repair-stage-intent-v1",
                       "stage": "reserve-campaign-budget",
                       "request_digest": m["request_digest"],
                       "max_cost": m["max_cost"], "max_seconds": m["max_seconds"]},
                      "budget reservation intent")
        try:
            reservation_ref = _reserve_budget(request)
        except RepairBridgeError as error:
            _record_run(request.directory,
                        {"status": "budget_reservation_failed",
                         "stage": "reserve-campaign-budget",
                         "request_digest": m["request_digest"],
                         "error": type(error).__name__ + ": " + str(error)})
            raise
        _record_run(request.directory,
                    {"status": "budget_reserved", "stage": "reserve-campaign-budget",
                     "request_digest": m["request_digest"],
                     "reservation_ref": reservation_ref})
        enqueue_argv = prefix + ["enqueue", m["project_id"], str(task_path)]
        _retain_exact(request.directory / "enqueue-intent.json",
                      {"kind": "research-repair-stage-intent-v1", "stage": "enqueue",
                       "request_digest": m["request_digest"], "argv": enqueue_argv},
                      "enqueue intent")
        enqueue = _invoke(enqueue_argv,
                          cwd=request.root, environment=env, timeout=60)
        enqueue_status = ("enqueue_returned" if enqueue["returncode"] == 0 else
                          "enqueue_unknown" if enqueue["returncode"] is None else
                          "enqueue_failed")
        _record_run(request.directory,
                    {"status": enqueue_status, "stage": "enqueue",
                     "request_digest": m["request_digest"], "enqueue": enqueue})
        if enqueue["returncode"] != 0:
            record = {"status": enqueue_status, "request_digest": m["request_digest"],
                      "registration": registration, "enqueue": enqueue, "worker": None}
            return {**result, **record, "task_path": str(task_path)}
        _atomic(intent_path, intent)
        worker = _invoke(worker_argv, cwd=request.root, environment=env,
                         timeout=m["max_seconds"])
        status = ("worker_returned" if worker["returncode"] == 0 else
                  "worker_unknown" if worker["returncode"] is None else "worker_failed")
        record = {"status": status, "request_digest": m["request_digest"],
                  "registration": registration, "enqueue": enqueue, "worker": worker}
        _record_run(request.directory, record)
        return {**result, **record, "task_path": str(task_path)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--root", required=True)
    build.add_argument("--runtime-executable", required=True)
    build.add_argument("--db", required=True)
    build.add_argument("--project-id", required=True)
    build.add_argument("--owner", required=True)
    build.add_argument("--adapter", required=True)
    build.add_argument("--adapters-config", required=True)
    build.add_argument("--project", required=True)
    build.add_argument("--admission", required=True)
    build.add_argument("--input", action="append", required=True)
    build.add_argument("--instruction", action="append", required=True)
    build.add_argument("--output", action="append", required=True)
    build.add_argument("--max-cost", type=int, required=True)
    build.add_argument("--max-seconds", type=int, required=True)
    build.add_argument("--request-id", required=True)
    build.add_argument("--out", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("request")
    run.add_argument("--execute", action="store_true")
    run.add_argument("--approved-request-digest")
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            result = build_request(
                root=Path(args.root).resolve(), runtime_executable=Path(args.runtime_executable).resolve(),
                db=Path(args.db).resolve(), project_id=args.project_id, owner=args.owner,
                adapter=args.adapter, adapters_config=Path(args.adapters_config).resolve(),
                project=Path(args.project).resolve(),
                admission=Path(args.admission).resolve(),
                inputs=[Path(path).resolve() for path in args.input],
                instructions=[Path(path).resolve() for path in args.instruction],
                output_paths=args.output, max_cost=args.max_cost,
                max_seconds=args.max_seconds, request_id=args.request_id,
                out=Path(args.out).resolve())
        else:
            result = run_request(args.request, execute=args.execute,
                                 approved_digest=args.approved_request_digest)
        print(canonical(result))
    except (RepairBridgeError, OSError, subprocess.SubprocessError) as error:
        print(canonical({"status": "error", "error": type(error).__name__ + ": " + str(error)}))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
