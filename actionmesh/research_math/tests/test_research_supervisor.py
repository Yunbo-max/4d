"""Authored supervisor acceptance checks; execute only through the Local harness.

These use the installed canonical harness and real short CPU engineering jobs.
RESEARCH_AUTOPILOT_SKILL_DIR must identify that complete installed skill. No
scientific scorer/model/data is exercised. Removing identity, receipt, dependency,
STOP or deadline enforcement from the supervisor must fail the corresponding test.
"""
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[3] / "scripts/research_supervisor.py"
SPEC = importlib.util.spec_from_file_location("research_supervisor_under_test", SOURCE)
S = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(S)


class ResearchSupervisorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        location = os.environ.get("RESEARCH_AUTOPILOT_SKILL_DIR")
        if not location:
            raise RuntimeError("Local acceptance requires RESEARCH_AUTOPILOT_SKILL_DIR")
        cls.skill = Path(location).resolve(strict=True)
        cls.skill_hash = S.skill_digest(cls.skill)
        cls.H = S.load_harness(cls.skill, cls.skill_hash)
        cls.R = cls.H.R

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.pool = self.root / "pool"
        self.pool.mkdir()
        self.entries = []

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(S.canonical(value) + "\n")
        return {"path": relative, "sha256": S.file_digest(path)}

    def plan(self, name, *, dependencies=(), repair=None, fail=False, delay=0, input_refs=()):
        command = [sys.executable, "-c", "raise SystemExit(3)" if fail else
                   "from pathlib import Path; import time; time.sleep(%r); Path('result.txt').write_text('engineering')" % delay]
        native = self.R.make_plan(self.root, run_id=name, jobs=[{
            "trial_id": name, "command": command, "cwd": ".", "input_refs": list(input_refs),
            "code_refs": [], "output_paths": ["result.txt"], "seed": 1,
            "group": "engineering", "arm_role": "engineering"}],
            provenance={"git_revision": "engineering-fixture", "model_revision": "none",
                        "data_revision": "none", "environment_digest": "0" * 64},
            limits={"max_attempts": 1, "max_development_trials": 1,
                    "max_confirmation_trials": 0, "max_retries_per_trial": 0,
                    "wall_time_seconds": 5, "attempt_timeout_seconds": 4})
        native_ref = self.write("plans/" + name + "-native.json", native)
        harness = self.H.make_plan(self.root, batch_id=name, pool_dir=str(self.pool),
            tasks=[{"task_id": name, "idea_id": "engineering", "plan_ref": native_ref,
                    "depends_on": [], "priority": 1,
                    "resources": {"cpu_cores": 1, "ram_mib": 32, "gpu_count": 0,
                                  "gpu_peak_mib": None, "allow_gpu_share": False,
                                  "memory_profile_ref": None, "exclusive_keys": []}}],
            limits={"window_seconds": 30, "total_wall_seconds": 10,
                    "max_parallel_tasks": 1, "cpu_cores": 1, "ram_mib": 128,
                    "max_gpu_task_seconds": 0})
        entry = {"id": name, "plan_ref": self.write("plans/" + name + ".json", harness),
                 "plan_digest": harness["plan_digest"], "dependencies": list(dependencies),
                 "on_failure_of": repair}
        self.entries.append(entry)
        return harness

    def manifest(self, version=1):
        value = {"kind": "research-harness-campaign", "version": version,
                 "campaign_id": "engineering-supervisor", "root": str(self.root),
                 "python": str(Path(sys.executable).resolve()), "skill_root": str(self.skill),
                 "skill_digest": self.skill_hash, "pool_dir": str(self.pool),
                 "total_wall_seconds": 2400, "collection_reserve_seconds": 1800,
                 "plans": self.entries}
        if version == 2:
            for entry in value["plans"]:
                entry.setdefault("required_inputs", [])
        value["campaign_digest"] = S.campaign_digest(value)
        path = self.root / "campaign.json"
        path.write_text(S.canonical(value))
        return path, value

    def extended_manifest(self, base, *, path_name="campaign-next.json"):
        value = copy.deepcopy(base)
        value["plans"] = copy.deepcopy(self.entries)
        for entry in value["plans"]:
            entry.setdefault("required_inputs", [])
        value["version"] = 2
        value["campaign_digest"] = S.campaign_digest(value)
        path = self.root / path_name
        path.write_text(S.canonical(value))
        return path, value

    def execute(self, path, value, **options):
        return S.run_campaign(path, execute=True,
                              approved_digest=value["campaign_digest"], poll_seconds=.05, **options)

    def repair_receipt(self, campaign, *, bridge_status="worker_returned",
                       request_id="repair-a"):
        """Write a complete bridge-valid retained request and receipt chain."""
        failed_entry = campaign["plans"][0]
        failure_ref = self.write("repair-failure-" + request_id + ".json", {
            "status": "failed", "plan_digest": failed_entry["plan_digest"]})
        status_ref = self.write("repair-status-" + request_id + ".json", {
            "status": "inspection", "campaign_digest": campaign["campaign_digest"],
            "plans": {failed_entry["id"]: {"status": "failed", "repairable": True}}})
        budget_ref = self.write("repair-budget-" + request_id + ".json", {
            "kind": "research-repair-campaign-budget", "version": 1,
            "campaign_id": campaign["campaign_id"],
            "campaign_digest": campaign["campaign_digest"],
            "max_cost": 7, "max_seconds": 11})
        campaign_path = self.root / "campaign.json"
        campaign_ref = {"path": "campaign.json", "sha256": S.file_digest(campaign_path)}
        admission = {"kind": "research-repair-admission", "version": 1,
                     "campaign_id": campaign["campaign_id"],
                     "campaign_digest": campaign["campaign_digest"],
                     "failed_plan_id": failed_entry["id"],
                     "status": "terminal_failed", "classification": "code_error",
                     "gpu_allowed": False, "scientific_retry_allowed": False,
                     "reviewer": "local-reviewer", "reviewed_at": "2026-10-08T00:00:00Z",
                     "repair_scope": "candidate child source only",
                     "failure_ref": failure_ref,
                     "failed_plan_ref": failed_entry["plan_ref"],
                     "supervisor_status_ref": status_ref,
                     "budget_ref": budget_ref, "campaign_ref": campaign_ref}
        admission_ref = self.write("repair-admission-" + request_id + ".json",
                                   admission)
        instruction_ref = self.write("repair-instruction-" + request_id + ".json",
                                     {"scope": "repair reviewed code failure"})
        project_ref = self.write("repair-project-" + request_id + ".json",
            {"project_id": request_id, "root": str(self.root),
             "workflow_class": "scientific_method",
             "assigned_role": "web_supervisor", "adapter_allowlist": ["codex"],
             "skill_root": str(self.skill), "skill_digest": self.skill_hash,
             "max_cost": 7, "max_seconds": 11, "max_workers": 1})
        runtime = Path(sys.executable).resolve()
        database = self.root / "controller.sqlite3"
        if not database.exists():
            database.write_bytes(b"controller-fixture")
        adapters_path = self.root / "adapters.json"
        if not adapters_path.exists():
            adapters_path.write_text('{"adapters":{}}\n')
        request = {"kind": "research-runtime-repair-request", "version": 1,
                   "request_id": request_id, "root": str(self.root),
                   "runtime_executable": str(runtime),
                   "runtime_sha256": S.file_digest(runtime),
                   "db": str(database),
                   "db_device": database.stat().st_dev,
                   "db_inode": database.stat().st_ino,
                   "project_id": request_id, "dedicated_project": True,
                   "owner": "repair-owner",
                   "adapter": "codex",
                   "adapters_config": str(adapters_path),
                   "adapters_sha256": S.file_digest(adapters_path),
                   "project_ref": project_ref, "admission_ref": admission_ref,
                   "inputs": [failure_ref, failed_entry["plan_ref"], status_ref,
                              budget_ref, campaign_ref],
                   "instructions": [instruction_ref],
                   "output_paths": ["repairs/" + campaign["campaign_id"] + "/" +
                                    request_id + "/patch.diff"],
                   "max_cost": 7, "max_seconds": 11,
                   "capabilities": ["filesystem_read", "filesystem_write"]}
        request["request_digest"] = hashlib.sha256(
            S.canonical({key: value for key, value in request.items()
                         if key != "request_digest"}).encode()).hexdigest()
        directory = (self.root / "runs/supervisor" / campaign["campaign_id"] /
                     "repair-requests" / request_id)
        events = directory / "events"
        events.mkdir(parents=True)
        (directory / "owner.lock").write_text("")
        (directory / "request.json").write_text(S.canonical(request) + "\n")
        task = {"task_id": request_id, "action": "candidate_code",
                "inputs": list(request["inputs"]),
                "instructions": [request["admission_ref"],
                                 *request["instructions"]],
                "dependencies": [], "output_paths": list(request["output_paths"]),
                "completion": {"kind": "outputs_verified"},
                "max_cost": request["max_cost"],
                "max_seconds": request["max_seconds"],
                "capabilities": list(request["capabilities"]),
                "adapter": request["adapter"],
                "parameters": {"campaign_digest": campaign["campaign_digest"],
                               "failed_plan_id": campaign["plans"][0]["id"],
                               "gpu_allowed": False,
                               "scientific_retry_allowed": False,
                               "requires_reviewed_child_version": True}}
        (directory / "task.json").write_text(S.canonical(task) + "\n")
        prefix = [request["runtime_executable"], "--db", request["db"]]
        registration_argv = prefix + ["register-project",
                                      str(self.root / project_ref["path"])]
        enqueue_argv = prefix + ["enqueue", request_id,
                                 str(directory / "task.json")]
        worker_argv = prefix + ["worker", request_id, "--owner", request["owner"],
                                "--adapter", request["adapter"],
                                "--adapters-config", request["adapters_config"]]
        registration = self.repair_invocation(registration_argv)
        enqueue = self.repair_invocation(enqueue_argv)
        worker_code = (None if bridge_status == "worker_unknown" else
                       7 if bridge_status == "worker_failed" else 0)
        worker = self.repair_invocation(worker_argv, worker_code)
        values = []
        registration_status = (bridge_status if bridge_status.startswith(
            "project_registration_") else "project_registration_returned")
        registration_value = self.repair_invocation(
            registration_argv, None if registration_status.endswith("unknown") else
            7 if registration_status.endswith("failed") else 0)
        values.append({"status": registration_status, "stage": "register-project",
                       "request_digest": request["request_digest"],
                       "registration": registration_value})
        self.repair_stage_intent(directory, "register-project")
        if registration_status == "project_registration_returned" and not bridge_status.startswith(
                "project_registration_"):
            if bridge_status == "budget_reservation_failed":
                values.append({"status": bridge_status,
                               "stage": "reserve-campaign-budget",
                               "request_digest": request["request_digest"],
                               "error": "RepairBridgeError: fixture"})
                self.repair_stage_intent(directory, "reserve-campaign-budget")
            else:
                reservation_ref = self.write(
                    "repair-reservation-" + request_id + ".json",
                    {"kind": "research-repair-reservation", "version": 1,
                     "campaign_digest": campaign["campaign_digest"],
                     "request_id": request_id,
                     "request_digest": request["request_digest"],
                     "max_cost": request["max_cost"],
                     "max_seconds": request["max_seconds"]})
                values.append({"status": "budget_reserved",
                               "stage": "reserve-campaign-budget",
                               "request_digest": request["request_digest"],
                               "reservation_ref": reservation_ref})
                self.repair_stage_intent(directory, "reserve-campaign-budget")
                if bridge_status not in {"budget_reserved"}:
                    enqueue_status = (bridge_status if bridge_status.startswith("enqueue_")
                                      else "enqueue_returned")
                    enqueue_value = self.repair_invocation(
                        enqueue_argv, None if enqueue_status.endswith("unknown") else
                        7 if enqueue_status.endswith("failed") else 0)
                    values.append({"status": enqueue_status, "stage": "enqueue",
                                   "request_digest": request["request_digest"],
                                   "enqueue": enqueue_value})
                    self.repair_stage_intent(directory, "enqueue")
                    if (enqueue_status == "enqueue_returned" and
                            bridge_status.startswith("worker_")):
                        values.append({"status": bridge_status,
                                       "request_digest": request["request_digest"],
                                       "registration": registration,
                                       "enqueue": enqueue,
                                       "worker": worker})
                        self.repair_worker_intent(directory)
        previous = None
        encoded = []
        event_name = None
        for sequence, value in enumerate(values, 1):
            event = dict(value, sequence=sequence,
                         previous_event_sha256=previous)
            event_hash = hashlib.sha256(S.canonical(event).encode()).hexdigest()
            event_name = f"{sequence:020d}-" + event_hash + ".json"
            (events / event_name).write_text(S.canonical(event) + "\n")
            encoded.append(S.canonical(event))
            previous = event_hash
        (directory / "journal.jsonl").write_text("\n".join(encoded) + "\n")
        (directory / "last-status.json").write_text(S.canonical({
            "event": event_name, "status": bridge_status,
            "request_digest": request["request_digest"]}) + "\n")
        return directory

    def repair_invocation(self, argv, returncode=0):
        unknown = returncode is None
        return {"argv": argv, "returncode": returncode,
                "transport_error": "OSError: fixture" if unknown else None,
                "termination": {"process_group": 123, "term_sent": False,
                                "kill_sent": False, "direct_child_reaped": not unknown},
                "started_epoch": 1.0, "finished_epoch": 2.0,
                "stdout": "", "stdout_sha256": hashlib.sha256(b"").hexdigest(),
                "stdout_bytes": 0, "stdout_truncated": False,
                "stderr": "", "stderr_sha256": hashlib.sha256(b"").hexdigest(),
                "stderr_bytes": 0, "stderr_truncated": False}

    def rewrite_repair_chain(self, directory, mutate):
        paths = sorted((directory / "events").glob("*.json"))
        values = [json.loads(path.read_text()) for path in paths]
        mutate(values)
        # Worker events repeat the predecessor envelopes. Keep an intentionally
        # rehashed mutation internally consistent so argv checks are reached.
        registration = next((v['registration'] for v in values
                             if v.get('stage') == 'register-project'), None)
        enqueue = next((v['enqueue'] for v in values
                        if v.get('stage') == 'enqueue'), None)
        for value in values:
            if value.get('status', '').startswith('worker_'):
                if registration is not None:
                    value['registration'] = registration
                if enqueue is not None:
                    value['enqueue'] = enqueue
        for path in paths:
            path.unlink()
        previous, encoded, event_name = None, [], None
        for sequence, original in enumerate(values, 1):
            value = {key: item for key, item in original.items()
                     if key not in {"sequence", "previous_event_sha256"}}
            value.update(sequence=sequence, previous_event_sha256=previous)
            digest = hashlib.sha256(S.canonical(value).encode()).hexdigest()
            event_name = f"{sequence:020d}-" + digest + ".json"
            (directory / "events" / event_name).write_text(
                S.canonical(value) + "\n")
            encoded.append(S.canonical(value))
            previous = digest
        (directory / "journal.jsonl").write_text("\n".join(encoded) + "\n")
        request = json.loads((directory / "request.json").read_text())
        (directory / "last-status.json").write_text(S.canonical({
            "event": event_name, "status": values[-1]["status"],
            "request_digest": request["request_digest"]}) + "\n")

    def repair_stage_intent(self, directory, stage):
        request = json.loads((directory / "request.json").read_text())
        prefix = [request["runtime_executable"], "--db", request["db"]]
        if stage == "register-project":
            value = {"kind": "research-repair-stage-intent-v1", "stage": stage,
                     "request_digest": request["request_digest"],
                     "argv": prefix + ["register-project",
                                       str(self.root / request["project_ref"]["path"])]}
            name = "registration-intent.json"
        elif stage == "reserve-campaign-budget":
            value = {"kind": "research-repair-stage-intent-v1", "stage": stage,
                     "request_digest": request["request_digest"],
                     "max_cost": request["max_cost"],
                     "max_seconds": request["max_seconds"]}
            name = "budget-reservation-intent.json"
        elif stage == "enqueue":
            value = {"kind": "research-repair-stage-intent-v1", "stage": stage,
                     "request_digest": request["request_digest"],
                     "argv": prefix + ["enqueue", request["project_id"],
                                       str(directory / "task.json")]}
            name = "enqueue-intent.json"
        else:
            raise AssertionError("unknown repair stage fixture")
        (directory / name).write_text(S.canonical(value) + "\n")
        return directory / name

    def repair_worker_intent(self, directory):
        request = json.loads((directory / "request.json").read_text())
        value = {"kind": "research-repair-worker-intent-v1",
                 "request_digest": request["request_digest"],
                 "project_id": request["project_id"],
                 "task_id": request["request_id"], "owner": request["owner"],
                 "adapter": request["adapter"],
                 "argv": [request["runtime_executable"], "--db", request["db"],
                          "worker", request["project_id"], "--owner",
                          request["owner"], "--adapter", request["adapter"],
                          "--adapters-config", request["adapters_config"]],
                 "timeout_seconds": request["max_seconds"]}
        (directory / "worker-intent.json").write_text(S.canonical(value) + "\n")
        return directory / "worker-intent.json"

    def test_default_inspection_never_creates_runtime_state(self):
        self.plan("a")
        path, _ = self.manifest()
        result = S.run_campaign(path)
        self.assertEqual(result["status"], "inspection")
        self.assertFalse((self.root / "runs").exists())

    def test_canonical_preload_is_recompiled_from_verified_source_bytes(self):
        previous = sys.modules["run_harness"]
        previous.untrusted_cached_attribute = True
        refreshed = S.load_harness(self.skill, self.skill_hash)
        self.assertIsNot(refreshed, previous)
        self.assertFalse(hasattr(refreshed, "untrusted_cached_attribute"))

    def test_exact_campaign_digest_is_required_before_any_launch(self):
        self.plan("a")
        path, _ = self.manifest()
        with self.assertRaisesRegex(S.CampaignError, "approval"):
            S.run_campaign(path, execute=True, approved_digest="0" * 64)
        self.assertFalse((self.root / "runs/harness").exists())

    def test_unknown_manifest_permission_key_is_rejected(self):
        self.plan("a")
        path, value = self.manifest()
        value["gpu_approved"] = True
        value["campaign_digest"] = S.campaign_digest(value)
        path.write_text(S.canonical(value))
        with self.assertRaises(S.CampaignError):
            S.run_campaign(path)

    def test_changed_plan_bytes_fail_before_launch(self):
        self.plan("a")
        path, value = self.manifest()
        (self.root / "plans/a.json").write_text("{}")
        with self.assertRaises(S.CampaignError):
            self.execute(path, value)

    def test_duplicate_native_run_across_plans_is_rejected(self):
        self.plan("a")
        second = self.plan("b")
        second["tasks"][0]["plan_ref"] = self.H.C.reference(self.root, self.root / "plans/a-native.json")
        second["plan_digest"] = self.H.plan_digest(second)
        self.entries[1]["plan_ref"] = self.write("plans/b.json", second)
        self.entries[1]["plan_digest"] = second["plan_digest"]
        path, _ = self.manifest()
        with self.assertRaisesRegex(S.CampaignError, "native identity"):
            S.run_campaign(path)

    def test_dependency_cycle_is_rejected(self):
        self.plan("a", dependencies=("b",))
        self.plan("b", dependencies=("a",))
        path, _ = self.manifest()
        with self.assertRaisesRegex(S.CampaignError, "cycle"):
            S.run_campaign(path)

    def test_malformed_repair_trigger_cannot_become_ordinary_work(self):
        self.plan("a", fail=True)
        self.plan("repair", repair="a")
        path, value = self.manifest()
        for trigger in ("", "not an identifier", "../a", 7, False):
            with self.subTest(trigger=trigger):
                value["plans"][1]["on_failure_of"] = trigger
                value["campaign_digest"] = S.campaign_digest(value)
                path.write_text(S.canonical(value))
                with self.assertRaisesRegex(S.CampaignError, "repair trigger"):
                    self.execute(path, value)
                self.assertFalse((self.root / "runs/harness").exists())

    def delayed_recheck_cannot_dispatch(self, consumed_until):
        self.plan("a")
        path, value = self.manifest()
        clock, calls = [1000.], [0]
        real_recheck = S.Campaign.recheck
        def delayed_recheck(campaign):
            real_recheck(campaign)  # Retain real source, plan, and input validation.
            calls[0] += 1
            if calls[0] == 2:  # The final recheck before durable dispatch intent.
                clock[0] = consumed_until
        class ForbiddenTransport:
            def start(self, *args):
                raise AssertionError("expired dispatch budget must prevent process creation")
        with patch.object(S.Campaign, "recheck", delayed_recheck):
            result = self.execute(path, value, now=lambda: clock[0],
                                  transport=ForbiddenTransport())
        self.assertEqual(calls[0], 2)
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["deadline_epoch"], 3400.)
        self.assertFalse((self.root / "runs/harness").exists())

    def test_recheck_reaching_collection_cutoff_prevents_dispatch(self):
        self.delayed_recheck_cannot_dispatch(1600.)

    def test_recheck_leaving_insufficient_fresh_plan_time_prevents_dispatch(self):
        # Five workload seconds remain before reserve; the frozen plan needs ten.
        self.delayed_recheck_cannot_dispatch(1595.)

    def test_pending_project_recovery_is_not_mutated_by_inspection(self):
        self.plan("a")
        path, _ = self.manifest()
        pending = self.root / ".pending-commit.json"
        pending.write_text("retained recovery evidence")
        with self.assertRaisesRegex(S.CampaignError, "project recovery"):
            S.run_campaign(path)
        self.assertEqual(pending.read_text(), "retained recovery evidence")

    def test_duplicate_trial_identity_cannot_be_hidden_behind_new_native_run(self):
        self.plan("a")
        self.plan("b")
        native = json.loads((self.root / "plans/b-native.json").read_text())
        native["jobs"][0]["trial_id"] = "a"
        native["plan_digest"] = self.R.plan_digest(native)
        native_ref = self.write("plans/b-native.json", native)
        plan = json.loads((self.root / "plans/b.json").read_text())
        plan["tasks"][0]["plan_ref"] = native_ref
        plan["plan_digest"] = self.H.plan_digest(plan)
        self.entries[1]["plan_ref"] = self.write("plans/b.json", plan)
        self.entries[1]["plan_digest"] = plan["plan_digest"]
        path, _ = self.manifest()
        with self.assertRaisesRegex(S.CampaignError, "trial identity"):
            S.run_campaign(path)

    def test_failure_blocks_only_dependents_and_independent_plan_runs(self):
        self.plan("a", fail=True)
        self.plan("b", dependencies=("a",))
        self.plan("c")
        path, value = self.manifest()
        result = self.execute(path, value)
        self.assertEqual(result["plans"]["a"], "failed")
        self.assertEqual(result["plans"]["b"], "blocked")
        self.assertEqual(result["plans"]["c"], "completed")
        self.assertFalse((self.root / "runs/attempts/b").exists())

    def test_preapproved_repair_runs_once_without_retrying_failed_identity(self):
        self.plan("a", fail=True)
        self.plan("repair", repair="a")
        path, value = self.manifest()
        self.execute(path, value)
        result = self.execute(path, value)
        self.assertEqual(result["plans"]["repair"], "completed")
        receipt = json.loads((self.root / "runs/attempts/a/receipt.json").read_text())
        self.assertEqual(len(receipt["attempts"]), 1)
        self.assertEqual(receipt["attempts"][0]["retry_index"], 0)

    def test_completed_campaign_resume_preserves_receipt_and_deadline(self):
        self.plan("a")
        path, value = self.manifest()
        first = self.execute(path, value)
        receipt = self.root / "runs/attempts/a/receipt.json"
        before = receipt.read_bytes()
        second = self.execute(path, value)
        self.assertEqual(first["deadline_epoch"], second["deadline_epoch"])
        self.assertEqual(receipt.read_bytes(), before)

    def test_completed_campaign_remains_completed_after_original_deadline(self):
        self.plan("a")
        path, value = self.manifest()
        first = self.execute(path, value)
        resumed = self.execute(path, value, now=lambda: first["deadline_epoch"] + 1)
        self.assertEqual(resumed["status"], "completed")
        self.assertEqual(resumed["deadline_epoch"], first["deadline_epoch"])

    def test_stop_file_prevents_first_batch(self):
        self.plan("a")
        path, value = self.manifest()
        directory = self.root / "runs/supervisor/engineering-supervisor"
        directory.mkdir(parents=True)
        (directory / "STOP").touch()
        result = self.execute(path, value)
        self.assertEqual(result["status"], "paused")
        self.assertFalse((self.root / "runs/harness").exists())

    def test_live_stop_hands_off_then_resumes_same_worker_without_duplicate(self):
        self.plan("a", delay=1.5)
        path, value = self.manifest()
        process_file = self.root / "runs/harness/a/tasks/a/process.json"
        stop_file = self.root / "runs/supervisor/engineering-supervisor/STOP"
        harness = self.H
        class StopWhenWorkerLive(S.HarnessTransport):
            def start(self, argv, directory, environment):
                child = super().start(argv, directory, environment)
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    if process_file.is_file() and harness._alive(json.loads(process_file.read_text())):
                        stop_file.touch()
                        return child
                    if child.poll() is not None:
                        return child
                    time.sleep(.02)
                child.send_signal(S.signal.SIGINT)
                child.wait(timeout=5)
                raise AssertionError("fixture worker did not become live")
        paused = self.execute(path, value, transport=StopWhenWorkerLive())
        self.assertEqual(paused["status"], "paused")
        stop_file.unlink()
        resumed = self.execute(path, value)
        self.assertEqual(resumed["status"], "completed")
        receipt = json.loads((self.root / "runs/attempts/a/receipt.json").read_text())
        self.assertEqual(len(receipt["attempts"]), 1)
        self.assertEqual(receipt["attempts"][0]["status"], "completed")

    def held_pool_case(self, action):
        self.plan("a")
        path, value = self.manifest()
        stream = (self.pool / "driver.lock").open("a+")
        S.fcntl.flock(stream, S.fcntl.LOCK_EX)
        clock = [1000.]
        release = threading.Event()
        observed = []
        directory = self.root / "runs/supervisor/engineering-supervisor"
        journal = directory / "journal.jsonl"
        def react():
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and not release.is_set():
                if journal.exists() and '"event":"host_driver_wait"' in journal.read_text():
                    observed.append(True)
                    if action == "release":
                        S.fcntl.flock(stream, S.fcntl.LOCK_UN)
                    elif action == "stop":
                        (directory / "STOP").touch()
                    else:
                        clock[0] = 5000.
                    return
                time.sleep(.01)
            if not release.is_set():
                clock[0] = 5000.  # Bound a broken fixture instead of hanging acceptance.
        watcher = threading.Thread(target=react)
        watcher.start()
        try:
            result = self.execute(path, value, now=lambda: clock[0])
        finally:
            release.set()
            watcher.join(timeout=11)
            S.fcntl.flock(stream, S.fcntl.LOCK_UN)
            stream.close()
        self.assertEqual(observed, [True], "fixture must actually observe the held-lock wait")
        events = [json.loads(line) for line in journal.read_text().splitlines()]
        self.assertEqual(sum(event["event"] == "host_driver_wait" for event in events), 1)
        return result

    def test_existing_pool_driver_is_waited_out_then_next_plan_runs(self):
        result = self.held_pool_case("release")
        self.assertEqual(result["status"], "completed")
        receipt = json.loads((self.root / "runs/attempts/a/receipt.json").read_text())
        self.assertEqual(len(receipt["attempts"]), 1)

    def test_stop_during_pool_wait_launches_nothing(self):
        result = self.held_pool_case("stop")
        self.assertEqual(result["status"], "paused")
        self.assertFalse((self.root / "runs/harness").exists())

    def test_original_deadline_applies_during_pool_wait(self):
        result = self.held_pool_case("deadline")
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["deadline_epoch"], 3400.)
        self.assertFalse((self.root / "runs/harness").exists())

    def test_elapsed_downtime_does_not_reset_deadline(self):
        self.plan("a")
        path, value = self.manifest()
        directory = self.root / "runs/supervisor/engineering-supervisor"
        directory.mkdir(parents=True)
        stop = directory / "STOP"
        stop.touch()
        first = self.execute(path, value, now=lambda: 1000.)
        stop.unlink()
        result = self.execute(path, value, now=lambda: 4000.)
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["deadline_epoch"], first["deadline_epoch"])
        self.assertFalse((self.root / "runs/harness").exists())

    def test_missing_state_with_retained_native_identity_never_launches(self):
        self.plan("a")
        path, value = self.manifest()
        (self.root / "runs/attempts/a").mkdir(parents=True)
        result = self.execute(path, value)
        self.assertEqual(result["status"], "reconciliation_required")
        self.assertFalse((self.root / "runs/harness/a/state.json").exists())

    def test_forged_completed_state_without_receipt_is_unknown(self):
        harness = self.plan("a")
        path, value = self.manifest()
        batch = self.root / "runs/harness/a"
        batch.mkdir(parents=True)
        (batch / "plan.json").write_text(S.canonical(harness))
        (batch / "state.json").write_text(S.canonical({
            "format": "research-harness-state-v1", "batch_id": "a", "root": str(self.root),
            "plan_digest": harness["plan_digest"], "boot_id": self.H._self_identity()["boot_id"],
            "tasks": {"a": {"status": "completed"}}, "status": "completed"}))
        result = self.execute(path, value)
        self.assertEqual(result["status"], "reconciliation_required")

    def test_tampered_completed_output_blocks_resume(self):
        self.plan("a")
        path, value = self.manifest()
        self.execute(path, value)
        output = next((self.root / "runs/attempts/a").glob("*/workspace/result.txt"))
        output.write_text("changed")
        result = self.execute(path, value)
        self.assertEqual(result["status"], "reconciliation_required")

    def test_plan_symlink_is_rejected(self):
        self.plan("a")
        path, _ = self.manifest()
        plan = self.root / "plans/a.json"
        actual = self.root / "plans/actual.json"
        plan.rename(actual)
        plan.symlink_to(actual)
        with self.assertRaisesRegex(S.CampaignError, "symlink"):
            S.run_campaign(path)

    def gpu_plan(self):
        plan = self.plan("gpu")
        plan["gpus"]["uuids"] = ["GPU-engineering-fixture-never-launched"]
        plan["tasks"][0]["resources"]["gpu_count"] = 1
        plan["limits"]["max_gpu_task_seconds"] = 5
        plan["plan_digest"] = self.H.plan_digest(plan)
        self.entries[0]["plan_ref"] = self.write("plans/gpu.json", plan)
        self.entries[0]["plan_digest"] = plan["plan_digest"]

    def test_gpu_stop_default_never_reaches_transport(self):
        self.gpu_plan()
        path, value = self.manifest()
        class ForbiddenTransport:
            def start(self, *args):
                raise AssertionError("GPU STOP must prevent process creation")
        result = self.execute(path, value, transport=ForbiddenTransport())
        self.assertEqual(result["status"], "blocked_gpu_stop")
        self.assertFalse((self.root / "runs/harness").exists())

    def test_future_local_resume_acknowledgment_uses_only_pinned_harness_argv(self):
        self.gpu_plan()
        path, value = self.manifest()
        captured = []
        class TransportBoundaryReached(RuntimeError):
            pass
        class NoExecutionTransport:
            def start(self, argv, directory, environment):
                captured.append(argv)
                raise TransportBoundaryReached("test boundary; nothing executed")
        with self.assertRaises(TransportBoundaryReached):
            self.execute(path, value, transport=NoExecutionTransport(),
                         allow_gpu_after_user_resume=True)
        self.assertEqual(captured, [[value["python"], "-B", str(self.skill / "scripts/run_harness.py"),
                                   str(self.root / "plans/gpu.json"), "--root", str(self.root),
                                   "--execute", "--approved-plan-digest", self.entries[0]["plan_digest"]]])
        self.assertFalse((self.root / "runs/harness").exists())
        resumed_without_acknowledgment = self.execute(path, value)
        # A transport exception after durable intent cannot establish no launch.
        self.assertEqual(resumed_without_acknowledgment["status"], "reconciliation_required")


    def missing_input_campaign(self):
        input_path = self.root / "inputs/required.txt"
        input_path.parent.mkdir()
        input_path.write_text("retained exact input")
        ref = {"path": "inputs/required.txt", "sha256": S.file_digest(input_path)}
        self.plan("waiting", input_refs=[ref])
        self.entries[-1]["required_inputs"] = [ref]
        self.plan("independent")
        path, value = self.manifest(version=2)
        content = input_path.read_bytes()
        input_path.unlink()
        return path, value, input_path, content

    def test_missing_declared_input_isolates_only_dependent_plan(self):
        path, value, _, _ = self.missing_input_campaign()
        before = path.read_bytes()
        result = self.execute(path, value)
        self.assertEqual(result["status"], "waiting_inputs")
        self.assertEqual(result["plans"]["independent"], "completed")
        self.assertEqual(result["plans"]["waiting"], "waiting_inputs")
        self.assertEqual(path.read_bytes(), before)
        self.assertFalse((self.root / "runs/attempts/waiting").exists())

    def test_exact_input_arrival_continues_without_new_manifest_or_budget(self):
        path, value, input_path, content = self.missing_input_campaign()
        heartbeat = self.root / "runs/supervisor/engineering-supervisor/heartbeat.json"
        delivered = threading.Event()
        errors = []
        def deliver():
            try:
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline:
                    if heartbeat.is_file():
                        observed = json.loads(heartbeat.read_text())
                        if observed["plans"].get("independent") == "completed":
                            temp = input_path.with_suffix(".incoming")
                            temp.write_bytes(content)
                            os.replace(temp, input_path)
                            delivered.set()
                            return
                    time.sleep(.02)
                errors.append("independent real harness batch never completed")
                S.request_stop(path, value["campaign_digest"])
            except Exception as error:
                errors.append(repr(error))
        publisher = threading.Thread(target=deliver)
        publisher.start()
        try:
            result = self.execute(path, value, watch_ready=True, heartbeat_seconds=.05)
        finally:
            publisher.join(timeout=20)
        self.assertEqual(errors, [])
        self.assertTrue(delivered.is_set())
        self.assertEqual(result["status"], "completed")
        self.assertEqual(set(result["plans"].values()), {"completed"})
        state = json.loads((heartbeat.parent / "state.json").read_text())
        self.assertEqual(state["starts"], {"waiting": 1, "independent": 1})
        self.assertEqual(state["deadline_epoch"], state["started_epoch"] + value["total_wall_seconds"])

    def test_final_readiness_loss_does_not_poison_dispatch_state(self):
        input_path = self.root / "required.txt"
        input_path.write_text("exact bytes")
        ref = {"path": "required.txt", "sha256": S.file_digest(input_path)}
        self.plan("waiting", input_refs=[ref])
        self.entries[-1]["required_inputs"] = [ref]
        path, value = self.manifest(version=2)
        real_recheck = S.Campaign.recheck
        calls = [0]
        def remove_before_final_recheck(campaign):
            calls[0] += 1
            if calls[0] == 2:
                input_path.unlink()
            real_recheck(campaign)
        class ForbiddenTransport:
            def start(self, *args):
                raise AssertionError("lost readiness cannot reach process creation")
        with patch.object(S.Campaign, "recheck", remove_before_final_recheck):
            result = self.execute(path, value, transport=ForbiddenTransport())
        self.assertEqual(result["status"], "waiting_inputs")
        state = json.loads((self.root / "runs/supervisor/engineering-supervisor/state.json").read_text())
        self.assertEqual(state["starts"]["waiting"], 0)
        self.assertIsNone(state["active"])
        self.assertFalse((self.root / "runs/harness").exists())

    def test_stop_after_intent_before_spawn_keeps_resumable_zero_start(self):
        self.plan("a")
        path, value = self.manifest()
        real_atomic = S._atomic
        injected = [False]
        def stop_after_intent(target, payload):
            real_atomic(target, payload)
            if (target.name == "state.json" and payload.get("starts", {}).get("a") == 1
                    and not injected[0]):
                injected[0] = True
                (target.parent / "STOP").touch()  # Legacy direct sentinel/signal timing.
        class ForbiddenTransport:
            def start(self, *args):
                raise AssertionError("observed STOP before Popen must prevent launch")
        with patch.object(S, "_atomic", stop_after_intent):
            paused = self.execute(path, value, transport=ForbiddenTransport())
        self.assertEqual(paused["status"], "paused")
        state = json.loads((self.root / "runs/supervisor/engineering-supervisor/state.json").read_text())
        self.assertEqual(state["starts"]["a"], 0)
        self.assertEqual(self.execute(path, value, resume=True)["status"], "completed")

    def test_changed_arriving_input_is_rejected_without_launch(self):
        path, value, input_path, _ = self.missing_input_campaign()
        input_path.write_text("wrong bytes")
        with self.assertRaisesRegex(S.CampaignError, "pinned bytes"):
            self.execute(path, value, watch_ready=True)
        self.assertFalse((self.root / "runs/harness").exists())

    def test_two_real_batches_continue_with_receipt_bound_dependency(self):
        self.plan("first")
        self.plan("second", dependencies=("first",))
        path, value = self.manifest()
        result = self.execute(path, value, heartbeat_seconds=.05)
        self.assertEqual(result["status"], "completed")
        for name in ("first", "second"):
            receipt = json.loads((self.root / ("runs/attempts/" + name + "/receipt.json")).read_text())
            self.assertEqual(receipt["status"], "completed")
            self.assertEqual(len(receipt["attempts"]), 1)
        inspected = S.run_campaign(path)
        self.assertIn("heartbeat", inspected["retained_status"])
        self.assertEqual(inspected["retained_status"]["online_repair_agent"], "not_connected")

    def test_repair_bridge_worker_receipt_is_observed_without_dispatch_or_patch_acceptance(self):
        self.plan("first")
        path, value = self.manifest()
        self.repair_receipt(value)
        inspected = S.run_campaign(path)
        retained = inspected["retained_status"]
        self.assertEqual(retained["online_repair_agent"],
                         "worker_returned_unreviewed")
        self.assertEqual(retained["repair_bridge"]["requests"]["repair-a"]["bridge_status"],
                         "worker_returned")
        self.assertFalse((self.root / "runs/harness").exists())

    def test_repair_bridge_unknown_requires_reconciliation_and_survives_heartbeat(self):
        self.plan("first")
        path, value = self.manifest()
        self.repair_receipt(value, bridge_status="worker_unknown")
        result = self.execute(path, value, heartbeat_seconds=.05)
        self.assertEqual(result["online_repair_agent"], "reconcile_required")
        heartbeat = json.loads((self.root / "runs/supervisor/engineering-supervisor/heartbeat.json").read_text())
        self.assertEqual(heartbeat["kind"], "research-supervisor-heartbeat-v2")
        self.assertEqual(heartbeat["online_repair_agent"], "reconcile_required")
        self.assertEqual(heartbeat["repair_bridge_status"], "reconcile_required")

    def test_repair_worker_intent_without_terminal_event_requires_reconciliation(self):
        self.plan("first")
        path, value = self.manifest()
        directory = self.repair_receipt(value, bridge_status="enqueue_returned")
        self.repair_worker_intent(directory)
        retained = S.run_campaign(path)["retained_status"]
        self.assertEqual(retained["online_repair_agent"], "reconcile_required")
        self.assertEqual(retained["repair_bridge"]["requests"]["repair-a"]
                         ["bridge_status"], "worker_intent_without_terminal_event")

    def test_repair_enqueue_returned_without_worker_is_not_reported_ready(self):
        self.plan("first")
        path, value = self.manifest()
        directory = self.repair_receipt(value, bridge_status="enqueue_returned")
        self.repair_stage_intent(directory, "enqueue")
        retained = S.run_campaign(path)["retained_status"]
        self.assertEqual(retained["online_repair_agent"], "reconcile_required")

    def test_exact_stage_failures_remain_failed(self):
        self.plan("first")
        path, value = self.manifest()
        cases = (("project_registration_failed", "register-project"),
                 ("budget_reservation_failed", "reserve-campaign-budget"),
                 ("enqueue_failed", "enqueue"))
        for index, (status, stage) in enumerate(cases):
            with self.subTest(stage=stage):
                request_id = "repair-failure-" + str(index)
                directory = self.repair_receipt(
                    value, bridge_status=status, request_id=request_id)
                self.repair_stage_intent(directory, stage)
                retained = S.run_campaign(path)["retained_status"]
                item = retained["repair_bridge"]["requests"][request_id]
                self.assertEqual(item["status"], "failed")

    def test_mutated_stage_or_worker_intent_fails_closed(self):
        self.plan("first")
        path, value = self.manifest()
        directory = self.repair_receipt(value, bridge_status="worker_returned")
        stage_path = self.repair_stage_intent(directory, "enqueue")
        stage_value = json.loads(stage_path.read_text())
        del stage_value["argv"]
        stage_path.write_text(S.canonical(stage_value) + "\n")
        with self.assertRaisesRegex(S.CampaignError, "stage intent identity"):
            S.run_campaign(path)
        stage_path.unlink()
        worker_path = self.repair_worker_intent(directory)
        worker_value = json.loads(worker_path.read_text())
        worker_value["timeout_seconds"] += 1
        worker_path.write_text(S.canonical(worker_value) + "\n")
        with self.assertRaisesRegex(S.CampaignError, "worker intent identity"):
            S.run_campaign(path)

    def test_bridge_invalid_live_controller_identity_is_rejected(self):
        self.plan("first")
        path, value = self.manifest()
        directory = self.repair_receipt(value, bridge_status="worker_returned")
        request = json.loads((directory / "request.json").read_text())
        database = Path(request["db"])
        database.unlink()
        database.mkdir()
        with self.assertRaisesRegex(S.CampaignError, "runtime identity"):
            S.run_campaign(path)

    def test_unowned_preflight_output_is_never_reported_configured(self):
        self.plan("first")
        path, value = self.manifest()
        directory = self.repair_receipt(value, bridge_status="worker_returned")
        request = json.loads((directory / "request.json").read_text())
        for name in ("registration-intent.json", "budget-reservation-intent.json",
                     "enqueue-intent.json", "worker-intent.json", "last-status.json",
                     "journal.jsonl"):
            (directory / name).unlink()
        for event in (directory / "events").iterdir():
            event.unlink()
        (directory / "events").rmdir()
        output = self.root / request["output_paths"][0]
        output.parent.mkdir(parents=True)
        output.write_text("unowned")
        retained = S.run_campaign(path)["retained_status"]
        item = retained["repair_bridge"]["requests"]["repair-a"]
        self.assertEqual(item["status"], "reconcile_required")
        self.assertEqual(item["bridge_status"], "unowned_repair_output")

    def test_complete_worker_receipt_requires_all_intents(self):
        self.plan("first")
        path, value = self.manifest()
        directory = self.repair_receipt(value, bridge_status="worker_returned")
        (directory / "worker-intent.json").unlink()
        retained = S.run_campaign(path)["retained_status"]
        self.assertEqual(retained["online_repair_agent"], "reconcile_required")
        self.assertEqual(retained["repair_bridge"]["requests"]["repair-a"]
                         ["bridge_status"], "invalid_bridge_progression")

    def test_each_stage_result_requires_its_exact_intent(self):
        self.plan("first")
        path, value = self.manifest()
        directory = self.repair_receipt(value, bridge_status="worker_returned")
        cases = (("register-project", "registration-intent.json"),
                 ("reserve-campaign-budget", "budget-reservation-intent.json"),
                 ("enqueue", "enqueue-intent.json"))
        for stage, missing in cases:
            with self.subTest(stage=stage):
                (directory / missing).unlink()
                retained = S.run_campaign(path)["retained_status"]
                item = retained["repair_bridge"]["requests"]["repair-a"]
                self.assertEqual(item["status"], "reconcile_required")
                self.repair_stage_intent(directory, stage)

    def test_hash_consistent_registration_argv_mutation_fails_closed(self):
        self.plan("first")
        path, value = self.manifest()
        directory = self.repair_receipt(value, bridge_status="worker_returned")
        self.rewrite_repair_chain(directory, lambda events:
            events[0]["registration"]["argv"].append("--wrong"))
        with self.assertRaisesRegex(S.CampaignError,
                                    "registration invocation identity"):
            S.run_campaign(path)

    def test_hash_consistent_enqueue_argv_mutation_fails_closed(self):
        self.plan("first")
        path, value = self.manifest()
        directory = self.repair_receipt(value, bridge_status="worker_returned")
        self.rewrite_repair_chain(directory, lambda events:
            events[2]["enqueue"]["argv"].append("--wrong"))
        with self.assertRaisesRegex(S.CampaignError, "enqueue invocation identity"):
            S.run_campaign(path)

    def test_hash_consistent_stage_result_missing_payload_fails_closed(self):
        self.plan("first")
        path, value = self.manifest()
        directory = self.repair_receipt(value, bridge_status="enqueue_failed")
        last_path = directory / "last-status.json"
        last = json.loads(last_path.read_text())
        old_path = directory / "events" / last["event"]
        event = json.loads(old_path.read_text())
        del event["enqueue"]
        digest = hashlib.sha256(S.canonical(event).encode()).hexdigest()
        new_name = event["sequence"].__format__("020d") + "-" + digest + ".json"
        new_path = old_path.with_name(new_name)
        old_path.rename(new_path)
        new_path.write_text(S.canonical(event) + "\n")
        lines = (directory / "journal.jsonl").read_text().splitlines()
        lines[-1] = S.canonical(event)
        (directory / "journal.jsonl").write_text("\n".join(lines) + "\n")
        last["event"] = new_name
        last_path.write_text(S.canonical(last) + "\n")
        with self.assertRaisesRegex(S.CampaignError, "contract fields"):
            S.run_campaign(path)

    def test_hash_consistent_worker_argv_mutation_fails_closed(self):
        self.plan("first")
        path, value = self.manifest()
        directory = self.repair_receipt(value, bridge_status="worker_returned")
        last_path = directory / "last-status.json"
        last = json.loads(last_path.read_text())
        old_path = directory / "events" / last["event"]
        event = json.loads(old_path.read_text())
        event["worker"]["argv"][-1] = str(self.root / "wrong-adapters.json")
        digest = hashlib.sha256(S.canonical(event).encode()).hexdigest()
        new_name = f"{event['sequence']:020d}-" + digest + ".json"
        new_path = old_path.with_name(new_name)
        old_path.rename(new_path)
        new_path.write_text(S.canonical(event) + "\n")
        lines = (directory / "journal.jsonl").read_text().splitlines()
        lines[-1] = S.canonical(event)
        (directory / "journal.jsonl").write_text("\n".join(lines) + "\n")
        last["event"] = new_name
        last_path.write_text(S.canonical(last) + "\n")
        with self.assertRaisesRegex(S.CampaignError, "worker invocation identity"):
            S.run_campaign(path)

    def test_hash_bound_wrong_reservation_identity_fails_closed(self):
        self.plan("first")
        path, value = self.manifest()
        directory = self.repair_receipt(value, bridge_status="budget_reserved")
        last_path = directory / "last-status.json"
        last = json.loads(last_path.read_text())
        old_path = directory / "events" / last["event"]
        event = json.loads(old_path.read_text())
        receipt_path = self.root / event["reservation_ref"]["path"]
        receipt = json.loads(receipt_path.read_text())
        receipt["max_cost"] += 1
        receipt_path.write_text(S.canonical(receipt) + "\n")
        event["reservation_ref"]["sha256"] = S.file_digest(receipt_path)
        digest = hashlib.sha256(S.canonical(event).encode()).hexdigest()
        new_name = f"{event['sequence']:020d}-" + digest + ".json"
        new_path = old_path.with_name(new_name)
        old_path.rename(new_path)
        new_path.write_text(S.canonical(event) + "\n")
        lines = (directory / "journal.jsonl").read_text().splitlines()
        lines[-1] = S.canonical(event)
        (directory / "journal.jsonl").write_text("\n".join(lines) + "\n")
        last["event"] = new_name
        last_path.write_text(S.canonical(last) + "\n")
        with self.assertRaisesRegex(S.CampaignError, "reservation receipt identity"):
            S.run_campaign(path)

    def test_repair_owner_lock_contention_defers_observation(self):
        self.plan("first")
        path, value = self.manifest()
        directory = self.repair_receipt(value)
        lock = (directory / "owner.lock").open("r")
        S.fcntl.flock(lock, S.fcntl.LOCK_EX)
        try:
            retained = S.run_campaign(path)["retained_status"]
        finally:
            S.fcntl.flock(lock, S.fcntl.LOCK_UN)
            lock.close()
        request = retained["repair_bridge"]["requests"]["repair-a"]
        self.assertEqual(request["status"], "reconcile_required")
        self.assertEqual(request["reason"], "bridge_owner_active")

    def test_repair_historical_event_or_journal_reorder_fails_closed(self):
        self.plan("first")
        path, value = self.manifest()
        directory = self.repair_receipt(value, bridge_status="worker_returned")
        original = next((directory / "events").glob("00000000000000000001-*.json"))
        retained_bytes = original.read_bytes()
        first = json.loads(retained_bytes)
        original.write_text(S.canonical(dict(first, stage="wrong-stage")) + "\n")
        with self.assertRaisesRegex(S.CampaignError, "event inventory"):
            S.run_campaign(path)
        original.write_bytes(retained_bytes)
        lines = (directory / "journal.jsonl").read_text().splitlines()
        (directory / "journal.jsonl").write_text(
            lines[1] + "\n" + lines[0] + "\n" + "\n".join(lines[2:]) + "\n")
        retained = S.run_campaign(path)["retained_status"]
        self.assertEqual(retained["online_repair_agent"], "reconcile_required")
        self.assertEqual(retained["repair_bridge"]["requests"]["repair-a"]
                         ["bridge_status"], "unindexed_event")

    def test_parent_repair_receipt_survives_extension_and_lost_ack_readback(self):
        self.plan("first")
        base_path, base = self.manifest(version=2)
        self.repair_receipt(base)
        self.execute(base_path, base)
        self.plan("second", dependencies=("first",))
        next_path, proposed = self.extended_manifest(base)
        extended = S.extend_campaign(
            base_path, next_path,
            approved_current_digest=base["campaign_digest"],
            approved_next_digest=proposed["campaign_digest"])
        self.assertEqual(extended["status"], "extended")
        self.assertEqual(extended["online_repair_agent"], "readback_required")
        inspected = S.run_campaign(next_path)
        self.assertEqual(inspected["retained_status"]["online_repair_agent"],
                         "worker_returned_unreviewed")
        repeated = S.extend_campaign(
            base_path, next_path,
            approved_current_digest=base["campaign_digest"],
            approved_next_digest=proposed["campaign_digest"])
        self.assertEqual(repeated["status"], "already_extended")
        self.assertEqual(repeated["online_repair_agent"], "readback_required")

    def test_repair_bridge_tamper_or_cross_campaign_receipt_fails_closed(self):
        self.plan("first")
        path, value = self.manifest()
        directory = self.repair_receipt(value)
        last = json.loads((directory / "last-status.json").read_text())
        event = directory / "events" / last["event"]
        event.write_text(S.canonical({"status": "worker_returned",
                                      "request_digest": "0" * 64}) + "\n")
        with self.assertRaisesRegex(S.CampaignError, "event .*identity"):
            S.run_campaign(path)

    def test_repair_event_symlink_root_fails_closed(self):
        self.plan("first")
        path, value = self.manifest()
        directory = self.repair_receipt(value)
        real = directory / "events-real"
        (directory / "events").rename(real)
        (directory / "events").symlink_to(real, target_is_directory=True)
        with self.assertRaisesRegex(S.CampaignError, "event root"):
            S.run_campaign(path)

    def test_reviewed_append_only_extension_preserves_budget_and_completed_history(self):
        self.plan("first")
        base_path, base = self.manifest(version=2)
        first = self.execute(base_path, base)
        state_path = self.root / "runs/supervisor/engineering-supervisor/state.json"
        before = json.loads(state_path.read_text())
        first_receipt = self.root / "runs/attempts/first/receipt.json"
        retained_receipt = first_receipt.read_bytes()

        self.plan("second", dependencies=("first",))
        next_path, proposed = self.extended_manifest(base)
        extended = S.extend_campaign(
            base_path, next_path,
            approved_current_digest=base["campaign_digest"],
            approved_next_digest=proposed["campaign_digest"])

        self.assertEqual(extended["status"], "extended")
        self.assertEqual(extended["added_plans"], ["second"])
        migrated = json.loads(state_path.read_text())
        self.assertEqual(migrated["started_epoch"], before["started_epoch"])
        self.assertEqual(migrated["deadline_epoch"], before["deadline_epoch"])
        self.assertEqual(migrated["starts"], {"first": 1, "second": 0})
        self.assertEqual(first_receipt.read_bytes(), retained_receipt)
        repeated = S.extend_campaign(
            base_path, next_path,
            approved_current_digest=base["campaign_digest"],
            approved_next_digest=proposed["campaign_digest"])
        self.assertEqual(repeated["status"], "already_extended")

        finished = self.execute(next_path, proposed)
        self.assertEqual(finished["status"], "completed")
        self.assertEqual(finished["deadline_epoch"], before["deadline_epoch"])
        self.assertEqual(json.loads(state_path.read_text())["starts"], {"first": 1, "second": 1})

    def test_extension_recovers_lost_ack_between_manifest_and_state_replace(self):
        self.plan("first")
        base_path, base = self.manifest(version=2)
        self.execute(base_path, base)
        before = json.loads((self.root / "runs/supervisor/engineering-supervisor/state.json").read_text())
        self.plan("second")
        next_path, proposed = self.extended_manifest(base)
        real_atomic = S._atomic
        interrupted = [False]
        def lose_after_manifest(target, value):
            real_atomic(target, value)
            if target.name == "manifest.json" and value == proposed and not interrupted[0]:
                interrupted[0] = True
                raise OSError("injected lost acknowledgement after manifest replacement")
        with patch.object(S, "_atomic", lose_after_manifest):
            with self.assertRaisesRegex(OSError, "lost acknowledgement"):
                S.extend_campaign(
                    base_path, next_path,
                    approved_current_digest=base["campaign_digest"],
                    approved_next_digest=proposed["campaign_digest"])
        recovered = S.extend_campaign(
            base_path, next_path,
            approved_current_digest=base["campaign_digest"],
            approved_next_digest=proposed["campaign_digest"])
        self.assertEqual(recovered["status"], "extended")
        state = json.loads((self.root / "runs/supervisor/engineering-supervisor/state.json").read_text())
        self.assertEqual(state["started_epoch"], before["started_epoch"])
        self.assertEqual(state["deadline_epoch"], before["deadline_epoch"])
        self.assertEqual(state["starts"], {"first": 1, "second": 0})

    def test_extension_rejects_rewriting_a_retained_plan_or_budget(self):
        self.plan("first")
        base_path, base = self.manifest(version=2)
        self.execute(base_path, base)
        original_state = (self.root / "runs/supervisor/engineering-supervisor/state.json").read_bytes()
        self.plan("second")
        next_path, proposed = self.extended_manifest(base)

        for mutation in ("plan", "budget"):
            with self.subTest(mutation=mutation):
                changed = copy.deepcopy(proposed)
                if mutation == "plan":
                    changed["plans"][0]["dependencies"] = ["second"]
                else:
                    changed["total_wall_seconds"] += 1
                changed["campaign_digest"] = S.campaign_digest(changed)
                next_path.write_text(S.canonical(changed))
                with self.assertRaisesRegex(S.CampaignError, "append-only|fixed campaign envelope"):
                    S.extend_campaign(
                        base_path, next_path,
                        approved_current_digest=base["campaign_digest"],
                        approved_next_digest=changed["campaign_digest"])
                self.assertEqual(
                    (self.root / "runs/supervisor/engineering-supervisor/state.json").read_bytes(),
                    original_state)

    def test_extension_requires_both_exact_reviewed_digests(self):
        self.plan("first")
        base_path, base = self.manifest(version=2)
        self.execute(base_path, base)
        self.plan("second")
        next_path, proposed = self.extended_manifest(base)
        with self.assertRaisesRegex(S.CampaignError, "current campaign approval"):
            S.extend_campaign(base_path, next_path,
                              approved_current_digest="0" * 64,
                              approved_next_digest=proposed["campaign_digest"])
        with self.assertRaisesRegex(S.CampaignError, "next campaign approval"):
            S.extend_campaign(base_path, next_path,
                              approved_current_digest=base["campaign_digest"],
                              approved_next_digest="0" * 64)

    def test_extension_rejects_manifest_replacement_after_approval_snapshot(self):
        self.plan("first")
        base_path, base = self.manifest(version=2)
        self.execute(base_path, base)
        self.plan("second")
        next_path, proposed = self.extended_manifest(base)
        changed = copy.deepcopy(proposed)
        changed["plans"][1]["dependencies"] = ["first"]
        changed["campaign_digest"] = S.campaign_digest(changed)
        original_init = S.Campaign.__init__
        replaced = [False]
        def replace_before_second_read(campaign, source):
            if Path(source) == next_path and not replaced[0]:
                replaced[0] = True
                next_path.write_text(S.canonical(changed))
            original_init(campaign, source)
        with patch.object(S.Campaign, "__init__", replace_before_second_read):
            with self.assertRaisesRegex(S.CampaignError, "approved manifest changed"):
                S.extend_campaign(
                    base_path, next_path,
                    approved_current_digest=base["campaign_digest"],
                    approved_next_digest=proposed["campaign_digest"])

    def test_extension_archives_exact_old_state_and_rejects_history_symlink(self):
        self.plan("first")
        base_path, base = self.manifest(version=2)
        self.execute(base_path, base)
        directory = self.root / "runs/supervisor/engineering-supervisor"
        old_state = (directory / "state.json").read_bytes()
        self.plan("second")
        next_path, proposed = self.extended_manifest(base)
        history = directory / "manifest-history"
        history.mkdir()
        target = history / (base["campaign_digest"] + ".json")
        target.symlink_to(directory / "manifest.json")
        with self.assertRaisesRegex(S.CampaignError, "symlink"):
            S.extend_campaign(
                base_path, next_path,
                approved_current_digest=base["campaign_digest"],
                approved_next_digest=proposed["campaign_digest"])
        target.unlink()
        S.extend_campaign(
            base_path, next_path,
            approved_current_digest=base["campaign_digest"],
            approved_next_digest=proposed["campaign_digest"])
        archived_state = history / (base["campaign_digest"] + "-state.json")
        self.assertEqual(archived_state.read_bytes(), old_state)

    def test_multi_hop_extension_does_not_accept_a_nonpredecessor_as_idempotent(self):
        self.plan("first")
        path_a, manifest_a = self.manifest(version=2)
        self.execute(path_a, manifest_a)
        self.plan("second")
        path_b, manifest_b = self.extended_manifest(manifest_a, path_name="campaign-b.json")
        S.extend_campaign(path_a, path_b,
                          approved_current_digest=manifest_a["campaign_digest"],
                          approved_next_digest=manifest_b["campaign_digest"])
        self.plan("third", dependencies=("second",))
        path_c, manifest_c = self.extended_manifest(manifest_b, path_name="campaign-c.json")
        S.extend_campaign(path_b, path_c,
                          approved_current_digest=manifest_b["campaign_digest"],
                          approved_next_digest=manifest_c["campaign_digest"])
        with self.assertRaisesRegex(S.CampaignError, "predecessor"):
            S.extend_campaign(path_a, path_c,
                              approved_current_digest=manifest_a["campaign_digest"],
                              approved_next_digest=manifest_c["campaign_digest"])

    def test_extension_refuses_unsettled_or_active_campaign(self):
        self.plan("first")
        base_path, base = self.manifest(version=2)
        directory = self.root / "runs/supervisor/engineering-supervisor"
        directory.mkdir(parents=True)
        (self.root / "runs/attempts/first").mkdir(parents=True)
        state = {"kind": "research-supervisor-state-v1",
                 "campaign_digest": base["campaign_digest"],
                 "started_epoch": time.time(), "deadline_epoch": time.time() + 2400,
                 "last_epoch": time.time(), "active": "first", "starts": {"first": 1}}
        state["deadline_epoch"] = state["started_epoch"] + base["total_wall_seconds"]
        (directory / "state.json").write_text(S.canonical(state))
        (directory / "manifest.json").write_text(S.canonical(base))
        self.plan("second")
        next_path, proposed = self.extended_manifest(base)
        with self.assertRaisesRegex(S.CampaignError, "settled campaign"):
            S.extend_campaign(base_path, next_path,
                              approved_current_digest=base["campaign_digest"],
                              approved_next_digest=proposed["campaign_digest"])

    def test_extension_rejects_malformed_retained_state_before_history_mutation(self):
        self.plan("first")
        base_path, base = self.manifest(version=2)
        self.execute(base_path, base)
        self.plan("second")
        next_path, proposed = self.extended_manifest(base)
        directory = self.root / "runs/supervisor/engineering-supervisor"
        state_path = directory / "state.json"
        original = json.loads(state_path.read_text())
        for field, value in (("last_epoch", "not-a-number"), ("starts", {"first": 33})):
            with self.subTest(field=field):
                changed = copy.deepcopy(original)
                changed[field] = value
                state_path.write_text(S.canonical(changed))
                with self.assertRaisesRegex(S.CampaignError, "retained campaign state"):
                    S.extend_campaign(
                        base_path, next_path,
                        approved_current_digest=base["campaign_digest"],
                        approved_next_digest=proposed["campaign_digest"])
                self.assertFalse((directory / "manifest-history").exists())
        state_path.write_text(S.canonical(original))
        S.extend_campaign(
            base_path, next_path,
            approved_current_digest=base["campaign_digest"],
            approved_next_digest=proposed["campaign_digest"])
        migrated = json.loads(state_path.read_text())
        migrated["starts"]["first"] = 33
        state_path.write_text(S.canonical(migrated))
        with self.assertRaisesRegex(S.CampaignError, "retained campaign state"):
            S.extend_campaign(
                base_path, next_path,
                approved_current_digest=base["campaign_digest"],
                approved_next_digest=proposed["campaign_digest"])

    def test_extension_rejects_symlink_or_wrong_identity_heartbeat(self):
        self.plan("first")
        base_path, base = self.manifest(version=2)
        self.execute(base_path, base)
        self.plan("second")
        next_path, proposed = self.extended_manifest(base)
        directory = self.root / "runs/supervisor/engineering-supervisor"
        heartbeat = directory / "heartbeat.json"
        retained = heartbeat.read_bytes()
        elsewhere = self.root / "outside-heartbeat.json"
        elsewhere.write_bytes(retained)
        heartbeat.unlink()
        heartbeat.symlink_to(elsewhere)
        with self.assertRaisesRegex(S.CampaignError, "symlink"):
            S.extend_campaign(
                base_path, next_path,
                approved_current_digest=base["campaign_digest"],
                approved_next_digest=proposed["campaign_digest"])
        heartbeat.unlink()
        heartbeat.write_bytes(retained)
        wrong = json.loads(heartbeat.read_text())
        wrong["campaign_digest"] = "0" * 64
        heartbeat.write_text(S.canonical(wrong))
        with self.assertRaisesRegex(S.CampaignError, "heartbeat identity"):
            S.extend_campaign(
                base_path, next_path,
                approved_current_digest=base["campaign_digest"],
                approved_next_digest=proposed["campaign_digest"])

    def test_lost_ack_after_real_completion_settles_without_duplicate(self):
        self.plan("first")
        self.plan("second", dependencies=("first",))
        path, value = self.manifest()
        class LostAcknowledgement(OSError):
            pass
        class LoseCompletedAck(S.HarnessTransport):
            def start(self, argv, directory, environment):
                child = super().start(argv, directory, environment)
                child.wait(timeout=15)
                raise LostAcknowledgement("actual completed driver response lost")
        with self.assertRaises(LostAcknowledgement):
            self.execute(path, value, transport=LoseCompletedAck())
        state_path = self.root / "runs/supervisor/engineering-supervisor/state.json"
        original = json.loads(state_path.read_text())
        first_receipt = self.root / "runs/attempts/first/receipt.json"
        retained = first_receipt.read_bytes()
        resumed = self.execute(path, value)
        self.assertEqual(resumed["status"], "completed")
        self.assertEqual(first_receipt.read_bytes(), retained)
        self.assertEqual(resumed["deadline_epoch"], original["deadline_epoch"])
        self.assertEqual(json.loads(state_path.read_text())["starts"]["first"], 1)

    def test_unknown_launch_ack_cannot_relaunch_absent_batch(self):
        self.plan("first")
        path, value = self.manifest()
        class LostBeforeIdentity(S.HarnessTransport):
            def start(self, *args):
                raise OSError("cannot establish whether transport created a process")
        with self.assertRaises(OSError):
            self.execute(path, value, transport=LostBeforeIdentity())
        result = self.execute(path, value)
        self.assertEqual(result["status"], "reconciliation_required")
        state = json.loads((self.root / "runs/supervisor/engineering-supervisor/state.json").read_text())
        self.assertEqual(state["starts"]["first"], 1)
        self.assertFalse((self.root / "runs/harness").exists())

    def test_killed_supervisor_reconciles_surviving_real_driver(self):
        self.plan("first", delay=1.5)
        self.plan("second", dependencies=("first",))
        path, value = self.manifest()
        command = [sys.executable, str(SOURCE), str(path), "--execute",
                   "--approved-campaign-digest", value["campaign_digest"],
                   "--poll-seconds", "0.05", "--heartbeat-seconds", "0.05"]
        process_file = self.root / "runs/harness/first/tasks/first/process.json"
        driver_file = self.root / "runs/supervisor/engineering-supervisor/first/driver.json"
        controller = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if process_file.is_file() and driver_file.is_file():
                    if self.H._alive(json.loads(process_file.read_text())):
                        break
                if controller.poll() is not None:
                    self.fail("supervisor exited before recovery injection: " + controller.stderr.read().decode())
                time.sleep(.01)
            else:
                self.fail("real worker did not become live before bounded recovery injection")
            controller.kill()  # Inject only controller failure; retain driver/workers.
            controller.wait(timeout=5)
            result = self.execute(path, value, heartbeat_seconds=.05)
            self.assertEqual(result["status"], "completed")
            for name in ("first", "second"):
                receipt = json.loads((self.root / ("runs/attempts/" + name + "/receipt.json")).read_text())
                self.assertEqual(len(receipt["attempts"]), 1)
                self.assertEqual(receipt["attempts"][0]["retry_index"], 0)
        finally:
            if controller.poll() is None:
                controller.send_signal(S.signal.SIGINT)
                controller.wait(timeout=15)
            controller.stderr.close()

    def test_stop_and_explicit_resume_keep_original_deadline(self):
        self.plan("a")
        path, value = self.manifest()
        stopped = S.request_stop(path, value["campaign_digest"])
        self.assertFalse(stopped["workers_terminated"])
        paused = self.execute(path, value)
        self.assertEqual(paused["status"], "paused")
        resumed = self.execute(path, value, resume=True)
        self.assertEqual(resumed["status"], "completed")
        self.assertEqual(resumed["deadline_epoch"], paused["deadline_epoch"])
        self.assertFalse(resumed["gpu_dispatch_enabled"])

    def test_newer_stop_during_resume_is_preserved(self):
        self.plan("a")
        path, value = self.manifest()
        S.request_stop(path, value["campaign_digest"])
        self.assertEqual(self.execute(path, value)["status"], "paused")
        original_init = S.Campaign.__init__
        def parse_then_new_stop(campaign, source):
            original_init(campaign, source)  # All real plan/source checks still run.
            S.request_stop(path, value["campaign_digest"])
        # Inject ordering only; actual STOP files, locks and parser are used.
        with patch.object(S.Campaign, "__init__", parse_then_new_stop):
            with self.assertRaisesRegex(S.CampaignError, "newer STOP"):
                self.execute(path, value, resume=True)
        self.assertTrue((self.root / "runs/supervisor/engineering-supervisor/STOP").is_file())
        self.assertFalse((self.root / "runs/harness").exists())

    def test_resume_without_retained_state_cannot_start_fresh_budget(self):
        self.plan("a")
        path, value = self.manifest()
        with self.assertRaisesRegex(S.CampaignError, "original retained state"):
            self.execute(path, value, resume=True)
        self.assertFalse((self.root / "runs/harness").exists())

if __name__ == "__main__":
    unittest.main()
