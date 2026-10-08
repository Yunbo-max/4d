"""Generated/unexecuted Local acceptance for the runtime repair bridge.

These engineering tests use a real subprocess executable and filesystem.  They
do not run project methods, tests, models, downloads, inference, or scoring.
"""
import importlib.util
import json
from pathlib import Path
import stat
import sys
import tempfile
import time
import unittest


SOURCE = Path(__file__).resolve().parents[3] / "scripts/research_repair_bridge.py"
SPEC = importlib.util.spec_from_file_location("research_repair_bridge_under_test", SOURCE)
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)


class ResearchRepairBridgeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        (self.root / "runs/supervisor/campaign").mkdir(parents=True)
        self.db = self.root / "controller.sqlite3"
        self.db.write_bytes(b"registered-controller-fixture")
        self.adapters = self.root / "adapters.json"
        self.adapters.write_text('{"adapters":{}}\n')
        self.runtime_log = self.root / "runtime-argv.jsonl"
        self.runtime = self.root / "research-autopilot"
        self.runtime.write_text(
            "#!" + sys.executable + "\n"
            "import json, os, subprocess, sys, time\n"
            "with open(os.environ['REPAIR_RUNTIME_LOG'], 'a') as stream:\n"
            "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            "if 'worker' in sys.argv and os.environ.get('REPAIR_RUNTIME_SLEEP_WORKER'):\n"
            "    if os.environ.get('REPAIR_RUNTIME_CHILD_HEARTBEAT'):\n"
            "        child_code = \"import os,time; p=os.environ['REPAIR_RUNTIME_CHILD_HEARTBEAT']; f=open(p,'a'); exec(\\\"while True:\\\\n f.write('x'); f.flush(); time.sleep(.05)\\\")\"\n"
            "        child = subprocess.Popen([sys.executable, '-c', child_code], env=os.environ)\n"
            "        with open(os.environ['REPAIR_RUNTIME_CHILD_PID'], 'w') as stream:\n"
            "            stream.write(str(child.pid))\n"
            "    print('partial-worker-output', flush=True)\n"
            "    sys.stderr.write('partial-worker-error\\n'); sys.stderr.flush()\n"
            "    time.sleep(float(os.environ['REPAIR_RUNTIME_SLEEP_WORKER']))\n"
            "print(json.dumps({'ok': True, 'argv': sys.argv[1:]}))\n")
        self.runtime.chmod(self.runtime.stat().st_mode | stat.S_IXUSR)
        self.project = self.root / "repair-project.json"
        self.project.write_text(R.canonical({
            "project_id": "repair-candidate-a-v2", "root": str(self.root),
            "workflow_class": "scientific_method", "assigned_role": "web_supervisor",
            "adapter_allowlist": ["codex"], "skill_root": str(self.root),
            "skill_digest": "c" * 64, "max_cost": 100000,
            "max_seconds": 900, "max_workers": 1}) + "\n")
        self.project_ref = {"path": "repair-project.json",
                            "sha256": R.file_digest(self.project)}
        self.input_ref = self.write(
            "failure-receipt.json", {"status": "failed", "plan_digest": "b" * 64})
        self.plan_ref = self.write("failed-plan.json", {"plan_digest": "b" * 64})
        (self.root / "pool").mkdir()
        campaign = {"kind": "research-harness-campaign", "version": 2,
                    "campaign_id": "campaign",
                    "root": str(self.root), "python": sys.executable,
                    "skill_root": str(self.root), "skill_digest": "c" * 64,
                    "pool_dir": str(self.root / "pool"), "total_wall_seconds": 1801,
                    "collection_reserve_seconds": 1800,
                    "plans": [{"id": "candidate-a", "plan_ref": self.plan_ref,
                               "plan_digest": "b" * 64, "dependencies": [],
                               "on_failure_of": None, "required_inputs": []}]}
        campaign["campaign_digest"] = R.campaign_manifest_digest(campaign)
        self.campaign_digest = campaign["campaign_digest"]
        self.campaign_ref = self.write("campaign.json", campaign)
        self.budget_ref = self.write(
            "repair-budget.json",
            {"kind": "research-repair-campaign-budget", "version": 1,
             "campaign_id": "campaign", "campaign_digest": self.campaign_digest,
             "max_cost": 100000, "max_seconds": 900})
        self.status_ref = self.write(
            "supervisor-inspection.json",
            {"status": "inspection", "campaign_digest": self.campaign_digest,
             "plans": {"candidate-a": {"status": "failed", "repairable": True}}})
        failure = {"kind": "research-repair-admission", "version": 1,
                   "campaign_id": "campaign", "campaign_digest": self.campaign_digest,
                   "failed_plan_id": "candidate-a", "status": "terminal_failed",
                   "classification": "code_error", "scientific_retry_allowed": False,
                   "gpu_allowed": False, "reviewer": "local_executor",
                   "reviewed_at": "2026-10-08T18:00:00Z",
                   "repair_scope": "produce a reviewed child source version only",
                   "failure_ref": self.input_ref, "failed_plan_ref": self.plan_ref,
                   "supervisor_status_ref": self.status_ref, "budget_ref": self.budget_ref,
                   "campaign_ref": self.campaign_ref}
        self.admission_ref = self.write("admission.json", failure)
        self.instruction_ref = self.write("repair.md", "repair exact failed code")

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, str):
            path.write_text(value)
        else:
            path.write_text(R.canonical(value) + "\n")
        return {"path": relative, "sha256": R.file_digest(path)}

    def request(self):
        value = {
            "kind": "research-runtime-repair-request", "version": 1,
            "request_id": "repair-candidate-a-v2", "root": str(self.root),
            "runtime_executable": str(self.runtime), "db": str(self.db),
            "db_device": self.db.stat().st_dev, "db_inode": self.db.stat().st_ino,
            "runtime_sha256": R.file_digest(self.runtime),
            "adapters_sha256": R.file_digest(self.adapters),
            "project_id": "repair-candidate-a-v2", "dedicated_project": True,
            "owner": "local-repair-owner",
            "adapter": "codex", "adapters_config": str(self.adapters),
            "project_ref": self.project_ref,
            "admission_ref": self.admission_ref,
            "inputs": [self.input_ref, self.plan_ref, self.status_ref, self.budget_ref,
                       self.campaign_ref],
            "instructions": [self.instruction_ref],
            "output_paths": ["repairs/campaign/repair-candidate-a-v2/patch.diff",
                             "repairs/campaign/repair-candidate-a-v2/review.json"],
            "max_cost": 100000, "max_seconds": 900,
            "capabilities": ["filesystem_read", "filesystem_write"]}
        value["request_digest"] = R.request_digest(value)
        path = self.root / "request.json"
        path.write_text(R.canonical(value) + "\n")
        return path, value

    def test_inspection_builds_candidate_code_task_without_writing_or_spawning(self):
        path, value = self.request()
        result = R.run_request(path)
        self.assertEqual(result["status"], "inspection")
        self.assertEqual(result["task"]["action"], "candidate_code")
        self.assertEqual(result["task"]["max_cost"], value["max_cost"])
        self.assertEqual(result["task"]["max_seconds"], value["max_seconds"])
        self.assertFalse(result["task"]["parameters"]["scientific_retry_allowed"])
        self.assertFalse(result["task"]["parameters"]["gpu_allowed"])
        self.assertFalse((self.root / "runs/supervisor/campaign/repair-requests").exists())

    def test_execute_requires_exact_digest_before_runtime_or_state_write(self):
        path, _ = self.request()
        with self.assertRaisesRegex(R.RepairBridgeError, "approval"):
            R.run_request(path, execute=True, approved_digest="0" * 64)
        self.assertFalse((self.root / "runs/supervisor/campaign/repair-requests").exists())
        self.assertFalse(self.runtime_log.exists())

    def test_execute_enqueues_then_runs_one_existing_runtime_worker(self):
        path, value = self.request()
        result = R.run_request(
            path, execute=True, approved_digest=value["request_digest"],
            environment={"REPAIR_RUNTIME_LOG": str(self.runtime_log)})
        self.assertEqual(result["status"], "worker_returned")
        calls = [json.loads(line) for line in self.runtime_log.read_text().splitlines()]
        self.assertEqual(calls[0][:3], ["--db", str(self.db), "register-project"])
        self.assertEqual(calls[1][:4], ["--db", str(self.db), "enqueue", "repair-candidate-a-v2"])
        self.assertEqual(calls[2][:4], ["--db", str(self.db), "worker", "repair-candidate-a-v2"])
        self.assertIn("--adapter", calls[2])
        self.assertIn("--adapters-config", calls[2])
        task = json.loads(Path(result["task_path"]).read_text())
        self.assertEqual(task["task_id"], value["request_id"])
        self.assertEqual(task["completion"], {"kind": "outputs_verified"})

    def test_repeated_execute_reconciles_status_without_second_worker(self):
        path, value = self.request()
        options = {"execute": True, "approved_digest": value["request_digest"],
                   "environment": {"REPAIR_RUNTIME_LOG": str(self.runtime_log)}}
        self.assertEqual(R.run_request(path, **options)["status"], "worker_returned")
        repeated = R.run_request(path, **options)
        self.assertEqual(repeated["status"], "reconcile_required")
        calls = [json.loads(line) for line in self.runtime_log.read_text().splitlines()]
        self.assertEqual(sum("worker" in call for call in calls), 1)
        self.assertEqual(calls[-1][:4], ["--db", str(self.db), "status", "repair-candidate-a-v2"])
        records = [json.loads(item.read_text()) for item in
                   sorted((self.root / "runs/supervisor/campaign/repair-requests/"
                           "repair-candidate-a-v2/events").glob("*.json"))]
        self.assertEqual([item["status"] for item in records],
                         ["project_registration_returned", "budget_reserved", "enqueue_returned",
                          "worker_returned", "reconcile_required"])

    def test_worker_timeout_is_immutable_unknown_and_never_redispatched(self):
        project = json.loads(self.project.read_text())
        project["max_seconds"] = 1
        self.project.write_text(R.canonical(project) + "\n")
        self.project_ref = {"path": self.project.name,
                            "sha256": R.file_digest(self.project)}
        path, value = self.request()
        value["max_seconds"] = 1
        value["request_digest"] = R.request_digest(value)
        path.write_text(R.canonical(value) + "\n")
        options = {
            "execute": True, "approved_digest": value["request_digest"],
            "environment": {"REPAIR_RUNTIME_LOG": str(self.runtime_log),
                            "REPAIR_RUNTIME_SLEEP_WORKER": "3",
                            "REPAIR_RUNTIME_CHILD_HEARTBEAT": str(self.root / "child-heartbeat"),
                            "REPAIR_RUNTIME_CHILD_PID": str(self.root / "child-pid")}}
        first = R.run_request(path, **options)
        self.assertEqual(first["status"], "worker_unknown")
        self.assertIsNone(first["worker"]["returncode"])
        self.assertIn("TimeoutExpired", first["worker"]["transport_error"])
        self.assertGreater(first["worker"]["stdout_bytes"], 0)
        self.assertGreater(first["worker"]["stderr_bytes"], 0)
        self.assertEqual(len(first["worker"]["stdout_sha256"]), 64)
        self.assertTrue(first["worker"]["termination"]["term_sent"])
        heartbeat = self.root / "child-heartbeat"
        self.assertTrue((self.root / "child-pid").is_file())
        size_after_return = heartbeat.stat().st_size
        time.sleep(.2)
        self.assertEqual(heartbeat.stat().st_size, size_after_return)
        self.assertLess(first["worker"]["started_epoch"],
                        first["worker"]["finished_epoch"])
        events = self.root / "runs/supervisor/campaign/repair-requests/repair-candidate-a-v2/events"
        first_event = next(json.loads(item.read_text()) for item in events.glob("*.json")
                           if json.loads(item.read_text())["status"] == "worker_unknown")
        self.assertEqual(first_event["status"], "worker_unknown")
        repeated = R.run_request(path, **options)
        self.assertEqual(repeated["status"], "reconcile_required")
        calls = [json.loads(line) for line in self.runtime_log.read_text().splitlines()]
        self.assertEqual(sum("worker" in call for call in calls), 1)
        self.assertEqual(calls[-1][:4],
                         ["--db", str(self.db), "status", "repair-candidate-a-v2"])
        self.assertEqual(len(list(events.glob("*.json"))), 5)

    def test_lost_ack_after_durable_worker_intent_never_dispatches_worker(self):
        path, value = self.request()
        request = R.Request(path)
        request.directory.mkdir(parents=True)
        (request.directory / "request.json").write_text(R.canonical(value) + "\n")
        (request.directory / "task.json").write_text(R.canonical(request.task) + "\n")
        intent = {"kind": "research-repair-worker-intent-v1",
                  "request_digest": value["request_digest"],
                  "project_id": value["project_id"], "task_id": value["request_id"],
                  "owner": value["owner"], "adapter": value["adapter"],
                  "argv": [str(self.runtime), "--db", str(self.db), "worker",
                           value["project_id"], "--owner", value["owner"],
                           "--adapter", value["adapter"], "--adapters-config",
                           str(self.adapters)], "timeout_seconds": value["max_seconds"]}
        (request.directory / "worker-intent.json").write_text(R.canonical(intent) + "\n")
        result = R.run_request(
            path, execute=True, approved_digest=value["request_digest"],
            environment={"REPAIR_RUNTIME_LOG": str(self.runtime_log)})
        self.assertEqual(result["status"], "reconcile_required")
        calls = [json.loads(line) for line in self.runtime_log.read_text().splitlines()]
        self.assertFalse(any("worker" in call for call in calls))
        self.assertEqual(calls[-1][2], "status")

    def test_retained_stage_intent_forbids_automatic_stage_replay(self):
        for filename, stage in (
                ("registration-intent.json", "register-project"),
                ("budget-reservation-intent.json", "reserve-campaign-budget"),
                ("enqueue-intent.json", "enqueue")):
            with self.subTest(stage=stage):
                path, value = self.request()
                request = R.Request(path)
                request.directory.mkdir(parents=True, exist_ok=True)
                prefix = [str(request.runtime), "--db", str(request.db)]
                expected = {
                    "register-project": {
                        "kind": "research-repair-stage-intent-v1", "stage": stage,
                        "request_digest": value["request_digest"],
                        "argv": prefix + ["register-project", str(request.project_path)]},
                    "reserve-campaign-budget": {
                        "kind": "research-repair-stage-intent-v1", "stage": stage,
                        "request_digest": value["request_digest"],
                        "max_cost": value["max_cost"],
                        "max_seconds": value["max_seconds"]},
                    "enqueue": {
                        "kind": "research-repair-stage-intent-v1", "stage": stage,
                        "request_digest": value["request_digest"],
                        "argv": prefix + ["enqueue", value["project_id"],
                                          str(request.directory / "task.json")]},
                }[stage]
                (request.directory / filename).write_text(
                    R.canonical(expected) + "\n")
                result = R.run_request(
                    path, execute=True, approved_digest=value["request_digest"],
                    environment={"REPAIR_RUNTIME_LOG": str(self.runtime_log)})
                self.assertEqual(result["status"], "reconcile_required")
                calls = [json.loads(line) for line in
                         self.runtime_log.read_text().splitlines()]
                self.assertEqual(calls[-1][:4],
                                 ["--db", str(self.db), "status",
                                  "repair-candidate-a-v2"])
                self.assertFalse(any(action in call for call in calls
                                     for action in ("register-project", "enqueue", "worker")))
                # Each subcase needs an independent retained request namespace.
                for child in request.directory.iterdir():
                    if child.is_file():
                        child.unlink()
                    elif child.is_dir():
                        for nested in child.iterdir():
                            nested.unlink()
                        child.rmdir()
                self.runtime_log.unlink()

    def test_mutated_retained_stage_intent_is_rejected(self):
        path, value = self.request()
        request = R.Request(path)
        request.directory.mkdir(parents=True)
        (request.directory / "enqueue-intent.json").write_text(R.canonical({
            "kind": "research-repair-stage-intent-v1", "stage": "enqueue",
            "request_digest": value["request_digest"]}) + "\n")
        with self.assertRaisesRegex(R.RepairBridgeError, "stage intent mismatch"):
            R.run_request(path, execute=True, approved_digest=value["request_digest"],
                          environment={"REPAIR_RUNTIME_LOG": str(self.runtime_log)})
        self.assertFalse(self.runtime_log.exists())

    def test_admission_cannot_authorize_gpu_or_retry_same_scientific_attempt(self):
        path, value = self.request()
        admission = json.loads((self.root / self.admission_ref["path"]).read_text())
        for key in ("gpu_allowed", "scientific_retry_allowed"):
            admission[key] = True
            self.admission_ref = self.write("admission.json", admission)
            value["admission_ref"] = self.admission_ref
            value["request_digest"] = R.request_digest(value)
            path.write_text(R.canonical(value) + "\n")
            with self.assertRaisesRegex(R.RepairBridgeError, "GPU|retry"):
                R.run_request(path)
            admission[key] = False

    def test_output_must_be_new_and_cannot_overlap_retained_runs(self):
        path, value = self.request()
        value["output_paths"] = ["runs/attempts/candidate-a/receipt.json"]
        value["request_digest"] = R.request_digest(value)
        path.write_text(R.canonical(value) + "\n")
        with self.assertRaisesRegex(R.RepairBridgeError, "output"):
            R.run_request(path)

    def test_runtime_and_adapter_registry_bytes_are_pinned(self):
        path, _ = self.request()
        self.adapters.write_text('{"changed":true}\n')
        with self.assertRaisesRegex(R.RepairBridgeError, "adapter registry bytes"):
            R.run_request(path)

    def test_campaign_budget_is_reserved_once_and_cannot_be_reset_by_project_id(self):
        path, value = self.request()
        value["max_cost"] = 100001
        value["request_digest"] = R.request_digest(value)
        path.write_text(R.canonical(value) + "\n")
        with self.assertRaisesRegex(R.RepairBridgeError, "campaign repair budget"):
            R.run_request(path, execute=True, approved_digest=value["request_digest"])

    def test_negative_retained_budget_reservation_cannot_offset_new_work(self):
        path, value = self.request()
        directory = self.root / "runs/supervisor/campaign"
        reservation = {"kind": "research-repair-reservation", "version": 1,
                       "campaign_digest": self.campaign_digest,
                       "request_id": "old-repair", "request_digest": "d" * 64,
                       "max_cost": -1, "max_seconds": -1}
        (directory / "repair-budget.json").write_text(
            R.canonical(json.loads((self.root / self.budget_ref["path"]).read_text())) + "\n")
        receipts = directory / "repair-reservations"
        receipts.mkdir()
        (receipts / ("old-repair-" + "d" * 64 + ".json")).write_text(
            R.canonical(reservation) + "\n")
        with self.assertRaisesRegex(R.RepairBridgeError, "invalid retained reservation"):
            R.run_request(
                path, execute=True, approved_digest=value["request_digest"],
                environment={"REPAIR_RUNTIME_LOG": str(self.runtime_log)})
        calls = [json.loads(line) for line in self.runtime_log.read_text().splitlines()]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][:3], ["--db", str(self.db), "register-project"])
        self.assertFalse(any("enqueue" in call or "worker" in call for call in calls))
        records = [json.loads(item.read_text()) for item in
                   sorted((directory / "repair-requests/repair-candidate-a-v2/events").glob("*.json"))]
        self.assertEqual(records[-1]["status"], "budget_reservation_failed")

    def test_failed_receipt_must_bind_selected_campaign_plan_digest(self):
        path, value = self.request()
        self.input_ref = self.write(
            "failure-receipt.json", {"status": "failed", "plan_digest": "e" * 64})
        admission = json.loads((self.root / self.admission_ref["path"]).read_text())
        admission["failure_ref"] = self.input_ref
        self.admission_ref = self.write("admission.json", admission)
        value["admission_ref"] = self.admission_ref
        value["inputs"][0] = self.input_ref
        value["request_digest"] = R.request_digest(value)
        path.write_text(R.canonical(value) + "\n")
        with self.assertRaisesRegex(R.RepairBridgeError, "receipt.*campaign plan"):
            R.run_request(path)

    def test_two_distinct_projects_share_one_campaign_budget(self):
        path, value = self.request()
        env = {"REPAIR_RUNTIME_LOG": str(self.runtime_log)}
        self.assertEqual(R.run_request(
            path, execute=True, approved_digest=value["request_digest"],
            environment=env)["status"], "worker_returned")
        second = json.loads(path.read_text())
        second_id = "repair-candidate-a-v3"
        project = json.loads(self.project.read_text())
        project.update(project_id=second_id, max_cost=1, max_seconds=1)
        second_project = self.root / "repair-project-v3.json"
        second_project.write_text(R.canonical(project) + "\n")
        second.update(request_id=second_id, project_id=second_id,
                      project_ref={"path": second_project.name,
                                   "sha256": R.file_digest(second_project)},
                      output_paths=["repairs/campaign/repair-candidate-a-v3/patch.diff"],
                      max_cost=1, max_seconds=1)
        second["request_digest"] = R.request_digest(second)
        second_path = self.root / "request-v3.json"
        second_path.write_text(R.canonical(second) + "\n")
        with self.assertRaisesRegex(R.RepairBridgeError, "campaign repair budget"):
            R.run_request(second_path, execute=True,
                          approved_digest=second["request_digest"], environment=env)

    def test_native_adapter_is_rejected(self):
        path, value = self.request()
        value["adapter"] = "native"
        value["request_digest"] = R.request_digest(value)
        path.write_text(R.canonical(value) + "\n")
        with self.assertRaisesRegex(R.RepairBridgeError, "ACP adapter"):
            R.run_request(path)

    def test_worker_requires_request_scoped_dedicated_project(self):
        path, value = self.request()
        value["project_id"] = "shared-project"
        value["request_digest"] = R.request_digest(value)
        path.write_text(R.canonical(value) + "\n")
        with self.assertRaisesRegex(R.RepairBridgeError, "dedicated"):
            R.run_request(path)

    def test_admission_must_bind_repairable_failed_supervisor_observation(self):
        path, value = self.request()
        status = {"status": "inspection", "campaign_digest": self.campaign_digest,
                  "plans": {"candidate-a": {"status": "unknown", "repairable": False}}}
        self.status_ref = self.write("supervisor-inspection.json", status)
        admission = json.loads((self.root / self.admission_ref["path"]).read_text())
        admission["supervisor_status_ref"] = self.status_ref
        self.admission_ref = self.write("admission.json", admission)
        value["admission_ref"] = self.admission_ref
        value["inputs"][2] = self.status_ref
        value["request_digest"] = R.request_digest(value)
        path.write_text(R.canonical(value) + "\n")
        with self.assertRaisesRegex(R.RepairBridgeError, "repairable failed"):
            R.run_request(path)

    def test_builder_pins_every_existing_input_and_emits_inspectable_request(self):
        out = self.root / "built-request.json"
        result = R.build_request(
            root=self.root, runtime_executable=self.runtime, db=self.db,
            project_id="repair-candidate-a-v2", owner="local-repair-owner", adapter="codex",
            adapters_config=self.adapters, project=self.project,
            admission=self.root / self.admission_ref["path"],
            inputs=[self.root / self.input_ref["path"], self.root / self.plan_ref["path"],
                    self.root / self.status_ref["path"], self.root / self.budget_ref["path"],
                    self.root / self.campaign_ref["path"]],
            instructions=[self.root / self.instruction_ref["path"]],
            output_paths=["repairs/campaign/repair-candidate-a-v2/patch.diff"], max_cost=100000,
            max_seconds=900, request_id="repair-candidate-a-v2", out=out)
        self.assertEqual(result["request_digest"], R.request_digest(result))
        self.assertEqual(R.run_request(out)["task"]["action"], "candidate_code")

    def test_builder_validation_failure_leaves_no_request(self):
        out = self.root / "invalid-request.json"
        with self.assertRaises(R.RepairBridgeError):
            R.build_request(
                root=self.root, runtime_executable=self.runtime, db=self.db,
                project_id="repair-candidate-a-v2", owner="local-repair-owner", adapter="codex",
                adapters_config=self.adapters, project=self.project,
                admission=self.root / self.admission_ref["path"],
                inputs=[self.root / self.input_ref["path"], self.root / self.plan_ref["path"],
                        self.root / self.status_ref["path"], self.root / self.budget_ref["path"],
                        self.root / self.campaign_ref["path"]],
                instructions=[self.root / self.instruction_ref["path"]],
                output_paths=["runs/attempts/forbidden"], max_cost=100000,
                max_seconds=900, request_id="repair-candidate-a-v2", out=out)
        self.assertFalse(out.exists())

    def test_completed_output_does_not_break_exact_request_reconciliation(self):
        path, _ = self.request()
        output = self.root / "repairs/campaign/repair-candidate-a-v2/patch.diff"
        output.parent.mkdir(parents=True)
        output.write_text("retained child patch")
        result = R.run_request(path)
        self.assertEqual(result["status"], "inspection")
        self.assertEqual(result["task"]["output_paths"][0],
                         "repairs/campaign/repair-candidate-a-v2/patch.diff")

    def test_first_execute_rejects_preexisting_output_without_retained_task(self):
        path, value = self.request()
        output = self.root / value["output_paths"][0]
        output.parent.mkdir(parents=True)
        output.write_text("unowned bytes")
        with self.assertRaisesRegex(R.RepairBridgeError, "unowned repair output"):
            R.run_request(path, execute=True, approved_digest=value["request_digest"])

    def test_task_without_worker_intent_does_not_adopt_preexisting_output(self):
        path, value = self.request()
        request = R.Request(path)
        request.directory.mkdir(parents=True)
        (request.directory / "request.json").write_text(R.canonical(value) + "\n")
        (request.directory / "task.json").write_text(R.canonical(request.task) + "\n")
        output = self.root / value["output_paths"][0]
        output.parent.mkdir(parents=True)
        output.write_text("unowned bytes")
        with self.assertRaisesRegex(R.RepairBridgeError, "unowned repair output"):
            R.run_request(path, execute=True, approved_digest=value["request_digest"])

    def test_retained_task_symlink_is_rejected(self):
        path, value = self.request()
        request = R.Request(path)
        request.directory.mkdir(parents=True)
        (request.directory / "request.json").write_text(R.canonical(value) + "\n")
        (request.directory / "task.json").symlink_to(path)
        with self.assertRaisesRegex(R.RepairBridgeError, "regular retained"):
            R.run_request(path, execute=True, approved_digest=value["request_digest"])


if __name__ == "__main__":
    unittest.main()
