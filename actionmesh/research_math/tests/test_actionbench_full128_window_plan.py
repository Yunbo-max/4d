"""Fail-closed software checks for the priced Full128 harness compiler.

Web authors these checks without executing them.  Local must run them through
the admitted software-acceptance harness before any generated window is used.
"""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from prepare_actionbench_full128_window import (
    R9_CAMPAIGN_PLAN_PATH,
    R9_STATE_SNAPSHOT_PATH,
    R9_STATUS_SNAPSHOT_PATH,
    _plan_digest,
    build_input_ref_closure,
    build_native_command,
    native_limits,
    select_priced_window,
    validate_active_batch_reconciliation,
    validate_environment_closure,
    validate_native_plan_set,
    window_limits,
)


def pricing_fixture():
    uids = [f"uid-{index:03d}" for index in range(128)]
    windows = []
    for number, start in enumerate(range(0, 128, 16), 1):
        selected = uids[start:start + 16]
        windows.append({
            "window_id": f"full128-window-{number:02d}",
            "population_start_index": start,
            "population_stop_index_exclusive": start + len(selected),
            "uids": selected,
            "unit_count": len(selected),
            "planned_workload_seconds": len(selected) * 1664,
        })
    return {
        "kind": "actionbench-full128-queue-pricing",
        "version": "1.0.0",
        "status": "priced_engineering_only",
        "population": {"dataset": "facebook/actionbench", "revision": "d" * 40,
                       "size": 128},
        "population_uid_sha256": hashlib.sha256(json.dumps(
            uids, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False).encode()).hexdigest(),
        "hard_window_seconds": 28800,
        "collection_reserve_seconds": 1800,
        "workload_budget_seconds": 27000,
        "unit_timeout_seconds": 1664,
        "units_per_window": 16,
        "window_count": 8,
        "planned_workload_seconds_per_full_window": 26624,
        "unallocated_workload_seconds_per_full_window": 376,
        "windows": windows,
        "queue_priced": True,
        "queue_approved": False,
        "queue_generated": False,
        "dispatch_ready": False,
        "scientific_effect_qualification": False,
        "native_scientific_qualification": False,
        "candidate_methods_tested": False,
    }


def evidence_fixture(root, pricing):
    pricing_path = root / "inputs/actionbench-full128-queue/pricing.json"
    pricing_path.parent.mkdir(parents=True, exist_ok=True)
    pricing_path.write_text(json.dumps(pricing))
    campaign_path = root / R9_CAMPAIGN_PLAN_PATH
    campaign_path.parent.mkdir(parents=True, exist_ok=True)
    archive_root = campaign_path.parent.parent.parent
    tasks = []
    native_paths = []
    for index in range(1, 10):
        relative = Path(
            f"plans/population-gpu-current-r9/{index:03d}/native.json")
        native_path = archive_root / relative
        native_path.parent.mkdir(parents=True, exist_ok=True)
        native = {
            "schema_id": "experiment-run-plan",
            "schema_version": "1.0.0",
            "run_id": f"population-gpu-current-r9-unit-{index:03d}",
            "purpose": "engineering",
            "evidence_mode": "developmental",
            "jobs": [],
        }
        native["plan_digest"] = _plan_digest(native)
        native_path.write_text(json.dumps(native))
        native_paths.append(native_path)
        tasks.append({
            "task_id": f"population-{index:03d}",
            "plan_ref": {
                "path": relative.as_posix(),
                "sha256": hashlib.sha256(native_path.read_bytes()).hexdigest(),
            },
        })
    campaign = {
        "schema_id": "harness-plan",
        "schema_version": "1.0.0",
        "batch_id": "population-gpu-current-r9",
        "tasks": tasks,
    }
    campaign["plan_digest"] = _plan_digest(campaign)
    campaign_path.write_text(json.dumps(campaign))
    status_path = root / R9_STATUS_SNAPSHOT_PATH
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps({
        "population_window": {
            "run_id": "population-gpu-current-r9",
            "plan_digest": campaign["plan_digest"],
            "indices": list(range(1, 10)),
            "observed_at": "2026-10-07T16:35:40Z",
            "completed": 3,
            "running": 1,
            "pending": 5,
            "failed": 0,
        },
    }))
    state_path = root / R9_STATE_SNAPSHOT_PATH
    state_path.parent.mkdir(parents=True, exist_ok=True)
    statuses = ["completed"] * 3 + ["running"] + ["pending"] * 5
    state_path.write_text(json.dumps({
        "format": "research-harness-state-v1",
        "batch_id": "population-gpu-current-r9",
        "plan_digest": campaign["plan_digest"],
        "status": "running",
        "tasks": {
            f"population-{index:03d}": {"status": task_status}
            for index, task_status in zip(range(1, 10), statuses)
        },
    }))
    refs = tuple({
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    } for path in (pricing_path, campaign_path, status_path))
    profile = {
        "run_id": "population-gpu-current-r9",
        "population_range": (1, 10),
        "plan_digest": campaign["plan_digest"],
        "campaign_plan_path": R9_CAMPAIGN_PLAN_PATH,
        "status_snapshot_path": R9_STATUS_SNAPSHOT_PATH,
        "state_snapshot_path": R9_STATE_SNAPSHOT_PATH,
    }
    return (*refs, profile, native_paths)


def reconciliation_fixture(pricing, pricing_ref, campaign_ref, status_ref,
                           profile):
    statuses = ["completed"] * 3 + ["running"] + ["pending"] * 5
    state = {
        "format": "research-harness-state-v1",
        "batch_id": "population-gpu-current-r9",
        "plan_digest": profile["plan_digest"],
        "status": "running",
        "tasks": {
            f"population-{index:03d}": {"status": task_status}
            for index, task_status in zip(range(1, 10), statuses)
        },
    }
    return {
        "kind": "actionbench-full128-active-batch-reconciliation",
        "version": "1.0.0",
        "status": "reconciled_for_plan_generation",
        "pricing_ref": pricing_ref,
        "population_uid_sha256": pricing["population_uid_sha256"],
        "observed_at": "2026-10-07T16:35:40Z",
        "source_runs": [{
            "run_id": "population-gpu-current-r9",
            "plan_digest": profile["plan_digest"],
            "population_start_index": 1,
            "population_stop_index_exclusive": 10,
            "campaign_plan_ref": campaign_ref,
            "status_snapshot_ref": status_ref,
            "state_snapshot_ref": {
                "path": profile["state_snapshot_path"],
                "sha256": hashlib.sha256(json.dumps(state).encode()).hexdigest(),
            },
            "dispositions": [{
                "population_index": index,
                "uid": f"uid-{index:03d}",
                "status": "completed" if index < 4 else (
                    "running" if index == 4 else "pending"),
            } for index in range(1, 10)],
        }],
        "queue_approved": False,
        "dispatch_ready": False,
        "scientific_effect_qualification": False,
        "candidate_methods_tested": False,
    }


class Full128WindowPlanTests(unittest.TestCase):
    def test_snapshot_builder_freezes_stable_state_and_status_bytes(self):
        """Catch reconciliation evidence that still points at mutable inputs."""
        from tempfile import TemporaryDirectory
        from prepare_actionbench_active_batch_snapshot import capture_snapshot
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            live = root / "live"
            live.mkdir()
            profile = {
                "run_id": "population-gpu-current-r9",
                "population_range": (1, 10),
                "plan_digest": "a" * 64,
                "campaign_plan_path": "archive/harness.json",
                "status_snapshot_path": "archive/status-snapshot.json",
                "state_snapshot_path": "archive/state.json",
            }
            statuses = ["completed"] * 3 + ["running"] + ["pending"] * 5
            state = {
                "format": "research-harness-state-v1",
                "batch_id": profile["run_id"],
                "plan_digest": profile["plan_digest"],
                "status": "running",
                "tasks": {
                    f"population-{index:03d}": {"status": task_status}
                    for index, task_status in zip(range(1, 10), statuses)
                },
            }
            status = {"population_window": {
                "run_id": profile["run_id"],
                "plan_digest": profile["plan_digest"],
                "indices": list(range(1, 10)),
                "observed_at": "2026-10-07T17:50:33Z",
                "completed": 3, "running": 1, "pending": 5, "failed": 0,
            }}
            live_state = live / "state.json"
            live_status = live / "STATUS.json"
            live_state.write_text(json.dumps(state))
            live_status.write_text(json.dumps(status))
            state_output = root / profile["state_snapshot_path"]
            status_output = root / profile["status_snapshot_path"]

            result = capture_snapshot(
                root=root,
                live_state_path=live_state,
                live_status_path=live_status,
                state_output_path=state_output,
                status_output_path=status_output,
                required_profile=profile,
            )

            self.assertEqual(state_output.read_bytes(), live_state.read_bytes())
            self.assertEqual(status_output.read_bytes(), live_status.read_bytes())
            self.assertEqual(result["counts"], {
                "completed": 3, "running": 1, "pending": 5, "failed": 0,
            })
            live_status.write_text("{}")
            self.assertEqual(json.loads(status_output.read_text()), status)

    def test_snapshot_builder_accepts_official_harness_status_observation(self):
        """Preserve the harness's read-only --status output without synthesizing STATUS."""
        from tempfile import TemporaryDirectory
        from prepare_actionbench_active_batch_snapshot import capture_snapshot
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            live = root / "live"
            live.mkdir()
            profile = {
                "run_id": "population-gpu-current-r9",
                "population_range": (1, 10),
                "plan_digest": "a" * 64,
                "campaign_plan_path": "archive/harness.json",
                "status_snapshot_path": "archive/status-snapshot.json",
                "state_snapshot_path": "archive/state.json",
            }
            statuses = ["completed"] * 3 + ["running"] + ["pending"] * 5
            state = {
                "format": "research-harness-state-v1",
                "batch_id": profile["run_id"],
                "plan_digest": profile["plan_digest"],
                "status": "running",
                "tasks": {
                    f"population-{index:03d}": {"status": task_status}
                    for index, task_status in zip(range(1, 10), statuses)
                },
            }
            observed = dict(state)
            observed.update({
                "observed_at": "2026-10-07T19:12:37.209259Z",
                "status_is_retained_observation": True,
            })
            live_state = live / "state.json"
            live_status = live / "harness-status.stdout.json"
            live_state.write_text(json.dumps(state))
            live_status.write_text(json.dumps(observed))
            state_output = root / profile["state_snapshot_path"]
            status_output = root / profile["status_snapshot_path"]

            result = capture_snapshot(
                root=root,
                live_state_path=live_state,
                live_status_path=live_status,
                state_output_path=state_output,
                status_output_path=status_output,
                required_profile=profile,
            )

            self.assertEqual(status_output.read_bytes(), live_status.read_bytes())
            self.assertEqual(result["observed_at"], observed["observed_at"])
            self.assertEqual(result["counts"], {
                "completed": 3, "running": 1, "pending": 5, "failed": 0,
            })

    def test_snapshot_builder_rejects_incoherent_counts_without_partial_output(self):
        """Catch archiving a state/status pair from different observations."""
        from tempfile import TemporaryDirectory
        from prepare_actionbench_active_batch_snapshot import capture_snapshot
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            live = root / "live"
            live.mkdir()
            profile = {
                "run_id": "population-gpu-current-r9",
                "population_range": (1, 3),
                "plan_digest": "a" * 64,
                "campaign_plan_path": "archive/harness.json",
                "status_snapshot_path": "archive/status-snapshot.json",
                "state_snapshot_path": "archive/state.json",
            }
            live_state = live / "state.json"
            live_status = live / "STATUS.json"
            live_state.write_text(json.dumps({
                "format": "research-harness-state-v1",
                "batch_id": profile["run_id"],
                "plan_digest": profile["plan_digest"],
                "tasks": {
                    "population-001": {"status": "completed"},
                    "population-002": {"status": "pending"},
                },
            }))
            live_status.write_text(json.dumps({"population_window": {
                "run_id": profile["run_id"],
                "plan_digest": profile["plan_digest"],
                "indices": [1, 2],
                "observed_at": "2026-10-07T17:50:33Z",
                "completed": 2, "running": 0, "pending": 0, "failed": 0,
            }}))
            state_output = root / profile["state_snapshot_path"]
            status_output = root / profile["status_snapshot_path"]

            with self.assertRaisesRegex(ValueError, "counts"):
                capture_snapshot(
                    root=root,
                    live_state_path=live_state,
                    live_status_path=live_status,
                    state_output_path=state_output,
                    status_output_path=status_output,
                    required_profile=profile,
                )

            self.assertFalse(state_output.exists())
            self.assertFalse(status_output.exists())

    def test_reconciliation_builder_derives_every_disposition_from_harness_state(self):
        """Catch a builder that copies unverified caller-supplied statuses."""
        from tempfile import TemporaryDirectory
        try:
            from prepare_actionbench_active_batch_reconciliation import (
                build_reconciliation,
            )
        except ModuleNotFoundError as error:
            self.fail(f"reconciliation builder is missing: {error}")
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            pricing = pricing_fixture()
            (pricing_ref, campaign_ref, status_ref, profile,
             _) = evidence_fixture(root, pricing)
            output = root / (
                "inputs/actionbench-full128-queue/"
                "active-batch-reconciliation.json")

            built = build_reconciliation(
                root=root,
                pricing_path=root / pricing_ref["path"],
                campaign_path=root / campaign_ref["path"],
                state_path=root / profile["state_snapshot_path"],
                status_path=root / status_ref["path"],
                output_path=output,
                required_profile=profile,
            )

            self.assertEqual(
                [row["status"] for row in built["source_runs"][0]["dispositions"]],
                ["completed"] * 3 + ["running"] + ["pending"] * 5,
            )
            self.assertEqual(json.loads(output.read_text()), built)

    def test_reconciliation_accepts_official_harness_status_unchanged(self):
        """Bind native --status output without inventing a STATUS wrapper."""
        from tempfile import TemporaryDirectory
        from prepare_actionbench_active_batch_reconciliation import (
            build_reconciliation,
        )
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            pricing = pricing_fixture()
            (_, campaign_ref, status_ref, profile, _) = evidence_fixture(
                root, pricing)
            state_path = root / profile["state_snapshot_path"]
            status_path = root / status_ref["path"]
            state = json.loads(state_path.read_text())
            observed = dict(state)
            observed.update({
                "observed_at": "2026-10-07T19:12:37.209259Z",
                "status_is_retained_observation": True,
            })
            raw_status = json.dumps(observed).encode()
            status_path.write_bytes(raw_status)
            output = root / (
                "inputs/actionbench-full128-queue/"
                "active-batch-reconciliation.json")

            built = build_reconciliation(
                root=root,
                pricing_path=root / "inputs/actionbench-full128-queue/pricing.json",
                campaign_path=root / campaign_ref["path"],
                state_path=state_path,
                status_path=status_path,
                output_path=output,
                required_profile=profile,
            )

            self.assertEqual(built["observed_at"], observed["observed_at"])
            self.assertEqual(status_path.read_bytes(), raw_status)

    def test_official_harness_status_must_match_the_exact_state(self):
        """Reject a marked observer payload whose task record was altered."""
        from tempfile import TemporaryDirectory
        from prepare_actionbench_full128_window import (
            retained_observation_window,
        )
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            pricing = pricing_fixture()
            (_, _, _, profile, _) = evidence_fixture(root, pricing)
            state = json.loads((root / profile["state_snapshot_path"]).read_text())
            observed = dict(state)
            observed.update({
                "observed_at": "2026-10-07T19:12:37.209259Z",
                "status_is_retained_observation": True,
            })
            observed["tasks"] = dict(observed["tasks"])
            observed["tasks"]["population-001"] = {"status": "failed"}

            with self.assertRaisesRegex(ValueError, "differs from harness state"):
                retained_observation_window(observed, state, profile)

    def test_active_batch_reconciliation_is_derived_from_exact_harness_state(self):
        """Catch accepting caller-authored dispositions without task state."""
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            pricing = pricing_fixture()
            (pricing_ref, campaign_ref, status_ref, profile,
             _) = evidence_fixture(root, pricing)
            state_path = root / "docs/archive/runs/harness/population-r9/state.json"
            state_path.parent.mkdir(parents=True)
            statuses = ["completed"] * 3 + ["running"] + ["pending"] * 5
            state_path.write_text(json.dumps({
                "format": "research-harness-state-v1",
                "batch_id": "population-gpu-current-r9",
                "plan_digest": profile["plan_digest"],
                "status": "running",
                "tasks": {
                    f"population-{index:03d}": {"status": status}
                    for index, status in zip(range(1, 10), statuses)
                },
            }))
            state_ref = {
                "path": state_path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(),
            }
            profile["state_snapshot_path"] = state_ref["path"]
            reconciliation = reconciliation_fixture(
                pricing, pricing_ref, campaign_ref, status_ref, profile)
            reconciliation["source_runs"][0]["state_snapshot_ref"] = state_ref

            try:
                refs = validate_active_batch_reconciliation(
                    root, pricing, reconciliation, pricing["windows"][1], profile)
            except ValueError as error:
                self.fail(f"valid harness state snapshot was rejected: {error}")

            self.assertIn(state_path.resolve(), refs)

    def test_active_batch_reconciliation_rejects_per_task_state_swaps(self):
        """Catch aggregate-count agreement masking wrong task dispositions."""
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            pricing = pricing_fixture()
            (pricing_ref, campaign_ref, status_ref, profile,
             _) = evidence_fixture(root, pricing)
            reconciliation = reconciliation_fixture(
                pricing, pricing_ref, campaign_ref, status_ref, profile)
            rows = reconciliation["source_runs"][0]["dispositions"]
            rows[0]["status"], rows[3]["status"] = (
                rows[3]["status"], rows[0]["status"])

            with self.assertRaisesRegex(
                    ValueError, "exact retained harness task state"):
                validate_active_batch_reconciliation(
                    root, pricing, reconciliation, pricing["windows"][1], profile)

    def test_archived_reconciliation_evidence_is_staged_without_root_rebinding(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            archived = root / "docs/archive/plans/r9/harness.json"
            archived.parent.mkdir(parents=True)
            archived.write_text(json.dumps({
                "plan_ref": {"path": "plans/r9/001/native.json",
                             "sha256": "a" * 64},
            }))
            refs = build_input_ref_closure(root, [], [archived])
            self.assertEqual(refs, [{
                "path": "docs/archive/plans/r9/harness.json",
                "sha256": hashlib.sha256(archived.read_bytes()).hexdigest(),
            }])
            self.assertFalse((root / "plans/r9/001/native.json").exists())

    def test_active_batch_reconciliation_allows_only_nonoverlapping_window(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            pricing = pricing_fixture()
            (pricing_ref, campaign_ref, status_ref, profile,
             native_paths) = evidence_fixture(root, pricing)
            reconciliation = reconciliation_fixture(
                pricing, pricing_ref, campaign_ref, status_ref, profile)
            refs = validate_active_batch_reconciliation(
                root, pricing, reconciliation, pricing["windows"][1], profile)
            self.assertEqual(refs, sorted({
                (root / pricing_ref["path"]).resolve(),
                (root / campaign_ref["path"]).resolve(),
                (root / status_ref["path"]).resolve(),
                (root / profile["state_snapshot_path"]).resolve(),
                *(path.resolve() for path in native_paths),
            }))
            with self.assertRaisesRegex(ValueError, "overlaps retained work"):
                validate_active_batch_reconciliation(
                    root, pricing, reconciliation, pricing["windows"][0], profile)
            reconciliation["source_runs"][0]["dispositions"][0]["status"] = "failed"
            status_path = root / status_ref["path"]
            status = json.loads(status_path.read_text())
            status["population_window"]["completed"] = 2
            status["population_window"]["failed"] = 1
            status_path.write_text(json.dumps(status))
            reconciliation["source_runs"][0]["status_snapshot_ref"] = {
                "path": status_ref["path"],
                "sha256": hashlib.sha256(status_path.read_bytes()).hexdigest(),
            }
            state_path = root / profile["state_snapshot_path"]
            state = json.loads(state_path.read_text())
            state["tasks"]["population-001"]["status"] = "failed"
            state_path.write_text(json.dumps(state))
            reconciliation["source_runs"][0]["state_snapshot_ref"] = {
                "path": profile["state_snapshot_path"],
                "sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(),
            }
            with self.assertRaisesRegex(ValueError, "overlaps retained work"):
                validate_active_batch_reconciliation(
                    root, pricing, reconciliation, pricing["windows"][0], profile)

    def test_active_batch_reconciliation_requires_r9_and_complete_range(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            pricing = pricing_fixture()
            (pricing_ref, campaign_ref, status_ref, profile,
             _) = evidence_fixture(root, pricing)
            valid = reconciliation_fixture(
                pricing, pricing_ref, campaign_ref, status_ref, profile)
            validate_active_batch_reconciliation(
                root, pricing, valid, pricing["windows"][1], profile)
            for mutation in ("missing-run", "missing-index", "duplicate-index",
                             "shifted-range", "short-range"):
                reconciliation = reconciliation_fixture(
                    pricing, pricing_ref, campaign_ref, status_ref, profile)
                if mutation == "missing-run":
                    reconciliation["source_runs"] = []
                elif mutation == "missing-index":
                    reconciliation["source_runs"][0]["dispositions"].pop()
                else:
                    run = reconciliation["source_runs"][0]
                    if mutation == "duplicate-index":
                        run["dispositions"][-1] = dict(run["dispositions"][0])
                    elif mutation == "shifted-range":
                        run["population_start_index"] = 16
                        run["population_stop_index_exclusive"] = 25
                        run["dispositions"] = [{
                            "population_index": index,
                            "uid": f"uid-{index:03d}",
                            "status": "pending",
                        } for index in range(16, 25)]
                    else:
                        run["population_stop_index_exclusive"] = 9
                        run["dispositions"].pop()
                with self.subTest(mutation=mutation), self.assertRaisesRegex(
                        ValueError,
                        "exactly the retained r9|Known active batch|"
                        "Complete source-run|Duplicate retained|"
                        "Exact retained r9 range"):
                    validate_active_batch_reconciliation(
                        root, pricing, reconciliation, pricing["windows"][1],
                        profile)

    def test_active_batch_reconciliation_rejects_identity_or_claim_drift(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            pricing = pricing_fixture()
            (pricing_ref, campaign_ref, status_ref, profile,
             _) = evidence_fixture(root, pricing)
            cases = (
                ("population_uid_sha256", "b" * 64),
                ("queue_approved", True),
                ("dispatch_ready", True),
                ("candidate_methods_tested", True),
            )
            for key, value in cases:
                reconciliation = reconciliation_fixture(
                    pricing, pricing_ref, campaign_ref, status_ref, profile)
                reconciliation[key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    validate_active_batch_reconciliation(
                        root, pricing, reconciliation, pricing["windows"][1],
                        profile)

    def test_rejects_unbound_campaign_status_or_plan_digest(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            pricing = pricing_fixture()
            (pricing_ref, campaign_ref, status_ref, profile,
             native_paths) = evidence_fixture(root, pricing)
            for mutation in ("campaign-path", "status-path", "plan-digest",
                             "status-counts"):
                reconciliation = reconciliation_fixture(
                    pricing, pricing_ref, campaign_ref, status_ref, profile)
                run = reconciliation["source_runs"][0]
                if mutation == "campaign-path":
                    run["campaign_plan_ref"] = pricing_ref
                elif mutation == "status-path":
                    run["status_snapshot_ref"] = pricing_ref
                elif mutation == "plan-digest":
                    run["plan_digest"] = "b" * 64
                else:
                    status_path = root / status_ref["path"]
                    status = json.loads(status_path.read_text())
                    status["population_window"]["completed"] = 2
                    status_path.write_text(json.dumps(status))
                    run["status_snapshot_ref"] = {
                        "path": status_ref["path"],
                        "sha256": hashlib.sha256(status_path.read_bytes()).hexdigest(),
                    }
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    validate_active_batch_reconciliation(
                        root, pricing, reconciliation, pricing["windows"][1],
                        profile)

    def test_rejects_unhashed_or_changed_retained_native_plans(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            pricing = pricing_fixture()
            (pricing_ref, campaign_ref, status_ref, profile,
             native_paths) = evidence_fixture(root, pricing)
            reconciliation = reconciliation_fixture(
                pricing, pricing_ref, campaign_ref, status_ref, profile)
            native_paths[0].write_text("{}")
            with self.assertRaisesRegex(ValueError, "Changed pinned file"):
                validate_active_batch_reconciliation(
                    root, pricing, reconciliation, pricing["windows"][1], profile)

            (pricing_ref, campaign_ref, status_ref, profile,
             _) = evidence_fixture(root, pricing)
            reconciliation = reconciliation_fixture(
                pricing, pricing_ref, campaign_ref, status_ref, profile)
            campaign_path = root / campaign_ref["path"]
            campaign = json.loads(campaign_path.read_text())
            campaign["tasks"][0]["plan_ref"].pop("sha256")
            campaign["plan_digest"] = _plan_digest(campaign)
            campaign_path.write_text(json.dumps(campaign))
            profile["plan_digest"] = campaign["plan_digest"]
            reconciliation["source_runs"][0]["plan_digest"] = campaign["plan_digest"]
            reconciliation["source_runs"][0]["campaign_plan_ref"] = {
                "path": campaign_ref["path"],
                "sha256": hashlib.sha256(campaign_path.read_bytes()).hexdigest(),
            }
            status_path = root / status_ref["path"]
            status = json.loads(status_path.read_text())
            status["population_window"]["plan_digest"] = campaign["plan_digest"]
            status_path.write_text(json.dumps(status))
            reconciliation["source_runs"][0]["status_snapshot_ref"] = {
                "path": status_ref["path"],
                "sha256": hashlib.sha256(status_path.read_bytes()).hexdigest(),
            }
            with self.assertRaisesRegex(ValueError, "hash-bound retained native"):
                validate_active_batch_reconciliation(
                    root, pricing, reconciliation, pricing["windows"][1], profile)

    def test_rejects_campaign_digest_tampering_or_additional_source_runs(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            pricing = pricing_fixture()
            (pricing_ref, campaign_ref, status_ref, profile,
             _) = evidence_fixture(root, pricing)
            reconciliation = reconciliation_fixture(
                pricing, pricing_ref, campaign_ref, status_ref, profile)
            campaign_path = root / campaign_ref["path"]
            campaign = json.loads(campaign_path.read_text())
            campaign["unbound_field"] = "tamper"
            campaign_path.write_text(json.dumps(campaign))
            reconciliation["source_runs"][0]["campaign_plan_ref"] = {
                "path": campaign_ref["path"],
                "sha256": hashlib.sha256(campaign_path.read_bytes()).hexdigest(),
            }
            with self.assertRaisesRegex(ValueError, "Campaign plan does not bind"):
                validate_active_batch_reconciliation(
                    root, pricing, reconciliation, pricing["windows"][1], profile)

            (pricing_ref, campaign_ref, status_ref, profile,
             _) = evidence_fixture(root, pricing)
            reconciliation = reconciliation_fixture(
                pricing, pricing_ref, campaign_ref, status_ref, profile)
            reconciliation["source_runs"].append(
                dict(reconciliation["source_runs"][0]))
            with self.assertRaisesRegex(ValueError, "exactly the retained r9"):
                validate_active_batch_reconciliation(
                    root, pricing, reconciliation, pricing["windows"][1], profile)

    def test_selects_one_exact_priced_window_without_changing_order(self):
        pricing = pricing_fixture()
        selected = select_priced_window(pricing, "full128-window-02")
        self.assertEqual(selected, pricing["windows"][1])
        self.assertEqual(selected["uids"], [f"uid-{index:03d}" for index in range(16, 32)])

    def test_rejects_approval_dispatch_or_noncanonical_budget_claims(self):
        for key, value in (("queue_approved", True), ("queue_generated", True),
                           ("dispatch_ready", True), ("unit_timeout_seconds", 1663),
                           ("workload_budget_seconds", 26999),
                           ("hard_window_seconds", 28799),
                           ("collection_reserve_seconds", 1799)):
            pricing = pricing_fixture()
            pricing[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                select_priced_window(pricing, "full128-window-01")

    def test_rejects_missing_reordered_or_overbudget_window(self):
        cases = []
        missing = pricing_fixture()
        missing["windows"] = missing["windows"][1:]
        missing["window_count"] = 7
        cases.append(missing)
        reordered = pricing_fixture()
        reordered["windows"][0]["uids"] = list(reversed(reordered["windows"][0]["uids"]))
        cases.append(reordered)
        over = pricing_fixture()
        over["windows"][0]["planned_workload_seconds"] = 27001
        cases.append(over)
        for pricing in cases:
            with self.assertRaises(ValueError):
                select_priced_window(pricing, "full128-window-01")

    def test_outer_limits_retain_collection_reserve_and_one_gpu_task_at_a_time(self):
        pricing = pricing_fixture()
        window = select_priced_window(pricing, "full128-window-01")
        self.assertEqual(window_limits(pricing, window), {
            "total_wall_seconds": 27000,
            "window_seconds": 28800,
            "max_parallel_tasks": 1,
            "cpu_cores": 8,
            "ram_mib": 32768,
            "max_gpu_task_seconds": 26624,
        })

    def test_native_command_binds_uid_window_price_and_no_fallback(self):
        root = Path("/project")
        args = SimpleNamespace(
            root=root,
            pricing=root / "inputs/actionbench-full128-queue/pricing.json",
            contract=root / "contract.json", population=root / "population.json",
            snapshot_contract=root / "snapshot-contract.json",
            snapshot_admission=root / "snapshot-admission.json",
            dataset_semantics=root / "dataset-semantics.json",
            unit_manifest=root / "unit-manifest.json",
            source_root=root / "source", dataset_root=root / "dataset",
            weights_root=root / "weights", gpu_uuid="GPU-test",
        )
        command = build_native_command(args, "uid-000", "full128-window-01", 1664)
        joined = " ".join(command)
        for fragment in ("--root ..", "--pricing ../inputs/actionbench-full128-queue/pricing.json",
                         "--uid uid-000", "--window-id full128-window-01",
                         "--wall-seconds 1664", "--gpu-uuid GPU-test"):
            self.assertIn(fragment, joined)
        for forbidden in ("--fast", "--low-ram", "--dtype", "--retry"):
            self.assertNotIn(forbidden, command)

    def test_native_command_uses_only_staged_paths_for_repository_inputs(self):
        root = Path("/project")
        args = SimpleNamespace(
            root=root,
            pricing=root / "inputs/actionbench-full128-queue/pricing.json",
            contract=root / "docs/contract.json", population=root / "assets/population.json",
            snapshot_contract=root / "docs/snapshot-contract.json",
            snapshot_admission=root / "inputs/snapshot-admission.json",
            dataset_semantics=root / "inputs/dataset-semantics.json",
            unit_manifest=root / "inputs/unit-manifest.json",
            source_root=Path("/native/source"), dataset_root=Path("/native/dataset"),
            weights_root=Path("/native/weights"), gpu_uuid="GPU-test",
        )
        command = build_native_command(args, "uid-000", "full128-window-01", 1664)
        self.assertNotIn("/project", " ".join(command))
        for expected in ("../docs/contract.json", "../assets/population.json",
                         "../inputs/snapshot-admission.json"):
            self.assertIn(expected, command)
        for external in ("/native/source", "/native/dataset", "/native/weights"):
            self.assertIn(external, command)

    def test_environment_closure_requires_exact_current_manifest_and_lock(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = root / "inputs/native-runtime"
            inputs.mkdir(parents=True)
            packages = {name: "1" for name in
                        ("numpy", "torch", "trimesh", "scipy", "pytorch3d")}
            dependency = {
                "conda_packages": [{"name": "python"}], "conda_prefix": "/conda",
                "kind": "installed-native-dependency-inventory", "packages": packages,
                "python_executable": __import__("sys").executable,
                "python_prefix": "/env", "python_version": "3.12",
                "scope": "observed", "version": "1.0.0",
            }
            dependency_path = inputs / "dependencies.json"
            dependency_path.write_text(json.dumps(dependency))
            digest = hashlib.sha256(dependency_path.read_bytes()).hexdigest()
            environment = {
                "execution_mode": "native_host",
                "python_executable": __import__("sys").executable,
                "python_version": "3.12", "python_prefix": "/env",
                "conda_prefix": "/conda", "gpu_uuid": "GPU-current",
                "packages": packages,
                "dependency_lock_refs": [{"path": "inputs/native-runtime/dependencies.json",
                                          "sha256": digest}],
                "captured_at": "2026-10-07T00:00:00+00:00",
                "gpu_identity_source": "controller", "gpu_identity_verified": False,
                "native_contract_qualified": False, "scope": "metadata",
            }
            environment_path = inputs / "environment.json"
            environment_path.write_text(json.dumps(environment))
            self.assertEqual(validate_environment_closure(
                root, environment_path, environment, "GPU-current"),
                [environment_path.resolve(), dependency_path.resolve()])
            environment["extra"] = True
            with self.assertRaises(ValueError):
                validate_environment_closure(
                    root, environment_path, environment, "GPU-current")

    def test_sixteen_native_plans_are_ordered_and_disable_retry_or_fallback(self):
        window = pricing_fixture()["windows"][0]
        plans = []
        for index, uid in enumerate(window["uids"]):
            plans.append({
                "limits": native_limits(1664),
                "jobs": [{
                    "trial_id": f"full128-{index:03d}-{uid}",
                    "cwd": "actionmesh",
                    "command": ["python", "-m", "runner", "--uid", uid],
                }],
            })
        validate_native_plan_set(window, plans, 1664)
        plans[7]["limits"]["max_retries_per_trial"] = 1
        with self.assertRaises(ValueError):
            validate_native_plan_set(window, plans, 1664)


if __name__ == "__main__":
    unittest.main()
