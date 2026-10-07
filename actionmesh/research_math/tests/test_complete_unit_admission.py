import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest

from research_math.complete_unit_admission import (
    INCIDENTAL_PYTHON_CACHE_PATHS,
    object_digest,
    validate_admission_contract,
    validate_completed_result,
    validate_execution_binding,
    validate_official_report,
    validate_telemetry,
    verify_output_closure,
)
from research_math.complete_unit_contract import complete_unit_output_paths
from research_math.control_scoring import file_ref


UID = "000-000_03b69da8d2c94b5999bcf2605ee2ecd9"


def completed_result():
    scores = {
        arm: {"cd_3d": 0.1, "cd_4d": 0.2, "cd_motion": 0.3}
        for arm in ("native", "world_gaussian", "body_gaussian")
    }
    return {
        "status": "completed",
        "scientific_effect_qualification": False,
        "candidate_methods_tested": False,
        "runtime_profile": "fp16-lowram-v1",
        "gpu_uuid": "GPU-test",
        "stages": {
            name: {"status": "completed", "elapsed_seconds": seconds}
            for name, seconds in (("generation", 100.0), ("export", 2.0),
                                  ("controls", 3.0), ("official_scoring", 40.0))
        },
        "scores": scores,
        "elapsed_seconds": 150.0,
        "device_memory": {
            "sample_interval_seconds": 1, "samples": 151,
            "observed_peak_mib": 10255.0, "exact_peak": False, "errors": [],
        },
        "host_resources": {
            "sample_interval_seconds": 1, "samples": 151,
            "observed_peak_rss_bytes": 12_000_000_000,
            "observed_peak_output_bytes": 4_000_000,
            "minimum_free_disk_bytes": 20_000_000_000,
            "exact_peak": False, "errors": [],
        },
        "outputs": [],
    }


class CompleteUnitAdmissionTests(unittest.TestCase):
    def test_raw_official_report_rebinds_seed_denominator_and_three_rows(self):
        report = {
            "seed": 44,
            "device": "cuda:0",
            "denominator": {"frozen": True, "n_declared": 3},
            "summary": {"n_total": 3, "n_success": 3, "n_failed": 0,
                        "success_rate": 1.0},
            "cases": [
                {"case_id": UID + "-" + arm, "uid": UID,
                 "status": "success", "n_frames": 16,
                 "cd_3d": 0.1, "cd_4d": 0.2, "cd_motion": 0.3}
                for arm in ("native", "world_gaussian", "body_gaussian")
            ],
        }
        scores = validate_official_report(report, UID)
        self.assertEqual(set(scores), {"native", "world_gaussian", "body_gaussian"})
        for key, value in (("seed", 45), ("device", "cpu")):
            broken = copy.deepcopy(report)
            broken[key] = value
            with self.assertRaises(ValueError):
                validate_official_report(broken, UID)
        broken = copy.deepcopy(report)
        broken["summary"]["n_failed"] = 1
        with self.assertRaises(ValueError):
            validate_official_report(broken, UID)

    def test_contract_is_prospective_and_cannot_claim_scientific_qualification(self):
        contract = {
            "kind": "actionbench-complete-unit-admission-contract",
            "version": "1.1.0",
            "source_unit": {
                "run_id": "complete-lowram-r7",
                "task_id": "complete-fp16-lowram-v1-three-arm-unit",
                "trial_id": "complete-fp16-lowram-v1-three-arm-unit",
                "approved_plan_digest": "a94aa69f8605266587f56f0977740001bbd21971463bb5b78d4e20eaf2bf574b",
                "uid": UID,
                "runtime_profile": "fp16-lowram-v1",
                "generation_seed": 42,
                "official_scoring_seed": 44,
                "successful_output_file_count": 117,
                "nested_pre_result_output_count": 121,
                "incidental_python_cache_paths": list(INCIDENTAL_PYTHON_CACHE_PATHS),
            },
            "required_status": "admitted_engineering_complete_unit",
            "scientific_effect_qualification": False,
            "native_scientific_qualification": False,
            "candidate_methods_tested": False,
            "queue_approved": False,
            "queue_generated": False,
        }
        validate_admission_contract(contract)
        for path, value in (("scientific_effect_qualification", True),
                            ("queue_approved", True),
                            ("queue_generated", True)):
            broken = copy.deepcopy(contract)
            broken[path] = value
            with self.assertRaises(ValueError):
                validate_admission_contract(broken)
        broken = copy.deepcopy(contract)
        broken["source_unit"]["successful_output_file_count"] = 1
        with self.assertRaises(ValueError):
            validate_admission_contract(broken)

    def test_completed_result_freezes_measured_cost_without_scientific_claim(self):
        measurement = validate_completed_result(completed_result(), UID, 22528)
        self.assertEqual(measurement["runtime_profile"], "fp16-lowram-v1")
        self.assertEqual(measurement["elapsed_seconds"], 150.0)
        self.assertEqual(measurement["observed_peak_mib"], 10255.0)
        self.assertEqual(set(measurement["scores"]),
                         {"native", "world_gaussian", "body_gaussian"})
        self.assertFalse(measurement["scientific_effect_qualification"])
        self.assertFalse(measurement["candidate_methods_tested"])
        self.assertFalse(measurement["queue_approved"])
        self.assertFalse(measurement["queue_generated"])

    def test_result_rejects_failure_claim_or_invalid_resource_measurement(self):
        base = completed_result()
        mutations = (
            ("status", "failed"),
            ("scientific_effect_qualification", True),
            ("candidate_methods_tested", True),
            ("elapsed_seconds", 27001.0),
        )
        for key, value in mutations:
            with self.subTest(key=key):
                broken = copy.deepcopy(base)
                broken[key] = value
                with self.assertRaises(ValueError):
                    validate_completed_result(broken, UID, 22528)
        for key, value in (("samples", 0), ("observed_peak_mib", 0),
                           ("observed_peak_mib", 22529), ("errors", ["poll failed"])):
            with self.subTest(device_key=key):
                broken = copy.deepcopy(base)
                broken["device_memory"][key] = value
                with self.assertRaises(ValueError):
                    validate_completed_result(broken, UID, 22528)

    def test_raw_telemetry_rebinds_counts_peaks_identity_and_errors(self):
        result = completed_result()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            device = root / "device-samples.jsonl"
            host = root / "host-samples.jsonl"
            start = datetime(2026, 10, 7, tzinfo=timezone.utc)
            device_rows, host_rows = [], []
            for index in range(151):
                device_rows.append({
                    "observed_at": (start + timedelta(seconds=index)).isoformat(),
                    "uuid": "GPU-test", "name": "GeForce RTX 2080 Ti",
                    "total_mib": 22528.0,
                    "used_mib": 10255.0 if index == 150 else 9000.0,
                    "utilization_percent": 99.0 if index == 150 else 95.0,
                })
                host_rows.append({
                    "epoch": start.timestamp() + index,
                    "process_tree_rss_bytes": (12_000_000_000 if index == 150
                                                else 10_000_000_000),
                    "output_bytes": 4_000_000 if index == 150 else 1_000_000,
                    "free_disk_bytes": 20_000_000_000 if index == 150
                                       else 21_000_000_000,
                })
            device.write_text("\n".join(map(json.dumps, device_rows)) + "\n")
            host.write_text("\n".join(map(json.dumps, host_rows)) + "\n")
            validate_telemetry(result, device, host, "GPU-test", 22528)

            broken = device.read_text().replace('"used_mib": 10255.0',
                                                '"used_mib": 10254.0')
            device.write_text(broken)
            with self.assertRaises(ValueError):
                validate_telemetry(result, device, host, "GPU-test", 22528)

            truncated_result = copy.deepcopy(result)
            truncated_result["device_memory"].update(samples=1,
                                                       observed_peak_mib=9000.0)
            device.write_text(json.dumps(device_rows[0]) + "\n")
            with self.assertRaises(ValueError):
                validate_telemetry(truncated_result, device, host, "GPU-test", 22528)

            shifted = copy.deepcopy(device_rows)
            for row in shifted:
                timestamp = datetime.fromisoformat(row["observed_at"])
                row["observed_at"] = (timestamp + timedelta(days=1)).isoformat()
            device.write_text("\n".join(map(json.dumps, shifted)) + "\n")
            with self.assertRaises(ValueError):
                validate_telemetry(result, device, host, "GPU-test", 22528)

    def test_execution_binding_rejects_stage_or_context_disagreement(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            origin_root = Path("/original/project")
            attempt_root = root / "runs/attempts/unit/attempt-1"
            task_root = root / "runs/harness/unit/tasks/task"
            (attempt_root / "workspace/actionmesh/unit-output").mkdir(parents=True)
            task_root.mkdir(parents=True)
            generation = {"status": "completed", "exit_code": 0,
                          "elapsed_seconds": 100.0}
            scoring = {"status": "completed", "exit_code": 0,
                       "elapsed_seconds": 40.0}
            for name, value in (("generation", generation),
                                ("official-scoring", scoring)):
                (attempt_root / ("workspace/actionmesh/unit-output/" + name +
                                 ".execution.json")).write_text(json.dumps(value))
            context = {
                "batch_plan_digest": "batch-digest",
                "native_plan_digest": "native-digest",
                "task_id": "task",
                "declared_resources": {"gpu_count": 1},
                "devices": ["GPU-test"],
                "CUDA_VISIBLE_DEVICES": "GPU-test",
                "gate_advanced": False,
                "process": {"pid": 7},
            }
            context_path = task_root / "execution-context.json"
            context_path.write_text(json.dumps(context))
            job = {"command": ["python", "-m", "runner",
                               str(origin_root / "input.json")], "cwd": "actionmesh"}
            origin_attempt = origin_root / "runs/attempts/unit/attempt-1"
            attempt = {
                "attempt_path": "runs/attempts/unit/attempt-1",
                "command": ["python", "-m", "runner",
                            str(origin_attempt / "workspace/input.json")],
                "cwd": str(origin_attempt / "workspace/actionmesh"),
                "retry_index": 0,
                "evidence_mode": "developmental",
                "provenance": {"revision": "pinned"},
                "process": {"pid": 7},
                "seconds": 151.0,
            }
            for name in ("stdout.log", "stderr.log"):
                (attempt_root / name).write_text("")
            guard = {
                "command": attempt["command"], "cwd": attempt["cwd"],
                "status": "completed", "exit_code": 0, "reason_code": None,
                "seconds": 150.5, "guard_process": {"parent_pid": 7},
            }
            guard_path = attempt_root / "process-guard.json"
            guard_path.write_text(json.dumps(guard))
            attempt.update({
                "process_guard_ref": file_ref(root, guard_path),
                "stdout_ref": file_ref(root, attempt_root / "stdout.log"),
                "stderr_ref": file_ref(root, attempt_root / "stderr.log"),
            })
            task = {"task_id": "task", "resources": {"gpu_count": 1}}
            task_result = {
                "execution_context_ref": file_ref(root, context_path),
                "devices": ["GPU-test"], "gate_advanced": False,
                "process": {"pid": 7},
            }
            result = completed_result()
            harness = {"plan_digest": "batch-digest"}
            native = {"plan_digest": "native-digest", "evidence_mode": "developmental",
                      "provenance": {"revision": "pinned"},
                      "limits": {"wall_time_seconds": 27000}}
            validate_execution_binding(root, origin_root, harness, native, task, job,
                                       attempt,
                                       task_result, result, task_root, attempt_root)
            generation["status"] = "failed"
            (attempt_root / "workspace/actionmesh/unit-output/"
             "generation.execution.json").write_text(json.dumps(generation))
            with self.assertRaises(ValueError):
                validate_execution_binding(root, origin_root, harness, native, task, job,
                                           attempt,
                                           task_result, result, task_root, attempt_root)

    def test_output_closure_requires_exact_117_receipt_refs_and_nested_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            attempt_path = "runs/attempts/unit/attempt-1"
            workspace = root / attempt_path / "workspace"
            output_paths = complete_unit_output_paths(UID)
            result = completed_result()
            refs = []
            nested = []
            for relative in output_paths:
                path = workspace / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                if relative.endswith("result.json"):
                    continue
                path.write_bytes(("evidence:" + relative).encode())
                ref = file_ref(root, path)
                refs.append(ref)
                nested.append({
                    "path": Path(relative).relative_to("actionmesh/unit-output").as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": ref["sha256"],
                })
            for relative in INCIDENTAL_PYTHON_CACHE_PATHS:
                path = workspace / "actionmesh/unit-output" / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(("cache:" + relative).encode())
                ref = file_ref(root, path)
                nested.append({"path": relative, "bytes": path.stat().st_size,
                               "sha256": ref["sha256"]})
            result["outputs"] = nested
            result_path = workspace / "actionmesh/unit-output/result.json"
            result_path.write_text(json.dumps(result))
            refs.append(file_ref(root, result_path))
            job = {"output_paths": output_paths}
            attempt = {"attempt_path": attempt_path, "output_refs": refs}

            verified = verify_output_closure(root, job, attempt, UID)
            self.assertEqual(len(verified), 117)

            missing = copy.deepcopy(attempt)
            missing["output_refs"] = missing["output_refs"][:-1]
            with self.assertRaises(ValueError):
                verify_output_closure(root, job, missing, UID)

            victim = workspace / output_paths[1]
            victim.write_bytes(b"changed after receipt")
            with self.assertRaises(ValueError):
                verify_output_closure(root, job, attempt, UID)

    def test_object_digest_is_canonical_and_rejects_non_json_numbers(self):
        self.assertEqual(object_digest({"b": 2, "a": 1}),
                         object_digest({"a": 1, "b": 2}))
        with self.assertRaises(ValueError):
            object_digest({"not_finite": float("nan")})
