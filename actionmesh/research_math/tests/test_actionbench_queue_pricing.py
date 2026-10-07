import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research_math.actionbench_queue_pricing import (
    build_pricing_manifest,
    validate_pricing_contract,
)
from research_math.complete_unit_admission import (
    INCIDENTAL_PYTHON_CACHE_PATHS,
    validate_completed_result,
)
from research_math.control_scoring import file_ref, resolve_ref


UIDS = [f"000-{index:03d}_uid{index:03d}" for index in range(128)]


def pricing_contract():
    return {
        "kind": "actionbench-full128-queue-pricing-contract",
        "version": "1.0.0",
        "source_admission": {
            "kind": "actionbench-complete-unit-admission",
            "version": "1.0.0",
            "status": "admitted_engineering_complete_unit",
            "run_id": "complete-lowram-r7",
            "runtime_profile": "fp16-lowram-v1",
            "successful_output_file_count": 117,
        },
        "population": {
            "dataset": "facebook/actionbench",
            "revision": "2796071cbe6248422fcbeab3101fa9f9886cb7b9",
            "size": 128,
        },
        "budget": {
            "hard_window_seconds": 28800,
            "collection_reserve_seconds": 1800,
            "workload_budget_seconds": 27000,
            "unit_runtime_headroom_numerator": 5,
            "unit_runtime_headroom_denominator": 4,
        },
        "output_target": "inputs/actionbench-full128-queue/pricing.json",
        "scientific_effect_qualification": False,
        "native_scientific_qualification": False,
        "candidate_methods_tested": False,
        "queue_approved": False,
        "queue_generated": False,
        "dispatch_ready": False,
    }


def population():
    return {
        "dataset": "facebook/actionbench",
        "revision": "2796071cbe6248422fcbeab3101fa9f9886cb7b9",
        "uids": UIDS,
    }


def admitted_sidecar(root: Path, admission_contract: Path):
    run_id = "complete-lowram-r7"
    digest = "a94aa69f8605266587f56f0977740001bbd21971463bb5b78d4e20eaf2bf574b"
    task_id = "complete-fp16-lowram-v1-three-arm-unit"
    batch = root / "runs/harness" / run_id
    task = batch / "tasks" / task_id
    run = root / "runs/attempts" / run_id
    attempt = run / "attempt-a1"
    output = attempt / "workspace/actionmesh/unit-output"
    output.mkdir(parents=True)
    scores = {
        arm: {metric: 0.1 for metric in ("cd_3d", "cd_4d", "cd_motion")}
        for arm in ("native", "world_gaussian", "body_gaussian")
    }
    result = {
        "status": "completed",
        "runtime_profile": "fp16-lowram-v1",
        "gpu_uuid": "GPU-test",
        "elapsed_seconds": 1330.4655511886813,
        "stages": {
            name: {"status": "completed", "elapsed_seconds": 1.0}
            for name in ("generation", "export", "controls", "official_scoring")
        },
        "device_memory": {
            "sample_interval_seconds": 1, "samples": 1, "exact_peak": False,
            "errors": [], "observed_peak_mib": 10255.0,
        },
        "host_resources": {
            "sample_interval_seconds": 1, "samples": 1, "exact_peak": False,
            "errors": [], "observed_peak_rss_bytes": 1,
            "observed_peak_output_bytes": 1, "minimum_free_disk_bytes": 1,
        },
        "scores": scores,
        "scientific_effect_qualification": False,
        "candidate_methods_tested": False,
    }
    inventory_paths = []
    for index in range(116):
        path = output / "declared" / f"output-{index:03d}.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(str(index).encode())
        inventory_paths.append(path)
    for relative in INCIDENTAL_PYTHON_CACHE_PATHS:
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"cache")
        inventory_paths.append(path)
    result["outputs"] = [
        {
            "path": path.relative_to(output).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_ref(root, path)["sha256"],
        }
        for path in sorted(inventory_paths)
    ]
    result_path = output / "result.json"
    result_path.write_text(json.dumps(result))
    output_refs = [file_ref(root, path) for path in inventory_paths[:116]]
    output_refs.append(file_ref(root, result_path))

    harness_plan = {
        "plan_digest": digest, "output_root": "runs/harness", "batch_id": run_id,
        "tasks": [{"task_id": task_id}],
    }
    native_plan = {
        "plan_digest": "native-digest", "output_root": "runs/attempts",
        "run_id": run_id,
    }
    guard = attempt / "process-guard.json"
    stdout = attempt / "stdout.log"
    stderr = attempt / "stderr.log"
    for path, content in ((guard, "{}"), (stdout, ""), (stderr, "")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    attempt_row = {
        "attempt_id": "attempt-a1",
        "attempt_path": attempt.relative_to(root).as_posix(),
        "process_guard_ref": file_ref(root, guard),
        "stdout_ref": file_ref(root, stdout),
        "stderr_ref": file_ref(root, stderr),
    }
    native_receipt = {
        "plan_digest": "native-digest", "status": "completed",
        "attempts": [attempt_row],
    }
    fixed_json = {
        root / "plans" / run_id / "harness.json": harness_plan,
        batch / "plan.json": harness_plan,
        batch / "state.json": {},
        batch / "report.json": {"plan_digest": digest, "status": "completed"},
        task / "task.json": {},
        task / "result.json": {},
        task / "execution-context.json": {},
        root / "plans" / run_id / "native.json": native_plan,
        run / "plan.json": native_plan,
        run / "receipt.json": native_receipt,
        attempt / "attempt.json": attempt_row,
        output / "generation.execution.json": {},
        output / "official-scoring.execution.json": {},
    }
    for path, value in fixed_json.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    origin_paths = [
        batch / "plan.json", batch / "state.json", batch / "report.json",
        task / "task.json", task / "result.json", run / "plan.json",
        task / "execution-context.json", run / "receipt.json",
        attempt / "attempt.json", guard, stdout, stderr,
        output / "generation.execution.json",
        output / "official-scoring.execution.json",
    ]
    unit_manifest = root / "inputs/actionbench-full128-snapshots/unit-manifest.json"
    unit_manifest.parent.mkdir(parents=True)
    unit_manifest.write_text("{}")
    measurement = validate_completed_result(result, UIDS[0], 11264)
    return {
        "kind": "actionbench-complete-unit-admission",
        "version": "1.0.0",
        "status": "admitted_engineering_complete_unit",
        "run_id": "complete-lowram-r7",
        "trial_id": "complete-fp16-lowram-v1-three-arm-unit",
        "attempt_id": "attempt-a1",
        "approved_plan_digest": digest,
        "contract_ref": file_ref(root, admission_contract),
        "unit_manifest_ref": file_ref(root, unit_manifest),
        "result_ref": file_ref(root, result_path),
        "origin_refs": [file_ref(root, path) for path in origin_paths],
        "output_refs": output_refs,
        "successful_output_file_count": 117,
        "measurement": measurement,
        "eligible_for_queue_pricing": True,
        "scientific_effect_qualification": False,
        "native_scientific_qualification": False,
        "candidate_methods_tested": False,
        "queue_approved": False,
        "queue_generated": False,
    }


class ActionBenchQueuePricingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.admission_contract = self.root / (
            "docs/research-math-20261006/"
            "actionbench-complete-unit-admission-contract.json")
        self.admission_contract.parent.mkdir(parents=True)
        self.admission_contract.write_text(json.dumps({"kind": "contract"}))
        self.reproduction_contract = self.root / (
            "docs/research-math-20261006/"
            "actionbench-full128-reproduction-contract.json")
        self.reproduction_contract.write_text(json.dumps({
            "kind": "actionbench-full-population-reproduction-contract",
            "version": "1.0.0",
            "purpose": "official-full-population-baseline-reproduction",
            "queue_policy": {
                "hard_window_seconds": 28800,
                "collection_reserve_seconds": 1800,
            },
            "scientific_effect_qualification": False,
            "candidate_methods_tested": False,
            "dispatch_ready": False,
        }))
        self.contract = pricing_contract()
        self.contract["source_admission"]["contract_ref"] = file_ref(
            self.root, self.admission_contract)
        self.contract["full128_reproduction_contract_ref"] = file_ref(
            self.root, self.reproduction_contract)
        self.population = population()
        self.population_path = self.root / (
            "actionmesh/research_overnight/assets/actionbench_population.json")
        self.population_path.parent.mkdir(parents=True)
        self.population_path.write_text(json.dumps(self.population))
        self.contract["population_ref"] = file_ref(
            self.root, self.population_path)
        self.admission = admitted_sidecar(self.root, self.admission_contract)
        expected = copy.deepcopy(self.admission)
        expected.pop("contract_ref")
        self.origin_patcher = patch(
            "research_math.actionbench_queue_pricing.verify_origin",
            return_value=(expected, expected["origin_refs"]))
        self.origin_patcher.start()
        self.addCleanup(self.origin_patcher.stop)

    def tearDown(self):
        self.temporary.cleanup()

    def test_admitted_measurement_prices_eight_equal_sixteen_uid_windows(self):
        result = build_pricing_manifest(
            self.root, self.contract, self.admission, self.population)

        expected_timeout = math.ceil(1330.4655511886813 * 5 / 4)
        self.assertEqual(expected_timeout, 1664)
        self.assertEqual(result["unit_timeout_seconds"], expected_timeout)
        self.assertEqual(result["units_per_window"], 16)
        self.assertEqual(result["window_count"], 8)
        self.assertEqual(len(result["windows"]), 8)
        self.assertTrue(all(len(window["uids"]) == 16
                            for window in result["windows"]))
        self.assertEqual(
            [uid for window in result["windows"] for uid in window["uids"]],
            UIDS)
        self.assertEqual(result["planned_workload_seconds_per_full_window"],
                         16 * 1664)
        self.assertEqual(result["unallocated_workload_seconds_per_full_window"],
                         27000 - 16 * 1664)
        self.assertTrue(result["queue_priced"])
        self.assertFalse(result["queue_approved"])
        self.assertFalse(result["queue_generated"])
        self.assertFalse(result["dispatch_ready"])
        self.assertFalse(result["scientific_effect_qualification"])

    def test_pricing_rejects_unadmitted_or_scientific_claiming_sidecar(self):
        for path, value in (
                (("eligible_for_queue_pricing",), False),
                (("queue_generated",), True),
                (("measurement", "runtime_profile"), "default"),
                (("measurement", "scientific_effect_qualification"), True),
                (("successful_output_file_count",), 116)):
            with self.subTest(path=path):
                changed = copy.deepcopy(self.admission)
                target = changed
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                with self.assertRaises(ValueError):
                    build_pricing_manifest(
                        self.root, self.contract, changed, self.population)
        changed = copy.deepcopy(self.admission)
        changed["output_refs"][1] = changed["output_refs"][0]
        with self.assertRaisesRegex(ValueError, "evidence closure"):
            build_pricing_manifest(
                self.root, self.contract, changed, self.population)

    def test_pricing_rehashes_every_admission_evidence_reference(self):
        result_path = resolve_ref(self.root, self.admission["result_ref"])
        cache = result_path.parent / INCIDENTAL_PYTHON_CACHE_PATHS[0]
        cache.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "inventory hash"):
            build_pricing_manifest(
                self.root, self.contract, self.admission, self.population)

    def test_pricing_rejects_population_reordering_or_success_only_subset(self):
        for uids in (UIDS[:-1], [UIDS[1], UIDS[0], *UIDS[2:]]):
            changed = dict(self.population, uids=uids)
            with self.assertRaises(ValueError):
                build_pricing_manifest(
                    self.root, self.contract, self.admission, changed)

    def test_contract_cannot_approve_or_generate_a_queue(self):
        validate_pricing_contract(self.contract)
        for key in ("queue_approved", "queue_generated", "dispatch_ready",
                    "scientific_effect_qualification"):
            changed = copy.deepcopy(self.contract)
            changed[key] = True
            with self.assertRaises(ValueError):
                validate_pricing_contract(changed)

    def test_sidecar_timing_cannot_diverge_from_retained_result(self):
        changed = copy.deepcopy(self.admission)
        changed["measurement"]["elapsed_seconds"] = 22000.0
        with self.assertRaisesRegex(ValueError, "differs from canonical source"):
            build_pricing_manifest(
                self.root, self.contract, changed, self.population)


if __name__ == "__main__":
    unittest.main()
