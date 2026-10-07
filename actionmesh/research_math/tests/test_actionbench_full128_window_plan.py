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
    build_native_command,
    native_limits,
    select_priced_window,
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


class Full128WindowPlanTests(unittest.TestCase):
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
