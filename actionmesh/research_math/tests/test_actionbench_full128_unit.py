"""Synthetic CPU tests for one priced Full128 execution unit.

These tests do not load models, invoke the scorer, or run a GPU workload.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
from unittest.mock import patch

from research_math import actionbench_full128_unit as full
from research_math.control_scoring import file_ref


UID = "uid-b"


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class Full128UnitTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "dataset"
        sample = self.dataset / "data" / UID
        (sample / "imgs").mkdir(parents=True)
        payloads = [("camera.json", b"camera"), ("surfaces.npy", b"surface")]
        payloads += [(f"imgs/{index:02d}.png", f"frame-{index}".encode())
                     for index in range(16)]
        self.records = []
        for name, payload in payloads:
            path = sample / name
            path.write_bytes(payload)
            self.records.append({
                "path": f"data/{UID}/{name}", "bytes": len(payload),
                "sha256": sha(payload), "etag": "fixture",
            })
        self.snapshot_path = self.root / "snapshot.json"
        self.semantics_path = self.root / "semantics.json"
        self.template_path = self.root / "unit-manifest.json"
        self.snapshot = {
            "kind": "actionbench-full128-snapshot-admission", "version": "1.0.0",
            "status": "admitted_engineering_snapshot",
            "all_revisions_immutable": True, "all_content_files_hashed": True,
            "scientific_effect_qualification": False, "dispatch_ready": False,
            "snapshots": {
                "dataset": {
                    "repository": "facebook/actionbench", "revision": "d" * 40,
                    "manifest_sha256": sha(full.canonical(self.records)),
                    "files": self.records,
                },
                **{key: {"repository": key, "revision": key[0] * 40,
                         "file_count": 1, "total_bytes": 1,
                         "manifest_sha256": sha(key.encode())}
                   for key in ("actionmesh", "triposg", "dinov2", "rmbg")},
            },
        }
        self.semantics = {
            "kind": "actionbench-full128-dataset-semantics-admission",
            "version": "1.0.0", "status": "admitted_engineering_dataset_semantics",
            "dataset": "facebook/actionbench", "revision": "d" * 40,
            "population_size": 2, "frames_per_sample": 16,
            "snapshot_dataset_manifest_sha256": sha(full.canonical(self.records)),
            "samples": [{"uid": "uid-a", "input_manifest_sha256": "0" * 64},
                        {"uid": UID,
                         "input_manifest_sha256": sha(full.canonical(self.records))}],
            "all_consumed_bytes_revalidated": True,
            "scientific_effect_qualification": False, "dispatch_ready": False,
        }
        self.template = {
            "kind": "actionbench-current-release-unit-manifest", "version": "1.0.0",
            "status": "frozen_engineering_current_release_unit",
            "population": {"dataset": "facebook/actionbench", "revision": "d" * 40,
                           "size": 2},
            "source": {"revision": "a" * 40, "tree": "b" * 40,
                       "required_files": []},
            "model_snapshots": {
                key: {field: self.snapshot["snapshots"][key][field]
                      for field in ("repository", "revision", "file_count",
                                    "total_bytes", "manifest_sha256")}
                for key in ("actionmesh", "triposg", "dinov2", "rmbg")
            },
            "generation": {"runtime_profile": "fp16-lowram-v1", "seed": 42,
                           "fast": False, "low_ram": True, "dtype": "float16",
                           "config": "actionmesh/configs/actionmesh_lowram.yaml",
                           "repair_parent_receipt_sha256":
                           "7b4223a3fdcf4a738c1cec646ecb8f8172ece850a46aa8936aa96d8626cd62f9"},
            "multi_arm_unit": {"arms": ["native", "world_gaussian", "body_gaussian"]},
            "required_outputs": {}, "natural_failure_policy": {},
            "inference_executed": False, "scientific_effect_qualification": False,
            "dispatch_ready": False, "queue_generated": False,
        }
        self.snapshot_path.write_text(json.dumps(self.snapshot))
        self.semantics_path.write_text(json.dumps(self.semantics))
        self.template_path.write_text(json.dumps(self.template))
        self.template["prerequisite_receipts"] = {
            "snapshot_admission_sha256": file_ref(self.root, self.snapshot_path)["sha256"],
            "dataset_semantics_sha256": file_ref(self.root, self.semantics_path)["sha256"],
        }
        self.template_path.write_text(json.dumps(self.template))
        self.admission = {"unit_manifest_ref": file_ref(self.root, self.template_path)}
        self.pricing = {
            "kind": "actionbench-full128-queue-pricing", "version": "1.0.0",
            "status": "priced_engineering_only", "queue_priced": True,
            "queue_approved": False, "queue_generated": False,
            "dispatch_ready": False, "scientific_effect_qualification": False,
            "native_scientific_qualification": False,
            "candidate_methods_tested": False,
            "population": {"dataset": "facebook/actionbench", "revision": "d" * 40,
                           "size": 2},
            "unit_timeout_seconds": 1664, "units_per_window": 2,
            "window_count": 1,
            "windows": [{"window_id": "full128-window-01", "uids": ["uid-a", UID],
                         "unit_count": 2, "population_start_index": 0,
                         "population_stop_index_exclusive": 2,
                         "planned_workload_seconds": 3328}],
        }

    def freeze(self, pricing=None, uid=UID):
        output = self.root / f"{uid}-manifest.json"
        with patch.object(full, "verify_runtime_pricing_evidence",
                          return_value=(pricing or self.pricing, self.admission)), \
                patch.object(full, "_verify_template_source"):
            return full.freeze_unit(
                self.root, pricing or self.pricing, self.snapshot, self.semantics,
                self.template, self.root / "source", self.dataset, uid,
                "full128-window-01", output,
                file_ref(self.root, self.snapshot_path)["sha256"],
                file_ref(self.root, self.semantics_path)["sha256"],
                historical_root=self.root,
                historical_manifest=self.root / "archive-manifest.json")

    def test_priced_uid_freezes_exact_inputs_without_execution_claim(self):
        result = self.freeze()
        self.assertEqual(result["status"], "frozen_engineering_full128_unit")
        self.assertEqual(result["selected_unit"]["uid"], UID)
        self.assertEqual(result["selected_unit"]["window_id"], "full128-window-01")
        self.assertEqual(len(result["dataset_inputs"]), 18)
        self.assertFalse(result["inference_executed"])
        self.assertFalse(result["scientific_effect_qualification"])
        self.assertFalse(result["dispatch_ready"])
        self.assertTrue(result["queue_priced"])
        for key in ("queue_approved", "queue_generated",
                    "native_scientific_qualification", "candidate_methods_tested"):
            self.assertFalse(result[key])

    def test_uid_must_belong_to_exact_priced_window(self):
        with self.assertRaisesRegex(ValueError, "priced window"):
            self.freeze(uid="uid-outside")

    def test_mutated_selected_frame_is_rejected(self):
        (self.dataset / "data" / UID / "imgs/03.png").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "admitted bytes"):
            self.freeze()

    def test_extra_image_or_mask_cannot_change_official_loader_inputs(self):
        image_root = self.dataset / "data" / UID / "imgs"
        for name in ("16.png", "00_mask.png"):
            path = image_root / name
            path.write_bytes(b"extra")
            with self.subTest(name=name), \
                    self.assertRaisesRegex(ValueError, "directory closure"):
                self.freeze()
            path.unlink()

    def test_template_must_be_the_admission_bound_manifest(self):
        output = self.root / f"{UID}-manifest.json"
        original_ref = copy.deepcopy(self.admission["unit_manifest_ref"])
        self.admission["unit_manifest_ref"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(
                ValueError, "Changed pinned file"):
            self.freeze()
        self.assertFalse(output.exists())

        self.admission["unit_manifest_ref"] = original_ref
        self.template["required_outputs"] = {"unbound": True}
        with self.assertRaisesRegex(
                ValueError, "Admission-bound template manifest reference"):
            self.freeze()
        self.assertFalse(output.exists())

    def test_pricing_flags_cannot_claim_approval_or_dispatch(self):
        for key in ("queue_approved", "queue_generated", "dispatch_ready"):
            changed = copy.deepcopy(self.pricing)
            changed[key] = True
            with self.subTest(key=key), self.assertRaises(ValueError):
                full.validate_pricing_shape(changed)


class PricingReceiptTests(unittest.TestCase):
    @staticmethod
    def source_paths(root):
        return [
            root / "docs/research-math-20261006/actionbench-full128-queue-pricing-contract.json",
            root / "inputs/complete-unit-admissions/complete-lowram-r7.json",
            root / "actionmesh/research_overnight/assets/actionbench_population.json",
        ]

    def test_receipt_is_recomputed_from_its_three_pinned_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.source_paths(root)
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}")
            rebuilt = {"kind": "actionbench-full128-queue-pricing"}
            pricing = {**rebuilt,
                       "contract_ref": file_ref(root, paths[0]),
                       "admission_ref": file_ref(root, paths[1]),
                       "population_ref": file_ref(root, paths[2])}
            with patch.object(full, "build_pricing_manifest", return_value=rebuilt), \
                    patch.object(full, "validate_pricing_shape"):
                returned, admission = full.verify_pricing_receipt(root, pricing)
                self.assertEqual(returned, pricing)
                self.assertEqual(admission, {})

    def test_receipt_rejects_a_divergent_recomputation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.source_paths(root)
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}")
            pricing = {"kind": "changed",
                       "contract_ref": file_ref(root, paths[0]),
                       "admission_ref": file_ref(root, paths[1]),
                       "population_ref": file_ref(root, paths[2])}
            with patch.object(full, "build_pricing_manifest",
                              return_value={"kind": "expected"}), \
                    self.assertRaisesRegex(ValueError, "recomputation"):
                full.verify_pricing_receipt(root, pricing)

    def test_receipt_rejects_noncanonical_source_locations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = [root / name for name in ("contract.json", "admission.json",
                                               "population.json")]
            for path in paths:
                path.write_text("{}")
            pricing = {"contract_ref": file_ref(root, paths[0]),
                       "admission_ref": file_ref(root, paths[1]),
                       "population_ref": file_ref(root, paths[2])}
            with self.assertRaisesRegex(ValueError, "Canonical pricing source"):
                full.verify_pricing_receipt(root, pricing)


class RuntimeHistoricalPricingClosureTests(Full128UnitTests):
    def test_real_freezer_revalidates_distinct_historical_pricing_root(self):
        """Exercise the actual freezer with a current root and archived R7 root."""
        from unittest.mock import patch

        historical = self.root / "historical-r7"
        historical.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=historical, check=True)
        subprocess.run(["git", "config", "user.name", "Runtime Fixture"],
                       cwd=historical, check=True)
        subprocess.run(["git", "config", "user.email", "runtime-fixture@example.invalid"],
                       cwd=historical, check=True)

        canonical = {
            "inputs/actionbench-full128-queue/pricing.json": None,
            "docs/research-math-20261006/actionbench-full128-queue-pricing-contract.json":
                {"kind": "fixture-contract"},
            "actionmesh/research_overnight/assets/actionbench_population.json":
                {"kind": "fixture-population"},
            "inputs/complete-unit-admissions/complete-lowram-r7.json": None,
            "inputs/fp16-lowram-v1/unit-manifest.json": self.template,
        }
        current_manifest = self.root / "inputs/fp16-lowram-v1/unit-manifest.json"
        current_manifest.parent.mkdir(parents=True, exist_ok=True)
        current_manifest.write_text(json.dumps(self.template))
        self.admission = {"unit_manifest_ref": file_ref(self.root, current_manifest)}
        canonical["inputs/complete-unit-admissions/complete-lowram-r7.json"] = self.admission

        pricing = dict(self.pricing)
        for relative in ("docs/research-math-20261006/actionbench-full128-queue-pricing-contract.json",
                         "inputs/complete-unit-admissions/complete-lowram-r7.json",
                         "actionmesh/research_overnight/assets/actionbench_population.json"):
            current = self.root / relative
            current.parent.mkdir(parents=True, exist_ok=True)
            value = canonical[relative]
            current.write_text(json.dumps(value))
            pricing[{"docs/research-math-20261006/actionbench-full128-queue-pricing-contract.json":
                     "contract_ref",
                     "inputs/complete-unit-admissions/complete-lowram-r7.json":
                     "admission_ref",
                     "actionmesh/research_overnight/assets/actionbench_population.json":
                     "population_ref"}[relative]] = file_ref(self.root, current)
        pricing_path = self.root / "inputs/actionbench-full128-queue/pricing.json"
        pricing_path.parent.mkdir(parents=True, exist_ok=True)
        pricing_path.write_text(json.dumps(pricing))
        canonical["inputs/actionbench-full128-queue/pricing.json"] = pricing

        archive_rows = []
        archive_dir = self.root / "historical-blobs"
        archive_dir.mkdir()
        for index, (relative, value) in enumerate(canonical.items()):
            path = historical / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
            blob = archive_dir / f"{index}.blob"
            blob.write_bytes(path.read_bytes())
            archive_rows.append({"source": relative, "archive_path":
                                 blob.relative_to(self.root).as_posix(),
                                 "bytes": blob.stat().st_size,
                                 "sha256": sha(blob.read_bytes()), "matches": True})
        subprocess.run(["git", "add", "."], cwd=historical, check=True)
        subprocess.run(["git", "commit", "-qm", "fixture"], cwd=historical, check=True)
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=historical,
                                  capture_output=True, text=True, check=True).stdout.strip()

        archive_manifest = self.root / "archive-manifest.json"
        archive_manifest.write_text(json.dumps({
            "original_bytes": True,
            "copied_to_canonical_project_paths": False,
            "files": archive_rows,
        }))
        expected_pricing = {key: value for key, value in pricing.items()
                            if key not in {"contract_ref", "admission_ref", "population_ref"}}
        output = self.root / "runtime-revalidated.json"
        with patch.object(full, "R7_SOURCE_REVISION", revision, create=True), \
                patch.object(full, "R7_ARCHIVE_MANIFEST_SHA256",
                             sha(archive_manifest.read_bytes()), create=True), \
                patch.object(full, "build_pricing_manifest",
                             return_value=expected_pricing), \
                patch.object(full, "validate_pricing_shape"), \
                patch.object(full, "_verify_template_source"):
            result = full.freeze_unit(
                self.root, pricing, self.snapshot, self.semantics, self.template,
                self.root / "source", self.dataset, UID, "full128-window-01",
                output, file_ref(self.root, self.snapshot_path)["sha256"],
                file_ref(self.root, self.semantics_path)["sha256"],
                historical_root=historical, historical_manifest=archive_manifest)
        self.assertEqual(result["status"], "frozen_engineering_full128_unit")
        self.assertEqual(result["selected_unit"]["uid"], UID)

        (self.root / "docs/research-math-20261006/actionbench-full128-queue-pricing-contract.json").write_text(
            json.dumps({"kind": "substituted-current-contract"}))
        with patch.object(full, "R7_SOURCE_REVISION", revision, create=True), \
                patch.object(full, "R7_ARCHIVE_MANIFEST_SHA256",
                             sha(archive_manifest.read_bytes()), create=True), \
                patch.object(full, "build_pricing_manifest",
                             return_value=expected_pricing), \
                patch.object(full, "validate_pricing_shape"), \
                patch.object(full, "_verify_template_source"), \
                self.assertRaisesRegex(ValueError, "Current pricing copy differs"):
            full.freeze_unit(
                self.root, pricing, self.snapshot, self.semantics, self.template,
                self.root / "source", self.dataset, UID, "full128-window-01",
                self.root / "runtime-rejected.json",
                file_ref(self.root, self.snapshot_path)["sha256"],
                file_ref(self.root, self.semantics_path)["sha256"],
                historical_root=historical, historical_manifest=archive_manifest)
if __name__ == "__main__":
    unittest.main()
