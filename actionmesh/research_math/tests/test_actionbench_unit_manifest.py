"""Engineering tests for the current-release unit manifest.

Fixtures are tiny synthetic byte inventories.  They are not benchmark samples,
do not load models and produce no scientific score.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from research_math import actionbench_unit_manifest as unit


UID = "uid-a"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class UnitManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "source"
        self.dataset = self.root / "dataset"
        self.source.mkdir()
        self.dataset.mkdir()
        self.required_source = ["entry.py", "config.yaml"]
        for name in self.required_source:
            (self.source / name).write_text(name)
        sample = self.dataset / "data" / UID
        (sample / "imgs").mkdir(parents=True)
        paths = [("camera.json", b"camera"), ("surfaces.npy", b"surface")]
        paths += [(f"imgs/{index:02d}.png", f"frame-{index}".encode())
                  for index in range(16)]
        self.records = []
        for name, payload in paths:
            path = sample / name
            path.write_bytes(payload)
            relative = "data/" + UID + "/" + name
            self.records.append({"path": relative, "bytes": len(payload),
                                 "sha256": sha(payload), "etag": "fixture"})
        self.population = {"dataset": "facebook/actionbench",
                           "revision": "d" * 40, "uids": [UID]}
        self.contract = {
            "kind": "actionbench-current-release-unit-contract",
            "version": "1.0.0",
            "population": {"dataset": "facebook/actionbench",
                           "revision": "d" * 40, "size": 1,
                           "uid_set_sha256": sha(unit.canonical([UID]))},
            "calibration_unit": {
                "uid": UID,
                "selection_rule": "first UID in the canonical lexicographically sorted released population",
                "role": "engineering",
            },
            "source": {"revision": "a" * 40, "tree": "b" * 40,
                       "required_files": self.required_source},
            "model_snapshots": {
                key: {"repository": key, "revision": key[0] * 40}
                for key in ("actionmesh", "triposg", "dinov2", "rmbg")
            },
            "generation": {"seed": 42, "dtype": "bfloat16",
                           "fast": False, "low_ram": False},
            "multi_arm_unit": {"arms": ["native", "world_gaussian",
                                          "body_gaussian"],
                               "official_scorer": {"sampling_seed": 44}},
            "required_outputs": {"native_generation": ["mesh_00.glb through mesh_15.glb"]},
            "natural_failure_policy": {"forbidden_automatic_changes": ["enable --low_ram"]},
        }
        dataset_manifest = sha(unit.canonical(self.records))
        snapshots = {"dataset": {"repository": "facebook/actionbench",
                                  "revision": "d" * 40,
                                  "manifest_sha256": dataset_manifest,
                                  "files": self.records}}
        for key, required in self.contract["model_snapshots"].items():
            snapshots[key] = {**required, "file_count": 1, "total_bytes": 1,
                              "manifest_sha256": key * 8}
        self.snapshot = {
            "kind": "actionbench-full128-snapshot-admission", "version": "1.0.0",
            "status": "admitted_engineering_snapshot",
            "all_revisions_immutable": True, "all_content_files_hashed": True,
            "scientific_effect_qualification": False, "dispatch_ready": False,
            "source": {"revision": "a" * 40, "tree": "b" * 40,
                       "tracked_worktree_clean": True},
            "snapshots": snapshots,
        }
        self.semantics = {
            "kind": "actionbench-full128-dataset-semantics-admission",
            "version": "1.0.0", "status": "admitted_engineering_dataset_semantics",
            "dataset": "facebook/actionbench", "revision": "d" * 40,
            "population_size": 1, "frames_per_sample": 16,
            "snapshot_dataset_manifest_sha256": dataset_manifest,
            "samples": [{"uid": UID,
                         "input_manifest_sha256": sha(unit.canonical(self.records))}],
            "all_consumed_bytes_revalidated": True,
            "scientific_effect_qualification": False, "dispatch_ready": False,
        }

    def git(self, command, cwd=None, capture_output=None, text=None, check=None):
        args = command[1:]
        if args == ["rev-parse", "HEAD"]:
            return SimpleNamespace(stdout="a" * 40 + "\n", returncode=0)
        if args == ["rev-parse", "HEAD^{tree}"]:
            return SimpleNamespace(stdout="b" * 40 + "\n", returncode=0)
        if args == ["status", "--porcelain", "--untracked-files=no"]:
            return SimpleNamespace(stdout="", returncode=0)
        if args[:3] == ["ls-files", "--error-unmatch", "--"]:
            return SimpleNamespace(stdout=args[3] + "\n", returncode=0)
        raise AssertionError(args)

    def freeze(self, output_name="unit.json"):
        with patch.object(unit.subprocess, "run", side_effect=self.git):
            return unit.freeze(self.contract, self.population, self.snapshot,
                               self.semantics, self.source, self.dataset,
                               self.root / output_name, "1" * 64, "2" * 64)

    def test_valid_manifest_binds_exact_three_arm_unit_without_execution(self):
        result = self.freeze()
        self.assertEqual(result["status"], "frozen_engineering_current_release_unit")
        self.assertEqual(result["multi_arm_unit"]["arms"],
                         ["native", "world_gaussian", "body_gaussian"])
        self.assertFalse(result["inference_executed"])
        self.assertEqual(result["official_scorer_invocations"], 0)
        self.assertFalse(result["dispatch_ready"])

    def test_nonprospective_uid_selection_is_rejected(self):
        self.contract["calibration_unit"]["uid"] = "other"
        with self.assertRaisesRegex(ValueError, "prospectively"):
            self.freeze()

    def population_case(self):
        self.population['uids'] = ['uid-a', 'uid-b']
        self.contract['population'].update(size=2, uid_set_sha256=sha(unit.canonical(['uid-a', 'uid-b'])))
        self.semantics['population_size'] = 2
        self.semantics['samples'].append({'uid': 'uid-b', 'input_manifest_sha256': '3' * 64})
        self.contract['kind'] = 'actionbench-current-release-population-unit-contract'
        self.contract['calibration_unit'] = {
            'uid': 'uid-b', 'population_index': 1,
            'selection_rule': 'canonical full-population index; no outcome-based exclusions',
        }

    def test_population_contract_admits_exact_second_member(self):
        self.population_case()
        uid, _, semantic = unit._validate(self.contract, self.population, self.snapshot, self.semantics)
        self.assertEqual(uid, 'uid-b')
        self.assertEqual(semantic['uid'], 'uid-b')

    def test_population_contract_rejects_index_uid_disagreement(self):
        self.population_case()
        self.contract['calibration_unit']['population_index'] = 0
        with self.assertRaises(ValueError):
            unit._validate(self.contract, self.population, self.snapshot, self.semantics)

    def test_original_calibration_contract_cannot_select_second_member(self):
        self.population_case()
        self.contract['kind'] = 'actionbench-current-release-unit-contract'
        self.contract['calibration_unit']['selection_rule'] = 'first UID in the canonical lexicographically sorted released population'
        with self.assertRaises(ValueError):
            unit._validate(self.contract, self.population, self.snapshot, self.semantics)

    def test_mutated_selected_frame_is_rejected(self):
        (self.dataset / "data" / UID / "imgs" / "03.png").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "admitted bytes"):
            self.freeze()

    def test_semantic_manifest_mismatch_is_rejected(self):
        self.semantics["samples"][0]["input_manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "semantic admission"):
            self.freeze()

    def test_source_revision_mismatch_is_rejected(self):
        self.contract["source"]["revision"] = "c" * 40
        with self.assertRaisesRegex(ValueError, "source identity"):
            self.freeze()

    def test_model_revision_mismatch_is_rejected(self):
        self.snapshot["snapshots"]["triposg"]["revision"] = "0" * 40
        with self.assertRaisesRegex(ValueError, "triposg"):
            self.freeze()

    def test_existing_output_is_preserved(self):
        path = self.root / "existing.json"
        path.write_text("keep")
        with self.assertRaises(FileExistsError):
            self.freeze("existing.json")
        self.assertEqual(path.read_text(), "keep")

    def test_contract_keeps_generation_and_scoring_seeds_distinct(self):
        result = self.freeze()
        self.assertEqual(result["generation"]["seed"], 42)
        self.assertEqual(result["multi_arm_unit"]["official_scorer"]["sampling_seed"], 44)

    def test_contract_forbids_silent_low_ram_fallback(self):
        result = self.freeze()
        self.assertIn("enable --low_ram",
                      result["natural_failure_policy"]["forbidden_automatic_changes"])

    def test_untracked_required_source_is_rejected(self):
        def untracked(command, **kwargs):
            result = self.git(command, **kwargs)
            if command[1:3] == ["ls-files", "--error-unmatch"]:
                result.returncode = 1
            return result
        with patch.object(unit.subprocess, "run", side_effect=untracked):
            with self.assertRaisesRegex(ValueError, "not tracked"):
                unit.freeze(self.contract, self.population, self.snapshot,
                            self.semantics, self.source, self.dataset,
                            self.root / "untracked.json", "1" * 64, "2" * 64)

    def test_escaping_required_source_path_is_rejected(self):
        self.contract["source"]["required_files"] = ["../escape.py"]
        with patch.object(unit.subprocess, "run", side_effect=self.git):
            with self.assertRaisesRegex(ValueError, "Nonescaping"):
                unit.freeze(self.contract, self.population, self.snapshot,
                            self.semantics, self.source, self.dataset,
                            self.root / "escape.json", "1" * 64, "2" * 64)


if __name__ == "__main__":
    unittest.main()
