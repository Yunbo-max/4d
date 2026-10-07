"""Engineering tests for immutable ActionBench snapshot admission.

These fixtures are not benchmark samples and produce no scientific score.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from research_math import snapshot_admission

try:
    snapshot_planner = importlib.import_module("prepare_actionbench_snapshots")
    run_experiments = importlib.import_module("run_experiments")
except ModuleNotFoundError:
    snapshot_planner = None
    run_experiments = None


class SnapshotAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.revision = "a" * 40
        self.population = {
            "dataset": "facebook/actionbench",
            "revision": self.revision,
            "uids": ["uid-a", "uid-b"],
        }
        self.contract = {
            "kind": "actionbench-full128-snapshot-contract",
            "version": "1.0.0",
            "source": {
                "repository": "facebookresearch/actionmesh",
                "revision": "b" * 40,
                "tree": "c" * 40,
                "submodules": {
                    "third_party/TripoSG": "d" * 40,
                },
            },
            "snapshots": {
                "dataset": {
                    "repository": "facebook/actionbench",
                    "repository_type": "dataset",
                    "revision": self.revision,
                },
                "actionmesh": {
                    "repository": "facebook/ActionMesh",
                    "repository_type": "model",
                    "revision": "e" * 40,
                },
                "triposg": {
                    "repository": "VAST-AI/TripoSG",
                    "repository_type": "model",
                    "revision": "f" * 40,
                },
                "dinov2": {
                    "repository": "facebook/dinov2-large",
                    "repository_type": "model",
                    "revision": "1" * 40,
                },
                "rmbg": {
                    "repository": "briaai/RMBG-1.4",
                    "repository_type": "model",
                    "revision": "2" * 40,
                },
            },
            "dataset_layout": {
                "population_size": 2,
                "frames_per_sample": 16,
                "data_directory": "data",
                "frame_directory": "imgs",
                "frame_pattern": "{index:02d}.png",
                "required_sample_files": ["camera.json", "surfaces.npy"],
                "uid_set_sha256": hashlib.sha256(json.dumps(
                    self.population["uids"], sort_keys=True, separators=(",", ":"),
                    ensure_ascii=False).encode()).hexdigest(),
            },
        }
        self.roots = {}
        (self.root / "source").mkdir()
        for key in self.contract["snapshots"]:
            directory = self.root / key
            directory.mkdir()
            self.roots[key] = directory
        self._write_hf_file("actionmesh", "model.bin", b"weights")
        self._write_hf_file("triposg", "model.bin", b"weights")
        self._write_hf_file("dinov2", "model.bin", b"weights")
        self._write_hf_file("rmbg", "model.bin", b"weights")
        for uid in self.population["uids"]:
            self._write_hf_file("dataset", f"data/{uid}/camera.json", b"{}")
            self._write_hf_file("dataset", f"data/{uid}/surfaces.npy", b"NUMPY")
            for index in range(16):
                self._write_hf_file(
                    "dataset", f"data/{uid}/imgs/{index:02d}.png", b"PNG")

    def _write_hf_file(self, key, relative, content, revision=None):
        target = self.roots[key] / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        metadata = (self.roots[key] / ".cache/huggingface/download" /
                    (relative + ".metadata"))
        metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata.write_text(
            (revision or self.contract["snapshots"][key]["revision"])
            + "\nfixture-etag\n0\n")
        return target

    def _git(self, root, args):
        values = {
            ("rev-parse", "HEAD"): self.contract["source"]["revision"],
            ("rev-parse", "HEAD^{tree}"): self.contract["source"]["tree"],
            ("status", "--porcelain", "--untracked-files=no"): "",
            ("submodule", "status", "--", "third_party/TripoSG"):
                " " + self.contract["source"]["submodules"]["third_party/TripoSG"]
                + " third_party/TripoSG (heads/main)",
        }
        return values[tuple(args)]

    def _admit(self, output="admission.json"):
        with patch.object(snapshot_admission, "git_output", side_effect=self._git):
            return snapshot_admission.admit(
                self.contract, self.population, self.root / "source", self.roots,
                self.root / output)

    def test_complete_exact_revision_snapshots_emit_closed_manifest(self):
        result = self._admit()
        self.assertEqual(result["status"], "admitted_engineering_snapshot")
        self.assertEqual(result["dataset"]["population_size"], 2)
        self.assertEqual(result["dataset"]["frames_per_sample"], 16)
        self.assertEqual(result["source"]["revision"], "b" * 40)
        self.assertEqual(set(result["snapshots"]), set(self.roots))
        for snapshot in result["snapshots"].values():
            self.assertGreater(snapshot["file_count"], 0)
            for item in snapshot["files"]:
                self.assertNotIn(".cache/huggingface", item["path"])
                self.assertEqual(len(item["sha256"]), 64)

    def test_missing_dataset_frame_is_rejected(self):
        (self.roots["dataset"] / "data/uid-a/imgs/15.png").unlink()
        with self.assertRaisesRegex(ValueError, "dataset file closure"):
            self._admit()

    def test_additional_dataset_uid_is_rejected(self):
        self._write_hf_file("dataset", "data/unreleased/surfaces.npy", b"NUMPY")
        with self.assertRaisesRegex(ValueError, "population differs"):
            self._admit()

    def test_wrong_hub_revision_metadata_is_rejected(self):
        metadata = (self.roots["actionmesh"] /
                    ".cache/huggingface/download/model.bin.metadata")
        metadata.write_text("9" * 40 + "\nfixture-etag\n0\n")
        with self.assertRaisesRegex(ValueError, "revision metadata mismatch"):
            self._admit()

    def test_missing_hub_metadata_is_rejected(self):
        (self.roots["rmbg"] /
         ".cache/huggingface/download/model.bin.metadata").unlink()
        with self.assertRaisesRegex(ValueError, "Missing Hugging Face metadata"):
            self._admit()

    def test_lfs_pointer_is_rejected_instead_of_hashed_as_model(self):
        path = self.roots["triposg"] / "model.bin"
        path.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:" +
                        "0" * 64 + "\nsize 123\n")
        with self.assertRaisesRegex(ValueError, "Git LFS pointer"):
            self._admit()

    def test_downloaded_html_error_is_rejected(self):
        (self.roots["dinov2"] / "model.bin").write_text(
            "<!doctype html><title>gateway error</title>")
        with self.assertRaisesRegex(ValueError, "HTML response"):
            self._admit()

    def test_symlink_is_rejected_even_when_target_stays_inside_root(self):
        link = self.roots["actionmesh"] / "alias.bin"
        link.symlink_to("model.bin")
        with self.assertRaisesRegex(ValueError, "Symlink"):
            self._admit()

    def test_nested_metadata_parent_symlink_is_rejected(self):
        nested = (self.roots["dataset"] /
                  ".cache/huggingface/download/data/uid-a/imgs")
        external = self.root / "external-metadata"
        nested.rename(external)
        nested.symlink_to(external, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "metadata directory"):
            self._admit()

    def test_empty_snapshot_is_rejected(self):
        (self.roots["rmbg"] / "model.bin").unlink()
        with self.assertRaisesRegex(ValueError, "contains no regular files"):
            self._admit()

    def test_dirty_or_wrong_git_source_is_rejected(self):
        def dirty(root, args):
            if tuple(args) == ("status", "--porcelain", "--untracked-files=no"):
                return " M actionmesh/pipeline.py"
            return self._git(root, args)
        with patch.object(snapshot_admission, "git_output", side_effect=dirty):
            with self.assertRaisesRegex(ValueError, "tracked modifications"):
                snapshot_admission.admit(
                    self.contract, self.population, self.root / "source",
                    self.roots, self.root / "admission.json")

    def test_submodule_mismatch_is_rejected(self):
        def wrong_submodule(root, args):
            if tuple(args[:2]) == ("submodule", "status"):
                return "+" + "9" * 40 + " third_party/TripoSG"
            return self._git(root, args)
        with patch.object(snapshot_admission, "git_output", side_effect=wrong_submodule):
            with self.assertRaisesRegex(ValueError, "submodule mismatch"):
                snapshot_admission.admit(
                    self.contract, self.population, self.root / "source",
                    self.roots, self.root / "admission.json")

    def test_existing_output_is_preserved(self):
        output = self.root / "admission.json"
        output.write_text("preserve")
        with self.assertRaises(FileExistsError):
            self._admit()
        self.assertEqual(output.read_text(), "preserve")

    def test_file_hashes_and_aggregate_are_content_bound(self):
        result = self._admit()
        item = result["snapshots"]["actionmesh"]["files"][0]
        self.assertEqual(
            item["sha256"],
            hashlib.sha256((self.roots["actionmesh"] / item["path"]).read_bytes()).hexdigest())
        canonical = json.dumps(
            result["snapshots"]["actionmesh"]["files"], sort_keys=True,
            separators=(",", ":"), ensure_ascii=False).encode()
        self.assertEqual(result["snapshots"]["actionmesh"]["manifest_sha256"],
                         hashlib.sha256(canonical).hexdigest())


class SnapshotPlanTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(snapshot_planner, "Snapshot plan builder missing")
        self.assertIsNotNone(run_experiments, "Installed harness modules missing")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        code = self.root / "actionmesh/research_math"
        code.mkdir(parents=True)
        (code / "__init__.py").write_text("")
        (code / "snapshot_admission.py").write_text("fixture source\n")
        (self.root / "actionmesh/prepare_actionbench_snapshots.py").write_text(
            "fixture plan source\n")
        self.population_path = (
            self.root / "actionmesh/research_overnight/assets/actionbench_population.json")
        self.population_path.parent.mkdir(parents=True)
        population = {"dataset": "facebook/actionbench", "revision": "a" * 40,
                      "uids": ["uid"]}
        self.population_path.write_text(json.dumps(population))
        population_ref = snapshot_planner.file_ref(self.root, self.population_path)
        self.contract_path = self.root / "docs/snapshot.json"
        self.contract_path.parent.mkdir()
        self.contract_path.write_text(json.dumps({
            "snapshots": {"dataset": {"revision": "a" * 40}},
            "dataset_layout": {"uid_source_ref": population_ref},
        }))
        self.source_root = self.root / "official-source"
        self.source_root.mkdir()
        self.snapshot_roots = {}
        for key in snapshot_admission.SNAPSHOT_KEYS:
            path = self.root / ("snapshot-" + key)
            path.mkdir()
            self.snapshot_roots[key] = path
        self.skill_dir = Path(run_experiments.__file__).resolve().parents[1]

    def build(self, name="plans"):
        completed = SimpleNamespace(stdout="f" * 40 + "\n")
        with patch.object(snapshot_planner.subprocess, "run", return_value=completed):
            return snapshot_planner.build_plan(
                self.root, skill_dir=self.skill_dir,
                contract_path=self.contract_path,
                population_path=self.population_path,
                source_root=self.source_root,
                snapshot_roots=self.snapshot_roots,
                plan_dir=self.root / name, run_id="fixture-snapshot",
                wall_seconds=120, ram_mib=512, cpu_cores=1)

    def test_plan_is_single_attempt_cpu_only_and_runs_only_admission(self):
        outer = self.build()
        native = json.loads((self.root / "plans/native.json").read_text())
        self.assertEqual(native["limits"]["max_attempts"], 1)
        self.assertEqual(native["jobs"][0]["command"][1:3],
                         ["-m", "research_math.snapshot_admission"])
        self.assertEqual(outer["tasks"][0]["resources"]["gpu_count"], 0)
        self.assertEqual(outer["limits"]["max_gpu_task_seconds"], 0)

    def test_plan_binds_the_exact_population_file(self):
        contract = json.loads(self.contract_path.read_text())
        contract["dataset_layout"]["uid_source_ref"]["sha256"] = "0" * 64
        self.contract_path.write_text(json.dumps(contract))
        with self.assertRaisesRegex(ValueError, "exact population file"):
            self.build()
        self.assertFalse((self.root / "plans").exists())

    def test_missing_snapshot_root_is_rejected_before_plan_write(self):
        self.snapshot_roots["rmbg"] = self.root / "missing-rmbg"
        with self.assertRaises(FileNotFoundError):
            self.build()
        self.assertFalse((self.root / "plans").exists())

    def test_existing_plan_directory_is_preserved(self):
        self.build()
        before = (self.root / "plans/native.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.build()
        self.assertEqual((self.root / "plans/native.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
