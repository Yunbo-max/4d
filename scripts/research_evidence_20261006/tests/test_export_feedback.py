"""File/receipt transport tests. Opaque bytes are never benchmark inputs."""
from __future__ import annotations

import hashlib
import fcntl
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "export_feedback.py"
spec = importlib.util.spec_from_file_location("export_feedback", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

RECORDED_PROTOCOL = {
    "n_pts_chamfer": 100000, "n_pts_icp": 10000, "icp_initial_rotations": 24,
    "icp_iterations": 200, "icp_learning_rate": .01,
    "icp_transform": "rotation + translation + anisotropic scale (official)",
    "cd_query_points_per_direction": 10000,
    "cd_query_subsampling_seeds": {"predicted": 44, "ground_truth": 45},
    "cd_reference_tree_points": 100000,
    "cd_distance": "unsquared Euclidean; sum of both directional means",
    "cd_3d_alignment": "independent ICP each frame",
    "cd_4d_alignment": "one ICP from frame zero, applied to all frames",
    "cd_motion_alignment": "same first-frame ICP as CD4D",
    "cd_motion_correspondence": "predicted points share frame-zero face IDs and barycentric coordinates; nearest-neighbor matches to GT established only in frame zero",
    "units": "raw GT coordinate units; no extra normalization or scale multiplier",
    "gt_correspondence_assumption": "surfaces.npy point index is a material correspondence across time, as assumed by the official evaluator",
    "no_extra_preprocessing": "no crop, camera fit, axis flip, interpolation, or frame truncation",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + "\n")


class ReceiptTransport(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        # macOS exposes /var through a system symlink to /private/var. Resolve
        # the fixture root so the exporter tests only exercise links created by
        # the fixture, not the host's canonical temporary-directory alias.
        self.root = Path(self.temporary.name).resolve() / "project"
        self.run = self.root / "outputs" / "original"
        self.run.mkdir(parents=True)
        self.dest = self.root / "outputs" / "export"
        self.uid = "receipt-transport-case"
        self.unit = self.run / "units" / ("gen-" + self.uid)
        self.unit.mkdir(parents=True)
        self.gt_name = "data/data/" + self.uid + "/surfaces.npy"
        for name in ["research_census_eval.py", "repo/actionbench/benchmark.py", self.gt_name, "weight.dat"]:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"opaque transport fixture; not executable scientific data\n")
        # This constant-only file is read by AST, never imported/executed.
        (self.root / "research_census_eval.py").write_text("PROTOCOL = " + repr(RECORDED_PROTOCOL) + "\n")
        frame_paths = []
        for index in range(16):
            path = self.root / "data" / "data" / self.uid / "imgs" / f"{index:02d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"opaque image transport bytes")
            frame_paths.append(path)
        source = str(self.root / "research_census_eval.py")
        self.protocol = {
            "schema_version": 1, "suite": "generation", "root": str(self.root),
            "runner_sources": {source: sha(self.root / "research_census_eval.py")},
            "native_source": {"native_files_sha256": {
                str(self.root / "repo/actionbench/benchmark.py"): sha(self.root / "repo/actionbench/benchmark.py")}},
            "generation": {"cohort": {"uids": [self.uid],
                "native_inference": {"frames": 16, "stage0_steps": 100, "stageI_steps": 30, "cfg": 7.5, "seed": 42},
                "native_evaluation": {"frames": 16, "sampling_seed": 44, "surface_samples": 100000,
                    "icp_samples": 10000, "icp_rotations": 24, "icp_iterations": 200}},
                "data_root": str(self.root / "data"),
                "data_records": {str(self.root / self.gt_name): {"sha256": sha(self.root / self.gt_name), "bytes": (self.root / self.gt_name).stat().st_size}},
                "weight_records": {str(self.root / "weight.dat"): sha(self.root / "weight.dat")}},
        }
        self.protocol["generation"]["data_records"].update({str(p): {"sha256": sha(p), "bytes": p.stat().st_size} for p in frame_paths})
        self.signature = hashlib.sha256(json.dumps(self.protocol, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        put(self.run / "protocol.json", self.protocol)
        put(self.run / "window.json", {"fingerprint": self.signature})
        self.queue = {"status": "inventory_complete", "unit_ids": ["gen-" + self.uid],
            "active_process": None, "active_unit": None,
            "units": {"gen-" + self.uid: {"family": "generation", "status": "complete", "attempts": [
                {"status": "completed", "returncode": 0, "ended_utc": "2026-10-06T00:00:00+00:00",
                 "log": str(self.run / "logs" / "attempt1.log")}]}}}
        put(self.run / "queue.json", self.queue)
        put(self.run / "completion-status.json", {"status": "complete", "fingerprint": self.signature, "process": None})
        (self.run / "logs").mkdir()
        (self.run / "logs" / "attempt1.log").write_text("attempt log\n")
        # Preserve this obsolete progress report; export must not rewrite it.
        put(self.run / "completion-verification.json", {"verified_complete_pairs": 0})
        self.native = {"cd_3d": 1.0, "cd_4d": 2.0, "cd_motion": 3.0}
        self.stationary = {"cd_3d": 2.0, "cd_4d": 4.0, "cd_motion": 6.0}
        self.phase_files = {}
        for phase in ["generation", "stationary", "native_score", "stationary_score"]:
            directory = self.unit / (phase + "__attempt1")
            directory.mkdir()
            if phase.endswith("score"):
                metric = self.native if phase == "native_score" else self.stationary
                generated_dir = self.unit / (("generation" if phase == "native_score" else "stationary") + "__attempt1")
                entry = {"case_id": phase, "uid": self.uid, "case_dir": str(generated_dir)}
                put(directory / "manifest.json", {"cases": [entry]})
                (directory / "score.log").write_text("opaque recorded score log\n")
                put(directory / "metrics.json", {"protocol": dict(RECORDED_PROTOCOL, sampling_seed=44),
                    "evaluator_sha256": sha(self.root / "research_census_eval.py"),
                    "official_source": {"sha256": {"benchmark.py": sha(self.root / "repo/actionbench/benchmark.py")}},
                    "denominator": {"frozen": True, "manifest": str(directory / "manifest.json"),
                        "manifest_sha256": sha(directory / "manifest.json"), "n_declared": 1},
                    "cases": [{**entry, "manifest_entry": entry, "status": "success", "generation_status": "completed", "inputs": {
                        "sequence": {"path": str(self.unit / (("generation" if phase == "native_score" else "stationary") + "__attempt1") / "sequence.npz"),
                            "sha256": sha(self.unit / (("generation" if phase == "native_score" else "stationary") + "__attempt1") / "sequence.npz")},
                        "ground_truth": {"path": str(self.root / self.gt_name), "sha256": sha(self.root / self.gt_name)},
                        "generation_report": {"path": str(generated_dir / "report.json"), "sha256": sha(generated_dir / "report.json")}},
                        "shapes": {"vertices": [16, 3, 3], "faces": [1, 3], "gt_positions": [16, 100000, 3]}, **metric}],
                    "summary": {"n_total": 1, "n_success": 1, "n_failed": 0, "n_pending": 0,
                        "n_validated_only": 0, "success_rate": 1., "n_unique_assets": 1, "n_assets_with_success": 1,
                        "successful_case_ids": [phase], "means": metric, "asset_balanced_means": metric,
                        "asset_results": [{"uid": self.uid, "n_runs": 1, "n_success": 1, "means": metric}]}})
                files = [directory / "metrics.json", directory / "manifest.json", directory / "score.log"]
                result = {"metrics": metric, "native_report": str(files[0])}
            else:
                (directory / "sequence.npz").write_bytes(b"opaque raw product, deliberately not an array")
                report = {"status": "completed", "uid": self.uid, "seed": 42, "frames": 16,
                          "stage0_steps": 100, "stage1_steps": 30, "guidance_scale": 7.5,
                          "vertices_per_frame": 3, "faces": 1,
                          "sha256": {"sequence.npz": sha(directory / "sequence.npz")}}
                nested = []
                if phase == "generation":
                    inputs, preprocessing = [], []
                    for index, source in enumerate(frame_paths):
                        names = {k: directory / (k + "-frames") / f"{index:02d}.png" for k in ("input", "masked", "processed")}
                        for path in names.values():
                            path.parent.mkdir(exist_ok=True)
                            path.write_bytes(b"opaque cached image bytes")
                        inputs.append({"index": index, "source": str(source), "source_sha256": sha(source),
                                       "saved_file": str(names["input"].relative_to(directory)), "saved_sha256": sha(names["input"])})
                        preprocessing.append({"index": index, "masked_file": str(names["masked"].relative_to(directory)),
                            "masked_sha256": sha(names["masked"]), "processed_file": str(names["processed"].relative_to(directory)), "processed_sha256": sha(names["processed"])})
                    put(directory / "inputs.json", {"uid": self.uid, "records": inputs,
                        "frame_indices": list(range(16)), "timesteps": list(range(16))})
                    put(directory / "preprocessing.json", {"records": preprocessing})
                    nested = [directory / "inputs.json", directory / "preprocessing.json"]
                    report["sha256"].update({p.name: sha(p) for p in nested})
                if phase == "stationary":
                    report["source_sequence_sha256"] = sha(self.unit / "generation__attempt1" / "sequence.npz")
                put(directory / "report.json", report)
                files = [directory / "sequence.npz", directory / "report.json", *nested]
                result = {"directory": str(directory)}
            self.phase_files[phase] = files
            self.receipt(self.unit / (phase + ".receipt.json"), self.signature + ":" + phase, files, result=result)
        pair = {"uid": self.uid, "native": self.native, "stationary": self.stationary,
                "native_minus_stationary": {k: self.native[k] - self.stationary[k] for k in self.native}}
        put(self.unit / "pair.json", pair)
        put(self.run / "summary.json", {"queue_status": "inventory_complete", "scientific_gate_pass": False,
            "candidate_methods_tested": False, "generation": {"declared_assets": 1, "complete_pairs": 1, "pairs": [pair],
                "pending_or_failed": [], "inspection_by_native_cd_motion": [self.uid], "failure_labels_inferred": False,
                "native_minus_stationary": {k: {"n_assets": 1, "mean": pair["native_minus_stationary"][k], "bootstrap_95_ci": None,
                    "scope": "conditional on this development cohort and inference seed; no method verdict"} for k in self.native}}})
        self.finalize()

    def receipt(self, path, fingerprint, files, **extra):
        put(path, {"status": "complete", "fingerprint": fingerprint,
                   "artifacts": {str(p.relative_to(path.parent)): {"sha256": sha(p), "bytes": p.stat().st_size} for p in files}, **extra})

    def finalize(self):
        files = [self.unit / "pair.json", *self.unit.glob("*.receipt.json")]
        files = [p for p in files if p.name != "receipt.json"]
        for phase in self.phase_files.values():
            files.extend(phase)
        self.receipt(self.unit / "receipt.json", self.signature, files, uid=self.uid, family="generation")

    def rehash_phase(self, phase):
        path = self.unit / (phase + ".receipt.json")
        value = json.loads(path.read_text())
        self.receipt(path, self.signature + ":" + phase, self.phase_files[phase], result=value["result"])
        self.finalize()

    def export(self, **kwargs):
        return module.export_feedback(self.root, self.run, self.dest, self.signature, **kwargs)

    def test_exports_the_full_referenced_raw_closure_and_preserves_old_checkpoint(self):
        old = (self.run / "completion-verification.json").read_bytes()
        result = self.export()
        self.assertEqual(result["verified_complete_pairs"], 1)
        self.assertFalse(result["scientific_gate_pass"])
        self.assertFalse(result["native_replay_performed"])
        self.assertEqual(old, (self.run / "completion-verification.json").read_bytes())
        # Parts concatenate byte-for-byte to one gzip tar; no extraction is needed here.
        members = {}
        data = b"".join((self.dest / p["name"]).read_bytes() for p in result["archives"])
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            for entry in archive.getmembers():
                self.assertTrue(entry.isfile())
                members[entry.name] = archive.extractfile(entry).read()
        for phase in self.phase_files.values():
            for path in phase:
                self.assertEqual(members["run/" + path.relative_to(self.run).as_posix()], path.read_bytes())
        self.assertIn("source/research_census_eval.py", members)

    def rejected(self):
        with self.assertRaises(module.EvidenceError):
            self.export()
        self.assertFalse(self.dest.exists())

    def test_incomplete_inventory_cannot_be_exported_as_final(self):
        self.queue["units"]["gen-" + self.uid]["status"] = "pending"
        put(self.run / "queue.json", self.queue)
        self.rejected()

    def test_final_receipt_must_include_nested_phase_products(self):
        path = self.unit / "receipt.json"
        value = json.loads(path.read_text())
        value["artifacts"].pop(str(self.phase_files["native_score"][0].relative_to(self.unit)))
        put(path, value)
        self.rejected()

    def test_missing_raw_product_is_not_replaced_by_its_hash(self):
        self.phase_files["generation"][0].unlink()
        self.rejected()

    def test_changed_raw_bytes_are_rejected(self):
        self.phase_files["generation"][0].write_bytes(b"changed")
        self.rejected()

    def test_absolute_or_parent_paths_in_receipts_are_rejected(self):
        path = self.unit / "receipt.json"
        value = json.loads(path.read_text())
        value["artifacts"][str(self.root / "weight.dat")] = {"sha256": sha(self.root / "weight.dat"), "bytes": (self.root / "weight.dat").stat().st_size}
        put(path, value)
        self.rejected()

    def test_symlink_is_rejected_even_when_it_points_inside_the_run(self):
        (self.run / "linked.log").symlink_to(self.run / "logs" / "attempt1.log")
        self.rejected()

    def test_bound_protocol_cannot_be_changed(self):
        self.protocol["generation"]["cohort"]["native_inference"]["seed"] = 999
        put(self.run / "protocol.json", self.protocol)
        self.rejected()

    def test_source_and_input_and_weight_hashes_are_all_checked(self):
        for name in ["research_census_eval.py", self.gt_name, "weight.dat"]:
            with self.subTest(name=name):
                path = self.root / name
                old = path.read_bytes()
                path.write_bytes(b"changed")
                self.rejected()
                path.write_bytes(old)

    def test_pair_cannot_disagree_with_hash_valid_score_report(self):
        path = self.phase_files["native_score"][0]
        value = json.loads(path.read_text())
        value["cases"][0]["cd_motion"] = 9.0
        put(path, value)
        receipt = self.unit / "native_score.receipt.json"
        value = json.loads(receipt.read_text())
        self.rehash_phase("native_score")
        self.rejected()

    def test_native_sampling_budget_cannot_be_reduced(self):
        path = self.phase_files["native_score"][0]
        value = json.loads(path.read_text())
        value["protocol"]["n_pts_chamfer"] = 1000
        put(path, value)
        receipt = self.unit / "native_score.receipt.json"
        value = json.loads(receipt.read_text())
        self.rehash_phase("native_score")
        self.rejected()

    def test_summary_cannot_hide_a_pair(self):
        put(self.run / "summary.json", {"generation": {"declared_assets": 1, "complete_pairs": 1, "pairs": []}})
        self.rejected()

    def test_duplicate_json_keys_are_rejected(self):
        (self.run / "queue.json").write_text('{"status":"running","status":"inventory_complete"}')
        self.rejected()

    def test_nonfinite_metric_report_is_rejected(self):
        path = self.phase_files["native_score"][0]
        path.write_text(path.read_text().replace('"cd_motion": 3.0', '"cd_motion": 1e999'))
        receipt = self.unit / "native_score.receipt.json"
        value = json.loads(receipt.read_text())
        self.rehash_phase("native_score")
        self.rejected()

    def test_output_parent_symlink_cannot_redirect_export_into_the_run(self):
        link = self.root / "export_link"
        link.symlink_to(self.run, target_is_directory=True)
        self.dest = link / "new_export"
        self.rejected()

    def test_generation_report_must_itself_be_phase_bound(self):
        path = self.unit / "generation.receipt.json"
        value = json.loads(path.read_text())
        value["artifacts"].pop("generation__attempt1/report.json")
        put(path, value)
        self.finalize()
        self.rejected()

    def test_live_worker_is_rejected(self):
        self.queue["active_process"] = {"pid": os.getpid(), "host": socket.gethostname(), "start_ticks": None, "boot_id": None}
        put(self.run / "queue.json", self.queue)
        self.rejected()

    def test_unreconciled_remote_supervisor_is_rejected(self):
        put(self.run / "completion-status.json", {"status": "complete", "fingerprint": self.signature,
            "process": {"pid": 1, "host": "unreconciled-other-host", "start_ticks": None, "boot_id": None}})
        self.rejected()

    def test_existing_destination_is_preserved(self):
        self.dest.mkdir()
        (self.dest / "keep.txt").write_text("preserve me")
        with self.assertRaises(module.EvidenceError):
            self.export()
        self.assertEqual((self.dest / "keep.txt").read_text(), "preserve me")

    def test_destination_inside_run_is_rejected_without_mutation(self):
        self.dest = self.run / "export"
        self.rejected()

    def test_failed_attempt_logs_remain_in_the_bundle(self):
        (self.run / "logs" / "failed_attempt.log").write_text("failure retained\n")
        result = self.export()
        data = b"".join((self.dest / p["name"]).read_bytes() for p in result["archives"])
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            self.assertEqual(archive.extractfile("run/logs/failed_attempt.log").read(), b"failure retained\n")

    def test_archive_parts_obey_the_transport_limit(self):
        (self.run / "retained_opaque_output.bin").write_bytes(os.urandom(180000))
        result = self.export(part_bytes=65536)
        self.assertGreater(len(result["archives"]), 1)
        for part in result["archives"]:
            path = self.dest / part["name"]
            self.assertLessEqual(path.stat().st_size, 65536)
            self.assertEqual(sha(path), part["sha256"])
        data = b"".join((self.dest / p["name"]).read_bytes() for p in result["archives"])
        self.assertEqual(hashlib.sha256(data).hexdigest(), result["archive_sha256"])

    def test_relocated_project_retains_original_path_and_fingerprint(self):
        moved = self.root.with_name("relocated")
        relative_run = self.run.relative_to(self.root)
        self.root.rename(moved)
        self.root, self.run, self.dest = moved, moved / relative_run, moved / "outputs" / "export"
        result = self.export()
        self.assertEqual(result["fingerprint"], self.signature)

    def test_cli_reports_success_without_claiming_native_replay(self):
        result = subprocess.run([sys.executable, str(SCRIPT), "--project-root", str(self.root),
            "--run-dir", str(self.run), "--destination", str(self.dest), "--expected-fingerprint", self.signature],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        report = json.loads(result.stdout)
        self.assertEqual(report["verified_complete_pairs"], 1)
        self.assertFalse(report["native_replay_performed"])

    def test_cli_returns_blocked_without_a_partial_export(self):
        self.phase_files["generation"][0].unlink()
        result = subprocess.run([sys.executable, str(SCRIPT), "--project-root", str(self.root),
            "--run-dir", str(self.run), "--destination", str(self.dest), "--expected-fingerprint", self.signature],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 2, result.stderr + result.stdout)
        self.assertEqual(json.loads(result.stdout)["status"], "blocked")
        self.assertFalse(self.dest.exists())

    def test_held_original_runner_lock_blocks_export(self):
        with (self.run / "runner.lock").open("w") as stream:
            stream.write("existing owner\n")
            stream.flush()
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.rejected()

    def test_reduced_query_budget_or_changed_query_seed_is_rejected(self):
        path = self.phase_files["native_score"][0]
        original = json.loads(path.read_text())
        for key, bad in [("cd_query_points_per_direction", 100), ("cd_reference_tree_points", 100),
                         ("cd_query_subsampling_seeds", {"predicted": 1, "ground_truth": 2})]:
            with self.subTest(key=key):
                value = copy.deepcopy(original)
                value["protocol"][key] = bad
                put(path, value)
                self.rehash_phase("native_score")
                self.rejected()

    def test_one_frame_recorded_shape_cannot_be_certified_as_full_timeline(self):
        path = self.phase_files["native_score"][0]
        value = json.loads(path.read_text())
        value["cases"][0]["shapes"]["vertices"][0] = 1
        value["cases"][0]["shapes"]["gt_positions"][0] = 1
        put(path, value)
        self.rehash_phase("native_score")
        self.rejected()

    def test_denominator_cannot_point_to_an_absent_manifest(self):
        path = self.phase_files["native_score"][0]
        value = json.loads(path.read_text())
        value["denominator"]["manifest"] = str(path.parent / "absent.json")
        put(path, value)
        self.rehash_phase("native_score")
        self.rejected()

    def test_nested_saved_input_or_processed_image_cannot_be_missing(self):
        for prefix in ["input", "masked", "processed"]:
            with self.subTest(prefix=prefix):
                path = self.unit / "generation__attempt1" / (prefix + "-frames") / "00.png"
                original = path.read_bytes()
                path.unlink()
                self.rejected()
                path.write_bytes(original)

    def test_summary_aggregates_and_pending_state_cannot_contradict_pairs(self):
        path = self.run / "summary.json"
        original = json.loads(path.read_text())
        for field, bad in [("mean", 9876), ("n_assets", 999), ("bootstrap_95_ci", [0, 99])]:
            with self.subTest(field=field):
                value = copy.deepcopy(original)
                value["generation"]["native_minus_stationary"]["cd_motion"][field] = bad
                put(path, value)
                self.rejected()
        value = copy.deepcopy(original)
        value["generation"]["pending_or_failed"] = [{"uid": self.uid, "status": "failed"}]
        put(path, value)
        self.rejected()

    def test_preflight_process_and_log_are_reconciled(self):
        self.queue["preflight_attempts"] = [{"status": "completed", "returncode": 0,
            "ended_utc": "2026-10-06T00:00:00+00:00", "log": str(self.run / "logs/attempt1.log"),
            "process": {"pid": os.getpid(), "host": socket.gethostname(), "start_ticks": None}}]
        put(self.run / "queue.json", self.queue)
        self.rejected()
        self.queue["preflight_attempts"][0]["process"] = None
        self.queue["preflight_attempts"][0]["log"] = str(self.run / "logs/absent.log")
        put(self.run / "queue.json", self.queue)
        self.rejected()

    def test_transient_json_rewrite_is_not_reset_by_repeated_hashing(self):
        path = self.unit / "generation__attempt1/report.json"
        original = path.read_bytes()
        reader = module.read_json

        def transient(target, *args, **kwargs):
            if Path(target) != path:
                return reader(target, *args, **kwargs)
            changed = json.loads(original)
            changed["transient_note"] = "a concurrent writer changed this file"
            put(path, changed)
            try:
                return reader(target, *args, **kwargs)
            finally:
                path.write_bytes(original)

        with patch.object(module, "read_json", transient):
            self.rejected()

    def test_malformed_official_source_returns_blocked(self):
        path = self.phase_files["native_score"][0]
        value = json.loads(path.read_text())
        value["official_source"] = None
        put(path, value)
        self.rehash_phase("native_score")
        self.rejected()


if __name__ == "__main__":
    unittest.main()
