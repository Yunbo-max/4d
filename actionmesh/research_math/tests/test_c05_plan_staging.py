"""Real C05 plan staging checks; authored for Local execution only.

The tests invoke the installed runner's real ``_stage`` copier/remapper, but
never execute either staged command.  Opaque fixture bytes stand in for native
arrays, so this is an isolation/plan contract check, not scientific evidence.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import prepare_c05_mode_bank as mode_plan
import prepare_c05_native_acceptance as acceptance_plan
import prepare_spatial_mode_candidate as candidate_plan
import run_experiments
from research_math.spatial_mode_candidate import METHOD_IDS


PROJECT = Path(__file__).resolve().parents[3]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def _ref(root: Path, path: Path) -> dict:
    return {"path": path.relative_to(root).as_posix(), "sha256": _digest(path)}


def _local_ref(directory: Path, path: Path) -> dict:
    return {"path": path.relative_to(directory).as_posix(),
            "sha256": _digest(path), "bytes": path.stat().st_size}


def _sized_ref(root: Path, path: Path) -> dict:
    return {**_ref(root, path), "bytes": path.stat().st_size}


def _flag(command: list[str], name: str) -> Path:
    return Path(command[command.index(name) + 1]).resolve()


def _flag_values(command: list[str], name: str) -> list[Path]:
    return [Path(command[index + 1]).resolve()
            for index, value in enumerate(command) if value == name]


def _copy_project_sources(root: Path) -> None:
    names = (
        "research_math/__init__.py",
        "research_math/c05_mode_bank.py",
        "research_math/spatial_mode_candidate.py",
        "research_math/c05_candidate_artifacts.py",
        "prepare_c05_mode_bank.py",
        "prepare_spatial_mode_candidate.py",
    )
    for name in names:
        source = PROJECT / "actionmesh" / name
        destination = root / "actionmesh" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _copy_acceptance_sources(root: Path) -> None:
    """Mirror the real acceptance import closure into the controller fixture."""
    for source in acceptance_plan.acceptance_sources(PROJECT):
        destination = root / source.relative_to(PROJECT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _controller_fixture(root: Path) -> dict:
    """Create only the metadata and opaque files needed to build both plans."""
    _copy_project_sources(root)
    source = root / "vendor/actionmesh-source"
    for relative in (
        "actionmesh/pipeline.py",
        "actionmesh/model/temporal_autoencoder.py",
        "actionmesh/model/utils/storage.py",
        "actionmesh/scheduler/scheduler.py",
        "actionmesh/io/video_input.py",
    ):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# staged native source fixture\n")
    config = source / "actionmesh/configs/actionmesh-medium.py"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text("# staged prospective config fixture\n")

    frames = root / "inputs/actionbench/unit/images"
    frames.mkdir(parents=True)
    for index in range(16):
        (frames / f"{index:02d}.png").write_bytes(
            b"opaque PNG fixture " + str(index).encode())

    retained = root / "inputs/actionbench/unit/b0"
    retained.mkdir(parents=True)
    b0_sequence = retained / "sequence.npz"
    b0_sequence.write_bytes(b"opaque retained B0 sequence fixture\n")
    b0_report = retained / "report.json"
    _write_json(b0_report, {
        "status": "completed", "uid": "c05-staging-fixture", "seed": 42,
        "sha256": {"sequence.npz": _digest(b0_sequence)},
    })

    environment = root / "inputs/native-runtime/environment.json"
    _write_json(environment, {"gpu_uuid": "GPU-c05-staging-fixture"})
    cache = root / "model-cache"
    weight_rows = []
    for model_root in ("ActionMesh", "TripoSG", "dinov2", "RMBG"):
        weight = cache / "weights" / model_root / "tiny.bin"
        weight.parent.mkdir(parents=True)
        weight.write_bytes(("opaque " + model_root + " cache fixture\n").encode())
        weight_rows.append({
            "path": f"weights/{model_root}/tiny.bin", "size": weight.stat().st_size,
            "sha256": _digest(weight),
        })
    weights_manifest = cache / "weights-manifest.json"
    _write_json(weights_manifest, weight_rows)
    authorization = root / "inputs/c05/gpu-resume-authorization.json"
    _write_json(authorization, {
        "kind": mode_plan.AUTHORIZATION_KIND, "version": 1,
        "candidate_id": mode_plan.CANDIDATE_ID,
        "scope": "single_c05_mode_bank_attempt", "status": "approved",
        "gpu_stop_lifted": True, "uid": "c05-staging-fixture",
        "outer_seed": 42, "branch_count": 3,
        "gpu_uuid": "GPU-c05-staging-fixture", "run_id": "c05-mode-fixture",
        "environment_sha256": _digest(environment), "max_attempts": 1,
        "max_retries_per_trial": 0,
        "expires_at": (datetime.now(timezone.utc) + timedelta(days=2)).isoformat(),
    })
    return {
        "source": source, "frames": frames, "config": config,
        "b0_sequence": b0_sequence, "b0_report": b0_report,
        "environment": environment, "weights_manifest": weights_manifest,
        "authorization": authorization,
        "skill": Path(run_experiments.__file__).resolve().parents[1],
    }


def _build_mode_plan(root: Path, fixture: dict, *, plan_name: str):
    return mode_plan.build_plans(
        root, skill_dir=fixture["skill"], source_root=fixture["source"],
        frames=fixture["frames"], b0_sequence=fixture["b0_sequence"],
        b0_report=fixture["b0_report"], authorization=fixture["authorization"],
        environment=fixture["environment"],
        weights_manifest=fixture["weights_manifest"], config=fixture["config"],
        plan_dir=root / "plans" / plan_name,
        output=root / "retained/mode-bank", uid="c05-staging-fixture",
        outer_seed=42, branch_count=3, run_id="c05-mode-fixture",
        wall_seconds=300, gpu_peak_mib=1024)


def _materialize_opaque_mode_bank(root: Path, fixture: dict, native: dict,
                                  staged_command: list[str], staged_cwd: Path,
                                  workspace: Path, attempt: Path
                                  ) -> tuple[Path, Path, Path]:
    """Build producer metadata exactly, without running the scientific producer."""
    request_path = workspace / "plans/mode-for-candidate/request.json"
    request = json.loads(request_path.read_text())
    mode_root = workspace / request["output_relative"]
    sequences = mode_root / "sequences"
    sequences.mkdir(parents=True)
    bank = mode_root / "mode-bank.npz"
    bank.write_bytes(b"opaque mode-bank fixture; never interpreted by this test\n")
    seeds = [42, 104771, 209500]
    branches = []
    for index, seed in enumerate(seeds):
        path = sequences / f"branch-{index:04d}.npz"
        path.write_bytes(b"opaque branch fixture " + str(index).encode() + b"\n")
        branches.append({"branch_index": index, "stage1_seed": seed,
                         "sequence_ref": _local_ref(mode_root, path)})
    producer_inputs = [*request["producer_input_refs"],
                       _ref(workspace, request_path)]
    false_scope = {
        "native_scientific_qualification": False,
        "scientific_effect_qualification": False,
        "candidate_methods_tested": False,
        "dispatch_ready": False,
    }
    manifest = {
        "kind": candidate_plan.MODE_BANK_KIND, "version": 1,
        "status": "completed_unqualified", "candidate_id": "C05",
        "outer_generation_seed": 42, "inner_stage1_seeds": seeds,
        "branch_count": 3, "full_sequences_retained": 3,
        "generation_parameters": request["generation_parameters"],
        "provenance": request["provenance"],
        "producer_code_sha256": _digest(
            root / "actionmesh/research_math/c05_mode_bank.py"),
        "producer_input_refs": producer_inputs,
        "producer_code_refs": request["producer_code_refs"],
        "mode_bank_ref": _local_ref(mode_root, bank), "branches": branches,
        **false_scope,
    }
    manifest_path = mode_root / "raw-manifest.json"
    _write_json(manifest_path, manifest)
    result = {
        "kind": "c05-same-anchor-stage1-mode-bank-result", "version": 1,
        "status": "completed_unqualified", "candidate_id": "C05",
        "outer_generation_seed": 42, "inner_stage1_seeds": seeds,
        "full_sequences_retained": 3,
        "manifest_ref": _local_ref(mode_root, manifest_path),
        "producer_input_refs": producer_inputs,
        "producer_code_refs": request["producer_code_refs"],
        **false_scope,
    }
    result_path = mode_root / "result.json"
    _write_json(result_path, result)
    parity_path = workspace / "retained/mode-bank-b0-parity.json"
    _write_json(parity_path, {
        "kind": "c05-b0-parity-receipt", "version": 1,
        "candidate_id": mode_plan.CANDIDATE_ID,
        "uid": "c05-staging-fixture", "outer_seed": 42, "matches": True,
        "comparison": "exact array equality for all native sequence fields",
        "mode_bank_result_ref": _ref(workspace, result_path),
        "mode_bank_manifest_ref": _ref(workspace, manifest_path),
        "mode_bank_array_ref": _ref(workspace, bank),
        "branch_zero_ref": _ref(workspace, sequences / "branch-0000.npz"),
        "retained_b0_sequence_ref": _ref(
            workspace, workspace / fixture["b0_sequence"].relative_to(root)),
        "retained_b0_report_ref": _ref(
            workspace, workspace / fixture["b0_report"].relative_to(root)),
        "candidate_methods_tested": False,
        "scientific_effect_qualification": False, "native_qualified": False,
    })
    job = native["jobs"][0]
    output_refs = [_ref(root, workspace / relative) for relative in job["output_paths"]]
    attempt_record = {
        "attempt_id": attempt.name,
        "attempt_path": attempt.relative_to(root).as_posix(),
        "trial_id": job["trial_id"], "retry_index": 0,
        "status": "completed", "exit_code": 0,
        "command": staged_command, "cwd": str(staged_cwd),
        "input_refs": job["input_refs"], "code_refs": job["code_refs"],
        "seed": job["seed"], "group": job["group"], "arm_role": job["arm_role"],
        "output_refs": output_refs,
    }
    receipt = {
        "status": "completed", "run_id": native["run_id"],
        "plan_digest": native["plan_digest"], "purpose": native["purpose"],
        "evidence_mode": native["evidence_mode"],
        "provenance": native["provenance"], "attempts": [attempt_record],
    }
    receipt_path = root / native["output_root"] / native["run_id"] / "receipt.json"
    _write_json(receipt_path, receipt)
    return mode_root, parity_path, receipt_path


class C05RealStagingTests(unittest.TestCase):
    def test_exact_external_cache_rejects_every_inventory_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            fixture = _controller_fixture(root)
            manifest = json.loads(fixture["weights_manifest"].read_text())
            cache = fixture["weights_manifest"].parent
            initial = mode_plan.validate_exact_cache(cache, manifest)
            self.assertEqual(len(initial), 4)

            extra = cache / "weights/ActionMesh/unlisted.bin"
            extra.write_bytes(b"unlisted\n")
            with self.assertRaisesRegex(ValueError, "inventory"):
                mode_plan.validate_exact_cache(cache, manifest)
            extra.unlink()

            partial = cache / "weights/ActionMesh/tiny.bin.aria2"
            partial.write_bytes(b"partial\n")
            with self.assertRaisesRegex(ValueError, "inventory|differs"):
                mode_plan.validate_exact_cache(cache, manifest)
            partial.unlink()

            link = cache / "weights/model-link"
            link.symlink_to(cache / "weights/ActionMesh", target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "non-symbolic"):
                mode_plan.validate_exact_cache(cache, manifest)
            link.unlink()

            changed = cache / manifest[0]["path"]
            changed.write_bytes(changed.read_bytes() + b"changed after preflight\n")
            with self.assertRaisesRegex(ValueError, "differs"):
                mode_plan.validate_exact_cache(cache, manifest)

    def test_mode_bank_plan_freezes_every_noncache_input_cross_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            fixture = _controller_fixture(root)
            native, _ = _build_mode_plan(root, fixture, plan_name="mode-stage")
            job = native["jobs"][0]
            attempt = root.parent / (root.name + "-cross-root-attempt")
            attempt.mkdir()
            self.addCleanup(shutil.rmtree, attempt, True)
            command, cwd, workspace = run_experiments._stage(root, job, attempt)

            self.assertEqual(cwd, workspace)
            self.assertEqual(_flag(command, "--external-cache-root"),
                             fixture["weights_manifest"].parent.resolve())
            for option in ("--request", "--project-sentinel", "--source-sentinel",
                           "--authorization", "--b0-sequence", "--b0-report",
                           "--environment", "--weights-manifest", "--config"):
                self.assertTrue(_flag(command, option).is_relative_to(workspace))
            frames = _flag_values(command, "--frame")
            closure = _flag_values(command, "--closure-file")
            self.assertEqual([path.name for path in frames],
                             [f"{index:02d}.png" for index in range(16)])
            self.assertTrue(all(path.is_relative_to(workspace) for path in frames))
            self.assertEqual(set(closure), {
                workspace / ref["path"]
                for ref in job["input_refs"] + job["code_refs"]})
            allowed_external = fixture["weights_manifest"].parent.resolve()
            for argument in command[2:]:
                value = Path(argument)
                if value.is_absolute() and value.is_relative_to(root):
                    self.assertEqual(value, allowed_external)
            request = json.loads(_flag(command, "--request").read_text())
            absolute_values = []
            def collect(value):
                if isinstance(value, dict):
                    for child in value.values():
                        collect(child)
                elif isinstance(value, list):
                    for child in value:
                        collect(child)
                elif isinstance(value, str) and Path(value).is_absolute():
                    absolute_values.append(value)
            collect(request)
            self.assertEqual(absolute_values, [])

            staged_frame = frames[0]
            staged_code = _flag(command, "--source-sentinel")
            frame_bytes, code_bytes = staged_frame.read_bytes(), staged_code.read_bytes()
            (fixture["frames"] / "00.png").write_bytes(b"controller mutation\n")
            (fixture["source"] / "actionmesh/pipeline.py").write_bytes(
                b"controller source mutation\n")
            self.assertEqual(staged_frame.read_bytes(), frame_bytes)
            self.assertEqual(staged_code.read_bytes(), code_bytes)
            self.assertTrue(all((workspace / relative).is_relative_to(workspace)
                                for relative in job["output_paths"]))
            self.assertEqual(job["output_paths"][0],
                             "retained/mode-bank/result.json")
            self.assertIn("retained/mode-bank-b0-parity.json",
                          job["output_paths"])

    def test_cpu_candidate_plan_uses_only_staged_recursive_closure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            fixture = _controller_fixture(root)
            producer, _ = _build_mode_plan(
                root, fixture, plan_name="mode-for-candidate")
            producer_attempt = (root / producer["output_root"] /
                                producer["run_id"] / "attempt-fixture")
            producer_attempt.mkdir(parents=True)
            producer_command, producer_cwd, producer_workspace = \
                run_experiments._stage(
                    root, producer["jobs"][0], producer_attempt)
            mode_root, parity, producer_receipt = _materialize_opaque_mode_bank(
                root, fixture, producer, producer_command, producer_cwd,
                producer_workspace, producer_attempt)
            native, _ = candidate_plan.build_plans(
                root, skill_dir=fixture["skill"], mode_bank_root=mode_root,
                mode_bank_plan=root / "plans/mode-for-candidate/native.json",
                mode_bank_receipt=producer_receipt,
                parity_receipt=parity,
                b0_sequence=(producer_workspace /
                             fixture["b0_sequence"].relative_to(root)),
                b0_report=(producer_workspace /
                           fixture["b0_report"].relative_to(root)),
                output=root / "outputs/c05",
                plan_dir=root / "plans/candidate", run_id="c05-cpu-fixture",
                landmark_count=16, cluster_radius=0.2,
                natural_gate_min_fraction=0.1, localized_radius=0.25,
                temperature=1.0, unary_weight=1.0, spatial_weight=1.0,
                max_sweeps=4, displacement_clip_multiplier=2.0,
                face_chunk_size=128, max_artifact_bytes=1024 * 1024,
                wall_seconds=300, ram_mib=2048)
            job = native["jobs"][0]
            attempt = (root / native["output_root"] / native["run_id"] /
                       "candidate-attempt-fixture")
            attempt.mkdir(parents=True)
            command, cwd, workspace = run_experiments._stage(root, job, attempt)

            self.assertEqual(cwd, workspace)
            for option in ("--project-sentinel", "--mode-bank-result",
                           "--parity-receipt", "--b0-sequence", "--b0-report"):
                self.assertTrue(_flag(command, option).is_relative_to(workspace))
            closure = _flag_values(command, "--closure-file")
            expected = {workspace / ref["path"]
                        for ref in job["input_refs"] + job["code_refs"]}
            self.assertEqual(set(closure), expected)
            self.assertEqual(int(command[
                command.index("--expected-closure-count") + 1]), len(expected))

            staged_bank = workspace / mode_root.relative_to(root) / "mode-bank.npz"
            staged_code = _flag(command, "--project-sentinel")
            bank_bytes, code_bytes = staged_bank.read_bytes(), staged_code.read_bytes()
            self.assertTrue(all(relative.startswith("outputs/c05/")
                                for relative in job["output_paths"]))
            self.assertFalse(any(relative.endswith("sequence.npz")
                                 for relative in job["output_paths"]))
            self.assertEqual(len(job["output_paths"]), 11)
            self.assertTrue(all((workspace / relative).is_relative_to(workspace)
                                for relative in job["output_paths"]))
            self.assertEqual(command[command.index("--output-relative") + 1],
                             "outputs/c05")
            self.assertFalse(any(Path(argument).is_absolute()
                                 and Path(argument).is_relative_to(root)
                                 and not Path(argument).is_relative_to(workspace)
                                 for argument in command[2:]))

            for relative in job["output_paths"]:
                path = workspace / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(("opaque terminal output " + relative + "\n").encode())
            # The staging contract test needs a builder-valid terminal document
            # and recursive source closure, but never interprets scientific arrays.
            staged_mode_root = workspace / mode_root.relative_to(root)
            staged_mode = staged_mode_root / "mode-bank.npz"
            staged_producer_root = workspace / producer_workspace.relative_to(root)
            staged_parity = workspace / parity.relative_to(root)
            staged_b0_sequence = (
                staged_producer_root / fixture["b0_sequence"].relative_to(root))
            staged_b0_report = (
                staged_producer_root / fixture["b0_report"].relative_to(root))
            staged_implementation = (
                workspace / "actionmesh/research_math/c05_candidate_artifacts.py")
            staged_math = workspace / "actionmesh/research_math/spatial_mode_candidate.py"
            staged_mode_implementation = (
                workspace / "actionmesh/research_math/c05_mode_bank.py")
            candidate_document = {
                "kind": "c05-spatial-mode-candidate", "version": 1,
                "candidate_id": candidate_plan.CANDIDATE_ID,
                "status": "completed_unqualified", "method_ids": METHOD_IDS,
                "input_refs": {
                    "mode_bank_result": _sized_ref(
                        workspace, staged_mode_root / "result.json"),
                    "mode_bank_manifest": _sized_ref(
                        workspace, staged_mode_root / "raw-manifest.json"),
                    "mode_bank": _sized_ref(workspace, staged_mode),
                    "branch0_sequence": _sized_ref(
                        workspace, staged_mode_root / "sequences/branch-0000.npz"),
                    "b0_sequence": _sized_ref(workspace, staged_b0_sequence),
                    "b0_report": _sized_ref(workspace, staged_b0_report),
                    "parity_receipt": _sized_ref(workspace, staged_parity),
                    "artifact_implementation": _sized_ref(
                        workspace, staged_implementation),
                    "math_implementation": _sized_ref(workspace, staged_math),
                    "mode_bank_implementation": _sized_ref(
                        workspace, staged_mode_implementation),
                },
                "native_scientific_qualification": False,
                "scientific_effect_qualification": False,
                "local_method_verified": False, "dispatch_ready": False,
            }
            _write_json(workspace / "outputs/c05/candidate.json", candidate_document)
            for role in acceptance_plan.ROLES:
                sequence = workspace / f"outputs/c05/roles/{role}/sequence.npz"
                sequence.parent.mkdir(parents=True, exist_ok=True)
                sequence.write_bytes(b"opaque role sequence for staging only\n")
            attempt_record = {
                "attempt_id": attempt.name,
                "attempt_path": attempt.relative_to(root).as_posix(),
                "trial_id": job["trial_id"], "retry_index": 0,
                "status": "completed", "exit_code": 0,
                "command": command, "cwd": str(cwd),
                "input_refs": job["input_refs"], "code_refs": job["code_refs"],
                "seed": job["seed"], "group": job["group"],
                "arm_role": job["arm_role"],
                "output_refs": [_ref(root, workspace / relative)
                                for relative in job["output_paths"]],
            }
            candidate_receipt = root / native["output_root"] / native["run_id"] / "receipt.json"
            _write_json(candidate_receipt, {
                "status": "completed", "run_id": native["run_id"],
                "plan_digest": native["plan_digest"], "purpose": native["purpose"],
                "evidence_mode": native["evidence_mode"],
                "provenance": native["provenance"], "attempts": [attempt_record],
            })
            _, _, retained_workspace, retained_candidate = \
                acceptance_plan._candidate_execution(
                    root, root / "plans/candidate/native.json",
                    candidate_receipt, run_experiments)
            self.assertEqual(retained_workspace, workspace)
            self.assertEqual(retained_candidate,
                             workspace / "outputs/c05/candidate.json")
            complete_receipt = json.loads(candidate_receipt.read_text())
            incomplete_receipt = json.loads(candidate_receipt.read_text())
            incomplete_receipt["attempts"][0]["output_refs"].pop()
            _write_json(candidate_receipt, incomplete_receipt)
            with self.assertRaisesRegex(ValueError, "output receipt inventory"):
                acceptance_plan._candidate_execution(
                    root, root / "plans/candidate/native.json",
                    candidate_receipt, run_experiments)
            _write_json(candidate_receipt, complete_receipt)

            _copy_acceptance_sources(root)
            acceptance_native, _ = acceptance_plan.build_plans(
                root, skill_dir=fixture["skill"],
                artifact_candidate=retained_candidate,
                candidate_plan=root / "plans/candidate/native.json",
                candidate_receipt=candidate_receipt,
                plan_dir=root / "plans/acceptance", run_id="c05-acceptance-fixture",
                wall_seconds=300, ram_mib=2048)
            acceptance_job = acceptance_native["jobs"][0]
            acceptance_attempt = (root / acceptance_native["output_root"] /
                                  acceptance_native["run_id"] /
                                  "acceptance-attempt-fixture")
            acceptance_attempt.mkdir(parents=True)
            acceptance_command, acceptance_cwd, acceptance_workspace = \
                run_experiments._stage(root, acceptance_job, acceptance_attempt)
            staged_candidate = Path(acceptance_command[3]).resolve()
            staged_sentinel = Path(acceptance_command[4]).resolve()
            sentinel_relative = Path(acceptance_command[5])
            self.assertEqual(staged_candidate,
                             acceptance_workspace / retained_candidate.relative_to(root))
            self.assertEqual(staged_sentinel,
                             acceptance_workspace /
                             (workspace / sentinel_relative).relative_to(root))
            derived_evidence_root = staged_sentinel
            for _ in sentinel_relative.parts:
                derived_evidence_root = derived_evidence_root.parent
            expected_evidence_root = acceptance_workspace / workspace.relative_to(root)
            self.assertEqual(derived_evidence_root, expected_evidence_root)
            self.assertTrue(staged_candidate.is_relative_to(derived_evidence_root))
            self.assertEqual(acceptance_cwd, acceptance_workspace / "actionmesh")
            self.assertFalse(any(
                Path(argument).is_absolute() and Path(argument).is_relative_to(root)
                and not Path(argument).is_relative_to(acceptance_workspace)
                for argument in acceptance_command[2:]))
            self.assertEqual(
                {acceptance_workspace / ref["path"]
                 for ref in acceptance_job["input_refs"] + acceptance_job["code_refs"]},
                {path for path in acceptance_workspace.rglob("*") if path.is_file()})
            acceptance_input_paths = {
                acceptance_workspace / ref["path"]
                for ref in acceptance_job["input_refs"]}
            producer_request = (
                staged_producer_root / "plans/mode-for-candidate/request.json")
            required_second_boundary = {
                producer_request,
                staged_mode_root / "result.json",
                staged_mode_root / "raw-manifest.json",
                staged_mode,
                staged_mode_root / "sequences/branch-0000.npz",
                staged_b0_sequence, staged_b0_report, staged_parity,
                staged_implementation, staged_math, staged_mode_implementation,
            }
            self.assertTrue({
                acceptance_workspace / path.relative_to(root)
                for path in required_second_boundary
            }.issubset(acceptance_input_paths))
            staged_candidate_bytes = staged_candidate.read_bytes()
            retained_candidate.write_bytes(b"controller candidate mutation\n")
            self.assertEqual(staged_candidate.read_bytes(), staged_candidate_bytes)

            (mode_root / "mode-bank.npz").write_bytes(b"controller mutation\n")
            (root / "actionmesh/research_math/c05_candidate_artifacts.py").write_bytes(
                b"controller source mutation\n")
            self.assertEqual(staged_bank.read_bytes(), bank_bytes)
            self.assertEqual(staged_code.read_bytes(), code_bytes)


if __name__ == "__main__":
    unittest.main()
