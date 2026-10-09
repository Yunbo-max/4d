"""Receipt-bound C20 decoder-consensus target producer.

This GPU inner task replays one exact retained ActionMesh Stage-II context at
three source times (0, 8 and 15) while keeping the Stage-I latent bank, decoder,
target clock and mesh identity fixed.  The equal-weight cross-source coordinate
mean defines a falsifiable model self-consistency correction target; it is not
ground truth, uncertainty calibration, camera evidence or a native metric.

The task is source-authored only.  It may run only through the emitted harness
plan after an exact, expiring GPU-resume authorization lifts the current STOP.
"""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


CANDIDATE_ID = "4d-math-20261006-c20"
FREEZE_KIND = "c20-decoder-consensus-input-freeze"
TARGET_KIND = "c20-decoder-consensus-target"
SOURCE_INDICES = (0, 8, 15)
ACTIONMESH_REVISION = "fb69228ba8a4df684907b5d259cff3c22fb722f1"
ACTIONBENCH_REVISION = "2796071cbe6248422fcbeab3101fa9f9886cb7b9"
CONTROLLER_ENV_PREFIX = "C20_TARGET_CONTROLLER"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def canonical_digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object: " + str(path))
    return value


def file_ref(root: Path, path: Path) -> dict:
    root, path = Path(root).resolve(), Path(path).resolve()
    return {"path": path.relative_to(root).as_posix(), "sha256": digest(path)}


def resolve_ref(root: Path, ref: dict) -> Path:
    if (not isinstance(ref, dict) or set(ref) != {"path", "sha256"}
            or not isinstance(ref.get("path"), str)
            or not isinstance(ref.get("sha256"), str)
            or len(ref["sha256"]) != 64):
        raise ValueError("Exact path/sha256 reference required")
    relative = Path(ref["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Project-relative nonescaping reference required")
    root = Path(root).resolve(); unresolved = root / relative
    if unresolved.is_symlink():
        raise ValueError("Symlinked evidence is not accepted")
    path = unresolved.resolve(); path.relative_to(root)
    if not path.is_file() or digest(path) != ref["sha256"]:
        raise ValueError("Missing or stale C20 evidence: " + ref["path"])
    return path


def _argument_path_for_ref(root: Path, value: str, ref: dict) -> Path:
    """Map a plan's original absolute argv path into the current staged root."""
    relative = Path(ref["path"])
    argument = Path(value)
    if argument.is_absolute():
        parts = argument.parts
        suffix = relative.parts
        if len(parts) < len(suffix) or parts[-len(suffix):] != suffix:
            raise ValueError("C20 plan argument differs from pinned ref path")
    elif argument != relative:
        raise ValueError("C20 plan relative argument differs from pinned ref path")
    return resolve_ref(root, ref)


def validate_target_plan(root: Path, plan: dict, authorization: dict,
                         authorization_path: Path) -> dict:
    """Fix the complete GPU task semantics; a generic valid plan is insufficient."""
    root = Path(root).resolve()
    auth_ref = file_ref(root, authorization_path)
    freeze_path = resolve_ref(root, authorization["input_freeze_ref"])
    environment_path = resolve_ref(root, authorization["environment_ref"])
    freeze = read_json(freeze_path)
    _, freeze_refs = validate_freeze(root, freeze)
    environment = read_json(environment_path)
    dependency_refs = environment.get("dependency_lock_refs")
    if not isinstance(dependency_refs, list) or len(dependency_refs) != 1:
        raise ValueError("Exact C20 dependency ref required")
    dependency_path = resolve_ref(root, dependency_refs[0])
    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 1:
        raise ValueError("Exact one-job C20 target plan required")
    job = jobs[0]
    command = job.get("command")
    if not isinstance(command, list) or len(command) != 19:
        raise ValueError("Exact C20 target command required")
    expected_flags = ["-m", "research_math.c20_consensus_target", "--root", "..",
        "--freeze", command[6], "--source-root", command[8],
        "--weights-root", command[10], "--environment", command[12],
        "--authorization", command[14], "--gpu-uuid", authorization["gpu_uuid"], "--output",
        "c20-consensus-target-output"]
    if command[1:] != expected_flags or command[0] != environment["python_executable"]:
        raise ValueError("C20 target command/module differs from authorization")
    _argument_path_for_ref(root, command[6], authorization["input_freeze_ref"])
    _argument_path_for_ref(root, command[12], authorization["environment_ref"])
    _argument_path_for_ref(root, command[14], auth_ref)
    cache = (Path("inputs/c20-target/staged") /
             (freeze["uid"] + "-" + str(freeze["generation_seed"])) /
             ACTIONMESH_REVISION)
    source_relative, weights_relative = cache / "source", cache / "weights"
    source_arg, weights_arg = Path(command[8]), Path(command[10])
    for argument, relative in ((source_arg, source_relative),
                               (weights_arg, weights_relative)):
        expected_relative = Path("..") / relative
        if argument != expected_relative:
            raise ValueError("Canonical immutable C20 runtime cache required")
    source_root, weights_root = root / source_relative, root / weights_relative
    runtime_paths = sorted((path for parent in (source_root, weights_root)
                            for path in parent.rglob("*") if path.is_file()),
                           key=lambda path: path.as_posix())
    if not runtime_paths or any(path.is_symlink() for path in runtime_paths):
        raise ValueError("Complete physical C20 runtime cache required")
    expected_inputs = [file_ref(root, path) for path in
                       (environment_path, dependency_path, freeze_path)]
    expected_inputs += [*freeze_refs, auth_ref]
    expected_inputs += [file_ref(root, path) for path in runtime_paths]
    by_path = {}
    for ref in expected_inputs:
        if ref["path"] in by_path and by_path[ref["path"]] != ref:
            raise ValueError("Conflicting C20 target input identity")
        by_path[ref["path"]] = ref
    actual_inputs = {ref.get("path"): ref for ref in job.get("input_refs", [])
                     if isinstance(ref, dict)}
    code_paths = [root / "actionmesh/research_math" / name for name in (
        "__init__.py", "c20_consensus_target.py", "actionbench_parity.py",
        "control_scoring.py", "decoder_observer.py",
        "pipeline_decoder_observer.py", "native_context_runner.py")]
    code_paths += [root / "actionmesh" / name for name in (
        "official_actionbench_adapter.py", "research_census_eval.py",
        "prepare_c20_consensus_target.py")]
    expected_code = [file_ref(root, path) for path in code_paths]
    output = "actionmesh/c20-consensus-target-output/"
    expected_outputs = [output + name for name in (
        "producer-report.json", "producer-manifest.json", "consensus-target.npz")]
    actual_input_list = job.get("input_refs", [])
    if (len(actual_input_list) != len(actual_inputs)
            or len(expected_inputs) != len(by_path)
            or actual_inputs != by_path or job.get("code_refs") != expected_code
            or job.get("cwd") != "actionmesh"
            or job.get("output_paths") != expected_outputs
            or job.get("trial_id") != "c20-decoder-consensus-target"
            or job.get("group") != "c20-target-producer"
            or job.get("arm_role") != "decoder-consensus-no-camera-no-gt"
            or job.get("seed") != authorization["generation_seed"]
            or plan.get("purpose") != "engineering"
            or plan.get("evidence_mode") != "developmental"
            or plan.get("protocol_ref") is not None
            or plan.get("protocol_digest") is not None
            or plan.get("run_id") != authorization["run_id"]
            or plan.get("limits", {}).get("max_attempts") != 1
            or plan.get("limits", {}).get("max_retries_per_trial") != 0
            or plan.get("limits", {}).get("wall_time_seconds") !=
               authorization["wall_seconds"]):
        raise ValueError("C20 target plan is not the exact authorized method task")
    return job


def validate_execution_authorization(root: Path, authorization_path: Path,
                                     gpu_uuid: str) -> dict:
    from datetime import datetime
    authorization_path = Path(authorization_path).resolve()
    authorization_path.relative_to(Path(root).resolve())
    authorization = read_json(authorization_path)
    controller_root_value = os.environ.get(CONTROLLER_ENV_PREFIX + "_ROOT")
    consumption_value = os.environ.get(CONTROLLER_ENV_PREFIX + "_CONSUMPTION_PATH")
    consumption_sha = os.environ.get(CONTROLLER_ENV_PREFIX + "_CONSUMPTION_SHA256")
    if not all((controller_root_value, consumption_value, consumption_sha)):
        raise ValueError("Finalized C20 target controller claim injection required")
    controller_root = Path(controller_root_value).resolve()
    consumption_path = Path(consumption_value).resolve()
    consumption_path.relative_to(controller_root)
    if (consumption_path.is_symlink() or not consumption_path.is_file()
            or digest(consumption_path) != consumption_sha):
        raise ValueError("Physical finalized C20 target consumption required")
    consumed = read_json(consumption_path)
    controller_auth = resolve_ref(controller_root, consumed["authorization_ref"])
    auth_required = {
        "kind", "version", "candidate_id", "scope", "run_id", "uid",
        "generation_seed", "gpu_uuid", "input_freeze_ref", "environment_ref",
        "issued_at", "expires_at", "wall_seconds",
        "explicit_resume_for_exact_attempt", "no_scientific_retry",
        "authorization_digest",
    }
    consumption_required = {
        "kind", "version", "state", "authorization_ref", "run_id",
        "gpu_uuid", "native_plan_ref", "native_plan_digest",
        "harness_plan_ref", "harness_plan_digest", "no_scientific_retry",
        "consumed_at", "consumption_digest",
    }
    expected_consumption_path = (controller_root /
        "inputs/c20-target/authorization-consumption" /
        (digest(controller_auth) + ".json")).resolve()
    if digest(controller_auth) != digest(authorization_path):
        raise ValueError("Staged/controller C20 authorization identity differs")
    native_plan_path = resolve_ref(controller_root, consumed["native_plan_ref"])
    harness_plan_path = resolve_ref(controller_root, consumed["harness_plan_ref"])
    native_plan = read_json(native_plan_path)
    harness_plan = read_json(harness_plan_path)
    validate_target_plan(controller_root, native_plan, authorization, controller_auth)
    try:
        issued = datetime.fromisoformat(authorization["issued_at"].replace("Z", "+00:00"))
        expires = datetime.fromisoformat(authorization["expires_at"].replace("Z", "+00:00"))
        consumed_at = datetime.fromisoformat(consumed["consumed_at"].replace("Z", "+00:00"))
    except (KeyError, AttributeError, ValueError) as error:
        raise ValueError("C20 target controller times are invalid") from error
    if (set(authorization) != auth_required
            or set(consumed) != consumption_required
            or consumption_path != expected_consumption_path
            or authorization.get("kind") != "c20-target-gpu-resume-authorization"
            or authorization.get("version") != 1
            or authorization.get("candidate_id") != CANDIDATE_ID
            or authorization.get("scope") != "single_c20_target_attempt"
            or authorization.get("explicit_resume_for_exact_attempt") is not True
            or authorization.get("no_scientific_retry") is not True
            or type(authorization.get("wall_seconds")) is not int
            or not 60 <= authorization["wall_seconds"] <= 26940
            or authorization.get("authorization_digest") != canonical_digest({
            key: value for key, value in authorization.items()
            if key != "authorization_digest"})
            or authorization.get("gpu_uuid") != gpu_uuid
            or consumed.get("kind") != "c20-target-authorization-consumption"
            or consumed.get("version") != 1
            or consumed.get("state") != "consumed_for_exact_plan"
            or consumed.get("gpu_uuid") != gpu_uuid
            or consumed.get("run_id") != authorization.get("run_id")
            or consumed.get("native_plan_digest") != native_plan.get("plan_digest")
            or consumed.get("harness_plan_digest") != harness_plan.get("plan_digest")
            or harness_plan.get("gpus", {}).get("uuids") != [gpu_uuid]
            or consumed.get("no_scientific_retry") is not True
            or consumed.get("consumption_digest") != canonical_digest({
                key: value for key, value in consumed.items()
                if key != "consumption_digest"})
            or issued.tzinfo is None or expires.tzinfo is None
            or consumed_at.tzinfo is None or not issued <= consumed_at < expires):
        raise ValueError("Exact in-window C20 target controller consumption required")
    cache_prefix = "inputs/c20-target/staged/"
    runtime_refs = [ref for ref in native_plan["jobs"][0]["input_refs"]
                    if ref["path"].startswith(cache_prefix)]
    if not runtime_refs:
        raise ValueError("C20 target plan lacks immutable runtime snapshot")
    return {
        "consumption": consumed,
        "runtime_refs": runtime_refs,
        "freeze_ref": authorization["input_freeze_ref"],
        "environment_ref": authorization["environment_ref"],
        "authorization_ref": consumed["authorization_ref"],
        "source_root": (Path("inputs/c20-target/staged") /
                        (authorization["uid"] + "-" +
                         str(authorization["generation_seed"])) /
                        ACTIONMESH_REVISION / "source"),
        "weights_root": (Path("inputs/c20-target/staged") /
                         (authorization["uid"] + "-" +
                          str(authorization["generation_seed"])) /
                         ACTIONMESH_REVISION / "weights"),
    }


def validate_freeze(root: Path, freeze: dict) -> tuple[dict, list[dict]]:
    core = {key: value for key, value in freeze.items() if key != "freeze_digest"}
    required = {
        "kind", "version", "candidate_id", "uid", "generation_seed",
        "benchmark_revision", "source_indices", "target_indices",
        "source_sequence_ref", "source_report_ref", "generation_identity_ref",
        "capture_identity_ref", "window_record_ref", "decoder_record_ref",
        "decoder_inputs_ref", "decoder_tensors_ref", "weights_manifest_ref",
        "source_review_ref", "frozen_without_candidate_or_confirmation_outcomes",
        "max_source_self_map_rms_over_diagonal",
        "max_consensus_target_rms_over_diagonal",
        "min_action_seed_rms_over_diagonal",
        "freeze_digest",
    }
    if (set(freeze) != required
            or freeze.get("freeze_digest") != canonical_digest(core)
            or freeze.get("kind") != FREEZE_KIND or freeze.get("version") != 1
            or freeze.get("candidate_id") != CANDIDATE_ID
            or freeze.get("benchmark_revision") != ACTIONBENCH_REVISION
            or freeze.get("generation_seed") not in (42, 314, 2718)
            or freeze.get("source_indices") != list(SOURCE_INDICES)
            or freeze.get("target_indices") != list(range(16))
            or freeze.get("frozen_without_candidate_or_confirmation_outcomes") is not True):
        raise ValueError("Exact outcome-blind C20 decoder-consensus freeze required")
    for name in ("max_source_self_map_rms_over_diagonal",
                 "max_consensus_target_rms_over_diagonal",
                 "min_action_seed_rms_over_diagonal"):
        value = freeze.get(name)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not np.isfinite(value) or value <= 0):
            raise ValueError("Positive prospective C20 bound required: " + name)
    if (not isinstance(freeze.get("uid"), str) or not freeze["uid"]
            or Path(freeze["uid"]).name != freeze["uid"]):
        raise ValueError("Canonical C20 UID required")
    names = [name for name in required if name.endswith("_ref")]
    paths = {name: resolve_ref(root, freeze[name]) for name in names}
    sequence, report = paths["source_sequence_ref"], read_json(paths["source_report_ref"])
    if (sequence.name != "sequence.npz"
            or paths["source_report_ref"] != sequence.with_name("report.json")
            or report.get("status") != "completed"
            or report.get("uid") != freeze["uid"]
            or report.get("seed") != freeze["generation_seed"]
            or report.get("sha256", {}).get("sequence.npz") != digest(sequence)):
        raise ValueError("Exact completed native B0 source pair required")
    identity = read_json(paths["generation_identity_ref"])
    capture_identity = read_json(paths["capture_identity_ref"])
    if (identity != capture_identity
            or identity.get("kind") != "native-context-generation-identity"
            or identity.get("uid") != freeze["uid"]
            or identity.get("generation", {}).get("seed") != freeze["generation_seed"]
            or not isinstance(identity.get("upstream_source_sha256"), dict)
            or not identity["upstream_source_sha256"]
            or identity.get("native_context_qualified") is not False
            or identity.get("scientific_effect_qualification") is not False):
        raise ValueError("Exact unqualified retained native context identity required")
    manifest = identity.get("verified_unit_manifest")
    if (not isinstance(manifest, dict)
            or not isinstance(manifest.get("population"), dict)
            or manifest["population"].get("revision") != ACTIONBENCH_REVISION):
        raise ValueError("Retained native context has a different ActionBench revision")
    call = paths["decoder_record_ref"].parent
    window = paths["window_record_ref"].parent
    capture = paths["capture_identity_ref"].parent
    if (call != window / "decoder/call-0000" or window.parent != capture
            or paths["decoder_inputs_ref"] != call / "inputs.safetensors"
            or paths["decoder_tensors_ref"] != call / "tensors.safetensors"):
        raise ValueError("Canonical single-window C20 capture layout required")
    review = read_json(paths["source_review_ref"])
    if (review.get("kind") != "c20-decoder-consensus-source-review"
            or review.get("candidate_id") != CANDIDATE_ID
            or review.get("outcome") != "verified"
            or review.get("author") == review.get("reviewer")
            or review.get("generated_unexecuted") is not True):
        raise ValueError("Independent C20 decoder-consensus source review required")
    refs = [freeze[name] for name in names]
    return identity, refs


def _verify_weights(weights_root: Path, manifest_path: Path) -> dict:
    rows = json.loads(manifest_path.read_text())
    selected = [row for row in rows if isinstance(row, dict)
                and row.get("repo") == "facebook/ActionMesh"
                and row.get("revision") == ACTIONMESH_REVISION
                and isinstance(row.get("path"), str)
                and row["path"].startswith("weights/ActionMesh/autoencoder/")]
    if {row.get("file") for row in selected} != {
            "autoencoder/config.json", "autoencoder/model.safetensors"}:
        raise ValueError("Exact pinned ActionMesh autoencoder weight closure required")
    verified = {}
    weights_root = Path(weights_root).resolve()
    for row in selected:
        path = (weights_root / Path(row["path"]).relative_to("weights")).resolve()
        path.relative_to(weights_root)
        if (path.is_symlink() or not path.is_file()
                or path.stat().st_size != row.get("size")
                or digest(path) != row.get("sha256")):
            raise ValueError("Missing/stale ActionMesh weight: " + row["path"])
        verified[row["path"]] = row["sha256"]
    return verified


def _autocast_stack(stack: ExitStack, torch, record: dict) -> None:
    stack.enter_context(torch.inference_mode())
    stack.enter_context(torch.set_grad_enabled(False))
    for device in ("cpu", "cuda"):
        if record.get(device + "_autocast_enabled"):
            name = record[device + "_autocast_dtype"].removeprefix("torch.")
            if name not in ("float16", "bfloat16"):
                raise ValueError("Unsupported retained autocast dtype")
            stack.enter_context(torch.autocast(device_type=device,
                                               dtype=getattr(torch, name)))
        else:
            stack.enter_context(torch.autocast(device_type=device, enabled=False))


def _runtime_identity(root: Path, environment_path: Path,
                      environment: dict, gpu_uuid: str) -> dict:
    from research_math.actionbench_parity import validate_runtime_identity

    required_environment = {
        "execution_mode", "python_executable", "python_version",
        "python_prefix", "conda_prefix", "gpu_uuid", "packages",
        "dependency_lock_refs", "captured_at", "gpu_identity_source",
        "gpu_identity_verified", "native_contract_qualified", "scope",
    }
    if (set(environment) != required_environment
            or not isinstance(gpu_uuid, str) or not gpu_uuid.startswith("GPU-")):
        raise ValueError("Exact native environment schema/GPU identity required")
    required = ("numpy", "torch", "trimesh", "scipy", "pytorch3d")
    packages = {name: metadata.version(name) for name in required}
    validate_runtime_identity(
        environment, gpu_uuid=gpu_uuid,
        visible_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
        python_executable=sys.executable, package_versions=packages)
    environment_path = Path(environment_path).resolve()
    if environment_path.parts[-3:] != ("inputs", "native-runtime", "environment.json"):
        raise ValueError("Canonical staged native environment required")
    refs = environment.get("dependency_lock_refs")
    if (not isinstance(refs, list) or len(refs) != 1
            or refs[0].get("path") != "inputs/native-runtime/dependencies.json"):
        raise ValueError("Canonical pinned dependency inventory required")
    dependency_path = resolve_ref(root, refs[0])
    dependency = read_json(dependency_path)
    dependency_keys = {
        "conda_packages", "conda_prefix", "kind", "packages",
        "python_executable", "python_prefix", "python_version", "scope", "version",
    }
    if (set(dependency) != dependency_keys
            or dependency.get("kind") != "installed-native-dependency-inventory"
            or dependency.get("version") != "1.0.0"
            or dependency.get("python_executable") != environment["python_executable"]
            or dependency.get("python_prefix") != environment["python_prefix"]
            or dependency.get("python_version") != environment["python_version"]
            or dependency.get("conda_prefix") != environment["conda_prefix"]
            or any(dependency.get("packages", {}).get(name) != value
                   for name, value in environment["packages"].items())
            or not isinstance(dependency.get("conda_packages"), list)
            or not dependency["conda_packages"]):
        raise ValueError("Native dependency inventory differs from environment")
    command = ["nvidia-smi", "--id=" + gpu_uuid,
               "--query-gpu=uuid,name,memory.total", "--format=csv,noheader,nounits"]
    observed = subprocess.run(command, check=True, capture_output=True, text=True)
    values = [value.strip() for value in observed.stdout.strip().split(",")]
    if len(values) != 3 or values[0] != gpu_uuid:
        raise ValueError("Physical C20 GPU identity differs from authorization")
    return {"gpu_uuid": values[0], "name": values[1],
            "memory_total_mib": float(values[2]), "command": command,
            "stdout_sha256": hashlib.sha256(observed.stdout.encode()).hexdigest(),
            "stderr_sha256": hashlib.sha256(observed.stderr.encode()).hexdigest(),
            "packages": packages, "python_executable": sys.executable,
            "dependency_lock_ref": file_ref(root, dependency_path)}


def build_target(root: Path, freeze_path: Path, source_root: Path,
                 weights_root: Path, environment_path: Path,
                 authorization_path: Path, gpu_uuid: str, output: Path) -> dict:
    """Run one frozen three-source decoder-consensus producer."""
    if any(os.environ.get(name) != "1" for name in
           ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")):
        raise ValueError("Offline native environment required")
    # Resolve every path before importing upstream ActionMesh.  Its audited
    # loader temporarily changes cwd to the upstream checkout, so retaining a
    # caller-relative output here would redirect terminal evidence.
    root = Path(root).resolve()
    freeze_path = Path(freeze_path).resolve()
    source_root = Path(source_root).resolve()
    weights_root = Path(weights_root).resolve()
    output = Path(output).resolve()
    environment_path = Path(environment_path).resolve()
    authorization_path = Path(authorization_path).resolve()
    for path in (freeze_path, environment_path, authorization_path, output.parent):
        path.relative_to(root)
    if output != root / "actionmesh/c20-consensus-target-output":
        raise ValueError("Canonical C20 target output path required")
    execution = validate_execution_authorization(root, authorization_path, gpu_uuid)
    expected_freeze = resolve_ref(root, execution["freeze_ref"])
    expected_environment = resolve_ref(root, execution["environment_ref"])
    expected_authorization = resolve_ref(root, execution["authorization_ref"])
    expected_source = (root / execution["source_root"]).resolve()
    expected_weights = (root / execution["weights_root"]).resolve()
    if (freeze_path != expected_freeze
            or environment_path != expected_environment
            or authorization_path != expected_authorization
            or source_root != expected_source
            or weights_root != expected_weights):
        raise ValueError("Current C20 invocation differs from exact controller-authorized plan")
    runtime_snapshot = {ref["path"]: digest(resolve_ref(root, ref))
                        for ref in execution["runtime_refs"]}
    freeze = read_json(freeze_path)
    identity, refs = validate_freeze(root, freeze)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "kind": TARGET_KIND, "version": 1, "candidate_id": CANDIDATE_ID,
        "status": "error", "uid": freeze["uid"],
        "seed": freeze["generation_seed"], "source_indices": list(SOURCE_INDICES),
        "input_freeze_ref": file_ref(root, freeze_path), "input_refs": refs,
        "environment_ref": file_ref(root, environment_path),
        "target_semantics": "equal-source decoder coordinate self-consistency surrogate; not truth",
        "camera_or_gt_used": False, "candidate_or_confirmation_outcomes_used": False,
        "native_qualified": False, "scientific_admission": False,
        "generated_unexecuted_at_authoring": True,
    }
    started = time.monotonic()
    try:
        import torch
        import trimesh
        from research_math.decoder_observer import load_capture
        from research_math.pipeline_decoder_observer import load_window
        from research_math.native_context_runner import _source_modules
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("Exactly one parent-harness allocated CUDA device required")
        runtime_identity = _runtime_identity(
            root, environment_path, read_json(environment_path), gpu_uuid)
        previous_cwd = Path.cwd()
        try:
            _, Autoencoder, _ = _source_modules(
                source_root, identity["upstream_source_sha256"])
        finally:
            os.chdir(previous_cwd)
        from actionmesh.preprocessing.mesh_processor import get_mesh_features
        weight_hashes = _verify_weights(
            weights_root, resolve_ref(root, freeze["weights_manifest_ref"]))
        model = Autoencoder.from_pretrained(
            str(Path(weights_root) / "ActionMesh/autoencoder"),
            local_files_only=True).eval().to("cuda")
        if getattr(model, "prediction_mode", None) != "direct":
            raise ValueError("C20 consensus is frozen to native direct-coordinate mode")
        call = resolve_ref(root, freeze["decoder_record_ref"]).parent
        window = resolve_ref(root, freeze["window_record_ref"]).parent
        tensors, record = load_capture(call)
        window_tensors, window_record = load_window(window)
        if (record.get("training") or record.get("step_callback_present")
                or window_record.get("mapping") !=
                "source-first native window IDs; exact normalized-alpha check; no subsampling"):
            raise ValueError("Exact eval-mode source-first decoder capture required")
        with np.load(resolve_ref(root, freeze["source_sequence_ref"]),
                     allow_pickle=False) as saved:
            arrays = {name: saved[name].copy() for name in saved.files}
        vertices, faces = arrays["vertices"], arrays["faces"]
        if (vertices.ndim != 3 or vertices.shape[0] != 16
                or vertices.shape[2] != 3 or not np.isfinite(vertices).all()
                or faces.ndim != 2 or faces.shape[1] != 3
                or not np.array_equal(arrays["frame_indices"], np.arange(16))):
            raise ValueError("Complete finite 16-frame native source required")
        if (not np.array_equal(vertices[0], window_tensors["anchor_vertices"].numpy())
                or not np.array_equal(vertices[1:], window_tensors["vertices"].numpy())
                or not np.array_equal(faces, window_tensors["faces"].numpy())
                or not np.array_equal(
                    arrays["timesteps"][:1],
                    window_tensors["source_timesteps"].numpy().reshape(-1))
                or not np.array_equal(
                    arrays["timesteps"][1:],
                    window_tensors["target_timesteps"].numpy().reshape(-1))
                or ("query_vertex_ids" in arrays and not np.array_equal(
                    arrays["query_vertex_ids"], np.arange(vertices.shape[1])))):
            raise ValueError(
                "B0 source geometry/time identity differs from retained decoder window")
        values = {name: tensors[name].to(device=record["input_devices"][name])
                  for name in ("latent", "framestep", "source_alpha", "target_alphas", "query")}
        with ExitStack() as stack:
            _autocast_stack(stack, torch, record)
            replay = model(**values)
        if not torch.equal(replay.detach().cpu(), tensors["output"]):
            raise ValueError("Pinned model no longer exactly replays source-zero capture")
        alphas = torch.cat((values["source_alpha"][:, None],
                            values["target_alphas"]), dim=1)
        if (alphas.shape != (1, 16)
                or not torch.all(alphas[:, 1:] > alphas[:, :-1]).item()):
            raise ValueError("Exact increasing 16-frame normalized clock required")
        diagonal = float(np.linalg.norm(np.ptp(vertices[0].astype(np.float64), axis=0)))
        if not np.isfinite(diagonal) or diagonal <= 0:
            raise ValueError("Positive native anchor diagonal required")
        trajectories, self_map = [], []
        for source_index in SOURCE_INDICES:
            if source_index == 0:
                query = values["query"]
            else:
                mesh = trimesh.Trimesh(vertices[source_index], faces, process=False)
                query_np = get_mesh_features(mesh, with_normals=True).numpy()
                query = torch.from_numpy(query_np)[None].to(
                    device=record["input_devices"]["query"])
            source_alpha = alphas[:, source_index]
            call_values = {name: values[name] for name in ("latent", "framestep")}
            call_values.update(source_alpha=source_alpha,
                               target_alphas=alphas, query=query)
            with ExitStack() as stack:
                _autocast_stack(stack, torch, record)
                raw = model(**call_values)
                decoded = model.apply_displacement(query[..., :3], raw)
            current = decoded[0].detach().cpu().to(torch.float64).numpy()
            if current.shape != vertices.shape or not np.isfinite(current).all():
                raise ValueError("Incomplete C20 source-conditioned trajectory")
            self_rms = float(np.sqrt(np.mean(
                (current[source_index] - vertices[source_index].astype(np.float64)) ** 2))
                / diagonal)
            if self_rms > freeze["max_source_self_map_rms_over_diagonal"]:
                raise ValueError("Source-conditioned decoder self-map exceeds frozen bound")
            self_map.append(self_rms)
            trajectories.append(current)
        bank = np.stack(trajectories)
        raw_consensus = np.mean(bank, axis=0)
        consensus = raw_consensus.copy()
        consensus[0] = vertices[0].astype(np.float64)
        target = consensus - vertices.astype(np.float64)
        action_seed = vertices.astype(np.float64) - vertices[0:1].astype(np.float64)
        action_seed[0] = 0.0
        target_rms = float(np.sqrt(np.mean(target ** 2)) / diagonal)
        action_rms = float(np.sqrt(np.mean(action_seed ** 2)) / diagonal)
        if target_rms > freeze["max_consensus_target_rms_over_diagonal"]:
            raise ValueError("Consensus correction exceeds frozen small-target bound")
        if action_rms < freeze["min_action_seed_rms_over_diagonal"]:
            raise ValueError("Native action seed is too small for amplitude construction")
        weights = np.ones_like(target, dtype=np.float64)
        target_path = output / "consensus-target.npz"
        np.savez_compressed(
            target_path, vertices=vertices, faces=faces,
            frame_indices=arrays["frame_indices"], timesteps=arrays["timesteps"],
            source_indices=np.asarray(SOURCE_INDICES, dtype=np.int64),
            source_trajectories=bank, raw_consensus_vertices=raw_consensus,
            consensus_vertices=consensus,
            target=target, action_amplitude_seed=action_seed, weights=weights)
        post_weights = _verify_weights(
            weights_root, resolve_ref(root, freeze["weights_manifest_ref"]))
        post_sources = {relative: digest(source_root / relative)
                        for relative in identity["upstream_source_sha256"]}
        post_runtime = {ref["path"]: digest(resolve_ref(root, ref))
                        for ref in execution["runtime_refs"]}
        if (post_weights != weight_hashes or post_runtime != runtime_snapshot
                or post_sources != identity["upstream_source_sha256"]):
            raise ValueError("C20 source/weight closure changed during producer execution")
        report.update(
            status="completed", target_sha256=digest(target_path),
            source_sequence_sha256=freeze["source_sequence_ref"]["sha256"],
            source_report_sha256=freeze["source_report_ref"]["sha256"],
            freeze_sha256=digest(freeze_path), verified_weight_sha256=weight_hashes,
            runtime_identity=runtime_identity,
            frame_count=16, vertex_count=int(vertices.shape[1]),
            source_replay_count=len(SOURCE_INDICES), target_anchor_exact=True,
            action_seed_anchor_exact=True,
            anchor_diagonal=diagonal,
            source_self_map_rms_over_diagonal=self_map,
            cross_source_rms=float(np.sqrt(np.mean((bank - consensus[None]) ** 2))),
            consensus_target_rms=float(np.sqrt(np.mean(target ** 2))),
            consensus_target_rms_over_diagonal=target_rms,
            action_seed_rms_over_diagonal=action_rms,
            input_clock="retained native normalized decoder clock",
            output_clock="original source sequence timesteps; no evaluator retiming")
    except Exception as error:
        report.update(exception_type=type(error).__name__[:256],
                      error=(str(error) or "<empty>")[:4096])
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["implementation_sha256"] = digest(Path(__file__))
        report["report_digest"] = canonical_digest(report)
        (output / "producer-report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n")
        manifest = {
            "kind": "c20-decoder-consensus-target-manifest", "version": 1,
            "candidate_id": CANDIDATE_ID, "status": report["status"],
            "uid": freeze["uid"], "seed": freeze["generation_seed"],
            "report": {"path": "producer-report.json",
                       "sha256": digest(output / "producer-report.json")},
            "conditional_outputs": ([{"path": "consensus-target.npz",
                                      "sha256": digest(output / "consensus-target.npz")}] if
                                    (output / "consensus-target.npz").is_file() else []),
        }
        manifest["manifest_digest"] = canonical_digest(manifest)
        (output / "producer-manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return report


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "freeze", "source-root", "weights-root",
                 "environment", "authorization", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    args = parser.parse_args(argv)
    result = build_target(args.root, args.freeze, args.source_root,
                          args.weights_root, args.environment,
                          args.authorization, args.gpu_uuid, args.output)
    print(json.dumps({"status": result["status"], "native_qualified": False,
                      "scientific_admission": False}))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
