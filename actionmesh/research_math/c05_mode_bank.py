"""Receipt-bound C05 empirical mode-bank producer for ActionMesh.

This module freezes Stage 0 once, encodes the video context once, and runs K
independent Stage-I noise branches through the same Stage-II anchor/query.  Its
outputs are equal-mass empirical *model-output* modes.  They are deliberately
not described as posterior correspondences or calibrated probabilities.

The producer accepts an already constructed ActionMesh pipeline and input.  It
does not load benchmark labels, an evaluator, or any external coordinate bank.
Web-authored source and tests are ``generated_unexecuted`` until Local accepts
them on the pinned native environment.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Sequence

import numpy as np


class C05ModeBankError(RuntimeError):
    """A frozen producer invariant or retained-output check failed."""


PROVENANCE_KEYS = (
    "source_ref",
    "model_ref",
    "input_ref",
    "config_ref",
    "environment_ref",
)
SCOPE_FLAGS = (
    "native_scientific_qualification",
    "scientific_effect_qualification",
    "candidate_methods_tested",
    "dispatch_ready",
)
GENERATION_PARAMETER_KEYS = (
    "stage_0_steps",
    "stage_1_steps",
    "face_decimation",
    "floaters_threshold",
    "guidance_scales",
    "anchor_idx",
)
SEED_DOMAIN = b"ActionMesh/C05/same-anchor-stage-I/v1\0"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(_canonical_json(value))


def _array(value: Any, *, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    if array.dtype.hasobject or not np.issubdtype(array.dtype, np.number):
        raise C05ModeBankError(f"{name} must be a numeric array")
    if not np.isfinite(array).all():
        raise C05ModeBankError(f"{name} must contain only finite values")
    return array


def _array_sha256(value: Any, *, name: str) -> str:
    array = np.ascontiguousarray(_array(value, name=name))
    header = _canonical_json({"dtype": array.dtype.str, "shape": list(array.shape)})
    return _sha256_bytes(header + array.tobytes(order="C"))


def _safe_relative_path(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{name} must be a canonical relative POSIX path")
    path = PurePosixPath(value)
    if (path.is_absolute() or path.as_posix() != value
            or any(part in ("", ".", "..") for part in path.parts)):
        raise ValueError(f"{name} must be a canonical relative POSIX path")
    return value


def _sha256(value: str, *, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a canonical lowercase SHA-256")
    return value


def validate_provenance(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate immutable caller-supplied source/model/input/runtime receipts."""
    if not isinstance(value, Mapping):
        raise ValueError("provenance must be a mapping")
    expected = {"generation_uid", "gpu_uuid", *PROVENANCE_KEYS}
    if set(value) != expected:
        raise ValueError("provenance must contain only the frozen receipt fields")
    generation_uid = value["generation_uid"]
    gpu_uuid = value["gpu_uuid"]
    if not isinstance(generation_uid, str) or not generation_uid.strip():
        raise ValueError("nonempty generation_uid required")
    if not isinstance(gpu_uuid, str) or not gpu_uuid.strip():
        raise ValueError("nonempty gpu_uuid required")
    result: dict[str, Any] = {
        "generation_uid": generation_uid,
        "gpu_uuid": gpu_uuid,
    }
    for key in PROVENANCE_KEYS:
        ref = value[key]
        if not isinstance(ref, Mapping) or set(ref) != {"path", "sha256"}:
            raise ValueError(f"{key} must contain exactly path and sha256")
        result[key] = {
            "path": _safe_relative_path(ref["path"], name=f"{key}.path"),
            "sha256": _sha256(ref["sha256"], name=f"{key}.sha256"),
        }
    return result


def validate_ref_closure(value: Sequence[Mapping[str, Any]], *, name: str) -> list[dict[str, str]]:
    """Validate a bounded ordered content-reference closure without reading it."""
    if (not isinstance(value, (list, tuple)) or not value or len(value) > 128):
        raise ValueError(f"{name} must contain 1..128 content refs")
    result: list[dict[str, str]] = []
    paths: set[str] = set()
    for index, ref in enumerate(value):
        if not isinstance(ref, Mapping) or set(ref) != {"path", "sha256"}:
            raise ValueError(f"{name}[{index}] must contain exactly path and sha256")
        path = _safe_relative_path(ref["path"], name=f"{name}[{index}].path")
        if path in paths:
            raise ValueError(f"duplicate {name} path: {path}")
        paths.add(path)
        result.append({"path": path,
                       "sha256": _sha256(ref["sha256"], name=f"{name}[{index}].sha256")})
    return result


def validate_generation_parameters(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact B0-affecting parameters used by every branch."""
    if not isinstance(value, Mapping) or set(value) != set(GENERATION_PARAMETER_KEYS):
        raise ValueError("complete exact generation parameters required")
    integer_names = ("stage_0_steps", "stage_1_steps", "face_decimation", "anchor_idx")
    for name in integer_names:
        if type(value[name]) is not int:
            raise ValueError(f"{name} must be an integer")
    if any(value[name] <= 0 for name in integer_names[:-1]):
        raise ValueError("generation step/decimation counts must be positive")
    if value["anchor_idx"] != 0:
        raise ValueError("C05 freezes the ActionBench anchor at frame zero")
    threshold = value["floaters_threshold"]
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold)) or not 0.0 <= float(threshold) <= 1.0):
        raise ValueError("floaters_threshold must be finite and in [0,1]")
    guidance = value["guidance_scales"]
    if (not isinstance(guidance, (list, tuple)) or not guidance
            or any(isinstance(item, bool) or not isinstance(item, (int, float))
                   or not math.isfinite(float(item)) for item in guidance)):
        raise ValueError("nonempty finite guidance_scales required")
    return {
        "stage_0_steps": value["stage_0_steps"],
        "stage_1_steps": value["stage_1_steps"],
        "face_decimation": value["face_decimation"],
        "floaters_threshold": float(threshold),
        "guidance_scales": [float(item) for item in guidance],
        "anchor_idx": 0,
    }


def derive_inner_stage1_seeds(outer_seed: int, count: int) -> tuple[int, ...]:
    """Return a domain-separated deterministic schedule with exact B0 first.

    Branch zero intentionally reuses the outer ActionMesh seed: Stage 0 and the
    first Stage-I branch therefore reproduce the ordinary B0 random choices.
    Later seeds are derived, recorded internal nuisance seeds and never replace
    the outer G01 replicate identity.
    """
    if type(outer_seed) is not int or not 0 <= outer_seed < 2 ** 63:
        raise ValueError("outer_seed must be an integer in [0,2**63)")
    if type(count) is not int or count < 2:
        raise ValueError("at least two empirical branches required")
    seeds = [outer_seed]
    index = 1
    while len(seeds) < count:
        material = (SEED_DOMAIN + outer_seed.to_bytes(8, "big")
                    + index.to_bytes(8, "big"))
        candidate = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
        candidate &= (2 ** 63 - 1)
        if candidate not in seeds:
            seeds.append(candidate)
        index += 1
    return tuple(seeds)


def _copy_item(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "clone"):
        return value.detach().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if hasattr(value, "copy"):
        try:
            return value.copy()
        except TypeError:
            pass
    return copy.deepcopy(value)


def clone_pristine_bank(bank: Any) -> Any:
    """Clone one Stage-0 bank without sharing mutable tensors or meshes."""
    cloned = copy.copy(bank)
    cloned.timesteps = list(bank.timesteps)
    cloned.items = [_copy_item(item) for item in bank.items]
    if cloned.timesteps is bank.timesteps or cloned.items is bank.items:
        raise C05ModeBankError("bank clone retained mutable storage")
    return cloned


def _bank_sha256(bank: Any, *, name: str) -> str:
    if len(bank.items) != len(bank.timesteps):
        raise C05ModeBankError(f"{name} bank item/timestep count mismatch")
    payload = [{
        "timestep": float(timestep),
        "item_sha256": _mesh_sha256(item, name=f"{name}.item")
        if hasattr(item, "vertices") and hasattr(item, "faces")
        else _array_sha256(item, name=f"{name}.item"),
    } for timestep, item in zip(bank.timesteps, bank.items)]
    return _sha256_bytes(_canonical_json(payload))


def _mesh_sha256(mesh: Any, *, name: str) -> str:
    payload = {
        "vertices": _array_sha256(mesh.vertices, name=f"{name}.vertices"),
        "faces": _array_sha256(mesh.faces, name=f"{name}.faces"),
    }
    return _sha256_bytes(_canonical_json(payload))


def _apply_generation_parameters(pipeline: Any, parameters: Mapping[str, Any]) -> None:
    pipeline.cfg.model.image_to_3D_denoiser.num_inference_steps = parameters["stage_0_steps"]
    pipeline.scheduler.num_inference_steps = parameters["stage_1_steps"]
    pipeline.mesh_process.face_decimation = parameters["face_decimation"]
    pipeline.mesh_process.floaters_threshold = parameters["floaters_threshold"]
    pipeline.cf_guidance.guidance_scales = list(parameters["guidance_scales"])
    pipeline.cfg.anchor_idx = parameters["anchor_idx"]


def _model_scope(pipeline: Any):
    """Use inference/autocast when torch is present; keep fixtures dependency-light."""
    try:
        import torch
    except ImportError:  # pragma: no cover - native ActionMesh itself requires torch
        return nullcontext(), nullcontext()
    inference = torch.inference_mode()
    device = getattr(pipeline, "device", None)
    device_type = getattr(device, "type", str(device).split(":")[0])
    dtype = getattr(pipeline, "_dtype", None)
    autocast = (torch.autocast(device_type="cuda", dtype=dtype)
                if device_type == "cuda" and dtype is not None else nullcontext())
    return inference, autocast


def _ordered_meshes(bank: Any) -> tuple[list[Any], np.ndarray]:
    try:
        meshes, timesteps = bank.get_ordered(device="cpu")
    except TypeError:
        meshes, timesteps = bank.get_ordered()
    return list(meshes), _array(timesteps, name="mesh timesteps").astype(np.float32)


def _sequence_arrays(mesh_bank: Any, *, expected_frames: int) -> dict[str, np.ndarray]:
    meshes, timesteps = _ordered_meshes(mesh_bank)
    if len(meshes) != expected_frames or timesteps.shape != (expected_frames,):
        raise C05ModeBankError("every branch must export the complete frozen frame set")
    if not np.all(np.diff(timesteps.astype(np.float64)) > 0.0):
        raise C05ModeBankError("branch timesteps must be strictly increasing")
    vertices = []
    faces = None
    for frame, mesh in enumerate(meshes):
        current_vertices = _array(mesh.vertices, name=f"mesh[{frame}].vertices")
        current_faces = _array(mesh.faces, name=f"mesh[{frame}].faces")
        if current_vertices.ndim != 2 or current_vertices.shape[1] != 3:
            raise C05ModeBankError("mesh vertices must have shape (V,3)")
        if current_faces.ndim != 2 or current_faces.shape[1] != 3:
            raise C05ModeBankError("triangular mesh faces required")
        if faces is None:
            faces = current_faces.astype(np.int64, copy=True)
        elif not np.array_equal(faces, current_faces):
            raise C05ModeBankError("Stage-II sequence changed the frozen anchor topology")
        vertices.append(current_vertices.astype(np.float32, copy=False))
    stacked = np.stack(vertices, axis=0)
    assert faces is not None
    return {
        "vertices": stacked,
        "faces": faces,
        "frame_indices": np.arange(expected_frames, dtype=np.int64),
        "timesteps": timesteps,
        "query_vertex_ids": np.arange(stacked.shape[1], dtype=np.int64),
    }


def _save_sequence(path: Path, sequence: Mapping[str, np.ndarray]) -> None:
    np.savez_compressed(path, **sequence)


def _content_ref(root: Path, path: Path, *, content_sha256: str | None = None) -> dict[str, Any]:
    ref: dict[str, Any] = {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if content_sha256 is not None:
        ref["content_sha256"] = content_sha256
    return ref


def _fail_record(output: Path, *, error: Exception, outer_seed: int,
                 inner_seeds: Sequence[int]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    record = {
        "kind": "c05-same-anchor-stage1-mode-bank-result",
        "version": 1,
        "status": "failed",
        "error_type": type(error).__name__,
        "error": str(error),
        "outer_generation_seed": outer_seed,
        "inner_stage1_seeds": list(inner_seeds),
        **{name: False for name in SCOPE_FLAGS},
    }
    path = output / "result.json"
    if not path.exists():
        _write_json(path, record)


def produce_same_anchor_mode_bank(
    pipeline: Any,
    model_input: Any,
    output_dir: str | Path,
    *,
    outer_seed: int,
    branch_count: int,
    generation_parameters: Mapping[str, Any],
    provenance: Mapping[str, Any],
    producer_input_refs: Sequence[Mapping[str, Any]],
    producer_code_refs: Sequence[Mapping[str, Any]],
    expected_frames: int = 16,
) -> dict[str, Any]:
    """Generate and retain K same-anchor, different-Stage-I output sequences.

    Branch 0 follows the ordinary B0 algorithmic seed path: both Stage 0 and
    Stage I use ``outer_seed``.  Exact retained-array parity is established only
    by the separate receipt emitted by ``prepare_c05_mode_bank.py``.  Branches
    1..K-1 vary only the domain-separated Stage-I initialization seed.  Every
    branch starts from clones of the pristine Stage-0 latent and mesh banks and
    reuses the same encoded video context.
    """
    parameters = validate_generation_parameters(generation_parameters)
    frozen_provenance = validate_provenance(provenance)
    frozen_input_refs = validate_ref_closure(
        producer_input_refs, name="producer_input_refs")
    frozen_code_refs = validate_ref_closure(
        producer_code_refs, name="producer_code_refs")
    inner_seeds = derive_inner_stage1_seeds(outer_seed, branch_count)
    if type(expected_frames) is not int or expected_frames != 16:
        raise ValueError("C05 current-release production requires exactly 16 frames")
    if getattr(model_input, "n_frames", None) != expected_frames:
        raise ValueError("model input must be the exact complete 16-frame window")
    input_timesteps = _array(model_input.timesteps, name="input timesteps")
    if (input_timesteps.shape != (expected_frames,)
            or not np.all(np.diff(input_timesteps.astype(np.float64)) > 0.0)):
        raise ValueError("model input must have 16 strictly increasing timesteps")
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("C05 output directory must be new and append-only")
    output.mkdir(parents=True)
    (output / "sequences").mkdir()

    pristine_latent = pristine_mesh = None
    anchor_latent_sha256 = anchor_mesh_sha256 = context_sha256 = ""
    try:
        _apply_generation_parameters(pipeline, parameters)
        inference, _ = _model_scope(pipeline)
        with inference:
            pipeline._load_background_removal()
            model_input.frames = pipeline.background_removal.process_images(model_input.frames)
            pipeline._unload_model("background_removal")
            model_input.frames = pipeline.image_process.process_images(model_input.frames)

            pipeline._load_image_to_3d()
            pristine_latent, pristine_mesh = pipeline.init_banks_from_anchor(
                model_input, seed=outer_seed)
            pipeline._unload_model("image_to_3d_pipe")
            if len(pristine_latent.items) != 1 or len(pristine_mesh.items) != 1:
                raise C05ModeBankError("Stage 0 must yield exactly one anchor in each bank")
            anchor_latent_sha256 = _bank_sha256(pristine_latent, name="stage0 latent")
            anchor_mesh_sha256 = _bank_sha256(pristine_mesh, name="stage0 mesh")

            pipeline._load_image_encoder()
            context = pipeline.encode_all_frames(model_input)
            pipeline._unload_model("image_encoder")
            context_sha256 = _array_sha256(context, name="encoded video context")

            branch_latents = []
            pipeline._load_temporal_denoiser()
            _, autocast = _model_scope(pipeline)
            with autocast:
                for seed in inner_seeds:
                    latent_bank = clone_pristine_bank(pristine_latent)
                    branch_latents.append(pipeline.generate_3d_latents(
                        model_input, context=context, latent_bank=latent_bank, seed=seed))
            pipeline._unload_model("temporal_3D_denoiser")

            sequences: list[dict[str, np.ndarray]] = []
            branch_refs = []
            pipeline._load_temporal_vae()
            _, autocast = _model_scope(pipeline)
            with autocast:
                for branch_index, (seed, latent_bank) in enumerate(
                        zip(inner_seeds, branch_latents)):
                    mesh_bank = clone_pristine_bank(pristine_mesh)
                    completed = pipeline.generate_mesh_animation(
                        latent_bank=latent_bank, mesh_bank=mesh_bank)
                    sequence = _sequence_arrays(completed, expected_frames=expected_frames)
                    sequences.append(sequence)
                    branch_path = output / "sequences" / f"branch-{branch_index:04d}.npz"
                    _save_sequence(branch_path, sequence)
                    sequence_content = _sha256_bytes(_canonical_json({
                        key: _array_sha256(value, name=f"branch {branch_index} {key}")
                        for key, value in sequence.items()
                    }))
                    branch_refs.append({
                        "branch_index": branch_index,
                        "role": ("b0_algorithmic_path_unverified"
                                 if branch_index == 0 else "empirical_mode"),
                        "stage1_seed": seed,
                        "sequence_ref": _content_ref(
                            output, branch_path, content_sha256=sequence_content),
                    })
            pipeline._unload_model("temporal_3D_vae")

        if (_bank_sha256(pristine_latent, name="stage0 latent") != anchor_latent_sha256
                or _bank_sha256(pristine_mesh, name="stage0 mesh") != anchor_mesh_sha256):
            raise C05ModeBankError("pristine Stage-0 banks were mutated by a branch")

        common = sequences[0]
        anchor_vertices = _array(pristine_mesh.items[0].vertices,
                                 name="pristine anchor vertices").astype(np.float32)
        for branch_index, sequence in enumerate(sequences):
            if (not np.array_equal(sequence["faces"], common["faces"])
                    or not np.array_equal(sequence["timesteps"], common["timesteps"])
                    or not np.array_equal(sequence["frame_indices"], common["frame_indices"])
                    or not np.array_equal(sequence["query_vertex_ids"],
                                          common["query_vertex_ids"])):
                raise C05ModeBankError(
                    f"branch {branch_index} changed Stage-II topology/query/time identity")
            if not np.array_equal(sequence["vertices"][0], anchor_vertices):
                raise C05ModeBankError(
                    f"branch {branch_index} changed the frozen Stage-0 anchor coordinates")

        vertices = np.stack([sequence["vertices"] for sequence in sequences], axis=0)
        scores = np.full(branch_count, 1.0 / branch_count, dtype=np.float64)
        mode_bank_path = output / "mode-bank.npz"
        np.savez_compressed(
            mode_bank_path,
            vertices=vertices,
            faces=common["faces"],
            frame_indices=common["frame_indices"],
            timesteps=common["timesteps"],
            query_vertex_ids=common["query_vertex_ids"],
            equal_mass_scores=scores,
            inner_stage1_seeds=np.asarray(inner_seeds, dtype=np.int64),
            outer_generation_seed=np.asarray(outer_seed, dtype=np.int64),
        )
        bank_content_sha256 = _sha256_bytes(_canonical_json({
            "vertices": _array_sha256(vertices, name="mode vertices"),
            "faces": _array_sha256(common["faces"], name="mode faces"),
            "frame_indices": _array_sha256(common["frame_indices"], name="frame indices"),
            "timesteps": _array_sha256(common["timesteps"], name="mode timesteps"),
            "query_vertex_ids": _array_sha256(
                common["query_vertex_ids"], name="query vertex ids"),
            "equal_mass_scores": _array_sha256(scores, name="equal mass scores"),
            "inner_stage1_seeds": _array_sha256(
                np.asarray(inner_seeds, dtype=np.int64), name="inner seeds"),
        }))
        source_path = Path(__file__)
        manifest = {
            "kind": "c05-same-anchor-stage1-mode-bank-manifest",
            "version": 1,
            "status": "completed_unqualified",
            "candidate_id": "C05",
            "outer_generation_seed": outer_seed,
            "inner_stage1_seeds": list(inner_seeds),
            "seed_schedule": "sha256-domain-separated-v1; branch0=outer-seed",
            "branch_count": branch_count,
            "frames": expected_frames,
            "vertices_per_frame": int(vertices.shape[2]),
            "generation_parameters": parameters,
            "provenance": frozen_provenance,
            "producer_input_refs": frozen_input_refs,
            "producer_code_refs": frozen_code_refs,
            "producer_code_sha256": _sha256_file(source_path),
            "stage0_anchor_latent_sha256": anchor_latent_sha256,
            "stage0_anchor_mesh_sha256": anchor_mesh_sha256,
            "encoded_video_context_sha256": context_sha256,
            "stage2_query_source_sha256": anchor_mesh_sha256,
            "stage2_query_contract": "get_mesh_features(anchor_mesh,with_normals=True)",
            "mode_semantics": "equal-mass empirical model-output modes",
            "posterior_correspondence_claim": False,
            "calibrated_probability_claim": False,
            "uniform_score_implications": {
                "independent_top1_tie_break": "branch-0000",
                "temperature_reweighting_changes_weights": False,
            },
            "mode_bank_ref": _content_ref(
                output, mode_bank_path, content_sha256=bank_content_sha256),
            "branches": branch_refs,
            **{name: False for name in SCOPE_FLAGS},
        }
        manifest_path = output / "raw-manifest.json"
        _write_json(manifest_path, manifest)
        result = {
            "kind": "c05-same-anchor-stage1-mode-bank-result",
            "version": 1,
            "status": "completed_unqualified",
            "candidate_id": "C05",
            "manifest_ref": _content_ref(output, manifest_path),
            "mode_bank_ref": manifest["mode_bank_ref"],
            "producer_input_refs": frozen_input_refs,
            "producer_code_refs": frozen_code_refs,
            "outer_generation_seed": outer_seed,
            "inner_stage1_seeds": list(inner_seeds),
            "branch0_exact_b0": False,
            "branch0_parity_requirement": (
                "A separate receipt must compare every branch-0000 native array "
                "against the retained B0 sequence before candidate materialization."),
            "stage0_runs": 1,
            "context_encodes": 1,
            "full_sequences_retained": branch_count,
            "mode_semantics": manifest["mode_semantics"],
            "posterior_correspondence_claim": False,
            **{name: False for name in SCOPE_FLAGS},
        }
        _write_json(output / "result.json", result)
        return result
    except Exception as error:
        for attribute in ("background_removal", "image_to_3d_pipe", "image_encoder",
                          "temporal_3D_denoiser", "temporal_3D_vae"):
            try:
                pipeline._unload_model(attribute)
            except Exception:
                pass
        _fail_record(output, error=error, outer_seed=outer_seed, inner_seeds=inner_seeds)
        raise


def validate_retained_mode_bank(output_dir: str | Path,
                                project_root: str | Path | None = None) -> dict[str, Any]:
    """Rehash a completed producer result and all retained K sequences."""
    root = Path(output_dir)
    result_path = root / "result.json"
    manifest_path = root / "raw-manifest.json"
    result = json.loads(result_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if (result.get("kind") != "c05-same-anchor-stage1-mode-bank-result"
            or result.get("version") != 1
            or result.get("status") != "completed_unqualified"
            or manifest.get("kind") != "c05-same-anchor-stage1-mode-bank-manifest"
            or manifest.get("version") != 1
            or manifest.get("status") != "completed_unqualified"):
        raise C05ModeBankError("completed unqualified C05 producer records required")
    if any(result.get(name) is not False or manifest.get(name) is not False
           for name in SCOPE_FLAGS):
        raise C05ModeBankError("producer receipt overclaims scientific scope")
    if result.get("manifest_ref", {}).get("sha256") != _sha256_file(manifest_path):
        raise C05ModeBankError("C05 manifest hash mismatch")
    refs = [manifest["mode_bank_ref"]] + [row["sequence_ref"]
                                                for row in manifest["branches"]]
    expected_paths = {"mode-bank.npz", "raw-manifest.json", "result.json"}
    expected_paths.update(ref["path"] for ref in refs)
    actual_paths = {path.relative_to(root).as_posix() for path in root.rglob("*")
                    if path.is_file()}
    if actual_paths != expected_paths:
        raise C05ModeBankError("missing or unlisted C05 retained output")
    for ref in refs:
        relative = _safe_relative_path(ref["path"], name="retained ref path")
        path = root / relative
        if (not path.is_file() or path.is_symlink()
                or ref.get("bytes") != path.stat().st_size
                or ref.get("sha256") != _sha256_file(path)):
            raise C05ModeBankError("retained C05 output hash/size mismatch: " + relative)
    if (manifest.get("branch_count") != len(manifest.get("branches", []))
            or manifest.get("inner_stage1_seeds") != result.get("inner_stage1_seeds")
            or manifest.get("outer_generation_seed") != result.get("outer_generation_seed")):
        raise C05ModeBankError("C05 seed/count receipt mismatch")
    for field in ("producer_input_refs", "producer_code_refs"):
        closure = validate_ref_closure(manifest.get(field), name=field)
        if result.get(field) != closure:
            raise C05ModeBankError("C05 result/manifest source closure mismatch: " + field)
        if project_root is not None:
            project = Path(project_root).resolve()
            for ref in closure:
                path = project / ref["path"]
                if (not path.is_file() or path.is_symlink()
                        or path.resolve().relative_to(project) != Path(ref["path"])
                        or _sha256_file(path) != ref["sha256"]):
                    raise C05ModeBankError(
                        "C05 producer closure hash mismatch: " + ref["path"])
    with np.load(root / manifest["mode_bank_ref"]["path"], allow_pickle=False) as archive:
        required = {"vertices", "faces", "frame_indices", "timesteps",
                    "query_vertex_ids", "equal_mass_scores", "inner_stage1_seeds",
                    "outer_generation_seed"}
        if set(archive.files) != required:
            raise C05ModeBankError("complete C05 mode-bank fields required")
        if archive["vertices"].shape[0] != manifest["branch_count"]:
            raise C05ModeBankError("mode-bank branch axis differs from receipt")
        expected_mass = np.full(manifest["branch_count"],
                                1.0 / manifest["branch_count"], dtype=np.float64)
        if not np.array_equal(archive["equal_mass_scores"], expected_mass):
            raise C05ModeBankError("C05 branches must retain exact equal masses")
        if not np.array_equal(archive["inner_stage1_seeds"],
                              np.asarray(manifest["inner_stage1_seeds"], dtype=np.int64)):
            raise C05ModeBankError("C05 stored seed schedule differs from receipt")
        bank = {name: archive[name].copy() for name in required}
    bank_content = _sha256_bytes(_canonical_json({
        "vertices": _array_sha256(bank["vertices"], name="mode vertices"),
        "faces": _array_sha256(bank["faces"], name="mode faces"),
        "frame_indices": _array_sha256(bank["frame_indices"], name="frame indices"),
        "timesteps": _array_sha256(bank["timesteps"], name="mode timesteps"),
        "query_vertex_ids": _array_sha256(bank["query_vertex_ids"], name="query vertex ids"),
        "equal_mass_scores": _array_sha256(bank["equal_mass_scores"], name="equal mass scores"),
        "inner_stage1_seeds": _array_sha256(bank["inner_stage1_seeds"], name="inner seeds"),
    }))
    if manifest["mode_bank_ref"].get("content_sha256") != bank_content:
        raise C05ModeBankError("C05 aggregate content digest mismatch")
    sequence_fields = ("vertices", "faces", "frame_indices", "timesteps",
                       "query_vertex_ids")
    for index, row in enumerate(manifest["branches"]):
        if (row.get("branch_index") != index
                or row.get("stage1_seed") != manifest["inner_stage1_seeds"][index]
                or row.get("role") != ("b0_algorithmic_path_unverified"
                                        if index == 0 else "empirical_mode")):
            raise C05ModeBankError("C05 branch identity/seed/role mismatch")
        path = root / row["sequence_ref"]["path"]
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(sequence_fields):
                raise C05ModeBankError("complete native branch sequence required")
            sequence = {name: archive[name].copy() for name in sequence_fields}
        expected = {"vertices": bank["vertices"][index],
                    **{name: bank[name] for name in sequence_fields[1:]}}
        if any(not np.array_equal(sequence[name], expected[name])
               for name in sequence_fields):
            raise C05ModeBankError("C05 aggregate/branch array mismatch")
        content = _sha256_bytes(_canonical_json({
            name: _array_sha256(sequence[name], name=f"branch {index} {name}")
            for name in sequence_fields
        }))
        if row["sequence_ref"].get("content_sha256") != content:
            raise C05ModeBankError("C05 branch content digest mismatch")
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a retained C05 same-anchor Stage-I mode bank")
    parser.add_argument("operation", choices=("validate",))
    parser.add_argument("output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Validation-only CLI; native production remains harness/plan owned."""
    args = parse_args(argv)
    manifest = validate_retained_mode_bank(args.output)
    print(json.dumps({
        "status": "validated_unqualified",
        "branch_count": manifest["branch_count"],
        "outer_generation_seed": manifest["outer_generation_seed"],
        **{name: False for name in SCOPE_FLAGS},
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
