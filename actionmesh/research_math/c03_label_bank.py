"""Build C03 development labels from official tracked ActionBench queries.

This is a development-only GPU inner task.  It reuses retained Stage-II latent
contexts, replaces only the decoder query with the released frame-zero tracked
points/normals, and compares the decoded coordinates with the same tracked
indices at frames 1..15.  It never maps a tracked GT point to a generated mesh
vertex.  The fitted C03 operator is global; transfer to generated-query vertices
is an explicit covariate-transfer assumption checked later from residual support.
"""
from __future__ import annotations

from contextlib import ExitStack
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


CANDIDATE_ID = "4d-math-20261006-c03"
BANK_KIND = "c03-tracked-gt-development-label-bank"
INVENTORY_KIND = "c03-development-label-inventory"
BENCHMARK_REVISION = "2796071cbe6248422fcbeab3101fa9f9886cb7b9"
ACTIONMESH_REVISION = "fb69228ba8a4df684907b5d259cff3c22fb722f1"
SOURCE_REVIEW_CHECKS = {
    "tracked_point_index_is_stable_across_all_16_released_frames",
    "surface_columns_are_xyz_then_normal_xyz",
    "decoder_query_contract_is_xyz_plus_l2_normalized_normal",
    "retained_window_is_source_first_with_15_target_times",
    "development_and_confirmation_families_are_disjoint",
}
SOURCE_REVIEW_REFS = {
    "actionmesh/repo/actionbench/README.md":
        "0dcfafddb40fcfb120dcc94d7801128fe152e451dd1470d5510badd43b9c3f0a",
    "actionmesh/repo/actionmesh/pipeline.py":
        "0934b6801676a9e6f188e5b07450109e0c3359c10fa1230a9b9c69f056fd50f2",
    "actionmesh/repo/actionmesh/model/temporal_autoencoder.py":
        "69c18c2fe73de8ecd95f062b4faafac4337589c53bac18284fccaa36d4aa37f2",
    "actionmesh/repo/actionmesh/preprocessing/mesh_processor.py":
        "0ce2c86f28cc84eb23beda5471c55b45f29420abb9c1d39d8b3aa69b81c6da65",
}
CANONICAL_INPUT_REFS = {
    "dataset_snapshot_ref": {
        "path": "docs/research-math-20261006/longgoal-20261007/admission-evidence-r6/inputs/actionbench-full128-snapshots/admission.json",
        "sha256": "be4d61684f3f1f3a6a049415fc69f0342de8c9897a093b7e06cfeacb2a0efc04"},
    "dataset_semantics_ref": {
        "path": "docs/research-math-20261006/longgoal-20261007/admission-evidence-r6/inputs/actionbench-full128-snapshots/dataset-semantics.json",
        "sha256": "5557c7496ebe98c30b1e3e618fbae1e33512f4248141046f12094b7c9b56499b"},
    "source_review_ref": {
        "path": "docs/research-math-20261006/longgoal-20261007/c03-tracked-query-source-review.json",
        "sha256": "420c8c5ffdf38573a92086e95f25636b8acc1896bf72aa5e7cd581128f14cefe"},
    "weights_manifest_ref": {
        "path": "actionmesh/weights-manifest.json",
        "sha256": "55bdd725afe926efa8cafdcd67df1be89d31de06eaaeba9651cb53ce0299f24e"},
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object: " + str(path))
    return value


def read_json_value(path: Path):
    return json.loads(Path(path).read_text())


def canonical_digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


def resolve_ref(root: Path, ref: dict) -> Path:
    if (not isinstance(ref, dict) or set(ref) != {"path", "sha256"}
            or not isinstance(ref.get("path"), str)
            or not isinstance(ref.get("sha256"), str)
            or len(ref["sha256"]) != 64):
        raise ValueError("Exact path/sha256 reference required")
    relative = Path(ref["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Project-relative nonescaping reference required")
    root = Path(root).resolve(); path = (root / relative).resolve()
    path.relative_to(root)
    if (path.is_symlink() or not path.is_file()
            or any(parent.is_symlink() for parent in path.parents
                   if parent != root.parent)
            or digest(path) != ref["sha256"]):
        raise ValueError("Missing, symlinked or stale evidence: " + ref["path"])
    return path


def collect_refs(value) -> list[dict]:
    refs = []
    if isinstance(value, dict):
        if set(value) == {"path", "sha256"}:
            refs.append(value)
        else:
            for item in value.values():
                refs.extend(collect_refs(item))
    elif isinstance(value, list):
        for item in value:
            refs.extend(collect_refs(item))
    return refs


def validate_inventory(root: Path, inventory: dict) -> tuple[dict, list[dict]]:
    core = {key: value for key, value in inventory.items()
            if key != "inventory_digest"}
    if (inventory.get("inventory_digest") != canonical_digest(core)
            or inventory.get("kind") != INVENTORY_KIND
            or inventory.get("version") != 1
            or inventory.get("candidate_id") != CANDIDATE_ID
            or inventory.get("generation_seed") != 42
            or inventory.get("benchmark_revision") != BENCHMARK_REVISION
            or inventory.get("frame_indices") != list(range(16))
            or inventory.get("query_policy") !=
            "official_tracked_gt_positions_and_normals_same_index_no_generated_vertex_mapping"):
        raise ValueError("Current digest-bound C03 development inventory required")
    required_refs = ("dataset_snapshot_ref", "dataset_semantics_ref",
                     "g01_design_ref", "family_evidence_ref", "family_split_ref",
                     "source_review_ref", "weights_manifest_ref")
    if any(name not in inventory for name in required_refs):
        raise ValueError("Complete C03 semantic/source/weight references required")
    ref_paths = {name: resolve_ref(root, inventory[name]) for name in required_refs}
    if any(inventory[name] != expected
           for name, expected in CANONICAL_INPUT_REFS.items()):
        raise ValueError("C03 canonical admission/review/weight reference differs")
    snapshot = read_json(ref_paths["dataset_snapshot_ref"])
    semantics = read_json(ref_paths["dataset_semantics_ref"])
    split = read_json(ref_paths["family_split_ref"])
    review = read_json(ref_paths["source_review_ref"])
    from research_math import g01_design as g01
    design = g01.read_json(ref_paths["g01_design_ref"])
    if (inventory["g01_design_ref"].get("path") !=
            "docs/research-math-20261006/longgoal-20261007/G01_DESIGN.json"
            or design.get("design_digest") != g01.EXPECTED_DESIGN_DIGEST
            or design.get("design_digest") != g01.canonical_record_digest(
                design, "design_digest")):
        raise ValueError("Exact canonical G01 design required")
    evidence = g01.read_json(ref_paths["family_evidence_ref"])
    derived_split = g01.derive_family_split(root, design, evidence)
    dataset = snapshot.get("snapshots", {}).get("dataset", {})
    if (snapshot.get("kind") != "actionbench-full128-snapshot-admission"
            or snapshot.get("status") != "admitted_engineering_snapshot"
            or dataset.get("revision") != BENCHMARK_REVISION
            or not isinstance(dataset.get("files"), list)):
        raise ValueError("Pinned ActionBench engineering snapshot admission required")
    if (semantics.get("kind") != "actionbench-full128-dataset-semantics-admission"
            or semantics.get("status") != "admitted_engineering_dataset_semantics"
            or semantics.get("dataset") != "facebook/actionbench"
            or semantics.get("revision") != BENCHMARK_REVISION
            or semantics.get("frames_per_sample") != 16
            or not isinstance(semantics.get("samples"), list)):
        raise ValueError("Pinned ActionBench 16-frame semantics admission required")
    if (split.get("kind") != "g01-family-split"
            or split.get("version") != "1.0.0"
            or split.get("benchmark_revision") != BENCHMARK_REVISION
            or split.get("design_digest") != design["design_digest"]
            or split.get("split_digest") != g01.canonical_record_digest(
                split, "split_digest") or split != derived_split):
        raise ValueError("Pinned G01 family split required")
    review_refs = review.get("source_refs")
    review_core = {key: value for key, value in review.items()
                   if key != "review_digest"}
    if (review.get("kind") != "c03-tracked-query-source-review"
            or review.get("version") != 1 or review.get("outcome") != "verified"
            or review.get("candidate_id") != CANDIDATE_ID
            or review.get("author") == review.get("reviewer")
            or not all(isinstance(review.get(name), str) and review[name]
                       for name in ("author", "reviewer"))
            or set(review.get("checks", [])) != SOURCE_REVIEW_CHECKS
            or review.get("benchmark_revision") != BENCHMARK_REVISION
            or review.get("generated_unexecuted") is not True
            or review.get("review_digest") != canonical_digest(review_core)
            or not isinstance(review_refs, list)
            or {item.get("path"): item.get("sha256") for item in review_refs
                if isinstance(item, dict)} != SOURCE_REVIEW_REFS):
        raise ValueError("Complete C03 tracked-query source review required")
    for ref in review_refs:
        refs_path = resolve_ref(root, ref)
        if refs_path != (Path(root).resolve() / ref["path"]).resolve():
            raise ValueError("C03 source review reference escaped canonical path")
    transitive_refs = [design["benchmark"]["population_ref"],
                       design["cohort"]["exposure_ref"],
                       *collect_refs(evidence), *review_refs]
    for ref in transitive_refs:
        resolve_ref(root, ref)
    development = inventory.get("development_uids")
    d2 = inventory.get("d2_uids")
    confirmation = inventory.get("confirmation_uids")
    families = inventory.get("uid_to_family")
    if (not isinstance(development, list) or not development
            or development != sorted(set(development))
            or not isinstance(confirmation, list)
            or confirmation != sorted(set(confirmation))
            or not isinstance(d2, list) or d2 != sorted(set(d2))
            or set(development) & set(confirmation) or set(development) & set(d2)
            or set(d2) & set(confirmation)
            or not isinstance(families, dict)
            or set(families) != set(development) | set(d2) | set(confirmation)
            or {families[x] for x in development} & {families[x] for x in d2}
            or {families[x] for x in development} & {families[x] for x in confirmation}
            or {families[x] for x in d2} & {families[x] for x in confirmation}
            or development != split.get("d1_ids")
            or d2 != split.get("d2_ids")
            or confirmation != split.get("confirmation_ids")
            or any(split.get("family_by_uid", {}).get(uid) != family
                   for uid, family in families.items())):
        raise ValueError("Disjoint complete UID/family split required")
    file_rows = {row.get("path"): row for row in dataset["files"]
                 if isinstance(row, dict) and isinstance(row.get("path"), str)}
    semantic_rows = {row.get("uid"): row for row in semantics["samples"]
                     if isinstance(row, dict) and isinstance(row.get("uid"), str)}
    rows = inventory.get("development_rows")
    if (not isinstance(rows, list)
            or [row.get("uid") if isinstance(row, dict) else None for row in rows]
               != development):
        raise ValueError("Exactly one ordered retained context per development UID required")
    refs = []
    expected = {"uid", "family", "capture_identity_ref", "window_record_ref",
                "decoder_record_ref", "decoder_inputs_ref", "decoder_tensors_ref",
                "surfaces_ref"}
    for row in rows:
        if (set(row) != expected or row["family"] != families[row["uid"]]):
            raise ValueError("Exact C03 development row schema/family required")
        paths = {name: resolve_ref(root, row[name]) for name in expected if name.endswith("_ref")}
        identity = read_json(paths["capture_identity_ref"])
        if (identity.get("kind") != "native-context-generation-identity"
                or identity.get("uid") != row["uid"]
                or identity.get("generation", {}).get("seed") != 42):
            raise ValueError("Retained seed42 native context identity required")
        call = paths["decoder_record_ref"].parent
        window = paths["window_record_ref"].parent
        capture = paths["capture_identity_ref"].parent
        if (call != window / "decoder/call-0000"
                or window.parent != capture
                or paths["decoder_inputs_ref"] != call / "inputs.safetensors"
                or paths["decoder_tensors_ref"] != call / "tensors.safetensors"):
            raise ValueError("Canonical single-window decoder capture layout required")
        surface_key = f"data/{row['uid']}/surfaces.npy"
        surface_admission = file_rows.get(surface_key)
        surface_semantics = semantic_rows.get(row["uid"], {}).get("surfaces", {})
        if (not row["surfaces_ref"].get("path", "").endswith(surface_key)
                or not isinstance(surface_admission, dict)
                or surface_admission.get("sha256") != row["surfaces_ref"]["sha256"]
                or surface_semantics.get("shape") != [16, 100000, 6]
                or surface_semantics.get("dtype") != "<f4"
                or surface_semantics.get("zero_normal_count") != 0):
            raise ValueError("UID-bound admitted ActionBench surface tensor required")
        refs.extend(row[name] for name in expected if name.endswith("_ref"))
    refs.extend(inventory[name] for name in required_refs)
    refs.extend(transitive_refs)
    unique = {}
    for ref in refs:
        if ref["path"] in unique and unique[ref["path"]] != ref:
            raise ValueError("Conflicting C03 transitive input reference")
        unique[ref["path"]] = ref
    return inventory, list(unique.values())


def _verify_weights(weights_root: Path, manifest_path: Path) -> dict:
    rows = read_json_value(manifest_path)
    if not isinstance(rows, list):
        raise ValueError("Pinned weights manifest must be a JSON list")
    selected = [row for row in rows if isinstance(row, dict)
                and row.get("repo") == "facebook/ActionMesh"
                and row.get("revision") == ACTIONMESH_REVISION
                and isinstance(row.get("path"), str)
                and row["path"].startswith("weights/ActionMesh/autoencoder/")]
    if not selected or {row.get("file") for row in selected} != {
            "autoencoder/config.json", "autoencoder/model.safetensors"}:
        raise ValueError("Exact pinned ActionMesh autoencoder manifest closure required")
    verified = {}
    weights_root = Path(weights_root).resolve()
    for row in selected:
        relative = Path(row["path"]).relative_to("weights")
        path = (weights_root / relative).resolve(); path.relative_to(weights_root)
        if (path.is_symlink() or not path.is_file()
                or path.stat().st_size != row.get("size")
                or digest(path) != row.get("sha256")):
            raise ValueError("Missing or stale pinned ActionMesh weight: " + row["path"])
        verified[row["path"]] = row["sha256"]
    return verified


def _selection(uid: str, count: int, total: int, seed: int) -> np.ndarray:
    if type(count) is not int or not 1 <= count <= total:
        raise ValueError("points_per_uid must be within released tracked population")
    material = hashlib.sha256(f"{seed}:{uid}".encode()).digest()[:8]
    rng = np.random.default_rng(int.from_bytes(material, "little"))
    return np.sort(rng.choice(total, size=count, replace=False).astype(np.int64))


def _autocast_stack(stack, torch, record):
    stack.enter_context(torch.inference_mode())
    stack.enter_context(torch.set_grad_enabled(False))
    for device in ("cpu", "cuda"):
        if record.get(device + "_autocast_enabled"):
            name = record[device + "_autocast_dtype"].removeprefix("torch.")
            if name not in ("float16", "bfloat16"):
                raise ValueError("Unsupported captured autocast dtype")
            stack.enter_context(torch.autocast(device_type=device,
                                               dtype=getattr(torch, name)))
        else:
            stack.enter_context(torch.autocast(device_type=device, enabled=False))


def build_bank(root: Path, inventory_path: Path, source_root: Path,
               weights_root: Path, output: Path, *, points_per_uid: int,
               selection_seed: int) -> dict:
    """Execute the exact retained-context GT-query producer once."""
    if any(os.environ.get(key) != "1" for key in
           ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")):
        raise ValueError("Offline native environment required")
    root = Path(root).resolve(); output = Path(output)
    inventory = read_json(inventory_path)
    _, inventory_refs = validate_inventory(root, inventory)
    output.mkdir(parents=True, exist_ok=False)
    report = {"kind": BANK_KIND, "version": 1, "candidate_id": CANDIDATE_ID,
              "status": "error", "inventory_sha256": digest(inventory_path),
              "inventory_ref": {"path": inventory_path.resolve().relative_to(root).as_posix(),
                                "sha256": digest(inventory_path)},
              "input_refs": inventory_refs,
              "generated_unexecuted_at_authoring": True,
              "native_qualified": False, "scientific_admission": False,
              "transfer_claim": "global residual-response calibration only; no pointwise GT-to-generated mapping"}
    started = time.monotonic()
    try:
        import torch
        from research_math.decoder_observer import load_capture
        from research_math.pipeline_decoder_observer import load_window
        from research_math.native_context_runner import _source_modules
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("Exactly one parent-harness allocated CUDA device required")
        identities = [read_json(resolve_ref(root, row["capture_identity_ref"]))
                      for row in inventory["development_rows"]]
        source_hashes = identities[0]["upstream_source_sha256"]
        if any(identity.get("upstream_source_sha256") != source_hashes
               for identity in identities):
            raise ValueError("All development contexts require one exact ActionMesh source")
        _, Autoencoder, _ = _source_modules(source_root, source_hashes)
        weight_hashes = _verify_weights(
            weights_root, resolve_ref(root, inventory["weights_manifest_ref"]))
        model = Autoencoder.from_pretrained(
            str(Path(weights_root) / "ActionMesh/autoencoder"),
            local_files_only=True).eval().to("cuda")
        residual_rows, error_rows, uid_rows, chosen_rows = [], [], [], []
        per_uid = []
        for row, identity in zip(inventory["development_rows"], identities):
            call = resolve_ref(root, row["decoder_record_ref"]).parent
            window = resolve_ref(root, row["window_record_ref"]).parent
            window_tensors, window_record = load_window(window)
            tensors, record = load_capture(call)
            if (record.get("training") or record.get("step_callback_present")
                    or getattr(model, "prediction_mode", None) != record.get("prediction_mode")
                    or window_record.get("mapping") !=
                       "source-first native window IDs; exact normalized-alpha check; no subsampling"
                    or not torch.equal(window_tensors["anchor_vertices"].to(tensors["query"].dtype),
                                       tensors["query"][0, :, :3])):
                raise ValueError("Exact eval decoder mode without callback required")
            if (window_tensors["source_timesteps"].shape != (1,)
                    or not torch.equal(window_tensors["source_timesteps"],
                                       torch.zeros_like(window_tensors["source_timesteps"]))
                    or window_tensors["target_timesteps"].shape != (1, 15)
                    or not torch.equal(window_tensors["target_timesteps"],
                        torch.arange(1, 16,
                            dtype=window_tensors["target_timesteps"].dtype,
                            device=window_tensors["target_timesteps"].device)[None])):
                raise ValueError("C03 labels require exact absolute native times 0 -> 1..15")
            captured_values = {name: tensors[name].to(device=record["input_devices"][name])
                               for name in ("query", "latent", "framestep",
                                            "source_alpha", "target_alphas")}
            with ExitStack() as stack:
                _autocast_stack(stack, torch, record)
                replayed = model(**captured_values)
            if not torch.equal(replayed.detach().cpu(), tensors["output"]):
                raise ValueError("Pinned model does not exactly replay retained native decoder call")
            surface_path = resolve_ref(root, row["surfaces_ref"])
            surfaces = np.load(surface_path, allow_pickle=False)
            if (surfaces.shape != (16, 100000, 6)
                    or not np.issubdtype(surfaces.dtype, np.floating)
                    or not np.isfinite(surfaces).all()):
                raise ValueError("Released finite ActionBench [16,100000,6] surface required")
            indices = _selection(row["uid"], points_per_uid, 100000, selection_seed)
            query_np = np.asarray(surfaces[0, indices], dtype=np.float32)
            normal_norm = np.linalg.norm(query_np[:, 3:6], axis=1, keepdims=True)
            if np.any(~np.isfinite(normal_norm)) or np.any(normal_norm <= 0):
                raise ValueError("Tracked GT probe contains a zero/nonfinite normal")
            # Match get_mesh_features(..., with_normals=True): float32 XYZ and
            # L2-normalized float32 normals.  Released normals are not assumed
            # unit length merely because the dataset card names them normals.
            query_np[:, 3:6] /= normal_norm
            values = {name: tensors[name].to(device=record["input_devices"][name])
                      for name in ("latent", "framestep", "source_alpha", "target_alphas")}
            query = torch.from_numpy(query_np)[None].to(device=record["input_devices"]["query"])
            with ExitStack() as stack:
                _autocast_stack(stack, torch, record)
                raw_target = model(query=query, **values)
                predicted = model.apply_displacement(query[..., :3], raw_target)
                raw_source = model(query=query, **{**values,
                    "target_alphas": values["source_alpha"][:, None]})
                predicted_source = model.apply_displacement(query[..., :3], raw_source)
            target = predicted[0].detach().cpu().to(torch.float64).numpy()
            source = predicted_source[0, 0].detach().cpu().to(torch.float64).numpy()
            if target.shape != (15, points_per_uid, 3) or source.shape != (points_per_uid, 3):
                raise ValueError("C03 decoder query did not return complete 15-frame coordinates")
            residual_rows.append(source - surfaces[0, indices, :3].astype(np.float64))
            error_rows.append(target - surfaces[1:, indices, :3].astype(np.float64))
            uid_rows.extend([row["uid"]] * points_per_uid)
            chosen_rows.append(indices)
            per_uid.append({"uid": row["uid"], "family": row["family"],
                            "points": points_per_uid,
                            "surfaces_sha256": row["surfaces_ref"]["sha256"],
                            "capture_identity_sha256": row["capture_identity_ref"]["sha256"],
                            "selected_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest()})
        residual = np.concatenate(residual_rows).astype(np.float64)
        errors = np.concatenate(error_rows, axis=1).astype(np.float64)
        uids = np.asarray(uid_rows, dtype="U")
        family_counts = {}
        for row in inventory["development_rows"]:
            family_counts[row["family"]] = family_counts.get(row["family"], 0) + 1
        weights = np.asarray([1.0 / (len(family_counts) * family_counts[inventory["uid_to_family"][uid]]
                                     * points_per_uid) for uid in uids], dtype=np.float64)
        bank = output / "development-bank.npz"
        np.savez_compressed(bank, reference_residual=residual,
                            labeled_error=errors, weights=weights, uids=uids)
        report.update(status="completed", data_sha256=digest(bank), rows=len(uids),
                      points_per_uid=points_per_uid, selection_seed=selection_seed,
                      development_uids=inventory["development_uids"], per_uid=per_uid,
                      d2_uids=inventory["d2_uids"],
                      confirmation_uids=inventory["confirmation_uids"],
                      verified_weight_sha256=weight_hashes,
                      weight_sum=float(weights.sum()),
                      coordinate_policy="normalized_actionbench_tracked_gt_query_global_affine_transfer",
                      tracked_index_policy="same released point index across all 16 frames",
                      label_definition="decoded tracked query coordinate minus released tracked coordinate",
                      reference_definition="decoded source-time tracked query coordinate minus frame-zero tracked coordinate")
    except Exception as error:
        report.update(exception_type=type(error).__name__, error=str(error)[:4096])
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["report_digest"] = canonical_digest(report)
        (output / "producer-report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "inventory", "source-root", "weights-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--points-per-uid", type=int, required=True)
    parser.add_argument("--selection-seed", type=int, required=True)
    args = parser.parse_args(argv)
    result = build_bank(args.root, args.inventory, args.source_root,
                        args.weights_root, args.output,
                        points_per_uid=args.points_per_uid,
                        selection_seed=args.selection_seed)
    print(json.dumps({"status": result["status"], "native_qualified": False}))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
