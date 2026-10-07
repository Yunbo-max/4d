"""Fail-closed semantic admission for an already admitted ActionBench dataset.

This is an engineering input check.  It revalidates the bytes consumed from the
dataset snapshot, then checks the released tensor, camera and PNG structures.  It
does not load model weights, run inference, score predictions, or qualify a
scientific protocol.  Execute it only through the harness plan emitted by
``prepare_actionbench_dataset_semantics.py``.
"""
from __future__ import annotations

import argparse
import binascii
import hashlib
import io
import json
from pathlib import Path
import stat
import struct
import zlib

import numpy as np


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _regular_file(path: Path, label: str):
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode):
        raise ValueError("Symlink forbidden in semantic input: " + label)
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("Non-regular semantic input: " + label)
    return info


def _read_bound(path: Path, expected: dict, label: str,
                max_bytes: int | None = None) -> bytes:
    before = _regular_file(path, label)
    if expected.get("path") != label or expected.get("bytes") != before.st_size:
        raise ValueError("Semantic input differs from snapshot admission: " + label)
    if max_bytes is not None and before.st_size > max_bytes:
        raise ValueError("Semantic input exceeds frozen parser byte bound: " + label)
    payload = path.read_bytes()
    after = _regular_file(path, label)
    identity_before = (before.st_dev, before.st_ino, before.st_size,
                       before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size,
                      after.st_mtime_ns)
    if identity_before != identity_after:
        raise ValueError("Semantic input changed while reading: " + label)
    digest = hashlib.sha256(payload).hexdigest()
    if expected.get("sha256") != digest:
        raise ValueError("Semantic input differs from snapshot admission: " + label)
    return payload


def _physical_child(root: Path, relative: str) -> Path:
    candidate = root / relative
    parent = candidate.parent
    while parent != root:
        info = parent.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ValueError("Symlink or non-directory semantic parent: " + relative)
        parent = parent.parent
    return candidate


def _admitted_files(admission: dict, revision: str, population_size: int,
                    uid_digest: str) -> tuple[dict, str]:
    if (admission.get("kind") != "actionbench-full128-snapshot-admission" or
            admission.get("version") != "1.0.0" or
            admission.get("status") != "admitted_engineering_snapshot" or
            admission.get("all_revisions_immutable") is not True or
            admission.get("all_content_files_hashed") is not True or
            admission.get("scientific_effect_qualification") is not False or
            admission.get("dispatch_ready") is not False):
        raise ValueError("A successful snapshot admission is required")
    dataset = admission.get("snapshots", {}).get("dataset", {})
    if (dataset.get("repository") != "facebook/actionbench" or
            dataset.get("revision") != revision):
        raise ValueError("Snapshot admission dataset revision mismatch")
    records = dataset.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("Snapshot admission has no dataset file manifest")
    indexed = {}
    for record in records:
        path = record.get("path") if isinstance(record, dict) else None
        if not isinstance(path, str) or not path or path in indexed:
            raise ValueError("Invalid or duplicate admitted dataset path")
        indexed[path] = record
    manifest_digest = hashlib.sha256(canonical(records)).hexdigest()
    if (dataset.get("file_count") != len(records) or
            dataset.get("manifest_sha256") != manifest_digest):
        raise ValueError("Snapshot admission dataset manifest mismatch")
    layout = admission.get("dataset", {})
    if (layout.get("dataset") != "facebook/actionbench" or
            layout.get("revision") != revision or
            layout.get("population_size") != population_size or
            layout.get("frames_per_sample") != 16 or
            layout.get("uid_manifest_sha256") != uid_digest or
            layout.get("required_files_per_sample") != 18):
        raise ValueError("Snapshot admission dataset layout mismatch")
    return indexed, manifest_digest


def _png_ihdr(payload: bytes, label: str) -> dict:
    if len(payload) < 33 or payload[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Invalid PNG signature: " + label)
    length = struct.unpack(">I", payload[8:12])[0]
    if length != 13 or payload[12:16] != b"IHDR":
        raise ValueError("PNG does not start with a complete IHDR: " + label)
    width, height, depth, colour, compression, filtering, interlace = \
        struct.unpack(">IIBBBBB", payload[16:29])
    if width <= 0 or height <= 0 or width != height:
        raise ValueError("ActionBench frame must be a nonempty square PNG: " + label)
    if colour != 6:
        raise ValueError("ActionBench frame is not encoded as RGBA PNG: " + label)
    if (depth not in {8, 16} or compression != 0 or filtering != 0 or
            interlace not in {0, 1}):
        raise ValueError("Unsupported ActionBench PNG encoding: " + label)
    offset = 8
    chunks = []
    idat = []
    while offset < len(payload):
        if offset + 12 > len(payload):
            raise ValueError("Truncated PNG chunk: " + label)
        chunk_length = struct.unpack(">I", payload[offset:offset + 4])[0]
        end = offset + 12 + chunk_length
        if end > len(payload):
            raise ValueError("Truncated PNG chunk payload: " + label)
        chunk_type = payload[offset + 4:offset + 8]
        chunk_data = payload[offset + 8:offset + 8 + chunk_length]
        expected_crc = struct.unpack(">I", payload[end - 4:end])[0]
        actual_crc = binascii.crc32(chunk_type + chunk_data) & 0xFFFFFFFF
        if expected_crc != actual_crc:
            raise ValueError("PNG chunk CRC mismatch: " + label)
        chunks.append(chunk_type)
        if chunk_type == b"IDAT":
            idat.append(chunk_data)
        if chunk_type == b"IEND" and chunk_length != 0:
            raise ValueError("Invalid PNG IEND payload: " + label)
        offset = end
        if chunk_type == b"IEND":
            break
    if (offset != len(payload) or not chunks or chunks[0] != b"IHDR" or
            b"IDAT" not in chunks or chunks[-1] != b"IEND"):
        raise ValueError("Incomplete PNG chunk sequence: " + label)
    try:
        decoder = zlib.decompressobj()
        scanlines = decoder.decompress(b"".join(idat)) + decoder.flush()
    except zlib.error as error:
        raise ValueError("Invalid PNG compressed image data: " + label) from error
    if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError("Incomplete PNG compressed image stream: " + label)
    if interlace == 0:
        row_bytes = 1 + width * 4 * (depth // 8)
        if len(scanlines) != height * row_bytes or any(
                scanlines[row * row_bytes] > 4 for row in range(height)):
            raise ValueError("Invalid PNG scanline structure: " + label)
    return {"width": width, "height": height, "bit_depth": depth,
            "colour_type": colour, "interlace": interlace}


def _finite_array(value, shape, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != tuple(shape) or not np.isfinite(array).all():
        raise ValueError("Invalid finite shape for " + label)
    return array


def _camera(payload: bytes, contract: dict, label: str) -> dict:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Invalid camera JSON: " + label) from error
    keys = contract["camera"]["required_keys"]
    if not isinstance(value, dict) or not set(keys).issubset(value):
        raise ValueError("Required camera keys missing: " + label)
    rotation = _finite_array(value["R"], (3, 3), label + ":R")
    translation = _finite_array(value["T"], (3,), label + ":T")
    focal = _finite_array(value["focal_length_ndc"], (2,), label + ":focal")
    principal = _finite_array(value["principal_point_ndc"], (2,),
                              label + ":principal")
    if not np.all(focal > 0):
        raise ValueError("Camera focal lengths must be positive: " + label)
    tolerance = contract["camera"]["rotation_tolerance"]
    orthogonal_error = float(np.max(np.abs(rotation @ rotation.T - np.eye(3))))
    determinant = float(np.linalg.det(rotation))
    if orthogonal_error > tolerance or abs(determinant - 1.0) > tolerance:
        raise ValueError("Camera rotation is not a proper orthonormal matrix: " + label)
    return {
        "rotation_orthogonal_max_abs_error": orthogonal_error,
        "rotation_determinant": determinant,
        "translation": translation.tolist(),
        "focal_length_ndc": focal.tolist(),
        "principal_point_ndc": principal.tolist(),
        "extra_keys": sorted(set(value) - set(keys)),
    }


def _surfaces(payload: bytes, contract: dict, label: str) -> dict:
    try:
        value = np.load(io.BytesIO(payload), allow_pickle=False)
    except (ValueError, OSError) as error:
        raise ValueError("Invalid non-pickle NumPy surface tensor: " + label) from error
    expected_shape = tuple(contract["surfaces"]["shape"])
    if value.shape != expected_shape:
        raise ValueError("ActionBench surface tensor shape mismatch: " + label)
    if value.dtype.kind != "f" or value.dtype.itemsize not in \
            contract["surfaces"]["allowed_float_item_bytes"]:
        raise ValueError("ActionBench surface tensor dtype mismatch: " + label)
    if not np.isfinite(value).all():
        raise ValueError("ActionBench surface tensor contains nonfinite values: " + label)
    positions = value[..., :3]
    normals = value[..., 3:]
    lower, upper = contract["surfaces"]["position_bounds"]
    tolerance = contract["surfaces"]["position_bound_tolerance"]
    position_min = float(positions.min())
    position_max = float(positions.max())
    if position_min < lower - tolerance or position_max > upper + tolerance:
        raise ValueError("ActionBench positions exceed normalized-space bound: " + label)
    normal_norms = np.linalg.norm(normals.astype(np.float64), axis=-1)
    if not np.isfinite(normal_norms).all():
        raise ValueError("ActionBench normal norms overflow: " + label)
    return {
        "shape": list(value.shape),
        "dtype": value.dtype.str,
        "position_min": position_min,
        "position_max": position_max,
        "normal_norm_min": float(normal_norms.min()),
        "normal_norm_max": float(normal_norms.max()),
        "zero_normal_count": int(np.count_nonzero(normal_norms == 0)),
    }


def validate_contract(contract: dict, population: dict):
    if (contract.get("kind") != "actionbench-full128-dataset-semantics-contract" or
            contract.get("version") != "1.0.0"):
        raise ValueError("Unsupported ActionBench dataset semantics contract")
    if (population.get("dataset") != "facebook/actionbench" or
            population.get("revision") != contract.get("dataset_revision")):
        raise ValueError("Semantic contract/population dataset mismatch")
    uids = population.get("uids")
    if (not isinstance(uids, list) or len(uids) != contract.get("population_size") or
            len(uids) != len(set(uids))):
        raise ValueError("Semantic contract/population cardinality mismatch")
    expected_digest = contract.get("uid_set_sha256")
    if hashlib.sha256(canonical(uids)).hexdigest() != expected_digest:
        raise ValueError("Semantic contract released UID digest mismatch")
    surfaces = contract.get("surfaces", {})
    if (surfaces.get("shape") != [16, 100000, 6] or
            surfaces.get("position_bounds") != [-1.0, 1.0] or
            surfaces.get("allowed_float_item_bytes") != [4, 8] or
            not isinstance(surfaces.get("position_bound_tolerance"), (int, float)) or
            not 0 <= surfaces["position_bound_tolerance"] <= 1e-4):
        raise ValueError("Invalid frozen ActionBench surface semantics")
    camera = contract.get("camera", {})
    if (camera.get("required_keys") != ["R", "T", "focal_length_ndc",
                                         "principal_point_ndc"] or
            not isinstance(camera.get("rotation_tolerance"), (int, float)) or
            not 0 < camera["rotation_tolerance"] <= 1e-3):
        raise ValueError("Invalid frozen ActionBench camera semantics")
    limits = contract.get("parser_resource_limits", {})
    if (limits.get("camera_max_bytes") != 65536 or
            limits.get("surface_max_bytes") != 83886080 or
            limits.get("frame_max_bytes") != 67108864):
        raise ValueError("Invalid frozen ActionBench parser resource limits")


def admit(contract: dict, population: dict, snapshot_admission: dict,
          dataset_root: Path, output: Path):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Preserve existing dataset semantics admission: " +
                              str(output))
    validate_contract(contract, population)
    revision = contract["dataset_revision"]
    admitted, snapshot_manifest_digest = _admitted_files(
        snapshot_admission, revision, contract["population_size"],
        contract["uid_set_sha256"])
    dataset_root = Path(dataset_root)
    if not dataset_root.is_dir() or dataset_root.is_symlink():
        raise ValueError("Invalid admitted ActionBench dataset root")
    dataset_root = dataset_root.resolve()
    records = []
    image_encodings = {}
    limits = contract["parser_resource_limits"]
    for uid in population["uids"]:
        prefix = "data/" + uid + "/"
        used = []
        camera_label = prefix + "camera.json"
        camera_bytes = _read_bound(_physical_child(dataset_root, camera_label),
                                   admitted.get(camera_label, {}), camera_label,
                                   limits["camera_max_bytes"])
        used.append(admitted[camera_label])
        camera = _camera(camera_bytes, contract, camera_label)
        surface_label = prefix + "surfaces.npy"
        surface_bytes = _read_bound(_physical_child(dataset_root, surface_label),
                                    admitted.get(surface_label, {}), surface_label,
                                    limits["surface_max_bytes"])
        used.append(admitted[surface_label])
        surfaces = _surfaces(surface_bytes, contract, surface_label)
        frame_layout = None
        for frame in range(contract["frames_per_sample"]):
            frame_label = prefix + "imgs/" + f"{frame:02d}.png"
            frame_bytes = _read_bound(_physical_child(dataset_root, frame_label),
                                      admitted.get(frame_label, {}), frame_label,
                                      limits["frame_max_bytes"])
            used.append(admitted[frame_label])
            parsed = _png_ihdr(frame_bytes, frame_label)
            if frame_layout is None:
                frame_layout = parsed
            elif parsed != frame_layout:
                raise ValueError("Frame encoding differs within sample: " + uid)
        encoding_key = canonical(frame_layout).decode()
        image_encodings.setdefault(encoding_key, {**frame_layout, "sample_count": 0})
        image_encodings[encoding_key]["sample_count"] += 1
        records.append({
            "uid": uid,
            "input_manifest_sha256": hashlib.sha256(canonical(used)).hexdigest(),
            "camera": camera,
            "surfaces": surfaces,
            "frames": {"count": contract["frames_per_sample"], **frame_layout},
        })
    result = {
        "kind": "actionbench-full128-dataset-semantics-admission",
        "version": "1.0.0",
        "status": "admitted_engineering_dataset_semantics",
        "scope": ("Byte-bound tensor/camera/RGBA structure validation for the pinned "
                  "current-public-release ActionBench inputs; not inference, scoring, "
                  "tracked-correspondence proof, runtime qualification, or scientific "
                  "effect evidence"),
        "dataset": "facebook/actionbench",
        "revision": revision,
        "population_size": len(records),
        "frames_per_sample": contract["frames_per_sample"],
        "snapshot_dataset_manifest_sha256": snapshot_manifest_digest,
        "image_encodings": [image_encodings[key]
                            for key in sorted(image_encodings)],
        "samples": records,
        "sample_semantics_sha256": hashlib.sha256(canonical(records)).hexdigest(),
        "all_consumed_bytes_revalidated": True,
        "scientific_effect_qualification": False,
        "dispatch_ready": False,
        "queue_generated": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2, ensure_ascii=False,
                                allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--snapshot-admission", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = admit(
        json.loads(args.contract.read_text()),
        json.loads(args.population.read_text()),
        json.loads(args.snapshot_admission.read_text()),
        args.dataset_root, args.output)
    print(json.dumps({"output": str(args.output), "status": result["status"],
                      "scientific_effect_qualification": False,
                      "dispatch_ready": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
