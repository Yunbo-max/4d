"""Hash-qualified cache I/O shared by research runners; no GT or model imports."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np

BASE_COMMIT = "81f4f48330aef6d0c00dce9826335a3b1287850e"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"not JSON serializable: {type(item).__name__}")
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, default=convert, allow_nan=False)+"\n")
    temp.replace(path)


def validate_mesh(tracks, faces, times):
    if (tracks.ndim != 3 or tracks.shape[-1] != 3 or min(tracks.shape[:2]) < 1
            or tracks.dtype.kind != "f" or not np.isfinite(tracks).all()):
        raise ValueError("finite floating-point trajectories [T,V,3] required")
    if (faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0
            or faces.dtype.kind not in "iu" or faces.min() < 0 or faces.max() >= tracks.shape[1]
            or np.any(np.diff(np.sort(faces, axis=1), axis=1) == 0)):
        raise ValueError("nonempty integer triangles with distinct valid vertex IDs required")
    if times.shape != (len(tracks),) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("finite strictly increasing actual frame times required")


def load_census_case(directory):
    """Read any completed native census UID/seed, not a hardcoded two-asset set.

    The published cache's declared hashes, full clock, anchor and vertex identity
    must match. This checks provenance and structure, not physical correctness.
    """
    directory = Path(directory)
    report = json.loads((directory/"report.json").read_text())
    uid, seed = report.get("uid"), report.get("seed")
    if (report.get("status") != "completed" or report.get("frames") != 16
            or not isinstance(uid, str) or not uid or Path(uid).name != uid or uid in (".", "..")
            or isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        raise ValueError("completed native 16-frame case with valid UID and seed required")
    hashes = {"report.json": digest(directory/"report.json")}
    for name in ("prepared.npz", "denoised.npz", "sequence.npz"):
        hashes[name] = digest(directory/name)
        if hashes[name] != report.get("sha256", {}).get(name):
            raise ValueError("native cache hash mismatch: " + name)
    with np.load(directory/"prepared.npz", allow_pickle=False) as data:
        prepared = {key: data[key].copy() for key in ("timesteps", "anchor_latent", "anchor_timesteps",
            "anchor_vertices", "anchor_faces", "anchor_query_features", "query_vertex_ids", "seed", "context")}
    with np.load(directory/"denoised.npz", allow_pickle=False) as data:
        latents, times, latent_seed = data["latents"].copy(), data["timesteps"].copy(), data["seed"].copy()
    with np.load(directory/"sequence.npz", allow_pickle=False) as data:
        tracks, faces = data["vertices"].copy(), data["faces"].copy()
        frame_ids, mesh_times, ids = data["frame_indices"].copy(), data["timesteps"].copy(), data["query_vertex_ids"].copy()
    validate_mesh(tracks, faces, mesh_times)
    if (latents.shape != (16, 2048, 64) or latents.dtype.kind != "f" or not np.isfinite(latents).all()
            or prepared["context"].ndim != 3 or len(prepared["context"]) != 16
            or min(prepared["context"].shape) < 1 or prepared["context"].dtype.kind != "f"
            or not np.isfinite(prepared["context"]).all()):
        raise ValueError("finite native latents [16,2048,64] and context [16,S,D] required")
    for clock in (times, mesh_times, frame_ids, prepared["timesteps"]):
        if not np.array_equal(clock, np.arange(16)):
            raise ValueError("every native frame 0..15 must occur once in order")
    for saved_seed in (latent_seed, prepared["seed"]):
        if saved_seed.shape != () or saved_seed.dtype.kind not in "iu" or saved_seed.item() != seed:
            raise ValueError("cache and report seeds differ")
    if (prepared["anchor_latent"].shape != (1, 2048, 64)
            or not np.array_equal(prepared["anchor_timesteps"], [0])
            or not np.array_equal(prepared["anchor_latent"][0], latents[0])
            or not np.array_equal(prepared["anchor_vertices"], tracks[0])
            or not np.array_equal(prepared["anchor_faces"], faces)):
        raise ValueError("native source anchor changed")
    if not np.array_equal(ids, np.arange(tracks.shape[1])) or not np.array_equal(ids, prepared["query_vertex_ids"]):
        raise ValueError("material query IDs changed")
    query = prepared["anchor_query_features"]
    if (query.shape != (tracks.shape[1], 6) or query.dtype != np.float32
            or not np.isfinite(query).all() or not np.array_equal(query[:, :3], tracks[0])
            or not np.allclose(np.linalg.norm(query[:, 3:], axis=1), 1., rtol=0, atol=1e-4)):
        raise ValueError("original finite FP32 XYZ/unit-normal queries required")
    return report, hashes, latents, tracks, faces, prepared


def save_sequence(directory, tracks, faces, times, query_ids=None):
    validate_mesh(tracks, faces, times)
    ids = np.arange(tracks.shape[1], dtype=np.int64) if query_ids is None else query_ids
    np.savez_compressed(Path(directory)/"sequence.npz", vertices=tracks, faces=faces,
        timesteps=times, frame_indices=np.arange(len(tracks), dtype=np.int64), query_vertex_ids=ids)
    return digest(Path(directory)/"sequence.npz")
