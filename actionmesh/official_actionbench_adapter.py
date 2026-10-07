"""Format-only adapter around the pinned official ActionBench dataset CLI.

The adapter converts a frozen NPZ mesh sequence to the GLB directory layout
accepted by ``actionbench/evaluate_dataset.py``, executes that official script in
a fresh process, and normalizes its one-row CSV into retained JSON.  It does not
implement or modify any metric.  Scientific execution belongs inside the
research-autopilot harness.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback


METRICS = ("cd_3d", "cd_4d", "cd_motion")
OFFICIAL_FILES = (
    "benchmark.py", "chamfer.py", "icp.py", "sample_mesh.py",
    "sample_point_cloud.py", "evaluate_dataset.py",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def manifest_cases(case_root: Path, manifest_path: Path) -> tuple[list[dict], dict]:
    case_root = Path(case_root).resolve()
    manifest_path = Path(manifest_path).resolve()
    manifest = read_json(manifest_path)
    entries = manifest.get("cases")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Frozen manifest must contain a nonempty cases list")
    cases, seen = [], set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Manifest case must be an object")
        uid, case_id = entry.get("uid"), entry.get("case_id")
        relative = Path(str(entry.get("case_dir", "")))
        if (not isinstance(uid, str) or not uid or Path(uid).name != uid or
                uid in (".", "..")):
            raise ValueError("Invalid ActionBench UID")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError("Case IDs must be nonempty and unique")
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Case directories must be relative and nonescaping")
        directory = (case_root / relative).resolve()
        directory.relative_to(case_root)
        seen.add(case_id)
        cases.append({"uid": uid, "case_id": case_id, "case_dir": str(directory)})
    return cases, {
        "frozen": True,
        "manifest": str(manifest_path),
        "manifest_sha256": digest(manifest_path),
        "n_declared": len(cases),
    }


def parse_official_csv(path: Path, uid: str) -> dict:
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1 or rows[0].get("uid") != uid:
        raise ValueError("Official CLI must return exactly the requested UID")
    row = rows[0]
    if row.get("status") != "success":
        raise RuntimeError("Official CLI row failed: " + row.get("error_message", ""))
    values = {}
    for metric in METRICS:
        value = float(row[metric])
        if not math.isfinite(value) or value < 0:
            raise ValueError("Official CLI returned an invalid metric: " + metric)
        values[metric] = value
    frames = int(row["n_frames"])
    if frames <= 0:
        raise ValueError("Official CLI returned an invalid frame count")
    return {"uid": uid, "status": "success", "n_frames": frames, **values}


def export_official_prediction(sequence: Path, destination: Path) -> dict:
    """Export one sequence and prove the retained GLBs round-trip exactly."""
    import numpy as np
    import trimesh

    with np.load(sequence, allow_pickle=False) as saved:
        vertices = saved["vertices"].copy()
        faces = saved["faces"].copy()
        frame_indices = saved["frame_indices"].copy() if "frame_indices" in saved else None
    if (vertices.ndim != 3 or vertices.shape[-1] != 3 or
            not np.issubdtype(vertices.dtype, np.floating) or
            not np.isfinite(vertices).all()):
        raise ValueError("Prediction vertices must be finite floating [T,V,3]")
    if (faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or
            not np.issubdtype(faces.dtype, np.integer) or faces.min() < 0 or
            faces.max() >= vertices.shape[1]):
        raise ValueError("Prediction faces must be valid integer [F,3]")
    if frame_indices is not None and not np.array_equal(frame_indices, np.arange(len(vertices))):
        raise ValueError("Prediction timeline must be complete and ordered")
    if vertices.dtype != np.float32:
        raise ValueError("Official GLB parity export requires source float32 vertices")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    files = []
    for index, frame in enumerate(vertices):
        path = destination / f"mesh_{index:05d}.glb"
        mesh = trimesh.Trimesh(vertices=frame, faces=faces, process=False, validate=False)
        mesh.export(path, file_type="glb")
        loaded = trimesh.load(path, force="mesh", process=False)
        loaded_vertices = np.asarray(loaded.vertices, dtype=np.float32)
        loaded_faces = np.asarray(loaded.faces, dtype=faces.dtype)
        if not np.array_equal(loaded_vertices, frame) or not np.array_equal(loaded_faces, faces):
            raise ValueError(f"GLB round-trip changed frame {index}")
        files.append({"frame_index": index, "path": str(path), "sha256": digest(path),
                      "bytes": path.stat().st_size})
    return {"source_sequence": str(Path(sequence).resolve()),
            "source_sequence_sha256": digest(sequence),
            "vertices_dtype": str(vertices.dtype), "faces_dtype": str(faces.dtype),
            "n_frames": len(vertices), "n_vertices": vertices.shape[1],
            "n_faces": faces.shape[0], "files": files,
            "round_trip": "exact vertices and faces for every frame"}


def official_command(script: Path, gt_root: Path, pred_root: Path,
                     output_csv: Path, device: str, seed: int,
                     cpu_knn_backward: bool = False) -> list[str]:
    entry = ([str(Path(__file__).with_name('deterministic_actionbench_entry.py')),
              '--official-script', str(script)] if cpu_knn_backward else [str(script)])
    return [sys.executable, *entry, "--gt_root", str(gt_root),
            "--pred_root", str(pred_root), "--output_csv", str(output_csv),
            "--device", device, "--n_pts_icp", "10000",
            "--n_pts_chamfer", "100000", "--seed", str(seed),
            "--mesh_pattern", "mesh_*.glb", "--recompute"]


def prepare_official_source(source: Path, destination: Path) -> dict:
    """Copy official sources and apply only the recorded CPU-RNG device fix.

    Upstream samples mesh surfaces on CPU before moving point clouds to the
    requested CUDA device.  Its ``fork_rng`` call nevertheless passes a CPU
    device to the CUDA RNG API.  The faithful wrapper already applies this same
    source-local compatibility change.  Refuse every other source shape.
    """
    destination.mkdir(parents=True, exist_ok=False)
    original = {}
    for name in OFFICIAL_FILES:
        original[name] = digest(source / name)
        shutil.copyfile(source / name, destination / name)
    sample_mesh = destination / "sample_mesh.py"
    content = sample_mesh.read_text()
    old = "devices=[verts.device], enabled=True"
    new = "devices=([verts.device] if verts.is_cuda else []), enabled=True"
    if content.count(old) != 1:
        raise RuntimeError("Unexpected official CPU RNG source; review required")
    sample_mesh.write_text(content.replace(old, new))
    return {"original_sha256": original,
            "patched_source_sha256": {name: digest(destination / name)
                                       for name in OFFICIAL_FILES},
            "compatibility_patch": {
                "file": "sample_mesh.py",
                "function": "get_baryc_sampling_mesh",
                "old": old, "new": new,
                "reason": "CPU surface sampler must not query CUDA RNG state for a CPU device",
                "metric_or_draw_change": False,
            }}


def evaluate(args: argparse.Namespace) -> int:
    case_root, gt_root = args.case_dir.resolve(), args.gt_dir.resolve()
    output, repo_root = args.output.resolve(), args.repo_root.resolve()
    if output.exists() or output.with_name(output.name + ".official").exists():
        raise FileExistsError("Official adapter outputs are single-use")
    cases, denominator = manifest_cases(case_root, args.manifest)
    official_root = repo_root / "actionbench"
    if not all((official_root / name).is_file() for name in OFFICIAL_FILES):
        raise FileNotFoundError("Pinned official ActionBench source closure is incomplete")
    work = output.with_name(output.name + ".official")
    work.mkdir(parents=True, exist_ok=False)
    executed_source = work / "official-source"
    source_provenance = prepare_official_source(official_root, executed_source)
    report = {
        "schema_version": 1,
        "adapter_role": "format-only wrapper around official evaluate_dataset.py",
        "metric_implementation": "none; metrics come from the official CLI subprocess",
        "denominator": denominator,
        "device": args.device,
        "seed": args.seed,
        "python": platform.python_version(),
        "adapter_sha256": digest(Path(__file__)),
        "official_source": source_provenance,
        "cases": [{**case, "status": "pending"} for case in cases],
    }
    write_json(output, report)
    for result in report["cases"]:
        started = time.monotonic()
        case_work = work / result["case_id"]
        try:
            case_work.mkdir(parents=True, exist_ok=False)
            report_path = Path(result["case_dir"]) / "report.json"
            generation = read_json(report_path)
            if generation.get("uid") != result["uid"] or generation.get("status") != "completed":
                raise ValueError("Prediction report does not bind a completed requested UID")
            sequence = Path(result["case_dir"]) / "sequence.npz"
            pred_root = case_work / "predictions"
            export = export_official_prediction(sequence, pred_root / result["uid"])
            write_json(case_work / "export-manifest.json", export)
            gt = gt_root / result["uid"] / "surfaces.npy"
            if not gt.is_file():
                raise FileNotFoundError("Released GT missing for requested UID")
            csv_path = case_work / "official.csv"
            command = official_command(executed_source / "evaluate_dataset.py", gt_root,
                                       pred_root, csv_path, args.device, args.seed,
                                       getattr(args, 'cpu_knn_backward', False))
            environment = os.environ.copy()
            completed = subprocess.run(command, cwd=executed_source, env=environment,
                                       capture_output=True, text=True, timeout=args.timeout_seconds)
            (case_work / "stdout.log").write_text(completed.stdout)
            (case_work / "stderr.log").write_text(completed.stderr)
            execution = {"command": command, "cwd": str(executed_source),
                         "exit_code": completed.returncode,
                         "stdout_sha256": digest(case_work / "stdout.log"),
                         "stderr_sha256": digest(case_work / "stderr.log")}
            write_json(case_work / "execution.json", execution)
            if completed.returncode != 0:
                raise RuntimeError(f"Official ActionBench CLI exited {completed.returncode}")
            values = parse_official_csv(csv_path, result["uid"])
            result.update(values)
            if getattr(args, 'cpu_knn_backward', False):
                result['additional_runtime_compatibility'] = read_json(csv_path.with_suffix('.backend.json'))
            result["inputs"] = {"sequence": {"path": str(sequence), "sha256": digest(sequence)},
                                "ground_truth": {"path": str(gt), "sha256": digest(gt)}}
            result["official_outputs"] = {
                "csv": {"path": str(csv_path), "sha256": digest(csv_path)},
                "summary": {"path": str(csv_path.with_suffix(".summary.json")),
                            "sha256": digest(csv_path.with_suffix(".summary.json"))},
                "export_manifest": {"path": str(case_work / "export-manifest.json"),
                                    "sha256": digest(case_work / "export-manifest.json")},
                "execution": {"path": str(case_work / "execution.json"),
                              "sha256": digest(case_work / "execution.json")},
            }
        except Exception as exc:
            result.update(status="error", error=f"{type(exc).__name__}: {exc}",
                          traceback=traceback.format_exc())
        result["elapsed_seconds"] = time.monotonic() - started
        write_json(output, report)
    successes = [row for row in report["cases"] if row["status"] == "success"]
    report["summary"] = {"n_total": len(report["cases"]), "n_success": len(successes),
                         "n_failed": len(report["cases"]) - len(successes),
                         "success_rate": len(successes) / len(report["cases"])}
    write_json(output, report)
    return 0 if len(successes) == len(report["cases"]) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("case-dir", "gt-dir", "output", "manifest", "repo-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cuda:0"), required=True)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument('--cpu-knn-backward', action='store_true')
    args = parser.parse_args()
    if args.seed != 44 or not 1 <= args.timeout_seconds <= 27000:
        parser.error("Frozen scorer seed 44 and finite timeout <=27000 required")
    return evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
