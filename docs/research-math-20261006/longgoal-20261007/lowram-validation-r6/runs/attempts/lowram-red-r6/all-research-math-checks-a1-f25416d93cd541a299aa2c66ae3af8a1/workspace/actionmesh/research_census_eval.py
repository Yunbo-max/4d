"""Evaluate a frozen ActionBench census without changing official metric budgets.

Each case contains sequence.npz (vertices[T,V,3], faces[F,3], optionally
frame_indices[T]) and report.json (uid). GT is GT_ROOT/uid/surfaces.npy.
An optional manifest contains cases [{case_id, uid, case_dir}]; every entry is
reported, including absent/failed predictions. case_dir is relative to CASE_ROOT.

The official ActionBench functions are imported, not reimplemented. Their
100,000 surface samples, 10,000 ICP samples, 24 initial rotations, 200 ICP
iterations, and final CD query subsampling are retained. A documented CPU RNG
device fix avoids passing a CPU device to torch.cuda.get_rng_state in the
official synchronized sampler. It changes neither draws nor sampling math.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
from pathlib import Path
import platform
import sys
import tempfile
import time
import traceback


OFFICIAL_FILES = (
    "benchmark.py", "chamfer.py", "icp.py", "sample_mesh.py",
    "sample_point_cloud.py", "evaluate_dataset.py",
)
PROTOCOL = {
    "n_pts_chamfer": 100_000,
    "n_pts_icp": 10_000,
    "icp_initial_rotations": 24,
    "icp_iterations": 200,
    "icp_learning_rate": 0.01,
    "icp_transform": "rotation + translation + anisotropic scale (official)",
    "cd_query_points_per_direction": 10_000,
    "cd_query_subsampling_seeds": {"predicted": 44, "ground_truth": 45},
    "cd_reference_tree_points": 100_000,
    "cd_distance": "unsquared Euclidean; sum of both directional means",
    "cd_3d_alignment": "independent ICP each frame",
    "cd_4d_alignment": "one ICP from frame zero, applied to all frames",
    "cd_motion_alignment": "same first-frame ICP as CD4D",
    "cd_motion_correspondence": (
        "predicted points share frame-zero face IDs and barycentric coordinates; "
        "nearest-neighbor matches to GT established only in frame zero"
    ),
    "units": "raw GT coordinate units; no extra normalization or scale multiplier",
    "gt_correspondence_assumption": (
        "surfaces.npy point index is a material correspondence across time, "
        "as assumed by the official evaluator"
    ),
    "no_extra_preprocessing": "no crop, camera fit, axis flip, interpolation, or frame truncation",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object: {path}")
    return value


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def discover_cases(case_root: Path, manifest: Path | None) -> tuple[list[dict], dict]:
    """Never intersect predictions with GT: failed/missing cases remain visible."""
    if manifest is None and (case_root / "manifest.json").is_file():
        manifest = case_root / "manifest.json"
    if manifest is not None:
        content = read_json(manifest)
        entries = content.get("cases")
        if not isinstance(entries, list) or not entries:
            raise ValueError("Manifest must have a nonempty cases list")
        provenance = {"frozen": True, "manifest": str(manifest),
                      "manifest_sha256": digest(manifest)}
    else:
        # A report without a sequence is an attempted case and must be counted.
        dirs = {p.parent for p in case_root.rglob("sequence.npz")}
        dirs.update(p.parent for p in case_root.rglob("report.json"))
        if (case_root / "sequence.npz").is_file() or (case_root / "report.json").is_file():
            dirs.add(case_root)
        if not dirs:
            raise ValueError("No cases found; supply a frozen manifest to count missing runs")
        entries = []
        for directory in sorted(dirs):
            report = read_json(directory / "report.json") if (directory / "report.json").is_file() else {}
            relative = str(directory.relative_to(case_root))
            entries.append({"case_id": relative if relative != "." else directory.name,
                            "uid": report.get("uid", directory.name), "case_dir": relative})
        provenance = {"frozen": False, "warning": "Discovered attempts only; not a preregistered cohort denominator"}
    cases, seen = [], set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Each manifest case must be an object")
        uid = str(entry.get("uid", ""))
        if not uid or Path(uid).name != uid or uid in (".", ".."):
            raise ValueError(f"Invalid UID: {uid!r}")
        case_id = str(entry.get("case_id", uid))
        if not case_id or case_id in seen:
            raise ValueError(f"Empty/duplicate case_id: {case_id!r}")
        seen.add(case_id)
        directory = case_root / str(entry.get("case_dir", uid))
        cases.append({"case_id": case_id, "uid": uid, "case_dir": str(directory),
                      "manifest_entry": entry})
    provenance["n_declared"] = len(cases)
    return cases, provenance


def load_arrays(sequence: Path, gt_path: Path, required_gt_points: int = 100_000):
    import numpy as np
    with np.load(sequence, allow_pickle=False) as saved:
        vertices = saved["vertices"].copy()
        faces = saved["faces"].copy()
        frame_indices = saved["frame_indices"].copy() if "frame_indices" in saved else None
    gt = np.load(gt_path, mmap_mode="r", allow_pickle=False)
    if vertices.ndim != 3 or vertices.shape[-1] != 3 or min(vertices.shape[:2]) < 1:
        raise ValueError("vertices must have shape [T,V,3]")
    if not np.issubdtype(vertices.dtype, np.floating) or not np.isfinite(vertices).all():
        raise ValueError("vertices must be finite floating-point coordinates")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError("faces must have nonempty shape [F,3]")
    if not np.issubdtype(faces.dtype, np.integer) or faces.min() < 0 or faces.max() >= vertices.shape[1]:
        raise ValueError("faces must be valid integer vertex indices")
    if gt.ndim != 3 or gt.shape[-1] != 6 or gt.shape[1] != required_gt_points:
        raise ValueError(f"GT must be [T,{required_gt_points},6]; got {gt.shape}")
    if vertices.shape[0] != gt.shape[0]:
        raise ValueError(f"Full timeline required: predicted {vertices.shape[0]} frames, GT {gt.shape[0]}")
    if frame_indices is not None:
        if not np.issubdtype(frame_indices.dtype, np.integer) or not np.array_equal(frame_indices, np.arange(len(gt))):
            raise ValueError("frame_indices must match the complete GT timeline exactly")
    if not np.isfinite(gt[..., :3]).all():
        raise ValueError("GT positions contain nonfinite coordinates")
    for frame in vertices:
        triangles = frame[faces]
        area2 = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                       triangles[:, 2] - triangles[:, 0]), axis=-1)
        if not np.isfinite(area2).all() or area2.sum() <= 0:
            raise ValueError("A predicted frame has no finite positive surface area")
    return vertices, faces, np.array(gt[..., :3], dtype=np.float32, copy=True)


def load_official(repo: Path, cpu_rng_fix: bool = True):
    bench_path = repo / "actionbench"
    if not all((bench_path / name).is_file() for name in OFFICIAL_FILES):
        raise FileNotFoundError(f"Incomplete official ActionBench source: {bench_path}")
    # The upstream source uses top-level imports. Refuse an unrelated cached module.
    for name in ("benchmark", "sample_mesh", "sample_point_cloud", "icp", "chamfer"):
        if name in sys.modules:
            origin = Path(getattr(sys.modules[name], "__file__", "")).resolve()
            if origin.parent != bench_path.resolve():
                raise RuntimeError(f"Conflicting imported module {name}: {origin}")
    sys.path.insert(0, str(bench_path.resolve()))
    import benchmark
    import sample_mesh
    patch = None
    if cpu_rng_fix:
        function = sample_mesh.get_baryc_sampling_mesh
        source = inspect.getsource(function)
        old = "devices=[verts.device], enabled=True"
        new = "devices=([verts.device] if verts.is_cuda else []), enabled=True"
        if old in source:
            if source.count(old) != 1:
                raise RuntimeError("Unexpected official CPU RNG source; review required")
            fixed = source.replace(old, new)
            namespace = dict(sample_mesh.__dict__)
            exec(compile(fixed, "<recorded-actionbench-cpu-rng-fix>", "exec"), namespace)
            sample_mesh.get_baryc_sampling_mesh = namespace[function.__name__]
            patch = {"function": "sample_mesh.get_baryc_sampling_mesh",
                     "old": old, "new": new,
                     "reason": "CPU sampler must not ask CUDA for the RNG state of a CPU device",
                     "patched_function_sha256": hashlib.sha256(fixed.encode()).hexdigest()}
        elif new not in source:
            raise RuntimeError("Official synchronized sampler changed; review CPU RNG compatibility")
    provenance = {
        "source_root": str(repo.resolve()),
        "sha256": {name: digest(bench_path / name) for name in OFFICIAL_FILES},
        "runtime_compatibility_patch": patch,
    }
    return benchmark.compute_chamfer_3d_4d, provenance


def summarize(results: list[dict]) -> dict:
    successful = [r for r in results if r["status"] == "success"]
    # Repeated inference seeds are repeated runs of the same asset, not new assets.
    uids = sorted({r.get("uid", r["case_id"]) for r in results})
    assets = []
    for uid in uids:
        runs = [r for r in results if r.get("uid", r["case_id"]) == uid]
        ok = [r for r in runs if r["status"] == "success"]
        assets.append({"uid": uid, "n_runs": len(runs), "n_success": len(ok),
                       "means": {key: sum(r[key] for r in ok) / len(ok) if ok else None
                                 for key in ("cd_3d", "cd_4d", "cd_motion")}})
    successful_assets = [a for a in assets if a["n_success"]]
    return {
        "n_total": len(results), "n_success": len(successful),
        "n_failed": sum(r["status"] == "error" for r in results),
        "n_pending": sum(r["status"] == "pending" for r in results),
        "n_validated_only": sum(r["status"] == "validated_only" for r in results),
        "success_rate": len(successful) / len(results) if results else 0.0,
        "mean_scope": "successful cases only; never treat failures as zero",
        "means_weighting": "case/run weighted; asset-balanced means also provided",
        "n_unique_assets": len(assets),
        "n_assets_with_success": len(successful_assets),
        "asset_results": assets,
        "asset_balanced_means": {
            key: sum(a["means"][key] for a in successful_assets) / len(successful_assets)
            if successful_assets else None for key in ("cd_3d", "cd_4d", "cd_motion")
        },
        "successful_case_ids": [r["case_id"] for r in successful],
        "means": {key: sum(r[key] for r in successful) / len(successful) if successful else None
                  for key in ("cd_3d", "cd_4d", "cd_motion")},
    }


def evaluate(args) -> int:
    import numpy as np
    cases, denominator = discover_cases(args.case_dir, args.manifest)
    results = [dict(case, status="pending") for case in cases]
    report = {"schema_version": 1, "protocol": dict(PROTOCOL, sampling_seed=args.seed),
              "denominator": denominator, "cases": results,
              "device": args.device, "python": platform.python_version(),
              "evaluator_sha256": digest(Path(__file__)), "official_source": None,
              "packages": {}}
    for package in ("numpy", "torch", "trimesh", "scipy", "pytorch3d"):
        try:
            report["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][package] = None
    backend = None
    backend_error = None
    if not args.validate_only:
        try:
            backend, report["official_source"] = load_official(args.repo_root, not args.disable_cpu_rng_fix)
        except Exception:
            backend_error = traceback.format_exc()
            report["backend_error"] = backend_error
    for result in results:
        started = time.monotonic()
        try:
            directory = Path(result["case_dir"])
            paths = {"sequence": directory / "sequence.npz",
                     "ground_truth": args.gt_dir / result["uid"] / "surfaces.npy"}
            if (directory / "report.json").is_file():
                paths["generation_report"] = directory / "report.json"
                generation = read_json(paths["generation_report"])
                result["generation_status"] = generation.get("status")
                if generation.get("uid") not in (None, result["uid"]):
                    raise ValueError("Manifest UID differs from generation report UID")
                if generation.get("status") in ("error", "failed"):
                    raise ValueError("Generation report declares failure; stale sequence not evaluated")
            result["inputs"] = {}
            for name, path in paths.items():
                result["inputs"][name] = {"path": str(path), "sha256": digest(path)}
            vertices, faces, gt = load_arrays(paths["sequence"], paths["ground_truth"])
            result["shapes"] = {"vertices": list(vertices.shape), "faces": list(faces.shape),
                                "gt_positions": list(gt.shape)}
            if args.validate_only:
                result["status"] = "validated_only"
            else:
                if backend_error:
                    raise RuntimeError("Official evaluator unavailable; see backend_error")
                import torch
                import trimesh
                meshes = [trimesh.Trimesh(vertices=v, faces=faces, process=False) for v in vertices]
                values = backend(torch.from_numpy(gt), meshes, device=args.device, is_4D=True,
                                 n_pts_icp=10_000, n_pts_chamfer=100_000, seed=args.seed)
                for key, value in zip(("cd_3d", "cd_4d", "cd_motion"), values):
                    value = float(value)
                    if not math.isfinite(value) or value < 0:
                        raise ValueError(f"Invalid official metric {key}: {value}")
                    result[key] = value
                result["status"] = "success"
        except Exception as exc:
            result.update(status="error", error=f"{type(exc).__name__}: {exc}",
                          traceback=traceback.format_exc())
        result["elapsed_seconds"] = time.monotonic() - started
        report["summary"] = summarize(results)
        write_json(args.output, report)
        print(json.dumps({k: result.get(k) for k in ("case_id", "uid", "status", "cd_3d", "cd_4d", "cd_motion", "error")}))
    return 1 if any(r["status"] == "error" for r in results) else 0


def self_test() -> dict:
    """No model dependencies: validate loaders and denominator accounting only."""
    import numpy as np
    checks = []
    with tempfile.TemporaryDirectory(prefix="actionbench-census-eval-") as temporary:
        root = Path(temporary)
        vertices = np.array([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        gt = np.concatenate((vertices, np.zeros_like(vertices)), axis=-1)
        np.save(root / "surfaces.npy", gt)
        np.savez(root / "sequence.npz", vertices=vertices, faces=faces, frame_indices=[0])
        got = load_arrays(root / "sequence.npz", root / "surfaces.npy", required_gt_points=3)
        assert np.array_equal(got[0], vertices) and np.array_equal(got[1], faces)
        checks.append("loader preserves vertex/face order and positions exactly")
        np.savez(root / "sequence.npz", vertices=vertices, faces=faces, frame_indices=[1])
        try:
            load_arrays(root / "sequence.npz", root / "surfaces.npy", required_gt_points=3)
        except ValueError as exc:
            assert "timeline" in str(exc)
        else:
            raise AssertionError("Wrong frame mapping accepted")
        checks.append("mismatched frame mapping rejected")
        manifest = root / "manifest.json"
        write_json(manifest, {"cases": [{"uid": "present"}, {"uid": "missing"}]})
        cases, denominator = discover_cases(root, manifest)
        assert len(cases) == 2 and denominator["n_declared"] == 2
        checks.append("absent predictions retained in frozen denominator")
        stats = summarize([{"case_id": "a", "status": "success", "cd_3d": 2.,
                            "cd_4d": 3., "cd_motion": 4.}, {"case_id": "b", "status": "error"}])
        assert stats["success_rate"] == .5 and stats["means"]["cd_motion"] == 4.
        checks.append("failures do not enter means as zeros")
    return {"status": "pass", "checks": checks,
            "scope": "I/O and census accounting; not an execution of PyTorch3D ICP or official metrics"}


def metric_self_test(repo: Path) -> dict:
    """Small analytic fixtures run the actual official Chamfer functions."""
    import numpy as np
    import torch
    spec = importlib.util.spec_from_file_location("census_official_chamfer", repo / "actionbench/chamfer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    points = torch.tensor([[0., 0., 0.], [10., 0., 0.]], dtype=torch.float64)
    shifted = points + torch.tensor([0., 3., 4.], dtype=torch.float64)
    assert abs(module.compute_chamfer_score(points, points)) < 1e-12
    assert abs(module.compute_chamfer_score(points, shifted) - 10.) < 1e-12
    truth = torch.stack((points, shifted))
    swapped = torch.stack((points, shifted.flip(0)))
    assert abs(module.compute_motion_chamfer_score(truth, truth)) < 1e-12
    # Identical per-frame surfaces, but the two material identities swap in frame 1.
    assert abs(module.compute_chamfer_score(swapped[1], truth[1])) < 1e-12
    assert abs(module.compute_motion_chamfer_score(swapped, truth) - 10.) < 1e-12
    return {"status": "pass", "checks": ["identity zero", "unsquared bidirectional distance",
            "material swap has zero per-frame CD but nonzero CD-M"],
            "source_sha256": digest(repo / "actionbench/chamfer.py"),
            "scope": "official distance formulas; ICP/sampling integration is separate"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path)
    parser.add_argument("--gt-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent / "repo")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--disable-cpu-rng-fix", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--metric-self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test or args.metric_self_test:
        result = metric_self_test(args.repo_root) if args.metric_self_test else self_test()
        if args.output:
            write_json(args.output, result)
        print(json.dumps(result, indent=2))
        return 0
    if any(getattr(args, name) is None for name in ("case_dir", "gt_dir", "output")):
        parser.error("--case-dir, --gt-dir and --output are required")
    return evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
