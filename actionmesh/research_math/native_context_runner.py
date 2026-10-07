"""Future harness inner task for paired official generation and decoder replay.

This source is generated/unexecuted. It neither starts a campaign nor creates
an approved GPU plan. Run only inside a separately admitted engineering harness
attempt after GPU STOP is lifted. Every outcome remains scientifically unqualified.
The frozen complete-unit runner and its output closure are not changed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback


PREREQUISITE_PATHS = (
    "contract", "population", "snapshot_contract", "snapshot_admission",
    "dataset_semantics", "unit_manifest", "source_root", "dataset_root", "weights_root",
)
INPUT_FILES = PREREQUISITE_PATHS[:6]
SOURCE_FILES = (
    "inference/video_to_animated_mesh.py", "actionmesh/pipeline.py",
    "actionmesh/model/temporal_autoencoder.py", "actionmesh/model/utils/embeddings.py",
    "actionmesh/io/mesh_io.py", "actionmesh/io/video_input.py",
    "actionmesh/configs/actionmesh.yaml", "actionmesh/configs/actionmesh_lowram.yaml",
)
EFFECTIVE_PARAMETERS = {
    "stage_0_steps": 100, "stage_1_steps": 30, "face_decimation": 40000,
    "floaters_threshold": .02, "guidance_scales": [7.5], "anchor_idx": 0,
    "temporal_context_size": 16, "sliding_window_denoiser": 15,
    "subsampling_level": 1, "sliding_window_autoencoder": 15,
}


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load(path):
    return json.loads(Path(path).read_text())


def tolerance(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Explicit finite nonnegative comparison tolerance required")
    return float(value)


def generation_settings(generation, source_root, frames, output):
    """Translate the exact frozen profile into the actual official Python API."""
    from research_math.complete_unit_contract import validate_generation_profile
    profile = validate_generation_profile(generation)
    if generation.get("effective_parameters") != EFFECTIVE_PARAMETERS:
        raise ValueError("Frozen complete native generation parameters changed")
    pipeline = {
        "config_name": Path(generation["config"]).name,
        "config_dir": str(Path(source_root) / "actionmesh/configs"),
        "dtype": generation["dtype"], "lazy_loading": profile == "fp16-lowram-v1",
    }
    run = {"input": str(frames), "output_dir": str(output), "seed": 42, "blender_path": None}
    for key in ("stage_0_steps", "stage_1_steps", "face_decimation", "floaters_threshold",
                "guidance_scales", "anchor_idx"):
        run[key] = generation["effective_parameters"][key]
    return pipeline, run


def stage_environment(source_root, project_actionmesh):
    environment = os.environ.copy()
    environment.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
        HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
        PYTHONPATH=os.pathsep.join(map(str, (source_root,
            Path(source_root) / "third_party/TripoSG", project_actionmesh))),
    )
    return environment


def physical_file(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or any(p.is_symlink() for p in path.parents):
        raise ValueError("Physical retained file required: " + str(path))
    return path


def capture_windows(capture_root):
    """Resolve only canonical relative, ordered and hash-bound capture rows."""
    from research_math.pipeline_decoder_observer import load_window
    root = Path(capture_root)
    index = load(physical_file(root / "index.json"))
    if (index.get("kind") != "pipeline-decoder-observation" or index.get("version") != 1
            or index.get("status") != "captured_unqualified"
            or index.get("scientific_effect_qualification") is not False
            or index.get("native_context_qualified") is not False
            or index.get("identity_sha256") != digest(physical_file(root / "identity.json"))):
        raise ValueError("Complete unqualified capture index with identity binding required")
    rows = index.get("windows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("At least one complete native window required")
    paths = []
    for number, row in enumerate(rows):
        expected = "window-" + format(number, "04d")
        if (not isinstance(row, dict) or row.get("path") != expected
                or row.get("status") != "captured_unqualified"):
            raise ValueError("Canonical consecutive native window references required")
        path = root / expected
        if row.get("record_sha256") != digest(physical_file(path / "record.json")):
            raise ValueError("Capture window reference changed")
        load_window(path)
        paths.append(path)
    if {p.name for p in root.glob("window-*")} != {p.name for p in paths}:
        raise ValueError("Unlisted or missing native window directory")
    return paths


def read_sequence(path):
    import numpy as np
    from research_math.complete_unit_export import validate_sequence
    with np.load(physical_file(path), allow_pickle=False) as archive:
        expected = {"vertices", "faces", "frame_indices", "timesteps", "query_vertex_ids"}
        if set(archive.files) != expected:
            raise ValueError("Complete native sequence fields required")
        arrays = {key: archive[key].copy() for key in archive.files}
    validated = validate_sequence(arrays["vertices"], arrays["faces"])
    if arrays["vertices"].dtype != np.float32:
        raise ValueError("Official exported full float32 coordinates required")
    for key in ("frame_indices", "timesteps", "query_vertex_ids"):
        if not np.array_equal(arrays[key], validated[key]):
            raise ValueError("Original native identity/time mapping required: " + key)
    return arrays


def compare_sequences(unobserved_path, observed_path, *, atol, rtol):
    import numpy as np
    atol, rtol = tolerance(atol), tolerance(rtol)
    left, right = read_sequence(unobserved_path), read_sequence(observed_path)
    same_shape = left["vertices"].shape == right["vertices"].shape
    topology = np.array_equal(left["faces"], right["faces"])
    identities = all(np.array_equal(left[key], right[key])
                     for key in ("frame_indices", "timesteps", "query_vertex_ids"))
    coordinates = bool(same_shape and np.allclose(left["vertices"], right["vertices"],
                                                 atol=atol, rtol=rtol))
    return {
        "kind": "paired-native-generation-comparison", "version": 1,
        "matches": bool(topology and identities and coordinates),
        "topology_matches": bool(topology), "identity_matches": bool(identities),
        "coordinates_match": coordinates, "frames": 16, "atol": atol, "rtol": rtol,
        "max_abs_error": float(np.max(np.abs(left["vertices"].astype(np.float64)
                - right["vertices"].astype(np.float64)))) if same_shape else None,
        "unobserved_sha256": digest(unobserved_path), "observed_sha256": digest(observed_path),
        "coordinate_space": "official GLB coordinates after lossless deformation-axis restoration",
        "scientific_effect_qualification": False, "native_context_qualified": False,
    }


def verify_capture_export(capture_root, sequence_path):
    """Bind the full native capture to all 16 exported frames, including anchor."""
    import numpy as np
    from research_math.pipeline_decoder_observer import load_window
    sequence = read_sequence(sequence_path)
    windows = capture_windows(capture_root)
    # The frozen current-release unit is exactly 16 unsubsampled frames in one
    # Stage-II context. Reject other schedules instead of inventing a merge rule.
    if len(windows) != 1:
        raise ValueError("Exactly one full 16-frame native context required")
    meshes, _ = load_window(windows[0])
    if (not np.array_equal(meshes["source_timesteps"].numpy().reshape(-1), [0.])
            or not np.array_equal(meshes["target_timesteps"].numpy().reshape(-1), np.arange(1, 16))
            or not np.array_equal(meshes["faces"].numpy(), sequence["faces"])
            or not np.array_equal(meshes["anchor_vertices"].numpy().astype(np.float32),
                                  sequence["vertices"][0])
            or not np.array_equal(meshes["vertices"].numpy().astype(np.float32),
                                  sequence["vertices"][1:])):
        raise ValueError("Captured full topology/coordinates/times differ from official float32 export")
    return {"status": "capture_export_matches_unqualified", "frames": 16,
            "sequence_sha256": digest(sequence_path),
            "capture_index_sha256": digest(Path(capture_root) / "index.json"),
            "scientific_effect_qualification": False, "native_context_qualified": False}


def aggregate_comparisons(pair, replays):
    matches = bool(pair.get("matches") is True and replays and all(
        row.get("status") == "replayed_unqualified"
        and row.get("raw_matches") is True and row.get("mesh_matches") is True
        for row in replays))
    return {"all_comparisons_match": matches, "replay_windows": len(replays),
            "replay_qualified": False, "native_context_qualified": False,
            "native_scientific_qualification": False, "scientific_effect_qualification": False,
            "candidate_methods_tested": False, "dispatch_ready": False}


def _source_modules(source_root, expected):
    """Import inspected official files from the verified source checkout only."""
    source_root = Path(source_root).resolve()
    for relative, expected_hash in expected.items():
        if digest(source_root / relative) != expected_hash:
            raise ValueError("Upstream source changed before import: " + relative)
    os.chdir(source_root)
    sys.path[:0] = [str(source_root), str(source_root / "third_party/TripoSG")]
    from actionmesh.pipeline import ActionMeshPipeline
    from actionmesh.model.temporal_autoencoder import ActionMeshAutoencoder
    for cls, relative in ((ActionMeshPipeline, "actionmesh/pipeline.py"),
                          (ActionMeshAutoencoder, "actionmesh/model/temporal_autoencoder.py")):
        if Path(inspect.getfile(cls)).resolve() != source_root / relative:
            raise ValueError("Imported upstream class escaped verified source root")
    path = source_root / "inference/video_to_animated_mesh.py"
    spec = importlib.util.spec_from_file_location("_native_context_official_entry", path)
    if spec is None or spec.loader is None:
        raise ValueError("Official inference module unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return ActionMeshPipeline, ActionMeshAutoencoder, module.run_actionmesh


def _verify_instrument_code(identity):
    project = Path(__file__).resolve().parents[1]
    for relative, expected_hash in identity["instrument_code_sha256"].items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Canonical instrument source reference required")
        if digest(project / path) != expected_hash:
            raise ValueError("Instrument source changed across stages: " + relative)


def _stage(request_path, expected_hash):
    """Private subprocess entry; the parent harness owns its bounded lifetime."""
    request_path = physical_file(Path(request_path).resolve())
    if digest(request_path) != expected_hash:
        raise ValueError("Stage request hash mismatch")
    request = load(request_path)
    if request.get("kind") != "native-context-instrument-stage" or request.get("version") != 1:
        raise ValueError("Unsupported instrument stage")
    if any(os.environ.get(key) != "1" for key in
           ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")):
        raise ValueError("Offline native stage environment required")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != request["gpu_uuid"]:
        raise ValueError("Exact parent-harness GPU allocation required")
    output = Path(request["output"])
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"status": "failed", "kind": "native-context-stage-result", "version": 1,
              "stage": request["stage"], "request_sha256": expected_hash,
              "scientific_effect_qualification": False, "native_context_qualified": False}
    try:
        _verify_instrument_code(request["identity"])
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("One allocated CUDA device required; no CPU fallback")
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats()
        Pipeline, Autoencoder, run_actionmesh = _source_modules(
            request["source_root"], request["identity"]["upstream_source_sha256"])
        if request["stage"] in ("unobserved", "observed"):
            from research_math.complete_unit_runner import collect_native_sequence
            from research_math.pipeline_decoder_observer import PipelineDecoderObserver
            settings, run = generation_settings(request["identity"]["generation"],
                Path(request["source_root"]), Path(request["frames"]), output)
            settings["dtype"] = getattr(torch, settings["dtype"])
            pipeline = Pipeline(**settings)
            pipeline.to("cuda")
            if request["stage"] == "observed":
                with PipelineDecoderObserver(pipeline, Path(request["capture_root"]),
                        request["identity"], max_bytes=request["max_capture_bytes"], max_windows=1):
                    run_actionmesh(pipeline=pipeline, **run)
            else:
                run_actionmesh(pipeline=pipeline, **run)
            collect_native_sequence(output, request["uid"])
            if request["stage"] == "observed":
                report["capture_export"] = verify_capture_export(
                    Path(request["capture_root"]), output / "sequence.npz")
        elif request["stage"] == "replay":
            from research_math.pipeline_decoder_observer import replay_window
            capture_root = Path(request["capture_root"])
            if load(capture_root / "identity.json") != request["identity"]:
                raise ValueError("Capture identity differs from verified generation")
            pair = compare_sequences(Path(request["unobserved_sequence"]),
                Path(request["observed_sequence"]), atol=request["atol"], rtol=request["rtol"])
            report["paired_comparison"] = pair
            report["capture_export"] = verify_capture_export(
                capture_root, Path(request["observed_sequence"]))
            # Same Stage-II class, checkpoint, fp32 parameter loading and eval
            # mode as ActionMeshPipeline._load_temporal_vae. Replay restores
            # the actual captured autocast/inference modes around each call.
            model = Autoencoder.from_pretrained(
                str(Path(request["weights_root"]) / "ActionMesh/autoencoder"),
                local_files_only=True).eval().to("cuda")
            reports = []
            for window in capture_windows(capture_root):
                reports.append(replay_window(model, window, output / window.name,
                    identity=request["identity"], atol=request["atol"], rtol=request["rtol"],
                    source_time_query=request["source_time_query"] and pair["matches"]))
            report["windows"] = reports
            report.update(aggregate_comparisons(pair, reports))
        else:
            raise ValueError("Unknown instrument stage")
        report["status"] = "completed_unqualified"
        report["torch_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["torch_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    except BaseException as error:
        report.update(error=type(error).__name__ + ": " + str(error), traceback=traceback.format_exc())
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        save(output / "stage-result.json", report)
    return 0


def _identity(args, manifest, output):
    """Retain byte identities; these records are evidence, never authorization."""
    inputs = output / "inputs"
    inputs.mkdir()
    refs = {}
    names = list(INPUT_FILES)
    names += [name for name in ("pricing", "historical_manifest") if getattr(args, name) is not None]
    for name in names:
        source = Path(getattr(args, name))
        target = inputs / (name + ".json")
        target.write_bytes(source.read_bytes())
        refs[name] = {"path": target.relative_to(output).as_posix(), "sha256": digest(target)}
    project = Path(__file__).resolve().parents[1]
    code_paths = sorted((project / "research_math").glob("*.py"))
    code_paths += [project / name for name in
                   ("official_actionbench_adapter.py", "research_census_eval.py",
                    "deterministic_actionbench_entry.py")]
    identity = {
        "kind": "native-context-generation-identity", "version": 1,
        "scope": "paired engineering observer/replay; no candidate or scorer execution",
        "uid": manifest.get("selected_unit", manifest.get("calibration_unit", {}))["uid"],
        "generation": manifest["generation"], "verified_unit_manifest": manifest,
        "retained_input_refs": refs,
        "upstream_source_sha256": {relative: digest(args.source_root / relative) for relative in SOURCE_FILES},
        "instrument_code_sha256": {path.relative_to(project).as_posix(): digest(path) for path in code_paths},
        "python_executable": sys.executable, "python_version": sys.version,
        "gpu_uuid": args.gpu_uuid, "atol": args.atol, "rtol": args.rtol,
        "scientific_effect_qualification": False, "native_context_qualified": False,
    }
    save(output / "generation-identity.json", identity)
    return identity


def execute(args):
    from research_math.complete_unit_runner import run_stage, verify_prerequisites
    from research_math.control_scoring import DeviceSamples
    from research_math.unit_resources import HostSamples
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {"kind": "paired-native-context-instrument", "version": 1, "status": "failed",
              "scientific_effect_qualification": False, "native_context_qualified": False,
              "replay_qualified": False, "candidate_methods_tested": False,
              "dispatch_ready": False, "stages": {}, "gpu_uuid": args.gpu_uuid,
              "instrument_wall_seconds": args.instrument_wall_seconds,
              "prerequisite_unit_wall_seconds": args.wall_seconds,
              "atol": args.atol, "rtol": args.rtol,
              "source_time_query_requested": args.source_time_query}
    save(output / "result.json", result)
    monitor = host = None
    manifest = identity = None
    try:
        host = HostSamples(output, output / "host-samples.jsonl")
        host.start()
        manifest = verify_prerequisites(args, output)
        generation_settings(manifest["generation"], args.source_root, Path("unused"), Path("unused"))
        identity = _identity(args, manifest, output)
        uid = identity["uid"]
        monitor = DeviceSamples(args.gpu_uuid, output / "device-samples.jsonl")
        monitor.start()
        environment = stage_environment(args.source_root, Path(__file__).resolve().parents[1])
        result["offline_environment"] = {key: environment[key] for key in
            ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "PYTHONPATH")}
        requests = output / "requests"
        requests.mkdir()
        for stage in ("unobserved", "observed", "replay"):
            remaining = args.instrument_wall_seconds - (time.monotonic() - started)
            if remaining <= 1:
                raise TimeoutError("Instrument budget exhausted before " + stage)
            request = {
                "kind": "native-context-instrument-stage", "version": 1, "stage": stage,
                "source_root": str(args.source_root.resolve()),
                "weights_root": str(args.weights_root.resolve()),
                "frames": str((args.dataset_root / "data" / uid / "imgs").resolve()),
                "uid": uid, "gpu_uuid": args.gpu_uuid, "identity": identity,
                "output": str(output / stage), "capture_root": str(output / "capture"),
                "unobserved_sequence": str(output / "unobserved/sequence.npz"),
                "observed_sequence": str(output / "observed/sequence.npz"),
                "atol": args.atol, "rtol": args.rtol,
                "source_time_query": args.source_time_query,
                "max_capture_bytes": args.max_capture_bytes,
            }
            path = requests / (stage + ".json")
            save(path, request)
            command = [sys.executable, "-m", "research_math.native_context_runner",
                       "--stage-request", str(path), "--stage-request-sha256", digest(path)]
            result["stages"][stage] = run_stage(stage, command, args.source_root,
                                                output, remaining, environment)
            result["stages"][stage]["report_sha256"] = digest(output / stage / "stage-result.json")
            save(output / "result.json", result)
            if stage == "observed":
                pair = compare_sequences(output / "unobserved/sequence.npz",
                    output / "observed/sequence.npz", atol=args.atol, rtol=args.rtol)
                save(output / "paired-comparison.json", pair)
                result["paired_comparison"] = pair
        replay = load(output / "replay/stage-result.json")
        result["replay_reports"] = replay["windows"]
        result.update(aggregate_comparisons(result["paired_comparison"], replay["windows"]))
        result["status"] = ("completed_unqualified" if result["all_comparisons_match"]
                            else "comparison_mismatch")
    except BaseException as error:
        result.update(status="failed", error=type(error).__name__ + ": " + str(error),
                      traceback=traceback.format_exc())
    finally:
        # Recheck source, all four weight snapshots, native input/prerequisite
        # manifests even after a failed generation. Never overwrite the first
        # revalidation or use a new configuration as an automatic fallback.
        if manifest is not None:
            try:
                final_dir = output / "final-integrity"
                final_dir.mkdir()
                checked = verify_prerequisites(args, final_dir)
                if checked != manifest:
                    raise ValueError("Prerequisite manifest changed across instrument stages")
                if identity is not None:
                    for name, ref in identity["retained_input_refs"].items():
                        if digest(getattr(args, name)) != ref["sha256"]:
                            raise ValueError("Prerequisite input bytes changed: " + name)
                    for relative, expected_hash in identity["upstream_source_sha256"].items():
                        if digest(args.source_root / relative) != expected_hash:
                            raise ValueError("Imported source changed across instrument stages: " + relative)
                    _verify_instrument_code(identity)
                result["final_integrity"] = "matched"
            except BaseException as error:
                result.update(status="failed", final_integrity="failed",
                              final_integrity_error=type(error).__name__ + ": " + str(error))
        if monitor is not None:
            monitor.close()
            result["device_memory"] = {"sample_interval_seconds": 1, "samples": len(monitor.memory),
                "observed_peak_mib": max(monitor.memory, default=None),
                "exact_peak": False, "errors": monitor.errors}
            if monitor.errors:
                result["status"] = "failed"
        if host is not None:
            host.close()
            # Keep error samples too; HostSamples.summary assumes all rows
            # have successful measurement fields, which errors do not have.
            result["host_resources"] = {"sample_interval_seconds": 1, "samples": len(host.rows),
                "exact_peak": False, "errors": host.errors}
            for key, reducer in (("process_tree_rss_bytes", max), ("output_bytes", max),
                                 ("free_disk_bytes", min)):
                result["host_resources"][key] = reducer(
                    (row[key] for row in host.rows if key in row), default=None)
            if host.errors:
                result["status"] = "failed"
        result["outputs"] = [{"path": path.relative_to(output).as_posix(),
                              "bytes": path.stat().st_size, "sha256": digest(path)}
                             for path in sorted(output.rglob("*"))
                             if path.is_file() and path != output / "result.json"]
        # Collection hashing is part of this instrument's measured boundary.
        result["elapsed_seconds"] = time.monotonic() - started
        if result["elapsed_seconds"] > args.instrument_wall_seconds:
            result.update(status="failed", budget_exceeded=True)
        save(output / "result.json", result)
    return 0 if result["status"] == "completed_unqualified" else 1


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (*PREREQUISITE_PATHS, "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    for name in ("root", "pricing", "historical_root", "historical_manifest"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path)
    parser.add_argument("--uid")
    parser.add_argument("--window-id")
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--wall-seconds", type=int, default=27000,
        help="Unchanged prerequisite one-unit wall budget; in Full128 mode equals frozen queue price")
    parser.add_argument("--instrument-wall-seconds", type=int, required=True,
        help="Separately admitted total budget for both generations, replay and integrity collection")
    parser.add_argument("--atol", type=float, required=True)
    parser.add_argument("--rtol", type=float, required=True)
    parser.add_argument("--source-time-query", action="store_true")
    parser.add_argument("--max-capture-bytes", type=int, default=64 * 1024 * 1024)
    args = parser.parse_args(argv)
    try:
        args.atol, args.rtol = tolerance(args.atol), tolerance(args.rtol)
        if not 1 <= args.wall_seconds <= 27000 or not 1 <= args.instrument_wall_seconds <= 27000:
            raise ValueError("Both bounded budgets must be in 1..27000 seconds")
        if args.max_capture_bytes < 1:
            raise ValueError("Positive native capture byte bound required")
        if not args.gpu_uuid.startswith("GPU-") or any(c in args.gpu_uuid for c in "\n\r, "):
            raise ValueError("One physical harness-allocated GPU UUID required")
    except ValueError as error:
        parser.error(str(error))
    return args


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--stage-request" in argv:
        parser = argparse.ArgumentParser(description="Internal harness-owned instrument stage")
        parser.add_argument("--stage-request", type=Path, required=True)
        parser.add_argument("--stage-request-sha256", required=True)
        args = parser.parse_args(argv)
        return _stage(args.stage_request, args.stage_request_sha256)
    return execute(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
