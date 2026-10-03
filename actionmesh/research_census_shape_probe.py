"""Native Stage-I isosurface versus Stage-II mesh diagnostic, no inference edit.

Decode selected saved latents serially using the exact native TripoSG flash
extractor. A new output directory is mandatory. No downloads, adaptive resolution
fallback, ICP, latent normalization, weight updates, or correspondence metric.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback

from research_three_ideas import Resources, digest, json_write, setup

SAMPLES = 50000
BOUNDS = (-1.005, -1.005, -1.005, 1.005, 1.005, 1.005)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def validate_mesh(vertices, faces):
    import numpy as np
    vertices, faces = np.asarray(vertices), np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError('Mesh vertices must be nonempty [N,3]')
    if not np.isfinite(vertices).all():
        raise ValueError('Mesh vertices are nonfinite')
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError('Mesh faces must be nonempty [F,3]')
    if not np.issubdtype(faces.dtype, np.integer):
        raise ValueError('Mesh faces must be integer indices')
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError('Mesh faces reference out-of-range vertices')


def surface_sample(mesh, count, seed):
    """Trimesh area-weighted sampling with an explicit private RNG seed."""
    import numpy as np
    import trimesh
    if 'seed' not in inspect.signature(trimesh.sample.sample_surface).parameters:
        raise RuntimeError('This Trimesh lacks sample_surface(seed=...); no global-RNG fallback permitted')
    if not np.isfinite(mesh.area_faces).all() or float(mesh.area) <= 0:
        raise ValueError('Cannot sample a mesh with invalid or zero surface area')
    points, face_indices = trimesh.sample.sample_surface(mesh, count, seed=int(seed))
    points = np.asarray(points, dtype=np.float32)
    face_indices = np.asarray(face_indices, dtype=np.int64)
    if points.shape != (count, 3) or face_indices.shape != (count,) or not np.isfinite(points).all():
        raise ValueError('Surface sampler returned invalid samples')
    return points, face_indices


def symmetric_surface_distance(first, second):
    """0.5*(mean nearest distance A->B + mean nearest distance B->A)."""
    import numpy as np
    from scipy.spatial import cKDTree
    first, second = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    for points in (first, second):
        if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
            raise ValueError('Distance requires finite nonempty [N,3] point arrays')
    # One CPU thread avoids unexpected oversubscription beside GPU jobs.
    ab = cKDTree(second).query(first, k=1, workers=1)[0]
    ba = cKDTree(first).query(second, k=1, workers=1)[0]
    return {'symmetric_mean_unsquared': float(0.5 * (ab.mean() + ba.mean())),
            'stageI_to_stageII_mean_unsquared': float(ab.mean()),
            'stageII_to_stageI_mean_unsquared': float(ba.mean()),
            'stageI_to_stageII_p95_unsquared': float(np.quantile(ab, .95)),
            'stageII_to_stageI_p95_unsquared': float(np.quantile(ba, .95)),
            'units': 'Native canonical coordinate units; no scale fitting or ICP',
            'definition': '0.5 * (mean nearest Euclidean distance A->B + mean nearest Euclidean distance B->A)',
            'scope': 'Monte Carlo surface-set discrepancy, not official ActionBench CD and not material-motion CD-M'}


def cpu_self_test():
    import numpy as np
    import trimesh
    grid = np.asarray([(x, y, 0.) for x in (0., .5, 1.) for y in (0., .5, 1.)])
    same = symmetric_surface_distance(grid, grid.copy())
    translated = symmetric_surface_distance(grid, grid + np.array([0., 0., .25]))
    assert same['symmetric_mean_unsquared'] == 0.
    assert abs(translated['symmetric_mean_unsquared'] - .25) < 1e-12
    assert abs(symmetric_surface_distance(grid + [0, 0, .25], grid)['symmetric_mean_unsquared'] - .25) < 1e-12
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    validate_mesh(vertices, faces)
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    rng_before = np.random.get_state()
    first, ids = surface_sample(mesh, 4096, 101)
    second, ids2 = surface_sample(mesh, 4096, 101)
    rng_after = np.random.get_state()
    assert all(np.array_equal(a, b) for a, b in zip(rng_before, rng_after)), 'Global NumPy RNG changed'
    assert np.array_equal(first, second) and np.array_equal(ids, ids2)
    assert symmetric_surface_distance(first, second)['symmetric_mean_unsquared'] == 0.
    shifted_mesh = trimesh.Trimesh(vertices + [0, 0, .25], faces, process=False)
    shifted, _ = surface_sample(shifted_mesh, 4096, 101)
    assert abs(symmetric_surface_distance(first, shifted)['symmetric_mean_unsquared'] - .25) < 1e-7
    different, _ = surface_sample(mesh, 4096, 102)
    assert not np.array_equal(first, different)
    for frame in (0, 8, 15):
        points, _ = surface_sample(mesh, 2048, 101 + frame)
        assert symmetric_surface_distance(points, points.copy())['symmetric_mean_unsquared'] == 0.
    rejected = 0
    for bad_vertices, bad_faces in [(vertices * np.nan, faces), (vertices, faces + 9)]:
        try:
            validate_mesh(bad_vertices, bad_faces)
        except ValueError:
            rejected += 1
    assert rejected == 2
    return {'status': 'passed', 'exact_grid_same_surface_zero': True,
            'parallel_plane_normal_translation_025': True,
            'trimesh_same_seed_samples_bitwise_equal': True,
            'global_numpy_rng_preserved': True, 'same_sequence_selected_frames_zero': True,
            'invalid_geometry_rejected': True, 'torch_or_cuda_used': False}


def load_case(case_dir, frames):
    import numpy as np
    with np.load(case_dir / 'denoised.npz', allow_pickle=False) as cached:
        latents = cached['latents'].copy()
        latent_times = cached['timesteps'].copy()
    with np.load(case_dir / 'sequence.npz', allow_pickle=False) as cached:
        vertices, faces = cached['vertices'].copy(), cached['faces'].copy()
        mesh_times = cached['timesteps'].copy()
    if latents.shape != (16, 2048, 64) or not np.isfinite(latents).all():
        raise ValueError('Expected 16 finite [2048,64] native shape latents')
    if vertices.ndim != 3 or vertices.shape[0] != 16 or vertices.shape[2] != 3:
        raise ValueError('Expected 16 native Stage-II mesh frames')
    if latent_times.shape != (16,) or not np.array_equal(latent_times, mesh_times):
        raise ValueError('Latent and Stage-II mesh timesteps differ')
    if not np.array_equal(latent_times, np.arange(16)):
        raise ValueError('Expected original native frame indices 0..15')
    for frame in frames:
        validate_mesh(vertices[frame], faces)
    report = json.loads((case_dir / 'report.json').read_text())
    if report.get('status') != 'completed':
        raise ValueError('Source native inference report must be completed')
    for filename in ('denoised.npz', 'sequence.npz'):
        expected = report.get('sha256', {}).get(filename)
        if not expected or expected != digest(case_dir / filename):
            raise ValueError(f'Missing or mismatched source hash for {filename}')
    return latents, vertices, faces, report


def source_provenance(repo):
    files = ['actionmesh/external/triposg.py', 'actionmesh/model/temporal_autoencoder.py',
             'actionmesh/preprocessing/mesh_processor.py',
             'third_party/TripoSG/triposg/inference_utils.py',
             'third_party/TripoSG/triposg/models/autoencoders/autoencoder_kl_triposg.py',
             'third_party/TripoSG/triposg/pipelines/pipeline_triposg.py']
    return {name: digest(repo / name) for name in files}


def run_probe(args, repo, torch, started, state):
    import numpy as np
    import scipy
    import trimesh
    sys.path.insert(0, str(repo / 'third_party/TripoSG'))
    from triposg.models.autoencoders import TripoSGVAEModel
    from triposg.inference_utils import flash_extract_geometry

    def check(stage):
        elapsed = time.monotonic() - started
        state['stage'] = stage
        state['elapsed_seconds'] = elapsed
        json_write(args.output / 'progress.json', state)
        print(f'SHAPE_PROBE stage={stage} elapsed={elapsed:.2f}s', flush=True)
        if elapsed > args.max_seconds:
            raise TimeoutError(f'{args.max_seconds}s stage-check budget exceeded at {stage}; external timeout needed to interrupt an in-flight CUDA call')

    check('loading_source_caches')
    latents, stage2_vertices, stage2_faces, source_report = load_case(args.case_dir, args.frames)
    np.save(args.output / 'selected-source-latents.npy', latents[args.frames], allow_pickle=False)
    check('loading_standalone_vae')
    weights = repo / 'pretrained_weights/TripoSG'
    if not (weights / 'vae').is_dir():
        raise FileNotFoundError(f'Local VAE weights absent: {weights / "vae"}')
    vae = TripoSGVAEModel.from_pretrained(str(weights), subfolder='vae',
        torch_dtype=torch.float16, local_files_only=True).eval().to(device='cuda', dtype=torch.float16)
    parameter_count = 0
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
        parameter_count += parameter.numel()
    if not parameter_count or any(p.requires_grad for p in vae.parameters()):
        raise RuntimeError('Could not verify all VAE weights frozen')
    vae.set_flash_decoder()
    if vae.training or vae.decoder.training:
        raise RuntimeError('VAE must be in evaluation mode to disable geometry gradients')
    floating_buffers = {name: str(buffer.dtype) for name, buffer in vae.named_buffers() if buffer.is_floating_point()}
    if any(buffer.is_floating_point() and buffer.dtype != torch.float16 for buffer in vae.buffers()):
        raise RuntimeError('VAE floating buffers must match native recursive FP16 conversion')
    json_write(args.output / 'provenance.json', {
        'source_case_dir': str(args.case_dir), 'source_uid': source_report.get('uid'),
        'source_seed': source_report.get('seed'),
        'source_hashes': {name: digest(args.case_dir / name) for name in ('denoised.npz', 'sequence.npz', 'report.json')},
        'code_sha256': source_provenance(repo), 'checkpoint_directory': str(weights / 'vae'),
        'checkpoint_config_sha256': digest(weights / 'vae/config.json'),
        'versions': {'torch': torch.__version__, 'numpy': np.__version__, 'scipy': scipy.__version__, 'trimesh': trimesh.__version__},
        'frozen_parameter_count': parameter_count, 'source_latent_dtype': str(latents.dtype),
        'decode_latent_dtype': 'float16', 'latent_rescaling': False,
        'decode_autocast': 'cuda float16', 'floating_buffer_dtypes': floating_buffers,
        'vae_training': vae.training, 'decoder_training': vae.decoder.training,
        'bounds': list(BOUNDS), 'octree_depth': 9, 'num_chunks': 10000,
        'memory_fraction': args.memory_fraction,
        'memory_cap_scope': 'PyTorch caching allocator fraction of physical GPU memory; not a hard cap on all CUDA/driver allocations',
        'timeout_scope': 'Checks between stages; cannot interrupt a single CUDA call. Caller must also impose an external wall timeout.',
    })
    check('vae_ready')
    for frame in args.frames:
        check(f'frame_{frame:02d}_decode_start')
        frame_started = time.monotonic()
        with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.float16):
            z = torch.from_numpy(latents[frame:frame + 1]).to(device='cuda', dtype=torch.float16)
            if not torch.isfinite(z).all():
                raise ValueError(f'Latent cast became nonfinite at frame {frame}')
            torch.cuda.synchronize()
            decoded_vertices, decoded_faces = flash_extract_geometry(z, vae, bounds=BOUNDS,
                octree_depth=9, num_chunks=10000)[0]
            torch.cuda.synchronize()
            del z
        if decoded_vertices is None or decoded_faces is None:
            raise RuntimeError(f'Native flash extraction returned no surface at frame {frame}; no lower-resolution retry')
        validate_mesh(decoded_vertices, decoded_faces)
        decoded_vertices = np.asarray(decoded_vertices, dtype=np.float32)
        decoded_faces = np.asarray(decoded_faces, dtype=np.int64)
        frame_dir = args.output / f'frame_{frame:02d}'
        frame_dir.mkdir()
        np.savez_compressed(frame_dir / 'stageI-raw-mesh.npz', vertices=decoded_vertices,
                            faces=decoded_faces, frame=np.asarray(frame, dtype=np.int64))
        mesh = trimesh.Trimesh(decoded_vertices, decoded_faces, process=False)
        mesh.export(frame_dir / 'stageI-raw.glb')
        decode_seconds = time.monotonic() - frame_started
        check(f'frame_{frame:02d}_decoded_saved')
        stage2_mesh = trimesh.Trimesh(stage2_vertices[frame], stage2_faces, process=False)
        seed = args.seed + frame
        a, a_faces = surface_sample(mesh, SAMPLES, seed)
        b, b_faces = surface_sample(stage2_mesh, SAMPLES, seed)
        np.savez_compressed(frame_dir / 'surface-samples.npz', stageI_xyz=a, stageII_xyz=b,
            stageI_face_indices=a_faces, stageII_face_indices=b_faces, seed=np.asarray(seed, dtype=np.int64))
        check(f'frame_{frame:02d}_samples_saved')
        metrics = symmetric_surface_distance(a, b)
        row = {'frame': frame, 'decode_and_mesh_save_seconds': decode_seconds,
               'elapsed_frame_seconds': time.monotonic() - frame_started,
               'stageI_vertices': len(decoded_vertices), 'stageI_faces': len(decoded_faces),
               'stageII_vertices': len(stage2_vertices[frame]), 'stageII_faces': len(stage2_faces),
               'stageI_bounds': mesh.bounds.tolist(), 'stageII_bounds': stage2_mesh.bounds.tolist(),
               'samples_per_mesh': SAMPLES, 'sample_seed': seed,
               'finite': True, 'metric': metrics,
               'frame0_calibration': frame == 0,
               'sha256': {name: digest(frame_dir / name) for name in ('stageI-raw-mesh.npz', 'stageI-raw.glb', 'surface-samples.npz')}}
        json_write(frame_dir / 'measurement.json', row)
        state['measurements'].append(row)
        del mesh, stage2_mesh, decoded_vertices, decoded_faces, a, b
        torch.cuda.empty_cache()
        check(f'frame_{frame:02d}_completed')
    state.update(status='completed', stage='completed', uid=source_report.get('uid'), seed=source_report.get('seed'),
        inference_edit=False, training=False, backprop=False, independent_topology=True,
        interpretation='Stage-I/Stage-II surface discrepancy only. Frame0 differs partly because native anchor was cleaned, decimated and floater-filtered. StageII clamps xyz to [-1,1], extraction bounds are +/-1.005. Independent mesh indices do not measure physical correspondence. No CD-M claim or automatic attribution of material-motion failure.')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true', help='Real CPU geometry/sampling tests; needs numpy/scipy/trimesh only')
    parser.add_argument('--root', type=Path, default=Path('/root/rivermind-data/actionmesh-repro'))
    parser.add_argument('--case-dir', type=Path)
    parser.add_argument('--frames', nargs='+', type=int, default=[0, 8, 15])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--memory-fraction', type=float, default=.30)
    parser.add_argument('--max-seconds', type=float, default=300.)
    parser.add_argument('--seed', type=int, default=20261002)
    parser.add_argument('--gpu-index', default='0')
    args = parser.parse_args(argv)
    if args.self_test:
        return args
    if args.case_dir is None or args.output is None:
        parser.error('--case-dir and --output are required unless --self-test')
    if not 0 < args.memory_fraction <= 1 or not 0 < args.max_seconds < float('inf'):
        parser.error('memory-fraction must be in (0,1] and max-seconds finite and positive')
    if not args.frames or len(args.frames) != len(set(args.frames)) or any(t < 0 or t >= 16 for t in args.frames):
        parser.error('frames must be unique native indices from 0 through 15')
    if args.seed < 0:
        parser.error('seed must be nonnegative')
    for name in ('root', 'case_dir', 'output'):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.self_test:
        print(json.dumps(cpu_self_test(), indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = {'status': 'running', 'started_utc': utc_now(), 'requested_frames': args.frames,
             'measurements': [], 'stage': 'setup'}
    json_write(args.output / 'command.json', {
        'argv': sys.argv if argv is None else [str(Path(__file__)), *argv],
        'script_sha256': digest(Path(__file__)), 'started_utc': state['started_utc'],
        'offline': True, 'case_dir': str(args.case_dir), 'output': str(args.output),
        'memory_fraction': args.memory_fraction, 'max_seconds': args.max_seconds,
        'frames': args.frames, 'no_resolution_or_frame_fallback': True,
    })
    json_write(args.output / 'progress.json', state)
    monitor = None
    try:
        repo, torch = setup(args.root)
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device=torch.cuda.current_device())
        monitor = Resources(args.output, torch, args.gpu_index)
        monitor.start()
        run_probe(args, repo, torch, started, state)
    except Exception as exc:
        state.update(status='failed', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        state.update(ended_utc=utc_now(), elapsed_seconds=time.monotonic() - started)
        if monitor:
            try:
                state['resources'] = monitor.finish()
            except Exception as exc:
                state['resource_monitor_error'] = str(exc)
        json_write(args.output / 'report.json', state)
        json_write(args.output / 'progress.json', state)


if __name__ == '__main__':
    main()
