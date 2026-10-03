"""Bounded frozen Stage-II time-coordinate diagnostic, NOT reverse diffusion.

Decode the same cached Stage-I latents and source queries in three ways:
forward reproduction, row permutation preserving timestamps, then t -> 15-t.
No input video generation, denoising, training, GT, averaging, or guidance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np
from research_three_ideas import Resources, digest, json_write

ARMS = ('forward', 'row_permutation', 'time_reversal')
FROZEN_UIDS = ('000-037_1358c424008a43cbaa35eba5e58551ac', '000-043_061697e330d44524bd11f8cf95772e2d')
RESERVATION_MIB = 6144
ALLOCATOR_CAP_MIB = 5120
PERMUTATION_RMS_OVER_D_LIMIT = 1e-4
PERMUTATION_MAX_COORD_OVER_D_LIMIT = 1e-3


def bitwise_equal(first, second):
    return first.dtype == second.dtype and first.shape == second.shape and first.tobytes() == second.tobytes()


def array_digest(array):
    value = hashlib.sha256()
    value.update(str(array.dtype).encode())
    value.update(json.dumps(list(array.shape)).encode())
    value.update(np.ascontiguousarray(array).tobytes())
    return value.hexdigest()


def arm_mapping(arm):
    physical_rows = np.arange(16, dtype=np.int64)
    if arm == 'row_permutation':
        physical_rows = physical_rows[::-1].copy()
    elif arm not in ARMS:
        raise ValueError('Unknown arm: ' + arm)
    physical_to_clock = np.arange(16, dtype=np.float32)
    if arm == 'time_reversal':
        physical_to_clock = 15 - physical_to_clock
    target_physical = np.arange(1, 16, dtype=np.int64)
    return {'latent_row_physical_ids': physical_rows,
            'latent_row_clock_times': physical_to_clock[physical_rows],
            'source_physical_frame': 0, 'source_row_index': int(np.flatnonzero(physical_rows == 0)[0]),
            'source_clock_time': float(physical_to_clock[0]),
            'target_physical_frames': target_physical,
            'target_clock_times': physical_to_clock[target_physical],
            'output_physical_frames': np.arange(16, dtype=np.int64),
            'output_clock_times': physical_to_clock}


def load_source(case_dir):
    report = json.loads((case_dir / 'report.json').read_text())
    if report.get('status') != 'completed' or report.get('frames') != 16:
        raise ValueError('Require completed native 16-frame source')
    if report.get('uid') not in FROZEN_UIDS or report.get('seed') != 42:
        raise ValueError('This diagnostic is frozen to the first two census assets, seed42')
    hashes = {'report.json': digest(case_dir / 'report.json')}
    for name in ('prepared.npz', 'denoised.npz', 'sequence.npz'):
        hashes[name] = digest(case_dir / name)
        if hashes[name] != report.get('sha256', {}).get(name):
            raise ValueError('Native source hash mismatch: ' + name)
    with np.load(case_dir / 'prepared.npz', allow_pickle=False) as saved:
        prepared = {key: saved[key].copy() for key in ('timesteps', 'anchor_latent', 'anchor_timesteps',
                    'anchor_vertices', 'anchor_faces', 'anchor_query_features', 'query_vertex_ids', 'seed')}
    with np.load(case_dir / 'denoised.npz', allow_pickle=False) as saved:
        latents, times, latent_seed = saved['latents'].copy(), saved['timesteps'].copy(), int(saved['seed'])
    with np.load(case_dir / 'sequence.npz', allow_pickle=False) as saved:
        vertices, faces = saved['vertices'].copy(), saved['faces'].copy()
        frame_ids, mesh_times, query_ids = saved['frame_indices'].copy(), saved['timesteps'].copy(), saved['query_vertex_ids'].copy()
    if latents.shape != (16, 2048, 64) or not np.isfinite(latents).all():
        raise ValueError('Expected finite native [16,2048,64] latent sequence')
    if vertices.ndim != 3 or vertices.shape[0] != 16 or vertices.shape[2] != 3 or not np.isfinite(vertices).all():
        raise ValueError('Invalid native 16-frame mesh sequence')
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or not np.issubdtype(faces.dtype, np.integer):
        raise ValueError('Invalid fixed native triangle topology')
    if faces.min() < 0 or faces.max() >= vertices.shape[1]:
        raise ValueError('Native face index out of bounds')
    for values in (times, mesh_times, frame_ids, prepared['timesteps']):
        if not np.array_equal(values, np.arange(16)):
            raise ValueError('Require every physical frame 0..15 exactly once')
    if latent_seed != 42 or int(prepared['seed']) != 42 or not np.array_equal(prepared['anchor_timesteps'], [0]):
        raise ValueError('Cached source seed/anchor time mismatch')
    if not bitwise_equal(prepared['anchor_vertices'], vertices[0]) or not bitwise_equal(prepared['anchor_faces'], faces):
        raise ValueError('Cached anchor differs from native mesh identity')
    # Native LatentBank stacks a FP16 Stage0 anchor with FP32 denoised rows,
    # promoting the anchor to FP32. Require exact values, not identical dtypes.
    if (not np.array_equal(prepared['anchor_latent'][0], latents[0]) or
            not bitwise_equal(prepared['anchor_latent'][0].astype(latents.dtype), latents[0])):
        raise ValueError('Native Stage-I changed the held source latent')
    if not bitwise_equal(query_ids, prepared['query_vertex_ids']) or not np.array_equal(query_ids, np.arange(vertices.shape[1])):
        raise ValueError('Cached source query vertex IDs differ')
    query = prepared['anchor_query_features']
    if query.shape != (vertices.shape[1], 6) or query.dtype != np.float32 or not np.isfinite(query).all():
        raise ValueError('Expected finite original FP32 XYZ+normal source queries')
    if not bitwise_equal(query[:, :3], vertices[0]):
        raise ValueError('Cached query positions differ from held anchor vertices')
    return report, hashes, latents, vertices, faces, prepared


def discrepancy(vertices, reference, diagonal):
    delta = vertices.astype(np.float64) - reference.astype(np.float64)
    moving = delta[1:]
    return {'bitwise_equal': bitwise_equal(vertices, reference),
            'max_abs_coordinate': float(np.max(np.abs(delta))),
            'moving_rms_xyz_over_anchor_diagonal': float(np.sqrt(np.mean(moving ** 2)) / diagonal),
            'moving_max_abs_coordinate_over_anchor_diagonal': float(np.max(np.abs(moving)) / diagonal),
            'per_frame_mean_vertex_distance_over_anchor_diagonal': (np.linalg.norm(delta, axis=-1).mean(axis=1) / diagonal).tolist(),
            'interpretation': 'Matched-query prediction disagreement; not ground-truth error.'}


def run(args, state, started):
    import torch
    import trimesh

    def check(stage):
        state['stage'] = stage
        state['elapsed_seconds'] = time.monotonic() - started
        json_write(args.output / 'progress.json', state)
        print('TIME_DIRECTION', stage, round(state['elapsed_seconds'], 2), flush=True)
        if state['elapsed_seconds'] >= args.max_seconds:
            raise TimeoutError('Diagnostic asset budget exceeded; no reduced-frame fallback')

    check('source_validation')
    source, hashes, latents, native, faces, prepared = load_source(args.case_dir)
    diagonal = float(np.linalg.norm(np.ptp(native[0].astype(np.float64), axis=0)))
    if not np.isfinite(diagonal) or diagonal <= 0:
        raise ValueError('Invalid source anchor diagonal')
    repo = args.root / 'repo'
    code = json.loads((args.case_dir / 'code-provenance.json').read_text())
    source_code = {}
    for name in ('actionmesh/pipeline.py', 'actionmesh/model/temporal_autoencoder.py',
                 'actionmesh/configs/actionmesh.yaml', 'actionmesh/configs/actionmesh_lowram.yaml'):
        source_code[name] = digest(repo / name)
        if source_code[name] != code.get('sha256', {}).get(name):
            raise ValueError('Native source implementation changed: ' + name)
    for name in ('actionmesh/model/utils/embeddings.py', 'actionmesh/model/utils/rotary_embedding.py',
                 'actionmesh/model/utils/block.py', 'actionmesh/preprocessing/mesh_processor.py'):
        source_code[name] = digest(repo / name)
    hashes['code-provenance.json'] = digest(args.case_dir / 'code-provenance.json')
    checkpoint = repo / 'pretrained_weights/ActionMesh/autoencoder'
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    checkpoint_hashes = {str(path.relative_to(checkpoint)): digest(path)
                         for path in sorted(checkpoint.rglob('*')) if path.is_file() and '.cache' not in path.parts}
    if not checkpoint_hashes:
        raise ValueError('No local temporal autoencoder checkpoint files')
    state.update(uid=source['uid'], seed=42, source_hashes=hashes, source_case_dir=str(args.case_dir),
                 checkpoint_directory=str(checkpoint), checkpoint_sha256=checkpoint_hashes,
                 code_sha256=source_code, anchor_diagonal=diagonal,
                 source_latents_array_sha256=array_digest(latents), cached_query_sha256=array_digest(prepared['anchor_query_features']),
                 source_anchor_latent_dtype=str(prepared['anchor_latent'].dtype), source_stacked_latent_dtype=str(latents.dtype),
                 query_vertex_ids_sha256=array_digest(prepared['query_vertex_ids']),
                 checkpoint_identity_note='Current local checkpoint bytes are recorded. Native per-case report did not hash checkpoints; exact forward reproduction is required.')
    json_write(args.output / 'source-provenance.json', {k: state[k] for k in ('uid', 'seed', 'source_hashes', 'source_case_dir', 'checkpoint_directory', 'checkpoint_sha256', 'code_sha256')})
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    sys.path.insert(0, str(repo))
    from actionmesh.model.temporal_autoencoder import ActionMeshAutoencoder
    from actionmesh.preprocessing.mesh_processor import get_mesh_features
    from actionmesh.model.utils.embeddings import get_scaling, apply_scaling
    # Audit reconstructed features, but always decode the original cached query tensor.
    reconstructed = get_mesh_features(trimesh.Trimesh(native[0], faces, process=False), with_normals=True).numpy()
    state['recomputed_query_bitwise_matches_cache'] = bitwise_equal(reconstructed, prepared['anchor_query_features'])
    state['query_recompute_max_abs_difference'] = float(np.max(np.abs(reconstructed - prepared['anchor_query_features'])))
    device = torch.device(args.device)
    if device.type != 'cuda' or device.index is None or not torch.cuda.is_available():
        raise ValueError('Explicit available CUDA device required for later authorized execution')
    torch.cuda.set_device(device)
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1., ALLOCATOR_CAP_MIB * 1024 ** 2 / total), device=device)
    torch.set_num_threads(2)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    check('loading_frozen_temporal_decoder_only')
    decoder = ActionMeshAutoencoder.from_pretrained(str(checkpoint), local_files_only=True).eval().to(device)
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    if decoder.training or any(parameter.requires_grad for parameter in decoder.parameters()):
        raise ValueError('Decoder must remain frozen in evaluation mode')
    if decoder.prediction_mode != 'direct' or decoder.temporal_context_size != 16:
        raise ValueError('Expected native direct-output, 16-frame temporal decoder')
    if any(parameter.is_floating_point() and parameter.dtype != torch.float32 for parameter in decoder.parameters()):
        raise ValueError('Native temporal checkpoint parameter storage must remain FP32')
    state['decoder_parameter_count'] = sum(parameter.numel() for parameter in decoder.parameters())
    query = torch.from_numpy(prepared['anchor_query_features']).unsqueeze(0).to(device)
    decoded = {}
    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.float16):
        for arm in ARMS:
            row = state['arms'][arm]
            folder = args.output / 'variants' / arm
            check(arm + '_start')
            mapping = arm_mapping(arm)
            physical = mapping['latent_row_physical_ids']
            z_array = np.ascontiguousarray(latents[physical])
            # Confirm row rearrangement only; no rescaling or new latent values.
            if not bitwise_equal(z_array[np.argsort(physical)], latents):
                raise ValueError('Arm changed physical latent identity')
            z = torch.from_numpy(z_array).unsqueeze(0).to(device)
            times = torch.from_numpy(mapping['latent_row_clock_times']).unsqueeze(0).to(device)
            source_time = torch.tensor([mapping['source_clock_time']], dtype=times.dtype, device=device)
            targets = torch.from_numpy(mapping['target_clock_times']).unsqueeze(0).to(device)
            minimum, span = get_scaling(times)
            source_alpha = apply_scaling(source_time, minimum, span)
            target_alphas = apply_scaling(targets, minimum, span)
            row.update(status='running', uid=source['uid'], seed=42, frames=16,
                       mapping={k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in mapping.items()},
                       source_alpha=source_alpha.cpu().tolist(), target_alphas=target_alphas.cpu().tolist(),
                       latent_input_sha256=array_digest(z_array), latent_dtype=str(z.dtype), query_dtype=str(query.dtype))
            json_write(folder / 'report.json', row)
            arm_started = time.monotonic()
            def callback(step, count):
                check(f'{arm}_target_{step}_of_{count}')
            displacement = decoder(latent=z, framestep=times, source_alpha=source_alpha,
                                   target_alphas=target_alphas, query=query, step_callback=callback)
            # Native direct mode ignores the vertex argument; pass documented XYZ shape.
            result = decoder.apply_displacement(vertex=query[..., :3], displacement=displacement)
            torch.cuda.synchronize(device)
            values = result[0].detach().cpu().numpy().astype(native.dtype)
            if values.shape != native[1:].shape or not np.isfinite(values).all():
                raise ValueError('Expected 15 finite target outputs in physical order')
            vertices = np.concatenate((native[0:1].copy(), values), axis=0)
            if not bitwise_equal(vertices[0], native[0]) or len(np.unique(mapping['output_physical_frames'])) != 16:
                raise ValueError('Missing physical frame or changed source anchor')
            sequence = folder / 'sequence.npz'
            np.savez_compressed(sequence, vertices=vertices, faces=faces,
                frame_indices=mapping['output_physical_frames'], timesteps=np.arange(16, dtype=np.float32),
                decoder_clock_times=mapping['output_clock_times'], query_vertex_ids=prepared['query_vertex_ids'])
            row.update(status='completed', elapsed_seconds=time.monotonic() - arm_started,
                       sequence_sha256=digest(sequence), sha256={'sequence.npz': digest(sequence)},
                       anchor_bitwise_unchanged=True, faces_bitwise_unchanged=True,
                       all_16_physical_frames_once=True, source_query_ids_unchanged=True,
                       discrepancy_vs_native=discrepancy(vertices, native, diagonal),
                       training=False, stageI_regenerated=False, stageII_only=True)
            decoded[arm] = vertices
            if arm == 'forward':
                row['control_pass'] = bitwise_equal(vertices, native)
                if not row['control_pass']:
                    row['status'] = 'control_failed'
                json_write(folder / 'report.json', row)
                if not row['control_pass']:
                    raise ValueError('Forward control is not bitwise native reproduction; do not interpret direction')
            elif arm == 'row_permutation':
                comparison = discrepancy(vertices, decoded['forward'], diagonal)
                row['discrepancy_vs_forward'] = comparison
                row['control_pass'] = (comparison['moving_rms_xyz_over_anchor_diagonal'] <= PERMUTATION_RMS_OVER_D_LIMIT
                    and comparison['moving_max_abs_coordinate_over_anchor_diagonal'] <= PERMUTATION_MAX_COORD_OVER_D_LIMIT)
                if not row['control_pass']:
                    row['status'] = 'control_failed'
                json_write(folder / 'report.json', row)
                if not row['control_pass']:
                    raise ValueError('Timestamp-preserving permutation exceeds frozen numerical tolerance; adapter unqualified')
            else:
                row['discrepancy_vs_forward'] = discrepancy(vertices, decoded['forward'], diagonal)
                floor = max(state['arms']['row_permutation']['discrepancy_vs_forward']['moving_rms_xyz_over_anchor_diagonal'], 1e-8)
                row['rms_to_permutation_floor_ratio'] = row['discrepancy_vs_forward']['moving_rms_xyz_over_anchor_diagonal'] / floor
                row['direction_disagreement_above_10x_floor'] = row['rms_to_permutation_floor_ratio'] > 10
                row['interpretation'] = 'Direction dependence only; neither reconstruction error nor useful uncertainty has been established.'
                json_write(folder / 'report.json', row)
            del displacement, result, z
            check(arm + '_complete')
            json_write(args.output / 'report.json', state)
    for filename, expected in hashes.items():
        if digest(args.case_dir / filename) != expected:
            raise ValueError('Source bytes changed during diagnostic: ' + filename)
    state.update(status='completed', controls_passed=True, source_hashes_unchanged=True,
                 scientific_scope='Frozen Stage-II direction diagnostic, not reversed diffusion, quality improvement or a novel method.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--case-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu-index', default='0')
    parser.add_argument('--max-seconds', type=float, default=300)
    args = parser.parse_args()
    if not 0 < args.max_seconds <= 300:
        parser.error('Per-asset budget must be in (0,300] seconds')
    for key in ('root', 'case_dir', 'output'):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    if args.output == args.case_dir or args.case_dir in args.output.parents:
        parser.error('New diagnostic output must be outside the immutable source case')
    args.output.mkdir(parents=True, exist_ok=False)
    state = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
             'arms': {arm: {'status': 'not_started', 'variant': arm} for arm in ARMS},
             'implementation_sha256': digest(Path(__file__)), 'max_seconds': args.max_seconds,
             'gpu_reservation_mib': RESERVATION_MIB, 'allocator_cap_mib': ALLOCATOR_CAP_MIB,
             'resource_note': '6144 MiB reservation is conservative relative to parent-audited native StageII peaks3431/3427MiB over50samples each. One decoder, three serial arms. This adapter peak is not yet measured; allocator cap does not cap driver/other-process memory.',
             'timeout_note': 'Callbacks bound stages; launch with an external 300-second process-group deadline to bound an in-flight CUDA call.',
             'control_protocol': {'forward': 'bitwise equality to original native sequence required',
                 'permutation_rms_over_D_limit': PERMUTATION_RMS_OVER_D_LIMIT,
                 'permutation_max_coordinate_over_D_limit': PERMUTATION_MAX_COORD_OVER_D_LIMIT,
                 'reversal_signal_floor_multiplier': 10},
             'GT_read': False, 'training': False, 'new_method_claim': False}
    for arm, report in state['arms'].items():
        folder = args.output / 'variants' / arm
        folder.mkdir(parents=True)
        json_write(folder / 'report.json', report)
    json_write(args.output / 'command.json', {'argv': sys.argv, 'implementation_sha256': state['implementation_sha256']})
    started, monitor = time.monotonic(), None
    try:
        import torch
        # This monitor starts no model and records this process's allocator plus physical totals.
        monitor = Resources(args.output, torch, args.gpu_index)
        monitor.start()
        run(args, state, started)
    except Exception as exc:
        state.update(status='failed', error=repr(exc), traceback=traceback.format_exc(), controls_passed=False)
        for arm, report in state['arms'].items():
            if report['status'] == 'running':
                report.update(status='failed', error=repr(exc))
            elif report['status'] == 'not_started':
                report['reason'] = 'Earlier failure/control rejection or budget; no adaptive retry'
            json_write(args.output / 'variants' / arm / 'report.json', report)
    finally:
        if monitor:
            try:
                state['resources'] = monitor.finish()
            except Exception as exc:
                state['resource_monitor_error'] = repr(exc)
        state['elapsed_seconds'] = time.monotonic() - started
        json_write(args.output / 'report.json', state)
        json_write(args.output / 'progress.json', state)
    print(json.dumps({'status': state['status'], 'output': str(args.output)}), flush=True)
    return int(state['status'] != 'completed')


if __name__ == '__main__':
    raise SystemExit(main())
