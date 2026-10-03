"""Resume ONLY a deadline-interrupted Stage-II reversal using verified v1 controls.

This is a fresh, separately bounded engineering attempt. V1 files are immutable;
passed forward/permutation controls are verified from artifacts, never rerun.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

import numpy as np
import research_census_time_direction as original
from research_three_ideas import Resources, digest, json_write


def read(path, inventory):
    value = json.loads(path.read_text())
    inventory[str(path)] = digest(path)
    return value


def require_hash(path, expected, inventory):
    actual = digest(path)
    if actual != expected:
        raise ValueError('Resume source hash mismatch: ' + str(path))
    inventory[str(path)] = actual
    return actual


def argv_path(argv, flag):
    if not isinstance(argv, list) or argv.count(flag) != 1:
        raise ValueError('Missing/ambiguous receipt command argument: ' + flag)
    index = argv.index(flag)
    if index + 1 >= len(argv):
        raise ValueError('Missing receipt command value: ' + flag)
    return Path(argv[index + 1]).expanduser().resolve()


def verify_controls(args):
    """CPU-only provenance and independent control recalculation before model load."""
    inventory = {}
    prior = read(args.source_attempt / 'report.json', inventory)
    provenance = read(args.source_attempt / 'source-provenance.json', inventory)
    receipt = read(args.timeout_receipt, inventory)
    old_script_hash = digest(Path(original.__file__))
    if prior.get('implementation_sha256') != old_script_hash or receipt.get('source_sha256') != old_script_hash:
        raise ValueError('Original runner differs from the v1 run/timeout receipt')
    if receipt.get('status') != 'stopped_after_failure':
        raise ValueError('Expected preserved failed-launch receipt')
    matching = [job for job in receipt.get('jobs', []) if
                argv_path(job.get('argv'), '--case-dir') == args.case_dir and
                argv_path(job.get('argv'), '--output') == args.source_attempt]
    if len(matching) != 1:
        raise ValueError('Timeout receipt must identify this exact case and source attempt once')
    terminated = matching[0]
    if terminated.get('status') != 'failed' or terminated.get('termination_reason') != 'deadline' or terminated.get('exit_code') != -15:
        raise ValueError('Resume is scoped to the recorded external-deadline termination')
    if (args.source_attempt / 'variants/time_reversal/sequence.npz').exists():
        raise FileExistsError('Reversal sequence already exists; reconcile it instead of rerunning')
    source, source_hashes, latents, native, faces, prepared = original.load_source(args.case_dir)
    source_hashes['code-provenance.json'] = digest(args.case_dir / 'code-provenance.json')
    if source_hashes != provenance.get('source_hashes') or source_hashes != prior.get('source_hashes'):
        raise ValueError('Native case caches changed since the completed controls')
    for name, value in source_hashes.items():
        require_hash(args.case_dir / name, value, inventory)
    identity = (source['uid'], source['seed'])
    if (prior.get('uid'), prior.get('seed')) != identity or (provenance.get('uid'), provenance.get('seed')) != identity:
        raise ValueError('Control attempt/native case UID or seed mismatch')
    diagonal = float(np.linalg.norm(np.ptp(native[0].astype(np.float64), axis=0)))
    if prior.get('anchor_diagonal') != diagonal or not np.isfinite(diagonal) or diagonal <= 0:
        raise ValueError('Saved control normalization changed')
    expected_arrays = {'source_latents_array_sha256': original.array_digest(latents),
                       'cached_query_sha256': original.array_digest(prepared['anchor_query_features']),
                       'query_vertex_ids_sha256': original.array_digest(prepared['query_vertex_ids'])}
    if any(prior.get(key) != value for key, value in expected_arrays.items()):
        raise ValueError('Control latent/query/vertex identity changed')
    expected_protocol = {'forward': 'bitwise equality to original native sequence required',
        'permutation_rms_over_D_limit': original.PERMUTATION_RMS_OVER_D_LIMIT,
        'permutation_max_coordinate_over_D_limit': original.PERMUTATION_MAX_COORD_OVER_D_LIMIT,
        'reversal_signal_floor_multiplier': 10}
    if prior.get('control_protocol') != expected_protocol:
        raise ValueError('Original control tolerances differ; no relaxation allowed')
    if prior.get('code_sha256') != provenance.get('code_sha256'):
        raise ValueError('Original code provenance reports disagree')
    for filename, expected in provenance['code_sha256'].items():
        require_hash(args.root / 'repo' / filename, expected, inventory)
    checkpoint = args.root / 'repo/pretrained_weights/ActionMesh/autoencoder'
    current = {str(path.relative_to(checkpoint)): digest(path) for path in sorted(checkpoint.rglob('*'))
               if path.is_file() and '.cache' not in path.parts}
    if not current or current != provenance.get('checkpoint_sha256') or current != prior.get('checkpoint_sha256'):
        raise ValueError('Temporal decoder checkpoint changed since passed controls')
    for filename, value in current.items():
        inventory[str(checkpoint / filename)] = value
    controls, arrays = {}, {}
    for arm in ('forward', 'row_permutation'):
        folder = args.source_attempt / 'variants' / arm
        report = read(folder / 'report.json', inventory)
        if prior.get('arms', {}).get(arm) != report:
            raise ValueError('Completed control report disagrees with saved v1 snapshot')
        if (report.get('status') != 'completed' or report.get('control_pass') is not True or
                (report.get('uid'), report.get('seed'), report.get('variant')) != (*identity, arm)):
            raise ValueError('Control was not completed and accepted: ' + arm)
        path = folder / 'sequence.npz'
        require_hash(path, report.get('sequence_sha256'), inventory)
        if report.get('sha256', {}).get('sequence.npz') != report.get('sequence_sha256'):
            raise ValueError('Control sequence hashes disagree')
        with np.load(path, allow_pickle=False) as saved:
            vertices = saved['vertices'].copy()
            if (vertices.shape != native.shape or vertices.dtype != native.dtype or not np.isfinite(vertices).all()
                    or not original.bitwise_equal(saved['faces'], faces)
                    or not original.bitwise_equal(vertices[0], native[0])
                    or not original.bitwise_equal(saved['query_vertex_ids'], prepared['query_vertex_ids'])
                    or not np.array_equal(saved['frame_indices'], np.arange(16))
                    or not np.array_equal(saved['timesteps'], np.arange(16))
                    or not np.array_equal(saved['decoder_clock_times'], np.arange(16))):
                raise ValueError('Completed control lost geometry/frame/query identity: ' + arm)
        expected_mapping = {k: value.tolist() if isinstance(value, np.ndarray) else value
                            for k, value in original.arm_mapping(arm).items()}
        if report.get('mapping') != expected_mapping:
            raise ValueError('Control physical/time mapping mismatch')
        ordered_latents = latents[original.arm_mapping(arm)['latent_row_physical_ids']]
        if report.get('latent_input_sha256') != original.array_digest(ordered_latents):
            raise ValueError('Control used different ordered latents')
        comparison = original.discrepancy(vertices, native, diagonal)
        if comparison != report.get('discrepancy_vs_native'):
            raise ValueError('Control disagreement report cannot be reproduced from arrays')
        if arm == 'forward' and not original.bitwise_equal(vertices, native):
            raise ValueError('Forward control no longer reproduces native vertices bitwise')
        controls[arm] = {'source_report': str(folder / 'report.json'),
                         'source_report_sha256': inventory[str(folder / 'report.json')],
                         'source_sequence': str(path), 'source_sequence_sha256': inventory[str(path)],
                         'status': 'verified_reused_control', 'gpu_recomputed': False,
                         'independent_discrepancy_recalculation': comparison}
        arrays[arm] = vertices
    permutation = original.discrepancy(arrays['row_permutation'], arrays['forward'], diagonal)
    if (permutation != prior['arms']['row_permutation'].get('discrepancy_vs_forward') or
            permutation['moving_rms_xyz_over_anchor_diagonal'] > original.PERMUTATION_RMS_OVER_D_LIMIT or
            permutation['moving_max_abs_coordinate_over_anchor_diagonal'] > original.PERMUTATION_MAX_COORD_OVER_D_LIMIT):
        raise ValueError('Original permutation control fails independent unchanged-tolerance check')
    return {'source': source, 'latents': latents, 'native': native, 'faces': faces, 'prepared': prepared,
            'diagonal': diagonal, 'checkpoint': checkpoint, 'inventory': inventory, 'controls': controls,
            'permutation': permutation, 'timeout_job': terminated,
            'prior_report': prior,
            'source_attempt_status': prior.get('status'), 'source_script_sha256': old_script_hash,
            'source_protocol_sha256': receipt.get('protocol_sha256')}


def run(args, state, started):
    def check(stage):
        state.update(stage=stage, elapsed_seconds=time.monotonic() - started)
        json_write(args.output / 'progress.json', state)
        print('TIME_DIRECTION_RESUME', stage, round(state['elapsed_seconds'], 2), flush=True)
        if state['elapsed_seconds'] >= args.max_seconds:
            raise TimeoutError('New reversal-only attempt reached its independent deadline')
    check('verifying_immutable_controls_on_cpu')
    verified = verify_controls(args)
    native, latents, prepared = verified['native'], verified['latents'], verified['prepared']
    state.update(uid=verified['source']['uid'], seed=42, controls=verified['controls'],
                 controls_verified_without_gpu_reexecution=True, source_files_sha256=verified['inventory'],
                 source_attempt_status=verified['source_attempt_status'],
                 preserved_timeout_job=verified['timeout_job'], source_script_sha256=verified['source_script_sha256'],
                 source_protocol_sha256=verified['source_protocol_sha256'],
                 source_timeout_interpretation='Engineering deadline; no completed reversal observation or scientific failure in the original attempt.')
    prior = verified['prior_report']
    state.update(source_hashes=prior['source_hashes'], query_vertex_ids_sha256=prior['query_vertex_ids_sha256'],
                 cached_query_sha256=prior['cached_query_sha256'], source_latents_array_sha256=prior['source_latents_array_sha256'],
                 source_case_dir=str(args.case_dir), checkpoint_sha256=prior['checkpoint_sha256'],
                 code_sha256=prior['code_sha256'], controls_passed=False, GT_read=False, training=False,
                 arms={arm: prior['arms'][arm] for arm in ('forward', 'row_permutation')})
    for arm in ('forward', 'row_permutation'):
        destination = args.output / 'variants' / arm
        destination.mkdir(parents=True, exist_ok=False)
        for filename in ('report.json', 'sequence.npz'):
            source_path = args.source_attempt / 'variants' / arm / filename
            shutil.copyfile(source_path, destination / filename)
            if digest(destination / filename) != verified['inventory'][str(source_path)]:
                raise ValueError('Reused control copy differs from its immutable source')
    json_write(args.output / 'reused-controls.json', {'controls': verified['controls'],
        'copy_policy': 'Original sequence/report bytes copied exactly; original timings retained; zero additional control GPU passes.',
        'source_attempt': str(args.source_attempt), 'source_attempt_report_sha256': verified['inventory'][str(args.source_attempt / 'report.json')]})
    json_write(args.output / 'control-verification.json', {k: state[k] for k in ('controls', 'source_files_sha256',
               'preserved_timeout_job', 'source_attempt_status', 'source_script_sha256', 'source_protocol_sha256')})
    json_write(args.output / 'report.json', state)
    check('controls_verified_loading_decoder')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    import torch
    sys.path.insert(0, str(args.root / 'repo'))
    from actionmesh.model.temporal_autoencoder import ActionMeshAutoencoder
    from actionmesh.model.utils.embeddings import get_scaling, apply_scaling
    device = torch.device(args.device)
    if device.type != 'cuda' or device.index is None or not torch.cuda.is_available():
        raise ValueError('An explicit available CUDA device is required for the new pass')
    torch.cuda.set_device(device)
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1., original.ALLOCATOR_CAP_MIB * 1024 ** 2 / total), device=device)
    torch.set_num_threads(2)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    monitor = Resources(args.output, torch, args.gpu_index)
    monitor.start()
    try:
        decoder = ActionMeshAutoencoder.from_pretrained(str(verified['checkpoint']), local_files_only=True).eval().to(device)
        for parameter in decoder.parameters():
            parameter.requires_grad_(False)
        if (decoder.training or decoder.prediction_mode != 'direct' or decoder.temporal_context_size != 16
                or any(p.requires_grad or (p.is_floating_point() and p.dtype != torch.float32) for p in decoder.parameters())):
            raise ValueError('Native frozen FP32 direct decoder configuration changed')
        mapping = original.arm_mapping('time_reversal')
        z_array = np.ascontiguousarray(latents[mapping['latent_row_physical_ids']])
        query = torch.from_numpy(prepared['anchor_query_features']).unsqueeze(0).to(device)
        z = torch.from_numpy(z_array).unsqueeze(0).to(device)
        times = torch.from_numpy(mapping['latent_row_clock_times']).unsqueeze(0).to(device)
        source_time = torch.tensor([mapping['source_clock_time']], dtype=times.dtype, device=device)
        targets = torch.from_numpy(mapping['target_clock_times']).unsqueeze(0).to(device)
        minimum, span = get_scaling(times)
        source_alpha = apply_scaling(source_time, minimum, span)
        target_alphas = apply_scaling(targets, minimum, span)
        arm = {'status': 'running', 'variant': 'time_reversal', 'uid': state['uid'], 'seed': 42, 'frames': 16,
               'mapping': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in mapping.items()},
               'source_alpha': source_alpha.cpu().tolist(), 'target_alphas': target_alphas.cpu().tolist(),
               'latent_input_sha256': original.array_digest(z_array), 'latent_dtype': str(z.dtype), 'query_dtype': str(query.dtype),
               'resumed_from': str(args.source_attempt), 'passed_controls_reused': True}
        state['reversal'] = arm
        state['arms']['time_reversal'] = arm
        folder = args.output / 'variants/time_reversal'
        json_write(folder / 'report.json', arm)
        check('reversal_start')
        pass_started = time.monotonic()
        def callback(step, count):
            check(f'reversal_target_{step}_of_{count}')
        with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.float16):
            displacement = decoder(latent=z, framestep=times, source_alpha=source_alpha,
                target_alphas=target_alphas, query=query, step_callback=callback)
            result = decoder.apply_displacement(vertex=query[..., :3], displacement=displacement)
        torch.cuda.synchronize(device)
        values = result[0].detach().cpu().numpy().astype(native.dtype)
        if values.shape != native[1:].shape or not np.isfinite(values).all():
            raise ValueError('Expected all 15 finite nonanchor targets')
        vertices = np.concatenate((native[:1].copy(), values))
        check('reversal_all_16_frames_ready')
        sequence = folder / 'sequence.npz'
        np.savez_compressed(sequence, vertices=vertices, faces=verified['faces'],
            frame_indices=mapping['output_physical_frames'], timesteps=np.arange(16, dtype=np.float32),
            decoder_clock_times=mapping['output_clock_times'], query_vertex_ids=prepared['query_vertex_ids'])
        contrast = original.discrepancy(vertices, native, verified['diagonal'])
        floor = max(verified['permutation']['moving_rms_xyz_over_anchor_diagonal'], 1e-8)
        ratio = contrast['moving_rms_xyz_over_anchor_diagonal'] / floor
        arm.update(status='completed', elapsed_seconds=time.monotonic() - pass_started,
            sequence_sha256=digest(sequence), sha256={'sequence.npz': digest(sequence)},
            anchor_bitwise_unchanged=original.bitwise_equal(vertices[0], native[0]), faces_bitwise_unchanged=True,
            all_16_physical_frames_once=True, source_query_ids_unchanged=True,
            discrepancy_vs_native=contrast, discrepancy_vs_forward=contrast,
            rms_to_permutation_floor_ratio=ratio, direction_disagreement_above_10x_floor=ratio > 10,
            training=False, stageI_regenerated=False, stageII_only=True,
            interpretation='Direction dependence only; not ground-truth error, quality improvement or a new method.')
        json_write(folder / 'report.json', arm)
        for name, expected in verified['inventory'].items():
            require_hash(Path(name), expected, {})
        check('immutable_sources_rechecked')
        state.update(status='completed', controls_passed=True, source_hashes_unchanged=True, new_gpu_decoder_passes=1)
    finally:
        state['resources'] = monitor.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'case-dir', 'source-attempt', 'timeout-receipt', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu-index', default='0')
    parser.add_argument('--max-seconds', type=float, default=300)
    args = parser.parse_args()
    for name in ('root', 'case_dir', 'source_attempt', 'timeout_receipt', 'output'):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if not 0 < args.max_seconds <= 300:
        parser.error('New engineering attempt must remain within 300 seconds')
    if any(args.output == path or path in args.output.parents for path in (args.case_dir, args.source_attempt)):
        parser.error('New output must be outside both immutable source directories')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'variants/time_reversal').mkdir(parents=True)
    started = time.monotonic()
    state = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
             'resume_script_sha256': digest(Path(__file__)), 'source_attempt': str(args.source_attempt),
             'timeout_receipt': str(args.timeout_receipt), 'max_seconds': args.max_seconds,
             'gpu_reservation_mib': original.RESERVATION_MIB, 'allocator_cap_mib': original.ALLOCATOR_CAP_MIB,
             'control_tolerances_unchanged': True, 'controls_gpu_reexecuted': False,
             'timeout_note': 'Caller must impose a separate300s process-group deadline; the original300s attempt and its cost remain preserved.',
             'scope': 'Engineering resume of one incomplete reversal; no new method or scientific result assumed.'}
    json_write(args.output / 'command.json', {'argv': sys.argv, 'resume_script_sha256': state['resume_script_sha256']})
    try:
        run(args, state, started)
    except Exception as exc:
        state.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        if state.get('reversal', {}).get('status') == 'running':
            state['reversal'].update(status='failed', error=repr(exc))
            json_write(args.output / 'variants/time_reversal/report.json', state['reversal'])
    finally:
        state['elapsed_seconds'] = time.monotonic() - started
        json_write(args.output / 'report.json', state)
        json_write(args.output / 'progress.json', state)
    print(json.dumps({'status': state['status'], 'output': str(args.output)}), flush=True)
    return int(state['status'] != 'completed')


if __name__ == '__main__':
    raise SystemExit(main())
