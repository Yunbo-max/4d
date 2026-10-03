"""Exploratory ActionMesh pilots; offline caches, no training, no formal gate claims.

Use the existing inference environment. prepare performs Stage 0 (100 steps) and
image encoding once. run reuses that exact NPZ for standard 30-step Stage I and
frozen Stage II. No rendering or Blender process is launched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import types


def json_write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def cap_norms(norms, quantile=0.75):
    """Exact linear quantile and projection scales; no learned uncertainty claim."""
    if not norms or not 0 <= quantile <= 1:
        raise ValueError('Nonempty norms and quantile in [0,1] required')
    if any(not math.isfinite(x) or x < 0 for x in norms):
        raise ValueError('Norms must be finite and nonnegative')
    ordered = sorted(norms)
    position = quantile * (len(ordered) - 1)
    low, high = math.floor(position), math.ceil(position)
    threshold = ordered[low] + (position - low) * (ordered[high] - ordered[low])
    scales = [min(1.0, threshold / x) if x > 0 else 1.0 for x in norms]
    return threshold, scales


def setup(root):
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    repo = root / 'repo'
    if not (repo / 'actionmesh/pipeline.py').is_file():
        raise FileNotFoundError(f'ActionMesh repo not found: {repo}')
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    import torch
    import actionmesh.utils

    def require_local(repo_id, local_dir):
        p = Path(local_dir)
        if not p.is_dir() or not any(p.iterdir()):
            raise FileNotFoundError(f'Offline asset absent: {p} ({repo_id}); downloads disabled')
        return str(p)

    actionmesh.utils.download_if_missing = require_local
    if not torch.cuda.is_available():
        raise RuntimeError('This runner requires the existing CUDA environment')
    return repo, torch


def make_pipeline(repo, torch):
    from actionmesh.pipeline import ActionMeshPipeline
    pipeline = ActionMeshPipeline(
        config_name='actionmesh_lowram', config_dir=str(repo / 'actionmesh/configs'),
        dtype=torch.float16, lazy_loading=True,
    ).to('cuda')
    # Explicit protocol, rather than relying on a fast preset or changed defaults.
    pipeline.cfg.model.image_to_3D_denoiser.num_inference_steps = 100
    pipeline.scheduler.num_inference_steps = 30
    pipeline.cfg.anchor_idx = 0
    pipeline.cfg.subsampling_level = 1
    if list(map(list, pipeline.cf_guidance.guidance_at_inference)) != [[0, 1], [1, 1]]:
        raise ValueError('Expected exactly two CFG branches: anchor-only, image+anchor')
    if list(pipeline.cf_guidance.guidance_scales) != [7.5]:
        raise ValueError('Expected official guidance scale 7.5')
    return pipeline


def freeze_loaded(model):
    """Freeze nn.Modules and components of a Diffusers pipeline explicitly."""
    modules = [model]
    if hasattr(model, 'components'):
        modules.extend(model.components.values())
    for module in modules:
        if hasattr(module, 'parameters'):
            for parameter in module.parameters():
                parameter.requires_grad_(False)


class Resources:
    def __init__(self, out, torch, gpu_index):
        self.out, self.torch, self.gpu_index = out, torch, gpu_index
        self.process = self.stream = None
        self.monitor_error = None
        self.started = time.monotonic()

    def start(self):
        self.torch.cuda.reset_peak_memory_stats()
        self.stream = (self.out / 'gpu-memory.csv').open('w')
        try:
            self.process = subprocess.Popen([
                'nvidia-smi', '-i', self.gpu_index,
                '--query-gpu=timestamp,memory.used,memory.total,utilization.gpu',
                '--format=csv,noheader,nounits', '-l', '1',
            ], stdout=self.stream, stderr=subprocess.DEVNULL)
        except OSError as exc:
            self.monitor_error = str(exc)

    def finish(self):
        try:
            self.torch.cuda.synchronize()
        except Exception:
            pass
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.stream:
            self.stream.close()
        samples = []
        for line in (self.out / 'gpu-memory.csv').read_text().splitlines():
            try:
                samples.append(float(line.split(',')[1]))
            except (IndexError, ValueError):
                pass
        return {
            'elapsed_seconds': time.monotonic() - self.started,
            'peak_allocated_bytes': self.torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': self.torch.cuda.max_memory_reserved(),
            'physical_gpu_sampled_peak_mib': max(samples) if samples else None,
            'gpu_samples': len(samples), 'gpu_monitor_error': self.monitor_error,
            'gpu': self.torch.cuda.get_device_name(),
            'memory_scope': 'torch: this process; nvidia-smi: physical GPU total, sampled each second',
        }


def stage_timer(torch, durations, name, function):
    torch.cuda.synchronize()
    started = time.monotonic()
    result = function()
    torch.cuda.synchronize()
    durations[name] = time.monotonic() - started
    return result


def prepare(args, repo, torch, durations):
    import numpy as np
    from actionmesh.io.video_input import load_frames

    pipeline = make_pipeline(repo, torch)
    input_path = args.input or str(repo / 'assets/examples/kangaroo')
    inputs = load_frames(path=input_path, max_frames=16)
    if inputs.n_frames != 16:
        raise ValueError(f'Pilot requires exactly 16 real frames; got {inputs.n_frames}')
    raw_image_hashes = [hashlib.sha256(im.tobytes()).hexdigest() for im in inputs.frames]
    with torch.inference_mode():
        pipeline._load_background_removal()
        freeze_loaded(pipeline.background_removal)
        inputs.frames = stage_timer(torch, durations, 'background_removal',
            lambda: pipeline.background_removal.process_images(inputs.frames))
        pipeline._unload_model('background_removal')
        inputs.frames = pipeline.image_process.process_images(inputs.frames)

        pipeline._load_image_to_3d()
        freeze_loaded(pipeline.image_to_3d_pipe)
        latent_bank, mesh_bank = stage_timer(torch, durations, 'stage0',
            lambda: pipeline.init_banks_from_anchor(inputs, args.seed))
        pipeline._unload_model('image_to_3d_pipe')
        anchor_latent, anchor_times = latent_bank.get_ordered()
        anchor_mesh = mesh_bank.get_ordered(device='cpu')[0][0]
        anchor_array = anchor_latent.detach().cpu().numpy()

        pipeline._load_image_encoder()
        freeze_loaded(pipeline.image_encoder)
        context = stage_timer(torch, durations, 'image_encoder', lambda: pipeline.encode_all_frames(inputs))
        context_array = context.detach().cpu().numpy()
        pipeline._unload_model('image_encoder')

    arrays = {
        'schema_version': np.asarray(1, dtype=np.int64),
        'timesteps': inputs.timesteps.cpu().numpy().astype(np.float32),
        'context': context_array,
        'anchor_latent': anchor_array,
        'anchor_timesteps': anchor_times.detach().cpu().numpy().astype(np.float32),
        'anchor_vertices': np.asarray(anchor_mesh.vertices, dtype=np.float32),
        'anchor_faces': np.asarray(anchor_mesh.faces, dtype=np.int64),
        'seed': np.asarray(args.seed, dtype=np.int64),
    }
    for name in ('context', 'anchor_latent', 'anchor_vertices'):
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f'Nonfinite prepared array: {name}')
    cache = args.output / 'prepared.npz'
    np.savez_compressed(cache, **arrays)
    anchor_mesh.export(args.output / 'anchor.glb')
    frames_dir = args.output / 'processed-frames'
    frames_dir.mkdir()
    for i, frame in enumerate(inputs.frames):
        frame.save(frames_dir / f'{i:02d}.png')
    metadata = {
        'status': 'prepared', 'schema_version': 1, 'input': input_path,
        'seed': args.seed, 'stage0_steps': 100, 'stage1_steps_planned': 30,
        'frames': 16, 'config': 'actionmesh_lowram', 'dtype': 'float16',
        'cache': str(cache), 'cache_sha256': digest(cache),
        'source_decoded_image_hashes': raw_image_hashes,
        'shapes': {key: list(value.shape) for key, value in arrays.items()},
        'scope': 'Exploratory shared cache; no formal performance/quality gate',
    }
    json_write(args.output / 'prepared.json', metadata)
    return metadata


def load_cache(path, expected_seed):
    import numpy as np
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    required = {'schema_version', 'timesteps', 'context', 'anchor_latent',
                'anchor_timesteps', 'anchor_vertices', 'anchor_faces', 'seed'}
    if not required.issubset(arrays) or int(arrays['schema_version']) != 1:
        raise ValueError('Unsupported/incomplete cache schema')
    if int(arrays['seed']) != expected_seed:
        raise ValueError('Run seed must match prepared anchor seed for this pilot')
    times = arrays['timesteps']
    if times.shape != (16,) or not np.all(np.diff(times) > 0):
        raise ValueError('Cache must contain 16 ordered timesteps')
    if arrays['context'].ndim != 3 or arrays['context'].shape[0] != 16:
        raise ValueError('Bad context shape')
    if arrays['anchor_latent'].ndim != 3 or arrays['anchor_latent'].shape[0] != 1:
        raise ValueError('Expected exactly one anchor latent')
    if arrays['anchor_timesteps'].shape != (1,) or arrays['anchor_timesteps'][0] != times[0]:
        raise ValueError('Expected frame-0 anchor')
    for key in required:
        if not np.issubdtype(arrays[key].dtype, np.number) or not np.isfinite(arrays[key]).all():
            raise ValueError(f'Unsafe/nonfinite cache array: {key}')
    metadata_path = path.with_name('prepared.json')
    if not metadata_path.is_file():
        raise FileNotFoundError('prepared.json with cache hash is required')
    expected = json.loads(metadata_path.read_text())['cache_sha256']
    if digest(path) != expected:
        raise ValueError('Prepared cache hash mismatch')
    return arrays


def global_cap_scale(norms, selective_scales):
    """Global scale matching the selective cap's aggregate conditional RMS."""
    denominator = sum(x * x for x in norms)
    return math.sqrt(sum((x * scale) ** 2 for x, scale in zip(norms, selective_scales)) / denominator) if denominator > 0 else 1.0


def initialize_sampling(pipeline, arrays, torch, seed):
    from actionmesh.model.utils.storage import LatentBank
    bank = LatentBank(empty_dims=pipeline._denoiser_latent_shape)
    bank.update(torch.from_numpy(arrays['anchor_timesteps']), torch.from_numpy(arrays['anchor_latent']))
    times = torch.from_numpy(arrays['timesteps']).float()
    conditioned, mask = bank.get(times, device='cuda', add_batch_dim=True)
    noise = pipeline.scheduler.get_noise(
        batch_size=1, latent_shape=pipeline._denoiser_latent_shape, n_timesteps=16,
        generator=torch.Generator(device='cuda').manual_seed(seed), device='cuda')
    latent = conditioned * mask[..., None, None] + noise * (1.0 - mask[..., None, None])
    # Official _denoise_latents passes CPU frame times; retain that exactly so
    # RoPE trig/normalization does not silently move from CPU to CUDA.
    return latent, mask.to(latent.dtype), times[None]


def flow_segment(pipeline, torch, latent, context, mask, framestep, start, end, mode, anchor_latent, records, stats_path=None, parity_first=None):
    """Transparent native Euler flow loop for a slice of the original 30 steps.

    Mirrors SchedulerFlow._flow_sample: exact get_schedule, CFG expansion,
    _diffusion_forward, aggregate_cfg, signed distance, masked in-place update.
    Only the explicitly named suffix intervention changes the guided velocity.
    RoPE cache is recomputed at a restart (it depends on shape/times, not noise).
    """
    guidance, scheduler = pipeline.cf_guidance, pipeline.scheduler
    timesteps, distances = scheduler.get_schedule()
    timesteps, distances = timesteps.to(latent.device), distances.to(latent.device)
    if len(distances) != 30 or not scheduler.is_additive:
        raise ValueError('Expected 30-step additive native schedule')
    unobserved = guidance.get_unobserved_mask(mask)
    freqs_rot = None
    for index in range(start, end):
        t = timesteps[index]
        hidden, ctx, branch_mask, frames = guidance.cfg_at_inference(latent, context, mask, framestep)
        diffusion_time = torch.tensor([t], dtype=latent.dtype, device=latent.device).expand(hidden.shape[0])
        prediction, freqs_rot = scheduler._diffusion_forward(
            pipeline.temporal_3D_denoiser, hidden, ctx, frames, branch_mask, diffusion_time, freqs_rot)
        if prediction.ndim != 4 or prediction.shape[0] != 2:
            raise ValueError('Expected two CFG branches with batch size one')
        # aggregate_cfg mutates the base view: preserve both original branches.
        v_anchor, v_cond = (x.clone() for x in prediction.chunk(2, dim=0))
        delta = v_cond - v_anchor
        norms = delta.float().square().mean(dim=(-2, -1)).sqrt()[0].cpu().tolist()
        threshold, selective = cap_norms(norms[1:], 0.75)
        velocity = guidance.aggregate_cfg(prediction)
        original_velocity = velocity.clone()
        scales = [1.0] * 16
        energy_logs = None
        active = index >= 20 and mode != 'baseline'
        if active and mode in ('cfg-cap', 'global-cap'):
            scales[1:] = selective if mode == 'cfg-cap' else [global_cap_scale(norms[1:], selective)] * 15
            scale_tensor = torch.tensor(scales, device=delta.device, dtype=delta.dtype)[None, :, None, None]
            velocity = v_anchor + float(guidance.guidance_scales[0]) * delta * scale_tensor
        elif active and mode in ('moment-energy', 'rms-energy'):
            # This module owns only an endpoint-energy graph, no denoiser graph.
            from research_moment_energy import correction
            with torch.inference_mode(False), torch.enable_grad():
                # Fresh ordinary tensors avoid inference-tensor autograd restrictions.
                correction_value, energy_logs = correction(
                    latent.detach().clone(), original_velocity.detach().clone(),
                    v_anchor.detach().clone(), v_cond.detach().clone(),
                    anchor_latent.detach().clone(), float(t.item()) / 1000.0,
                    mode.replace('-', '_'))
            if not energy_logs.get('finite', False):
                raise ValueError(f'Energy module reported a finite-check failure: {energy_logs}')
            velocity = velocity + correction_value.detach().to(velocity)
        if not torch.isfinite(velocity).all():
            raise ValueError(f'Nonfinite velocity at step {index}')
        change = float((velocity - original_velocity).float().square().mean().sqrt().item())
        # Same additive update and anchor mask as official _flow_sample.
        flow_step = latent + distances[index] * velocity
        latent[unobserved] = flow_step[unobserved]
        if not torch.isfinite(latent).all():
            raise ValueError(f'Nonfinite latent at step {index}')
        parity = None
        if parity_first is not None and index == start:
            parity = {'bitwise_equal': bool(torch.equal(latent, parity_first)),
                      'max_abs_error': float((latent - parity_first).abs().max().item()),
                      'allclose_rtol_1e5_atol_1e6': bool(torch.allclose(latent, parity_first, rtol=1e-5, atol=1e-6))}
            if not parity['allclose_rtol_1e5_atol_1e6']:
                raise ValueError(f'Native first-step parity failed: {parity}')
        row = {'step_index': index, 'step': index + 1, 'timestep': float(t.item()),
               'distance': float(distances[index].item()), 'active': active,
               'threshold_rms': threshold, 'frame_rms_before': norms, 'scales': scales,
               'clipped_movable_frames': sum(x < 1.0 - 1e-7 for x in scales[1:]),
               'guided_prediction_rms_change': change, 'energy': energy_logs,
               'native_first_step_parity': parity}
        records.append(row)
        if stats_path is not None:
            json_write(stats_path, records)
        print(f'PROBE step={index+1}/30 mode={mode} change_rms={change:.6g}', flush=True)
    return latent


def prefix(args, repo, torch, durations):
    import numpy as np
    arrays = load_cache(args.cache, args.seed)
    pipeline = make_pipeline(repo, torch)
    records = []
    with torch.inference_mode():
        latent, mask, times = initialize_sampling(pipeline, arrays, torch, args.seed)
        initial = latent.detach().cpu().numpy().copy()
        context = torch.from_numpy(arrays['context']).to('cuda')[None]
        pipeline._load_temporal_denoiser()
        freeze_loaded(pipeline.temporal_3D_denoiser)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # One extra native forward verifies our actual model/precision/update
            # wiring. This is one-step parity, not a full 30-step parity claim.
            native_generator = pipeline.scheduler._flow_sample(
                pipeline.temporal_3D_denoiser, pipeline.cf_guidance,
                init_latent=latent.clone(), context=context, device='cuda',
                mask=mask, framestep=times, disable_prog=True)
            parity_first = stage_timer(torch, durations, 'native_first_step_canary',
                lambda: next(native_generator)[0].clone())
            native_generator.close()
            latent = stage_timer(torch, durations, 'prefix20', lambda: flow_segment(
                pipeline, torch, latent, context, mask, times, 0, 20, 'baseline',
                torch.from_numpy(arrays['anchor_latent']).to('cuda'), records,
                args.output / 'sampling-stats.json', parity_first))
        pipeline._unload_model('temporal_3D_denoiser')
        schedule, distances = pipeline.scheduler.get_schedule()
        saved = args.output / 'prefix.npz'
        np.savez_compressed(saved, latent_boundary=latent.cpu().numpy(), initial_latent=initial,
            timesteps_schedule=schedule.numpy(), distances=distances.numpy(),
            mask=mask.cpu().numpy(), framestep=times.cpu().numpy(),
            seed=np.asarray(args.seed, dtype=np.int64), next_step=np.asarray(20, dtype=np.int64))
    json_write(args.output / 'sampling-stats.json', records)
    result = {'status': 'prefix_prepared', 'cache_sha256': digest(args.cache),
              'prefix_sha256': digest(saved), 'seed': args.seed, 'native_schedule_steps': 30,
              'executed_steps': 20, 'next_step': 20,
              'native_first_step_parity': records[0]['native_first_step_parity'],
              'parity_scope': 'One real-model step compared to native sampler; exact saved schedule. No full official 30-step parity run.'}
    json_write(args.output / 'prefix.json', result)
    return result


def load_prefix(path, cache_path, pipeline, seed):
    import numpy as np
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    metadata = json.loads(path.with_name('prefix.json').read_text())
    if digest(path) != metadata['prefix_sha256'] or digest(cache_path) != metadata['cache_sha256']:
        raise ValueError('Prefix or preparation hash mismatch')
    if int(arrays['seed']) != seed or int(arrays['next_step']) != 20:
        raise ValueError('Prefix seed/boundary mismatch')
    schedule, distances = pipeline.scheduler.get_schedule()
    if not np.array_equal(arrays['timesteps_schedule'], schedule.numpy()) or not np.array_equal(arrays['distances'], distances.numpy()):
        raise ValueError('Prefix schedule differs from native 30-step schedule')
    for key, value in arrays.items():
        if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
            raise ValueError(f'Unsafe/nonfinite prefix array: {key}')
    return arrays


def run(args, repo, torch, durations):
    import numpy as np
    import trimesh
    from PIL import Image
    from actionmesh.io.video_input import ActionMeshInput
    from actionmesh.model.utils.storage import LatentBank, MeshBank

    arrays = load_cache(args.cache, args.seed)
    pipeline = make_pipeline(repo, torch)
    if tuple(arrays['anchor_latent'].shape[1:]) != tuple(pipeline._denoiser_latent_shape):
        raise ValueError('Cache latent shape does not match pipeline config')
    # Frame pixels are never read in generate_3d_latents; context is cached exactly.
    inputs = ActionMeshInput(frames=[Image.new('RGB', (1, 1)) for _ in range(16)],
                             timesteps=torch.from_numpy(arrays['timesteps']).float())
    latent_bank = LatentBank(empty_dims=pipeline._denoiser_latent_shape)
    latent_bank.update(torch.from_numpy(arrays['anchor_timesteps']), torch.from_numpy(arrays['anchor_latent']).to('cuda'))
    mesh_bank = MeshBank()
    anchor_mesh = trimesh.Trimesh(vertices=arrays['anchor_vertices'], faces=arrays['anchor_faces'], process=False)
    mesh_bank.update(torch.from_numpy(arrays['anchor_timesteps']), [anchor_mesh])
    saved_prefix = load_prefix(args.prefix, args.cache, pipeline, args.seed)
    records = []
    with torch.inference_mode():
        context = torch.from_numpy(arrays['context']).to('cuda')[None]
        boundary = torch.from_numpy(saved_prefix['latent_boundary']).to('cuda')
        mask = torch.from_numpy(saved_prefix['mask']).to('cuda')
        times = torch.from_numpy(saved_prefix['framestep'])
        expected_shape = (1, 16, *pipeline._denoiser_latent_shape)
        if tuple(boundary.shape) != expected_shape:
            raise ValueError('Prefix boundary shape mismatch')
        anchor_latent = torch.from_numpy(arrays['anchor_latent']).to('cuda')
        pipeline._load_temporal_denoiser()
        freeze_loaded(pipeline.temporal_3D_denoiser)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            denoised = stage_timer(torch, durations, 'suffix10', lambda: flow_segment(
                pipeline, torch, boundary, context, mask, times, 20, 30, args.mode, anchor_latent, records,
                args.output / 'sampling-stats.json'))
        if len(records) != 10:
            raise ValueError(f'Expected ten suffix steps; recorded {len(records)} calls')
        latent_bank.update(torch.from_numpy(arrays['timesteps']), denoised)
        pipeline._unload_model('temporal_3D_denoiser')
        latents, times = latent_bank.get_ordered()
        latent_array = latents.detach().cpu().numpy()
        if not np.isfinite(latent_array).all():
            raise ValueError('Nonfinite denoised latent')
        np.savez_compressed(args.output / 'denoised.npz', latents=latent_array,
                            timesteps=times.detach().cpu().numpy(),
                            seed=np.asarray(args.seed, dtype=np.int64))
        json_write(args.output / 'sampling-stats.json', records)
        del context

        pipeline._load_temporal_vae()
        freeze_loaded(pipeline.temporal_3D_vae)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            mesh_bank = stage_timer(torch, durations, 'stage2', lambda: pipeline.generate_mesh_animation(latent_bank, mesh_bank))
        pipeline._unload_model('temporal_3D_vae')
    meshes = mesh_bank.get_ordered(device='cpu')[0]
    vertices = np.stack([np.asarray(mesh.vertices) for mesh in meshes]).astype(np.float32)
    faces = np.asarray(meshes[0].faces, dtype=np.int64)
    if vertices.shape[0] != 16 or not np.isfinite(vertices).all():
        raise ValueError('Expected 16 finite decoded meshes')
    if not all(np.array_equal(mesh.faces, faces) for mesh in meshes):
        raise ValueError('Decoded topology differs across frames')
    np.save(args.output / 'deformations_vertices.npy', vertices, allow_pickle=False)
    np.save(args.output / 'deformations_faces.npy', faces, allow_pickle=False)
    for i, mesh in enumerate(meshes):
        mesh.export(args.output / f'mesh_{i:04d}.glb')
    velocity = np.diff(vertices, axis=0)
    acceleration = np.diff(vertices, n=2, axis=0)
    result = {
        'status': 'completed_exploratory_run', 'mode': args.mode, 'frames': 16,
        'seed': args.seed, 'stage1_steps': 30, 'guidance_scale': 7.5,
        'cache': str(args.cache), 'cache_sha256': digest(args.cache),
        'prefix': str(args.prefix), 'prefix_sha256': digest(args.prefix),
        'shared_native_prefix_steps': 20, 'executed_suffix_steps': 10,
        'denoised_sha256': digest(args.output / 'denoised.npz'),
        'vertices_per_frame': int(vertices.shape[1]), 'faces': int(len(faces)),
        'finite_geometry': True, 'fixed_topology': True,
        'mean_vertex_speed': float(np.linalg.norm(velocity, axis=-1).mean()),
        'mean_vertex_acceleration': float(np.linalg.norm(acceleration, axis=-1).mean()),
        'max_displacement_from_anchor': float(np.linalg.norm(vertices - vertices[:1], axis=-1).max()),
        'changed_steps': sum(r['guided_prediction_rms_change'] > 0 for r in records),
        'clipped_frame_steps': sum(r['clipped_movable_frames'] for r in records),
        'validation_scope': 'Execution, finite geometry and intervention activity only. Smoothness is not accuracy; no GT or formal gate claim.',
        'training': False, 'denoiser_backprop': False,
        'small_energy_autograd': args.mode in ('moment-energy', 'rms-energy'),
    }
    return result


def cpu_canary(root):
    """Real scheduler, deterministic toy field: native vs shared-prefix parity."""
    sys.path.insert(0, str(root / 'repo'))
    import torch
    from actionmesh.scheduler.scheduler import SchedulerFlow
    from actionmesh.scheduler.guidance import ClassifierFreeGuidance

    class ToyField:
        def forward(self, hidden_states, context, framestep, mask, diffusion_time, freqs_rot):
            velocity = (0.13 * hidden_states + 0.01 * context.mean(dim=(-2, -1), keepdim=True)
                        + 1e-5 * diffusion_time[:, None, None, None])
            return velocity, freqs_rot

    scheduler = SchedulerFlow(num_inference_steps=30, is_additive=True, split_cfg_batch=True)
    guidance = ClassifierFreeGuidance(guidance_at_inference=[[0, 1], [1, 1]], guidance_scales=[7.5])
    pipeline = types.SimpleNamespace(scheduler=scheduler, cf_guidance=guidance, temporal_3D_denoiser=ToyField())
    generator = torch.Generator(device='cpu').manual_seed(1234)
    initial = torch.randn(1, 16, 8, 4, generator=generator)
    context = torch.randn(1, 16, 3, 4, generator=generator)
    mask = torch.zeros(1, 16)
    mask[:, 0] = 1
    times = torch.arange(16, dtype=torch.float32)[None]
    with torch.inference_mode():
        for native, _ in scheduler._flow_sample(ToyField(), guidance, initial.clone(), context,
                                               device='cpu', mask=mask, framestep=times):
            pass
        records = []
        boundary = flow_segment(pipeline, torch, initial.clone(), context, mask, times,
                                0, 20, 'baseline', initial[:, :1], records)
        continued = flow_segment(pipeline, torch, boundary.clone(), context, mask, times,
                                 20, 30, 'baseline', initial[:, :1], records)
    assert torch.equal(native, continued), 'Shared-prefix baseline differs from native scheduler'
    assert torch.equal(continued[:, 0], initial[:, 0]), 'Anchor mask did not hold'
    threshold, scales = cap_norms([1., 2., 3., 100.], .75)
    global_scale = global_cap_scale([1., 2., 3., 100.], scales)
    selective_square = sum((x * s) ** 2 for x, s in zip([1., 2., 3., 100.], scales))
    global_square = sum((x * global_scale) ** 2 for x in [1., 2., 3., 100.])
    assert abs(selective_square - global_square) < 1e-8
    return {'status': 'passed', 'device': 'cpu', 'toy_native_30_vs_shared_20_10_bitwise_equal': True,
            'anchor_held': True, 'global_selective_norm_matched': True,
            'scope': 'Real scheduler with toy velocity field; not real-model GPU parity'}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    canary = sub.add_parser('cpu-canary', help='CPU-only native flow parity and projection checks')
    canary.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    for name in ('prepare', 'prefix', 'run'):
        child = sub.add_parser(name)
        child.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
        child.add_argument('--output', required=True, type=Path, help='New output directory; existing paths are rejected')
        child.add_argument('--seed', type=int, default=42)
        child.add_argument('--gpu-index', default='0', help='Physical GPU index for nvidia-smi telemetry')
        if name == 'prepare':
            child.add_argument('--input', help='Default: repo/assets/examples/kangaroo; first 16 real frames')
        else:
            child.add_argument('--cache', required=True, type=Path)
            if name == 'run':
                child.add_argument('--prefix', required=True, type=Path)
                child.add_argument('--mode', choices=['baseline', 'cfg-cap', 'global-cap', 'moment-energy', 'rms-energy'], required=True)
    args = parser.parse_args(argv)
    for name in ('root', 'output', 'cache', 'prefix'):
        value = getattr(args, name, None)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    if args.command == 'prepare' and args.input:
        args.input = str(Path(args.input).expanduser().resolve())
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.command == 'cpu-canary':
        print(json.dumps(cpu_canary(args.root), indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    json_write(args.output / 'command.json', {'argv': sys.argv, 'root': str(args.root),
        'script_sha256': digest(Path(__file__).resolve()), 'offline': True,
        'scope': 'Exploratory pilot; no formal NaturalGate0 or quality claim'})
    monitor = None
    durations = {}
    result = {'status': 'failed'}
    try:
        repo, torch = setup(args.root)
        monitor = Resources(args.output, torch, args.gpu_index)
        monitor.start()
        result = {'prepare': prepare, 'prefix': prefix, 'run': run}[args.command](args, repo, torch, durations)
    except Exception as exc:
        result = {'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc),
                  'traceback': traceback.format_exc()}
        raise
    finally:
        result['stage_seconds'] = durations
        if monitor:
            result['resources'] = monitor.finish()
        json_write(args.output / 'report.json', result)


if __name__ == '__main__':
    main()
