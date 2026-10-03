"""One offline, unmodified ActionMesh video-to-4D baseline for a failure census.

Requires exactly 16 PNG frames, an existing CUDA environment and local weights.
No training, sampler intervention, download, renderer or comparison metric is run.
All output goes into a new directory; a failed attempt must get a new directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess
import sys
import traceback

from research_three_ideas import (
    Resources, digest, json_write, make_pipeline, setup, stage_timer,
)


def freeze_loaded(model):
    """Freeze module parameters without calling Diffusers' metadata dictionary.

    Some Diffusers pipelines expose ``parameters`` as a dict rather than the
    callable nn.Module method. Their actual trainable modules live in components.
    """
    modules = [model]
    components = getattr(model, 'components', None)
    if isinstance(components, dict):
        modules.extend(components.values())
    parameters = {}
    for module in modules:
        iterator = getattr(module, 'parameters', None)
        if callable(iterator):
            for parameter in iterator():
                parameters[id(parameter)] = parameter
    if not parameters:
        raise ValueError('No callable module parameters found; cannot verify frozen weights')
    for parameter in parameters.values():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in parameters.values()):
        raise RuntimeError('A loaded model parameter remained trainable after freezing')
    return len(parameters)


def freeze_cpu_canary():
    """Dependency-free regression fixture for pipeline metadata/module freezing."""
    class Parameter:
        def __init__(self):
            self.requires_grad = True
            self.freeze_calls = 0

        def requires_grad_(self, value):
            self.requires_grad = value
            self.freeze_calls += 1
            return self

    class Module:
        def __init__(self, *parameters):
            self.values = parameters

        def parameters(self):
            return iter(self.values)

    first, second, direct = Parameter(), Parameter(), Parameter()
    first_module = Module(first, second)

    class Pipeline:
        parameters = {'metadata': 'not callable'}
        components = {'model': first_module, 'shared_module': first_module,
                      'scheduler': object(), 'optional': None}

    assert freeze_loaded(Pipeline()) == 2
    assert not first.requires_grad and not second.requires_grad
    assert first.freeze_calls == second.freeze_calls == 1
    assert freeze_loaded(Module(direct)) == 1 and not direct.requires_grad

    class RefusesFreeze(Parameter):
        def requires_grad_(self, value):
            return self

    try:
        freeze_loaded(Module(RefusesFreeze()))
    except RuntimeError:
        pass
    else:
        raise AssertionError('Failed to detect a parameter remaining trainable')
    try:
        freeze_loaded(object())
    except ValueError:
        pass
    else:
        raise AssertionError('Unverifiable empty model accepted')
    return {'status': 'passed', 'dict_pipeline_parameters_skipped': True,
            'components_and_direct_module_frozen': True, 'shared_parameters_deduplicated': True,
            'remaining_trainable_parameter_rejected': True, 'empty_model_rejected': True,
            'scope': 'CPU mock regression; no Torch/CUDA/model forward required'}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def natural_key(path):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r'(\d+)', path.name)]


def load_input(directory, output, torch):
    import numpy as np
    from PIL import Image
    from actionmesh.io.video_input import ActionMeshInput
    if not directory.is_dir():
        raise ValueError(f'Input must be an extracted frames directory: {directory}')
    paths = sorted((p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == '.png'), key=natural_key)
    if len(paths) != 16:
        raise ValueError(f'Expected exactly 16 real PNG frames; found {len(paths)}. No padding, subsampling or truncation is performed.')
    frame_dir = output / 'input-frames'
    frame_dir.mkdir()
    frames, records = [], []
    for index, path in enumerate(paths):
        with Image.open(path) as opened:
            original_mode = opened.mode
            frame = opened.convert('RGBA')
        saved = frame_dir / f'{index:02d}.png'
        frame.save(saved)
        frames.append(frame)
        records.append({'index': index, 'source': str(path), 'source_sha256': digest(path),
                        'source_mode': original_mode, 'size_wh': list(frame.size),
                        'decoded_rgba_sha256': hashlib.sha256(frame.tobytes()).hexdigest(),
                        'saved_file': str(saved.relative_to(output)), 'saved_sha256': digest(saved)})
    if len({frame.size for frame in frames}) != 1:
        raise ValueError('All input frames must have the same dimensions for shared cropping')
    return ActionMeshInput(frames, torch.arange(16, dtype=torch.float32)), records


def preprocessing(inputs, pipeline, torch, output, durations):
    """Call official preprocessing and record its precise pixel-coordinate map."""
    import numpy as np
    from actionmesh.preprocessing.image_processor import is_valid_alpha, aggregate_bboxes
    valid = [bool(is_valid_alpha(np.asarray(frame)[..., 3])) for frame in inputs.frames]
    # The official remover also returns valid RGBA unchanged. Avoid loading its
    # weights when all alpha masks are already meaningful, preserving those bytes.
    if not all(valid):
        pipeline._load_background_removal()
        freeze_loaded(pipeline.background_removal)
        inputs.frames = stage_timer(torch, durations, 'background_removal',
            lambda: pipeline.background_removal.process_images(inputs.frames))
        pipeline._unload_model('background_removal')
    else:
        durations['background_removal'] = 0.0
    masked_dir, processed_dir = output / 'masked-frames', output / 'processed-frames'
    masked_dir.mkdir()
    processed_dir.mkdir()
    bboxes, masked_records = [], []
    for index, frame in enumerate(inputs.frames):
        alpha = np.asarray(frame)[..., 3]
        if not is_valid_alpha(alpha):
            raise ValueError(f'Invalid foreground/background alpha after preprocessing: frame {index}')
        ys, xs = np.nonzero(alpha > 0)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
        bboxes.append(bbox)
        path = masked_dir / f'{index:02d}.png'
        frame.save(path)
        masked_records.append({'index': index, 'original_alpha_preserved': valid[index],
                               'foreground_bbox_xywh': bbox, 'size_wh': list(frame.size),
                               'masked_rgba_sha256': hashlib.sha256(frame.tobytes()).hexdigest(),
                               'masked_file': str(path.relative_to(output)), 'masked_sha256': digest(path)})
    crop_boxes = bboxes if pipeline.image_process.independent_cropping else [aggregate_bboxes(bboxes)] * 16
    inputs.frames = stage_timer(torch, durations, 'image_preprocessing',
        lambda: pipeline.image_process.process_images(inputs.frames))
    records = []
    for index, (frame, bbox) in enumerate(zip(inputs.frames, crop_boxes)):
        x, y, w, h = map(int, bbox)
        max_dim = max(w, h)
        pad_base = int(max_dim * pipeline.image_process.padding_ratio)
        px, py = pad_base + (max_dim - w) // 2, pad_base + (max_dim - h) // 2
        if frame.size != (w + 2 * px, h + 2 * py):
            raise ValueError('Official processed dimensions differ from recorded crop/pad map')
        path = processed_dir / f'{index:02d}.png'
        frame.save(path)
        records.append({**masked_records[index], 'crop_bbox_xywh': [x, y, w, h],
                        'padding_lrtb': [px, px, py, py], 'processed_size_wh': list(frame.size),
                        'source_to_processed_xy': [[1, 0, px - x], [0, 1, py - y], [0, 0, 1]],
                        'processed_file': str(path.relative_to(output)), 'processed_sha256': digest(path),
                        'processed_pixel_sha256': hashlib.sha256(frame.tobytes()).hexdigest()})
    info = {'independent_cropping': bool(pipeline.image_process.independent_cropping),
            'padding_ratio': float(pipeline.image_process.padding_ratio),
            'background_rgb_0_1': pipeline.image_process.bg_color.tolist(),
            'all_original_alpha_preserved': all(valid), 'records': records,
            'coordinate_scope': 'Source pixels to saved processed pixels only. Image encoders may resize internally; no camera calibration inferred.'}
    json_write(output / 'preprocessing.json', info)
    return info


def save_npz(path, **arrays):
    import numpy as np
    for key, value in arrays.items():
        if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
            raise ValueError(f'Unsafe or nonfinite array {key} for {path.name}')
    with path.with_suffix(path.suffix + '.tmp').open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    path.with_suffix(path.suffix + '.tmp').replace(path)


def code_provenance(repo):
    paths = ['actionmesh/pipeline.py', 'actionmesh/preprocessing/image_processor.py',
             'actionmesh/preprocessing/background_removal.py', 'actionmesh/model/temporal_autoencoder.py',
             'actionmesh/model/temporal_denoiser.py', 'actionmesh/scheduler/scheduler.py',
             'actionmesh/configs/actionmesh.yaml', 'actionmesh/configs/actionmesh_lowram.yaml']
    try:
        head = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], check=True, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(['git', '-C', str(repo), 'status', '--porcelain'], check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.SubprocessError):
        head, dirty = None, None
    return {'git_head': head, 'git_status_porcelain': dirty,
            'sha256': {path: digest(repo / path) for path in paths},
            'helpers_sha256': digest(Path(__file__).with_name('research_three_ideas.py'))}


def run_case(args, repo, torch, durations):
    import numpy as np
    from omegaconf import OmegaConf
    from actionmesh.preprocessing.mesh_processor import get_mesh_features

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    pipeline = make_pipeline(repo, torch)
    json_write(args.output / 'resolved-config.json', OmegaConf.to_container(pipeline.cfg, resolve=True))
    json_write(args.output / 'code-provenance.json', code_provenance(repo))
    inputs, frame_records = load_input(args.input, args.output, torch)
    json_write(args.output / 'inputs.json', {'uid': args.uid, 'records': frame_records,
        'frame_order': 'Natural filename order; exactly 16, no duplicates introduced',
        'frame_indices': list(range(16)), 'timesteps': list(range(16)),
        'time_units': 'Frame index, matching official directory loader; physical fps unknown'})
    stage_progress = []

    def mark(stage, **extra):
        stage_progress.append({'stage': stage, 'utc': utc_now(), **extra})
        json_write(args.output / 'progress.json', stage_progress)
        print(f'CENSUS uid={args.uid} seed={args.seed} stage={stage} {extra}', flush=True)

    with torch.inference_mode():
        prep = preprocessing(inputs, pipeline, torch, args.output, durations)
        mark('preprocessed', original_alpha_preserved=prep['all_original_alpha_preserved'])
        pipeline._load_image_to_3d()
        freeze_loaded(pipeline.image_to_3d_pipe)
        latent_bank, mesh_bank = stage_timer(torch, durations, 'stage0',
            lambda: pipeline.init_banks_from_anchor(inputs, args.seed))
        pipeline._unload_model('image_to_3d_pipe')
        anchor_latent, anchor_times = latent_bank.get_ordered()
        anchor_mesh = mesh_bank.get_ordered(device='cpu')[0][0]
        anchor_array = anchor_latent.detach().cpu().numpy()
        query = get_mesh_features(anchor_mesh, with_normals=True).cpu().numpy()
        prepared = {
            'schema_version': np.asarray(1, dtype=np.int64),
            'timesteps': inputs.timesteps.numpy(), 'anchor_latent': anchor_array,
            'anchor_timesteps': anchor_times.detach().cpu().numpy().astype(np.float32),
            'anchor_vertices': np.asarray(anchor_mesh.vertices, dtype=np.float32),
            'anchor_faces': np.asarray(anchor_mesh.faces, dtype=np.int64),
            'anchor_query_features': query,
            'query_vertex_ids': np.arange(len(anchor_mesh.vertices), dtype=np.int64),
            'seed': np.asarray(args.seed, dtype=np.int64),
        }
        save_npz(args.output / 'stage0.npz', **prepared)
        anchor_mesh.export(args.output / 'anchor.glb')
        mark('stage0_completed', vertices=len(anchor_mesh.vertices), faces=len(anchor_mesh.faces))

        pipeline._load_image_encoder()
        freeze_loaded(pipeline.image_encoder)
        context = stage_timer(torch, durations, 'image_encoder', lambda: pipeline.encode_all_frames(inputs))
        prepared['context'] = context.detach().cpu().numpy()
        pipeline._unload_model('image_encoder')
        save_npz(args.output / 'prepared.npz', **prepared)
        json_write(args.output / 'prepared.json', {
            'status': 'prepared', 'uid': args.uid, 'seed': args.seed, 'schema_version': 1,
            'cache_sha256': digest(args.output / 'prepared.npz'),
            'shapes': {key: list(value.shape) for key, value in prepared.items()},
            'stage0_steps': 100, 'stage1_steps_planned': 30, 'config': 'actionmesh_lowram',
            'dtype': 'float16 autocast, official FP32 temporal weight storage',
        })
        mark('prepared_cache_saved')

        def stage1_callback(step, total, window_idx, total_windows):
            mark('stage1_step', step=int(step), total=int(total), window=int(window_idx), windows=int(total_windows))

        pipeline._load_temporal_denoiser()
        freeze_loaded(pipeline.temporal_3D_denoiser)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            latent_bank = stage_timer(torch, durations, 'stage1', lambda: pipeline.generate_3d_latents(
                inputs, context=context, latent_bank=latent_bank, seed=args.seed, step_callback=stage1_callback))
        pipeline._unload_model('temporal_3D_denoiser')
        latents, times = latent_bank.get_ordered()
        latent_array = latents.detach().cpu().numpy()
        if latent_array.shape != (16, *pipeline._denoiser_latent_shape):
            raise ValueError(f'Unexpected final latent shape: {latent_array.shape}')
        if not np.array_equal(latent_array[0], anchor_array[0]):
            raise ValueError('Stage I changed fixed anchor latent')
        save_npz(args.output / 'denoised.npz', latents=latent_array,
                 timesteps=times.detach().cpu().numpy().astype(np.float32), seed=np.asarray(args.seed, dtype=np.int64))
        step_records = [r for r in stage_progress if r['stage'] == 'stage1_step']
        if len(step_records) != 30 or any(r['total'] != 30 or r['windows'] != 1 for r in step_records):
            raise ValueError('Expected one native 30-step denoising window')
        del context
        mark('stage1_completed')

        pipeline._load_temporal_vae()
        freeze_loaded(pipeline.temporal_3D_vae)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            mesh_bank = stage_timer(torch, durations, 'stage2',
                lambda: pipeline.generate_mesh_animation(latent_bank=latent_bank, mesh_bank=mesh_bank))
        pipeline._unload_model('temporal_3D_vae')
        mark('stage2_completed')

    meshes, output_times = mesh_bank.get_ordered(device='cpu')
    vertices = np.stack([np.asarray(mesh.vertices) for mesh in meshes]).astype(np.float32)
    faces = np.asarray(meshes[0].faces, dtype=np.int64)
    if vertices.shape != (16, len(anchor_mesh.vertices), 3) or not np.isfinite(vertices).all():
        raise ValueError('Expected 16 finite meshes preserving vertex count')
    if not all(np.array_equal(mesh.faces, faces) for mesh in meshes):
        raise ValueError('Output faces changed across frames')
    if not np.array_equal(vertices[0], prepared['anchor_vertices']):
        raise ValueError('Output frame 0 differs from saved anchor')
    if not np.array_equal(output_times.numpy(), inputs.timesteps.numpy()):
        raise ValueError('Output times do not match input frame times')
    save_npz(args.output / 'sequence.npz', vertices=vertices, faces=faces,
             timesteps=output_times.numpy(), frame_indices=np.arange(16, dtype=np.int64),
             query_vertex_ids=prepared['query_vertex_ids'])
    np.save(args.output / 'deformations_vertices.npy', vertices, allow_pickle=False)
    np.save(args.output / 'deformations_faces.npy', faces, allow_pickle=False)
    # Shared topology + arrays avoid exporting 16 redundant GLB files.
    outputs = ['stage0.npz', 'anchor.glb', 'prepared.npz', 'denoised.npz', 'sequence.npz',
               'deformations_vertices.npy', 'deformations_faces.npy', 'preprocessing.json', 'inputs.json']
    mark('outputs_saved')
    return {
        'status': 'completed', 'uid': args.uid, 'seed': args.seed, 'frames': 16,
        'input': str(args.input), 'output': str(args.output),
        'stage0_steps': 100, 'stage1_steps': 30, 'guidance_scale': 7.5,
        'config': 'actionmesh_lowram', 'dtype': 'float16 autocast; official lazy-loaded models',
        'vertices_per_frame': int(vertices.shape[1]), 'faces': int(len(faces)),
        'finite_geometry': True, 'finite_caches': True, 'fixed_topology': True,
        'anchor_latent_held_exactly': True, 'anchor_mesh_held_exactly': True,
        'original_alpha_preserved': prep['all_original_alpha_preserved'],
        'training': False, 'steering': False, 'native_stage1_and_stage2': True,
        'backprop': False, 'sha256': {path: digest(args.output / path) for path in outputs},
        'validation_scope': 'Native inference execution and finite fixed-topology outputs only. Vertex identity is decoder query index, not a claim of correct physical correspondence. No quality metric computed.',
        'rng': {'python_numpy_torch_cuda_seed': args.seed, 'stage0_generator_seed': args.seed,
                'stage1_generator_seed': args.seed, 'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                'bitwise_repeatability_claimed': False},
        'versions': {'torch': torch.__version__, 'cuda': torch.version.cuda, 'numpy': np.__version__},
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/root/rivermind-data/actionmesh-repro'))
    parser.add_argument('--input', type=Path, required=True, help='Directory containing exactly 16 PNG frames')
    parser.add_argument('--uid', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-index', default='0', help='Physical GPU index for telemetry')
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2**32:
        parser.error('seed must be in [0, 2**32) for NumPy compatibility')
    for key in ('root', 'input', 'output'):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    return args


def main(argv=None):
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    started = utc_now()
    json_write(args.output / 'command.json', {
        'argv': list(sys.argv) if argv is None else [str(Path(__file__)), *argv],
        'uid': args.uid, 'seed': args.seed, 'started_utc': started,
        'script_sha256': digest(Path(__file__).resolve()), 'offline': True,
        'scope': 'Native ActionMesh failure-census baseline; no formal quality claim',
    })
    monitor, durations = None, {}
    result = {'status': 'failed', 'uid': args.uid, 'seed': args.seed}
    try:
        repo, torch = setup(args.root)
        monitor = Resources(args.output, torch, args.gpu_index)
        monitor.start()
        result = run_case(args, repo, torch, durations)
    except Exception as exc:
        result.update(status='failed', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        result.update(started_utc=started, ended_utc=utc_now(), stage_seconds=durations)
        if monitor:
            result['resources'] = monitor.finish()
        json_write(args.output / 'report.json', result)


if __name__ == '__main__':
    main()
