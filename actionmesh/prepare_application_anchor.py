"""Create one static TripoSG anchor for image + text -> video + mesh -> 4D.

This only reconstructs the supplied image; it generates neither a video nor an
animation. It calls the same single-image components and defaults as official
ActionMesh stage 0 without constructing a 16-frame ActionMeshInput. All model
files must already be present under --root and match the local manifest.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_weights(root):
    root = root.resolve()
    manifest = root / 'weights-manifest.json'
    if not manifest.is_file():
        manifest = root / 'manifests/weights.json'
    records = json.loads(manifest.read_text())
    selected = [item for item in records if item['repo'] in ('VAST-AI/TripoSG', 'briaai/RMBG-1.4')]
    if {item['repo'] for item in selected} != {'VAST-AI/TripoSG', 'briaai/RMBG-1.4'}:
        raise ValueError('Manifest must include both TripoSG and RMBG')
    checked = []
    for item in selected:
        path = (root / item['path']).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f'Weight path escapes --root: {item["path"]}')
        if not path.is_file() or path.stat().st_size != item['size'] or path.with_name(path.name + '.aria2').exists():
            raise ValueError(f'Missing or incomplete local weight: {path}')
        expected = item.get('sha256')
        if not expected or sha256(path) != expected:
            raise ValueError(f'Missing or mismatched SHA256 for weight: {path}')
        checked.append({'path': item['path'], 'bytes': item['size'], 'sha256': expected,
                        'repo': item['repo'], 'revision': item['revision']})
    return {'manifest': str(manifest), 'manifest_sha256': sha256(manifest),
            'files': checked, 'network_downloads_enabled': False}


def validate_mesh(mesh):
    import numpy as np
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError('Static anchor has empty, malformed or nonfinite vertices')
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or not np.issubdtype(faces.dtype, np.integer):
        raise ValueError('Static anchor has empty or malformed triangle indices')
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError('Static anchor has out-of-range triangle indices')
    extents = np.ptp(vertices, axis=0)
    if float(extents.max()) <= 1e-6 or not np.isfinite(mesh.area) or mesh.area <= 0:
        raise ValueError('Static anchor has degenerate geometry')
    return {'vertices': int(len(vertices)), 'triangles': int(len(faces)),
            'finite_vertices': True, 'valid_triangle_indices': True,
            'bounds': mesh.bounds.tolist(), 'extents': extents.tolist(),
            'surface_area': float(mesh.area), 'watertight': bool(mesh.is_watertight),
            'validation_scope': 'Finite, nonempty, indexed triangle geometry and GLB roundtrip; does not establish reconstruction quality.'}


class StageRecorder:
    def __init__(self, torch, output):
        self.torch, self.output, self.stages = torch, output, []

    @contextmanager
    def stage(self, name):
        torch = self.torch
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        success = False
        try:
            yield
            torch.cuda.synchronize()
            success = True
        finally:
            self.stages.append({'name': name, 'status': 'complete' if success else 'failed',
                                'elapsed_seconds': time.monotonic() - started,
                                'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                                'peak_reserved_bytes': torch.cuda.max_memory_reserved()})
            write_json(self.output / 'stages.json', self.stages)


def run(args):
    import numpy as np
    from PIL import Image
    import torch
    import trimesh
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required; CPU fallback is not enabled')
    repo, out = args.root / 'repo', args.output
    sys.path.insert(0, str(repo / 'third_party/TripoSG'))
    sys.path.insert(0, str(repo))
    from actionmesh.external.triposg import TripoSGPipelinePlus
    from actionmesh.preprocessing.background_removal import BackgroundRemover
    from actionmesh.preprocessing.image_processor import ImagePreprocessor
    from actionmesh.preprocessing.mesh_processor import MeshPostprocessor
    from actionmesh.utils import force_memory_cleanup, load_config

    cfg = load_config('actionmesh_lowram.yaml', str(repo / 'actionmesh/configs'))
    parameters = {'stage_0_steps': int(cfg.model.image_to_3D_denoiser.num_inference_steps),
                  'guidance_scale': float(cfg.model.image_to_3D_denoiser.guidance_scale),
                  'face_decimation': int(cfg.model.mesh_process.face_decimation),
                  'floaters_threshold': float(cfg.model.mesh_process.floaters_threshold)}
    if parameters != {'stage_0_steps': 100, 'guidance_scale': 7.5,
                      'face_decimation': 40000, 'floaters_threshold': 0.02}:
        raise ValueError(f'Upstream defaults differ from the expected reproduction configuration: {parameters}')
    torch.manual_seed(args.seed)
    recorder = StageRecorder(torch, out)
    versions = {}
    for name in ('torch', 'torchvision', 'diffusers', 'transformers', 'trimesh', 'numpy', 'Pillow'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    write_json(out / 'environment.json', {'python': platform.python_version(), 'packages': versions,
                                         'gpu': torch.cuda.get_device_name(), 'torch_cuda': torch.version.cuda})
    sources = ('actionmesh/pipeline.py', 'actionmesh/external/triposg.py',
               'actionmesh/preprocessing/background_removal.py', 'actionmesh/preprocessing/image_processor.py',
               'actionmesh/preprocessing/mesh_processor.py', 'actionmesh/configs/actionmesh.yaml')
    write_json(out / 'config.json', {**parameters, 'seed': args.seed, 'triposg_precision': 'float16',
                                   'background_removal_precision': 'official float32 default',
                                   'single_image_count': 1, 'temporal_generation': False,
                                   'upstream_sources': {name: sha256(repo / name) for name in sources},
                                   'coordinate_policy': 'Preserve native TripoSG vertices and official stage-0 mesh postprocessing; no recentering, scaling or axis remapping here. ActionMesh mesh-input backend normalizes and restores its input coordinates.',
                                   'memory_policy': 'RMBG unloaded before loading TripoSG',
                                   'condition_image': 'condition.png is the same white-background crop fed to TripoSG; it may also condition image-to-video.'})
    with Image.open(args.image) as image:
        image = image.convert('RGBA')
    with torch.inference_mode():
        with recorder.stage('background_removal_load_and_process'):
            remover = BackgroundRemover(str(args.root / 'weights/RMBG')).eval().to('cuda')
            alpha_was_valid = remover._has_a_valid_alpha_mask(image)
            foreground = remover.process_image(image)
            foreground.save(out / 'foreground.png')
        del remover
        force_memory_cleanup()
        with recorder.stage('official_image_preprocessing'):
            condition = ImagePreprocessor().process_images([foreground])[0]
            condition.save(out / 'condition.png')
        with recorder.stage('triposg_model_load'):
            pipe = TripoSGPipelinePlus.from_pretrained(str(args.root / 'weights/TripoSG'), local_files_only=True).to(torch.float16)
            pipe.to('cuda')
        with recorder.stage('triposg_100_step_generation_and_extraction'):
            latent, mesh = pipe(image=condition,
                                generator=torch.Generator(device=pipe.device).manual_seed(args.seed),
                                num_inference_steps=parameters['stage_0_steps'],
                                guidance_scale=parameters['guidance_scale'])
            raw_geometry = validate_mesh(mesh)
        del latent, pipe
        force_memory_cleanup()
        with recorder.stage('official_mesh_postprocessing'):
            mesh = MeshPostprocessor(face_decimation=parameters['face_decimation'],
                                     floaters_threshold=parameters['floaters_threshold']).process_mesh(mesh, seed=args.seed)
            geometry = validate_mesh(mesh)
        with recorder.stage('glb_export_and_roundtrip_validation'):
            anchor = out / 'anchor.glb'
            mesh.export(str(anchor), file_type='glb')
            restored = trimesh.load(str(anchor), force='mesh', process=False)
            validate_mesh(restored)
            if not np.array_equal(mesh.faces, restored.faces) or not np.allclose(mesh.vertices, restored.vertices, rtol=0, atol=1e-6):
                raise ValueError('GLB roundtrip changed anchor topology or coordinates')
    report = {'status': 'verified_static_anchor', 'animation_complete': False,
              'source_image': str(args.image), 'source_sha256': sha256(args.image),
              'input_had_valid_alpha': alpha_was_valid, 'seed': args.seed, 'parameters': parameters,
              'raw_geometry': raw_geometry, 'geometry': geometry,
              'anchor_sha256': sha256(anchor), 'condition_sha256': sha256(out / 'condition.png'),
              'elapsed_gpu_stages_seconds': sum(stage['elapsed_seconds'] for stage in recorder.stages),
              'peak_allocated_bytes': max(stage['peak_allocated_bytes'] for stage in recorder.stages),
              'peak_reserved_bytes': max(stage['peak_reserved_bytes'] for stage in recorder.stages),
              'timing_scope': 'Stages cover model loading, image processing, static generation, mesh processing and export; exclude weight hashing and imports.',
              'next_step': 'Generate a real motion video from condition.png plus the action prompt, then pass that video and anchor.glb to run_application_backend.py --mesh.'}
    write_json(out / 'report.json', report)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(os.environ.get('ACTIONMESH_WORKDIR', Path(__file__).resolve().parent)))
    parser.add_argument('--image', required=True, type=Path, help='One input image, not a video frame sequence')
    parser.add_argument('--output', required=True, type=Path, help='New output directory; existing paths are rejected')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    for name in ('root', 'image', 'output'):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


def main(argv=None):
    args = parse_args(argv)
    if not args.image.is_file():
        raise FileNotFoundError(args.image)
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {args.output}')
    if not (args.root / 'repo/actionmesh/pipeline.py').is_file():
        raise FileNotFoundError('ActionMesh source is missing under --root/repo')
    busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if busy:
        raise RuntimeError(f'GPU already has compute processes: {busy}')
    os.environ.update({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1'})
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        write_json(out / 'status.json', {'status': 'verifying_local_weights', 'animation_complete': False})
        shutil.copy2(args.image, out / ('source-image' + args.image.suffix.lower()))
        write_json(out / 'input.json', {'source': str(args.image), 'sha256': sha256(args.image), 'images': 1})
        write_json(out / 'weights-verified.json', verify_weights(args.root))
        write_json(out / 'command.json', {'argv': sys.argv, 'python': sys.executable, 'cwd': str(Path.cwd()), 'offline': True})
        write_json(out / 'status.json', {'status': 'generating_static_anchor', 'animation_complete': False})
        run(args)
        write_json(out / 'status.json', {'status': 'verified_static_anchor', 'animation_complete': False,
                                        'elapsed_total_seconds': time.monotonic() - started})
    except BaseException as exc:
        write_json(out / 'status.json', {'status': 'failed', 'animation_complete': False,
                                        'elapsed_total_seconds': time.monotonic() - started,
                                        'error': f'{type(exc).__name__}: {exc}', 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
