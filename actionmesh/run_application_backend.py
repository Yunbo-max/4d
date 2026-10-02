"""Run the ActionMesh backend on an actually generated video, offline on CUDA.

This is the video-to-4D half of an application, not a text/image video generator.
Exactly 16 real video frames are selected uniformly, including both endpoints.
This explicit preprocessing differs from the official loader's first-31 policy.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import runpy
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


def select_frame_indices(total, count=16):
    if total < count:
        raise ValueError(f'Need at least {count} decoded frames; got {total}. Frames will not be duplicated.')
    return [round(index * (total - 1) / (count - 1)) for index in range(count)]


def prepare_frames(video, out):
    import cv2
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f'Cannot open video: {video}')
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total = 0
    try:
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            total += 1
    finally:
        cap.release()
    indices = select_frame_indices(total)
    frames_dir = out / 'input-frames'
    frames_dir.mkdir()
    chosen = dict(zip(indices, range(len(indices))))
    cap = cv2.VideoCapture(str(video))
    saved = []
    try:
        for index in range(total):
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f'Video changed or failed to decode on second pass at frame {index}')
            if index in chosen:
                path = frames_dir / f'{chosen[index]:02d}.png'
                if not cv2.imwrite(str(path), frame):
                    raise IOError(f'Could not save input frame: {path}')
                saved.append({'file': str(path.relative_to(out)), 'source_frame_index': index,
                              'timestamp_seconds': index / fps if math.isfinite(fps) and fps > 0 else None,
                              'width': int(frame.shape[1]), 'height': int(frame.shape[0]), 'sha256': sha256(path)})
    finally:
        cap.release()
    if len(saved) != 16:
        raise ValueError(f'Expected 16 selected frames, got {len(saved)}')
    metadata = {'source_video': str(video), 'source_sha256': sha256(video),
                'source_decoded_frames': total, 'source_fps': fps if math.isfinite(fps) else None,
                'selected_frames': 16, 'frame_selection': 'uniformly spaced real decoded frame indices, including first and last',
                'preprocessing_is_official_default': False,
                'note': 'No frame duplication or interpolation. Official ActionMesh consumes these PNGs with sequential timesteps 0..15; source timestamps are recorded but are not passed to the model.',
                'frames': saved}
    write_json(out / 'input.json', metadata)
    shutil.copy2(video, out / ('source-video' + video.suffix.lower()))
    return metadata


def verify_local_weights(root):
    manifest = root / 'weights-manifest.json'
    if not manifest.is_file():
        manifest = root / 'manifests/weights.json'
    if not manifest.is_file():
        raise FileNotFoundError('Local weights manifest is required; no downloads will be attempted.')
    checked = []
    for item in json.loads(manifest.read_text()):
        path = root / item['path']
        if not path.is_file() or path.stat().st_size != item['size'] or path.with_name(path.name + '.aria2').exists():
            raise ValueError(f'Missing or incomplete local weight: {path}')
        expected = item.get('sha256')
        if not expected or sha256(path) != expected:
            raise ValueError(f'Missing or mismatched SHA256 for local weight: {path}')
        checked.append({'path': item['path'], 'bytes': item['size'], 'sha256': expected})
    return {'manifest': str(manifest), 'files': checked, 'network_downloads_enabled': False}


def validate_geometry(out, expected_frames=16, mesh=None):
    import numpy as np
    import trimesh
    vertices = np.load(out / 'deformations_vertices.npy', allow_pickle=False)
    faces = np.load(out / 'deformations_faces.npy', allow_pickle=False)
    if vertices.ndim != 3 or vertices.shape[0] != expected_frames or vertices.shape[1] == 0 or vertices.shape[2] != 3 or not np.isfinite(vertices).all():
        raise ValueError('Invalid, nonfinite, or empty deformation vertices')
    if faces.ndim != 2 or faces.shape[0] == 0 or faces.shape[1] != 3 or not np.issubdtype(faces.dtype, np.integer) or faces.min() < 0 or faces.max() >= vertices.shape[1]:
        raise ValueError('Invalid deformation faces')
    files = sorted(out.glob('mesh_*.glb'))
    if len(files) != expected_frames:
        raise ValueError(f'Expected {expected_frames} per-frame GLBs, found {len(files)}')
    for path in files:
        generated = trimesh.load(str(path), force='mesh', process=False)
        if not np.isfinite(generated.vertices).all() or len(generated.vertices) != vertices.shape[1] or not np.array_equal(generated.faces, faces):
            raise ValueError(f'Per-frame mesh has invalid vertices or different topology: {path}')
    motion = float(np.linalg.norm(vertices - vertices[:1], axis=-1).max())
    if motion <= 1e-6:
        raise ValueError('No detectable mesh motion')
    result = {'frames': int(vertices.shape[0]), 'vertices_per_frame': int(vertices.shape[1]),
              'triangles_per_frame': int(len(faces)), 'finite_vertices': True, 'fixed_topology': True,
              'max_vertex_distance_from_frame0': motion,
              'validation_scope': 'File integrity, finite geometry, fixed topology and nonzero motion. Does not establish prompt fidelity or reconstruction quality.'}
    if mesh:
        from actionmesh.io.mesh_io import load_glb
        anchor = load_glb(str(mesh))
        preserved = len(anchor.vertices) == vertices.shape[1] and np.array_equal(anchor.faces, faces)
        if not preserved:
            raise ValueError('Input mesh topology was not preserved')
        result['input_topology_preserved'] = True
    return result


def child(args):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required')
    repo, out = args.root / 'repo', args.output
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    # Prevent missing local assets from silently becoming Hub downloads.
    import actionmesh.utils
    def require_local(repo_id, local_dir):
        path = Path(local_dir)
        if not path.is_dir() or not any(path.iterdir()):
            raise FileNotFoundError(f'Offline model missing: {path} ({repo_id})')
        return str(path)
    actionmesh.utils.download_if_missing = require_local
    script = repo / 'inference' / ('video_and_3d_to_animated_mesh.py' if args.mesh else 'video_to_animated_mesh.py')
    argv = [str(script), '--input', str(out / 'input-frames'), '--output_dir', str(out),
            '--low_ram', '--dtype', 'float16', '--seed', str(args.seed)]
    mesh = out / 'input-mesh.glb' if args.mesh else None
    if mesh:
        argv += ['--mesh_input', str(mesh)]
    write_json(out / 'command.json', {'cwd': str(repo), 'entrypoint_argv': argv,
                                     'execution': 'runpy.run_path of absolute entrypoint in CUDA child process',
                                     'python': sys.executable, 'offline': True, 'fast': False,
                                     'config': 'actionmesh_lowram.yaml', 'stage_0_steps': 100, 'stage_1_steps': 30})
    sys.argv = argv
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    try:
        runpy.run_path(str(script), run_name='__main__')
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        report = validate_geometry(out, mesh=mesh)
        report.update({'status': 'verified', 'backend': 'ActionMesh', 'seed': args.seed, 'precision': 'float16',
                       'low_ram': True, 'fast': False, 'stage_0_steps': 100, 'stage_1_steps': 30,
                       'gpu': torch.cuda.get_device_name(), 'elapsed_inference_seconds': elapsed,
                       'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                       'timing_scope': 'Official ActionMesh entrypoint including its optional PyTorch3D rendering. Excludes prior video generation, frame extraction, hash verification and Blender export.'})
        write_json(out / 'report.json', report)
    finally:
        write_json(out / 'runtime.json', {'elapsed_process_seconds': time.monotonic() - started,
                                         'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                                         'peak_reserved_bytes': torch.cuda.max_memory_reserved()})


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(os.environ.get('ACTIONMESH_WORKDIR', Path(__file__).resolve().parent)))
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--mesh', type=Path, help='Optional existing GLB to animate and retain its topology')
    parser.add_argument('--output', type=Path, required=True, help='New output directory; existing directory is rejected')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--timeout', type=int, default=3600, help='Maximum backend child runtime in seconds')
    parser.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    for name in ['root', 'video', 'mesh', 'output']:
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    return args


def main(argv=None):
    args = parse_args(argv)
    os.environ.update({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1'})
    if args.child:
        child(args)
        return
    if not args.video.is_file():
        raise FileNotFoundError(args.video)
    if args.mesh and (not args.mesh.is_file() or args.mesh.suffix.lower() != '.glb'):
        raise ValueError('--mesh must point to an existing GLB')
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite an existing output: {args.output}')
    if not (args.root / 'repo/inference/video_to_animated_mesh.py').is_file():
        raise FileNotFoundError('ActionMesh repo is missing under --root')
    busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if busy:
        raise RuntimeError(f'GPU already has compute processes: {busy}')
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        write_json(out / 'status.json', {'status': 'preparing'})
        prepare_frames(args.video, out)
        if args.mesh:
            shutil.copy2(args.mesh, out / 'input-mesh.glb')
            write_json(out / 'mesh-input.json', {'source': str(args.mesh), 'sha256': sha256(out / 'input-mesh.glb')})
        write_json(out / 'weights-verified.json', verify_local_weights(args.root))
        versions = {}
        for name in ['torch', 'torchvision', 'pytorch3d', 'numpy', 'trimesh', 'diffusers', 'transformers']:
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        write_json(out / 'environment.json', {'python': platform.python_version(), 'packages': versions,
                    'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], text=True).strip()})
        cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--child', '--root', str(args.root),
               '--video', str(args.video), '--output', str(out), '--seed', str(args.seed)]
        if args.mesh:
            cmd += ['--mesh', str(args.mesh)]
        write_json(out / 'runner-command.json', {'argv': cmd, 'timeout_seconds': args.timeout})
        write_json(out / 'status.json', {'status': 'running'})
        with (out / 'gpu-memory.csv').open('w') as gpu, (out / 'inference.log').open('w') as log:
            sampler = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,index,memory.used,utilization.gpu', '--format=csv,nounits', '-l', '1'], stdout=gpu)
            try:
                result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
            finally:
                sampler.terminate()
                try:
                    sampler.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    sampler.kill()
                    sampler.wait()
        if result.returncode:
            raise RuntimeError(f'ActionMesh exited with code {result.returncode}; see {out / "inference.log"}')
        report = json.loads((out / 'report.json').read_text())
        with (out / 'gpu-memory.csv').open() as stream:
            rows = list(csv.reader(stream))[1:]
        samples = [int(row[2].strip()) for row in rows if len(row) == 4 and row[1].strip() == '0']
        report['nvidia_smi_peak_mib'] = max(samples) if samples else None
        report['nvidia_smi_scope'] = 'Physical GPU 0 total memory sampled every second; torch metrics apply to the inference CUDA process.'
        write_json(out / 'report.json', report)
        write_json(out / 'status.json', {'status': 'verified', 'exit_code': 0, 'elapsed_backend_total_seconds': time.monotonic() - started})
        print(json.dumps(report), flush=True)
    except BaseException as exc:
        write_json(out / 'status.json', {'status': 'failed', 'error': str(exc), 'error_type': type(exc).__name__, 'elapsed_backend_total_seconds': time.monotonic() - started})
        (out / 'runner-error.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
