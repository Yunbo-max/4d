"""Reproduce supported input modes on one CUDA GPU; leave upstream code unchanged."""
import argparse
import hashlib
import json
import importlib.metadata
import platform
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
import traceback

ROOT = Path(os.environ.get('ACTIONMESH_WORKDIR', Path(__file__).resolve().parent)).resolve()
REPO = ROOT / 'repo'
SUITE = ROOT / 'outputs/input-modes-20261001'
BLENDER = ROOT / 'tools/blender-3.5.1-linux-x64/blender'


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def prepare():
    import cv2
    import numpy as np
    from PIL import Image
    from actionmesh.io.video_input import load_frames
    dest = SUITE / 'inputs'
    dest.mkdir(parents=True, exist_ok=True)
    for name in ['kangaroo', 'panda']:
        frames = sorted((REPO / 'assets/examples' / name).glob('*.png'))
        assert len(frames) == 16
        video = dest / f'{name}.mp4'
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 8, (512, 512))
        assert writer.isOpened()
        for path in frames:
            im = Image.open(path).convert('RGBA')
            assert im.size == (512, 512)
            bg = Image.new('RGBA', im.size, 'white')
            bg.alpha_composite(im)
            writer.write(cv2.cvtColor(np.array(bg.convert('RGB')), cv2.COLOR_RGB2BGR))
        writer.release()
        assert load_frames(video, max_frames=31).n_frames == 16
    single = dest / 'single-image'
    single.mkdir(exist_ok=True)
    import shutil
    shutil.copy2(REPO / 'assets/examples/kangaroo/00.png', single / '00.png')
    checks = []
    for name, value in [('text', 'a kangaroo boxing'), ('single_png', REPO / 'assets/examples/kangaroo/00.png'), ('one_frame_folder', single)]:
        try:
            load_frames(value, max_frames=31)
        except (ValueError, AssertionError) as exc:
            checks.append({'input': name, 'expected_rejection': True, 'error_type': type(exc).__name__, 'message': str(exc)})
        else:
            raise AssertionError('Unexpectedly accepted ' + name)
    write(SUITE / 'unsupported-inputs.json', checks)
    write(dest / 'manifest.json', {'source': 'Official ActionMesh example PNG sequences', 'frames': 16, 'fps': 8, 'size': [512, 512], 'encoding': 'OpenCV mp4v, RGBA composited on white; no masks passed to inference', 'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.glob('*.mp4')}})


def child(name):
    import numpy as np
    import torch
    import trimesh
    assert torch.cuda.is_available(), 'CUDA GPU required'
    out = SUITE / name
    mesh_mode = name == 'panda-video-mesh'
    script = REPO / 'inference' / ('video_and_3d_to_animated_mesh.py' if mesh_mode else 'video_to_animated_mesh.py')
    video = SUITE / 'inputs' / ('panda.mp4' if mesh_mode else 'kangaroo.mp4')
    args = [str(script), '--input', str(video), '--output_dir', str(out), '--low_ram', '--dtype', 'float16', '--seed', '42']
    if mesh_mode:
        args += ['--mesh_input', str(REPO / 'assets/examples/panda/panda.glb')]
    write(out / 'command.json', args)
    sys.argv = args
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    try:
        runpy.run_path(str(script), run_name='__main__')
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        v = np.load(out / 'deformations_vertices.npy')
        f = np.load(out / 'deformations_faces.npy')
        assert v.shape[0] == 16 and v.shape[2] == 3 and np.isfinite(v).all()
        assert f.ndim == 2 and f.shape[1] == 3 and f.min() >= 0 and f.max() < v.shape[1]
        assert len(list(out.glob('mesh_*.glb'))) == 16
        motion = float(np.linalg.norm(v - v[:1], axis=-1).max())
        assert motion > 1e-6
        report = {'status': 'verified', 'mode': name, 'frames': 16, 'vertices_per_frame': int(v.shape[1]), 'triangles_per_frame': len(f), 'max_vertex_distance_from_frame0': motion, 'finite_vertices': True, 'seed': 42, 'precision': 'float16', 'low_ram': True, 'fast': False, 'gpu': torch.cuda.get_device_name(), 'elapsed_inference_seconds': elapsed, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
        if mesh_mode:
            from actionmesh.io.mesh_io import load_glb
            anchor = load_glb(str(REPO / 'assets/examples/panda/panda.glb'))
            report['input_topology_preserved'] = bool(v.shape[1] == len(anchor.vertices) and np.array_equal(f, anchor.faces))
            assert report['input_topology_preserved']
        write(out / 'report.json', report)
    finally:
        write(out / 'runtime.json', {'elapsed_process_seconds': time.monotonic() - start, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--child', choices=['kangaroo-video', 'panda-video-mesh'])
    parser.add_argument('--retry', choices=['kangaroo-video', 'panda-video-mesh'], help='Retry one mode after preserving its failed output directory.')
    args = parser.parse_args()
    os.chdir(REPO)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    if args.child:
        child(args.child)
        return
    SUITE.mkdir(parents=True, exist_ok=bool(args.retry))
    busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    assert not busy, 'GPU already in use: ' + busy
    import torch
    versions = {}
    for package in ['torch', 'torchvision', 'pytorch3d', 'numpy', 'trimesh', 'diffusers']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    write(SUITE / 'environment.json', {
        'python': platform.python_version(), 'versions': versions, 'cuda': torch.version.cuda,
        'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], text=True).strip(),
        'note': 'Environment at most recent suite invocation. See failed-attempts for dependency failures; inference uses CUDA and Blender preview uses CPU.'
    })
    if not args.retry:
        prepare()
    statuses = json.loads((SUITE / 'status.json').read_text()) if args.retry else []
    if args.retry:
        statuses = [s for s in statuses if s['mode'] != args.retry]
        write(SUITE / 'status.json', statuses)
    for name in ([args.retry] if args.retry else ['kangaroo-video', 'panda-video-mesh']):
        out = SUITE / name
        out.mkdir()
        with (out / 'gpu-memory.csv').open('w') as gpu, (out / 'inference.log').open('w') as log:
            sampler = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,memory.used,utilization.gpu', '--format=csv', '-l', '1'], stdout=gpu)
            try:
                result = subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '--child', name], stdout=log, stderr=subprocess.STDOUT, timeout=3600)
            finally:
                sampler.terminate()
                sampler.wait(timeout=10)
        status = {'mode': name, 'exit_code': result.returncode}
        if result.returncode == 0:
            report = json.loads((out / 'report.json').read_text())
            rows = (out / 'gpu-memory.csv').read_text().splitlines()[1:]
            peak = max(int(r.split(',')[1].strip().split()[0]) for r in rows)
            report['nvidia_smi_peak_mib'] = peak
            write(out / 'report.json', report)
        statuses.append(status)
        write(SUITE / 'status.json', statuses)
        print(json.dumps(status), flush=True)
    if any(item['exit_code'] for item in statuses):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
