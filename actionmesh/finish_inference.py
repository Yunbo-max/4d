"""Bounded remote continuation: verify official weights, run one official sample."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(os.environ.get('ACTIONMESH_WORKDIR', Path(__file__).resolve().parent)).resolve()
STATE = ROOT / 'manifests/inference-status.json'

def status(stage, **kwargs):
    data = {'stage': stage, 'updated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kwargs}
    temp = STATE.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(STATE)
    print(json.dumps(data), flush=True)

def verify_weights():
    result = []
    for item in json.loads((ROOT / 'manifests/weights.json').read_text()):
        p = ROOT / item['path']
        if not p.is_file() or p.stat().st_size != item['size'] or p.with_name(p.name + '.aria2').exists():
            raise RuntimeError('Incomplete download: ' + str(p))
        expected = item.get('sha256')
        if expected:
            h = hashlib.sha256()
            with p.open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    h.update(block)
            if h.hexdigest() != expected:
                raise RuntimeError('SHA256 mismatch: ' + str(p))
        result.append({'path': item['path'], 'bytes': p.stat().st_size, 'sha256_verified': bool(expected)})
    (ROOT / 'manifests/weights-verified.json').write_text(json.dumps(result, indent=2))
    return len(result)

def run_inference():
    import runpy
    import numpy as np
    import torch
    import trimesh
    os.chdir(ROOT / 'repo')
    out = ROOT / 'outputs' / time.strftime('kangaroo-lowram-fp16-%Y%m%dT%H%M%S', time.gmtime())
    out.mkdir(parents=True, exist_ok=False)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    sys.argv = ['inference/video_to_animated_mesh.py', '--input', 'assets/examples/kangaroo', '--output_dir', str(out), '--low_ram', '--dtype', 'float16', '--seed', '42']
    (out / 'command.json').write_text(json.dumps(sys.argv, indent=2))
    status('inference_running', output=str(out), command=sys.argv)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    try:
        runpy.run_path(str(ROOT / 'repo/inference/video_to_animated_mesh.py'), run_name='__main__')
        vertices = np.load(out / 'deformations_vertices.npy')
        faces = np.load(out / 'deformations_faces.npy')
        meshes = sorted(out.glob('mesh_*.glb'))
        assert len(meshes) == 16 and vertices.shape[0] == 16
        assert vertices.ndim == 3 and vertices.shape[-1] == 3 and vertices.shape[1] > 0
        assert np.isfinite(vertices).all() and faces.shape[0] > 0
        assert faces.min() >= 0 and faces.max() < vertices.shape[1]
        for p in meshes:
            mesh = trimesh.load(p, force='mesh')
            assert len(mesh.vertices) > 0 and len(mesh.faces) > 0
        motion = float(np.max(np.abs(vertices[1:] - vertices[:1])))
        assert motion > 1e-6, 'No detectable mesh motion'
        status('inference_verified', output=str(out), frames=len(meshes), vertices=int(vertices.shape[1]), faces=int(faces.shape[0]), max_vertex_displacement=motion)
    finally:
        stats = {'elapsed_seconds': time.monotonic() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
        (out / 'runtime.json').write_text(json.dumps(stats, indent=2))

def main():
    os.chdir(ROOT)
    (ROOT / 'manifests').mkdir(exist_ok=True)
    (ROOT / 'logs').mkdir(exist_ok=True)
    if '--verify-only' in sys.argv:
        print('Verified files:', verify_weights())
        return
    if '--run' in sys.argv:
        verify_weights()
        run_inference()
        return
    pid = int(sys.argv[1])
    deadline = time.monotonic() + 8 * 3600
    status('downloading', downloader_pid=pid)
    while Path('/proc/' + str(pid)).exists():
        if time.monotonic() > deadline:
            raise TimeoutError('Download wait exceeded 8 hours; no inference started')
        time.sleep(30)
    status('verifying_weights')
    count = verify_weights()
    status('weights_verified', files=count)
    busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if busy:
        raise RuntimeError('GPU has another process; inference was not started: ' + busy)
    gpu_log = (ROOT / 'logs/gpu-memory.csv').open('w')
    sampler = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,memory.used,utilization.gpu', '--format=csv', '-l', '1'], stdout=gpu_log)
    try:
        with (ROOT / 'logs/inference.log').open('w') as log:
            r = subprocess.run([str(ROOT / 'inference-env/bin/python'), '-u', __file__, '--run'], stdout=log, stderr=subprocess.STDOUT, timeout=7200)
        if r.returncode:
            raise RuntimeError('Inference failed, exit ' + str(r.returncode) + '; see logs/inference.log')
    finally:
        sampler.terminate()
        sampler.wait(timeout=10)
        gpu_log.close()

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        status('failed', error=str(exc))
        raise
