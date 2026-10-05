"""Observe the selected physical GPU and qualify the existing offline environment."""
from __future__ import annotations

import csv
import importlib.metadata
import io
import json
import os
from pathlib import Path
import subprocess
import time

from .engine import atomic_json, digest, utc
from .protocol import resource_admission, verify_records, verify_source


def gpu_snapshot(index: str) -> dict:
    output = subprocess.check_output(['nvidia-smi', '-i', index,
        '--query-gpu=index,uuid,name,memory.total,memory.free,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True, timeout=5)
    rows = list(csv.reader(io.StringIO(output)))
    if len(rows) != 1 or len(rows[0]) != 7:
        raise RuntimeError('Exactly one physical GPU must be selected')
    row = [x.strip() for x in rows[0]]
    raw_apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name',
                 '--format=csv,noheader,nounits'], text=True, timeout=5)
    applications = []
    for app in csv.reader(io.StringIO(raw_apps)):
        if len(app) >= 3 and app[0].strip() == row[1]:
            applications.append({'pid': int(app[1]), 'name': app[2].strip()})
    return {'physical_index': row[0], 'uuid': row[1], 'name': row[2],
            'total_mib': int(row[3]), 'free_mib': int(row[4]), 'used_mib': int(row[5]),
            'utilization_percent': int(row[6]), 'applications': applications, 'utc': utc().isoformat()}


def require_available(snapshot: dict, suite: str) -> None:
    limits = resource_admission(suite)
    minimum_free, minimum_total = limits['minimum_free_mib'], limits['minimum_total_mib']
    if snapshot['applications']:
        raise RuntimeError('Selected GPU has a live compute process; reconcile it before launch')
    if snapshot['total_mib'] < minimum_total or snapshot['free_mib'] < minimum_free:
        raise RuntimeError(f'Insufficient observed VRAM for the unchanged {suite} protocol: '
                           f"{snapshot['free_mib']} free / {snapshot['total_mib']} total MiB; need {minimum_free} free")


def verify_weights(root: Path) -> dict:
    manifest_path = root / 'manifests/weights.json'
    if not manifest_path.is_file():
        raise FileNotFoundError('Existing weight manifest missing: ' + str(manifest_path) + '. Use the existing download_weights.py preparation flow; the overnight runner does not download new model weights.')
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, list) or not manifest:
        raise ValueError('Require the existing nonempty weight manifest list')
    records = []
    for item in manifest:
        path = (root / item['path']).resolve()
        path.relative_to(root.resolve())
        if not path.is_file() or path.stat().st_size != item['size'] or path.with_name(path.name + '.aria2').exists():
            raise ValueError('Incomplete cached weight: ' + str(path))
        actual = digest(path)
        if item.get('sha256') and actual != item['sha256']:
            raise ValueError('Cached weight hash mismatch: ' + str(path))
        records.append({'path': str(path), 'bytes': path.stat().st_size, 'sha256': actual,
                        'matches_expected_sha256': bool(item.get('sha256'))})
    return {'manifest_sha256': digest(manifest_path), 'files': records}


def freeze_weight_records(root: Path) -> dict:
    verified = verify_weights(root)
    manifest = (root / 'manifests/weights.json').resolve()
    return {str(manifest): {'sha256': verified['manifest_sha256'], 'bytes': manifest.stat().st_size},
            **{item['path']: {'sha256': item['sha256'], 'bytes': item['bytes']} for item in verified['files']}}


def doctor(root: Path, suite: str, index: str, output: Path, protocol: dict | None = None) -> dict:
    record = {'status': 'blocked', 'checked_utc': utc().isoformat(), 'suite': suite,
              'scope': 'resource/source/cache/import readiness, not native execution or scientific validation'}
    try:
        snapshot = gpu_snapshot(index)
        record['gpu'] = snapshot
        record['resource_admission'] = resource_admission(suite)
        require_available(snapshot, suite)
        os.environ.update(CUDA_VISIBLE_DEVICES=snapshot['uuid'], HF_HUB_OFFLINE='1',
                          TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1')
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('The selected inference Python has no working CUDA torch')
        if torch.cuda.device_count() != 1:
            raise RuntimeError('The worker must expose exactly one selected CUDA device')
        record['compute_capability'] = list(torch.cuda.get_device_capability(0))
        record['torch_gpu_name'] = torch.cuda.get_device_name(0)
        record['versions'] = {}
        for package in ('torch', 'numpy', 'transformers', 'diffusers', 'accelerate', 'scipy', 'trimesh', 'pytorch3d', 'qwen-vl-utils', 'opencv-python'):
            try:
                record['versions'][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                record['versions'][package] = None
        if suite in ('generation', 'both'):
            record['native_sources'] = verify_source(root)
            record['weights'] = verify_weights(root)
            if protocol and protocol.get('generation'):
                verify_records(protocol['generation']['weight_records'])
            from research_census_eval import load_official
            _, qualification = load_official(root / 'repo')
            record['native_scorer_import'] = qualification
        if suite in ('perception', 'both'):
            import cv2
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            from qwen_vl_utils import process_vision_info
            if not protocol or not protocol.get('perception'):
                raise ValueError('Perception data/model must be prepared explicitly')
            verify_records(protocol['perception']['model_records'])
        record['status'] = 'ready_for_baseline_calibration'
    except Exception as exc:
        record['error'] = f'{type(exc).__name__}: {exc}'
    atomic_json(output, record)
    print(json.dumps({'readiness': record['status'], 'error': record.get('error'), 'report': str(output)}), flush=True)
    return record


class GPUMonitor:
    def __init__(self, index: str, output: Path):
        self.index, self.output, self.last_sample = index, output, 0.0

    def __call__(self):
        if time.monotonic() - self.last_sample < 10:
            return
        self.last_sample = time.monotonic()
        sample = gpu_snapshot(self.index)
        with self.output.open('a') as stream:
            stream.write(json.dumps(sample) + '\n')
        if sample['free_mib'] < 256:
            raise RuntimeError('GPU headroom fell below 256 MiB; stop rather than modify the native protocol')
