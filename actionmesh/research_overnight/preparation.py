"""Prepare only published native assets; model weights remain the existing cache."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

from .engine import atomic_json, digest, execute, fingerprint
from .protocol import ACTIONMESH, SALT, cohort_manifest, data_records, verify_records, verify_source


def find_root(explicit: Path | None = None) -> Path:
    if explicit:
        candidates = [explicit]
    elif os.environ.get('ACTIONMESH_WORKDIR'):
        candidates = [Path(os.environ['ACTIONMESH_WORKDIR'])]
    else:
        candidates = [Path('/root/rivermind-data/actionmesh-repro'), ACTIONMESH]
    for candidate in candidates:
        path = candidate.expanduser().resolve()
        if (path / 'repo/actionmesh/pipeline.py').is_file() and (path / 'repo/pretrained_weights').is_dir():
            return path
    raise FileNotFoundError('Existing ActionMesh asset root not found. Set ACTIONMESH_WORKDIR or --root to the directory containing repo/ and the cached pretrained_weights. No new model download is started.')


def prepare_generation(root: Path, output: Path, count: int, data_root: Path | None, offline: bool) -> dict:
    verify_source(root)
    target = (data_root or root / 'data/actionbench-overnight-20261005').expanduser().resolve()
    manifest = cohort_manifest(count)
    output.mkdir(parents=True, exist_ok=True)
    saved = output / 'prepared-generation.json'
    if saved.exists():
        previous = json.loads(saved.read_text())
        if previous['cohort'] != manifest or previous['data_root'] != str(target):
            raise ValueError('Prepared cohort/data path changed; use a separate output directory')
        verify_records(previous['data_records'])
        verify_records(previous['weight_records'])
        return previous
    from .resources import freeze_weight_records
    weights = freeze_weight_records(root)
    atomic_json(output / 'cohort.json', manifest)
    required = [target / 'data' / uid / name for uid in manifest['uids']
                for name in ['camera.json', 'surfaces.npy', *[f'imgs/{i:02d}.png' for i in range(16)]]]
    missing = [p for p in required if not p.is_file() or p.with_name(p.name + '.aria2').exists()]
    download_records = []
    if missing:
        if offline:
            raise FileNotFoundError(f'{len(missing)} native data files missing; run prepare without --offline before 23:00')
        free = shutil.disk_usage(target.parent if target.parent.exists() else root).free
        if free < 2 * 1024**3:
            raise RuntimeError('Need at least 2 GiB free disk for the pinned native cohort')
        code = "from huggingface_hub import snapshot_download;import json,sys;snapshot_download(repo_id=sys.argv[1],repo_type='dataset',revision=sys.argv[2],local_dir=sys.argv[3],allow_patterns=json.loads(sys.argv[4]),max_workers=2)"
        patterns = [f'data/{uid}/*' for uid in manifest['uids']]
        preferred = os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com')
        endpoints = list(dict.fromkeys([preferred, 'https://huggingface.co']))
        for i, endpoint in enumerate(endpoints):
            env = os.environ.copy()
            for key in ('HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE'):
                env.pop(key, None)
            env.update(HF_ENDPOINT=endpoint, HF_HUB_DISABLE_XET='1', HF_HUB_ETAG_TIMEOUT='20', HF_HUB_DOWNLOAD_TIMEOUT='60')
            last = [0.0]
            def notice():
                if time.monotonic() - last[0] >= 30:
                    print('Preparing pinned ActionBench inputs; see ' + str(output / f'prepare-download-{i}.log'), flush=True)
                    last[0] = time.monotonic()
            result = execute([sys.executable, '-u', '-c', code, manifest['dataset'], manifest['revision'], str(target), json.dumps(patterns)],
                             output / f'prepare-download-{i}.log', 900, env=env, heartbeat=notice)
            download_records.append({'endpoint': endpoint, **result})
            if result['status'] == 'completed':
                break
        else:
            atomic_json(output / 'prepare-failure.json', {'downloads': download_records, 'missing_files': [str(p) for p in missing]})
            raise RuntimeError('Native data preparation failed; retained download logs. No GPU experiment started.')
    records = data_records(target, manifest)
    prepared = {'cohort': manifest, 'data_root': str(target), 'data_records': records,
                'weight_records': weights,
                'download_records': download_records, 'prepared_at_utc': __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()}
    atomic_json(saved, prepared)
    return prepared


def prepare_perception(qa_json: Path, video_root: Path, model_path: Path, count: int, output: Path) -> dict:
    from .perception import public_question
    qa_json, video_root, model_path = (p.expanduser().resolve() for p in (qa_json, video_root, model_path))
    rows = json.loads(qa_json.read_text())
    if not isinstance(rows, dict) or not rows:
        raise ValueError('Require the published 4D-Bench QA object keyed by question ID')
    groups = {}
    categories = set()
    for question_id, row in rows.items():
        _, _, category = public_question(row)
        groups.setdefault(question_id.split('_')[0], []).append(question_id)
        categories.add(category)
    uids = sorted(groups, key=lambda uid: hashlib.sha256(f'{SALT}:qa:{uid}'.encode()).hexdigest())
    chosen = []
    for uid in uids:
        question_ids = sorted(groups[uid])
        if len(chosen) + len(question_ids) > count:
            continue  # Never cut an object group to meet an arbitrary count.
        chosen.extend(question_ids)
        if len(chosen) == count:
            break
    if not chosen:
        raise ValueError('Question limit is too small for a complete native object group')
    config = json.loads((model_path / 'config.json').read_text())
    if config.get('model_type') != 'qwen2_vl':
        raise ValueError('The native QA adapter requires a local Qwen2-VL checkpoint')
    # Pin local checkpoint bytes and expose provenance rather than silently
    # substituting a smaller/quantized model when resources are tight.
    model_files = sorted(p for p in model_path.rglob('*') if p.is_file() and '.cache' not in p.parts)
    weights = [p for p in model_files if p.suffix == '.safetensors']
    if not weights or sum(p.stat().st_size for p in weights) < 13_000_000_000:
        raise ValueError('Require the unquantized Qwen2-VL-7B checkpoint; smaller/quantized substitutions are not enabled')
    input_paths = [qa_json]
    for uid in sorted({q.split('_')[0] for q in chosen}):
        input_paths.extend(video_root / uid / f'view_{view}_rgb_white_bg.mp4' for view in (1, 8, 16))
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError('Missing published QA input: ' + str(path))
    prepared = {'benchmark': '4D-Bench', 'source_code_commit': 'f40f49a7539c4c5b1485ad5e80370cf006bbaf53',
                'dataset_revision': 'c3b799ddac9c21db0690ad964c1e82a42c892744',
                'dataset_identity_status': 'local bytes pinned; user must supply the published QA archive at the stated revision',
                'qa_json': str(qa_json), 'video_root': str(video_root), 'model_path': str(model_path),
                'question_ids': chosen, 'independent_objects': len({q.split('_')[0] for q in chosen}),
                'categories': sorted(categories), 'model_repo': 'Qwen/Qwen2-VL-7B-Instruct',
                'model_revision': model_path.name if model_path.parent.name == 'snapshots' else None,
                'input_records': {str(p): {'sha256': digest(p), 'bytes': p.stat().st_size} for p in input_paths},
                'model_records': {str(p): {'sha256': digest(p), 'bytes': p.stat().st_size} for p in model_files},
                'view_ids': [1, 8, 16], 'frames_per_view': 6, 'max_new_tokens': 128,
                'runtime': {'dtype': 'float16', 'attention': 'sdpa', 'do_sample': False, 'quantized': False},
                'scope': 'native-task baseline input-organization comparison, no new interaction method'}
    saved = output / 'prepared-perception.json'
    if saved.exists() and fingerprint(json.loads(saved.read_text())) != fingerprint(prepared):
        raise ValueError('Prepared QA inputs/model changed; use a new output directory')
    atomic_json(saved, prepared)
    return prepared
