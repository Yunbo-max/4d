"""Download only a frozen public ActionBench cohort; retain attempts and hashes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('--manifest', type=Path, required=True)
p.add_argument('--root', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
m = json.loads(a.manifest.read_text())
dest = a.root / 'data/actionbench-census-20261002'
report = {'status': 'running', 'pid': os.getpid(), 'manifest_sha256': hashlib.sha256(a.manifest.read_bytes()).hexdigest(), 'attempts': [], 'files': []}
def save():
    q = a.output / 'download.json.tmp'
    q.write_text(json.dumps(report, indent=2) + '\n')
    q.replace(a.output / 'download.json')
save()
env = os.environ.copy()
env.update(HF_HUB_DISABLE_XET='1', HF_HUB_ETAG_TIMEOUT='20', HF_HUB_DOWNLOAD_TIMEOUT='60')
for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
    env.pop(key, None)
for index, endpoint in enumerate(('https://hf-mirror.com', 'https://huggingface.co')):
    env['HF_ENDPOINT'] = endpoint
    cmd = [str(a.root / 'inference-env/bin/hf'), 'download', m['dataset'],
           '--repo-type', 'dataset', '--revision', m['revision'], '--local-dir', str(dest),
           '--max-workers', '4', '--include', *[f'data/{u}/*' for u in m['uids']]]
    item = {'endpoint': endpoint, 'argv': cmd, 'status': 'running', 'timeout_seconds': 600}
    report['attempts'].append(item); save()
    started = time.monotonic()
    with (a.output / f'attempt-{index}.log').open('w') as log:
        try:
            x = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=600)
            item['exit_code'] = x.returncode
        except subprocess.TimeoutExpired:
            item['exit_code'] = 124
    item['seconds'] = time.monotonic() - started
    item['status'] = 'completed' if item['exit_code'] == 0 else 'failed'
    save()
    if item['exit_code'] == 0:
        break
ok = True
for f in m['files']:
    path = dest / f['path']
    row = {'path': f['path'], 'expected_bytes': f['size'], 'exists': path.is_file()}
    if path.is_file():
        row['bytes'] = path.stat().st_size
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for b in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                h.update(b)
        row['sha256'] = h.hexdigest()
        expected_sha = f.get('lfs', {}).get('oid')
        row['verified'] = row['bytes'] == row['expected_bytes'] and (expected_sha is None or expected_sha == row['sha256'])
    else:
        row['verified'] = False
    ok = ok and row['verified']; report['files'].append(row)
report['status'] = 'completed' if ok else 'failed'
report['destination'] = str(dest)
save()
print(json.dumps({'status': report['status'], 'files': len(report['files']), 'destination': str(dest)}), flush=True)
raise SystemExit(0 if ok else 1)
