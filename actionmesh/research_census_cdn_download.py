"""Download pinned public dataset files from resolved CDN URLs, never log queries."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

p = argparse.ArgumentParser()
p.add_argument('--urls', type=Path, required=True)
p.add_argument('--destination', type=Path, required=True)
p.add_argument('--report', type=Path, required=True)
p.add_argument('--workers', type=int, default=4)
a = p.parse_args()
if a.report.exists():
    raise FileExistsError('Download report already exists; reconcile instead of duplicating')
m = json.loads(a.urls.read_text())
records = []
started = time.monotonic()
def save(status):
    tmp = a.report.with_suffix('.tmp')
    tmp.write_text(json.dumps({'status': status, 'dataset': m['dataset'], 'revision': m['revision'],
        'expected_files': len(m['files']), 'seconds': time.monotonic() - started, 'files': records}, indent=2) + '\n')
    tmp.replace(a.report)
def download(f):
    row = {k: f[k] for k in ('path', 'size', 'sha256', 'hostname') if k in f}
    begin = time.monotonic()
    try:
        relative = Path(f['path'])
        if relative.is_absolute() or '..' in relative.parts or relative.parts[0] != 'data':
            raise ValueError('Invalid frozen path')
        hostname = urlsplit(f['url']).hostname
        if hostname != 'us.aws.cdn.hf.co':
            row['status'] = 'non_cdn_source'; return row
        target = a.destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size == f['size'] and hashlib.sha256(target.read_bytes()).hexdigest() == f['sha256']:
            row['status'] = 'already_verified'; return row
        temporary = target.with_suffix(target.suffix + '.cdn-part')
        h, size = hashlib.sha256(), 0
        with urlopen(Request(f['url']), timeout=45) as response, temporary.open('wb') as out:
            while True:
                if time.monotonic() - begin > 300:
                    raise TimeoutError('Per-file300second budget exhausted')
                block = response.read(1024 * 1024)
                if not block: break
                out.write(block); h.update(block); size += len(block)
        row.update(bytes=size, measured_sha256=h.hexdigest())
        if size != f['size'] or h.hexdigest() != f['sha256']:
            raise ValueError('Frozen size/hash mismatch')
        temporary.replace(target)
        row['status'] = 'verified'
    except Exception as e:
        # Error strings may contain public signed URLs; retain type/status only.
        row.update(status='failed', error_type=type(e).__name__, http_status=getattr(e, 'code', None))
    row['seconds'] = time.monotonic() - begin
    return row
save('running')
with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
    pending = [pool.submit(download, f) for f in m['files']]
    for done in concurrent.futures.as_completed(pending):
        row = done.result(); records.append(row); save('running')
        print(json.dumps({k: row.get(k) for k in ('path', 'status', 'bytes', 'seconds', 'error_type')}), flush=True)
ok = all(r['status'] in ('verified', 'already_verified') for r in records)
save('completed' if ok else 'partial_or_failed')
raise SystemExit(0 if ok else 1)
