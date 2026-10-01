"""Download pinned official weights via byte-identical domestic mirrors."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true', help='Write download manifest without network transfers')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifests = root / 'manifests'
    manifests.mkdir(exist_ok=True)
    (root / 'logs').mkdir(exist_ok=True)
    files = json.loads((root / 'weights-manifest.json').read_text())
    (manifests / 'weights.json').write_text(json.dumps(files, indent=2))
    entries = []
    for item in files:
        path = root / item['path']
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size == item['size'] and not path.with_name(path.name + '.aria2').exists():
            expected = item.get('sha256')
            if not expected:
                continue
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(block)
            if digest.hexdigest() == expected:
                continue
        url = item.get('mirror_url') or f"https://hf-mirror.com/{item['repo']}/resolve/{item['revision']}/{item['file']}"
        entries.extend([url, f' dir={path.parent}', f' out={path.name}'])
        if item.get('sha256'):
            entries.append(' checksum=sha-256=' + item['sha256'])
    download_list = manifests / 'aria2.txt'
    download_list.write_text('\n'.join(entries) + '\n')
    link = root / 'repo/pretrained_weights'
    if not link.exists() and not link.is_symlink():
        link.symlink_to('../weights', target_is_directory=True)
    print(f'{len(files)} required files; {sum(x["size"] for x in files) / 1e9:.2f} GB')
    if args.prepare_only:
        return
    if entries:
        aria2 = shutil.which('aria2c')
        if not aria2:
            raise SystemExit('Install aria2c first; no downloads started.')
        subprocess.run([aria2, '--input-file=' + str(download_list), '--continue=true', '--check-integrity=true', '--allow-overwrite=true', '--auto-file-renaming=false', '--disable-ipv6=true', '--max-concurrent-downloads=4', '--max-connection-per-server=8', '--split=8', '--min-split-size=4M', '--max-tries=3', '--retry-wait=10', '--connect-timeout=15', '--timeout=120', '--file-allocation=none'], check=True)
    subprocess.run([sys.executable, str(root / 'finish_inference.py'), '--verify-only'], check=True)

if __name__ == '__main__':
    main()
