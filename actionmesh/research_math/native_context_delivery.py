"""Receipt-bound raw output collection for the paired native instrument.

No model loads or score acceptance. The archive is a transport envelope; its
manifest proves file identity only. Failed/terminated attempts keep their raw
workspace and are never reconstructed into successful bundles.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tarfile


INPUT_NAMES = ('contract', 'population', 'snapshot_contract', 'snapshot_admission',
               'dataset_semantics', 'unit_manifest', 'environment')
OUTPUT_PREFIX = 'actionmesh/context-output'


def receipt_output_paths():
    return [OUTPUT_PREFIX + '/' + name for name in
            ('result.json', 'raw-manifest.json', 'raw-evidence.tar')]


def required_raw_paths(*, source_time_query):
    paths = ['generation-identity.json', 'paired-comparison.json',
             'revalidated-unit-manifest.json', 'final-integrity/revalidated-unit-manifest.json',
             'device-samples.jsonl', 'host-samples.jsonl']
    paths += ['inputs/' + name + '.json' for name in INPUT_NAMES]
    paths.append('inputs/dependencies.json')
    for stage in ('unobserved', 'observed', 'replay'):
        paths += ['requests/' + stage + '.json', stage + '/stage-result.json']
        paths += [stage + suffix for suffix in ('.stdout.log', '.stderr.log', '.execution.json')]
    for stage in ('unobserved', 'observed'):
        paths += [stage + '/' + name for name in
                  ('deformations_vertices.npy', 'deformations_faces.npy', 'sequence.npz', 'report.json')]
        paths += [stage + '/mesh_' + format(i, '02d') + '.glb' for i in range(16)]
    paths += ['capture/identity.json', 'capture/index.json',
              'capture/window-0000/record.json', 'capture/window-0000/meshes.safetensors',
              'capture/window-0000/decoder/identity.json']
    paths += ['capture/window-0000/decoder/call-0000/' + name for name in
              ('record.json', 'inputs.safetensors', 'tensors.safetensors')]
    paths += ['replay/window-0000/report.json', 'replay/window-0000/replay.safetensors']
    if source_time_query:
        paths.append('replay/window-0000/source-time.safetensors')
    return paths


def _hash(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def collection_snapshot(output):
    """Boundary observations, not an exact peak; returned into result.json."""
    import psutil
    root = Path(output)
    return {'output_bytes': sum(p.stat().st_size for p in root.rglob('*')
                                if p.is_file() and not p.is_symlink()),
            'free_disk_bytes': shutil.disk_usage(root).free,
            'collector_rss_bytes': psutil.Process(os.getpid()).memory_info().rss,
            'exact_peak': False}


def _inventory(root):
    rows = []
    excluded = {'result.json', 'raw-manifest.json', 'raw-evidence.tar'}
    for path in sorted(root.rglob('*')):
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise ValueError('Physical regular raw evidence required: ' + str(path))
        relative = path.relative_to(root).as_posix()
        if relative not in excluded:
            rows.append({'path': relative, 'bytes': path.stat().st_size, 'sha256': _hash(path)})
    return rows


def finalize_bundle(output, *, source_time_query):
    """Bind all raw files after writers/telemetry stop; no overwrites or links."""
    root = Path(output)
    if root.is_symlink() or any(p.is_symlink() for p in root.parents) or not root.is_dir():
        raise ValueError('Physical raw evidence directory required')
    archive_path, manifest_path = root / 'raw-evidence.tar', root / 'raw-manifest.json'
    if archive_path.exists() or archive_path.is_symlink() or manifest_path.exists() or manifest_path.is_symlink():
        raise FileExistsError('Preserve existing paired-instrument bundle')
    rows = _inventory(root)
    missing = set(required_raw_paths(source_time_query=source_time_query)) - {row['path'] for row in rows}
    if missing:
        raise ValueError('Missing raw instrument output: ' + ', '.join(sorted(missing)))
    # A failed archive stays partial evidence and cannot be silently overwritten.
    with tarfile.open(archive_path, mode='x:') as archive:
        for row in rows:
            path = root / row['path']
            info = archive.gettarinfo(str(path), arcname=row['path'])
            if not info.isfile() or info.size != row['bytes']:
                raise ValueError('Raw evidence changed during collection')
            info.uid = info.gid = 0
            info.uname = info.gname = ''
            info.mtime = 0
            info.mode = 0o600
            with path.open('rb') as stream:
                archive.addfile(info, stream)
    # Verify archive members themselves, not merely the files used to create it.
    with tarfile.open(archive_path, mode='r:') as archive:
        members = archive.getmembers()
        if [m.name for m in members] != [row['path'] for row in rows]:
            raise ValueError('Raw archive member inventory differs')
        for member, row in zip(members, rows):
            if not member.isfile() or member.size != row['bytes']:
                raise ValueError('Raw archive member type or size differs')
            value = hashlib.sha256()
            with archive.extractfile(member) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    value.update(block)
            if value.hexdigest() != row['sha256']:
                raise ValueError('Raw archive member hash differs: ' + row['path'])
    if _inventory(root) != rows:
        raise ValueError('Raw evidence changed across collection')
    archive_ref = {'path': archive_path.name, 'bytes': archive_path.stat().st_size,
                   'sha256': _hash(archive_path)}
    manifest = {'kind': 'paired-native-context-raw-bundle', 'version': 1,
                'files': rows, 'archive': archive_ref,
                'source_time_query': bool(source_time_query),
                'native_context_qualified': False, 'scientific_effect_qualification': False,
                'scope': 'Exact raw bytes; result.json is separately receipt-bound'}
    with manifest_path.open('x') as stream:
        stream.write(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    return {'archive': archive_ref,
            'manifest': {'path': manifest_path.name, 'sha256': _hash(manifest_path)},
            'raw_file_count': len(rows)}
