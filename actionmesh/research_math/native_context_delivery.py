"""Receipt-bound raw output collection for the paired native instrument.

No model loads or score acceptance. The archive is a transport envelope; its
manifest proves file identity only. Failed/terminated attempts keep their raw
workspace and are never reconstructed into successful bundles.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tarfile
import tempfile


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


def _expected_hash(value, label):
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in '0123456789abcdef' for character in value)):
        raise ValueError('Canonical lowercase SHA-256 required: ' + label)
    return value


def _physical_file(path):
    path = Path(path)
    if (path.is_symlink() or any(parent.is_symlink() for parent in path.parents)
            or not path.is_file() or not stat.S_ISREG(path.stat().st_mode)):
        raise ValueError('Physical regular bundle file required: ' + str(path))
    return path


def _copy_exact_snapshot(source, target, expected_bytes):
    """Copy no more than the preflight size; reject growth or truncation races."""
    copied = 0
    with source.open('rb') as incoming, target.open('xb') as destination:
        while True:
            block = incoming.read(min(1024 * 1024, expected_bytes - copied + 1))
            if not block:
                break
            copied += len(block)
            if copied > expected_bytes:
                raise ValueError('Bundle input grew during bounded snapshot: ' + str(source))
            destination.write(block)
    if copied != expected_bytes:
        raise ValueError('Bundle input truncated during bounded snapshot: ' + str(source))


def _member_path(value):
    if not isinstance(value, str) or not value or '\\' in value:
        raise ValueError('Archive member must be a canonical relative POSIX path')
    path = PurePosixPath(value)
    if path.is_absolute() or value != path.as_posix() or any(part in ('', '.', '..') for part in path.parts):
        raise ValueError('Archive member must be a canonical relative POSIX path')
    return path


def _positive_limit(value, label):
    if type(value) is not int or value < 1:
        raise ValueError('Positive integer expansion limit required: ' + label)
    return value


SCOPE_FIELDS = ('native_context_qualified', 'native_scientific_qualification',
                'scientific_effect_qualification', 'replay_qualified',
                'candidate_methods_tested', 'dispatch_ready')


def _false_scope(record, label):
    required = SCOPE_FIELDS
    if any(record.get(name) is not False for name in required):
        raise ValueError('Unsupported or overclaimed ' + label)


def _reject_positive_scope(record, label):
    if any(name in record and record[name] is not False for name in SCOPE_FIELDS):
        raise ValueError('Unsupported or overclaimed ' + label)


def validate_bundle(result_path, manifest_path, archive_path, *, expected_result_sha256,
                    expected_manifest_sha256, expected_archive_sha256, expected_uid,
                    expected_gpu_uuid, expected_generation_identity_sha256,
                    expected_source_time_query, max_files, max_unpacked_bytes,
                    max_member_bytes, max_archive_bytes, max_metadata_bytes):
    """Rehash a completed paired-context transport bundle without extracting it."""
    if not isinstance(expected_uid, str) or not expected_uid:
        raise ValueError('Nonempty expected producer identity UID required')
    if not isinstance(expected_gpu_uuid, str) or not expected_gpu_uuid:
        raise ValueError('Nonempty expected producer GPU UUID required')
    if type(expected_source_time_query) is not bool:
        raise ValueError('Boolean expected source-time-query value required')
    max_files = _positive_limit(max_files, 'files')
    max_unpacked_bytes = _positive_limit(max_unpacked_bytes, 'unpacked bytes')
    max_member_bytes = _positive_limit(max_member_bytes, 'member bytes')
    max_archive_bytes = _positive_limit(max_archive_bytes, 'archive bytes')
    max_metadata_bytes = _positive_limit(max_metadata_bytes, 'metadata bytes')
    paths = {
        'result.json': _physical_file(result_path),
        'raw-manifest.json': _physical_file(manifest_path),
        'raw-evidence.tar': _physical_file(archive_path),
    }
    if paths['raw-evidence.tar'].stat().st_size > max_archive_bytes:
        raise ValueError('Raw archive exceeds archive byte limit')
    if any(paths[name].stat().st_size > max_metadata_bytes
           for name in ('result.json', 'raw-manifest.json')):
        raise ValueError('Bundle metadata exceeds per-file metadata byte limit')
    expected = {
        'result.json': _expected_hash(expected_result_sha256, 'result'),
        'raw-manifest.json': _expected_hash(expected_manifest_sha256, 'manifest'),
        'raw-evidence.tar': _expected_hash(expected_archive_sha256, 'archive'),
    }
    actual = {name: _hash(path) for name, path in paths.items()}
    if actual != expected:
        raise ValueError('Paired-context bundle differs from explicitly pinned inputs')
    result = json.loads(paths['result.json'].read_text())
    manifest = json.loads(paths['raw-manifest.json'].read_text())
    if (result.get('kind') != 'paired-native-context-instrument'
            or result.get('version') != 1
            or result.get('status') != 'completed_unqualified'
            or result.get('all_comparisons_match') is not True
            or result.get('final_integrity') != 'matched'
            or result.get('gpu_uuid') != expected_gpu_uuid
            or result.get('source_time_query_requested') is not expected_source_time_query):
        raise ValueError('Completed matching but scientifically unqualified result required')
    _false_scope(result, 'paired-context result')
    if (manifest.get('kind') != 'paired-native-context-raw-bundle'
            or manifest.get('version') != 1
            or manifest.get('native_context_qualified') is not False
            or manifest.get('scientific_effect_qualification') is not False
            or manifest.get('source_time_query') is not expected_source_time_query):
        raise ValueError('Unsupported or overclaimed paired-context manifest')
    _reject_positive_scope(manifest, 'paired-context manifest')
    archive_ref = manifest.get('archive')
    if archive_ref != {'path': 'raw-evidence.tar',
                       'bytes': paths['raw-evidence.tar'].stat().st_size,
                       'sha256': actual['raw-evidence.tar']}:
        raise ValueError('Manifest archive reference differs from current archive')
    raw_bundle = result.get('raw_bundle')
    expected_bundle = {
        'archive': archive_ref,
        'manifest': {'path': 'raw-manifest.json', 'sha256': actual['raw-manifest.json']},
        'raw_file_count': len(manifest.get('files', [])),
    }
    if raw_bundle != expected_bundle:
        raise ValueError('Result does not bind the current raw bundle')
    rows, names = [], []
    if not isinstance(manifest.get('files'), list) or not manifest['files']:
        raise ValueError('Nonempty raw file inventory required')
    if len(manifest['files']) > max_files:
        raise ValueError('Raw member count exceeds expansion limit')
    unpacked_bytes = 0
    for row in manifest['files']:
        if not isinstance(row, dict) or set(row) != {'path', 'bytes', 'sha256'}:
            raise ValueError('Exact raw member record required')
        name = _member_path(row['path']).as_posix()
        if type(row['bytes']) is not int or row['bytes'] < 0:
            raise ValueError('Nonnegative raw member byte count required')
        if row['bytes'] > max_member_bytes:
            raise ValueError('Raw member exceeds per-file expansion limit')
        unpacked_bytes += row['bytes']
        if unpacked_bytes > max_unpacked_bytes:
            raise ValueError('Raw bundle exceeds total expansion limit')
        _expected_hash(row['sha256'], 'member ' + name)
        names.append(name); rows.append(dict(row))
    if names != sorted(names) or len(names) != len(set(names)):
        raise ValueError('Raw member inventory must be sorted and unique')
    required = set(required_raw_paths(source_time_query=manifest['source_time_query']))
    if not required.issubset(names):
        raise ValueError('Raw member inventory omits required native context bytes')
    payloads = {}
    inspected = {'generation-identity.json', 'paired-comparison.json',
                 'replay/stage-result.json'}
    with tarfile.open(paths['raw-evidence.tar'], mode='r:') as archive:
        members = archive.getmembers()
        if [member.name for member in members] != names:
            raise ValueError('Archive member inventory differs from manifest')
        for member, row in zip(members, rows):
            if (not member.isfile() or member.issym() or member.islnk()
                    or getattr(member, 'sparse', None) or member.size != row['bytes']):
                raise ValueError('Archive contains a non-regular or wrong-size member')
            value = hashlib.sha256()
            content = bytearray() if member.name in inspected else None
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError('Archive member payload unavailable: ' + member.name)
            with stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    value.update(block)
                    if content is not None:
                        content.extend(block)
            if value.hexdigest() != row['sha256']:
                raise ValueError('Archive member hash differs: ' + member.name)
            if content is not None:
                payloads[member.name] = bytes(content)
    identity_hash = _expected_hash(expected_generation_identity_sha256,
                                   'generation identity')
    identity = json.loads(payloads['generation-identity.json'])
    pair = json.loads(payloads['paired-comparison.json'])
    replay = json.loads(payloads['replay/stage-result.json'])
    if (hashlib.sha256(payloads['generation-identity.json']).hexdigest() != identity_hash
            or identity.get('kind') != 'native-context-generation-identity'
            or identity.get('version') != 1 or identity.get('uid') != expected_uid
            or identity.get('gpu_uuid') != expected_gpu_uuid
            or identity.get('native_context_qualified') is not False
            or identity.get('scientific_effect_qualification') is not False):
        raise ValueError('Raw generation identity differs from expected producer identity')
    _reject_positive_scope(identity, 'raw generation identity')
    if (pair.get('kind') != 'paired-native-generation-comparison'
            or pair.get('version') != 1 or pair.get('matches') is not True
            or pair.get('native_context_qualified') is not False
            or pair.get('scientific_effect_qualification') is not False):
        raise ValueError('Raw paired comparison is absent, mismatched, or overclaimed')
    _reject_positive_scope(pair, 'raw paired comparison')
    if (replay.get('kind') != 'native-context-stage-result'
            or replay.get('version') != 1 or replay.get('stage') != 'replay'
            or replay.get('status') != 'completed_unqualified'
            or replay.get('all_comparisons_match') is not True):
        raise ValueError('Raw replay stage is absent or did not complete matching')
    _false_scope(replay, 'raw replay stage')
    if (result.get('paired_comparison') != pair
            or replay.get('paired_comparison') != pair
            or result.get('replay_reports') != replay.get('windows')):
        raise ValueError('Producer result/replay semantic cross-links differ')
    return {
        'status': 'validated_unqualified', 'files': rows,
        'source_time_query': manifest['source_time_query'],
        'input_sha256': actual, 'native_context_qualified': False,
        'scientific_effect_qualification': False, 'expected_uid': expected_uid,
        'expected_gpu_uuid': expected_gpu_uuid,
        'generation_identity_sha256': identity_hash,
        'limits': {'max_files': max_files, 'max_unpacked_bytes': max_unpacked_bytes,
                   'max_member_bytes': max_member_bytes,
                   'max_archive_bytes': max_archive_bytes,
                   'max_metadata_bytes': max_metadata_bytes},
    }


def consume_bundle(result_path, manifest_path, archive_path, output, **contract):
    """Validate fully, then extract with controlled writes into a single-use root."""
    output = Path(output)
    parent = output.parent.resolve()
    if (not parent.is_dir() or parent.is_symlink()
            or output.parent.absolute() != parent):
        raise ValueError('Physical output parent directory required')
    if output.exists() or output.is_symlink():
        raise FileExistsError('Preserve existing consumed bundle output')
    sources = tuple(_physical_file(path) for path in
                    (result_path, manifest_path, archive_path))
    source_bytes = tuple(path.stat().st_size for path in sources)
    max_archive_bytes = _positive_limit(contract.get('max_archive_bytes'), 'archive bytes')
    max_unpacked_bytes = _positive_limit(contract.get('max_unpacked_bytes'), 'unpacked bytes')
    max_metadata_bytes = _positive_limit(contract.get('max_metadata_bytes'), 'metadata bytes')
    if source_bytes[2] > max_archive_bytes:
        raise ValueError('Raw archive exceeds archive byte limit')
    if any(value > max_metadata_bytes for value in source_bytes[:2]):
        raise ValueError('Bundle metadata exceeds per-file metadata byte limit')
    required_free = (source_bytes[2] + 2 * sum(source_bytes[:2])
                     + max_unpacked_bytes + 1024 * 1024)
    if shutil.disk_usage(parent).free < required_free:
        raise ValueError('Insufficient free disk for bounded snapshot and extraction')
    with tempfile.TemporaryDirectory(dir=parent, prefix='.native-context-snapshot-') as directory:
        snapshot = Path(directory)
        staged = []
        for source, expected_bytes, name in zip(sources, source_bytes,
                ('result.json', 'raw-manifest.json', 'raw-evidence.tar')):
            target = snapshot/name
            _copy_exact_snapshot(source, target, expected_bytes)
            staged.append(target)
        record = validate_bundle(*staged, **contract)
        output.mkdir(exist_ok=False)
        raw = output/'raw'; raw.mkdir()
        with tarfile.open(staged[2], mode='r:') as archive:
            members = archive.getmembers()
            if [member.name for member in members] != [row['path'] for row in record['files']]:
                raise ValueError('Snapshot archive changed after validation')
            for member, row in zip(members, record['files']):
                if (not member.isfile() or member.issym() or member.islnk()
                        or getattr(member, 'sparse', None) or member.size != row['bytes']):
                    raise ValueError('Snapshot archive member changed after validation')
                target = raw.joinpath(*PurePosixPath(row['path']).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError('Archive member payload unavailable: ' + member.name)
                value = hashlib.sha256()
                with stream, target.open('xb') as destination:
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        destination.write(block); value.update(block)
                if target.stat().st_size != row['bytes'] or value.hexdigest() != row['sha256']:
                    raise ValueError('Extracted raw member differs: ' + row['path'])
        shutil.copyfile(staged[0], output/'bundle-result.json')
        shutil.copyfile(staged[1], output/'bundle-manifest.json')
        if (_hash(output/'bundle-result.json') != record['input_sha256']['result.json']
                or _hash(output/'bundle-manifest.json') != record['input_sha256']['raw-manifest.json']):
            raise ValueError('Copied bundle metadata changed after validation')
        summary = {
            'status': 'consumed_unqualified', 'extracted_files': len(record['files']),
            'source_time_query': record['source_time_query'],
            'input_sha256': record['input_sha256'],
            'producer_uid': record['expected_uid'],
            'producer_gpu_uuid': record['expected_gpu_uuid'],
            'generation_identity_sha256': record['generation_identity_sha256'],
            'limits': record['limits'],
            'native_context_qualified': False, 'scientific_effect_qualification': False,
            'replay_qualified': False, 'candidate_methods_tested': False,
            'dispatch_ready': False,
            'scope': 'Verified transport extraction only; no model, candidate, scorer or scientific admission',
        }
        with (output/'bundle-consumption.json').open('x') as stream:
            stream.write(json.dumps(summary, indent=2, allow_nan=False) + '\n')
        return summary


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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-result-sha256', required=True)
    parser.add_argument('--expected-manifest-sha256', required=True)
    parser.add_argument('--expected-archive-sha256', required=True)
    parser.add_argument('--expected-uid', required=True)
    parser.add_argument('--expected-gpu-uuid', required=True)
    parser.add_argument('--expected-generation-identity-sha256', required=True)
    parser.add_argument('--expected-source-time-query', choices=('true', 'false'), required=True)
    parser.add_argument('--max-files', type=int, required=True)
    parser.add_argument('--max-unpacked-bytes', type=int, required=True)
    parser.add_argument('--max-member-bytes', type=int, required=True)
    parser.add_argument('--max-archive-bytes', type=int, required=True)
    parser.add_argument('--max-metadata-bytes', type=int, required=True)
    args = parser.parse_args(argv)
    result = consume_bundle(args.result, args.manifest, args.archive, args.output,
        expected_result_sha256=args.expected_result_sha256,
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_archive_sha256=args.expected_archive_sha256,
        expected_uid=args.expected_uid, expected_gpu_uuid=args.expected_gpu_uuid,
        expected_generation_identity_sha256=args.expected_generation_identity_sha256,
        expected_source_time_query=args.expected_source_time_query == 'true',
        max_files=args.max_files, max_unpacked_bytes=args.max_unpacked_bytes,
        max_member_bytes=args.max_member_bytes,
        max_archive_bytes=args.max_archive_bytes,
        max_metadata_bytes=args.max_metadata_bytes)
    print(json.dumps(result))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
