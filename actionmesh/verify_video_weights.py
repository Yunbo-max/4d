"""Offline, bounded-memory audit of Wan files against the pinned manifest and ledger.

No downloads, conversion or deletion are performed. Converted source shards are
normally absent: their historical source hashes are bound to the manifest, while
derived bytes, tensor layout, shard keys, storage dtype and finiteness are checked
directly. This does not independently re-prove the absent source-to-output cast.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import time


KEEP_FP32 = ['time_embedder', 'scale_shift_table', 'norm1', 'norm2', 'norm3']
INDEX_REL = 'transformer/diffusion_pytorch_model.safetensors.index.json'
DTYPE_BYTES = {'BOOL': 1, 'U8': 1, 'I8': 1, 'U16': 2, 'I16': 2, 'U32': 4,
               'I32': 4, 'U64': 8, 'I64': 8, 'F16': 2, 'BF16': 2, 'F32': 4, 'F64': 8}
FLOAT_DTYPES = {'F16', 'BF16', 'F32', 'F64'}


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}')
    with temporary.open('w') as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def read_tensor_header(path):
    """Validate the Safetensors header and contiguous data layout without loading it."""
    size = path.stat().st_size
    with path.open('rb') as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f'Truncated Safetensors header: {path}')
        length = struct.unpack('<Q', prefix)[0]
        if length < 2 or length > 100_000_000 or length + 8 > size:
            raise ValueError(f'Invalid Safetensors header length: {path}')
        header = json.loads(stream.read(length), object_pairs_hook=unique_object)
    specs, intervals = {}, []
    for key, value in header.items():
        if key == '__metadata__':
            continue
        dtype, shape, offsets = value['dtype'], value['shape'], value['data_offsets']
        if dtype not in DTYPE_BYTES or not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError(f'Invalid dtype or shape: {path}:{key}')
        if len(offsets) != 2 or any(type(n) is not int or n < 0 for n in offsets):
            raise ValueError(f'Invalid data offsets: {path}:{key}')
        start, end = offsets
        expected_bytes = math.prod(shape) * DTYPE_BYTES[dtype]
        if end - start != expected_bytes:
            raise ValueError(f'Tensor shape does not match byte layout: {path}:{key}')
        specs[key] = {'dtype': dtype, 'shape': shape, 'bytes': expected_bytes}
        intervals.append((start, end, key))
    cursor = 0
    for start, end, key in sorted(intervals):
        if start != cursor:
            raise ValueError(f'Overlapping or noncontiguous tensor data: {path}:{key}')
        cursor = end
    if cursor + length + 8 != size:
        raise ValueError(f'Truncated or trailing tensor data: {path}')
    if not specs:
        raise ValueError(f'No tensors: {path}')
    return specs, intervals, length + 8


def scan_safetensors(path, converted=False, expected_specs=None, expected_keys=None):
    """Use <=8 MiB read buffers; no complete tensor, shard, Torch or GPU in memory."""
    import numpy as np
    specs, intervals, payload_offset = read_tensor_header(path)
    if expected_keys is not None and set(specs) != set(expected_keys):
        raise ValueError(f'Shard keys differ from index: {path}; missing={sorted(set(expected_keys)-set(specs))[:5]}, extra={sorted(set(specs)-set(expected_keys))[:5]}')
    if expected_specs is not None and specs != expected_specs:
        raise ValueError(f'Tensor keys, shape or dtype differ from expected metadata: {path}')
    checked = 0
    with path.open('rb') as stream:
        for start, end, key in sorted(intervals):
            dtype = specs[key]['dtype']
            if converted and dtype in FLOAT_DTYPES:
                wanted = 'F32' if any(part in key for part in KEEP_FP32) else 'F16'
                if dtype != wanted:
                    raise ValueError(f'Wrong converted storage dtype: {path}:{key}; expected {wanted}, got {dtype}')
            if dtype not in FLOAT_DTYPES:
                continue
            # IEEE exponent bits detect both NaN and Inf, including BF16, without
            # casting BF16 to FP16 or materializing multi-gigabyte embeddings.
            unsigned, mask = {'F16': ('<u2', 0x7c00), 'BF16': ('<u2', 0x7f80),
                              'F32': ('<u4', 0x7f800000), 'F64': ('<u8', 0x7ff0000000000000)}[dtype]
            stream.seek(payload_offset + start)
            remaining = end - start
            while remaining:
                block = stream.read(min(8 * 1024 * 1024, remaining))
                if not block:
                    raise ValueError(f'Truncated tensor during scan: {path}:{key}')
                values = np.frombuffer(block, dtype=unsigned)
                if np.any((values & mask) == mask):
                    raise ValueError(f'Non-finite tensor value: {path}:{key}')
                remaining -= len(block)
            checked += 1
    return {'tensor_count': len(specs), 'floating_tensors_checked_finite': checked,
            'all_floating_tensors_finite': True, 'tensor_bytes': sum(v['bytes'] for v in specs.values()),
            'dtype_tensor_counts': dict(Counter(v['dtype'] for v in specs.values())), 'tensor_specs': specs}


def expected_revision(manifest, item):
    return item.get('revision', manifest['huggingface_revision'])


def validate_binding(manifest, item, entry):
    rel = item['path']
    if entry.get('source_sha256') != item['sha256'] or entry.get('source_bytes') != item['size']:
        raise ValueError(f'Ledger source hash/size does not match pinned manifest: {rel}')
    if entry.get('converted_to_fp16') is not bool(item['convert_to_fp16']):
        raise ValueError(f'Ledger conversion policy mismatch: {rel}')
    wanted_keep = item.get('preserve_fp32_module_names', KEEP_FP32) if item['convert_to_fp16'] else []
    if entry.get('kept_fp32_modules', []) != wanted_keep:
        raise ValueError(f'Ledger keep-FP32 policy mismatch: {rel}')
    if item['convert_to_fp16'] and wanted_keep != KEEP_FP32:
        raise ValueError(f'Unsupported keep-FP32 policy in manifest: {rel}')
    if entry.get('source_revision', expected_revision(manifest, item)) != expected_revision(manifest, item):
        raise ValueError(f'Ledger revision mismatch: {rel}')
    if entry.get('source_repo', item.get('repo', manifest['model_id'])) != item.get('repo', manifest['model_id']):
        raise ValueError(f'Ledger source repository mismatch: {rel}')


def expected_index_keys(dest, index_rel):
    path = dest / index_rel
    index = json.loads(path.read_text())
    by_file = {}
    for key, filename in index['weight_map'].items():
        shard = Path(index_rel).parent / filename
        if shard.is_absolute() or '..' in shard.parts:
            raise ValueError(f'Unsafe shard filename: {filename}')
        by_file.setdefault(str(shard), set()).add(key)
    return index, by_file


def audit(root, report_path=None):
    manifest_path = root / 'video-weights-manifest.json'
    ledger_path = root / 'manifests/wan-weights-verified.json'
    manifest = json.loads(manifest_path.read_text())
    ledger = json.loads(ledger_path.read_text())
    dest = root / 'weights/Wan2.2-TI2V-5B-Diffusers'
    report_path = report_path or root / 'manifests/wan-weights-audit.json'
    report = {'status': 'running', 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'model_id': manifest['model_id'], 'revision': manifest['huggingface_revision'],
              'manifest_sha256': digest(manifest_path), 'download_ledger_sha256': digest(ledger_path),
              'source_verification_scope': 'Unconverted file hashes are checked directly against pinned source hashes. For converted shards, absent source content is historically bound by matching ledger source hashes; derived content and tensor layout/dtypes/finiteness are directly checked. Source-to-output casting equality cannot be independently re-proven without source files.',
              'files': {}}
    try:
        items = {item['path']: item for item in manifest['files']}
        if len(items) != len(manifest['files']):
            raise ValueError('Duplicate manifest paths')
        for rel, item in items.items():
            path = dest / rel
            if Path(rel).is_absolute() or '..' in Path(rel).parts or path.is_symlink():
                raise ValueError(f'Unsafe manifest path or symlink: {rel}')
            if not path.is_file() or path.with_name(path.name + '.aria2').exists():
                raise ValueError(f'Missing/incomplete file: {rel}')
            if rel not in ledger:
                raise ValueError(f'Missing ledger entry: {rel}')
            entry = ledger[rel]
            validate_binding(manifest, item, entry)
            actual_hash = digest(path)
            if actual_hash != entry.get('stored_sha256') or path.stat().st_size != entry.get('stored_bytes'):
                raise ValueError(f'Stored file differs from ledger: {rel}')
            if not item['convert_to_fp16'] and rel != INDEX_REL and actual_hash != item['sha256']:
                raise ValueError(f'Unconverted file differs from pinned source: {rel}')
            report['files'][rel] = {'source_sha256': item['sha256'], 'source_revision': expected_revision(manifest, item),
                    'source_revision_binding': 'recorded_and_matched' if 'source_revision' in entry else 'bound_from_matching_source_sha256_to_pinned_manifest',
                    'stored_sha256': actual_hash, 'stored_bytes': path.stat().st_size,
                    'converted_to_fp16': item['convert_to_fp16']}
        transformer_index, transformer_keys = expected_index_keys(dest, INDEX_REL)
        text_index, text_keys = expected_index_keys(dest, 'text_encoder/model.safetensors.index.json')
        key_maps = {**transformer_keys, **text_keys}
        if any(rel not in items for rel in key_maps):
            raise ValueError('Index references a shard absent from pinned manifest')
        for rel, item in items.items():
            if rel.endswith('.safetensors'):
                validation = scan_safetensors(dest / rel, converted=item['convert_to_fp16'],
                         expected_specs=ledger[rel].get('tensor_validation', {}).get('tensor_specs'), expected_keys=key_maps.get(rel))
                report['files'][rel]['tensor_validation'] = validation
                print('AUDITED_TENSORS', rel, validation['tensor_count'], flush=True)
        for index, keys, label in [(transformer_index, transformer_keys, 'transformer'), (text_index, text_keys, 'text_encoder')]:
            tensor_bytes = sum(report['files'][rel]['tensor_validation']['tensor_bytes'] for rel in keys)
            if index.get('metadata', {}).get('total_size') != tensor_bytes:
                raise ValueError(f'{label} index total_size does not match stored tensor data')
        if digest(ledger_path) != report['download_ledger_sha256'] or digest(manifest_path) != report['manifest_sha256']:
            raise ValueError('Ledger or manifest changed while audit was running; rerun after downloader has exited')
        report.update({'status': 'verified', 'file_count': len(items), 'converted_shard_count': sum(i['convert_to_fp16'] for i in items.values()),
                       'stored_bytes': sum(v['stored_bytes'] for v in report['files'].values()), 'all_safetensors_finite': True,
                       'all_index_shard_keys_match': True, 'all_tensor_shape_byte_layouts_match': True, 'converted_dtype_policy_verified': True})
        atomic_json(report_path, report)
        print('WAN_WEIGHTS_AUDIT_VERIFIED', str(report_path), flush=True)
        return report
    except Exception as exc:
        report.update({'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc)})
        atomic_json(report_path, report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(os.environ.get('ACTIONMESH_WORKDIR', Path(__file__).resolve().parent)))
    parser.add_argument('--report', type=Path, help='Default: ROOT/manifests/wan-weights-audit.json')
    args = parser.parse_args()
    audit(args.root.expanduser().resolve(), args.report.expanduser().resolve() if args.report else None)


if __name__ == '__main__':
    main()
