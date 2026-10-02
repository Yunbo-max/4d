"""Download pinned official Wan weights via verified domestic mirror; convert storage dtype."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess

from verify_video_weights import (KEEP_FP32, FLOAT_DTYPES, INDEX_REL, atomic_json,
    digest, expected_revision, read_tensor_header, scan_safetensors, validate_binding)


def derived_specs(source_specs):
    result = {}
    for key, spec in source_specs.items():
        value = dict(spec)
        if value['dtype'] in FLOAT_DTYPES:
            value['dtype'] = 'F32' if any(part in key for part in KEEP_FP32) else 'F16'
            value['bytes'] = (4 if value['dtype'] == 'F32' else 2) * math.prod(spec['shape'])
        result[key] = value
    return result


def validate_existing(manifest, item, entry, output):
    """Accept legacy ledger entries only when source hash, bytes and policy match."""
    validate_binding(manifest, item, entry)
    if output.is_symlink() or not output.is_file() or output.with_name(output.name + '.aria2').exists():
        return None
    if output.stat().st_size != entry.get('stored_bytes') or digest(output) != entry.get('stored_sha256'):
        return None
    if not item['convert_to_fp16'] and item['path'] != INDEX_REL and digest(output) != item['sha256']:
        raise ValueError('Unconverted cached file differs from official source: ' + item['path'])
    if output.suffix == '.safetensors':
        return scan_safetensors(output, converted=item['convert_to_fp16'],
                                expected_specs=entry.get('tensor_validation', {}).get('tensor_specs'))
    return {}


def record_source(manifest, item):
    return {'source_sha256': item['sha256'], 'source_bytes': item['size'],
            'source_revision': expected_revision(manifest, item),
            'source_repo': item.get('repo', manifest['model_id']),
            'converted_to_fp16': item['convert_to_fp16'],
            'kept_fp32_modules': KEEP_FP32 if item['convert_to_fp16'] else []}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(os.environ.get('ACTIONMESH_WORKDIR', Path(__file__).resolve().parent)))
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / 'video-weights-manifest.json').read_text())
    dest = root / 'weights/Wan2.2-TI2V-5B-Diffusers'
    temp = root / 'weights/wan-download-temporary'
    record = root / 'manifests/wan-weights-verified.json'
    dest.mkdir(parents=True, exist_ok=True)
    temp.mkdir(parents=True, exist_ok=True)
    record.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(record.read_text()) if record.exists() else {}
    files = sorted(manifest['files'], key=lambda item: (not item['convert_to_fp16'], item['path']))
    for item in files:
        rel = item['path']
        if Path(rel).is_absolute() or '..' in Path(rel).parts:
            raise ValueError('Unsafe manifest path: ' + rel)
        if item['convert_to_fp16'] and item.get('preserve_fp32_module_names', KEEP_FP32) != KEEP_FP32:
            raise ValueError('Unsupported FP32-preservation policy: ' + rel)
        output = dest / rel
        output.parent.mkdir(parents=True, exist_ok=True)
        if rel in results:
            validation = validate_existing(manifest, item, results[rel], output)
            if validation is not None:
                results[rel].update(record_source(manifest, item))
                if validation:
                    results[rel]['tensor_validation'] = validation
                    if item['convert_to_fp16']:
                        results[rel]['tensor_bytes'] = validation['tensor_bytes']
                atomic_json(record, results)
                print('VERIFIED_EXISTING', rel, flush=True)
                continue
        source = temp / (rel.replace('/', '__') + '.source')
        sidecar = source.with_name(source.name + '.aria2')
        if source.is_symlink() or sidecar.is_symlink() or output.is_symlink():
            raise ValueError('Refusing to overwrite a weight/temp symlink: ' + rel)
        if not source.is_file() or source.stat().st_size != item['size'] or digest(source) != item['sha256']:
            # A completed corrupt temporary file must not be mistaken by aria2
            # for an already-finished download. Only this dedicated temp path
            # is removed; partially downloaded sources with sidecars can resume.
            if source.is_file() and not sidecar.exists():
                source.unlink()
            if not source.exists() and sidecar.exists():
                sidecar.unlink()
            subprocess.run(['aria2c', '--continue=true', '--allow-overwrite=true', '--auto-file-renaming=false', '--file-allocation=none',
                '--check-integrity=true', '--checksum=sha-256=' + item['sha256'], '--max-tries=5', '--retry-wait=5', '--connect-timeout=20',
                '--timeout=60', '-x', '8', '-s', '8', '--console-log-level=warn', '--summary-interval=30', '--dir', str(temp), '--out', source.name, item['mirror_url']], check=True, timeout=14400)
        if not source.is_file() or source.stat().st_size != item['size'] or digest(source) != item['sha256']:
            raise ValueError('Incomplete or incorrect official source: ' + rel)
        if sidecar.exists():
            sidecar.unlink()  # Full file SHA256 has just been verified.
        raw_tensor_bytes = None
        validation = None
        if item['convert_to_fp16']:
            import torch
            from safetensors import safe_open
            from safetensors.torch import save_file
            torch.set_num_threads(4)
            source_specs, _, _ = read_tensor_header(source)
            expected_specs = derived_specs(source_specs)
            tensors = {}
            with safe_open(str(source), framework='pt', device='cpu') as handle:
                for key in handle.keys():
                    tensor = handle.get_tensor(key)
                    if tensor.is_floating_point():
                        dtype = torch.float32 if any(part in key for part in KEEP_FP32) else torch.float16
                        tensor = tensor.to(dtype)
                    tensors[key] = tensor.contiguous()
                metadata = handle.metadata()
            raw_tensor_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
            staged = output.with_name(output.name + '.converting')
            save_file(tensors, str(staged), metadata=metadata)
            del tensors, tensor
            # Scan the actual saved bytes before replacing output or deleting
            # the verified source. FP32->FP16 overflow is an explicit failure.
            validation = scan_safetensors(staged, converted=True, expected_specs=expected_specs)
            staged.replace(output)
        else:
            if source.name.endswith('.safetensors.source'):
                validation = scan_safetensors(source)
            source.replace(output)
        results[rel] = {**record_source(manifest, item), 'stored_sha256': digest(output), 'stored_bytes': output.stat().st_size,
                        'tensor_bytes': raw_tensor_bytes, **({'tensor_validation': validation} if validation else {})}
        atomic_json(record, results)
        if item['convert_to_fp16']:
            source.unlink()  # Delete only this verified temporary source, after durable validation/ledger.
        print('COMPLETE', rel, results[rel]['stored_bytes'], flush=True)
    index_path = dest / INDEX_REL
    index = json.loads(index_path.read_text())
    index['metadata']['total_size'] = sum(results[item['path']]['tensor_bytes'] for item in files if item['convert_to_fp16'])
    atomic_json(index_path, index)
    results[str(index_path.relative_to(dest))]['stored_sha256'] = digest(index_path)
    results[str(index_path.relative_to(dest))]['stored_bytes'] = index_path.stat().st_size
    results[str(index_path.relative_to(dest))]['note'] = 'Only tensor total_size metadata adjusted for half-precision storage.'
    atomic_json(record, results)
    print('WAN_WEIGHTS_READY', str(dest), flush=True)


if __name__ == '__main__':
    main()
