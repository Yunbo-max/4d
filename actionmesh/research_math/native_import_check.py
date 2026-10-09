"""Harness-only CPU dependency diagnostic; no checkpoint, native run or score."""
import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve previous import report')
    result = {'status': 'failed', 'imports': [], 'gpu_count': 0,
              'scientific_qualification': False, 'model_loaded': False,
              'python_executable': sys.executable}
    try:
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
            raise ValueError('Zero-GPU harness allocation required')
        inventory = json.loads(args.inventory.read_text())
        source = Path(inventory['source_root'])
        for row in inventory['files']:
            path = source / row['path']
            if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError('Official source bytes changed: ' + row['path'])
        result['source_inventory_sha256'] = hashlib.sha256(args.inventory.read_bytes()).hexdigest()
        result['packages'] = {name: importlib.metadata.version(name) for name in
                              ('torch', 'torchvision', 'pytorch3d', 'diffusers', 'transformers', 'diso')}
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
        sys.path[:0] = [str(source), str(source / 'third_party/TripoSG')]
        for name in ('torch', 'torchvision', 'pytorch3d.ops', 'diso', 'actionmesh.pipeline',
                     'actionmesh.model.temporal_autoencoder'):
            module = importlib.import_module(name)
            if name.startswith('actionmesh.') and not Path(module.__file__).resolve().is_relative_to(source):
                raise ValueError('Native module escaped inspected source')
            result['imports'].append({'name': name, 'file': module.__file__})
        entry = source / 'inference/video_to_animated_mesh.py'
        spec = importlib.util.spec_from_file_location('_native_import_entry', entry)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result['imports'].append({'name': 'official_inference_entry', 'file': str(entry)})
        import torch
        from actionmesh.model.utils.block import FlowMatchingBlock
        result['component_checks'] = []
        # Actual denoiser and autoencoder call-site options, not the unused
        # FlowMatchingBlock default. Tiny CPU fixtures load no model weights.
        for norm in (None, 'layer_norm'):
            block = FlowMatchingBlock(dim=16, num_attention_heads=2,
                cross_attention_dim=16, cross_attention_norm_type=norm)
            with torch.no_grad():
                output = block(torch.zeros(1, 3, 16), torch.zeros(1, 2, 16))
            if output.shape != (1, 3, 16) or not torch.isfinite(output).all():
                raise ValueError('Native attention component API check failed')
            result['component_checks'].append({'cross_attention_norm': norm,
                                               'status': 'completed', 'device': 'cpu'})
        result['cuda_initialized'] = torch.cuda.is_initialized()
        if result['cuda_initialized']:
            raise ValueError('Import unexpectedly initialized CUDA')
        result['status'] = 'completed'
    except Exception as error:
        result.update(error=type(error).__name__ + ': ' + str(error), traceback=traceback.format_exc())
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'completed_imports': len(result['imports']),
                      'error': result.get('error')}))
    return 0 if result['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
