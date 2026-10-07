"""Observe actual Stage-II windows and replay their complete decoder context.

This is an engineering instrument, not C01/C02 or a scientific admission. The
caller must supply an independently verified generation identity and use the
existing harness. No downloaded weights, synthetic motion target, ground truth,
candidate correction, or qualification flag is introduced here.
"""
from __future__ import annotations

from contextlib import ExitStack
import inspect
import json
import math
from pathlib import Path

from .decoder_observer import (DecoderObserver, FIELDS, load_capture, sha256,
                               tensor_inventory, write_record)


def _positive_integer(value, label):
    if type(value) is not int or value < 1:
        raise ValueError(label + ' must be a positive integer')


def _json_copy(value):
    if not isinstance(value, dict) or not value:
        raise ValueError('Nonempty generation identity required')
    return json.loads(json.dumps(value, allow_nan=False))


def _regular(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Physical capture file required: ' + str(path))
    if any(parent.is_symlink() for parent in path.parents):
        raise ValueError('Symlink capture parent: ' + str(path))
    return path


def _direct_eval(model):
    if (model is None or model.training or
            getattr(model, 'prediction_mode', None) != 'direct'):
        raise ValueError('Loaded eval-mode direct native decoder required')


def _window_inputs(values):
    import numpy as np
    import torch
    names = ('latents', 'window_timesteps', 'source_alpha', 'target_alphas')
    if any(not isinstance(values[name], torch.Tensor) for name in names):
        raise ValueError('Native window arguments must be tensors')
    latent, times, source, target = [values[name] for name in names]
    if (latent.ndim != 4 or latent.shape[0] != 1 or times.ndim != 2 or
            times.shape != latent.shape[:2] or times.shape[1] < 2 or
            source.shape != (1,) or target.ndim != 2 or
            target.shape[0] != 1 or target.shape[1] < 1):
        raise ValueError('Complete single-batch native window shapes required')
    if any(not torch.isfinite(values[name]).all().item() for name in names):
        raise ValueError('Non-finite native window inputs')
    # This instrument covers the current forward, anchor-zero generation path.
    # Backward/nonmonotone windows require their own reviewed mapping policy.
    if (not (times[:, 1:] > times[:, :-1]).all().item() or
            not (target[:, 1:] > target[:, :-1]).all().item() or
            not ((target >= 0) & (target <= 1)).all().item() or
            not torch.equal(source, torch.zeros_like(source))):
        raise ValueError('Forward source-first time mapping required')
    mesh = values['anchor_mesh']
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3 or
            not np.isfinite(vertices).all() or faces.ndim != 2 or
            faces.shape[1] != 3 or not len(faces) or
            not np.issubdtype(faces.dtype, np.integer) or
            faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError('Finite complete anchor geometry/topology required')
    low = times.min(dim=1).values
    width = times.max(dim=1).values - low
    native_targets = (times[:, 1:] - low[:, None]) / width[:, None]
    if not torch.equal(target, native_targets):
        raise ValueError('Native unsubsampled frame mapping required')
    return {
        'anchor_vertices': torch.from_numpy(vertices.copy()).contiguous(),
        'faces': torch.from_numpy(faces.copy()).contiguous(),
        'target_timesteps': times[:, 1:].detach().cpu().clone(),
        'source_timesteps': times[:, 0].detach().cpu().clone(),
    }


class PipelineDecoderObserver:
    """Temporarily wrap one pipeline instance without changing native results.

    The original bound method is called once with the original arguments. The
    decoder is obtained at call time, after the official lazy loader loads it.
    max_bytes bounds each window's persisted tensor payload, including both
    decoder input copies, raw output and meshes; file headers and JSON are
    excluded. Copies add synchronization, RAM and disk overhead; they do not
    establish matched-generation equivalence.
    """
    def __init__(self, pipeline, root: Path, identity: dict, *,
                 max_bytes=64 * 1024 * 1024, max_windows=64):
        _positive_integer(max_bytes, 'Capture byte bound')
        _positive_integer(max_windows, 'Window bound')
        self.pipeline = pipeline
        self.root = Path(root)
        self.identity = _json_copy(identity)
        self.max_bytes = max_bytes
        self.max_windows = max_windows
        self.windows = []
        self.attached = False
        self.active = False

    def _index(self, status):
        write_record(self.root / 'index.json', {
            'kind': 'pipeline-decoder-observation', 'version': 1,
            'status': status, 'windows': self.windows,
            'identity_sha256': sha256(self.root / 'identity.json'),
            'max_bytes_per_window': self.max_bytes, 'max_windows': self.max_windows,
            'byte_bound_scope': 'persisted_tensor_payload_excludes_headers',
            'scientific_effect_qualification': False, 'native_context_qualified': False,
        })

    def __enter__(self):
        if (self.attached or getattr(self.pipeline, '_research_observer_active', False) or
                hasattr(self.pipeline, '_research_decoder_capture')):
            raise ValueError('Pipeline observer already attached')
        self.original = self.pipeline._decode_displacement
        self.signature = inspect.signature(self.original)
        needed = {'latents', 'window_timesteps', 'source_alpha', 'target_alphas', 'anchor_mesh'}
        if not needed.issubset(self.signature.parameters):
            raise ValueError('Explicit native pipeline decoder signature required')
        self.had_override = '_decode_displacement' in vars(self.pipeline)
        self.old_override = vars(self.pipeline).get('_decode_displacement')
        if any(parent.is_symlink() for parent in self.root.parents):
            raise ValueError('Physical observation destination required')
        self.root.mkdir(parents=True, exist_ok=False)
        write_record(self.root / 'identity.json', self.identity)
        self._index('attached')
        self.pipeline._research_observer_active = True
        self.pipeline._decode_displacement = self._capture
        self.attached = True
        return self

    def _capture(self, *args, **kwargs):
        import numpy as np
        import torch
        from safetensors.torch import save_file
        if self.active or len(self.windows) >= self.max_windows:
            raise ValueError('Nested call or native window capture limit')
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        model = self.pipeline.temporal_3D_vae
        _direct_eval(model)
        meshes = _window_inputs(values)
        size = sum(item.numel() * item.element_size() for item in meshes.values())
        expected_shape = (values['target_alphas'].shape[1],) + tuple(meshes['anchor_vertices'].shape)
        # Native Trimesh construction stores each output vertex as float64.
        # Reserve the complete mesh payload before either decoder file is saved.
        mesh_reserve = size + math.prod(expected_shape) * np.dtype(np.float64).itemsize
        if mesh_reserve >= self.max_bytes:
            raise ValueError('Complete mesh capture exceeds byte bound')
        name = f'window-{len(self.windows):04d}'
        path = self.root / name
        path.mkdir(exist_ok=False)
        record = {
            'kind': 'pipeline-decoder-window', 'version': 1, 'status': 'started',
            'identity_sha256': sha256(self.root / 'identity.json'),
            'prediction_mode': 'direct', 'coordinate_bounds': [-1.0, 1.0],
            'mapping': 'source-first native window IDs; exact normalized-alpha check; no subsampling',
            'callback_present': values.get('step_callback') is not None,
            'max_bytes': self.max_bytes, 'scientific_effect_qualification': False,
            'byte_bound_scope': 'persisted_tensor_payload_excludes_headers',
            'reserved_mesh_payload_bytes': mesh_reserve,
            'replay_qualified': False, 'native_context_qualified': False,
        }
        self.windows.append({'path': name, 'status': 'started'})
        write_record(path / 'record.json', record)
        self._index('capturing')
        self.active = True
        try:
            with DecoderObserver(model, path / 'decoder', self.identity,
                                 max_bytes=self.max_bytes - mesh_reserve) as observer:
                output = self.original(*args, **kwargs)
            if observer.calls != 1:
                raise ValueError('Exactly one native decoder call per window required')
            if (not isinstance(output, (list, tuple)) or len(output) != expected_shape[0] or
                    any(not np.array_equal(np.asarray(item.faces), meshes['faces'].numpy())
                        for item in output)):
                raise ValueError('Native output topology/frame count changed')
            vertices = np.stack([np.asarray(item.vertices) for item in output])
            if vertices.shape != expected_shape or not np.isfinite(vertices).all():
                raise ValueError('Finite full native output coordinates required')
            meshes['vertices'] = torch.from_numpy(vertices.copy()).contiguous()
            captured, _ = load_capture(path / 'decoder' / 'call-0000')
            # Captured contains the second input copy plus raw output; add the
            # first copy retained in inputs.safetensors as well as all meshes.
            total = sum(t.numel() * t.element_size()
                        for t in [*meshes.values(), *captured.values(),
                                  *(captured[name] for name in FIELDS)])
            if total > self.max_bytes:
                raise ValueError('Complete window capture exceeds byte bound')
            save_file(meshes, str(path / 'meshes.safetensors'))
            record.update(status='captured_unqualified', tensors=tensor_inventory(meshes),
                          meshes_sha256=sha256(path / 'meshes.safetensors'),
                          decoder_record_sha256=sha256(path / 'decoder/call-0000/record.json'),
                          payload_bytes=total)
            return output
        except BaseException as error:
            record.update(status='capture_failed', error=type(error).__name__ + ': ' + str(error))
            raise
        finally:
            self.active = False
            write_record(path / 'record.json', record)
            self.windows[-1].update(status=record['status'], record_sha256=sha256(path / 'record.json'))
            self._index('captured_unqualified' if record['status'] == 'captured_unqualified' else 'failed')

    def __exit__(self, typ, error, tb):
        if self.attached:
            if self.had_override:
                self.pipeline._decode_displacement = self.old_override
            else:
                delattr(self.pipeline, '_decode_displacement')
            delattr(self.pipeline, '_research_observer_active')
            self.attached = False
            failed = error is not None or any(row['status'] != 'captured_unqualified' for row in self.windows)
            self._index('failed' if failed else 'captured_unqualified' if self.windows else 'no_windows')
        return False


def load_window(path: Path):
    """Check stored byte bindings and geometry/decoder agreement; no promotion."""
    import torch
    from safetensors.torch import load_file
    path = Path(path)
    record = json.loads(_regular(path / 'record.json').read_text())
    if (record.get('kind') != 'pipeline-decoder-window' or record.get('version') != 1 or
            record.get('status') != 'captured_unqualified' or record.get('prediction_mode') != 'direct'):
        raise ValueError('Complete direct native window capture required')
    for key, target in (
            ('identity_sha256', path.parent / 'identity.json'),
            ('meshes_sha256', path / 'meshes.safetensors'),
            ('decoder_record_sha256', path / 'decoder/call-0000/record.json')):
        if sha256(_regular(target)) != record.get(key):
            raise ValueError('Window capture hash mismatch: ' + key)
    for relative in ('decoder/identity.json', 'decoder/call-0000/inputs.safetensors',
                     'decoder/call-0000/tensors.safetensors'):
        _regular(path / relative)
    if sha256(path / 'decoder/identity.json') != record['identity_sha256']:
        raise ValueError('Decoder and window identities differ')
    tensors = load_file(str(path / 'meshes.safetensors'), device='cpu')
    if (set(tensors) != {'anchor_vertices', 'faces', 'vertices', 'target_timesteps', 'source_timesteps'} or
            tensor_inventory(tensors) != record.get('tensors')):
        raise ValueError('Window geometry inventory mismatch')
    captured, decoder_record = load_capture(path / 'decoder/call-0000')
    if decoder_record['prediction_mode'] != 'direct':
        raise ValueError('Unsupported saved prediction mode')
    query = captured['query']
    if (query.ndim != 3 or query.shape != (1, len(tensors['anchor_vertices']), 6) or
            not torch.equal(query[0, :, :3], tensors['anchor_vertices'].to(query.dtype))):
        raise ValueError('Native query and anchor geometry differ')
    times, source, target = (captured[key] for key in ('framestep', 'source_alpha', 'target_alphas'))
    low = times.min(dim=1).values
    width = times.max(dim=1).values - low
    if (not torch.equal(target, (times[:, 1:] - low[:, None]) / width[:, None]) or
            not torch.equal(source, torch.zeros_like(source)) or
            not torch.equal(tensors['target_timesteps'], times[:, 1:]) or
            not torch.equal(tensors['source_timesteps'], times[:, 0]) or
            captured['output'].shape != (1,) + tensors['vertices'].shape):
        raise ValueError('Native frame mapping/output shape mismatch')
    # Mesh construction in the direct upstream path clamps before float64 storage.
    if not torch.equal(captured['output'].clamp(-1, 1)[0].to(tensors['vertices'].dtype), tensors['vertices']):
        raise ValueError('Native raw output and bounded mesh coordinates differ')
    return tensors, record


def _tolerance(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('Explicit finite nonnegative replay tolerance required')
    return float(value)


def _comparison(actual, expected, *, atol, rtol):
    import torch
    if actual.shape != expected.shape:
        return {'matches': False, 'reason': 'shape_mismatch', 'max_abs_error': None}
    a, b = actual.detach().cpu().to(torch.float64), expected.detach().cpu().to(torch.float64)
    finite = torch.isfinite(a).all().item() and torch.isfinite(b).all().item()
    return {'matches': bool(finite and torch.allclose(a, b, atol=atol, rtol=rtol)),
            'max_abs_error': float((a - b).abs().max()) if finite and a.numel() else None,
            'dtype_matches': actual.dtype == expected.dtype}


def replay_window(model, window_path: Path, output_path: Path, *, identity: dict,
                  atol: float, rtol: float, source_time_query=False):
    """Replay non-anchor queries first; source-time query is a separate output.

    Identity equality here compares the caller's already verified identity; it
    cannot authenticate a supplied model. CLI integration rechecks its assets.
    Never use these diagnostic booleans as a native/scientific admission.
    """
    import torch
    from safetensors.torch import save_file
    atol, rtol = _tolerance(atol), _tolerance(rtol)
    _direct_eval(model)
    window_path, output_path = Path(window_path), Path(output_path)
    meshes, window_record = load_window(window_path)
    expected_identity = json.loads((window_path.parent / 'identity.json').read_text())
    if _json_copy(identity) != expected_identity:
        raise ValueError('Replay generation identity mismatch')
    tensors, record = load_capture(window_path / 'decoder/call-0000')
    if record.get('step_callback_present') or window_record.get('callback_present'):
        raise ValueError('Replay of external callback semantics is unsupported')
    if record.get('training') or 'grad_enabled' not in record:
        raise ValueError('Eval capture with recorded gradient mode required')
    values = {name: tensors[name].to(device=record['input_devices'][name]) for name in FIELDS}
    output_path.mkdir(parents=True, exist_ok=False)
    report = {'kind': 'native-window-replay', 'version': 1, 'status': 'started',
              'atol': atol, 'rtol': rtol,
              'window_record_sha256': sha256(window_path / 'record.json'),
              'identity_sha256': window_record['identity_sha256'],
              'raw_matches': False, 'mesh_matches': False,
              'scientific_effect_qualification': False, 'replay_qualified': False,
              'native_context_qualified': False, 'source_time_requested': bool(source_time_query)}
    write_record(output_path / 'report.json', report)
    try:
        with ExitStack() as stack:
            stack.enter_context(torch.inference_mode(record['inference_mode']))
            stack.enter_context(torch.set_grad_enabled(record['grad_enabled']))
            for device in ('cpu', 'cuda'):
                if record[device + '_autocast_enabled']:
                    dtype_name = record[device + '_autocast_dtype'].removeprefix('torch.')
                    if dtype_name not in ('bfloat16', 'float16'):
                        raise ValueError('Unsupported recorded autocast dtype')
                    stack.enter_context(torch.autocast(device_type=device, dtype=getattr(torch, dtype_name)))
                else:
                    stack.enter_context(torch.autocast(device_type=device, enabled=False))
            raw = model(**values)
            bounded = model.apply_displacement(vertex=values['query'][:3], displacement=raw)
            raw_comparison = _comparison(raw, tensors['output'], atol=atol, rtol=rtol)
            mesh_comparison = _comparison(bounded[0], meshes['vertices'], atol=atol, rtol=rtol)
            report.update(raw=raw_comparison, mesh=mesh_comparison,
                          raw_matches=raw_comparison['matches'] and raw_comparison.get('dtype_matches', False),
                          mesh_matches=mesh_comparison['matches'])
            save_file({'raw': raw.detach().cpu().contiguous(),
                       'bounded': bounded.detach().cpu().contiguous()}, str(output_path / 'replay.safetensors'))
            if source_time_query and report['raw_matches'] and report['mesh_matches']:
                source_values = dict(values, target_alphas=values['source_alpha'][:, None])
                source_raw = model(**source_values)
                if (source_raw.shape != (1, 1, values['query'].shape[1], 3) or
                        not torch.isfinite(source_raw).all().item()):
                    raise ValueError('Source-time query returned invalid full field')
                save_file({'output': source_raw.detach().cpu().contiguous()},
                          str(output_path / 'source-time.safetensors'))
                report['source_time_sha256'] = sha256(output_path / 'source-time.safetensors')
            report.update(status='replayed_unqualified',
                          replay_sha256=sha256(output_path / 'replay.safetensors'))
    except BaseException as error:
        report.update(status='replay_failed', error=type(error).__name__ + ': ' + str(error))
        raise
    finally:
        write_record(output_path / 'report.json', report)
    return report
