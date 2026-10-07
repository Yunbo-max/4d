"""Capture complete native decoder windows through the real pipeline boundary.

This is an engineering observer, not a candidate or a replay qualification.
Attach inside a separately admitted harness job to an existing pipeline. No
models, data, scorer or GPU job are loaded or launched by this module.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import MethodType

from research_math.decoder_observer import DecoderObserver, load_capture, sha256, write_record


_ABSENT = object()
_FIELDS = ('latents', 'window_timesteps', 'source_alpha', 'target_alphas', 'anchor_mesh')


def _source(method):
    path = inspect.getsourcefile(method)
    if path is None or not Path(path).is_file():
        raise ValueError('Inspectable native Python source required')
    return {'path': str(Path(path).resolve()), 'sha256': sha256(path)}


class PipelineDecoderCapture:
    """Observe `_decode_displacement` without replacing its args or output.

    The decoder is looked up at call time, after official low-RAM lazy loading.
    All vertices, target times and input latents are retained. Capture performs
    device-to-host copies and disk writes; its overhead requires new measurement.
    Instance-local, single-threaded use only. Concurrent/nested attachment fails.
    A failed window poisons the session even if the caller catches the exception.
    Caller identity is retained provenance, never trusted scientific approval.
    """

    def __init__(self, pipeline, root, identity, *, max_windows=2,
                 max_tensor_bytes=64*1024*1024, max_geometry_bytes=64*1024*1024):
        for value in (max_windows, max_tensor_bytes, max_geometry_bytes):
            if type(value) is not int or value < 1:
                raise ValueError('Positive integer capture bounds required')
        self.pipeline = pipeline
        self.root = Path(root)
        self.identity = json.loads(json.dumps(identity, allow_nan=False))
        self.max_windows = max_windows
        self.max_tensor_bytes = max_tensor_bytes
        self.max_geometry_bytes = max_geometry_bytes
        self.windows = 0
        self.failed = False
        self.active = False

    def __enter__(self):
        if self.active or hasattr(self.pipeline, '_research_decoder_capture'):
            raise ValueError('Pipeline observer already attached')
        self.original = self.pipeline._decode_displacement
        self.signature = inspect.signature(self.original)
        if not set(_FIELDS).issubset(self.signature.parameters):
            raise ValueError('Explicit native pipeline decode signature required')
        self.pipeline_source = _source(self.original)
        self.prior = vars(self.pipeline).get('_decode_displacement', _ABSENT)
        self.root.mkdir(parents=True, exist_ok=False)
        write_record(self.root/'identity.json', self.identity)
        self.wrapper = MethodType(lambda _pipeline, *a, **kw: self._decode(*a, **kw), self.pipeline)
        self.pipeline._decode_displacement = self.wrapper
        self.pipeline._research_decoder_capture = self
        self.active = True
        return self

    def _decode(self, *args, **kwargs):
        if self.failed:
            raise ValueError('Preserve failed window; start a separately reviewed attempt')
        path = None
        try:
            if self.windows >= self.max_windows:
                raise ValueError('Decoder window bound exceeded')
            bound = self.signature.bind(*args, **kwargs)
            bound.apply_defaults()
            values = bound.arguments
            model = self.pipeline.temporal_3D_vae
            if model is None or model.training:
                raise ValueError('Loaded eval-mode native decoder required at call boundary')
            import numpy as np
            import torch

            anchor = values['anchor_mesh']
            vertices = np.asarray(anchor.vertices)
            faces = np.asarray(anchor.faces)
            target = values['target_alphas']
            if (not isinstance(target, torch.Tensor) or target.ndim != 2
                    or target.shape[0] != 1 or target.shape[1] < 1):
                raise ValueError('Single native sample with all target times required')
            if (vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3
                    or not np.issubdtype(vertices.dtype, np.floating)
                    or vertices.dtype.itemsize > 8
                    or not np.isfinite(vertices).all()):
                raise ValueError('Finite complete native anchor vertices required')
            if (faces.ndim != 2 or faces.shape[1] != 3 or len(faces) < 1
                    or not np.issubdtype(faces.dtype, np.integer)
                    or faces.min() < 0 or faces.max() >= len(vertices)):
                raise ValueError('Native triangle topology required')
            # Conservatively reserve float64 for every returned mesh and anchor.
            geometry_bytes = (target.shape[1] + 1)*vertices.size*8 + faces.size*8
            if geometry_bytes > self.max_geometry_bytes:
                raise ValueError('Full-window geometry exceeds byte bound')
            vertices = vertices.copy()
            faces = faces.copy()
            path = self.root/f'window-{self.windows:04d}'
            path.mkdir(exist_ok=False)
            self.windows += 1
            np.savez(path/'anchor.npz', vertices=vertices, faces=faces)
            record = {
                'kind': 'pipeline-decoder-window', 'version': 1,
                'status': 'started', 'window_index': self.windows-1,
                'pipeline_source': self.pipeline_source,
                'decoder_source': _source(model.forward),
                'identity_sha256': sha256(self.root/'identity.json'),
                'grad_enabled': torch.is_grad_enabled(),
                'native_context_qualified': False, 'replay_qualified': False,
                'scientific_effect_qualification': False,
            }
            write_record(path/'window.json', record)
            identity = dict(record, caller_identity=self.identity)
            with DecoderObserver(model, path/'decoder', identity,
                                 max_bytes=self.max_tensor_bytes) as observer:
                # The official pipeline constructs query XYZ+normals itself.
                # Preserve positional arguments, callback and returned objects.
                meshes = self.original(*args, **kwargs)
            if observer.calls != 1:
                raise ValueError('Exactly one complete decoder call per native window required')
            tensors, call = load_capture(path/'decoder/call-0000')
            if any(not torch.isfinite(value).all().item() for value in tensors.values()):
                raise ValueError('Nonfinite native decoder context or output')
            if not isinstance(meshes, list) or len(meshes) != target.shape[1]:
                raise ValueError('All native target meshes required')
            query = tensors['query']
            output = tensors['output']
            if (tuple(query.shape) != (1, len(vertices), 6)
                    or tuple(output.shape) != (1, len(meshes), len(vertices), 3)):
                raise ValueError('Complete XYZ+normal query and full output required')
            expected_inputs = {'latent': 'latents', 'framestep': 'window_timesteps',
                               'source_alpha': 'source_alpha', 'target_alphas': 'target_alphas'}
            if any(not torch.equal(tensors[k], values[v].detach().cpu())
                   for k, v in expected_inputs.items()):
                raise ValueError('Pipeline and decoder context disagree')
            if not torch.equal(query[0, :, :3], torch.as_tensor(vertices).to(query.dtype)):
                raise ValueError('Decoder query does not cover the original anchor in order')
            mode = call['prediction_mode']
            if mode == 'direct':
                expected = output.clamp(-1, 1)
            elif mode == 'residual':
                expected = (query[:, None, :, :3] + output).clamp(-1, 1)
            else:
                raise ValueError('Unsupported native prediction mode')
            returned = []
            for index, mesh in enumerate(meshes):
                if not np.array_equal(np.asarray(mesh.faces), faces):
                    raise ValueError('Returned mesh topology changed')
                current = np.asarray(mesh.vertices)
                if (current.shape != vertices.shape or current.dtype.itemsize > 8
                        or not np.issubdtype(current.dtype, np.floating)
                        or not np.isfinite(current).all()):
                    raise ValueError('Finite full returned native mesh required')
                # Compare float64 values, not rounded float32 values. Conversion
                # through float64 also supports captured bfloat16 tensors.
                if not np.array_equal(current, expected[0, index].double().numpy()):
                    raise ValueError('Native output-to-mesh conversion differs from pinned semantics')
                returned.append(current)
            np.savez(path/'geometry.npz', anchor_vertices=vertices, faces=faces,
                     vertices=np.stack(returned))
            record.update(status='captured_unqualified', targets=len(meshes),
                          vertices=len(vertices), faces=len(faces),
                          prediction_mode=mode, decoder_calls=observer.calls,
                          output_to_mesh_equal=True,
                          anchor_sha256=sha256(path/'anchor.npz'),
                          geometry_sha256=sha256(path/'geometry.npz'),
                          decoder_record_sha256=sha256(path/'decoder/call-0000/record.json'))
            write_record(path/'window.json', record)
            return meshes
        except BaseException as error:
            self.failed = True
            if path is not None:
                write_record(path/'failure.json', {'status': 'failed',
                    'error_type': type(error).__name__, 'error': str(error)})
            raise

    def __exit__(self, typ, error, tb):
        conflict = self.pipeline._decode_displacement is not self.wrapper
        # Do not overwrite a replacement installed by another owner.
        if not conflict:
            if self.prior is _ABSENT:
                del self.pipeline._decode_displacement
            else:
                self.pipeline._decode_displacement = self.prior
        if getattr(self.pipeline, '_research_decoder_capture', None) is self:
            del self.pipeline._research_decoder_capture
        self.active = False
        failure = None
        if conflict:
            failure = 'Pipeline method changed during capture'
        elif self.failed:
            failure = 'Capture contains a failed window'
        elif self.windows == 0:
            failure = 'No decoder window was observed'
        status = 'failed' if typ is not None or failure else 'captured_unqualified'
        records = [{'path': p.relative_to(self.root).as_posix(),
                    'bytes': p.stat().st_size, 'sha256': sha256(p)}
                   for p in sorted(self.root.rglob('*')) if p.is_file()]
        write_record(self.root/'manifest.json', {
            'kind': 'pipeline-decoder-capture', 'version': 1, 'status': status,
            'windows': self.windows, 'max_windows': self.max_windows,
            'max_tensor_bytes_per_window': self.max_tensor_bytes,
            'max_geometry_bytes_per_window': self.max_geometry_bytes,
            'files': records, 'error': str(error) if error else failure,
            'native_context_qualified': False, 'replay_qualified': False,
            'scientific_effect_qualification': False, 'candidate_methods_tested': False,
        })
        if typ is None and failure:
            raise ValueError(failure)
        return False
