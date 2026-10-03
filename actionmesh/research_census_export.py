"""Export an immutable completed census case to a NEW animated GLB directory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from export_application_result import read_glb


SOURCE_FILES = ('sequence.npz', 'deformations_vertices.npy', 'deformations_faces.npy')


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def load_case(case_dir):
    report = json.loads((case_dir / 'report.json').read_text())
    if report.get('status') != 'completed' or report.get('frames') != 16:
        raise ValueError('Export requires a completed 16-frame census report')
    hashes = {name: digest(case_dir / name) for name in (*SOURCE_FILES, 'report.json')}
    for name in SOURCE_FILES:
        if report.get('sha256', {}).get(name) != hashes[name]:
            raise ValueError('Missing or mismatched source hash: ' + name)
    vertices = np.load(case_dir / 'deformations_vertices.npy', allow_pickle=False)
    faces = np.load(case_dir / 'deformations_faces.npy', allow_pickle=False)
    if vertices.ndim != 3 or vertices.shape[0] != 16 or vertices.shape[2] != 3 or vertices.shape[1] < 3:
        raise ValueError('Expected vertices with shape [16,V,3]')
    if not np.issubdtype(vertices.dtype, np.floating) or not np.isfinite(vertices).all():
        raise ValueError('Expected finite floating-point geometry')
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0 or not np.issubdtype(faces.dtype, np.integer):
        raise ValueError('Expected integer triangle topology [F,3]')
    if faces.min() < 0 or faces.max() >= vertices.shape[1]:
        raise ValueError('Face index outside vertex array')
    if np.any(np.diff(np.sort(faces, axis=1), axis=1) == 0):
        raise ValueError('Triangle repeats a vertex index')
    with np.load(case_dir / 'sequence.npz', allow_pickle=False) as saved:
        if not np.array_equal(saved['vertices'], vertices) or not np.array_equal(saved['faces'], faces):
            raise ValueError('Sequence/deformation geometry or topology mismatch')
        if not np.array_equal(saved['frame_indices'], np.arange(16)):
            raise ValueError('Expected native frame indices 0..15')
        if not np.array_equal(saved['timesteps'], np.arange(16)):
            raise ValueError('Expected native timesteps 0..15')
    for field, actual in (('vertices_per_frame', vertices.shape[1]), ('faces', len(faces))):
        if field in report and report[field] != actual:
            raise ValueError('Report geometry count mismatch: ' + field)
    return vertices, faces, report, hashes


def scalar_f32(document, binary, index):
    accessor = document['accessors'][index]
    if (accessor.get('componentType') != 5126 or accessor.get('type') != 'SCALAR'
            or 'sparse' in accessor or binary is None):
        raise ValueError('Expected embedded scalar float32 animation accessor')
    view = document['bufferViews'][accessor['bufferView']]
    if view.get('buffer', 0) != 0:
        raise ValueError('Unexpected external animation buffer')
    offset = view.get('byteOffset', 0) + accessor.get('byteOffset', 0)
    stride = view.get('byteStride', 4)
    count = accessor['count']
    end = offset + (count - 1) * stride + 4
    if count < 1 or stride < 4 or end > len(binary) or end > view.get('byteOffset', 0) + view['byteLength']:
        raise ValueError('Invalid animation accessor bounds')
    return np.ndarray((count,), dtype='<f4', buffer=binary, offset=offset, strides=(stride,)).copy()


def verify_glb(path):
    document, binary = read_glb(path)
    channels = []
    for animation in document.get('animations', []):
        for channel in animation.get('channels', []):
            if channel['target'].get('path') != 'weights':
                continue
            node = document['nodes'][channel['target']['node']]
            primitives = document['meshes'][node['mesh']]['primitives']
            counts = [len(primitive.get('targets', [])) for primitive in primitives]
            if not counts or any(count != 16 for count in counts):
                raise ValueError('Expected exactly 16 morph targets per animated primitive')
            sampler = animation['samplers'][channel['sampler']]
            times = scalar_f32(document, binary, sampler['input'])
            weights = scalar_f32(document, binary, sampler['output'])
            if not np.array_equal(times, np.arange(16, dtype=np.float32) / 8):
                raise ValueError('Expected 16 animation times from 0 to 15/8 seconds')
            if sampler.get('interpolation', 'LINEAR') != 'LINEAR' or len(weights) != 256 or not np.isfinite(weights).all():
                raise ValueError('Invalid 16-frame morph-weight animation')
            channels.append({'node': channel['target']['node'], 'primitive_target_counts': counts,
                             'keyframes': len(times), 'first_time_seconds': float(times[0]),
                             'last_time_seconds': float(times[-1]), 'weight_count': len(weights)})
    if not channels:
        raise ValueError('Missing animated morph-weight channel')
    return channels


def export(root, case_dir, output):
    root, case_dir, output = (path.expanduser().resolve() for path in (root, case_dir, output))
    if output == case_dir or case_dir in output.parents:
        raise ValueError('Export output must be separate from the immutable case directory')
    if output.exists():
        raise FileExistsError('Export identity already exists: ' + str(output))
    raw, faces, source_report, source_hashes = load_case(case_dir)
    blender = root / 'tools/blender-3.5.1-linux-x64/blender'
    official_exporter = root / 'repo/actionmesh/io/glb_export.py'
    for path in (blender, official_exporter):
        if not path.is_file():
            raise FileNotFoundError(path)
    output.mkdir(parents=True, exist_ok=False)
    state = {'status': 'exporting', 'source_case_dir': str(case_dir),
             'source_uid': source_report.get('uid'), 'source_seed': source_report.get('seed'),
             'source_hashes': source_hashes, 'exporter_sha256': digest(Path(__file__)),
             'official_exporter_sha256': digest(official_exporter),
             'coordinate_transform': 'official save_deformation (-z,x,y); source arrays unchanged',
             'playback_fps': 8, 'frames': 16, 'device': 'CPU', 'blender_threads': 2,
             'playback_note': '8 FPS export timing; native inference and evaluation arrays are unchanged.',
             'verification_scope': 'Source hashes and arrays plus GLB binary parsing, 16 morph targets and 16 animation times; not a visual quality or Blender re-import playback test.'}

    def save():
        temporary = output / 'export.json.tmp'
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(output / 'export.json')

    save()
    started = time.monotonic()
    try:
        transformed = raw[:, :, [2, 0, 1]].copy()
        transformed[:, :, 0] *= -1
        np.save(output / 'export-coordinates.npy', transformed, allow_pickle=False)
        np.save(output / 'export-faces.npy', faces, allow_pickle=False)
        destination = output / 'animated_mesh.glb'
        command = [str(blender), '-b', '-t', '2', '-P', str(official_exporter), '--',
                   '--vertices_npy', str(output / 'export-coordinates.npy'),
                   '--faces_npy', str(output / 'export-faces.npy'),
                   '--output_glb', str(destination), '--fps', '8']
        state['command'] = command
        save()
        environment = os.environ.copy()
        environment['CUDA_VISIBLE_DEVICES'] = ''
        environment['LD_LIBRARY_PATH'] = str(root / 'tools/runtime-libs/usr/lib/x86_64-linux-gnu') + ':' + environment.get('LD_LIBRARY_PATH', '')
        with (output / 'export.log').open('x') as log:
            subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
        state['morph_weight_channels'] = verify_glb(destination)
        after = {name: digest(case_dir / name) for name in source_hashes}
        if source_hashes != after:
            raise ValueError('Source changed during export')
        state.update(status='exported_structure_verified', source_hashes_unchanged=True,
                     sha256={name: digest(output / name) for name in
                             ('animated_mesh.glb', 'export-coordinates.npy', 'export-faces.npy')})
    except BaseException as exc:
        state.update(status='failed', error=repr(exc))
        raise
    finally:
        state['elapsed_seconds'] = time.monotonic() - started
        save()
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--case-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    state = export(args.root, args.case_dir, args.output)
    print(json.dumps({'status': state['status'], 'output': str(args.output)}), flush=True)


if __name__ == '__main__':
    main()
