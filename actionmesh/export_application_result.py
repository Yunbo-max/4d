"""Export one verified application backend run with Blender 3.5.1.

The preview compares the exact selected source frames with generated geometry.
Preview/GLB playback FPS is explicit; it is not the source video's FPS or a
claim about ActionMesh's temporal sampling. No network or model loading occurs.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time
import zipfile


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def read_glb(path):
    data = path.read_bytes()
    if len(data) < 20:
        raise ValueError(f'Truncated GLB: {path}')
    magic, version, size = struct.unpack_from('<4sII', data)
    if magic != b'glTF' or version != 2 or size != len(data):
        raise ValueError(f'Invalid GLB header: {path}')
    cursor, document, binary = 12, None, None
    while cursor < len(data):
        length, kind = struct.unpack_from('<I4s', data, cursor)
        chunk = data[cursor + 8:cursor + 8 + length]
        if len(chunk) != length:
            raise ValueError(f'Truncated GLB chunk: {path}')
        if kind == b'JSON':
            document = json.loads(chunk)
        elif kind == b'BIN\x00':
            binary = chunk
        cursor += 8 + length
    if document is None:
        raise ValueError(f'GLB JSON missing: {path}')
    return document, binary


def texture_pixel_hashes(path):
    from PIL import Image
    doc, binary = read_glb(path)
    results = []
    for item in doc.get('images', []):
        if 'bufferView' not in item or binary is None:
            raise ValueError('Texture validation requires embedded GLB images')
        view = doc['bufferViews'][item['bufferView']]
        start = view.get('byteOffset', 0)
        with Image.open(io.BytesIO(binary[start:start + view['byteLength']])) as image:
            decoded = image.convert('RGBA')
            results.append((decoded.size, hashlib.sha256(decoded.tobytes()).hexdigest()))
    return Counter(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(os.environ.get('ACTIONMESH_WORKDIR', Path(__file__).resolve().parent)))
    parser.add_argument('--output', type=Path, required=True, help='Verified backend output directory')
    parser.add_argument('--blender', type=Path, help='Defaults to --root/tools/blender-3.5.1-linux-x64/blender')
    parser.add_argument('--fps', type=int, default=8, help='Explicit playback FPS for GIF and GLB (does not change inference)')
    parser.add_argument('--reverse-view', action='store_true', help='Reverse the geometry preview camera')
    parser.add_argument('--publish', type=Path, help='Optional new directory for portable results, inputs and records')
    args = parser.parse_args()
    if args.fps < 1 or args.fps > 100:
        parser.error('--fps must be in [1, 100]')
    root, out = args.root.expanduser().resolve(), args.output.expanduser().resolve()
    blender = (args.blender or root / 'tools/blender-3.5.1-linux-x64/blender').expanduser().resolve()
    if not blender.is_file():
        raise FileNotFoundError(blender)
    report = json.loads((out / 'report.json').read_text())
    state = json.loads((out / 'status.json').read_text())
    if report.get('status') != 'verified' or state.get('status') != 'verified':
        raise ValueError('Export requires a successfully verified backend run')
    if args.publish and args.publish.exists():
        raise FileExistsError(f'Publication directory already exists: {args.publish}')
    artifacts = ['animated_mesh.glb', 'preview.png', 'preview.gif', 'deformations.npz', 'per-frame-meshes.zip', 'preview_frames', 'export.json', 'export.log']
    for name in artifacts:
        if (out / name).exists():
            raise FileExistsError(f'Refusing to replace an existing artifact: {out / name}')
    # A failed attempt remains reviewable and cannot be silently overwritten.
    with (out / 'export-started.json').open('x') as stream:
        json.dump({'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'playback_fps': args.fps}, stream)
    import numpy as np
    from PIL import Image, ImageDraw, ImageOps
    vertices = np.load(out / 'deformations_vertices.npy', allow_pickle=False)
    faces = np.load(out / 'deformations_faces.npy', allow_pickle=False)
    inputs = json.loads((out / 'input.json').read_text())
    if len(vertices) != len(inputs['frames']) or len(vertices) != 16 or not np.isfinite(vertices).all():
        raise ValueError('Input/output frame count or geometry validation failed')
    for item in inputs['frames']:
        if sha256(out / item['file']) != item['sha256']:
            raise ValueError('Selected source frame changed after inference')
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = str(root / 'tools/runtime-libs/usr/lib/x86_64-linux-gnu') + ':' + env.get('LD_LIBRARY_PATH', '')
    commands = []
    cmd = [str(blender), '-b', '-P', str(root / 'repo/actionmesh/io/glb_export.py'), '--',
           '--vertices_npy', str(out / 'deformations_vertices.npy'), '--faces_npy', str(out / 'deformations_faces.npy'),
           '--output_glb', str(out / 'animated_mesh.glb'), '--fps', str(args.fps)]
    mesh = out / 'input-mesh.glb'
    if mesh.is_file():
        import trimesh
        sys.path.insert(0, str(root / 'repo'))
        from actionmesh.io.mesh_io import load_glb
        anchor = load_glb(str(mesh))
        if len(anchor.vertices) != vertices.shape[1] or not np.array_equal(anchor.faces, faces):
            raise ValueError('Input mesh topology does not match output')
        restored = trimesh.Trimesh(vertices=vertices[0], faces=anchor.faces.copy(), visual=anchor.visual.copy(), process=False)
        restored.export(str(out / 'anchor-for-export.glb'))
        cmd += ['--input_glb', str(out / 'anchor-for-export.glb')]
    started = time.monotonic()
    with (out / 'export.log').open('x') as log:
        commands.append(cmd)
        subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
        doc, _ = read_glb(out / 'animated_mesh.glb')
        if not doc.get('animations') or not doc.get('meshes'):
            raise ValueError('Exported GLB has no mesh animation')
        if mesh.is_file():
            original = texture_pixel_hashes(mesh)
            if original:
                if not doc.get('textures') or original != texture_pixel_hashes(out / 'animated_mesh.glb'):
                    raise ValueError('Input embedded texture pixels were not preserved')
                report['texture_pixels_preserved'] = True
                report['input_embedded_texture_images'] = sum(original.values())
            else:
                report['texture_pixels_preserved'] = None
                report['texture_note'] = 'The input GLB has no embedded texture images.'
        cmd = [str(blender), '-b', '-t', '8', '-P', str(root / 'render_preview.py'), '--', str(out)]
        if args.reverse_view:
            cmd.append('--reverse-view')
        commands.append(cmd)
        subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
    previews = sorted((out / 'preview_frames').glob('*.png'))
    if len(previews) != len(vertices):
        raise ValueError('Missing preview frames')
    frames = []
    for item, preview in zip(inputs['frames'], previews):
        canvas = Image.new('RGB', (1024, 552), '#eef1f6')
        with Image.open(out / item['file']) as source:
            actual = ImageOps.contain(source.convert('RGB'), (512, 512))
            canvas.paste(actual, ((512 - actual.width) // 2, 40 + (512 - actual.height) // 2))
        with Image.open(preview) as rendered:
            geometry = rendered.convert('RGBA')
            canvas.paste(geometry, (512, 40), geometry)
        draw = ImageDraw.Draw(canvas)
        draw.text((20, 12), f'Actual generated video: frame {item["source_frame_index"]}', fill='#24334a')
        draw.text((532, 12), 'ActionMesh geometry (blue preview material)', fill='#24334a')
        frames.append(canvas)
    frames[0].save(out / 'preview.png')
    duration = round(1000 / args.fps)
    frames[0].save(out / 'preview.gif', save_all=True, append_images=frames[1:], duration=duration, loop=0, optimize=False)
    with Image.open(out / 'preview.gif') as gif:
        if gif.n_frames != len(frames):
            raise ValueError('Exported GIF frame count changed')
    np.savez_compressed(out / 'deformations.npz', vertices=vertices, faces=faces)
    with zipfile.ZipFile(out / 'per-frame-meshes.zip', 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.glob('mesh_*.glb')):
            archive.write(path, path.name)
    report.update({'animations_in_glb': len(doc['animations']), 'embedded_texture_images': len(doc.get('images', [])),
                   'preview_note': 'Left: exactly the frames supplied to ActionMesh after explicit temporal sampling. Right: generated geometry with a fixed blue preview material. Animated GLB retains input textures when an input mesh has embedded textures.',
                   'preview_playback_fps': args.fps, 'source_video_fps': inputs['source_fps'],
                   'playback_timing_note': 'Chosen export playback FPS; source video timing is not preserved. This setting does not alter inference.'})
    write_json(out / 'report.json', report)
    write_json(out / 'export.json', {'status': 'verified', 'commands': commands, 'elapsed_seconds': time.monotonic() - started,
                                   'playback_fps': args.fps, 'gif_requested_frame_duration_ms': duration,
                                   'source_video_fps': inputs['source_fps'], 'render_device': 'CPU',
                                   'preview_camera': 'reverse' if args.reverse_view else 'default'})
    if args.publish:
        dest = args.publish.expanduser().resolve()
        dest.mkdir(parents=True, exist_ok=False)
        names = ['animated_mesh.glb', 'preview.gif', 'preview.png', 'deformations.npz', 'per-frame-meshes.zip',
                 'report.json', 'runtime.json', 'command.json', 'runner-command.json', 'gpu-memory.csv',
                 'inference.log', 'export.log', 'export.json', 'environment.json', 'input.json', 'status.json',
                 'weights-verified.json', 'mesh-input.json', 'input-mesh.glb']
        for name in names:
            if (out / name).is_file():
                shutil.copy2(out / name, dest / name)
        for path in out.glob('source-video.*'):
            shutil.copy2(path, dest / path.name)
        shutil.copytree(out / 'input-frames', dest / 'input-frames')
        (dest / 'SHA256SUMS').write_text(''.join(f'{sha256(path)}  {path.relative_to(dest)}\n' for path in sorted(dest.rglob('*')) if path.is_file()))
    print(json.dumps({'status': 'export_verified', 'output': str(out), 'publication': str(args.publish) if args.publish else None}), flush=True)


if __name__ == '__main__':
    main()
