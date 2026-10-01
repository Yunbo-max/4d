"""Export and validate the two additional input-mode experiments."""
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import time
import zipfile

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(os.environ.get('ACTIONMESH_WORKDIR', Path(__file__).resolve().parent)).resolve()
SUITE = ROOT / 'outputs/input-modes-20261001'
BLENDER = ROOT / 'tools/blender-3.5.1-linux-x64/blender'


def glb_json(path):
    data = path.read_bytes()
    magic, version, length = struct.unpack_from('<4sII', data)
    assert magic == b'glTF' and version == 2 and length == len(data)
    size, kind = struct.unpack_from('<I4s', data, 12)
    assert kind == b'JSON'
    return json.loads(data[20:20 + size])


def texture_pixels(path):
    data = path.read_bytes()
    size = struct.unpack_from('<I', data, 12)[0]
    doc = json.loads(data[20:20 + size])
    view = doc['bufferViews'][doc['images'][0]['bufferView']]
    start = 28 + size + view.get('byteOffset', 0)
    return np.asarray(Image.open(io.BytesIO(data[start:start + view['byteLength']])).convert('RGBA'))


def main():
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = str(ROOT / 'tools/runtime-libs/usr/lib/x86_64-linux-gnu') + ':' + env.get('LD_LIBRARY_PATH', '')
    for name in ['kangaroo-video', 'panda-video-mesh']:
        out = SUITE / name
        published = ROOT / 'publication/input-modes' / name
        if (published / 'SHA256SUMS').exists():
            for line in (published / 'SHA256SUMS').read_text().splitlines():
                digest, filename = line.split(maxsplit=1)
                assert hashlib.sha256((published / filename).read_bytes()).hexdigest() == digest
            print('ALREADY_EXPORTED', name, flush=True)
            continue
        deadline = time.monotonic() + 3600
        while True:
            if (SUITE / 'status.json').exists():
                statuses = json.loads((SUITE / 'status.json').read_text())
                if any(s['mode'] == name and s['exit_code'] != 0 for s in statuses):
                    raise RuntimeError('Inference failed: ' + name)
                if any(s['mode'] == name and s['exit_code'] == 0 for s in statuses):
                    break
            if time.monotonic() > deadline:
                raise TimeoutError('Waiting for inference: ' + name)
            time.sleep(10)
        report = json.loads((out / 'report.json').read_text())
        assert report['status'] == 'verified'
        panda = name.startswith('panda')
        source = ROOT / 'repo/assets/examples' / ('panda' if panda else 'kangaroo')
        cmd = [str(BLENDER), '-b', '-P', str(ROOT / 'repo/actionmesh/io/glb_export.py'), '--', '--vertices_npy', str(out / 'deformations_vertices.npy'), '--faces_npy', str(out / 'deformations_faces.npy'), '--output_glb', str(out / 'animated_mesh.glb'), '--fps', '8']
        if panda:
            import trimesh
            from actionmesh.io.mesh_io import load_glb
            anchor = load_glb(str(source / 'panda.glb'))
            restored = trimesh.Trimesh(vertices=np.load(out / 'deformations_vertices.npy')[0], faces=anchor.faces.copy(), visual=anchor.visual.copy(), process=False)
            restored.export(str(out / 'anchor-for-export.glb'))
            cmd += ['--input_glb', str(out / 'anchor-for-export.glb')]
        with (out / 'export.log').open('w') as log:
            subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
            doc = glb_json(out / 'animated_mesh.glb')
            assert doc.get('animations') and doc.get('meshes')
            if panda:
                assert doc.get('textures') and doc.get('images'), 'Textured input must keep texture in animated GLB'
                original_pixels = texture_pixels(source / 'panda.glb')
                assert np.array_equal(original_pixels, texture_pixels(out / 'animated_mesh.glb'))
                report['texture_pixels_preserved'] = True
                report['texture_size'] = [int(original_pixels.shape[1]), int(original_pixels.shape[0])]
            subprocess.run([str(BLENDER), '-b', '-t', '8', '-P', str(ROOT / 'render_preview.py'), '--', str(out)] + (['--reverse-view'] if panda else []), env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
        frames = []
        previews = sorted((out / 'preview_frames').glob('*.png'))
        assert len(previews) == 16
        import cv2
        cap = cv2.VideoCapture(str(SUITE / 'inputs' / ('panda.mp4' if panda else 'kangaroo.mp4')))
        for preview in previews:
            ok, rgb = cap.read()
            assert ok
            inp = Image.fromarray(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
            frame = Image.new('RGB', (1024, 552), '#eef1f6')
            frame.paste(inp, (0, 40))
            im = Image.open(preview).convert('RGBA')
            frame.paste(im, (512, 40), im)
            draw = ImageDraw.Draw(frame)
            draw.text((20, 12), 'Actual MP4 input', fill='#24334a')
            draw.text((532, 12), 'Generated geometry (blue preview material)', fill='#24334a')
            frames.append(frame)
        cap.release()
        frames[0].save(out / 'preview.png')
        frames[0].save(out / 'preview.gif', save_all=True, append_images=frames[1:], duration=125, loop=0, optimize=False)
        np.savez_compressed(out / 'deformations.npz', vertices=np.load(out / 'deformations_vertices.npy'), faces=np.load(out / 'deformations_faces.npy'))
        with zipfile.ZipFile(out / 'per-frame-meshes.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(out.glob('mesh_*.glb')):
                archive.write(path, path.name)
        report.update({'animations_in_glb': len(doc['animations']), 'embedded_texture_images': len(doc.get('images', [])), 'preview_note': 'Geometry preview uses blue material; animated GLB retains input texture for panda. Per-frame meshes are geometry-only.'})
        report['timing_scope'] = 'Official entrypoint including preprocessing and, when installed, PyTorch3D GPU rendering; excludes Blender export and file transfer.'
        report['pytorch3d_available_during_run'] = 'PyTorch3D is not installed' not in (out / 'inference.log').read_text()
        (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        dest = ROOT / 'publication/input-modes' / name
        dest.mkdir(parents=True, exist_ok=False)
        for filename in ['animated_mesh.glb', 'preview.gif', 'preview.png', 'deformations.npz', 'per-frame-meshes.zip', 'report.json', 'runtime.json', 'command.json', 'gpu-memory.csv', 'inference.log', 'export.log']:
            shutil.copy2(out / filename, dest / filename)
        for path in out.glob('*.mp4'):
            shutil.copy2(path, dest / path.name)
        (dest / 'SHA256SUMS').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n' for p in sorted(dest.iterdir())))
        print('EXPORTED', name, flush=True)
    dest = ROOT / 'publication/input-modes'
    shutil.copytree(SUITE / 'inputs', dest / 'inputs', dirs_exist_ok=True)
    shutil.copy2(SUITE / 'unsupported-inputs.json', dest / 'unsupported-inputs.json')
    shutil.copy2(SUITE / 'status.json', dest / 'status.json')
    shutil.copy2(SUITE / 'environment.json', dest / 'environment.json')


if __name__ == '__main__':
    main()
