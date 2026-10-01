"""Export an official ActionMesh run to animated GLB, GIF and a portable archive."""
import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import zipfile
import numpy as np
from PIL import Image, ImageDraw

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--blender', required=True, type=Path)
    parser.add_argument('--library-dir', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    out = args.output.resolve()
    blender = str(args.blender.resolve())
    env = os.environ.copy()
    if args.library_dir:
        env['LD_LIBRARY_PATH'] = str(args.library_dir.resolve()) + ':' + env.get('LD_LIBRARY_PATH', '')
    subprocess.run([blender, '-b', '-P', str(root / 'repo/actionmesh/io/glb_export.py'), '--', '--vertices_npy', str(out / 'deformations_vertices.npy'), '--faces_npy', str(out / 'deformations_faces.npy'), '--output_glb', str(out / 'animated_mesh.glb'), '--fps', '8'], env=env, check=True, timeout=300)
    glb = (out / 'animated_mesh.glb').read_bytes()
    magic, version, length = struct.unpack_from('<4sII', glb)
    assert magic == b'glTF' and version == 2 and length == len(glb)
    json_len, chunk_type = struct.unpack_from('<I4s', glb, 12)
    assert chunk_type == b'JSON'
    document = json.loads(glb[20:20 + json_len])
    assert document.get('animations') and document.get('meshes'), 'GLB must contain animation and geometry'
    subprocess.run([blender, '-b', '-t', '8', '-P', str(root / 'render_preview.py'), '--', str(out)], env=env, check=True, timeout=1800)
    frames = []
    inputs = sorted((root / 'repo/assets/examples/kangaroo').glob('*.png'))
    previews = sorted((out / 'preview_frames').glob('*.png'))
    assert len(previews) == 16 and len(inputs) == 16
    for inp, preview in zip(inputs, previews):
        frame = Image.new('RGB', (1024, 552), '#eef1f6')
        for x, path in [(0, inp), (512, preview)]:
            im = Image.open(path).convert('RGBA')
            im.thumbnail((512, 512))
            frame.paste(im, (x + (512 - im.width) // 2, 40 + (512 - im.height) // 2), im)
        draw = ImageDraw.Draw(frame)
        draw.text((20, 12), 'Input video', fill='#24334a')
        draw.text((532, 12), 'ActionMesh generated geometry', fill='#24334a')
        frames.append(frame)
    frames[0].save(out / 'preview.png')
    frames[0].save(out / 'preview.gif', save_all=True, append_images=frames[1:], duration=125, loop=0, optimize=False)
    v = np.load(out / 'deformations_vertices.npy')
    f = np.load(out / 'deformations_faces.npy')
    np.savez_compressed(out / 'deformations.npz', vertices=v, faces=f)
    with zipfile.ZipFile(out / 'per-frame-meshes.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.glob('mesh_*.glb')):
            archive.write(path, path.name)
    result = {'frames': len(previews), 'animations_in_glb': len(document['animations']), 'animation_fps': 8, 'preview': 'Left: supplied official example. Right: generated geometry rendered with fixed lighting; color is a visualization material, not inferred texture.'}
    (out / 'export.json').write_text(json.dumps(result, indent=2) + '\n')
    print('EXPORT_VERIFIED', json.dumps(result))

if __name__ == '__main__':
    main()
