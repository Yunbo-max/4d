"""Publish a verified actual-GLB front render as a portable GIF preview."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from PIL import Image


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',type=Path,required=True)
    parser.add_argument('--result',type=Path,required=True)
    args=parser.parse_args()
    report=json.loads((args.check/'render-check.json').read_text())
    if report.get('status')!='verified' or report.get('evaluated_animation_moves') is not True:
        raise ValueError('Actual GLB animation is not verified')
    if report.get('view_index')!=3:
        raise ValueError('Expected this experiment\'s selected front view 3')
    if report.get('source_sha256')!=sha(args.result/'animated_mesh.glb'):
        raise ValueError('Rendered GLB does not match published animation')
    targets=[args.result/n for n in ['animation-front','animated-preview.gif','animated-preview.png','animated-preview.json']]
    if any(p.exists() for p in targets):
        raise FileExistsError('Preview evidence already exists')
    images=[]
    for item in report['rendered_frames']:
        path=args.check/item['file']
        if sha(path)!=item['sha256']:
            raise ValueError('Rendered frame hash mismatch')
        with Image.open(path) as image:
            images.append(image.convert('RGB'))
    if len(images)!=16:
        raise ValueError('Expected all 16 animation frames')
    shutil.copytree(args.check,targets[0])
    images[0].save(targets[1],save_all=True,append_images=images[1:],duration=125,loop=0,optimize=False)
    images[0].save(targets[2])
    with Image.open(targets[1]) as gif:
        if gif.n_frames!=16:
            raise ValueError('GIF lost animation frames')
    targets[3].write_text(json.dumps({'status':'verified','source_glb_sha256':sha(args.result/'animated_mesh.glb'),'render_check_sha256':sha(args.check/'render-check.json'),'preview_sha256':sha(targets[1]),'frames':16,'requested_playback_fps':8,'gif_duration_note':'GIF stores time in centiseconds; requested 125ms may be quantized by encoder.','materials':'Actual imported GLB materials and UVs, preserved','camera_view_index':3},indent=2)+'\n')
    print(json.dumps({'status':'verified','result':str(args.result)}))


if __name__=='__main__':
    main()
