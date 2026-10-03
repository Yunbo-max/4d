"""Evaluation-only ActionBench GT camera convention check; no prediction fitting."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def sha(path):
    return hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()


def project(points, camera, width, height):
    points = np.asarray(points, dtype=np.float64)
    view = points @ np.asarray(camera['R']).T + np.asarray(camera['T'])
    depth = -view[:, 2]
    positive = np.isfinite(view).all(1) & (depth > 0)
    uv = np.full((len(points), 2), np.nan)
    fx, fy = camera['focal_length_ndc']
    px, py = camera['principal_point_ndc']
    uv[positive, 0] = width / 2 * (1 + fx * view[positive, 0] / depth[positive] + px)
    uv[positive, 1] = height / 2 * (1 - fy * view[positive, 1] / depth[positive] - py)
    return uv, positive


def self_test():
    c = dict(R=np.eye(3).tolist(), T=[0,0,0], focal_length_ndc=[2,2], principal_point_ndc=[0,0])
    uv, positive = project(np.array([[0,0,-2],[.5,.25,-2],[0,0,1]],float), c, 100, 100)
    assert np.array_equal(positive, [True,True,False])
    assert np.array_equal(uv[:2], [[50,50],[75,37.5]])
    # Non-symmetric rotation detects accidentally using R instead of R.T.
    c['R'] = [[0,-1,0],[1,0,0],[0,0,1]]
    uv, positive = project(np.array([[.5,0,-2]]),c,100,100)
    assert np.array_equal(uv, [[50,25]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    self_test()
    manifest = json.loads(args.manifest.read_text())
    frozen = dict(frames=[0,8,15], alpha_threshold=127, boundary_tolerance_pixels=2,
                  dilation='5x5 square Chebyshev pixel tolerance', min_inside_dilated_fraction=.99,
                  require_all_depths_positive=True, min_in_bounds_fraction=.999,
                  sampling='all100000GTsurfacepoints; no resampling',
                  guidance=False, camera_fit=False, image_coordinate='nearest integer pixel',
                  limits='Projection containment, not exact mesh IoU or full calibration of an inference camera; hidden GT points also project inside silhouette.')
    (args.output/'protocol.json').write_text(json.dumps(frozen,indent=2)+'\n')
    rows=[]
    sheet=Image.new('RGB',(3*300,8*332),'white')
    for index, uid in enumerate(manifest['uids']):
        root=args.data_root/uid
        camera=json.loads((root/'camera.json').read_text())
        surfaces=np.load(root/'surfaces.npy',mmap_mode='r',allow_pickle=False)
        if surfaces.shape!=(16,100000,6):raise ValueError('Unexpected GT shape')
        for col,t in enumerate(frozen['frames']):
            image_path=root/'imgs'/f'{t:02d}.png'
            rgba=Image.open(image_path).convert('RGBA');w,h=rgba.size
            alpha=np.asarray(rgba)[...,3]>127
            expanded=np.asarray(Image.fromarray((alpha*255).astype('uint8')).filter(ImageFilter.MaxFilter(5)))>0
            uv,positive=project(surfaces[t,:,:3],camera,w,h)
            xy=np.zeros((len(uv),2),dtype=np.int64);xy[positive]=np.rint(uv[positive]).astype(np.int64)
            inside=positive&(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
            hits=np.zeros(len(uv),bool);wide=hits.copy()
            hits[inside]=alpha[xy[inside,1],xy[inside,0]]
            wide[inside]=expanded[xy[inside,1],xy[inside,0]]
            passed=bool(positive.all() and inside.mean()>=.999 and wide.mean()>=.99)
            row=dict(uid=uid,frame=t,all_depths_positive=bool(positive.all()),in_bounds_fraction=float(inside.mean()),
                     inside_alpha_fraction=float(hits.mean()),inside_2px_fraction=float(wide.mean()),qualified=passed,
                     source_sha256={'camera':sha(root/'camera.json'),'image':sha(image_path)})
            rows.append(row)
            background=Image.new('RGBA',rgba.size,'white');background.alpha_composite(rgba);canvas=background.convert('RGB')
            draw=ImageDraw.Draw(canvas)
            for point in np.flatnonzero(inside)[::20]:
                x,y=xy[point];draw.point((int(x),int(y)), fill='#21aa54' if wide[point] else '#ee2255')
            canvas.save(args.output/f'{uid}__frame{t:02d}.png')
            canvas.thumbnail((300,300));sheet.paste(canvas,(col*300,index*332+26))
            ImageDraw.Draw(sheet).text((col*300+4,index*332+4),f'{uid.split("_")[0]} t={t} 2px={wide.mean():.3f}',fill='black')
    sheet.save(args.output/'projection-contact-sheet.png')
    result=dict(status='completed',rows=rows,qualified_count=sum(x['qualified'] for x in rows),denominator=len(rows),
                protocol=frozen,source_sha256=sha(Path(__file__)),manifest_sha256=sha(args.manifest),
                surface_sha256={uid:sha(args.data_root/uid/'surfaces.npy') for uid in manifest['uids']},
                interpretation='Evaluation camera convention only; the GT-fitted native alignment is never used here. No inference-time camera estimated.')
    (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','qualified_count','denominator']}))


if __name__=='__main__':main()
