"""Export completed native pilots to animated GLBs using the official exporter."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
import numpy as np
from export_application_result import read_glb


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--variants',nargs='+',required=True); a=p.parse_args()
    env=os.environ.copy(); env['LD_LIBRARY_PATH']=str(a.root/'tools/runtime-libs/usr/lib/x86_64-linux-gnu')+':'+env.get('LD_LIBRARY_PATH','')
    for name in a.variants:
        folder=a.output/name; dest=folder/'animated_mesh.glb'
        if dest.exists(): raise FileExistsError(dest)
        report=json.loads((folder/'report.json').read_text())
        if report['status']!='completed_exploratory_run': raise ValueError('Run incomplete')
        raw=np.load(folder/'deformations_vertices.npy',allow_pickle=False)
        v=raw[:,:,[2,0,1]].copy(); v[:,:,0]*=-1
        src=folder/'export-coordinates.npy'; np.save(src,v,allow_pickle=False)
        cmd=[str(a.root/'tools/blender-3.5.1-linux-x64/blender'),'-b','-t','2','-P',str(a.root/'repo/actionmesh/io/glb_export.py'),'--','--vertices_npy',str(src),'--faces_npy',str(folder/'deformations_faces.npy'),'--output_glb',str(dest),'--fps','8']
        started=time.monotonic()
        with (folder/'export.log').open('x') as log:
            subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
        doc,_=read_glb(dest)
        channels=[c for anim in doc.get('animations',[]) for c in anim.get('channels',[]) if c['target']['path']=='weights']
        target_counts=[len(p.get('targets',[])) for m in doc.get('meshes',[]) for p in m['primitives']]
        if not channels or not target_counts or min(target_counts)==0: raise ValueError('Missing morph animation')
        out={'status':'exported_structure_verified','coordinate_transform':'official save_deformation (-z,x,y); raw pilot arrays unchanged','playback_fps':8,'frames':len(raw),'morph_weight_channels':len(channels),'primitive_target_counts':target_counts,'elapsed_seconds':time.monotonic()-started,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'command':cmd,'verification_scope':'GLB parsing and animated morph structure; not a Blender re-import playback or quality test'}
        (folder/'export.json').write_text(json.dumps(out,indent=2)+'\n'); print(name,out['status'],flush=True)


if __name__=='__main__': main()
