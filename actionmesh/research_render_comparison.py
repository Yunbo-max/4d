"""Render paired mesh geometry with one fixed orthographic camera and scale."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from research_compare import load


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--variants', nargs='+', default=['baseline','cfg_cap','cfg_global','moment_energy','rms_energy'])
    a=p.parse_args()
    # Raw pilot arrays are pipeline/model coordinates. Match the documented
    # official save_deformation export convention (-z, x, y), with z up.
    values=[load(a.root/name) for name in a.variants]
    values=[(v[:,:,[2,0,1]]*np.array([-1,1,1]),f) for v,f in values]
    # Shared camera; fixed world transform, never align frames separately.
    direction=np.array([2.4,-4.0,1.7]); direction/=np.linalg.norm(direction)
    right=np.cross([0,0,1],direction); right/=np.linalg.norm(right)
    up=np.cross(direction,right)
    rot=np.stack([right,up,direction],axis=1)
    coords=[v@rot for v,f in values]
    lo=np.stack([v.min((0,1)) for v in coords]).min(0)
    hi=np.stack([v.max((0,1)) for v in coords]).max(0)
    center=(hi+lo)/2; scale=max((hi-lo)[:2])*1.2
    size=280; frames=[]
    try: font=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',16)
    except OSError: font=ImageFont.load_default()
    for t in range(16):
        im=Image.new('RGB',(size*len(values),size+62),'#f5f7fa'); draw=ImageDraw.Draw(im)
        for k,((v,faces),xyz) in enumerate(zip(values,coords)):
            q=xyz[t]; tri=q[faces]
            normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
            normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
            light=np.array([0.3,0.5,0.8]); light/=np.linalg.norm(light)
            shade=0.4+0.6*np.abs(normals@light)
            xy=(q[:,:2]-center[:2])/scale*size
            xy[:,0]+=size/2+k*size; xy[:,1]=size/2-xy[:,1]+40
            for i in np.argsort(tri[:,:,2].mean(1)):
                col=tuple((np.array([70,135,205])*shade[i]).astype(int))
                draw.polygon([tuple(x) for x in xy[faces[i]]],fill=col)
            draw.text((k*size+10,10),a.variants[k],fill='#152536',font=font)
        draw.text((10,size+42),f'Frame {t:02d}/15 | shared camera and scale | exploratory outputs, no GT',fill='#152536',font=font)
        frames.append(im)
    frames[0].save(a.root/'native-comparison.gif',save_all=True,append_images=frames[1:],duration=180,loop=0)
    grid=Image.new('RGB',(frames[0].width,frames[0].height*4),'white')
    for i,t in enumerate([0,5,10,15]): grid.paste(frames[t],(0,i*frames[0].height))
    grid.save(a.root/'native-comparison.png')


if __name__=='__main__': main()
