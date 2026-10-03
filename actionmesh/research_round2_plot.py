#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--skip-preview',action='store_true');a=p.parse_args();r=a.root
summary=json.loads((r/'summary.json').read_text())
methods=[('global','Global'),('local','Local'),('heldout','Held-out 3D'),('spatial_action_q50','Spatial split'),('utility','Idea U: edit utility'),('directional_covariance_q50','Idea G: geometry'),('temporal_action_q50','Idea T: time paths'),('pca_normal','PCA normal')]
fig,axes=plt.subplots(1,3,figsize=(13,5),layout='constrained')
for ax,key,title in zip(axes,['generated/mixed','normal/normal_clean','normal/normal_noisy'],['Constructed motion controls','ActionBench relief edit: clean','ActionBench relief edit: noisy']):
 group=summary['groups'][key];pairs=[(m,label) for m,label in methods if m in group['macro']]
 vals=[1000*group['macro'][m] for m,_ in pairs];y=np.arange(len(pairs))
 colors=['#287aab' if m in ('utility','directional_covariance_q50','temporal_action_q50') else '#a9b1b9' for m,_ in pairs]
 ax.barh(y,vals,color=colors,height=.6)
 for i,(m,_) in enumerate(pairs):
  v=[1000*x[m] for x in group['asset_means'].values()];ax.scatter(v,np.full(len(v),i),s=18,c='#222222',zorder=3)
 ax.set_yticks(y,[label for _,label in pairs]);ax.invert_yaxis();ax.set_title(title,fontsize=11);ax.set_xlabel('Edit-offset EPE / fixed diagonal × 1000')
 ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.18)
fig.suptitle('Three training-free output-space editing probes | lower error is better',fontsize=14)
fig.get_layout_engine().set(rect=(0,.08,1,.90));fig.text(.5,.015,'Bars: equal-asset means. Dots: 4 assets per panel, averaging 3 sampling seeds. Constructed and published inputs are different tasks.',ha='center',fontsize=9)
fig.savefig(r/'comparison.png',dpi=160);fig.savefig(r/'comparison.pdf');plt.close(fig)
# Fixed first UID/seed17, chosen before outcomes, no favorable example search.
files=sorted((r/'pilot').glob('*__s17__normal_clean__normal-tracks.npz'))
if files and not a.skip_preview:
 pth=files[0];z=np.load(pth,allow_pickle=False);names=['target','local','heldout','utility','directional_covariance_q50','temporal_action_q50']
 titles=['Target: supplied normals','Local transport','Held-out 3D selector','Idea U: edit utility','Idea G: geometry','Idea T: time paths']
 fig=plt.figure(figsize=(10,6));axes=[fig.add_subplot(2,3,i+1,projection='3d') for i in range(6)]
 allpts=z['source'];center=(allpts.max((0,1))+allpts.min((0,1)))/2;radius=np.ptp(allpts,axis=(0,1)).max()*.55
 def draw(frame):
  for ax,name,title in zip(axes,names,titles):
   ax.clear();pts=z[name][frame];ax.scatter(*pts.T,c=z['source'][0,:,2],cmap='viridis',s=3,depthshade=False)
   ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),zlim=(center[2]-radius,center[2]+radius));ax.set_box_aspect([1,1,1]);ax.view_init(18,45);ax.set_axis_off();ax.set_title(title,fontsize=10)
  fig.suptitle(f'Sampled surface-relief edit | fixed first ActionBench UID | frame {frame+1}/16',fontsize=12)
 draw(8);fig.savefig(r/'relief-preview.png',dpi=130)
 FuncAnimation(fig,draw,frames=16,interval=180).save(r/'relief-preview.gif',writer=PillowWriter(fps=6));plt.close(fig)
 (r/'preview-source.json').write_text(json.dumps({'input':pth.name,'selection':'predeclared first UID seed17 clean','representation':'512 sampled points, not fullmesh or newmodelgeneration'},indent=2))
print('Saved comparison and fixed-example relief preview')
