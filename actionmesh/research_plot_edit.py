"""Plot all asset/seed results; dots are repeated diagnostics, not confidence intervals."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',type=Path,required=True); a=p.parse_args()
    data=[json.loads((a.root/f'edit-seed{s}/results.json').read_text()) for s in [0,1,2]]
    assets=['kangaroo','octopus','mushroom','panda']
    methods=['world_offset','global_rigid_transport','local_rigid_transport','residual_gated_local_global']
    labels=['World offset','Global rigid','Local rigid','Residual gate']
    colors=['#a1a6af','#277da1','#e5a437','#7857a3']
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,reg,title in zip(axes,['two_region_articulation','global_rigid_noisy_observations'],['Articulated controlled motion','Noisy global rigid motion']):
        for i,(method,label,color) in enumerate(zip(methods,labels,colors)):
            y=np.array([[d['cases'][asset]['synthetic_controls'][reg]['methods'][method]['edit_offset_epe_over_scale']*100 for asset in assets] for d in data])
            x=np.arange(4)+(i-1.5)*.19
            ax.bar(x,y.mean(0),width=.17,color=color,label=label,alpha=.8)
            for j in range(3): ax.scatter(x+(j-1)*.025,y[j],color='black',s=9,zorder=4,alpha=.6)
        ax.set_xticks(range(4),assets); ax.set_title(title); ax.set_ylabel('Edit offset error (% of fixed object scale)')
        ax.spines[['top','right']].set_visible(False)
        if reg.endswith('observations'): ax.set_yscale('log')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False)
    fig.suptitle('Idea 3: analytical edit targets on generated shapes\n4 subjects, 3 sampling/noise seeds; no natural 4D quality ground truth',fontsize=12)
    fig.savefig(a.root/'edit-controls.png',dpi=160)
    plt.close(fig)


if __name__=='__main__': main()
