#!/usr/bin/env python3
"""Descriptive asset-macro summaries, no significance or model-quality inference."""
import argparse,csv,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('result',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
r=json.loads(a.result.read_text());cells=r['cells'];a.output.mkdir(parents=True,exist_ok=True)
def means(cs):
 by={}
 for c in cs:
  for m,v in c['methods'].items():
   if 'epe' in v:by.setdefault(c['asset'],{}).setdefault(m,[]).append(v['epe'])
 per={asset:{m:statistics.mean(v) for m,v in methods.items()} for asset,methods in by.items()}
 overall={m:statistics.mean(v[m] for v in per.values() if m in v) for m in sorted({m for v in per.values() for m in v})}
 return {'asset_means':per,'macro':overall,'cells':len(cs),'independent_assets':len(per)}
s={}
for kind in ['generated','normal']:
 for regime in sorted({c['regime'] for c in cells if c['kind']==kind}):
  s[f'{kind}/{regime}']=means([c for c in cells if c['kind']==kind and c['regime']==regime])
mixed=means([c for c in cells if c['regime'] in ['articulation','articulation_noisy','global_rigid_noisy']]);s['generated/mixed']=mixed
m=mixed['macro'];noisy=s['generated/global_rigid_noisy']['macro'];clean=s['generated/articulation']['macro']
def gain(candidate,baseline,values=m):return 1-values[candidate]/values[baseline] if values[baseline]>0 else None
def wins(candidate,baseline):return sum(v[candidate]<v[baseline] for v in mixed['asset_means'].values())
fixed=min(['local','global'],key=lambda n:m[n]);scalar=min(['scalar_covariance_q50','fit_residual_q50'],key=lambda n:m[n])
weak={}
for c in cells:
 if c['kind']=='generated' and c['regime']=='natural' and c['direction']=='pca_major':weak.setdefault(c['asset'],[]).append(c['weak_edited_count'])
weakassets=sum(any(x>0 for x in arr) for arr in weak.values())
u={'gain_vs_heldout':gain('utility','heldout'),'asset_wins_vs_heldout':wins('utility','heldout'),'gain_vs_best_fixed':gain('utility',fixed),'best_fixed':fixed,'global_noisy_regression_vs_heldout':-gain('utility','heldout',noisy)}
u['survives_declared_pilot']=u['gain_vs_heldout']>=.1 and u['asset_wins_vs_heldout']>=3 and u['gain_vs_best_fixed']>0 and u['global_noisy_regression_vs_heldout']<=.1
g={'gain_vs_best_scalar':gain('directional_covariance_q50',scalar),'best_scalar':scalar,'asset_wins_vs_best_scalar':wins('directional_covariance_q50',scalar),'clean_articulation_regression':-gain('directional_covariance_q50',scalar,clean),'weak_edited_counts_by_asset':weak,'weak_eligible_assets':weakassets,'gain_vs_spatial':gain('directional_covariance_q50','spatial_action_q50')}
g['survives_declared_pilot']=g['gain_vs_best_scalar']>=.1 and g['asset_wins_vs_best_scalar']>=3 and g['clean_articulation_regression']<=.05 and weakassets>=3 and g['gain_vs_spatial']>.05
t={'gain_vs_fit_residual':gain('temporal_action_q50','fit_residual_q50'),'gain_vs_matrix':gain('temporal_action_q50','temporal_matrix_q50'),'asset_wins_vs_fit_residual':wins('temporal_action_q50','fit_residual_q50'),'asset_wins_vs_matrix':wins('temporal_action_q50','temporal_matrix_q50'),'gain_vs_spatial':gain('temporal_action_q50','spatial_action_q50')}
bestnoisy=min(['local','global','heldout','fit_residual_q50','temporal_matrix_q50','spatial_action_q50'],key=lambda k:noisy[k]);t['noisy_strongest_baseline']=bestnoisy;t['global_noisy_regression']=-gain('temporal_action_q50',bestnoisy,noisy)
t['survives_declared_pilot']=t['gain_vs_fit_residual']>=.05 and t['gain_vs_matrix']>=.05 and min(t['asset_wins_vs_fit_residual'],t['asset_wins_vs_matrix'])>=3 and t['gain_vs_spatial']>.01 and t['global_noisy_regression']<=.05
summary={'groups':s,'decisions':{'utility':u,'geometry':g,'temporal':t},'status':r['status'],'cells':len(cells),'failures':r['failures'],'gpu':{k:r[k] for k in ['device','peak_allocated_bytes','peak_reserved_bytes','elapsed_seconds']},'notes':['Descriptive4assets pergroup; seeds/directions are repeatedconditions','Normals task uses observedGTgeometry and independentfuture-normal edit targets, notgenerated-video benchmark','No currentnatural-failureprevalence, secondgeneratororformalGateA']}
summary['checks']={'all_finite':all(v['finite'] for c in cells for v in c['methods'].values()),'anchor_error_max':max(v['anchor_max_error'] for c in cells for v in c['methods'].values()),'unedited_shift_max':max(v['unedited_max_shift'] for c in cells for v in c['methods'].values()),'axis_identity_error_max':max(c['axis_identity_max_error'] for c in cells),'global_rigid_worst_transport_epe':max(v for k,v in s['generated/global_rigid']['macro'].items() if k!='world')}
(a.output/'summary.json').write_text(json.dumps(summary,indent=2))
with (a.output/'all-cells.csv').open('w',newline='') as f:
 fields=['asset','kind','seed','regime','direction','method','epe','regret','weak_epe','fallback_fraction','switch_rate','amplitude_abs_error_mean'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for c in cells:
  for name,vals in c['methods'].items():w.writerow({**{k:c[k] for k in fields[:5]},'method':name,**{k:vals.get(k) for k in fields[6:]}})
print(json.dumps({'decisions':summary['decisions'],'gpu':summary['gpu'],'checks':summary['checks'],'cells':len(cells)},indent=2))
