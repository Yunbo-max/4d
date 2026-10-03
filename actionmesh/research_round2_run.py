#!/usr/bin/env python3
"""Frozen exploratory output-space edits. Not native model steering or training."""
from __future__ import annotations
import argparse, hashlib, json, math, time, traceback
from pathlib import Path
import numpy as np
import torch
from research_edit_probe import kabsch_rows, CASES
import research_round2_utility as utility
import research_round2_geometry as geometry
import research_round2_time as temporal


def jswrite(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False))


def rot(angle, axis):
    c,s=torch.cos(angle),torch.sin(angle); z=torch.zeros_like(c);o=torch.ones_like(c)
    rows = (o,z,z,z,c,s,z,-s,c) if axis==0 else (c,s,z,-s,c,z,z,z,o)
    return torch.stack(rows,-1).reshape(*angle.shape,3,3)


def controlled(x, regime, frames, scale, generator):
    phase=torch.linspace(0,1,frames,device=x.device)
    a=.9*torch.sin(2*math.pi*phase); b=.5*torch.sin(3*math.pi*phase)
    if 'articulation' in regime:
        group=x[:,0]>=x[:,0].median()
        sg=torch.where(group,1.,-1.)
        r=rot(a[:,None]*sg[None],2)@rot(b[:,None]*sg[None],0)
        ctr=torch.stack([x[~group].mean(0),x[group].mean(0)])[group.long()]
        clean=torch.einsum('ni,tnij->tnj',x-ctr,r)+ctr
    else:
        r=(rot(a,2)@rot(b,0))[:,None].expand(-1,len(x),-1,-1).clone()
        if regime=='affine_stretch':
            stretch=torch.stack([1+.4*torch.sin(math.pi*phase),1-.2*torch.sin(math.pi*phase),torch.ones_like(phase)],-1)
            r=torch.diag_embed(stretch)[:,None]@r
        ctr=x.mean(0)
        clean=torch.einsum('ni,tnij->tnj',x-ctr,r)+ctr
    obs=clean.clone()
    if regime.endswith('_noisy'):
        obs[1:]+=.005*scale*torch.randn(obs[1:].shape,generator=generator,device=x.device)
    return obs,r


def prepare(original, seed, device, pre_sampled=False, scale_override=None):
    rng=np.random.default_rng(seed)
    if original.shape[1]<1024: raise ValueError('fewer than predeclared1024 points')
    idx=np.arange(1024) if pre_sampled else rng.choice(original.shape[1],1024,replace=False)
    v=torch.tensor(np.array(original[:,idx,:3]),device=device,dtype=torch.float32)
    q=torch.arange(512,device=device);g=q.clone()
    dist=torch.cdist(v[0,q],v[0,g]);dist.fill_diagonal_(float('inf'))
    fit=g[dist.topk(32,largest=False).indices]
    held=torch.arange(512,768,device=device)[torch.cdist(v[0,q],v[0,512:768]).topk(8,largest=False).indices]
    report=torch.arange(768,1024,device=device)[torch.cdist(v[0,q],v[0,768:1024]).topk(8,largest=False).indices]
    assert not torch.isin(held,fit).any() and not torch.isin(report,torch.cat([fit.flatten(),held.flatten()])).any()
    scale=float(scale_override) if scale_override is not None else float(np.linalg.norm(np.ptp(original[0,:,:3].astype(np.float64),axis=0)))
    height=v[0,q,2];lo,hi=torch.quantile(height,torch.tensor([.65,.9],device=device))
    w=((height-lo)/(hi-lo).clamp_min(1e-8)).clamp(0,1)
    patch=v[0,fit];pc=patch-patch.mean(-2,keepdim=True)
    _,evec=torch.linalg.eigh(pc.transpose(-1,-2)@pc)
    directions={f'axis_{i}':torch.eye(3,device=device)[i].expand(512,3) for i in range(3)}
    directions.update(pca_major=evec[:,:,-1],pca_second=evec[:,:,-2])
    return v,q,g,fit,held,report,scale,w,directions,idx


def fit_cache(v,q,g,fit,scale):
    lr,lres,_=kabsch_rows(v[0,fit][None].expand(len(v),-1,-1,-1),v[:,fit])
    gr,gres,_=kabsch_rows(v[0,g][None].expand(len(v),-1,-1),v[:,g])
    # Preserve anchor exactly in candidates, including numerically rank-deficient patches.
    lr[0]=torch.eye(3,device=v.device);gr[0]=torch.eye(3,device=v.device)
    cache=temporal.build_cache(v,q,fit,g,lr,gr,kabsch_rows)
    patch=v[0,fit].double();pc=patch-patch.mean(-2,keepdim=True)
    scatter=pc.transpose(-1,-2)@pc
    tr=scatter.diagonal(dim1=-2,dim2=-1).sum(-1)
    info=tr[:,None,None]*torch.eye(3,device=v.device,dtype=torch.float64)-scatter
    eig=torch.linalg.eigvalsh(info).clamp_min(0)
    cache['weak_support']=eig[:,0]/eig.sum(-1).clamp_min(1e-15)<.01
    return lr,gr,lres/scale,gres/scale,cache


def ranked_mask(score,edited,coverage):
    mask=torch.zeros_like(score,dtype=torch.bool)
    ids=torch.where(edited)[0]
    for t in range(1,len(score)):
        selected=score[t,ids]
        if float(selected.max()-selected.min())<=1e-10: continue
        count=int(math.floor(len(ids)*coverage))
        mask[t,ids[torch.argsort(selected,descending=True,stable=True)[:count]]]=True
    return mask


def metrics(offset,d0,observed,q,report,scale,edited,truth=None,mask=None,oracle=None):
    amp=(offset.norm(dim=-1)-d0.norm(dim=-1)[None]).abs()/scale
    spokes=observed[:,report]-observed[:,q,None]
    response=(spokes*offset[:,:,None]).sum(-1)
    drift=(response-response[:1])/(scale*d0.norm(dim=-1)[None,:,None].clamp_min(1e-12))
    res={'amplitude_abs_error_mean':float(amp[:,edited].mean()),
         'anchor_max_error':float((offset[0]-d0).abs().max()/scale),
         'unedited_max_shift':float(offset[:,~edited].abs().max()/scale),
         'heldout_response_drift_rms_DIAGNOSTIC':float(drift[1:,edited].square().mean().sqrt()),
         'finite':bool(torch.isfinite(offset).all()),
         'fallback_fraction':float(mask[1:,edited].float().mean()) if mask is not None else None,
         'switch_rate':float((mask[2:,edited]!=mask[1:-1,edited]).float().mean()) if mask is not None else None}
    if truth is not None:
        errors=(offset-truth).norm(dim=-1)/scale
        res['epe']=float(errors[1:,edited].mean())
        if oracle is not None: res['regret']=float((errors-oracle)[1:,edited].mean())
    return res


def evaluate(v,d0,q,g,fit,held,report,scale,cache,truth,output,tag,normal_task=False,save=False):
    lr,gr,lres,gres,tc=cache
    loc=torch.einsum('qi,tqij->tqj',d0,lr);glob=torch.einsum('qi,tij->tqj',d0,gr)
    loc[0]=d0;glob[0]=d0
    edited=d0.norm(dim=-1)>0
    us=utility.compute(v,d0,q,held,lr,gr,scale)
    gs=geometry.compute(v,d0,q,fit,held,g,lr,gr,scale)
    ts=temporal.compute(tc,d0)
    scores={k:val for k,val in gs.items() if isinstance(val,torch.Tensor) and val.shape==lres.shape}
    scores.update(ts)
    scores['fit_residual']=lres-gres[:,None]
    offsets={'world':d0[None].expand_as(loc).clone(),'global':glob,'local':loc,'blend':.5*(loc+glob)}
    masks={}
    for name,score in [('utility',us['utility']),('heldout',us['heldout']),('axis_averaged',us['axis_averaged'])]:
        mask=score>=0;mask[0]=False
        offsets[name]=torch.where(mask[...,None],glob,loc);masks[name]=mask
    old=lres>.005;old[0]=False
    offsets['old_residual_gate']=torch.where(old[...,None],glob,loc);masks['old_residual_gate']=old
    for key,score in scores.items():
        if not torch.isfinite(score).all(): raise ValueError(f'nonfinite score {key}')
        for cov in [.25,.5,.75]:
            mask=ranked_mask(score,edited,cov);name=f'{key}_q{int(cov*100)}'
            offsets[name]=torch.where(mask[...,None],glob,loc);masks[name]=mask
    if normal_task:
        points=v[:,fit];center=points-points.mean(-2,keepdim=True)
        _,eig=torch.linalg.eigh(center.transpose(-1,-2)@center)
        normal=eig[...,:,0];sign=torch.where((normal*loc).sum(-1)>=0,1.,-1.)
        out=normal*sign[...,None]*d0.norm(dim=-1)[None,:,None];out[0]=d0
        offsets['pca_normal']=out
    oracle=None
    if truth is not None:
        el=(loc-truth).norm(dim=-1);eg=(glob-truth).norm(dim=-1)
        oracle=torch.minimum(el,eg)/scale
        offsets['oracle_NOT_EXECUTABLE']=torch.where((eg<el)[...,None],glob,loc)
    rows={m:metrics(off,d0,v,q,report,scale,edited,truth,masks.get(m),oracle) for m,off in offsets.items()}
    weak=tc['weak_support']&edited
    if truth is not None:
        for m,off in offsets.items():
            rows[m]['weak_epe']=float(((off-truth).norm(dim=-1)/scale)[1:,weak].mean()) if weak.any() else None
    record={'weak_edited_count':int(weak.sum()),'tag':tag,'edited_count':int(edited.sum()),'axis_identity_max_error':float(us['axis_identity_error'].abs().max()),'census':gs.get('census',{}),'fit_counts':tc['fit_counts'],'methods':rows,
            'scores':{k:{'min':float(s[1:,edited].min()),'median':float(s[1:,edited].median()),'max':float(s[1:,edited].max())} for k,s in scores.items()}}
    if save:
        arrays={'source':v[:,q].cpu().numpy(),'d0':d0.cpu().numpy(),'scale':np.array(scale)}
        if truth is not None: arrays['target']= (v[:,q]+truth).cpu().numpy()
        keep=['world','global','local','utility','heldout','directional_covariance_q50','temporal_action_q50','pca_normal']
        arrays.update({m:(v[:,q]+offsets[m]).cpu().numpy() for m in keep if m in offsets})
        np.savez_compressed(output/f'{tag}-tracks.npz',**arrays)
    return record


@torch.no_grad()
def run(args):
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; no silent CPU fallback')
    device=torch.device('cuda');torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    report={'scope':'exploratory output-space transport; no training or generator inference','status':'running','device':torch.cuda.get_device_name(),
            'torch_version':torch.__version__,'numpy_version':np.__version__,'protocol':args.protocol.read_text(),
            'protocol_sha256':hashlib.sha256(args.protocol.read_bytes()).hexdigest(),'seeds':[17,29,41],'inputs':{},'cells':[],'failures':[]}
    def checkpoint():
        torch.cuda.synchronize();report['elapsed_seconds']=time.perf_counter()-start
        report['peak_allocated_bytes']=torch.cuda.max_memory_allocated();report['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
        jswrite(out/'results.json',report)
    def budget():
        if time.perf_counter()-start>600: raise TimeoutError('Frozen600sGPUdiagnosticbudget')
    items=[(name,args.input_root/rel,'generated') for name,rel in CASES.items()]
    manifest=json.loads(args.manifest.read_text())
    if not args.only_generated:
        items.extend((uid,args.actionbench_root/'sampled'/(uid+'.npz'),'normal') for uid in manifest['uids'])
    for name,path,kind in items:
        try:
            raw=np.load(path,allow_pickle=False)
            if kind=='generated': original=np.array(raw['vertices']);raw.close()
            else: original=np.array(raw['s17'])
            if original.ndim!=3 or original.shape[-1]!=(3 if kind=='generated' else 6) or not np.isfinite(original).all(): raise ValueError('invalid input array')
            report['inputs'][name]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'shape':list(original.shape),'kind':kind}
            for seed in [17,29,41]:
                budget()
                if kind=='normal': original=np.array(raw[f's{seed}'])
                v,q,g,fit,held,rep,scale,w,dirs,ids=prepare(original,seed,device,pre_sampled=kind=='normal',scale_override=float(raw['full_scale']) if kind=='normal' else None)
                report['inputs'][name][f'indices_seed{seed}']=(raw[f'i{seed}'] if kind=='normal' else ids).tolist()
                report['inputs'][name]['scale']=scale
                generator=torch.Generator(device=device).manual_seed(seed)
                if kind=='generated':
                    regimes=['global_rigid','articulation','articulation_noisy','global_rigid_noisy','affine_stretch','natural']
                else:
                    normals=torch.tensor(np.array(original[:,ids,3:6]),device=device,dtype=torch.float32)
                    nlen=normals.norm(dim=-1,keepdim=True)
                    if (nlen<1e-8).any(): raise ValueError('zero provided normal; no silent exclusion')
                    normals=normals/nlen
                    dirs={'normal':normals[0,q]};regimes=['normal_clean','normal_noisy']
                for regime in regimes:
                    budget()
                    if regime=='natural': obs=v;mapping=None
                    elif kind=='generated': obs,mapping=controlled(v[0],regime,len(v),scale,generator)
                    else:
                        obs=v.clone();mapping=None
                        if regime.endswith('noisy'):obs[1:]+=.005*scale*torch.randn(obs[1:].shape,generator=generator,device=device)
                    cache=fit_cache(obs,q,g,fit,scale)
                    for direction,unit in dirs.items():
                        d0=.03*scale*w[:,None]*unit
                        truth=None
                        if mapping is not None: truth=torch.einsum('qi,tqij->tqj',d0,mapping[:,q])
                        elif kind=='normal':truth=.03*scale*w[None,:,None]*normals[:,q]
                        tag=f'{name}__s{seed}__{regime}__{direction}'
                        row=evaluate(obs,d0,q,g,fit,held,rep,scale,cache,truth,out,tag,kind=='normal',save=seed==17 and direction in ('axis_0','normal') and regime in ('natural','articulation_noisy','normal_clean'))
                        row.update(asset=name,seed=seed,regime=regime,direction=direction,kind=kind,has_gt=truth is not None)
                        report['cells'].append(row)
                    checkpoint()
                print(json.dumps({'asset':name,'seed':seed,'cells':len(report['cells']),'elapsed':report['elapsed_seconds']}),flush=True)
            if kind=='normal': raw.close()
        except Exception as exc:
            report['failures'].append({'asset':name,'type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()});checkpoint()
            if isinstance(exc,TimeoutError):break
    report['status']='failed' if report['failures'] else 'completed_exploratory'
    checkpoint();print(json.dumps({k:report[k] for k in ['status','elapsed_seconds','peak_allocated_bytes','failures']}),flush=True)
    if report['failures']:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['input-root','output','actionbench-root','manifest','protocol']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--only-generated',action='store_true',help='Run only missing generated-asset cells; retain independently completed normal cells')
    run(p.parse_args())
