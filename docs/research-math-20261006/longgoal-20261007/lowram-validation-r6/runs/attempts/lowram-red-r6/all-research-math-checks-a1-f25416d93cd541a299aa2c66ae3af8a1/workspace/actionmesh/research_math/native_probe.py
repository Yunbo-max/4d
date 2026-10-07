"""Bounded frozen ActionMesh query-interface diagnostic, not a quality benchmark.

Reuses all 16 Stage-I latents; measures three output times and selected original
vertices. Synthetic edit requests and residuals are explicitly constructed.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def verify_file(path, expected):
    actual=digest(path)
    if actual != expected: raise ValueError('Source hash mismatch: '+str(path))
    return actual


def serial(value):
    if is_dataclass(value): return serial(asdict(value))
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k):serial(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [serial(v) for v in value]
    return value


def save(path, value):
    Path(path).write_text(json.dumps(serial(value),indent=2,allow_nan=False)+'\n')


def measurement_maps(n):
    if not isinstance(n,int) or n<4: raise ValueError('At least four query points')
    A=np.zeros((3,9*n)); B=np.zeros_like(A)
    for axis in range(3):
        A[axis,3*n+axis:6*n:3]=1./n
        B[axis,6*n+axis:6*n+3*(n//2):3]=1./(n//2)
    return A,B


def query_control_basis(query):
    q=np.asarray(query,dtype=np.float64)
    if q.ndim!=2 or q.shape[1]!=6 or len(q)<4 or not np.isfinite(q).all():
        raise ValueError('Require finite XYZ+normal source queries')
    xyz=q[:,:3]; span=np.ptp(xyz,axis=0); axis=int(np.argmax(span))
    if span[axis]<=1e-12: raise ValueError('Degenerate source extent')
    weight=(xyz[:,axis]-xyz[:,axis].mean())/span[axis]
    D=np.zeros((len(q),6,6))
    for j in range(3): D[:,j,j]=1.; D[:,j,3+j]=weight
    return D


def run(args, report, check):
    import torch
    root=args.root
    sys.path.insert(0,str(root));sys.path.insert(0,str(root/'research/census-20261002'))
    sys.path.insert(0,str(root/'repo'))
    from research_census_time_direction import load_source
    from research_three_ideas import Resources
    from actionmesh.model.temporal_autoencoder import ActionMeshAutoencoder
    from actionmesh.model.utils.embeddings import get_scaling,apply_scaling
    from research_math.controllability import analyze_edit
    from research_math.correlated import apply_covariance_inverse
    source, hashes, latents, native, faces, prepared=load_source(args.case_dir)
    code=json.loads((args.case_dir/'code-provenance.json').read_text())
    code_hash=verify_file(root/'repo/actionmesh/model/temporal_autoencoder.py',code['sha256']['actionmesh/model/temporal_autoencoder.py'])
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fallback for native test')
    torch.cuda.set_device(0);torch.set_num_threads(2);torch.manual_seed(42)
    free,total=torch.cuda.mem_get_info()
    if free<8*1024**3:raise RuntimeError('Insufficient free GPU headroom; no competing task is stopped')
    torch.cuda.set_per_process_memory_fraction(min(1.,12*1024**3/total))
    monitor=Resources(args.output,torch,'0');monitor.start()
    report.update(uid=source['uid'],source_hashes=hashes,model_code_sha256=code_hash,
                  input_times=16,target_frames=[1,8,15],query_count=args.points,
                  weights_frozen=True,training=False,gt_used=False,
                  scope='Native frozen-decoder interface and constructed edit controls; not natural editing efficacy or reconstruction accuracy.')
    try:
        checkpoint=root/'repo/pretrained_weights/ActionMesh/autoencoder'
        check('load_decoder')
        model=ActionMeshAutoencoder.from_pretrained(str(checkpoint),local_files_only=True).eval().cuda()
        for p in model.parameters():p.requires_grad_(False)
        model.verbose=False
        ids=np.linspace(0,len(native[0])-1,args.points,dtype=np.int64)
        if len(np.unique(ids))!=args.points:raise ValueError('Insufficient distinct query vertices')
        query_np=prepared['anchor_query_features'][ids].copy()
        q=torch.from_numpy(query_np[None]).cuda()
        z=torch.from_numpy(latents[None]).cuda()
        times=torch.arange(16,dtype=torch.float32,device='cuda')[None]
        lo,span=get_scaling(times)
        src=apply_scaling(times[:,0],lo,span)
        target=apply_scaling(torch.tensor([[1.,8.,15.]],device='cuda'),lo,span)
        caches=[]
        def keep_cache(module, inputs, output):caches.append(output.detach().cpu().clone())
        hook=model.blocks[-2].register_forward_hook(keep_cache)
        try:
            check('native_three_target_decode')
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                base=model(latent=z,framestep=times,source_alpha=src,target_alphas=target,query=q)[0].cpu().numpy()
        finally:hook.remove()
        if len(caches)!=3:raise ValueError('Expected one temporal cache per target')
        diagonal=float(np.linalg.norm(np.ptp(native[0].astype(np.float64),axis=0)))
        ref=native[[1,8,15]][:,ids]
        rms=float(np.sqrt(np.mean((base.astype(float)-ref)**2))/diagonal)
        maximum=float(np.max(np.abs(base-ref))/diagonal)
        report['native_subset_parity']={'rms_over_diagonal':rms,'max_over_diagonal':maximum,
            'rms_limit':1e-4,'max_limit':1e-3,'passed':rms<=1e-4 and maximum<=1e-3}
        if not report['native_subset_parity']['passed']:raise ValueError('Native subset does not reproduce cached baseline within fixed tolerance')
        D=query_control_basis(query_np)
        def decode(a):
            check('cached_query_decode')
            shifted=query_np.astype(np.float64)+np.einsum('vcp,p->vc',D,a)
            query=torch.as_tensor(shifted[None],device='cuda',dtype=torch.float32)
            outputs=[]
            with torch.inference_mode(),torch.autocast('cuda',enabled=False):
                emb=model.embedder(query[...,:3]);emb=torch.cat([emb,query[...,3:]],dim=-1)
                for cache in caches:
                    kv=cache.cuda()
                    logits=model.fwd_cross_attn(kv,emb)
                    outputs.append((2*torch.sigmoid(logits)-1)[0].cpu().numpy())
                    del kv
            return np.stack(outputs)
        zero=np.zeros(6); replay=decode(zero)
        replay_error=float(np.max(np.abs(replay-base)))
        report['cache_replay']={'max_abs_error':replay_error,'passed':replay_error<=1e-6}
        if replay_error>1e-6:raise ValueError('Cache replay changes native outputs')
        h=.002*diagonal
        def jacobian(step):
            columns=[]
            for k in range(6):
                a=np.zeros(6);a[k]=step
                columns.append(((decode(a).astype(float)-decode(-a).astype(float))/(2*step)).ravel())
            return np.column_stack(columns)
        check('finite_difference_jacobian')
        J=jacobian(h);Jh=jacobian(h/2)
        rel=float(np.linalg.norm(J-Jh)/max(np.linalg.norm(Jh),1e-12))
        report['jacobian_check']={'step':h,'half_step':h/2,'relative_difference':rel,'limit':.15,'passed':rel<=.15}
        if rel>.15:raise ValueError('Finite difference derivative unstable at frozen step sizes')
        J=Jh;A,B=measurement_maps(args.points)
        radius=.01*diagonal*np.sqrt(args.points)
        rows=[]; saved={'native':base,'query_ids':ids,'query_basis':D,'jacobian':J,'contract_map':A,'edit_map':B,'target_frames':np.array([1,8,15])}
        for axis in range(3):
            d=np.zeros(3);d[axis]=.002*diagonal
            native_result=analyze_edit(J,A,B,d,radius=radius,rtol=1e-7)
            output_result=analyze_edit(np.eye(J.shape[0]),A,B,d,radius=radius,rtol=1e-7)
            ideal=np.asarray(native_result.step)
            a=np.linalg.lstsq(J,ideal,rcond=1e-7)[0]
            query_clip=min(1.,(.005*diagonal)/max(np.max(np.abs(np.einsum('vcp,p->vc',D,a)[:,:3])),1e-20))
            a*=query_clip
            trials=[];accepted=None
            for factor in [1.,.5,.25,.125,.0625,0.]:
                actual=decode(a*factor);delta=(actual-base).astype(float).ravel()
                contract=float(np.linalg.norm(A@delta));edit=float(np.linalg.norm(B@delta-d))
                feasible=contract<=2e-5*diagonal and np.linalg.norm(delta)<=radius*(1+1e-6)
                trials.append({'factor':factor,'contract_norm':contract,'edit_residual':edit,'step_norm':float(np.linalg.norm(delta)),'feasible':bool(feasible)})
                if feasible and edit<=np.linalg.norm(d)+1e-10:
                    accepted=actual;saved[f'edit_axis{axis}']=actual;break
            if accepted is None:raise RuntimeError('Original zero control must be feasible')
            direct=base+np.asarray(output_result.step).reshape(base.shape)
            saved[f'output_axis{axis}']=direct
            rows.append({'axis':axis,'requested_edit':d,'model_linear_result':native_result,
                         'output_linear_result':output_result,'query_clip':query_clip,'nonlinear_trials':trials,
                         'accepted_factor':trials[-1]['factor'],'baseline_label':'Unrestricted output point correction; does not establish valid continuous mesh or semantic edit'})
        report['B_constructed_native_edit_controls']=rows
        # GPU algebra check uses a real measured decoder Jacobian, but constructed RHS.
        # No generated-view likelihood or real scene confidence is inferred.
        rng=np.random.default_rng(20261002);F=J[:,:3];variances=np.array([.4,.2,.1]);rhs=rng.normal(size=J.shape[0]);sigma2=.3
        expected=apply_covariance_inverse(rhs,F,variances,sigma2=sigma2)
        Ft=torch.as_tensor(F*np.sqrt(variances),device='cuda',dtype=torch.float64)
        rt=torch.as_tensor(rhs,device='cuda',dtype=torch.float64)
        gpu=rt/sigma2-Ft@torch.linalg.solve(torch.eye(3,device='cuda',dtype=torch.float64)+(Ft.T@Ft)/sigma2,Ft.T@rt)/(sigma2*sigma2)
        error=float(np.max(np.abs(gpu.cpu().numpy()-expected)))
        report['A_cuda_linear_algebra']={'max_abs_cpu_cuda_difference':error,'passed':error<1e-8,'native_jacobian_used':True,'synthetic_rhs':True,'natural_generated_evidence_tested':False}
        if error>=1e-8:raise ValueError('CUDA covariance action differs from CPU')
        np.savez_compressed(args.output/'probe-arrays.npz',**saved)
        report.update(status='completed',model_parameter_gradients=any(p.grad is not None for p in model.parameters()),
                      all_loaded_parameters_frozen=all(not p.requires_grad for p in model.parameters()))
        for name,expected in hashes.items():verify_file(args.case_dir/name,expected)
        report['source_hashes_unchanged']=True
    finally:
        report['resources']=monitor.finish()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--case-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--points',type=int,default=24)
    p.add_argument('--max-seconds',type=float,default=600.)
    a=p.parse_args()
    if a.points<4 or not 0<a.max_seconds<=1200: p.error('Invalid bounded probe size/time')
    a.output.mkdir(parents=True,exist_ok=False)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1')
    started=time.monotonic();report={'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),'script_sha256':digest(__file__)}
    save(a.output/'command.json',{'argv':sys.argv,'points':a.points,'seconds_cap':a.max_seconds})
    def check(stage):
        elapsed=time.monotonic()-started
        save(a.output/'progress.json',{'stage':stage,'elapsed_seconds':elapsed})
        if elapsed>a.max_seconds:raise TimeoutError('Bounded native probe exceeded time cap')
    try:run(a,report,check)
    except Exception as exc:
        report.update(status='failed',error=str(exc),traceback=traceback.format_exc());raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started
        save(a.output/'report.json',report)


if __name__=='__main__':main()
