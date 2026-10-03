"""M09: three-branch, per-frame cross-condition CFG projection.

Branches must be v0 (unconditional), vm (mesh), vmv (mesh+video), all evaluated
at the SAME CURRENT sampling state. This module never replays cached velocities.
Unknown frame groups are complete token/channel vectors. No part labels or GT.
Projection is an existing mathematical operator, not an asserted novelty.
"""
import numpy as np


MODES=("projection","scalar","norm_matched","random")


def combine_guidance(v0, vm, vmv, known_mask, *, mesh_scale=1., video_scale=1.,
                     epsilon=1e-12, mode="projection", seed=42):
    """Return [B,T,N,C] velocity and diagnostics; leave known-frame CFG unchanged."""
    arrays=[np.asarray(x,dtype=float) for x in (v0,vm,vmv)]
    v0,vm,vmv=arrays;known=np.asarray(known_mask)
    if (v0.ndim!=4 or v0.size==0 or any(x.shape!=v0.shape or not np.isfinite(x).all() for x in arrays)
            or known.shape!=v0.shape[:2] or not np.isin(known,[0,1]).all()
            or mode not in MODES or not np.isfinite([mesh_scale,video_scale,epsilon]).all()
            or epsilon<=0 or mesh_scale<0 or video_scale<0):
        raise ValueError("finite equal [B,T,N,C] branches and binary [B,T] mask required")
    known=known.astype(bool)
    u=vm-v0;w=vmv-vm
    dot=np.sum(u*w,axis=(-2,-1),keepdims=True)
    norm2=np.sum(u*u,axis=(-2,-1),keepdims=True)
    active=(dot<0)&(norm2>epsilon)&(~known[...,None,None])
    coefficient=np.where(active,np.minimum(dot,0)/(norm2+epsilon),0.)
    projected=w-coefficient*u
    if mode=="projection":
        changed=projected
    elif mode=="scalar":
        changed=w
    elif mode=="norm_matched":
        norms=np.linalg.norm(w.reshape(*w.shape[:2],-1),axis=-1)[...,None,None]
        target=np.linalg.norm(projected.reshape(*w.shape[:2],-1),axis=-1)[...,None,None]
        changed=w*np.where(norms>0,target/np.maximum(norms,1e-300),1.)
    else:
        rng=np.random.default_rng(seed)
        direction=rng.normal(size=w.shape)
        length=np.sqrt(np.sum(direction**2,axis=(-2,-1),keepdims=True))
        budget=np.sqrt(np.sum((projected-w)**2,axis=(-2,-1),keepdims=True))
        changed=w+direction/np.maximum(length,1e-300)*budget
    # Exact original branch formula at known frames and every nonconflict frame.
    changed=np.where(active,changed,w)
    result=v0+mesh_scale*u+video_scale*changed
    return result,{"conflicting_unknown_frames":int(active.sum()),
                   "groups":int(np.prod(v0.shape[:2])),
                   "correction_l2":float(np.linalg.norm(changed-w)),
                   "branch_evaluations_per_state":3,"mode":mode}


def run_live_flow(predict, initial, timesteps, distances, known_mask, *,
                  mesh_scale=1., video_scale=1., mode="projection", epsilon=1e-12,seed=42):
    """NumPy reference additive-flow loop; predict is called afresh each step."""
    state=np.asarray(initial,dtype=float).copy();clock=np.asarray(timesteps,dtype=float)
    dt=np.asarray(distances,dtype=float);known=np.asarray(known_mask)
    if (state.ndim!=4 or state.size==0 or not np.isfinite(state).all() or known.shape!=state.shape[:2]
            or not np.isin(known,[0,1]).all() or clock.ndim!=1 or dt.ndim!=1
            or len(clock)!=len(dt)+1 or len(dt)==0 or not np.isfinite(clock).all()
            or not np.isfinite(dt).all() or np.any(dt<0) or np.any(np.diff(clock)>=0)):
        raise ValueError("invalid live-flow input")
    known=known.astype(bool);fixed=state.copy();records=[]
    for i,distance in enumerate(dt):
        branches=predict(state.copy(),float(clock[i]))
        if not isinstance(branches,(tuple,list)) or len(branches)!=3:
            raise ValueError("predict must return three live branches")
        velocity,info=combine_guidance(*branches,known,mesh_scale=mesh_scale,
                                      video_scale=video_scale,epsilon=epsilon,mode=mode,seed=seed+i)
        if velocity.shape!=state.shape:
            raise ValueError("branch shape differs from current state")
        state=state+distance*velocity
        state[known]=fixed[known]
        if not np.isfinite(state).all():
            raise ValueError("nonfinite live sampling state")
        records.append(info)
    return {"latents":state,"steps":len(dt),"branch_evaluations":3*len(dt),
            "records":records,"cached_velocity_replay":False}


def make_torch_guidance(base_guidance, *, mode="projection",epsilon=1e-12,seed=42):
    """Lazy native wrapper; delegates official expansion, hooks every aggregate.

    Replace pipeline.cf_guidance with this returned object for one live rollout.
    Base configuration must explicitly request [[0,0],[0,1],[1,1]] and two scales.
    This does not alter a two-branch native configuration silently. Restore the
    original object after the run. No model load occurs in this constructor.
    """
    import torch
    if (base_guidance.guidance_at_inference!=[[0,0],[0,1],[1,1]]
            or len(base_guidance.guidance_scales)!=2 or not base_guidance.inference_enabled):
        raise ValueError("explicit enabled three-branch CFG configuration required")
    if mode not in MODES or not np.isfinite(epsilon) or epsilon<=0:
        raise ValueError("invalid mode or epsilon")
    if any(not np.isfinite(s) or s<0 for s in base_guidance.guidance_scales):
        raise ValueError("guidance scales must be finite and nonnegative")

    class LiveGuidance:
        def __init__(self):
            self.base=base_guidance;self.known=None;self.records=[]
            self.guidance_at_inference=[[0,0],[0,1],[1,1]]
            self.guidance_scales=list(base_guidance.guidance_scales)
            self.inference_enabled=True
        def get_unobserved_mask(self,mask):
            if mask is not None and (mask.ndim!=2 or not torch.isfinite(mask).all()
                                     or not ((mask==0)|(mask==1)).all()):
                raise ValueError("native mask must be binary [B,T]")
            self.known=None if mask is None else mask.bool().clone()
            return self.base.get_unobserved_mask(mask)
        def cfg_at_inference(self,latent,context,mask,framestep):
            self.get_unobserved_mask(mask)
            return self.base.cfg_at_inference(latent,context,mask,framestep)
        def aggregate_cfg(self,aggregated):
            if aggregated.ndim!=4 or aggregated.numel()==0 or aggregated.shape[0]%3 or not torch.isfinite(aggregated).all():
                raise ValueError("three finite CFG batches required")
            v0,vm,vmv=aggregated.chunk(3,dim=0)
            # Promote BEFORE subtraction: finite fp16 branches may differ by
            # more than fp16's range, despite a representable final result.
            uf=vm.float()-v0.float();wf=vmv.float()-vm.float()
            dot=(uf*wf).sum(dim=(-2,-1),keepdim=True)
            norm2=uf.square().sum(dim=(-2,-1),keepdim=True)
            known=torch.zeros(v0.shape[:2],device=v0.device,dtype=torch.bool) if self.known is None else self.known.to(v0.device)
            if tuple(known.shape)!=tuple(v0.shape[:2]):
                raise ValueError("mask does not match live branch batch/time")
            active=(dot<0)&(norm2>epsilon)&(~known[...,None,None])
            coefficient=torch.where(active,torch.minimum(dot,torch.zeros_like(dot))/(norm2+epsilon),torch.zeros_like(dot))
            projected=wf-coefficient*uf
            if mode=="projection":
                changed=projected
            elif mode=="scalar":
                changed=wf
            elif mode=="norm_matched":
                original=wf.square().sum(dim=(-2,-1),keepdim=True).sqrt()
                desired=projected.square().sum(dim=(-2,-1),keepdim=True).sqrt()
                changed=wf*torch.where(original>0,desired/original.clamp_min(1e-30),torch.ones_like(original))
            else:
                generator=torch.Generator(device=wf.device).manual_seed(seed+len(self.records))
                direction=torch.randn(wf.shape,dtype=torch.float32,device=wf.device,generator=generator)
                length=direction.square().sum(dim=(-2,-1),keepdim=True).sqrt()
                budget=(projected-wf).square().sum(dim=(-2,-1),keepdim=True).sqrt()
                changed=wf+direction/length.clamp_min(1e-30)*budget
            changed=torch.where(active,changed,wf)
            result=(v0.float()+self.guidance_scales[0]*uf+self.guidance_scales[1]*changed).to(v0.dtype)
            if not torch.isfinite(result).all():
                raise ValueError("nonfinite guidance result")
            self.records.append({"step":len(self.records),"conflicting_unknown_frames":int(active.sum().item()),
                                 "mode":mode,"branch_evaluations":3})
            return result
    return LiveGuidance()


def demo():
    initial=np.zeros((1,3,1,2));initial[:,0]=[2.,3.]
    calls=[]
    def predict(state,t):
        calls.append(float(np.linalg.norm(state)))
        v0=np.zeros_like(state);vm=v0+[1.,0.]
        return v0,vm,vm+np.stack([-np.ones_like(state[...,0]),1.+state[...,1]],axis=-1)
    result=run_live_flow(predict,initial,[1.,.5,0.],[.5,.5],[[True,False,False]])
    return {"method":9,"evidence":"constructed live flow, not native benchmark",
            "state_norms_seen_by_predictor":calls,
            "fresh_branches":result["branch_evaluations"],
            "anchor_exact":bool(np.array_equal(result["latents"][:,0],initial[:,0])),
            "final":result["latents"].tolist(),"native_sampler_hook_available":True}
