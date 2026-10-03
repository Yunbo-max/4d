"""M10: bounded real callback rollouts with future overlap boundary conditions.

Requires >=4 windows and a caller-qualified shared representation identity.
No latent averaging is performed. Only caller-provided anchors are immutable.
The final window is first resampled against a reliable terminal pseudo-boundary,
freeing its generated starting overlap. Earlier windows then receive the revised
future overlap. A callback must actually sample anew from supplied conditions.
This is an executable rollout policy, not proof that future predictions are true.
"""
import numpy as np


def rollout_windows(callback, times, window_size, overlap, anchors, representation_id, *,
                    future_reliable, backward_updates=4, max_rollouts=16, seed=42):
    """callback(indices,known,direction,seed,representation_id) -> result dict.

    known maps GLOBAL frame indices to hard latent values. result must contain
    latents[len(indices),...] and the exact same representation_id. The identity
    is a required provenance assertion, not evidence inferred from numeric
    equality. Caller must qualify common anchor/token/coordinate meaning first.
    max_rollouts includes both directions; no hidden candidate passes.
    Every window has window_size distinct frames. A short tail shifts the last
    window back to end at the final real frame, increasing its overlap instead
    of padding or dropping observations. overlap is therefore a minimum.
    """
    times=np.asarray(times,dtype=float);reliable=np.asarray(future_reliable)
    if (times.ndim!=1 or len(times)<2 or not np.isfinite(times).all() or np.any(np.diff(times)<=0)
            or not isinstance(window_size,(int,np.integer)) or window_size<2
            or not isinstance(overlap,(int,np.integer)) or not 1<=overlap<window_size
            or reliable.shape!=times.shape or not np.isin(reliable,[0,1]).all()
            or not isinstance(backward_updates,(int,np.integer)) or backward_updates<0
            or not isinstance(max_rollouts,(int,np.integer)) or max_rollouts<1
            or isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or seed<0
            or not isinstance(representation_id,str) or not representation_id.strip()):
        raise ValueError("invalid windows, times, reliability, identity or budget")
    reliable=reliable.astype(bool)
    windows=[];start=0
    while True:
        if len(times)<window_size:
            raise ValueError("sequence is shorter than one complete window")
        start=min(start,len(times)-window_size)
        indices=np.arange(start,start+window_size,dtype=int)
        windows.append(indices)
        if indices[-1]==len(times)-1:
            break
        start+=window_size-overlap
    if len(windows)<4:
        raise ValueError("natural long-sequence protocol requires at least four windows")
    if max_rollouts<len(windows):
        raise ValueError("budget cannot cover one complete forward rollout")
    if not isinstance(anchors,dict) or not anchors:
        raise ValueError("explicit source anchor latents required")
    fixed={}
    shape=None
    for key,value in anchors.items():
        value=np.asarray(value)
        if (not isinstance(key,(int,np.integer)) or not 0<=key<len(times)
                or value.ndim<1 or value.dtype.kind not in "fiu" or not np.isfinite(value).all()):
            raise ValueError("invalid source anchor")
        if shape is None:
            shape=value.shape
        if value.shape!=shape:
            raise ValueError("inconsistent anchor representation shapes")
        fixed[int(key)]=value.copy()
    if 0 not in fixed:
        raise ValueError("initial window source frame0 must be fixed")
    bank={k:v.copy() for k,v in fixed.items()};records=[]
    def execute(i,direction,known):
        indices=windows[i]
        response=callback(indices.copy(),{k:v.copy() for k,v in known.items()},
                          direction,int(seed+i),representation_id)
        if not isinstance(response,dict) or response.get("representation_id")!=representation_id:
            raise ValueError("unqualified or mismatched latent representation identity")
        values=np.asarray(response.get("latents"))
        if values.shape!=(len(indices),*shape) or values.dtype.kind not in "fiu" or not np.isfinite(values).all():
            raise ValueError("invalid callback latent output")
        for local,frame in enumerate(indices):
            if int(frame) in known and not np.array_equal(values[local],known[int(frame)]):
                raise ValueError("callback changed a hard source/boundary anchor")
        records.append({"window":int(i),"direction":direction,"indices":indices.tolist(),
                        "known_indices":sorted(known),"seed":int(seed+i),
                        "new_sampling_call":True})
        return values
    for i,indices in enumerate(windows):
        known={int(k):bank[int(k)] for k in indices if int(k) in bank}
        values=execute(i,"forward",known)
        for frame,value in zip(indices,values):
            if int(frame) not in known:
                bank[int(frame)]=value.copy()
    forward=np.stack([bank[k] for k in range(len(times))])
    backward_count=0;skipped=[];revised=set();boundaries=[]
    for i in range(len(windows)-1,-1,-1):
        if backward_count>=backward_updates or len(records)>=max_rollouts:
            break
        indices=windows[i]
        if i==len(windows)-1:
            # Seed reverse propagation with the terminal tail, not the generated
            # source overlap; this allows later context to revise that overlap.
            future=[int(k) for k in indices[-overlap:] if reliable[k]]
            provenance="forward terminal pseudo-observation; reliability asserted by caller"
        else:
            if i+1 not in revised:
                skipped.append({"window":i,"reason":"later window was not revised"})
                continue
            future=[int(k) for k in np.intersect1d(indices,windows[i+1]) if reliable[k]]
            provenance="updated overlap from immediately later backward rollout"
        if not future:
            skipped.append({"window":i,"reason":"no reliable future overlap"})
            continue
        known={int(k):fixed[int(k)] for k in indices if int(k) in fixed}
        for k in future:
            known[k]=fixed[k] if k in fixed else bank[k]
        if len(known)==len(indices):
            skipped.append({"window":i,"reason":"all frames hard conditioned"})
            continue
        values=execute(i,"backward",known)
        for frame,value in zip(indices,values):
            if int(frame) not in known:
                bank[int(frame)]=value.copy()
        backward_count+=1
        revised.add(i)
        boundaries.append({"window":int(i),"boundary_indices":future,"provenance":provenance,
                           "reliability":"caller-supplied boolean; no GT assessment"})
    result=np.stack([bank[k] for k in range(len(times))])
    for k,value in fixed.items():
        if not np.array_equal(result[k],value):
            raise AssertionError("source-anchor invariant failed")
    return {"latents":result,"forward":forward,"times":times.copy(),
            "forward_rollouts":len(windows),"backward_rollouts":backward_count,
            "rollout_count":len(records),"max_rollouts":int(max_rollouts),
            "requested_backward_updates":int(backward_updates),
            "records":records,"skipped":skipped,"source_anchor_indices":sorted(fixed),
            "window_indices":[w.tolist() for w in windows],
            "actual_overlap_sizes":[int(len(np.intersect1d(a,b))) for a,b in zip(windows,windows[1:])],
            "backward_boundaries":boundaries,"terminal_boundary_is_ground_truth":False,
            "representation_id":representation_id,"unaligned_averaging":False,
            "natural_long_video_validated":False}


def demo():
    calls=[]
    def callback(indices,known,direction,seed,representation_id):
        keys=sorted(known);values=[float(known[k][0]) for k in keys]
        x=np.array([np.interp(k,keys,values)+(0. if k in known else .1)
                    for k in indices])[:,None]
        calls.append(direction)
        return {"latents":x,"representation_id":representation_id}
    result=rollout_windows(callback,np.arange(13.),4,1,{0:np.array([0.])},
                          "constructed-common-anchor",future_reliable=np.ones(13,bool),
                          backward_updates=4,max_rollouts=8)
    return {"method":10,"evidence":"constructed four-window callback rollout",
            "forward_rollouts":result["forward_rollouts"],"backward_rollouts":result["backward_rollouts"],
            "source_anchor_indices":result["source_anchor_indices"],
            "source_anchors_exact":bool(np.array_equal(
                result["latents"][result["source_anchor_indices"]],
                result["forward"][result["source_anchor_indices"]])),
            "forward":result["forward"].ravel().tolist(),"backward":result["latents"].ravel().tolist(),
            "calls":calls,"unaligned_averaging":False,"natural_long_video_validated":False}
