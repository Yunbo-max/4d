"""Lazy native adapters. No model loading, downloads, training or GT access.

Callers supply loaded frozen models and provenance-qualified shared contexts.
The decoder uses the FULL latent-context clock to normalize source/target alpha.
The window adapter calls the actual pipeline _denoise_latents with a fresh
LatentBank; it does not interpolate or average independently generated latents.
"""
import numpy as np


class ActionMeshDecoderAdapter:
    def __init__(self, decoder, latents, times, *, device="cuda:0", autocast=True):
        import torch
        self.torch=torch;self.device=torch.device(device);self.autocast=bool(autocast)
        self.decoder=decoder;self.times=np.asarray(times,dtype=float)
        if (self.times.ndim!=1 or len(self.times)<2 or not np.isfinite(self.times).all()
                or np.any(np.diff(self.times)<=0)):
            raise ValueError("finite strictly increasing context clock required")
        tensor=torch.as_tensor(latents,device=self.device).detach().clone()
        if tensor.ndim==3:
            tensor=tensor[None]
        if (tensor.ndim!=4 or tensor.shape[0]!=1 or tensor.shape[1]!=len(self.times)
                or not tensor.is_floating_point() or min(tensor.shape)<=0
                or not torch.isfinite(tensor).all()):
            raise ValueError("finite latent context [T,N,D] or [1,T,N,D] required")
        if decoder.training or any(p.requires_grad for p in decoder.parameters()):
            raise ValueError("caller must put decoder in eval mode and freeze its parameters")
        self.latents=tensor
        self.framestep=torch.as_tensor(self.times,dtype=torch.float32,device=self.device)[None]
        if not torch.all(self.framestep[:,1:]>self.framestep[:,:-1]):
            raise ValueError("clock loses distinctness in native float32")
        self.records=[]

    def __call__(self, source_time, target_times, positions, normals, vertex_ids):
        torch=self.torch
        targets=np.asarray(target_times,dtype=float);xyz=np.asarray(positions,dtype=float)
        normals=np.asarray(normals,dtype=float);ids=np.asarray(vertex_ids)
        if (not np.isfinite(source_time) or targets.ndim!=1 or len(targets)==0
                or not np.isfinite(targets).all() or xyz.ndim!=2 or xyz.shape[1]!=3
                or len(xyz)==0 or normals.shape!=xyz.shape or not np.isfinite(xyz).all()
                or not np.isfinite(normals).all()
                or not np.allclose(np.linalg.norm(normals,axis=1),1.,rtol=0,atol=1e-4)
                or ids.shape!=(len(xyz),) or ids.dtype.kind not in "iu"
                or np.any(ids<0) or len(np.unique(ids))!=len(ids)
                or min(source_time,targets.min())<self.times[0]
                or max(source_time,targets.max())>self.times[-1]):
            raise ValueError("invalid decoder times, material IDs, positions or unit normals")
        query=torch.as_tensor(np.concatenate([xyz,normals],axis=-1),dtype=torch.float32,
                              device=self.device)[None]
        # Exact get_scaling/apply_scaling algebra, using full native context.
        low=self.framestep.min(dim=1).values
        span=self.framestep.max(dim=1).values-low
        source=(torch.tensor([source_time],dtype=torch.float32,device=self.device)-low)/span
        target=(torch.as_tensor(targets,dtype=torch.float32,device=self.device)[None]-low[:,None])/span[:,None]
        with torch.inference_mode(), torch.autocast(device_type=self.device.type,
                dtype=torch.float16,enabled=self.autocast and self.device.type=="cuda"):
            displacement=self.decoder(self.latents,self.framestep,source,target,query)
            absolute=self.decoder.apply_displacement(vertex=query[...,:3],displacement=displacement)
        expected=(1,len(targets),len(xyz),3)
        if tuple(absolute.shape)!=expected or not torch.isfinite(absolute).all():
            raise ValueError("native decoder returned invalid absolute positions")
        self.records.append({"source_time":float(source_time),"target_times":targets.tolist(),
                             "points":len(ids),"point_targets":len(ids)*len(targets)})
        return absolute[0].float().cpu().numpy()


class ActionMeshWindowAdapter:
    def __init__(self, pipeline, input_data, context, representation_id, *,
                 autocast=True, step_callback=None):
        import torch
        self.torch=torch;self.pipeline=pipeline;self.input_data=input_data
        self.device=torch.device(pipeline.device);self.autocast=bool(autocast)
        self.step_callback=step_callback
        if not isinstance(representation_id,str) or not representation_id.strip():
            raise ValueError("provenance-qualified shared representation identity required")
        self.representation_id=representation_id
        self.times=input_data.timesteps.detach().cpu().numpy().astype(float)
        if (self.times.ndim!=1 or len(self.times)<16 or not np.isfinite(self.times).all()
                or np.any(np.diff(self.times)<1e-5)):
            raise ValueError("native clock must be distinct under LatentBank 1e-5 matching")
        self.context=torch.as_tensor(context,device=self.device).detach()
        if (self.context.ndim!=3 or len(self.context)!=len(self.times)
                or min(self.context.shape)<=0 or not torch.isfinite(self.context).all()):
            raise ValueError("full video context [T,S,D] required")
        model=pipeline.temporal_3D_denoiser
        if model is None or model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError("caller must load, eval and freeze the native denoiser")
        self.latent_shape=tuple(pipeline._denoiser_latent_shape)
        self.records=[]

    def __call__(self, indices, known, direction, seed, representation_id):
        import torch
        from actionmesh.model.utils.storage import LatentBank
        indices=np.asarray(indices)
        if (representation_id!=self.representation_id or direction not in ("forward","backward")
                or indices.ndim!=1 or indices.dtype.kind not in "iu" or len(indices)<16
                or np.any(np.diff(indices)<=0) or indices.min()<0 or indices.max()>=len(self.times)
                or not isinstance(seed,(int,np.integer)) or seed<0):
            raise ValueError("invalid native window: >=16 ordered global frames and shared identity required")
        if not known or any(not isinstance(k,(int,np.integer)) or k not in indices for k in known):
            raise ValueError("known conditions must have global IDs inside this window")
        if len(known)==len(indices):
            raise ValueError("native sampler needs at least one unknown frame")
        bank=LatentBank(empty_dims=self.latent_shape)
        keys=sorted(known)
        values=np.stack([np.asarray(known[k]) for k in keys])
        if values.shape!=(len(keys),*self.latent_shape) or not np.isfinite(values).all():
            raise ValueError("known latent shapes must match native denoiser")
        bank.update(torch.as_tensor(self.times[keys],dtype=torch.float32),
                    torch.as_tensor(values,dtype=torch.float32,device=self.device))
        selected=self.input_data.get(indices.tolist())
        context=self.context[torch.as_tensor(indices,dtype=torch.long,device=self.device)]
        with torch.inference_mode(),torch.autocast(device_type=self.device.type,dtype=torch.float16,
                enabled=self.autocast and self.device.type=="cuda"):
            result=self.pipeline._denoise_latents(selected,context,bank,seed=int(seed),
                                                   step_callback=self.step_callback)
        if tuple(result.shape)!=(1,len(indices),*self.latent_shape) or not torch.isfinite(result).all():
            raise ValueError("native sampler returned invalid latents")
        output=result[0].float().cpu().numpy()
        for key in keys:
            if not np.array_equal(output[np.flatnonzero(indices==key)[0]],values[keys.index(key)]):
                raise ValueError("native sampler changed a hard conditioning latent")
        self.records.append({"direction":direction,"indices":indices.tolist(),"seed":int(seed),
                             "known_global_ids":keys,"native_denoise_calls":1})
        return {"latents":output,"representation_id":self.representation_id}
