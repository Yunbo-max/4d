"""Non-intervening decoder tensor capture for later native-context admission.

Captured tensors are UNQUALIFIED engineering artifacts. Capturing a call does
not prove replay equivalence or authorize a candidate. Attach only around an
already-loaded decoder under a separately frozen observer experiment.
"""
from __future__ import annotations
import hashlib
import inspect
import json
from pathlib import Path

FIELDS=('latent','framestep','source_alpha','target_alphas','query')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_record(path, record):
    path.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')


def tensor_inventory(tensors):
    return {name:{'shape':list(value.shape),'dtype':str(value.dtype),
                  'bytes':value.numel()*value.element_size()}
            for name,value in tensors.items()}


def autocast_state(torch,device):
    """Read autocast metadata across PyTorch's pre-device and device APIs."""
    try:
        return (torch.is_autocast_enabled(device),torch.get_autocast_dtype(device))
    except TypeError:
        if device=='cuda':
            return (torch.is_autocast_enabled(),torch.get_autocast_gpu_dtype())
        if device=='cpu':
            return (torch.is_autocast_cpu_enabled(),torch.get_autocast_cpu_dtype())
        raise


class DecoderObserver:
    """Save actual forward arguments/output without substituting either.

    Hooks return None and never change the model, RNG, autocast or precision.
    Tensor copies synchronize devices and add overhead, which must be measured
    in the later admission experiment. One root and one directory per call.
    """
    def __init__(self,model,root:Path,identity:dict,*,max_bytes=64*1024*1024):
        if not isinstance(max_bytes,int) or max_bytes<1:
            raise ValueError('Positive capture byte bound required')
        self.model=model;self.root=Path(root)
        self.identity=json.loads(json.dumps(identity,allow_nan=False))
        self.max_bytes=max_bytes;self.handles=[];self.pending=None;self.calls=0
        self.signature=inspect.signature(model.forward)
        if not set(FIELDS).issubset(self.signature.parameters):
            raise ValueError('Explicit complete decoder forward signature required')
        self.root.mkdir(parents=True,exist_ok=False)
        write_record(self.root/'identity.json',self.identity)

    def _before(self,module,args,kwargs):
        import torch
        from safetensors.torch import save_file
        if self.pending is not None:raise ValueError('Nested capture is not supported')
        bound=self.signature.bind(*args,**kwargs);bound.apply_defaults()
        values={name:bound.arguments[name] for name in FIELDS}
        if any(not isinstance(value,torch.Tensor) for value in values.values()):
            raise ValueError('All five native decoder inputs must be tensors')
        size=sum(v.numel()*v.element_size() for v in values.values())
        if size>self.max_bytes:raise ValueError('Decoder capture exceeds byte bound')
        path=self.root/f'call-{self.calls:04d}';path.mkdir(exist_ok=False);self.calls+=1
        # copy=True prevents CPU aliases; clone is unnecessary after a forced copy.
        tensors={k:v.detach().to(device='cpu',copy=True).contiguous() for k,v in values.items()}
        save_file(tensors,str(path/'inputs.safetensors'))
        record={'kind':'decoder-call-capture','version':1,'status':'inputs_captured',
                'scientific_effect_qualification':False,'replay_qualified':False,
                'identity_sha256':sha256(self.root/'identity.json'),
                'prediction_mode':getattr(module,'prediction_mode',None),
                'training':bool(module.training),'inference_mode':torch.is_inference_mode_enabled(),
                'cuda_autocast_enabled':cuda_autocast_enabled,
                'cuda_autocast_dtype':str(cuda_autocast_dtype),
                'cpu_autocast_enabled':cpu_autocast_enabled,
                'cpu_autocast_dtype':str(cpu_autocast_dtype),
                'input_devices':{k:str(v.device) for k,v in values.items()},
                'step_callback_present':bound.arguments.get('step_callback') is not None,
                'inputs_sha256':sha256(path/'inputs.safetensors'),
                'tensors':tensor_inventory(tensors),'max_bytes':self.max_bytes}
        write_record(path/'record.json',record)
        self.pending=(path,tensors,record,size)
        # No replacement args or kwargs: the model receives the original objects.

    def _after(self,module,args,kwargs,output):
        import torch
        from safetensors.torch import save_file
        if self.pending is None:return
        path,tensors,record,size=self.pending
        self.pending=None
        if output is None:
            record['status']='forward_failed';write_record(path/'record.json',record);return
        if not isinstance(output,torch.Tensor):
            record['status']='unsupported_output';write_record(path/'record.json',record)
            raise ValueError('Native decoder tensor output required')
        if size+output.numel()*output.element_size()>self.max_bytes:
            record['status']='output_byte_bound_exceeded';write_record(path/'record.json',record)
            raise ValueError('Decoder output capture exceeds byte bound')
        tensors['output']=output.detach().to(device='cpu',copy=True).contiguous()
        save_file(tensors,str(path/'tensors.safetensors'))
        record.update(status='captured_unqualified',tensors=tensor_inventory(tensors),
                      tensors_sha256=sha256(path/'tensors.safetensors'))
        write_record(path/'record.json',record)
        # No replacement output: the caller receives the original object.

    def __enter__(self):
        if self.handles:raise ValueError('Observer already attached')
        if self.model.training:raise ValueError('Only an eval-mode native decoder may be observed')
        self.handles=[self.model.register_forward_pre_hook(self._before,with_kwargs=True),
                      self.model.register_forward_hook(self._after,with_kwargs=True,always_call=True)]
        return self

    def __exit__(self,typ,error,tb):
        for handle in self.handles:handle.remove()
        self.handles=[]
        if self.pending is not None:
            path,_,record,_=self.pending
            record['status']='forward_failed';write_record(path/'record.json',record)
            self.pending=None
        return False


def load_capture(root:Path):
    """Verify bytes and tensor metadata; this does NOT admit scientific use."""
    from safetensors.torch import load_file
    root=Path(root);record=json.loads((root/'record.json').read_text())
    if record.get('kind')!='decoder-call-capture' or record.get('version')!=1:
        raise ValueError('Unsupported decoder capture')
    if record.get('status')!='captured_unqualified':raise ValueError('Incomplete decoder call')
    for field,path in [('inputs_sha256',root/'inputs.safetensors'),
                       ('tensors_sha256',root/'tensors.safetensors'),
                       ('identity_sha256',root.parent/'identity.json')]:
        if sha256(path)!=record.get(field):raise ValueError('Capture byte hash mismatch: '+field)
    tensors=load_file(str(root/'tensors.safetensors'),device='cpu')
    if set(tensors)!=set(FIELDS)|{'output'} or tensor_inventory(tensors)!=record.get('tensors'):
        raise ValueError('Captured tensor inventory mismatch')
    inputs=load_file(str(root/'inputs.safetensors'),device='cpu')
    import torch
    if set(inputs)!=set(FIELDS) or any(not torch.equal(inputs[k],tensors[k]) for k in FIELDS):
        raise ValueError('Captured input copies disagree')
    return tensors,record
