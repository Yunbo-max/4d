"""Engineering hooks only: fixtures are not native method evidence."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch

class Decoder(torch.nn.Module):
    prediction_mode='direct'
    def forward(self,latent,framestep,source_alpha,target_alphas,query,step_callback=None):
        if step_callback:step_callback(1,1)
        return query[:,:,:3][:,None].expand(-1,target_alphas.shape[1],-1,-1).clone()

class DecoderObserverTests(unittest.TestCase):
    def inputs(self):
        return dict(latent=torch.zeros(1,2,3,4),framestep=torch.tensor([[0.,1.]]),
                    source_alpha=torch.zeros(1),target_alphas=torch.ones(1,1),
                    query=torch.arange(30,dtype=torch.float32).reshape(1,5,6)/30)

    def test_capture_is_bit_exact_nonmutating_and_preserves_rng(self):
        from research_math.decoder_observer import DecoderObserver, load_capture
        with tempfile.TemporaryDirectory() as d:
            args=self.inputs();model=Decoder().eval();expected=model(**args)
            rng=torch.get_rng_state().clone();before={k:v.clone() for k,v in args.items()}
            with DecoderObserver(model,Path(d)/'capture',{'scope':'engineering fixture'}) as observer:
                got=model(**args)
            self.assertTrue(torch.equal(expected,got));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertTrue(all(torch.equal(before[k],v) for k,v in args.items()))
            saved,record=load_capture(Path(d)/'capture/call-0000')
            self.assertTrue(torch.equal(saved['output'],got))
            self.assertTrue(all(torch.equal(saved[k],v) for k,v in args.items()))
            self.assertEqual(record['status'],'captured_unqualified')
            self.assertFalse(record['scientific_effect_qualification'])
            self.assertIn('cuda_autocast_enabled',record)
            self.assertIn('cuda_autocast_dtype',record)
            self.assertIn('cpu_autocast_enabled',record)
            self.assertIn('cpu_autocast_dtype',record)
            self.assertEqual(len(model._forward_pre_hooks),0);self.assertEqual(len(model._forward_hooks),0)

    def test_existing_output_is_preserved_and_tampering_rejected(self):
        from research_math.decoder_observer import DecoderObserver,load_capture
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'capture';model=Decoder().eval()
            with DecoderObserver(model,root,{}):model(**self.inputs())
            with self.assertRaises(FileExistsError):DecoderObserver(model,root,{})
            p=root/'call-0000/tensors.safetensors';p.write_bytes(p.read_bytes()+b'bad')
            with self.assertRaises(ValueError):load_capture(root/'call-0000')

    def test_byte_limit_rejects_capture_before_forward(self):
        from research_math.decoder_observer import DecoderObserver
        called=[];model=Decoder().eval()
        with tempfile.TemporaryDirectory() as d:
            with DecoderObserver(model,Path(d)/'capture',{},max_bytes=1):
                with self.assertRaises(ValueError):model(**self.inputs(),step_callback=lambda *x:called.append(x))
            self.assertEqual(called,[])

    def test_partial_hook_registration_is_rolled_back(self):
        from research_math.decoder_observer import DecoderObserver
        model=Decoder().eval()
        with tempfile.TemporaryDirectory() as d:
            observer=DecoderObserver(model,Path(d)/'capture',{})
            with patch.object(model,'register_forward_hook',
                              side_effect=RuntimeError('hook registration failed')):
                with self.assertRaisesRegex(RuntimeError,'hook registration failed'):
                    with observer:self.fail('Partially attached observer entered')
            self.assertEqual(len(model._forward_pre_hooks),0)
            self.assertEqual(len(model._forward_hooks),0)
            self.assertEqual(observer.handles,[])
            model(**self.inputs())
            self.assertEqual(observer.calls,0)
            self.assertFalse((Path(d)/'capture/call-0000').exists())

    def test_persisted_payload_limit_counts_both_input_files(self):
        from research_math.decoder_observer import DecoderObserver
        # Inputs: 96 + 8 + 4 + 4 + 120 = 232 bytes; output: 60 bytes.
        # Persisted payload is 232 + (232 + 60) = 524 bytes, not 292.
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'capture';model=Decoder().eval()
            with DecoderObserver(model,root,{},max_bytes=523):
                with self.assertRaisesRegex(ValueError,'byte bound'):
                    model(**self.inputs())
            call=root/'call-0000'
            self.assertTrue((call/'inputs.safetensors').exists())
            self.assertFalse((call/'tensors.safetensors').exists())
            record=json.loads((call/'record.json').read_text())
            self.assertEqual(record['status'],'output_byte_bound_exceeded')
            self.assertEqual(record['payload_bytes'],232)
            self.assertEqual(len(model._forward_pre_hooks),0)
            self.assertEqual(len(model._forward_hooks),0)

    def test_exact_persisted_payload_limit_accepts_complete_capture(self):
        from research_math.decoder_observer import DecoderObserver,load_capture
        from safetensors.torch import load_file
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'capture';model=Decoder().eval()
            with DecoderObserver(model,root,{},max_bytes=524):
                model(**self.inputs())
            call=root/'call-0000'
            _,record=load_capture(call)
            persisted=sum(value.numel()*value.element_size()
                          for name in ('inputs.safetensors','tensors.safetensors')
                          for value in load_file(str(call/name)).values())
            self.assertEqual(persisted,524)
            self.assertEqual(record['payload_bytes'],persisted)
            self.assertEqual(record['byte_bound_scope'],
                             'persisted_tensor_payload_excludes_headers')

    def test_forward_failure_keeps_inputs_and_detaches_hooks(self):
        from research_math.decoder_observer import DecoderObserver
        class Broken(Decoder):
            def forward(self,*args,**kwargs):raise RuntimeError('native failure')
        # Keep a concrete signature so the capture cannot silently omit arguments.
        model=Decoder().eval()
        def fail(*args):raise RuntimeError('native failure')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'capture'
            with self.assertRaisesRegex(RuntimeError,'native failure'):
                with DecoderObserver(model,root,{}):model(**self.inputs(),step_callback=fail)
            record=json.loads((root/'call-0000/record.json').read_text())
            self.assertEqual(record['status'],'forward_failed')
            self.assertTrue((root/'call-0000/inputs.safetensors').exists())
            self.assertEqual(len(model._forward_pre_hooks),0)

    def test_positional_inputs_and_bfloat16_roundtrip(self):
        from research_math.decoder_observer import DecoderObserver,load_capture
        with tempfile.TemporaryDirectory() as d:
            args=self.inputs();args['latent']=args['latent'].to(torch.bfloat16)
            model=Decoder().eval()
            with DecoderObserver(model,Path(d)/'capture',{}):
                model(*args.values())
            saved,_=load_capture(Path(d)/'capture/call-0000')
            self.assertEqual(saved['latent'].dtype,torch.bfloat16)
            self.assertTrue(torch.equal(saved['latent'],args['latent']))
