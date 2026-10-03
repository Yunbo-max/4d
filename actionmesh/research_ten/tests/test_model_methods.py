"""Analytic behavioral controls; no natural-model efficacy claims."""
import unittest
import importlib.util
import numpy as np

from research_ten.m02_cycles import vertex_normals, anchored_consensus, probe_cycles
from research_ten.m09_guidance import combine_guidance, run_live_flow
from research_ten.m10_windows import rollout_windows


@unittest.skipUnless(importlib.util.find_spec("torch"),"native Torch tests run on GPU environment")
class TorchAdapterTests(unittest.TestCase):
    def test_three_branch_parity_input_purity_and_half_extreme(self):
        import torch
        from research_ten.m09_guidance import make_torch_guidance
        class Base:
            guidance_at_inference=[[0,0],[0,1],[1,1]]
            guidance_scales=[1.,1.]
            inference_enabled=True
            def get_unobserved_mask(self,mask): return mask==0
        rng=np.random.default_rng(22)
        branches=[rng.normal(size=(1,3,2,4)).astype(np.float32) for _ in range(3)]
        packed=torch.from_numpy(np.concatenate(branches));before=packed.clone()
        for mode in ("projection","scalar","norm_matched"):
            wrapper=make_torch_guidance(Base(),mode=mode)
            wrapper.get_unobserved_mask(torch.tensor([[1,0,0]]))
            got=wrapper.aggregate_cfg(packed)
            reference,_=combine_guidance(*branches,[[1,0,0]],mode=mode)
            np.testing.assert_allclose(got.numpy(),reference,atol=1e-6,rtol=1e-6)
            self.assertTrue(torch.equal(packed,before))
        # Intermediate differences overflow fp16; scalar telescoping is finite.
        extreme=torch.tensor([-60000.,60000.,0.],dtype=torch.float16).reshape(3,1,1,1)
        wrapper=make_torch_guidance(Base(),mode="scalar")
        self.assertEqual(wrapper.aggregate_cfg(extreme).item(),0.)
        with self.assertRaises(ValueError):
            wrapper.aggregate_cfg(torch.empty((0,1,1,1)))

    def test_decoder_adapter_uses_full_context_clock_and_absolute_output(self):
        import torch
        from research_ten.actionmesh_adapter import ActionMeshDecoderAdapter
        class Decoder(torch.nn.Module):
            def forward(self,latents,clock,source,target,query):
                return (target-source[:,None])[:,:,None,None].expand(-1,-1,query.shape[1],3)
            def apply_displacement(self,vertex,displacement):
                return vertex[:,None]+displacement
        decoder=Decoder().eval()
        adapter=ActionMeshDecoderAdapter(decoder,np.zeros((4,2,3),np.float32),[2,4,6,8],device="cpu",autocast=False)
        result=adapter(4.,[6.,8.],[[0.,0.,0.]],[[0.,0.,1.]],[4])
        np.testing.assert_allclose(result[:,0,0],[1./3,2./3],rtol=1e-6)
        self.assertEqual(adapter.records[0]["point_targets"],2)


class CycleTests(unittest.TestCase):
    def setUp(self):
        self.vertices = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
        self.faces = np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]])

    def test_rigid_flow_composes_and_recomputes_intermediate_normals(self):
        calls = []
        def rotation(t):
            a=.2*t
            return np.array([[np.cos(a),-np.sin(a),0.],
                             [np.sin(a),np.cos(a),0.],[0.,0.,1.]])
        def decode(source, targets, positions, normals, ids):
            calls.append((source, normals.copy(), ids.copy()))
            original=(positions-np.array([source,0.,0.]))@rotation(source)
            return np.stack([original@rotation(t).T+[t,0.,0.] for t in targets])
        result=probe_cycles(decode,self.vertices,self.faces,[0.,2.,5.],[0,1])
        self.assertLess(result["max_cycle_residual"],1e-12)
        self.assertLess(result["max_composition_residual"],1e-12)
        np.testing.assert_array_equal(result["consensus"][0],self.vertices[[0,1]])
        at_b=next(c for c in calls if c[0]==2.)
        expected=vertex_normals(self.vertices,self.faces)[[0,1]]@rotation(2.).T
        np.testing.assert_allclose(at_b[1],expected,atol=1e-12)
        np.testing.assert_array_equal(at_b[2],[0,1])
        self.assertEqual(result["query_points"],16)

    def test_consensus_solves_anchored_graph_not_just_averaging(self):
        direct=np.zeros((3,1,3));direct[:,0,0]=[0.,3.,4.]
        edges=[(0,1,np.array([[1.,0.,0.]])),(1,2,np.array([[1.,0.,0.]]))]
        got=anchored_consensus(direct,edges,weight=1.)
        np.testing.assert_allclose(got[:,0,0],[0.,2.2,3.6],atol=1e-12)

    def test_budget_is_checked_before_decoder_work(self):
        calls=[]
        def decode(*args):
            calls.append(args)
            raise AssertionError("budget must reject first")
        with self.assertRaises(ValueError):
            probe_cycles(decode,self.vertices,self.faces,[0.,2.,5.],[0,1],max_query_points=1)
        self.assertFalse(calls)

    def test_degenerate_normals_are_not_silently_invented(self):
        with self.assertRaises(ValueError):
            vertex_normals(np.zeros((3,3)),np.array([[0,1,2]]))

    def test_inconsistent_path_is_detected_without_claiming_quality_gain(self):
        def decode(source,targets,positions,normals,ids):
            return np.stack([positions+[t-source+.1*(t-source)**2,0.,0.] for t in targets])
        result=probe_cycles(decode,self.vertices,self.faces,[0.,2.,5.],[0,1])
        self.assertGreater(result["max_cycle_residual"],0.1)
        self.assertGreater(result["max_composition_residual"],0.1)
        self.assertFalse(result["natural_quality_claim"])


class GuidanceTests(unittest.TestCase):
    def test_projection_removes_only_conflicting_component_and_respects_anchor(self):
        v0=np.zeros((1,2,1,2));vm=v0+np.array([1.,0.])
        vmv=vm+np.array([-1.,2.])
        out,info=combine_guidance(v0,vm,vmv,[[True,False]],mesh_scale=1.,video_scale=1.)
        np.testing.assert_array_equal(out[:,0],vmv[:,0])
        np.testing.assert_allclose(out[0,1,0],[1.,2.],atol=2e-12)
        self.assertEqual(info["conflicting_unknown_frames"],1)

    def test_zero_mesh_and_nonconflict_are_unchanged(self):
        v0=np.zeros((1,2,1,2));vm=v0.copy();vm[:,1]=[1.,0.]
        vmv=vm+np.array([1.,2.])
        got,_=combine_guidance(v0,vm,vmv,[[False,False]],mesh_scale=2.,video_scale=3.)
        np.testing.assert_array_equal(got,v0+2*(vm-v0)+3*(vmv-vm))

    def test_normmatched_and_random_controls_have_declared_budgets(self):
        v0=np.zeros((1,1,1,3));vm=v0+[1.,0.,0.];vmv=vm+[-1.,2.,0.]
        projected,_=combine_guidance(v0,vm,vmv,[[False]],mesh_scale=0.,video_scale=1.)
        matched,_=combine_guidance(v0,vm,vmv,[[False]],mesh_scale=0.,video_scale=1.,mode="norm_matched")
        random,_=combine_guidance(v0,vm,vmv,[[False]],mesh_scale=0.,video_scale=1.,mode="random",seed=12)
        self.assertAlmostEqual(np.linalg.norm(projected),np.linalg.norm(matched),places=12)
        self.assertAlmostEqual(np.linalg.norm(projected-(vmv-vm)),
                               np.linalg.norm(random-(vmv-vm)),places=12)

    def test_live_flow_requeries_changed_state_and_never_changes_known_latent(self):
        states=[]
        initial=np.zeros((1,2,1,2));initial[:,0]=[4.,5.]
        def predict(state,time):
            states.append(state.copy())
            v0=np.zeros_like(state);vm=v0+[1.,0.]
            vmv=vm+np.stack([-np.ones_like(state[...,0]),1.+state[...,1]],axis=-1)
            return v0,vm,vmv
        result=run_live_flow(predict,initial,[1.,.5,0.],[.5,.5],[[True,False]])
        self.assertEqual(len(states),2)
        self.assertFalse(np.array_equal(states[0],states[1]))
        np.testing.assert_array_equal(result["latents"][:,0],initial[:,0])
        self.assertEqual(result["branch_evaluations"],6)

    def test_invalid_shapes_and_nonfinite_are_rejected(self):
        x=np.zeros((1,2,1,2))
        with self.assertRaises(ValueError):
            combine_guidance(x,x,x,[[False]])
        with self.assertRaises(ValueError):
            combine_guidance(x,x,x+np.nan,[[False,False]])


class WindowTests(unittest.TestCase):
    def callback(self,records):
        def rollout(indices,known,direction,seed,representation_id):
            records.append((tuple(indices),{k:v.copy() for k,v in known.items()},direction,seed))
            keys=sorted(known)
            if not keys:
                raise AssertionError("a window must have an anchor")
            # Actual fresh rollout from supplied conditions; no cached velocity.
            values=np.array([np.interp(i,keys,[float(known[k][0]) for k in keys])
                             + (0. if i in known else .1) for i in indices])[:,None]
            return {"latents":values,"representation_id":representation_id}
        return rollout

    def test_four_windows_rerun_backward_and_preserve_original_source_anchors(self):
        records=[]
        result=rollout_windows(self.callback(records),np.arange(13.),4,1,{0:np.array([0.])},
                              "same-anchor",future_reliable=np.ones(13,bool),
                              backward_updates=4,max_rollouts=8)
        self.assertEqual(len(records),8)
        self.assertEqual(result["forward_rollouts"],4)
        self.assertEqual(result["backward_rollouts"],4)
        for frame in [0]:
            np.testing.assert_array_equal(result["latents"][frame],result["forward"][frame])
        self.assertTrue(any(x[2]=="backward" and len(x[1])>=2 for x in records))
        self.assertFalse(result["unaligned_averaging"])

    def test_later_context_updates_earlier_overlap_instead_of_self_conditioning(self):
        def run(future_value):
            def callback(indices,known,direction,seed,representation_id):
                keys=sorted(known)
                values=np.array([np.interp(i,keys,[known[k][0] for k in keys])
                                 for i in indices])[:,None]
                if indices[0]==9 and direction=="forward":
                    values[-1]=future_value  # independent later-window input evidence
                for k,v in known.items():
                    values[np.flatnonzero(indices==k)[0]]=v
                return {"latents":values,"representation_id":representation_id}
            return rollout_windows(callback,np.arange(13.),4,1,{0:np.array([0.])},"id",
                    future_reliable=np.ones(13,bool),backward_updates=4,max_rollouts=8)
        a=run(1.);b=run(2.)
        np.testing.assert_array_equal(a["forward"][:10],b["forward"][:10])
        self.assertGreater(b["latents"][3,0],a["latents"][3,0])
        self.assertEqual(a["source_anchor_indices"],[0])
        self.assertEqual(a["latents"][0,0],0.)

    def test_representation_mismatch_and_mutated_anchors_fail(self):
        def wrong(indices,known,direction,seed,representation_id):
            return {"latents":np.zeros((len(indices),1)),"representation_id":"wrong"}
        with self.assertRaises(ValueError):
            rollout_windows(wrong,np.arange(13.),4,1,{0:np.array([0.])},"id",
                            future_reliable=np.ones(13,bool),backward_updates=0,max_rollouts=4)
        def broken(indices,known,direction,seed,representation_id):
            return {"latents":np.ones((len(indices),1)),"representation_id":representation_id}
        with self.assertRaises(ValueError):
            rollout_windows(broken,np.arange(13.),4,1,{0:np.array([0.])},"id",
                            future_reliable=np.ones(13,bool),backward_updates=0,max_rollouts=4)

    def test_budget_and_short_input_reject_before_rollout(self):
        records=[]
        for total,cap in [(13,3),(7,20)]:
            with self.assertRaises(ValueError):
                rollout_windows(self.callback(records),np.arange(float(total)),4,1,
                                {0:np.array([0.])},"id",future_reliable=np.ones(total,bool),
                                max_rollouts=cap)
        self.assertFalse(records)

    def test_unreliable_future_is_not_used_as_backward_evidence(self):
        records=[]
        result=rollout_windows(self.callback(records),np.arange(13.),4,1,{0:np.array([0.])},
                              "id",future_reliable=np.zeros(13,bool),
                              backward_updates=3,max_rollouts=7)
        self.assertEqual(result["backward_rollouts"],0)
        self.assertEqual(len(records),4)


if __name__ == "__main__":
    unittest.main()
