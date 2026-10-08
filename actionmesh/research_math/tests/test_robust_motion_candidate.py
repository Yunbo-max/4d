"""C04 software contracts and retained-native acceptance; source only at authoring.

Synthetic arrays below verify algebra/engineering, never ActionBench outcomes.
The retained-native class requires a real receipt-bound candidate artifact.
"""
import json
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
from research_math import robust_motion_candidate as core


class RobustMotionMathematics(unittest.TestCase):
    def solve(self,d,g,s,**kw):
        options=dict(radius=.7,epsilon=.2,absolute_tolerance=1e-11,
                     relative_tolerance=1e-9,max_iterations=256)
        options.update(kw)
        return core.solve_anchored_conic_projection(d,g,s,**options)

    def arrays(self):
        d=np.zeros((3,2,3));d[1:]=np.arange(12).reshape(2,2,3)/10
        g=np.zeros_like(d);s=np.ones_like(d)
        return d,g,s

    def test_isotropic_ball_has_analytic_projection_and_dual_certificate(self):
        d,g,s=self.arrays(); x,c=self.solve(d,g,s)
        expected=d*(.2/.7)/np.linalg.norm(d)
        np.testing.assert_allclose(x,expected,atol=1e-8)
        self.assertLessEqual(c['support'],.2+1e-11)
        self.assertLess(c['primal_dual_gap'],1e-7)
        self.assertLess(c['stationarity_l2'],1e-8)
        np.testing.assert_array_equal(x[0],0)

    def test_deterministic_slab_analytic_and_zero_gradient_identity(self):
        d,g,s=self.arrays();g[1:,0,0]=1.
        x,c=self.solve(d,g,s,radius=0)
        amount=max(0,float(np.sum(g*d))-.2)/float(np.sum(g*g))
        np.testing.assert_allclose(x,d-amount*g,atol=1e-8)
        zero,c=self.solve(d,g*0,s,radius=0)
        np.testing.assert_array_equal(zero,d)
        self.assertEqual(c['primal_dual_gap'],0)

    def test_anisotropic_support_and_shape_rejection(self):
        d,g,s=self.arrays();g[1:]=.3;s[1:,1,:]=5.
        x,c=self.solve(d,g,s)
        self.assertLessEqual(core.ellipsoid_support(x,g,s,.7),.2+1e-11)
        self.assertLess(c['primal_dual_gap'],1e-7)
        self.assertLessEqual(c['dual_u_norm'],1+1e-12)
        for bad in (0.,-1.,float('nan')):
            t=s.copy();t[1,0,0]=bad
            with self.assertRaises(ValueError): self.solve(d,g,t)

    def test_exact_nonuniform_time_quadratic_remainder_and_hessian_bound(self):
        times=np.linspace(0,1,16)**1.3
        x=np.arange(16*4*3,dtype=np.float64).reshape(16,4,3)/500
        model=core.motion_surrogate(x,times,shape_floor=.1,shape_gain=.9)
        delta=np.sin(np.arange(x.size)).reshape(x.shape)*1e-4;delta[0]=0
        changed=core.motion_surrogate(x+delta,times,shape_floor=.1,shape_gain=.9)
        rem=.5*np.sum(np.diff(delta,axis=0)**2/np.diff(times)[:,None,None])/model['normalization']
        self.assertAlmostEqual(changed['value']-model['value'],float(np.sum(model['g0']*delta))+rem,places=11)
        self.assertLessEqual(rem,.5*model['lipschitz']*float(np.sum(delta*delta))+1e-12)
        self.assertTrue(np.all(model['shape_diagonal']>=.1))

    def test_backtrack_failure_retains_all_trials(self):
        x=np.ones((16,4,3),dtype=np.float32);times=np.linspace(0,1,16)
        model=core.motion_surrogate(x,times,shape_floor=.1,shape_gain=.9)
        projected=np.ones_like(x,dtype=float);projected[0]=0
        p=dict(trust_radius=10.,max_backtracks=3,epsilon=1e-20,
               finite_budget=1e-20,absolute_tolerance=1e-25)
        with self.assertRaises(core.BacktrackingFailure) as caught:
            core._backtrack(x,projected,times,model,p,1.)
        self.assertEqual(len(caught.exception.rejected),4)
        self.assertEqual([r['scale'] for r in caught.exception.rejected],
                         [caught.exception.rejected[0]['scale']*.5**i for i in range(4)])


class RobustMotionFailureRetention(unittest.TestCase):
    def test_actual_numerical_exhaustion_retains_robust_and_dependent_failures(self):
        # Engineering algebra fixture, never a native result. No solver or
        # failure boundary is mocked: the real bounded solver exhausts.
        source=np.zeros((16,4,3),dtype=np.float32)
        times=np.linspace(0.,1.,16)
        desired=np.ones_like(source,dtype=np.float64)*10.;desired[0]=0.
        model=core.motion_surrogate(source,times,shape_floor=.1,shape_gain=.9)
        p={name:.1 for name in core.FLOAT_PARAMETERS}
        p.update({name:64 for name in core.INTEGER_PARAMETERS})
        p.update(max_iterations=1,absolute_tolerance=1e-12,relative_tolerance=1e-12,
                 epsilon=.001,trust_radius=1e4,finite_budget=1e8)
        outcomes=core.construct_all_arms(source,times,desired,model,p)
        self.assertEqual(set(outcomes),set(core.ROLES))
        self.assertEqual(outcomes['deterministic_protection']['status'],'completed')
        self.assertEqual(outcomes['robust_conic_protection']['status'],'error')
        self.assertEqual(outcomes['strength_matched_repair']['status'],'error')
        self.assertIn('Conic',outcomes['robust_conic_protection']['error'])
        self.assertNotIn('vertices',outcomes['robust_conic_protection'])
        self.assertNotIn('vertices',outcomes['strength_matched_repair'])


class C04RetainedNativeAcceptance(unittest.TestCase):
    def test_receipt_bound_native_artifact_reconstruction(self):
        root=os.environ.get('C04_NATIVE_ROOT');artifact=os.environ.get('C04_NATIVE_ARTIFACT')
        if not root or not artifact:
            self.skipTest('Retained native acceptance requires C04_NATIVE_ROOT and C04_NATIVE_ARTIFACT; not covered by software fixtures')
        checked=core.validate_candidate_artifact(Path(root),Path(artifact))
        self.assertEqual(checked['candidate_id'],core.CANDIDATE_ID)
        self.assertEqual(tuple(checked['roles']),core.ROLES)
        self.assertEqual(len(checked['arms']),3)
        self.assertFalse(checked['native_qualified'])
        self.assertFalse(checked['scientific_admission'])
        # A retained terminal failure is valid transport evidence, not acceptance.
        self.assertEqual(checked['status'],'completed',
                         'At least one real native method arm failed; Local acceptance stays pending')
        self.assertTrue(all(r['status']=='completed' for r in checked['arms']))


if __name__=='__main__': unittest.main()
