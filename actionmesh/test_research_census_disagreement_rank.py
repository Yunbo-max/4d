"""CPU fixtures for ranking math, ties, locked sampling and GT isolation."""
import unittest
import numpy as np
import research_census_disagreement_rank as rank


class RankingFixtures(unittest.TestCase):
    def test_monotonic_and_reverse(self):
        values=np.arange(10,dtype=float)
        self.assertAlmostEqual(rank.spearman(values,values),1.)
        self.assertAlmostEqual(rank.spearman(values,-values),-1.)
        result=rank.risk_diagnostic(values,values,np.arange(10))
        self.assertEqual(result['rejected_query_ids'],[9])
        self.assertAlmostEqual(result['risk_enrichment'],2.)
        self.assertAlmostEqual(result['retained_risk'],4.)
        self.assertAlmostEqual(result['retained_risk_ratio'],4/4.5)

    def test_average_ties_and_arbitrary_stable_boundary(self):
        np.testing.assert_array_equal(rank.average_ranks(np.array([3.,1.,1.,2.])),[4.,1.5,1.5,3.])
        score=np.array([2.,2.,1.,0.,0.,0.,0.,0.,0.,0.])
        ids=np.array([40,30,20,10,50,60,70,80,90,100])
        result=rank.risk_diagnostic(score,np.arange(10,dtype=float),ids)
        self.assertEqual(result['rejected_query_ids'],[30])
        self.assertEqual(result['boundary_tie_total'],2)
        self.assertEqual(result['boundary_tie_rejected'],1)
        self.assertTrue(result['boundary_tie_is_split'])

    def test_constant_and_zero_target_are_undefined(self):
        self.assertIsNone(rank.spearman(np.ones(10),np.arange(10)))
        for score,outcome in [(np.ones(10),np.arange(10)),(np.arange(10),np.zeros(10))]:
            result=rank.risk_diagnostic(score,outcome,np.arange(10))
            self.assertEqual(result['status'],'undefined')
            self.assertIsNone(result['risk_enrichment'])
            self.assertIsNone(result['retained_risk_ratio'])

    def test_joint_permutation_preserves_rank_and_tie_selection(self):
        score=np.array([4.,4.,3.,2.,1.,0.,0.,2.,1.,3.]);outcome=np.arange(10,dtype=float)
        ids=np.array([70,10,50,30,20,100,40,80,90,60]);p=np.array([4,6,0,9,1,3,2,8,7,5])
        first=rank.risk_diagnostic(score,outcome,ids)
        second=rank.risk_diagnostic(score[p],outcome[p],ids[p])
        self.assertAlmostEqual(first['spearman'],second['spearman'])
        self.assertEqual(first['rejected_query_ids'],second['rejected_query_ids'])
        self.assertEqual(first['risk_enrichment'],second['risk_enrichment'])

    def test_fixed_barycentric_identity_and_locked_displacement_target(self):
        vertices=np.array([[[0,0,0],[2,0,0],[0,2,0]],[[0,0,1],[2,0,1],[0,2,1]]],dtype=np.float32)
        face=np.array([[0,1,2]],dtype=np.int64)
        raw=rank.material_samples(vertices,face,np.array([0]),np.array([[.25,.25,.5]],dtype=np.float32))
        np.testing.assert_array_equal(raw[:,0],[[.5,1,0],[.5,1,1]])
        # A constant anchor offset cancels in the conditional displacement EPE.
        predicted=np.array([[[2.,0,0]],[[2.,0,3]],[[2.,0,6]]])
        gt=np.array([[[0.,0,0]],[[0.,0,2]],[[0.,0,4]]])
        target,perframe,oracle=rank.conditional_target(predicted,gt,np.array([0]))
        np.testing.assert_array_equal(perframe,[[1.],[2.]])
        np.testing.assert_array_equal(target,[1.5]);np.testing.assert_array_equal(oracle,[2.])


if __name__=='__main__':unittest.main()
