import unittest
import numpy as np
from wo_boundary_decomposition import partition_sums, merge_sums, describe

class BoundaryTests(unittest.TestCase):
    def test_partition_merge_equals_full_same_G(self):
        rng=np.random.default_rng(123)
        a=rng.normal(size=(4,7,2,3)).astype('float32');o=rng.normal(size=a.shape).astype('float32')
        h=a+o;g=rng.normal(size=(4,7,5)).astype('float32')/28
        valid=np.ones((4,7),bool);valid[0,6]=False;g[0,6]=0
        def part(ix,chunk):
            return partition_sums(a[ix],o[ix],h[ix],g[ix],np.arange(4)[ix],valid[ix],token=3,sequence=2,chunk_tokens=chunk)
        whole=part(slice(None),7)
        merged=merge_sums([part([0,2],2),part([1,3],3)],range(4))
        np.testing.assert_allclose(whole['parts'],merged['parts'],rtol=1e-13,atol=1e-13)
        np.testing.assert_allclose(whole['amplitude'],merged['amplitude'],rtol=1e-13,atol=1e-13)
        expected=np.einsum('btnh,btd->nhd',a.astype('float64'),g.astype('float64'))
        np.testing.assert_allclose(merged['parts'][0,0],expected,rtol=1e-13,atol=1e-13)
        single=merged['parts'][2,0]
        self.assertAlmostEqual(np.linalg.norm(single)/np.linalg.norm(a[2,3].astype('float64')),np.linalg.norm(g[2,3].astype('float64')),places=12)
        with self.assertRaises(ValueError):merge_sums([part([0,2],2),part([0,1],2)],range(4))

    def test_cancellation_requires_matrix_merge_and_signed_projection(self):
        a=np.array([1.,-1.]).reshape(2,1,1,1).astype('float32');o=-.8*a
        g=np.ones((2,1,1),'float32');v=np.ones((2,1),bool)
        p=partition_sums(a,o,a+o,g,[0,1],v,token=0,sequence=0)
        self.assertEqual(describe(p)[0]['sum_gradient_norm'],0)
        self.assertLess(describe(p)[1]['O_projection_fraction'],0)
        h=(a+o).copy();h[0,0,0,0]+=.001
        shifted=partition_sums(a,o,h,g,[0,1],v,token=0,sequence=0)
        self.assertGreater(describe(shifted)[0]['rounding_gradient_relative_error'],.9)

if __name__=='__main__':unittest.main()
