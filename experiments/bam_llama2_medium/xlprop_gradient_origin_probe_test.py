import runpy,unittest
from pathlib import Path
import jax,jax.numpy as jnp
import numpy as np
m=runpy.run_path(str(Path(__file__).with_name('xlprop_gradient_origin_probe.py')))
class Tests(unittest.TestCase):
  def test_partition_and_global_normalization(self):
    masks=[m['cohort_weights'](8,16,jnp.int32(i))for i in range(9)]
    np.testing.assert_array_equal(masks[0],np.ones((8,16),bool))
    for group in [masks[1:5],masks[5:9]]:
      np.testing.assert_array_equal(sum(x.astype(int)for x in group),np.ones((8,16),int))
      def loss(w,mask):return jnp.sum((jnp.arange(128).reshape(8,16)*w)**2*mask)/128
      full=jax.grad(loss)(jnp.float32(2),masks[0]);parts=[jax.grad(loss)(jnp.float32(2),x)for x in group]
      np.testing.assert_allclose(sum(parts),full)
  def test_selected_family_and_scan_axis(self):
    p=lambda part:tuple(jax.tree_util.DictKey(x)for x in ['params','decoder','layers','block','self_attention',part,'kernel'])
    self.assertEqual(m['selected_leaf'](p('out'),(20,27,96,1920)),'W_O')
    self.assertEqual(m['selected_leaf'](p('W_lv_key'),(1920,27,32)),'V_read')
    self.assertIsNone(m['selected_leaf'](p('W_R_key'),(1920,27,32)))
    self.assertIsNone(m['selected_leaf'](p('query'),(1920,27,20,24)))
    with self.assertRaises(AssertionError):m['selected_leaf'](p('out'),(20,28,96,1920))
if __name__=='__main__':unittest.main()
