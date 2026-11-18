import runpy,unittest
from pathlib import Path
import numpy as np
import jax
m=runpy.run_path(str(Path(__file__).with_name('xlprop_wi1_column_probe.py')))
class Tests(unittest.TestCase):
  def test_scanned_parameter_identity_and_axis(self):
    path=lambda keys:tuple(jax.tree_util.DictKey(k) for k in keys.split('/'))
    p='params/decoder/layers/block/mlp/wi_1/kernel'
    self.assertTrue(m['select_wi1'](path(p),(1920,27,4048)))
    self.assertFalse(m['select_wi1'](path(p.replace('layers','final_local_layer')),(1920,4048)))
    self.assertFalse(m['select_wi1'](path(p.replace('mlp','memory_mlp')),(1920,27,4048)))
    with self.assertRaises(AssertionError):m['select_wi1'](path(p),(27,1920,4048))
  def test_column_energy_matches_matrix_norm(self):
    g=np.zeros((3,4048));g[0,51]=3;g[1,301]=4;g[2,2000]=12
    s=m['column_summary'](g)
    self.assertAlmostEqual(s['norm'],13)
    self.assertEqual(s['top_neurons'][0],2000)
    self.assertAlmostEqual(s['fixed_four_energy_fraction'],25/169)
    self.assertAlmostEqual(sum(s['column_squared_norms']),np.linalg.norm(g)**2)
  def test_token_projection_retains_cancellation(self):
    x=np.array([[[1.,0],[1,0],[0,1]]]);u=np.array([[[2.,1],[-1,-.5],[0,3]]])
    g=x.reshape(-1,2).T@u.reshape(-1,2)
    p,error=m['token_projection'](x,u,g)
    self.assertEqual(error,0);self.assertAlmostEqual(p.sum(),1)
    self.assertLess(p[0,1],0)
    _,bad=m['token_projection'](x,u,g*2)
    self.assertGreater(bad,.01)
if __name__=='__main__':unittest.main()
