import unittest,runpy
from pathlib import Path
import jax.numpy as jnp
import numpy as np
root=Path(__file__).parent
bridge=runpy.run_path(str(root/'xlprop_ordinary_parameter_bridge_probe.py'))
classify=runpy.run_path(str(root/'xlprop_parameter_family_probe.py'))['parameter_family']
selector=runpy.run_path(str(root/'xlprop_routing_localization_probe.py'))['selected_parameter']
class SelectionTest(unittest.TestCase):
 def test_only_selected_family_and_scanned_layers_change(self):
  x=jnp.zeros((2,27,3));y=jnp.ones_like(x)
  def tree(z):return {'decoder':{'layers':{'block':{'self_attention':{'out':{'kernel':z},'query':{'kernel':z}}}}},'embedding':x}
  peak=tree(x);old=tree(y)
  for layers in ([3],list(range(6))):
   changed,count,families=bridge['revert_parameters'](peak,old,classify,selector,{'attention_output_projection'},layers)
   out=np.asarray(changed['decoder']['layers']['block']['self_attention']['out']['kernel']);ref=np.zeros((2,27,3));ref[:,layers]=1
   np.testing.assert_array_equal(out,ref);self.assertEqual(count,6*len(layers))
   np.testing.assert_array_equal(changed['decoder']['layers']['block']['self_attention']['query']['kernel'],x)
   np.testing.assert_array_equal(peak['decoder']['layers']['block']['self_attention']['out']['kernel'],x)
 def test_empty_family_selection_rejected(self):
  with self.assertRaises(AssertionError):bridge['revert_parameters']({'x':jnp.zeros((2,27,3))},{'x':jnp.ones((2,27,3))},lambda *args:'unrelated',selector,{'attention_local_v'},[3])
if __name__=='__main__':unittest.main()
