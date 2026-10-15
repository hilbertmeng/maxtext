"""Serial Pythia: independent NeoX oracle, scan gradients and unchanged budgets."""
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
from pythia_family_test import PythiaFamilyTest
from pythia_test import PythiaTest

class PythiaSerialTest(PythiaFamilyTest):
  def config(self, **kwargs):
    names = {
        'PythiaMediumPropT2048': 'PythiaMediumPropT2048SerialMLP',
        'BamPythiaMediumPropT2048MShiftReadG050ReadAttentionSharedR256':
            'BamPythiaMediumPropT2048MShiftReadG050ReadAttentionSharedR256SerialMLP'}
    if kwargs.get('exp_class') in names:
      kwargs['exp_class'] = names[kwargs['exp_class']]
    kwargs['pythia_parallel_residual'] = False
    return super().config(**kwargs)

  def test_parameter_tree_and_initialization_unchanged(self):
    c=self.config(scan_layers=False)
    pc=PythiaTest.config(self,scan_layers=False)
    m,mesh,p=self.init(c);pm,_,pp=self.init(pc)
    a,b=flatten_dict(nn.unbox(p)),flatten_dict(nn.unbox(pp))
    self.assertEqual(set(a),set(b))
    for k in a:np.testing.assert_array_equal(a[k],b[k],err_msg=str(k))
    with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
      y=m.apply({'params':p},*self.inputs(),enable_dropout=False)[0]
      py=pm.apply({'params':pp},*self.inputs(),enable_dropout=False)[0]
    self.assertGreater(float(np.max(np.abs(y-py))),1e-5)

if __name__=='__main__':
  import unittest
  unittest.main()
