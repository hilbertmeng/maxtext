"""CPU semantics: uniform/signed scales and scanned vs unscanned layer controls."""
import os
import sys
from pathlib import Path
import unittest

os.environ.setdefault('BAM_RESIDUAL_ATTR_BASE_CONFIG','BamLlama2MediumV2')
sys.path.insert(0,str(Path(__file__).resolve().parent))
import jax.numpy as jnp
import numpy as np
from layers.attentions import _scale_row_cross
from row_cross_sign import controls, arm_matrix


class RowSignTest(unittest.TestCase):
  def test_scales(self):
    local=jnp.array([1.,2.,3.]); pos=jnp.array([2.,-1.,.5]); neg=jnp.array([-.7,.2,1.])
    total=local+pos+neg
    for scales,expected in [((1,1),total),((0,0),local),((.5,.5),local+.5*(pos+neg)),
                            ((1.5,1.5),local+1.5*(pos+neg)),((0,1),local+neg),((1,0),local+pos)]:
      np.testing.assert_allclose(_scale_row_cross(total,local,pos,neg,jnp.array(scales)),expected,atol=1e-6)
    b=total.astype(jnp.bfloat16)
    np.testing.assert_array_equal(_scale_row_cross(b,local,pos,neg,jnp.ones(2)),b)

  def test_layer_controls(self):
    names,scales=arm_matrix([11])
    p={'params':{'decoder':{f'layers_{i}':{'self_attention':{'abs_v_cache_projection':jnp.zeros((32,8))}} for i in range(24)}}}
    x=controls(p,jnp.asarray(scales[1]),False)
    for i in range(24):
      np.testing.assert_array_equal(x['decoder'][f'layers_{i}']['self_attention']['row_sign_scales'],0 if i==11 else 1)
    p={'params':{'decoder':{'layers':{'self_attention':{'abs_v_cache_projection':jnp.zeros((24,32,8))}}}}}
    x=controls(p,jnp.asarray(scales[1]),True)
    np.testing.assert_array_equal(x['decoder']['layers']['self_attention']['row_sign_scales'],scales[1])


if __name__=='__main__': unittest.main()
