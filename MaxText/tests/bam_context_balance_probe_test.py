"""Check the observer preserves a Flax forward and measures both paths directly."""
import importlib.util
from pathlib import Path
import unittest
from unittest import mock
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from layers import attentions
spec=importlib.util.spec_from_file_location('context_probe',Path(__file__).resolve().parents[2]/'experiments/bam_llama2_medium/h_context_balance_probe.py');probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
class FakeAttention(nn.Module):
  @nn.compact
  def __call__(self,x):
    context,_=attentions._attention_op(x);local=self._regularize_local_o_heads(x,True)
    return self.out_projection(3,context+local)
  def _regularize_local_o_heads(self,x,deterministic):return x*3
  def out_projection(self,output_dim,x):return x*5+.2
class ContextBalanceTest(unittest.TestCase):
  def test_observer_does_not_change_output(self):
    x=jnp.arange(12,dtype=jnp.float32).reshape(1,2,2,3)/10
    with mock.patch.object(attentions,'BamAttention',FakeAttention),mock.patch.object(attentions,'_attention_op',lambda x:(x*2,None)):
      model=FakeAttention();params=model.init(jax.random.PRNGKey(0),x);expected=model.apply(params,x)
      with probe.capture_context():actual,collections=model.apply(params,x,mutable=['intermediates'])
      np.testing.assert_array_equal(actual,expected)
      stats=np.asarray(collections['intermediates']['context_balance_probe'][0]);m=float(jnp.mean(x*x));np.testing.assert_allclose(stats,[[4*m,9*m,25*m,6*m]],rtol=1e-6)
if __name__=='__main__':unittest.main()
