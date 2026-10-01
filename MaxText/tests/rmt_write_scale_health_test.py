"""Forward-only write magnitudes must not alter resumed training math or parameters."""
import contextlib
import io
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from layers import rmt
from tests.rmt_xlprop_test import XLPropTest

EXP = 'RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOMPreNormLearnedScaleSharedWriteEmbedNormSeedKeyZeroInit'

class WriteScaleHealthTest(unittest.TestCase):
  config = XLPropTest.config
  model_args = XLPropTest.model_args

  def test_actual_update_includes_static_dynamic_cancellation(self):
    raw=jnp.array([4.,-4.]);static=jnp.ones((1,2,3,4))*2
    carry=jnp.ones_like(static)*4
    values=rmt._write_scale_health(raw,-static,static,carry)
    np.testing.assert_allclose(values,[4.,2.,2.,4.,0.],atol=1e-7)
    values=rmt._write_scale_health(raw,static,static,carry)
    np.testing.assert_allclose(values,[4.,2.,2.,4.,1.],atol=1e-7)
    self.assertTrue(all(np.isfinite(x) for x in rmt._write_scale_health(raw,static,static,jnp.zeros_like(carry))))

  def test_scanned_resume_equivalence_and_metric_export(self):
    cfg=self.config(EXP,base_num_decoder_layers=2,base_emb_dim=640,head_dim=32,
                    base_mlp_dim=128,vocab_size=128)
    cfg.get_keys().update(dtype=jnp.float32,rmt_record_dynamic_health=True,
                         rmt_record_write_scale_health=False)
    model,args=self.model_args(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
      params=nn.unbox(model.init(jax.random.key(2),**args)['params'])
      def loss(p):
        out,aux=model.apply({'params':p},**args,mutable=['intermediates'])
        return jnp.sum(out[0]),(out[0],aux)
      (old_loss,(old_output,old_aux)),old_grad=jax.jit(jax.value_and_grad(loss,has_aux=True))(params)
      cfg.get_keys()['rmt_record_write_scale_health']=True
      tree=nn.unbox(model.init(jax.random.key(2),**args)['params'])
      (new_loss,(new_output,new_aux)),new_grad=jax.jit(jax.value_and_grad(loss,has_aux=True))(params)
    for old,new in zip(jax.tree.leaves(params),jax.tree.leaves(tree)):
      np.testing.assert_array_equal(old,new)
    np.testing.assert_array_equal(old_loss,new_loss)
    np.testing.assert_array_equal(old_output,new_output)
    for old,new in zip(jax.tree.leaves(old_grad),jax.tree.leaves(new_grad)):
      np.testing.assert_array_equal(old,new)
    old_health=old_aux['intermediates']['decoder']['layers']['rmt_dynamic_health'][0]
    health=new_aux['intermediates']['decoder']['layers']['rmt_dynamic_health'][0]
    np.testing.assert_array_equal(health[:,:old_health.shape[-1]],old_health)
    names=rmt.dynamic_health_names(True,20,60,True)
    self.assertEqual(health.shape,(2,len(names)))
    self.assertEqual(len(names)-old_health.shape[-1],10)
    self.assertTrue(np.isfinite(health).all())
    from train import record_rmt_dynamic_health_metrics
    metrics={'scalar':{}}
    record_rmt_dynamic_health_metrics(metrics,new_aux,cfg)
    for arm in ['attn','mlp']:
      for stat in ['output_raw_rms','write_static_rms','write_dynamic_rms','write_carry_rms','write_delta_over_carry']:
        tag=f'rmt/dynamic/layer_001/{arm}_{stat}'
        self.assertIn(tag,metrics['scalar'])
        self.assertGreater(float(metrics['scalar'][tag]),0.)
    cfg.get_keys()['rmt_record_write_health']=False
    with contextlib.redirect_stdout(io.StringIO()):
      _,aux=model.apply({'params':params},**args,mutable=['intermediates'])
    metrics={'scalar':{}}
    record_rmt_dynamic_health_metrics(metrics,aux,cfg)
    self.assertIn('rmt/dynamic/layer_001/mlp_write_static_rms',metrics['scalar'])
    self.assertNotIn('rmt/dynamic/layer_001/mlp_write_tail40_ratio',metrics['scalar'])

if __name__=='__main__':unittest.main(defaultTest='WriteScaleHealthTest')
