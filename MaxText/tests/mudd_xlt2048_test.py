"""Historical XL Mudd: actual full shapes, complete history, consumed gradients."""
import contextlib,io,math,unittest
import jax,jax.numpy as jnp,numpy as np
from flax import linen as nn
from tests.mudd_xlprop_test import MuddXLPropTest

class MuddXLT2048Test(unittest.TestCase):
  cfg=MuddXLPropTest.cfg
  model=MuddXLPropTest.model
  check_history=MuddXLPropTest.check_history

  def test_full_budget_and_recipe(self):
    c=self.cfg(exp_class='MuddLlama2XLT2048Head16x128')
    self.assertEqual((c.emb_dim,c.num_decoder_layers,c.num_query_heads,c.head_dim,c.mlp_dim),(2048,24,16,128,5504))
    self.assertEqual((c.steps,c.learning_rate_schedule_steps,c.learning_rate),(50000,50000,2e-4))
    self.assertEqual(c.DATASET_VARIANT,'legacy2048')
    self.assertEqual(c.wd_mults,[])
    self.assertFalse(c.scan_layers)
    self.assertFalse(c.float32_logits)
    model,args=self.model(c)
    with contextlib.redirect_stdout(io.StringIO()):
      shapes=jax.eval_shape(lambda k:model.init(k,**args)['params'],jax.random.key(1))
    self.check_history(shapes,24)
    dec=nn.unbox(shapes)['decoder']
    widths=[round(5504*(i/23+.5)/128)*128 for i in range(24)]
    for i,w in enumerate(widths):
      self.assertEqual(dec[f'layers_{i}']['block']['mlp']['wi_0']['kernel'].shape,(2048,w))
    count=sum(math.prod(v.shape) for v in jax.tree.leaves(shapes))
    print('OLD_XL_MUDD_PARAMS',count,'MHA_DELTA',count-1420920832,'WIDTHS',widths,flush=True)
    self.assertGreater(count,1420920832)
    self.assertLess(count-1420920832,1420920832*.015)

  def test_every_history_block_has_finite_gradient(self):
    c=self.cfg(exp_class='MuddLlama2XLT2048Head16x128',base_num_decoder_layers=3,base_emb_dim=64,
      base_num_query_heads=2,base_num_kv_heads=2,head_dim=32,base_mlp_dim=256,vocab_size=128)
    model,args=self.model(c)
    with contextlib.redirect_stdout(io.StringIO()):
      params=nn.unbox(model.init(jax.random.key(2),**args)['params'])
      self.check_history(params,3)
      loss,grad=jax.jit(jax.value_and_grad(lambda p:jnp.mean(model.apply({'params':p},**args)[0]**2)))(params)
    self.assertTrue(np.isfinite(float(loss)))
    self.assertTrue(all(np.isfinite(v).all() for v in jax.tree.leaves(grad)))
    for i in range(3):
      self.assertGreater(sum(float(jnp.linalg.norm(v)) for v in jax.tree.leaves(grad['decoder'][f'layers_{i}']['block']['mlp'])),0.)

if __name__=='__main__':unittest.main()
