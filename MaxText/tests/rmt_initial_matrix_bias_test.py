"""Initial constant M bias budget, zero-init parity and gradient."""
import contextlib,io,math,unittest,copy
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from tests.rmt_xlprop_test import XLPropTest
NEW='RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroWriteContentBiasInitialMatrixBias'
class InitialMatrixBiasTest(unittest.TestCase):
 config=XLPropTest.config
 model_args=XLPropTest.model_args
 def test_budget(self):
  cfg=self.config(NEW);m,args=self.model_args(cfg)
  with contextlib.redirect_stdout(io.StringIO()):p=nn.unbox(jax.eval_shape(lambda k:m.init(k,**args)['params'],jax.random.key(0)))
  n=sum(math.prod(x.shape) for x in jax.tree.leaves(p));self.assertEqual(n,431759072+43200+3600);print('PARAMS',n)
  self.assertEqual(p['decoder']['initial_matrix_bias'].shape,(48,75))
 def test_parity_gradient_health(self):
  cfg=self.config(NEW,base_emb_dim=512,head_dim=32,base_mlp_dim=96,base_num_decoder_layers=2,vocab_size=128);m,args=self.model_args(cfg)
  with contextlib.redirect_stdout(io.StringIO()):p=nn.unbox(m.init(jax.random.key(4),**args)['params'])
  np.testing.assert_array_equal(p['decoder']['initial_matrix_bias'],0)
  def loss(p):
   y,aux=m.apply({'params':p},**args,mutable=['intermediates']);return jnp.mean(y[0]),aux
  with contextlib.redirect_stdout(io.StringIO()):(v,aux),g=jax.jit(jax.value_and_grad(loss,has_aux=True))(p)
  self.assertTrue(all(np.isfinite(x).all() for x in jax.tree.leaves((v,aux,g))))
  self.assertGreater(float(jnp.linalg.norm(g['decoder']['initial_matrix_bias'])),0)
  from train import record_rmt_dynamic_health_metrics
  metrics={'scalar':{}};record_rmt_dynamic_health_metrics(metrics,aux,cfg)
  self.assertEqual(float(metrics['scalar']['rmt/initial_matrix/bias_rms']),0)
  self.assertIn('rmt/carry/layer_001/token_mean_energy_fraction',metrics['scalar'])
  with contextlib.redirect_stdout(io.StringIO()):newv=loss(p)[0]
  cfg.get_keys()['rmt_initial_matrix_bias']=False;old,args=self.model_args(cfg);oldp=copy.deepcopy(p);del oldp['decoder']['initial_matrix_bias']
  with contextlib.redirect_stdout(io.StringIO()):oldv=jnp.mean(old.apply({'params':oldp},**args)[0])
  np.testing.assert_array_equal(newv,oldv)
if __name__=='__main__':unittest.main()
