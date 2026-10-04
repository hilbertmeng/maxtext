"""Terminal raw-M read followed by vector pre-vocabulary normalization."""
import contextlib, io, math, unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from tests import rmt_xlprop_test
from layers import rmt
BASE='RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNormSharedRawWrite'
NEW=BASE+'FinalReadoutNorm'
class FinalReadoutNormTest(unittest.TestCase):
 config=rmt_xlprop_test.XLPropTest.config
 model_args=rmt_xlprop_test.XLPropTest.model_args
 def test_full_budget_and_health_defaults(self):
  cfg=self.config(NEW);model,args=self.model_args(cfg)
  with contextlib.redirect_stdout(io.StringIO()):p=nn.unbox(jax.eval_shape(lambda k:model.init(k,**args)['params'],jax.random.key(0)))
  count=sum(math.prod(x.shape) for x in jax.tree.leaves(p))
  self.assertEqual(count,431778272)
  self.assertNotIn('final_matrix_norm',p['decoder'])
  self.assertEqual(p['decoder']['final_readout_norm']['scale'].shape,(1200,))
  self.assertEqual(p['decoder']['unembedding_vector_norm']['scale'].shape,(1200,))
  for k in ['rmt_record_dynamic_health','rmt_record_stability_health','rmt_record_write_scale_health']:self.assertTrue(cfg.get_keys()[k])
  self.assertEqual(cfg.rmt_carry_health_layers,'all');self.assertEqual(cfg.keep_period,0);self.assertEqual(cfg.max_to_keep,2)
  self.assertFalse(cfg.rmt_matrix_read_norm!='none');self.assertEqual(cfg.mlp_dim,4100)
  print('FINAL_READOUT_FULL_BUDGET_OK',count)
 def test_scanned_gradient_and_real_head_inputs(self):
  cfg=self.config(NEW,base_emb_dim=512,head_dim=32,base_mlp_dim=96,base_num_decoder_layers=2,vocab_size=128,loss_chunk_size=2)
  cfg.get_keys()['dtype']=jnp.float32;model,args=self.model_args(cfg)
  with contextlib.redirect_stdout(io.StringIO()):p=nn.unbox(model.init(jax.random.key(4),**args)['params'])
  captured={};logits=[]
  def intercept(next_method,args,kwargs,context):
   if context.module.name=='lm_head' and context.method_name=='__call__':captured['head_input']=args[0]
   out=next_method(*args,**kwargs)
   if context.module.name=='final_readout_norm':captured['normalized_hidden']=out
   if context.module.name=='lm_head' and context.method_name=='project_logits':logits.append(out)
   return out
  with contextlib.redirect_stdout(io.StringIO()),nn.intercept_methods(intercept):
   _,aux=model.apply({'params':p},**args,mutable=['intermediates'])
  # Interceptor sees both the override and its super call; health sows once per real chunk.
  self.assertEqual(len(aux['intermediates']['decoder']['lm_head']['rmt_final_logits_health']),2)
  self.assertGreaterEqual(len(logits),2)
  np.testing.assert_array_equal(captured['head_input'],captured['normalized_hidden'])
  # Tiny zero-initialized routes have low initial RMS, so epsilon is visible.
  np.testing.assert_allclose(jnp.sqrt(jnp.mean(captured['head_input']**2,axis=-1)),1.,atol=.005)
  from train import record_rmt_dynamic_health_metrics
  metrics={'scalar':{}};record_rmt_dynamic_health_metrics(metrics,aux,cfg);m=metrics['scalar']
  self.assertAlmostEqual(float(m['rmt/final_readout/matrix_raw_rms']),float(m['rmt/final_readout/matrix_read_rms']),places=6)
  self.assertAlmostEqual(float(m['rmt/final_readout/logits_rms']),float(jnp.sqrt(jnp.mean(jnp.concatenate(logits,axis=1)**2))),places=6)
  self.assertIn('rmt/carry/layer_001/output_raw_rms',m)
  self.assertIn('rmt/carry/layer_001/token_mean_energy_fraction',m)
  with contextlib.redirect_stdout(io.StringIO()):v,g=jax.jit(jax.value_and_grad(lambda p:jnp.mean(model.apply({'params':p},**args)[0])))(p)
  self.assertTrue(all(np.isfinite(a).all() for a in jax.tree.leaves((v,g))))
  for node,key in [('final_readout_norm','scale'),('dynamic_unembedding_read','key_kernel')]:self.assertGreater(float(jnp.linalg.norm(g['decoder'][node][key])),0.)
  # Disabled new policy retains the exact parent's parameter tree and forward.
  oldcfg=self.config(BASE,base_emb_dim=512,head_dim=32,base_mlp_dim=96,base_num_decoder_layers=2,vocab_size=128,loss_chunk_size=2);oldcfg.get_keys()['dtype']=jnp.float32
  offcfg=self.config(NEW,base_emb_dim=512,head_dim=32,base_mlp_dim=96,base_num_decoder_layers=2,vocab_size=128,loss_chunk_size=2);offcfg.get_keys().update(dtype=jnp.float32,rmt_final_readout_norm=False)
  with contextlib.redirect_stdout(io.StringIO()):
   old,_=self.model_args(oldcfg);off,_=self.model_args(offcfg)
   oldp=nn.unbox(old.init(jax.random.key(5),**args)['params']);offp=nn.unbox(off.init(jax.random.key(5),**args)['params'])
   for a,b in zip(jax.tree.leaves(oldp),jax.tree.leaves(offp)):np.testing.assert_array_equal(a,b)
   oldout=old.apply({'params':oldp},**args);offout=off.apply({'params':offp},**args)
  for a,b in zip(jax.tree.leaves(oldout),jax.tree.leaves(offout)):np.testing.assert_array_equal(a,b)
  print('FINAL_READOUT_REAL_INPUT_GRAD_PARENT_PARITY_OK',float(v))
if __name__=='__main__':unittest.main()
