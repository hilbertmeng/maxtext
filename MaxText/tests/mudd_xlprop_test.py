"""Mudd must retain every block in its history and gradient path."""
import contextlib, io, math, tempfile, unittest
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
import pyconfig,max_utils
from layers import models

class MuddXLPropTest(unittest.TestCase):
  def cfg(self, **kwargs):
    tmp=tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
    Path(tmp.name,'test').mkdir()
    with contextlib.redirect_stdout(io.StringIO()):
      return pyconfig.initialize([None,'MaxText/configs/base.yml'],exp_class='MuddLlama2XLProp',
        run_name='test',enable_checkpointing=False,base_output_directory=tmp.name+'/',jax_cache_dir='',
        log_config=False,dataset_type='synthetic',max_target_length=4,max_prefill_predict_length=4,
        query_chunk_size=2,per_device_batch_size=1.,attention='dot_product_chunk',**kwargs)
  def model(self,cfg):
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes)
    model=models.Transformer(config=cfg,mesh=mesh,quant=None)
    args=dict(decoder_input_tokens=jnp.array([[1,2,3,4]],jnp.int32),decoder_positions=jnp.arange(4)[None],
      decoder_target_tokens=jnp.ones((1,4),jnp.int32),decoder_target_mask=jnp.ones((1,4),jnp.float32),
      decoder_segment_ids=jnp.ones((1,4),jnp.int32),enable_dropout=False)
    return model,args
  def check_history(self,params,layers):
    dec=nn.unbox(params)['decoder']
    for i in range(1,layers):
      self.assertEqual(dec[f'layers_{i}']['compose']['mlp']['dynamic_dense_conn2']['kernel'].shape[-2:],(4,i+1))
      self.assertNotIn('compose_break',dec[f'layers_{i}'])
    self.assertEqual(dec['compose_final']['mlp']['dynamic_dense_conn2']['kernel'].shape[-2:],(1,layers+1))
  def test_full_shape_audit(self):
    cfg=self.cfg();model,args=self.model(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
      shapes=jax.eval_shape(lambda k:model.init(k,**args)['params'],jax.random.key(1))
    self.check_history(shapes,28)
    count=sum(math.prod(v.shape) for v in jax.tree.leaves(shapes))
    print('MUDD_FULL_PARAMS',count,flush=True)
    self.assertEqual(count,1438369841)
    widths=[round(5120*(i/27+.5)/128)*128 for i in range(28)]
    self.assertEqual(sum(widths),28*5120)
  def test_every_block_has_gradient_at_initialization(self):
    cfg=self.cfg(base_num_decoder_layers=3,base_emb_dim=64,base_num_query_heads=2,
      base_num_kv_heads=2,head_dim=32,base_mlp_dim=256,vocab_size=128)
    model,args=self.model(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
      params=nn.unbox(model.init(jax.random.key(2),**args)['params'])
      self.check_history(params,3)
      loss,grad=jax.jit(jax.value_and_grad(lambda p:jnp.sum(model.apply({'params':p},**args)[0])))(params)
    self.assertTrue(np.isfinite(loss))
    self.assertTrue(all(np.isfinite(v).all() for v in jax.tree.leaves(grad)))
    for i in range(3):
      g=grad['decoder'][f'layers_{i}']['block']['mlp']
      self.assertGreater(sum(float(jnp.linalg.norm(v)) for v in jax.tree.leaves(g)),0.)
    # Repeated calls must not accumulate list entries across remat traces.
    second=model.apply({'params':params},**args)[0]
    np.testing.assert_allclose(jnp.sum(second),loss,rtol=2e-3)
if __name__=='__main__':unittest.main()
