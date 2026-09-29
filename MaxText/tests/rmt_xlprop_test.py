"""Focused XL RMT shape, initialization, layer-scan and gradient checks."""
import contextlib
import io
import math
from pathlib import Path
import tempfile
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
import pyconfig, max_utils
from layers import models, rmt

EXP = 'RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoO'

class XLPropTest(unittest.TestCase):
  def config(self, name=EXP, **kwargs):
    directory = tempfile.TemporaryDirectory()
    self.addCleanup(directory.cleanup)
    Path(directory.name, 'test').mkdir()
    with contextlib.redirect_stdout(io.StringIO()):
      return pyconfig.initialize([None, 'MaxText/configs/base.yml'], exp_class=name,
          run_name='test', enable_checkpointing=False, base_output_directory=directory.name+'/',
          jax_cache_dir='', log_config=False, dataset_type='synthetic',
          max_target_length=4, max_prefill_predict_length=4, query_chunk_size=2,
          per_device_batch_size=1., **kwargs)

  def model_args(self, cfg):
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
    model = models.Transformer(config=cfg, mesh=mesh, quant=None)
    args = dict(decoder_input_tokens=jnp.array([[1,2,3,4]],jnp.int32),
        decoder_positions=jnp.arange(4)[None],decoder_target_tokens=jnp.ones((1,4),jnp.int32),
        decoder_target_mask=jnp.ones((1,4),jnp.float32),decoder_segment_ids=jnp.ones((1,4),jnp.int32),
        enable_dropout=False)
    return model,args

  def test_full_parameter_trees(self):
    for name, count in [(EXP,1432430680),
        ('RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoO',431773472)]:
      cfg=self.config(name)
      model,args=self.model_args(cfg)
      with contextlib.redirect_stdout(io.StringIO()):
        tree=jax.eval_shape(lambda k:model.init(k,**args)['params'],jax.random.key(1))
      self.assertEqual(sum(math.prod(p.shape) for p in jax.tree.leaves(tree)),count)
      if name==EXP:
        dec=nn.unbox(tree)['decoder']
        self.assertEqual(dec['dynamic_embedding_write']['address_down'].shape,(1920,384))
        self.assertEqual(dec['dynamic_embedding_write']['address_up'].shape,(384,1200))
        self.assertEqual(dec['dynamic_unembedding_read']['key_kernel'].shape,(1920,800))
        self.assertNotIn('compression',dec['dynamic_unembedding_read'])

  def test_scanned_forward_gradient_and_health(self):
    cfg=self.config(base_num_decoder_layers=2,base_emb_dim=640,head_dim=32,
                    base_mlp_dim=128,vocab_size=128)
    cfg.get_keys().update(dtype=jnp.float32,rmt_record_dynamic_health=True)
    model,args=self.model_args(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
      params=nn.unbox(model.init(jax.random.key(2),**args)['params'])
      def loss(p):
        output,aux=model.apply({'params':p},**args,mutable=['intermediates'])
        return jnp.sum(output[0]),aux
      (value,aux),gradient=jax.jit(jax.value_and_grad(loss,has_aux=True))(params)
    self.assertTrue(np.isfinite(value))
    self.assertTrue(all(np.isfinite(v).all() for v in jax.tree.leaves(gradient)))
    self.assertGreater(float(jnp.linalg.norm(gradient['decoder']['dynamic_unembedding_read']['key_kernel'])),0.)
    self.assertGreater(float(jnp.linalg.norm(gradient['decoder']['dynamic_embedding_write']['address_up'])),0.)
    health=aux['intermediates']['decoder']['layers']['rmt_dynamic_health'][0]
    names=rmt.dynamic_health_names(cfg.get_keys().get('rmt_record_write_health',True),20,60)
    self.assertEqual(health.shape,(2,len(names)))
    self.assertIn('attn_input_first20_rms',names)
    self.assertIn('mlp_input_tail40_rms',names)
    self.assertTrue(np.isfinite(health).all())
    np.testing.assert_array_equal(health[:,names.index('o_dynamic_rms')],0.)

if __name__=='__main__': unittest.main()
