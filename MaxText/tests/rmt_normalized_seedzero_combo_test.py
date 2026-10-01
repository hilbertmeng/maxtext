"""Focused combination scope, full-tree budget and scanned gradient checks."""
import contextlib, io, math, unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from tests.rmt_xlprop_test import XLPropTest
from layers import rmt

MEDIUM = "RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedWriteNormEmbedSeedZero"
XL = "RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOMPreNormLearnedScaleSharedWriteEmbedNormSeedKeyZeroInit"

class NormalizedSeedZeroComboTest(unittest.TestCase):
  config = XLPropTest.config
  model_args = XLPropTest.model_args

  def test_target_budgets_and_scope(self):
    for name, count, heads, rows, width, keep, recent in (
        (MEDIUM,431888672,16,48,4100,400,8),
        (XL,1432453720,20,60,6643,1000,8)):
      cfg = self.config(name)
      for flag in ("rmt_matrix_read_learned_scale", "rmt_static_write_content_norm",
                   "rmt_layer_write_content_norm", "rmt_embedding_shared_content",
                   "rmt_embedding_content_norm", "rmt_embedding_shared_write_norm",
                   "rmt_embedding_seed_key_zero_init", "rmt_vector_pre_norm",
                   "rmt_record_dynamic_health"):
        self.assertTrue(cfg.get_keys().get(flag, False), (name,flag))
      self.assertEqual(cfg.rmt_matrix_read_norm,"all")
      self.assertEqual(cfg.rmt_embedding_static_write_scale,1.)
      self.assertFalse(cfg.rmt_dynamic_o_enabled)
      self.assertTrue(cfg.scan_layers)
      self.assertFalse(cfg.rmt_block_scan)
      self.assertEqual(cfg.DATASET_VARIANT,"truepile4096")
      self.assertEqual(cfg.mlp_dim,width)
      self.assertEqual(cfg.keep_period,keep)
      self.assertEqual(cfg.max_to_keep,recent)
      for flag in ("rmt_pallas_write", "rmt_fused_attention_read", "rmt_fused_write_mlp_read",
                   "rmt_attn_write_key_zero_init", "rmt_mlp_write_key_zero_init"):
        self.assertFalse(cfg.get_keys().get(flag,False), (name,flag))
      model,args = self.model_args(cfg)
      with contextlib.redirect_stdout(io.StringIO()):
        p = nn.unbox(jax.eval_shape(lambda k:model.init(k,**args)["params"],jax.random.key(1)))
      self.assertEqual(sum(math.prod(v.shape) for v in jax.tree.leaves(p)),count)
      self.assertEqual(p["decoder"]["seed_key"].shape,(heads,rows))
      self.assertNotIn("embedding_write_content",p["decoder"])

  def test_scanned_initialization_gradients_and_health(self):
    for name, dim in ((MEDIUM,512),(XL,640)):
      cfg=self.config(name,base_num_decoder_layers=2,base_emb_dim=dim,
                      head_dim=32,base_mlp_dim=128,vocab_size=128)
      cfg.get_keys()["dtype"]=jnp.float32
      model,args=self.model_args(cfg)
      with contextlib.redirect_stdout(io.StringIO()):
        p=nn.unbox(model.init(jax.random.key(2),**args)["params"])
        np.testing.assert_array_equal(p["decoder"]["seed_key"],0.)
        for arm in ("attn","mlp"):
          self.assertGreater(float(jnp.linalg.norm(p["decoder"]["layers"][arm+"_write_key"])),0.)
          np.testing.assert_array_equal(p["decoder"]["layers"][arm+"_norm"]["scale"],1.)
        def loss(params):
          out,aux=model.apply({"params":params},**args,mutable=["intermediates"])
          return jnp.sum(out[0]),aux
        (value,aux),g=jax.jit(jax.value_and_grad(loss,has_aux=True))(p)
      self.assertTrue(all(np.isfinite(a).all() for a in jax.tree.leaves((value,aux,g))))
      self.assertGreater(float(jnp.linalg.norm(g["decoder"]["seed_key"])),0.)
      for arm in ("attn","mlp"):
        self.assertGreater(float(jnp.linalg.norm(g["decoder"]["layers"][arm+"_norm"]["scale"])),0.)
        self.assertGreater(float(jnp.linalg.norm(g["decoder"]["layers"][arm+"_write_key"])),0.)
      self.assertEqual(float(aux["intermediates"]["decoder"]["rmt_embedding_health"][0][1]),0.)
      self.assertIn("rmt_dynamic_health",aux["intermediates"]["decoder"]["layers"])

if __name__ == "__main__":unittest.main()
