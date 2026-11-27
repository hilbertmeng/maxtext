"""End-to-end forward & backward numerical equivalence test for Two-Pass Global Flash-DCMHA Pallas."""

import os
import shutil
import unittest
import jax
from jax import lax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np

import max_utils
import pyconfig
from layers import dc_pallas
from layers import models


class DcPallasTest(unittest.TestCase):

  def test_dc2_flash_attention_fp32_and_segment_parity(self):
    key = jax.random.PRNGKey(7)
    keys = jax.random.split(key, 11)
    b, t, h, d = 2, 512, 16, 64
    q = jax.random.normal(keys[0], (b, t, h, d), dtype=jnp.float32) * (d ** -0.5)
    k = jax.random.normal(keys[1], (b, t, h, d), dtype=jnp.float32)
    v = jax.random.normal(keys[2], (b, t, h, d), dtype=jnp.float32)
    pre_args = (
        jax.random.normal(keys[3], (b, t, 1, 2, h), dtype=jnp.float32) * 0.2,
        jax.random.normal(keys[4], (b, t, 1, 2, h), dtype=jnp.float32) * 0.2,
        jax.random.normal(keys[5], (b, t, 1, h), dtype=jnp.float32) * 0.1,
    )
    post_args = (
        jax.random.normal(keys[6], (b, t, 1, 2, h), dtype=jnp.float32) * 0.2,
        jax.random.normal(keys[7], (b, t, 1, 2, h), dtype=jnp.float32) * 0.2,
        jax.random.normal(keys[8], (b, t, 1, h), dtype=jnp.float32) * 0.1,
    )
    cot = jax.random.normal(keys[9], (b, t, h, d), dtype=jnp.float32)
    seg = jnp.concatenate([jnp.ones((b, t // 2), jnp.int32), 2 * jnp.ones((b, t // 2), jnp.int32)], axis=1)

    class _Cfg:
      dc_block_q = 128
      dc_block_kv = 256
      dc_block_kv_compute = 128
      dc_bwd_block_q = 128
      dc_bwd_block_kv = 256
      dc_bwd_block_kv_compute = 128
      dc_vmem_mib = 96

    for seg_ids in (None, seg):
      def ref_fn(q_, k_, v_, pre_, post_):
        out = dc_pallas._reference_dc2_attention(
            q_, k_, v_,
            dc_pallas.pack_dc2_weights(*pre_),
            dc_pallas.pack_dc2_weights(*post_),
            seg_ids=seg_ids,
            dot_prec=lax.Precision.HIGHEST,
        )
        return jnp.sum(out * cot), out

      def pal_fn(q_, k_, v_, pre_, post_):
        out = dc_pallas.dc2_flash_attention(
            q_, k_, v_, pre_, post_,
            decoder_segment_ids=seg_ids,
            config=_Cfg,
            mesh=None,
            dot_prec=lax.Precision.HIGHEST,
        )
        return jnp.sum(out * cot), out

      (_, out_ref), grads_ref = jax.jit(jax.value_and_grad(ref_fn, argnums=(0, 1, 2, 3, 4), has_aux=True))(
          q, k, v, pre_args, post_args
      )
      (_, out_pal), grads_pal = jax.jit(jax.value_and_grad(pal_fn, argnums=(0, 1, 2, 3, 4), has_aux=True))(
          q, k, v, pre_args, post_args
      )
      np.testing.assert_allclose(out_pal, out_ref, atol=2e-5, rtol=2e-5)
      for ga, gb in zip(jax.tree_util.tree_leaves(grads_pal), jax.tree_util.tree_leaves(grads_ref)):
        np.testing.assert_allclose(ga, gb, atol=5e-5, rtol=5e-5)

  def test_transformer_dc_flash_matches_qchunk(self):
    os.makedirs("/tmp/dc_pallas_unit_test/dc_pallas_unit_test", exist_ok=True)
    self.addCleanup(lambda: shutil.rmtree("/tmp/dc_pallas_unit_test", ignore_errors=True))
    common_args = [
        "train.py",
        "configs/base.yml",
        "run_name=dc_pallas_unit_test",
        "base_output_directory=/tmp/dc_pallas_unit_test",
        "dataset_type=synthetic",
        "enable_checkpointing=False",
        "float32_qk_product=True",
        "float32_logits=True",
        "per_device_batch_size=1",
        "base_num_decoder_layers=2",
        "max_prefill_predict_length=256",
        "max_target_length=512",
        "query_chunk_size=256",
        "dc_block_q=128",
        "dc_block_kv=256",
        "dc_block_kv_compute=128",
        "dc_bwd_block_q=128",
        "dc_bwd_block_kv=256",
        "dc_bwd_block_kv_compute=128",
    ]
    cfg_pal = pyconfig.initialize(common_args + ["exp_class=DC2GlobalLlama2MediumT4096"])
    cfg_ref = pyconfig.initialize(common_args + ["exp_class=DC2GlobalLlama2MediumT4096QChunk"])

    mesh = Mesh(max_utils.create_device_mesh(cfg_pal), cfg_pal.mesh_axes)
    m_pal = models.Transformer(cfg_pal, mesh, quant=None)
    m_ref = models.Transformer(cfg_ref, mesh, quant=None)

    rng = jax.random.PRNGKey(123)
    x = jax.random.randint(rng, (1, 512), 0, 1000)
    pos = jnp.broadcast_to(jnp.arange(512)[None, :], (1, 512))
    seg = jnp.ones((1, 512), dtype=jnp.int32)
    vars_init = m_ref.init({"params": rng, "dropout": rng, "aqt": rng}, x, pos, x, seg, decoder_segment_ids=seg, enable_dropout=False)

    def run_model(m):
      def loss_f(params):
        out = m.apply({"params": params}, x, pos, x, seg, decoder_segment_ids=seg, enable_dropout=False)
        logits = out[0] if isinstance(out, tuple) else out
        return jnp.mean(logits.astype(jnp.float32) ** 2), logits
      return jax.jit(jax.value_and_grad(loss_f, has_aux=True))(vars_init["params"])

    (loss_ref, logits_ref), grads_ref = run_model(m_ref)
    (loss_pal, logits_pal), grads_pal = run_model(m_pal)
    rel_logit_err = float(
        jnp.linalg.norm((logits_pal - logits_ref).astype(jnp.float32))
        / jnp.linalg.norm(logits_ref.astype(jnp.float32))
    )
    self.assertLess(rel_logit_err, 1e-2)
    self.assertAlmostEqual(float(loss_pal), float(loss_ref), places=1)
    for (p, gp), (_, gr) in zip(
        jax.tree_util.tree_leaves_with_path(grads_pal),
        jax.tree_util.tree_leaves_with_path(grads_ref),
    ):
      gn_ref = float(jnp.linalg.norm(gr.astype(jnp.float32)))
      if gn_ref > 1e-6:
        rel_g = float(jnp.linalg.norm((gp - gr).astype(jnp.float32))) / gn_ref
        self.assertLess(rel_g, 3.5e-2, f"Grad mismatch on {p}: {rel_g}")


if __name__ == "__main__":
  unittest.main()
