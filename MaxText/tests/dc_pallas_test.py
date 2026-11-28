"""End-to-end forward & backward numerical equivalence test for Unified (Ungrouped & Grouped) Flash-DCMHA Pallas."""

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

  def _check_kernel_fp32_parity(self, num_groups: int, rank: int):
    key = jax.random.PRNGKey(7 + num_groups * 13 + rank)
    keys = jax.random.split(key, 11)
    b, t, h, d = 2, 512, 16, 64
    m = h // num_groups
    q = jax.random.normal(keys[0], (b, t, h, d), dtype=jnp.float32) * (d ** -0.5)
    k = jax.random.normal(keys[1], (b, t, h, d), dtype=jnp.float32)
    v = jax.random.normal(keys[2], (b, t, h, d), dtype=jnp.float32)
    pre_args = (
        jax.random.normal(keys[3], (b, t, num_groups, rank, m), dtype=jnp.float32) * 0.2,
        jax.random.normal(keys[4], (b, t, num_groups, rank, m), dtype=jnp.float32) * 0.2,
        jax.random.normal(keys[5], (b, t, num_groups, m), dtype=jnp.float32) * 0.1,
    )
    post_args = (
        jax.random.normal(keys[6], (b, t, num_groups, rank, m), dtype=jnp.float32) * 0.2,
        jax.random.normal(keys[7], (b, t, num_groups, rank, m), dtype=jnp.float32) * 0.2,
        jax.random.normal(keys[8], (b, t, num_groups, m), dtype=jnp.float32) * 0.1,
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
      dc_grouped_block_q = 256
      dc_grouped_block_kv = 256
      dc_grouped_block_kv_compute = 128
      dc_grouped_bwd_block_q = 256
      dc_grouped_bwd_block_kv = 256
      dc_grouped_bwd_block_kv_compute = 128
      dc_vmem_mib = 96

    for seg_ids in (None, seg):
      def ref_fn(q_, k_, v_, pre_, post_):
        out = dc_pallas._reference_dc2_attention(
            q_, k_, v_, pre_, post_, seg_ids=seg_ids, dot_prec=lax.Precision.HIGHEST
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

  def test_ungrouped_g1_r2_fp32_and_segment_parity(self):
    self._check_kernel_fp32_parity(num_groups=1, rank=2)

  def test_grouped_g4_r1_fp32_and_segment_parity(self):
    self._check_kernel_fp32_parity(num_groups=4, rank=1)

  def _check_transformer_matches_qchunk(self, pallas_cls: str, qchunk_cls: str, tag: str):
    out_dir = f"/tmp/dc_pallas_unit_test_{tag}"
    os.makedirs(f"{out_dir}/dc_pallas_unit_test", exist_ok=True)
    self.addCleanup(lambda: shutil.rmtree(out_dir, ignore_errors=True))
    common_args = [
        "train.py",
        "configs/base.yml",
        "run_name=dc_pallas_unit_test",
        f"base_output_directory={out_dir}",
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
        "dc_grouped_block_q=256",
        "dc_grouped_block_kv=256",
        "dc_grouped_block_kv_compute=128",
        "dc_grouped_bwd_block_q=256",
        "dc_grouped_bwd_block_kv=256",
        "dc_grouped_bwd_block_kv_compute=128",
    ]
    cfg_pal = pyconfig.initialize(common_args + [f"exp_class={pallas_cls}"])
    cfg_ref = pyconfig.initialize(common_args + [f"exp_class={qchunk_cls}"])

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

  def test_transformer_ungrouped_g1_r2_matches_qchunk(self):
    self._check_transformer_matches_qchunk(
        "DC2GlobalLlama2MediumT4096", "DC2GlobalLlama2MediumT4096QChunk", "g1r2"
    )

  def test_transformer_grouped_g4_r1_matches_qchunk(self):
    self._check_transformer_matches_qchunk(
        "DC2G4R1GlobalLlama2MediumT4096", "DC2G4R1GlobalLlama2MediumT4096QChunk", "g4r1"
    )


if __name__ == "__main__":
  unittest.main()
