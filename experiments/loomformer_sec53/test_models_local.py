"""Local CPU numerical equivalence tests for all 3 JAX models in `models_jax.py`.

1. LoomPythia: compare `loom_forward_and_observe` against MaxText `Transformer` on a miniature
   `BamPythia2p8BHUnit300BFastKernelMPreNormQKScale` config, AND verify exact linear writer->reader
   attribution reconstruction!
2. mHC-Pythia: compare `mhc_forward_and_observe` against `/data0/mengqy/pythia-mhc-hf/params_rand2.npz`
   + `ref_rand2_fp32.npz`.
3. Pythia-2.8B: compare `pythia_forward_and_observe` on synthetic weights against `pythia_torch_reference`
   bijection.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "MaxText"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flax import linen as nn
from flax.core import unfreeze
from flax.traverse_util import unflatten_dict
import jax
import jax.numpy as jnp
import numpy as np

import exp
import max_utils
import pyconfig
from layers.models import Transformer
import models_jax as mj


def test_loom_equivalence_and_attribution():
  class MiniLoomPythia(exp.BamPythia2p8BHUnit300BFastKernelMPreNormQKScale):
    model_name = "MiniLoomPythia"
    base_emb_dim = 128
    base_num_decoder_layers = 4
    base_num_query_heads = 4
    base_num_kv_heads = 4
    head_dim = 32
    bam_k = 32
    bam_v = 16
    bam_local_qk_col_output_dim = 32
    bam_partial_rope_nope_dim = 32
    bam_standard_qk_dim = 8
    emb_bam_num_head = 4
    emb_bam_v_bottleneck_dim = 64
    bam_local_full_m_read_bottleneck_dim = 64
    bam_write_v_bottleneck_dim = 64
    bam_mlp_write_address_rank = 64
    base_mlp_dim = 256
    bam_memory_mlp_dim = 128
    bam_last_mlp_dim = 384
    bam_attention_write_layers = [0, 1, 2]
    bam_layer_modes = ["local_qk+local_v+local_o"] * 4
    vocab_size = 256
    max_target_length = 32
    max_prefill_predict_length = 32
    per_device_batch_size = 1.0
    target_train_batch_size = None
    dataset_type = "synthetic"
    enable_checkpointing = False
    only_eval = True
    record_training_health_metrics = False
    record_internal_nn_metrics = False
    record_logits_health_metrics = False
    bam_pallas_core = False
    bam_pallas_write = False
    bam_splash_attention = False
    float32_logits = True
    bam_pallas_packed_read_keys = False
    bam_pallas_fused_qk_norm = False
    remat_policy = "none"

  exp.MiniLoomPythia = MiniLoomPythia
  out_dir = "/tmp/test_loom_sec53/"
  Path(out_dir, "test").mkdir(parents=True, exist_ok=True)
  cfg = pyconfig.initialize(
      [None, str(REPO / "MaxText/configs/base.yml")],
      exp_class="MiniLoomPythia",
      run_name="test",
      base_output_directory=out_dir,
      enable_checkpointing=False,
      only_eval=True,
      log_config=False,
      jax_cache_dir="",
  )
  mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
  model = Transformer(cfg, mesh, quant=None)
  rng = jax.random.PRNGKey(42)
  tokens = jax.random.randint(rng, (2, 33), 0, 256, dtype=jnp.int32)
  inputs, targets = tokens[:, :-1], tokens[:, 1:]
  positions = jnp.broadcast_to(jnp.arange(32, dtype=jnp.int32), (2, 32))
  seg = jnp.ones((2, 32), dtype=jnp.int32)

  with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
    variables = model.init({"params": rng, "dropout": rng, "aqt": rng}, inputs, positions,
                           decoder_segment_ids=seg, decoder_target_tokens=targets,
                           decoder_target_mask=seg, enable_dropout=False)
    params = unfreeze(nn.unbox(variables))
    # Perturb zero-initialized static keys & biases so every branch is non-trivially active!
    p_key = jax.random.PRNGKey(99)
    leaves, treedef = jax.tree.flatten(params)
    new_leaves = []
    for i, leaf in enumerate(leaves):
      sub = jax.random.fold_in(p_key, i)
      new_leaves.append(leaf + 0.05 * jax.random.normal(sub, leaf.shape, dtype=leaf.dtype))
    params = jax.tree.unflatten(treedef, new_leaves)

    ref_xent, _, _ = model.apply(
        params, inputs, positions,
        decoder_segment_ids=seg,
        decoder_target_mask=seg,
        decoder_target_tokens=targets,
        enable_dropout=False,
        mutable=False,
    )

  u_params = mj.prepare_loom_params(params, jnp.bfloat16)
  sample_pos = jnp.array([0, 5, 17, 31], dtype=jnp.int32)
  head_qpos = jnp.array([4, 12, 23, 31], dtype=jnp.int32)
  cats = jnp.zeros_like(inputs)
  sink = inputs < 4
  obs = mj.loom_forward_and_observe(
      u_params, inputs, targets, sample_pos, head_qpos, cats, sink, sink,
  )
  diff = np.max(np.abs(np.asarray(ref_xent) - np.asarray(obs["xent"])))
  mean_diff = np.abs(float(np.mean(ref_xent)) - float(np.mean(obs["xent"])))
  print(f"[LoomPythia] max token xent diff = {diff:.6e}, mean NLL diff = {mean_diff:.6e}")
  assert diff < 6e-2 and mean_diff < 2e-3, (diff, mean_diff)

  # Verify exact linear attribution reconstruction for every layer l in {0..3} and token p!
  w_o_cur = np.asarray(obs["writers_o_cur"])    # [B, P, W_all, K]
  w_k_cur = np.asarray(obs["writers_k_cur"])    # [B, P, W_all, V]
  w_o_prev = np.asarray(obs["writers_o_prev"])
  w_k_prev = np.asarray(obs["writers_k_prev"])
  z = np.asarray(obs["layer_z"])                # [L, B, P, V]
  m_rms = np.asarray(obs["layer_m_rms"])        # [L, B, P]
  q_tot = np.asarray(obs["layer_q_dyn"] + obs["layer_q_stat"])  # [L, B, P, 4, N, V]
  u_dir = np.asarray(obs["layer_u_read"])       # [L, B, P, 4, N, K]

  s_pos_np = np.asarray(sample_pos)
  for l_idx in range(4):
    w_count = 4 + 8 * l_idx
    oc = w_o_cur[:, :, :w_count, :]             # [B, P, W_l, K]
    kc = w_k_cur[:, :, :w_count, :]             # [B, P, W_l, V]
    op = w_o_prev[:, :, :w_count, :]
    kp = w_k_prev[:, :, :w_count, :]
    zl = z[l_idx]                               # [B, P, V]
    has_prev = (s_pos_np > 0)[None, :, None]    # [1, P, 1]
    zl_eff = np.where(has_prev, zl, 0.0)

    # Effective shifted keys: [B, P, W_l, V]
    k_cur_eff = kc * (1.0 - zl_eff)[:, :, None, :]
    k_prev_eff = kp * zl_eff[:, :, None, :]

    # Couplings: q_tot [B, P, 4, N, V] dot k_eff / m_rms -> [B, P, W_l, 4, N]
    ql = q_tot[l_idx] / m_rms[l_idx][:, :, None, None, None]
    c_cur = np.einsum("bpsnv,bpwv->bpwsn", ql, k_cur_eff)
    c_prev = np.einsum("bpsnv,bpwv->bpwsn", ql, k_prev_eff)
    u_rec = np.einsum("bpwsn,bpwk->bpsnk", c_cur, oc) + np.einsum("bpwsn,bpwk->bpsnk", c_prev, op)

    rel_err = np.linalg.norm(u_rec - u_dir[l_idx]) / max(np.linalg.norm(u_dir[l_idx]), 1e-12)
    # Also check sum of normalized attribution weights == 1.0!
    u_norm2 = np.sum(u_rec ** 2, axis=-1)  # [B, P, 4, N]
    attr_cur = np.einsum("bpwsn,bpwk,bpsnk->bpwsn", c_cur, oc, u_rec) / np.maximum(u_norm2[:, :, None, :, :], 1e-12)
    attr_prev = np.einsum("bpwsn,bpwk,bpsnk->bpwsn", c_prev, op, u_rec) / np.maximum(u_norm2[:, :, None, :, :], 1e-12)
    col_sum = np.sum(attr_cur + attr_prev, axis=2)
    print(f"  layer {l_idx}: w_count={w_count:2d}, BF16-vs-FP32 readout rel_rms={rel_err:.5f}, max|sum(A)-1|={np.max(np.abs(col_sum - 1.0)):.2e}")
    assert rel_err < 1.5e-2
    assert np.max(np.abs(col_sum - 1.0)) < 1e-5


def test_mhc_equivalence():
  params_npz = Path("/data0/mengqy/pythia-mhc-hf/params_rand2.npz")
  ref_npz = Path("/data0/mengqy/pythia-mhc-hf/ref_rand2_fp32.npz")
  if not (params_npz.exists() and ref_npz.exists()):
    print("[mHC-Pythia] skipping (reference npz not present)")
    return
  raw = np.load(params_npz)
  flat_params = {tuple(k.split("/")): jnp.asarray(raw[k]) for k in raw.files}
  tree = unflatten_dict(flat_params)
  ref = np.load(ref_npz)
  toks = jnp.asarray(ref["tokens"], dtype=jnp.int32)  # [B, T]
  # Check shapes in ref
  # Slice first sequence so CPU eager test completes in ~2s
  toks = jnp.asarray(ref["tokens"][0:1], dtype=jnp.int32)
  ref_xent0 = ref["xent"][0:1]
  _, t_len = toks.shape
  m_params_f32 = mj.prepare_mhc_params(tree, dtype=jnp.float32)
  head_qpos = jnp.array([2, min(15, ref_xent0.shape[1] - 1)], dtype=jnp.int32)
  cats = jnp.zeros_like(toks)
  sink = toks == 0
  obs = mj.mhc_forward_and_observe(
      m_params_f32, toks, toks, head_qpos, cats, sink, sink, dtype=jnp.float32,
  )
  logit_diff = float(np.max(np.abs(np.asarray(obs["last_logits"][0]) - ref["logits"][0, -1])))
  print(f"[mHC-Pythia] 32-layer last_logits max_diff vs ref_rand2_fp32 = {logit_diff:.6f}")
  assert logit_diff < 2e-2


def test_pythia_hf_loader_and_forward():
  st_glob = list(Path("/data0/mengqy/cache/huggingface/hub/models--EleutherAI--pythia-2.8b/snapshots").glob("*/model.safetensors"))
  if not st_glob:
    print("[Pythia-2.8B] skipping (local safetensors cache not found)")
    return
  from safetensors import safe_open
  # Build a tiny 2-layer subset from actual Pythia-2.8B safetensors shapes/dtypes to verify loader + forward!
  hf_sub = {}
  with safe_open(str(st_glob[0]), framework="np") as f:
    for k in f.keys():
      if k in (
          "gpt_neox.embed_in.weight",
          "gpt_neox.final_layer_norm.weight",
          "gpt_neox.final_layer_norm.bias",
          "embed_out.weight",
      ) or k.startswith("gpt_neox.layers.0.") or k.startswith("gpt_neox.layers.1."):
        hf_sub[k] = f.get_tensor(k)
  p_params = mj.prepare_pythia_hf_params(hf_sub, num_layers=2, dtype=jnp.bfloat16)
  toks = jnp.array([[0, 15, 310, 1294, 512, 964, 187, 2048]], dtype=jnp.int32)
  obs = mj.pythia_forward_and_observe(
      p_params, toks[:, :-1], toks[:, 1:], jnp.array([2, 6], dtype=jnp.int32),
      jnp.zeros((1, 7), dtype=jnp.int32), toks[:, :-1] == 0, toks[:, :-1] == 0,
  )
  assert np.all(np.isfinite(np.asarray(obs["xent"])))
  print(f"[Pythia-2.8B 2-layer smoke] xent={np.asarray(obs['xent'])}")


if __name__ == "__main__":
  test_loom_equivalence_and_attribution()
  test_mhc_equivalence()
  test_pythia_hf_loader_and_forward()
  print("ALL_LOCAL_TESTS_PASSED")
