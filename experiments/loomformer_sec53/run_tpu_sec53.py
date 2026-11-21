"""TPU Mechanistic Analysis Runner for §5.3 of the LoomFormer ICML paper.

Runs on an idle retained FLEX_START `v6e-1` TPU (under worker lock, read-only Orbax restore,
isolated `/tmp` source checkout) across all three 300B-token 2.8B models on the exact same
32 truly held-out Pile validation sequences (`val_with_eos.tfrecord`, 65,536 tokens) + ICL prompts:

  1. `prepare_cohort`: deterministic 32-sequence held-out Pile cohort + Pythia lexical classes.
  2. `loom`: LoomPythia-2.8B (`BamPythia2p8BHUnit300BFastKernelMPreNormQKScale`, CP 143000)
     - Native MaxText `Transformer` numerical gate on sequence 0
     - Exact linear writer->reader vector attribution & readout reconstruction check
     - Layer-wise within-token effective wiring rank (full QKVO, QKV-only, cur+prev, per-port, projected A)
     - Layer-wise cross-token rewiring dynamics (cosine distance & top-k Jaccard switching per port)
     - Dynamic vs static read share, MShift previous-token share, Embedding/Attn/Memory-FFN writer share,
       and 32x32 source-depth -> target-depth routing matrix
     - Top-k writer reconstruction faithfulness + whole-network causal loss ablations
     - Structured ICL / syntax / IOI prompt micro-circuit attribution dump (`loom_icl_circuits.npz`)
     - Layer-wise attention head activation ratios & adjacent-layer stream cosine similarities
  3. `mhc`: mHC-Pythia-2.8B (`Pythia2p8BSerialQKNormMHC300B`, CP 143000)
     - Matched 32-sequence held-out Pile NLL, layer-wise attention head activation & sink metrics,
       and adjacent-layer cosine similarities
  4. `pythia`: Pythia-2.8B (`EleutherAI/pythia-2.8b`, released 143k step)
     - Matched 32-sequence held-out Pile NLL, layer-wise attention head activation & sink metrics,
       and adjacent-layer cosine similarities
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
import unicodedata
from typing import Any

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "MaxText"))
sys.path.insert(0, str(THIS_DIR))


# ---------------------------------------------------------------------------
# TFRecord reader without importing TensorFlow runtime
# ---------------------------------------------------------------------------

def _example_class():
  from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
  f = descriptor_pb2.FileDescriptorProto(name="sec53_example.proto", package="sec53_tf", syntax="proto3")

  def msg(name):
    return f.message_type.add(name=name)

  def fld(m, name, num, typ, repeated=False, type_name=None):
    v = m.field.add(name=name, number=num, type=typ, label=3 if repeated else 1)
    if type_name:
      v.type_name = ".sec53_tf." + type_name
    return v

  for name, typ in [("BytesList", 12), ("FloatList", 2), ("Int64List", 3)]:
    v = fld(msg(name), "value", 1, typ, True)
    if typ != 12:
      v.options.packed = True
  m = msg("Feature")
  m.oneof_decl.add(name="kind")
  for num, name, typ in [(1, "bytes_list", "BytesList"), (2, "float_list", "FloatList"), (3, "int64_list", "Int64List")]:
    fld(m, name, num, 11, type_name=typ).oneof_index = 0
  m = msg("Features")
  entry = m.nested_type.add(name="FeatureEntry")
  entry.options.map_entry = True
  fld(entry, "key", 1, 9)
  fld(entry, "value", 2, 11, type_name="Feature")
  fld(m, "feature", 1, 11, True, "Features.FeatureEntry")
  fld(msg("Example"), "features", 1, 11, type_name="Features")
  pool = descriptor_pool.DescriptorPool()
  pool.Add(f)
  return message_factory.GetMessageClass(pool.FindMessageTypeByName("sec53_tf.Example"))


def build_pythia_categories(vocab_size: int = 50304) -> tuple[np.ndarray, Any]:
  from huggingface_hub import hf_hub_download
  from tokenizers import Tokenizer
  tok_path = hf_hub_download("EleutherAI/pythia-2.8b", "tokenizer.json")
  tok = Tokenizer.from_file(tok_path)
  specials = {t["id"] for t in json.loads(tok.to_str())["added_tokens"] if t.get("special")}

  def classify(text: str, is_special: bool) -> int:
    if is_special:
      return 3
    s = text.strip()
    if not s:
      return 2
    if all(unicodedata.category(c).startswith("P") for c in s):
      return 1
    if all(unicodedata.category(c)[0] in "PS" for c in s):
      return 4
    return 0

  n_tok = tok.get_vocab_size()
  cats = np.full((vocab_size,), 3, dtype=np.int32)
  for i in range(min(n_tok, vocab_size)):
    cats[i] = classify(tok.decode([i], skip_special_tokens=False), i in specials)
  return cats, tok


def seq_sha256(arr: np.ndarray) -> str:
  return hashlib.sha256(np.asarray(arr, dtype=np.int32).tobytes()).hexdigest()[:16]


def cmd_prepare_cohort(args: argparse.Namespace) -> None:
  out_dir = Path(args.output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  cohort_path = out_dir / "cohort32_heldout_pile.npz"

  local_tfrecord = out_dir / "val_with_eos.tfrecord"
  if not local_tfrecord.exists():
    print(f"Downloading held-out Pile TFRecord from {args.tfrecord_gcs} ...", flush=True)
    subprocess.run(["gsutil", "-q", "cp", args.tfrecord_gcs, str(local_tfrecord)], check=True)

  Example = _example_class()
  pool = []
  with local_tfrecord.open("rb") as f:
    while len(pool) < 512:
      hdr = f.read(12)
      if len(hdr) < 12:
        break
      n = struct.unpack("<Q", hdr[:8])[0]
      body = f.read(n)
      f.read(4)
      ex = Example.FromString(body)
      ids = np.asarray(ex.features.feature["input_ids"].int64_list.value, dtype=np.int32)
      if len(ids) >= args.length + 1:
        pool.append(ids[: args.length + 1])

  if len(pool) < args.sequences:
    raise RuntimeError(f"Only found {len(pool)} full-length sequences, needed {args.sequences}")

  chosen = np.random.default_rng(args.seed).permutation(len(pool))[: args.sequences]
  tokens = np.stack([pool[i] for i in chosen], axis=0)  # [S, 2049]
  inputs = tokens[:, :-1]
  targets = tokens[:, 1:]
  hashes = np.array([seq_sha256(row) for row in tokens])
  cats, _ = build_pythia_categories(50304)

  np.savez_compressed(
      cohort_path,
      tokens=tokens,
      inputs=inputs,
      targets=targets,
      source_indices=chosen,
      sequence_hashes=hashes,
      categories=cats,
  )
  # Remove temporary TFRecord to keep disk tidy
  local_tfrecord.unlink(missing_ok=True)
  if args.gcs_output:
    subprocess.run(["gsutil", "-q", "cp", str(cohort_path), args.gcs_output.rstrip("/") + "/cohort32_heldout_pile.npz"], check=True)
  print(f"COHORT_READY path={cohort_path} sequences={len(inputs)} first_hash={hashes[0]}", flush=True)


# ---------------------------------------------------------------------------
# Orbax single-device checkpoint loader
# ---------------------------------------------------------------------------

def restore_orbax_params_single_device(checkpoint_items: str):
  import jax
  import orbax.checkpoint as ocp
  from flax import linen as nn
  from flax.core import unfreeze

  t0 = time.monotonic()
  sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
  cp = ocp.PyTreeCheckpointer()
  meta = cp.metadata(checkpoint_items)
  tree = meta.item_metadata.tree["params"] if "params" in meta.item_metadata.tree else meta.item_metadata.tree
  item_wrapped = "params" in meta.item_metadata.tree
  abstract = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=sharding), tree)
  if item_wrapped:
    restored = cp.restore(
        checkpoint_items,
        item={"params": abstract},
        transforms={},
        restore_args={"params": ocp.checkpoint_utils.construct_restore_args(abstract)},
    )["params"]
  else:
    restored = cp.restore(
        checkpoint_items,
        item=abstract,
        transforms={},
        restore_args=ocp.checkpoint_utils.construct_restore_args(abstract),
    )
  cp.close()
  params = jax.device_put(unfreeze(nn.unbox(restored)), sharding)
  del restored
  if isinstance(params, dict) and "params" in params:
    params = params["params"]
  jax.block_until_ready(params)
  print(f"RESTORE_OK {checkpoint_items} in {time.monotonic() - t0:.2f}s", flush=True)
  return params


def sink_masks_for_inputs(inputs_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  # Pythia / GPT-NeoX token IDs: '<|endoftext|>': 0, '.': 15, ' .': 964, '\n': 187, '\n\n': 535
  paper_ids = np.array([0, 15, 964, 187, 535], dtype=np.int32)
  strict_ids = np.array([0, 15, 187], dtype=np.int32)
  return np.isin(inputs_np, paper_ids), np.isin(inputs_np, strict_ids)


def summarize_head_and_cos(
    head_stats_list: list[np.ndarray],
    adj_cos_list: list[np.ndarray],
    cos_names: list[str],
    stat_names: tuple[str, ...],
) -> dict[str, Any]:
  # head_stats_list: S elements of [L, 1, H, Q, M] -> [S, L, H, Q, M]
  hs = np.concatenate(head_stats_list, axis=1)  # [L, S, H, Q, M]
  per_layer_mean = np.mean(hs, axis=(1, 2, 3))    # [L, M]
  per_seq_layer = np.mean(hs, axis=(2, 3))       # [L, S, M]
  per_layer_std = np.std(per_seq_layer, axis=1)  # [L, M] across sequences
  per_head_layer = np.mean(hs, axis=(1, 3))      # [L, H, M]

  ac = np.stack(adj_cos_list, axis=0)            # [S, C, L-1]
  ac_mean = np.mean(ac, axis=0)                  # [C, L-1]
  ac_std = np.std(ac, axis=0)                    # [C, L-1]

  head_dict = {}
  for m_i, m_name in enumerate(stat_names):
    head_dict[m_name] = {
        "layer_mean": per_layer_mean[:, m_i].tolist(),
        "layer_seq_std": per_layer_std[:, m_i].tolist(),
        "overall_mean": float(np.mean(per_layer_mean[:, m_i])),
        "deep_half_mean": float(np.mean(per_layer_mean[per_layer_mean.shape[0] // 2 :, m_i])),
        "last8_mean": float(np.mean(per_layer_mean[-8:, m_i])),
        "per_head_mean": per_head_layer[:, :, m_i].tolist(),
    }

  cos_dict = {
      name: {
          "layer_mean": ac_mean[c_i].tolist(),
          "layer_seq_std": ac_std[c_i].tolist(),
          "last8_mean": float(np.mean(ac_mean[c_i, -8:])),
      }
      for c_i, name in enumerate(cos_names)
  }
  return {"head_activation": head_dict, "adjacent_layer_cosine": cos_dict}


# ---------------------------------------------------------------------------
# Host-side exact LoomFormer mechanistic statistics
# ---------------------------------------------------------------------------

def svd_rank_metrics(mat: np.ndarray, eps: float = 1.0e-12) -> tuple[float, float, float, np.ndarray, np.ndarray]:
  """Return (amp_eff_rank, energy_eff_rank, stable_rank, cum_energy_at_k, norm_svals)."""
  s = np.linalg.svd(mat, compute_uv=False)
  s = np.maximum(s.astype(np.float64), 0.0)
  s_sum = max(float(np.sum(s)), eps)
  p_amp = s / s_sum
  amp_erank = float(np.exp(-np.sum(np.where(p_amp > 0, p_amp * np.log(np.maximum(p_amp, 1e-30)), 0.0))))

  e = s * s
  e_sum = max(float(np.sum(e)), eps)
  p_e = e / e_sum
  energy_erank = float(np.exp(-np.sum(np.where(p_e > 0, p_e * np.log(np.maximum(p_e, 1e-30)), 0.0))))
  stable_rank = float(e_sum / max(float(e[0]), eps))

  ks = (1, 3, 4, 8, 16, 32, 64)
  cum = np.cumsum(p_e)
  cum_at_k = np.array([float(cum[min(k, len(cum)) - 1]) for k in ks], dtype=np.float64)
  s_padded = np.zeros((128,), dtype=np.float64)
  n_copy = min(len(s), 128)
  s_padded[:n_copy] = s[:n_copy] / max(float(s[0]), eps)
  return amp_erank, energy_erank, stable_rank, cum_at_k, s_padded


def pairwise_cos_dist_mean(vecs: np.ndarray, eps: float = 1.0e-12) -> float:
  """vecs: [P, D] across P distinct tokens -> mean off-diagonal (1 - cos)."""
  p_cnt = vecs.shape[0]
  if p_cnt < 2:
    return 0.0
  vf = vecs.astype(np.float64)
  norms = np.linalg.norm(vf, axis=-1, keepdims=True)
  valid = (norms[:, 0] > eps)
  if np.sum(valid) < 2:
    return float("nan")
  vn = vf[valid] / norms[valid]
  gram = vn @ vn.T
  n_v = vn.shape[0]
  off_sum = np.sum(gram) - np.trace(gram)
  mean_cos = off_sum / (n_v * (n_v - 1))
  return float(1.0 - mean_cos)


def pairwise_topk_jaccard_dist(vecs: np.ndarray, k: int) -> float:
  """vecs: [P, W] -> mean off-diagonal (1 - Jaccard(TopK(|v(t1)|), TopK(|v(t2)|)))."""
  p_cnt, w_cnt = vecs.shape
  if p_cnt < 2:
    return 0.0
  k_eff = min(k, w_cnt)
  abs_v = np.abs(vecs)
  top_idx = np.argpartition(-abs_v, kth=k_eff - 1, axis=-1)[:, :k_eff]
  indicators = np.zeros((p_cnt, w_cnt), dtype=np.float32)
  np.put_along_axis(indicators, top_idx, 1.0, axis=-1)
  inter = indicators @ indicators.T  # [P, P]
  union = 2.0 * k_eff - inter
  jacc = inter / np.maximum(union, 1.0e-12)
  off_sum = np.sum(jacc) - np.trace(jacc)
  return float(1.0 - off_sum / (p_cnt * (p_cnt - 1)))


def analyze_loom_records(
    seq_records: list[dict[str, np.ndarray]],
    sample_pos: np.ndarray,
) -> dict[str, Any]:
  """Compute comprehensive layer-wise mechanistic statistics across all sampled tokens."""
  n_seq = len(seq_records)
  n_layers = seq_records[0]["layer_z"].shape[0]
  n_pos = len(sample_pos)
  n_heads = seq_records[0]["layer_q_dyn"].shape[4]
  port_names = ("Q", "K", "V", "O")
  topk_faith_ks = (1, 2, 4, 8, 16, 32, 64, 128)

  # Accumulators per layer
  rank_modes = ("cur_qkvo", "cur_qkv", "cur_prev_qkvo", "attr_qkvo", "port_Q", "port_K", "port_V", "port_O")
  erank_amp = {m: [[] for _ in range(n_layers)] for m in rank_modes}
  erank_energy = {m: [[] for _ in range(n_layers)] for m in rank_modes}
  srank = {m: [[] for _ in range(n_layers)] for m in rank_modes}
  cum_energy = {m: np.zeros((n_layers, 7), dtype=np.float64) for m in rank_modes}
  sval_spec = {m: np.zeros((n_layers, 128), dtype=np.float64) for m in rank_modes}

  recon_rel_err = np.zeros((n_layers,), dtype=np.float64)

  # Cross-token rewiring per (layer, port): within-sequence across P positions
  rewire_attr_cos = np.zeros((n_layers, 4), dtype=np.float64)
  rewire_coup_cos = np.zeros((n_layers, 4), dtype=np.float64)
  rewire_qdyn_cos = np.zeros((n_layers, 4), dtype=np.float64)
  rewire_qtot_cos = np.zeros((n_layers, 4), dtype=np.float64)
  rewire_jacc5 = np.zeros((n_layers, 4), dtype=np.float64)
  rewire_jacc10 = np.zeros((n_layers, 4), dtype=np.float64)
  rewire_jacc16 = np.zeros((n_layers, 4), dtype=np.float64)

  # Branch & source shares per (layer, port)
  dyn_norm_share = np.zeros((n_layers, 4), dtype=np.float64)
  dyn_signed_share = np.zeros((n_layers, 4), dtype=np.float64)
  prev_signed_share = np.zeros((n_layers, 4), dtype=np.float64)
  prev_abs_share = np.zeros((n_layers, 4), dtype=np.float64)
  writer_kind_signed = np.zeros((n_layers, 4, 3), dtype=np.float64)  # 0=emb, 1=attn, 2=mem_ffn
  writer_kind_posmass = np.zeros((n_layers, 4, 3), dtype=np.float64)
  depth_routing_posmass = np.zeros((n_layers, n_layers + 1, 4), dtype=np.float64)  # src: 0=emb, 1..31=layer l_src+1
  mshift_gate_mean = np.zeros((n_layers,), dtype=np.float64)
  mshift_gate_std = np.zeros((n_layers,), dtype=np.float64)

  # Top-k faithfulness per layer (averaged over ports & tokens) and per port
  faith_cos = np.zeros((n_layers, 4, len(topk_faith_ks)), dtype=np.float64)
  faith_r2 = np.zeros((n_layers, 4, len(topk_faith_ks)), dtype=np.float64)

  has_prev = (sample_pos > 0)  # [P]
  total_tokens = n_seq * n_pos

  for s_idx, rec in enumerate(seq_records):
    t_s = time.monotonic()
    w_o_cur = rec["writers_o_cur"][0]    # [P, W_all, K]
    w_k_cur = rec["writers_k_cur"][0]    # [P, W_all, V]
    w_o_prev = rec["writers_o_prev"][0]  # [P, W_all, K]
    w_k_prev = rec["writers_k_prev"][0]  # [P, W_all, V]
    w_norm_cur = np.linalg.norm(w_o_cur, axis=-1)    # [P, W_all]
    w_norm_prev = np.linalg.norm(w_o_prev, axis=-1)  # [P, W_all]

    for l_idx in range(n_layers):
      w_cnt = n_heads + 2 * n_heads * l_idx
      oc = w_o_cur[:, :w_cnt, :]          # [P, W_l, K]
      kc = w_k_cur[:, :w_cnt, :]          # [P, W_l, V]
      op = w_o_prev[:, :w_cnt, :]
      kp = w_k_prev[:, :w_cnt, :]
      nc = w_norm_cur[:, :w_cnt]          # [P, W_l]
      np_ = w_norm_prev[:, :w_cnt]        # [P, W_l]

      zl = rec["layer_z"][l_idx, 0]       # [P, V]
      m_rms = rec["layer_m_rms"][l_idx, 0]  # [P]
      q_dyn = rec["layer_q_dyn"][l_idx, 0]  # [P, 4, N, V]
      q_stat = rec["layer_q_stat"][l_idx, 0]
      q_tot = q_dyn + q_stat
      u_dir = rec["layer_u_read"][l_idx, 0]  # [P, 4, N, K]

      mshift_gate_mean[l_idx] += float(np.mean(zl)) / n_seq
      mshift_gate_std[l_idx] += float(np.std(zl)) / n_seq

      zl_eff = np.where(has_prev[:, None], zl, 0.0)
      k_cur_eff = kc * (1.0 - zl_eff)[:, None, :]  # [P, W_l, V]
      k_prev_eff = kp * zl_eff[:, None, :]         # [P, W_l, V]

      q_scaled = q_tot / m_rms[:, None, None, None]  # [P, 4, N, V]
      q_dyn_sc = q_dyn / m_rms[:, None, None, None]
      q_stat_sc = q_stat / m_rms[:, None, None, None]

      # Couplings: [P, W_l, 4, N]
      c_cur = np.einsum("psnv,pwv->pwsn", q_scaled, k_cur_eff)
      c_prev = np.einsum("psnv,pwv->pwsn", q_scaled, k_prev_eff)

      # Exact FP32 reconstructed readout u_rec [P, 4, N, K]
      u_cur = np.einsum("pwsn,pwk->psnk", c_cur, oc)
      u_prev = np.einsum("pwsn,pwk->psnk", c_prev, op)
      u_rec = u_cur + u_prev

      # Dynamic vs static readout vectors
      M_eff_p = (
          np.einsum("pwk,pwv->pkv", oc, k_cur_eff) + np.einsum("pwk,pwv->pkv", op, k_prev_eff)
      )  # [P, K, V]
      u_dyn_vec = np.einsum("pkv,psnv->psnk", M_eff_p, q_dyn_sc)
      u_stat_vec = np.einsum("pkv,psnv->psnk", M_eff_p, q_stat_sc)

      recon_rel_err[l_idx] += (
          float(np.linalg.norm(u_rec - u_dir) / max(float(np.linalg.norm(u_dir)), 1e-12)) / n_seq
      )

      u_norm2 = np.maximum(np.sum(u_rec * u_rec, axis=-1), 1.0e-12)  # [P, 4, N]
      # Vector attributions A_cur, A_prev: [P, W_l, 4, N]
      o_dot_u_cur = np.einsum("pwk,psnk->pwsn", oc, u_rec)
      o_dot_u_prev = np.einsum("pwk,psnk->pwsn", op, u_rec)
      attr_cur = (c_cur * o_dot_u_cur) / u_norm2[:, None, :, :]
      attr_prev = (c_prev * o_dot_u_prev) / u_norm2[:, None, :, :]

      # Effective scalar coupling weighted by writer output L2 norm: [P, W_l, 4, N]
      coup_cur = c_cur * nc[:, :, None, None]
      coup_prev = c_prev * np_[:, :, None, None]

      # 1. Within-token effective wiring rank across P positions
      for p_i in range(n_pos):
        mats = {
            "cur_qkvo": coup_cur[p_i].reshape(w_cnt, 4 * n_heads),
            "cur_qkv": coup_cur[p_i, :, :3, :].reshape(w_cnt, 3 * n_heads),
            "cur_prev_qkvo": np.concatenate((coup_cur[p_i], coup_prev[p_i]), axis=0).reshape(2 * w_cnt, 4 * n_heads),
            "attr_qkvo": attr_cur[p_i].reshape(w_cnt, 4 * n_heads),
            "port_Q": coup_cur[p_i, :, 0, :],
            "port_K": coup_cur[p_i, :, 1, :],
            "port_V": coup_cur[p_i, :, 2, :],
            "port_O": coup_cur[p_i, :, 3, :],
        }
        for m_name, mat in mats.items():
          ra, re, sr, ce, sp = svd_rank_metrics(mat)
          erank_amp[m_name][l_idx].append(ra)
          erank_energy[m_name][l_idx].append(re)
          srank[m_name][l_idx].append(sr)
          cum_energy[m_name][l_idx] += ce / total_tokens
          sval_spec[m_name][l_idx] += sp / total_tokens

      # 2. Cross-token rewiring dynamics per port & head (averaged over 32 heads & sequences)
      attr_both = np.concatenate((attr_cur, attr_prev), axis=1)  # [P, 2*W_l, 4, N]
      coup_both = np.concatenate((coup_cur, coup_prev), axis=1)
      for s_p in range(4):
        ac_h, cc_h, qd_h, qt_h, j5_h, j10_h, j16_h = [], [], [], [], [], [], []
        for h_i in range(n_heads):
          ac_h.append(pairwise_cos_dist_mean(attr_both[:, :, s_p, h_i]))
          cc_h.append(pairwise_cos_dist_mean(coup_both[:, :, s_p, h_i]))
          qd_h.append(pairwise_cos_dist_mean(q_dyn[:, s_p, h_i, :]))
          qt_h.append(pairwise_cos_dist_mean(q_tot[:, s_p, h_i, :]))
          j5_h.append(pairwise_topk_jaccard_dist(attr_both[:, :, s_p, h_i], 5))
          j10_h.append(pairwise_topk_jaccard_dist(attr_both[:, :, s_p, h_i], 10))
          j16_h.append(pairwise_topk_jaccard_dist(attr_both[:, :, s_p, h_i], 16))
        rewire_attr_cos[l_idx, s_p] += float(np.nanmean(ac_h)) / n_seq
        rewire_coup_cos[l_idx, s_p] += float(np.nanmean(cc_h)) / n_seq
        rewire_qdyn_cos[l_idx, s_p] += float(np.nanmean(qd_h)) / n_seq
        rewire_qtot_cos[l_idx, s_p] += float(np.nanmean(qt_h)) / n_seq
        rewire_jacc5[l_idx, s_p] += float(np.nanmean(j5_h)) / n_seq
        rewire_jacc10[l_idx, s_p] += float(np.nanmean(j10_h)) / n_seq
        rewire_jacc16[l_idx, s_p] += float(np.nanmean(j16_h)) / n_seq

      # 3. Dynamic vs static, cur vs prev, and writer-kind / source-depth shares
      nd = np.linalg.norm(u_dyn_vec, axis=-1)   # [P, 4, N]
      ns = np.linalg.norm(u_stat_vec, axis=-1)  # [P, 4, N]
      dyn_norm_share[l_idx] += np.mean(nd / np.maximum(nd + ns, 1e-12), axis=(0, 2)) / n_seq
      dyn_signed_share[l_idx] += np.mean(
          np.sum(u_dyn_vec * u_rec, axis=-1) / u_norm2, axis=(0, 2)
      ) / n_seq

      prev_signed = np.sum(attr_prev, axis=1)  # [P, 4, N]
      prev_signed_share[l_idx] += np.mean(prev_signed, axis=(0, 2)) / n_seq
      abs_cur = np.sum(np.abs(attr_cur), axis=1)
      abs_prev = np.sum(np.abs(attr_prev), axis=1)
      prev_abs_share[l_idx] += np.mean(abs_prev / np.maximum(abs_cur + abs_prev, 1e-12), axis=(0, 2)) / n_seq

      # Writer kind & source depth (combining cur + prev per writer i in 0..W_l-1)
      attr_w = attr_cur + attr_prev  # [P, W_l, 4, N]
      pos_w = np.maximum(attr_w, 0.0)
      pos_tot = np.maximum(np.sum(pos_w, axis=1, keepdims=True), 1e-12)
      pos_norm = pos_w / pos_tot  # [P, W_l, 4, N]

      # Embedding: 0..N-1
      emb_s = np.sum(attr_w[:, :n_heads], axis=1)     # [P, 4, N]
      emb_p = np.sum(pos_norm[:, :n_heads], axis=1)
      depth_routing_posmass[l_idx, 0] += np.mean(emb_p, axis=(0, 2)) / n_seq

      attn_s = np.zeros_like(emb_s)
      ffn_s = np.zeros_like(emb_s)
      attn_p = np.zeros_like(emb_p)
      ffn_p = np.zeros_like(emb_p)
      for src_l in range(l_idx):
        a0 = n_heads + 2 * n_heads * src_l
        a1 = a0 + n_heads
        m1 = a1 + n_heads
        a_sl_s = np.sum(attr_w[:, a0:a1], axis=1)
        m_sl_s = np.sum(attr_w[:, a1:m1], axis=1)
        a_sl_p = np.sum(pos_norm[:, a0:a1], axis=1)
        m_sl_p = np.sum(pos_norm[:, a1:m1], axis=1)
        attn_s += a_sl_s
        ffn_s += m_sl_s
        attn_p += a_sl_p
        ffn_p += m_sl_p
        depth_routing_posmass[l_idx, src_l + 1] += np.mean(a_sl_p + m_sl_p, axis=(0, 2)) / n_seq

      writer_kind_signed[l_idx, :, 0] += np.mean(emb_s, axis=(0, 2)) / n_seq
      writer_kind_signed[l_idx, :, 1] += np.mean(attn_s, axis=(0, 2)) / n_seq
      writer_kind_signed[l_idx, :, 2] += np.mean(ffn_s, axis=(0, 2)) / n_seq
      writer_kind_posmass[l_idx, :, 0] += np.mean(emb_p, axis=(0, 2)) / n_seq
      writer_kind_posmass[l_idx, :, 1] += np.mean(attn_p, axis=(0, 2)) / n_seq
      writer_kind_posmass[l_idx, :, 2] += np.mean(ffn_p, axis=(0, 2)) / n_seq

      # 4. Top-k writer reconstruction faithfulness (over combined cur+prev writers: 2*W_l)
      # Subsample 8 positions per sequence for top-k vector reconstruction speed
      sub_p = np.linspace(0, n_pos - 1, min(8, n_pos), dtype=np.int32)
      c_both_sub = np.concatenate((c_cur[sub_p], c_prev[sub_p]), axis=1)          # [Ps, 2*W_l, 4, N]
      o_both_sub = np.concatenate((oc[sub_p], op[sub_p]), axis=1)                 # [Ps, 2*W_l, K]
      a_both_sub = np.concatenate((attr_cur[sub_p], attr_prev[sub_p]), axis=1)    # [Ps, 2*W_l, 4, N]
      u_sub = u_rec[sub_p]                                                        # [Ps, 4, N, K]
      u_sub_n2 = np.maximum(np.sum(u_sub * u_sub, axis=-1), 1e-12)
      u_sub_n = np.sqrt(u_sub_n2)

      order = np.argsort(-a_both_sub, axis=1)  # descending by signed attribution A_{ij}
      max_k = min(max(topk_faith_ks), 2 * w_cnt)
      # Gather top max_k vectors: [Ps, max_k, 4, N, K]
      top_ord = order[:, :max_k, :, :]  # [Ps, max_k, 4, N]
      c_top = np.take_along_axis(c_both_sub, top_ord, axis=1)  # [Ps, max_k, 4, N]
      # Gather o_both_sub [Ps, 2*W_l, K] along axis 1
      ps_idx = np.arange(len(sub_p))[:, None, None, None]
      u_terms = c_top[..., None] * o_both_sub[ps_idx, top_ord, :]  # [Ps, max_k, 4, N, K]
      u_cum = np.cumsum(u_terms, axis=1)  # [Ps, max_k, 4, N, K]
      for k_i, k_val in enumerate(topk_faith_ks):
        ke = min(k_val, max_k) - 1
        uk = u_cum[:, ke]  # [Ps, 4, N, K]
        uk_n = np.maximum(np.linalg.norm(uk, axis=-1), 1e-12)
        cos_k = np.sum(uk * u_sub, axis=-1) / (uk_n * u_sub_n)
        r2_k = 1.0 - np.sum((u_sub - uk) ** 2, axis=-1) / u_sub_n2
        faith_cos[l_idx, :, k_i] += np.mean(cos_k, axis=(0, 2)) / n_seq
        faith_r2[l_idx, :, k_i] += np.mean(r2_k, axis=(0, 2)) / n_seq

    print(f"  [host stats] seq {s_idx + 1}/{n_seq} done in {time.monotonic() - t_s:.2f}s", flush=True)

  def pack_rank(d_list: dict[str, list[list[float]]]) -> dict[str, Any]:
    out = {}
    for m_name in rank_modes:
      arr = np.asarray(d_list[m_name], dtype=np.float64)  # [L, S*P]
      out[m_name] = {
          "mean": np.mean(arr, axis=-1).tolist(),
          "std": np.std(arr, axis=-1).tolist(),
          "p10": np.percentile(arr, 10, axis=-1).tolist(),
          "p50": np.percentile(arr, 50, axis=-1).tolist(),
          "p90": np.percentile(arr, 90, axis=-1).tolist(),
      }
    return out

  return {
      "num_sequences": n_seq,
      "sampled_positions_per_sequence": n_pos,
      "total_sampled_tokens": total_tokens,
      "port_names": list(port_names),
      "bf16_vs_fp32_readout_rel_rms_by_layer": recon_rel_err.tolist(),
      "effective_rank_amplitude": pack_rank(erank_amp),
      "effective_rank_energy": pack_rank(erank_energy),
      "stable_rank": pack_rank(srank),
      "cumulative_energy_ks": [1, 3, 4, 8, 16, 32, 64],
      "cumulative_energy": {m: cum_energy[m].tolist() for m in rank_modes},
      "singular_value_spectrum_norm1": {m: sval_spec[m].tolist() for m in rank_modes},
      "cross_token_rewiring": {
          "attr_cosine_dist": rewire_attr_cos.tolist(),
          "coupling_cosine_dist": rewire_coup_cos.tolist(),
          "q_dyn_cosine_dist": rewire_qdyn_cos.tolist(),
          "q_tot_cosine_dist": rewire_qtot_cos.tolist(),
          "top5_jaccard_dist": rewire_jacc5.tolist(),
          "top10_jaccard_dist": rewire_jacc10.tolist(),
          "top16_jaccard_dist": rewire_jacc16.tolist(),
      },
      "branch_and_source_shares": {
          "dynamic_norm_share": dyn_norm_share.tolist(),
          "dynamic_signed_share": dyn_signed_share.tolist(),
          "mshift_prev_signed_share": prev_signed_share.tolist(),
          "mshift_prev_abs_share": prev_abs_share.tolist(),
          "mshift_gate_mean": mshift_gate_mean.tolist(),
          "mshift_gate_std": mshift_gate_std.tolist(),
          "writer_kind_names": ["embedding", "attention", "memory_ffn"],
          "writer_kind_signed_share": writer_kind_signed.tolist(),
          "writer_kind_posmass_share": writer_kind_posmass.tolist(),
          "depth_routing_posmass_32x33x4": depth_routing_posmass.tolist(),
      },
      "topk_faithfulness": {
          "ks": list(topk_faith_ks),
          "cosine_similarity": faith_cos.tolist(),
          "explained_energy_r2": faith_r2.tolist(),
      },
  }


# ---------------------------------------------------------------------------
# Structured ICL / syntax micro-circuit dump (`loom_icl_circuits.npz`)
# ---------------------------------------------------------------------------

ICL_PROMPTS = [
    ("country_capital", "France: Paris\nJapan: Tokyo\nChina: Beijing\nSpain: Madrid\nGermany:"),
    ("antonym", "hot -> cold\nup -> down\ndark -> light\nfast -> slow\nhigh ->"),
    ("object_field", 'user.name = "Alice"\nitem.price = 42\nbook.title = "Dune"\nuser.'),
    ("ioi_store", "When Mary and John went to the store, John gave a drink to"),
    ("category_fruit", "apple: fruit\ncarrot: vegetable\nsalmon: fish\nbanana:"),
    ("lang_greeting", "English: Hello\nFrench: Bonjour\nSpanish: Hola\nGerman:"),
]


def run_loom_icl_prompts(
    u_params: dict[str, Any],
    cats_vocab: np.ndarray,
    tok: Any,
    out_dir: Path,
) -> dict[str, Any]:
  import jax
  import jax.numpy as jnp
  import models_jax as mj

  pad_len = 64
  sample_pos = jnp.arange(pad_len, dtype=jnp.int32)
  head_qpos = jnp.arange(2, pad_len, dtype=jnp.int32)
  jit_icl = jax.jit(
      lambda p, inp, tgt, cat, ps, ss, lpos: mj.loom_forward_and_observe(
          p, inp, tgt, sample_pos, head_qpos, cat, ps, ss, logit_pos=lpos
      )
  )

  prompt_meta = []
  npz_payload: dict[str, np.ndarray] = {}

  for name, text in ICL_PROMPTS:
    ids = tok.encode(text).ids
    assert 4 <= len(ids) <= pad_len, (name, len(ids))
    t_real = len(ids)
    inp_np = np.zeros((1, pad_len), dtype=np.int32)
    tgt_np = np.zeros((1, pad_len), dtype=np.int32)
    inp_np[0, :t_real] = ids
    tgt_np[0, : t_real - 1] = ids[1:]
    cat_np = cats_vocab[inp_np]
    ps_np, ss_np = sink_masks_for_inputs(inp_np)

    obs = jax.device_get(jit_icl(
        u_params,
        jnp.asarray(inp_np),
        jnp.asarray(tgt_np),
        jnp.asarray(cat_np),
        jnp.asarray(ps_np),
        jnp.asarray(ss_np),
        jnp.array([t_real - 1], dtype=jnp.int32),
    ))

    # Extract compact writer/reader factors for `[0..t_real-1]`
    w_o_c = obs["writers_o_cur"][0, :t_real]   # [T_real, 2016, 80]
    w_k_c = obs["writers_k_cur"][0, :t_real]   # [T_real, 2016, 64]
    w_o_p = obs["writers_o_prev"][0, :t_real]
    w_k_p = obs["writers_k_prev"][0, :t_real]
    z_l = obs["layer_z"][:, 0, :t_real]        # [32, T_real, 64]
    m_rms = obs["layer_m_rms"][:, 0, :t_real]  # [32, T_real]
    q_d = obs["layer_q_dyn"][:, 0, :t_real]    # [32, T_real, 4, 32, 64]
    q_s = obs["layer_q_stat"][:, 0, :t_real]   # [32, T_real, 4, 32, 64]
    u_r = obs["layer_u_read"][:, 0, :t_real]   # [32, T_real, 4, 32, 80]

    npz_payload[f"{name}__token_ids"] = np.asarray(ids, dtype=np.int32)
    npz_payload[f"{name}__writers_o_cur"] = w_o_c.astype(np.float16)
    npz_payload[f"{name}__writers_k_cur"] = w_k_c.astype(np.float16)
    npz_payload[f"{name}__writers_o_prev"] = w_o_p.astype(np.float16)
    npz_payload[f"{name}__writers_k_prev"] = w_k_p.astype(np.float16)
    npz_payload[f"{name}__layer_z"] = z_l.astype(np.float16)
    npz_payload[f"{name}__layer_m_rms"] = m_rms.astype(np.float32)
    npz_payload[f"{name}__layer_q_dyn"] = q_d.astype(np.float16)
    npz_payload[f"{name}__layer_q_stat"] = q_s.astype(np.float16)
    npz_payload[f"{name}__layer_u_read"] = u_r.astype(np.float16)

    logits_last = obs["last_logits"][0].astype(np.float64)
    probs = np.exp(logits_last - np.max(logits_last))
    probs /= np.sum(probs)
    top5 = np.argsort(-probs)[:5]
    top5_list = [
        {"id": int(idx), "token": tok.decode([int(idx)]), "prob": float(probs[idx])}
        for idx in top5
    ]
    print(f"  [ICL {name}] top1={top5_list[0]}", flush=True)

    prompt_meta.append({
        "name": name,
        "text": text,
        "length": t_real,
        "tokens": [{"pos": i, "id": int(tid), "str": tok.decode([int(tid)])} for i, tid in enumerate(ids)],
        "top5_next": top5_list,
    })

  npz_path = out_dir / "loom_icl_circuits.npz"
  np.savez_compressed(npz_path, **npz_payload)
  return {"prompts": prompt_meta, "npz_file": str(npz_path.name)}


# ---------------------------------------------------------------------------
# Subcommand 2: LoomPythia-2.8B full probe
# ---------------------------------------------------------------------------

def cmd_loom(args: argparse.Namespace) -> None:
  import jax
  import jax.numpy as jnp
  from flax import linen as nn
  import exp
  import max_utils
  import pyconfig
  from layers.models import Transformer
  import models_jax as mj

  out_dir = Path(args.output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  cohort = np.load(args.cohort)
  inputs_all = cohort["inputs"]          # [32, 2048]
  targets_all = cohort["targets"]        # [32, 2048]
  cats_vocab = cohort["categories"]      # [50304]
  seq_hashes = cohort["sequence_hashes"].tolist()
  n_seq, t_len = inputs_all.shape

  # 1. Restore Orbax checkpoint once
  raw_params = restore_orbax_params_single_device(args.checkpoint)

  # 2. Numerical gate on sequence 0 against MaxText `Transformer`!
  tmp_cfg_dir = "/tmp/loom_sec53_gate/gate/"
  Path(tmp_cfg_dir).mkdir(parents=True, exist_ok=True)
  cfg = pyconfig.initialize(
      [None, str(REPO_ROOT / "MaxText/configs/base.yml")],
      exp_class="BamPythia2p8BHUnit300BFastKernelMPreNormQKScale",
      run_name="gate",
      base_output_directory="/tmp/loom_sec53_gate/",
      enable_checkpointing=False,
      only_eval=True,
      dataset_type="synthetic",
      target_train_batch_size=None,
      per_device_batch_size=1.0,
      max_target_length=t_len,
      max_prefill_predict_length=t_len,
      record_internal_nn_metrics=False,
      record_training_health_metrics=False,
      jax_cache_dir="",
      log_config=False,
  )
  cfg.get_keys().update(
      bam_pallas_core=False,
      bam_pallas_write=False,
      bam_splash_attention=False,
      bam_pallas_packed_read_keys=False,
      bam_pallas_fused_qk_norm=False,
      float32_logits=True,
      record_logits_health_metrics=False,
      remat_policy="none",
  )
  mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
  mt_model = Transformer(cfg, mesh, quant=None)
  pos0 = jnp.arange(t_len, dtype=jnp.int32)[None, :]
  seg0 = jnp.ones((1, t_len), dtype=jnp.int32)
  with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
    jit_mt = jax.jit(
        lambda p, x, y: mt_model.apply(
            {"params": p}, x, pos0,
            decoder_segment_ids=seg0, decoder_target_mask=seg0,
            decoder_target_tokens=y, enable_dropout=False, mutable=False,
        )[0]
    )
    ref_xent0 = np.asarray(jax.device_get(jit_mt(raw_params, jnp.asarray(inputs_all[0:1]), jnp.asarray(targets_all[0:1]))))
  del jit_mt, mt_model
  gc.collect()

  u_params = mj.prepare_loom_params(raw_params, jnp.bfloat16)
  del raw_params
  gc.collect()

  sample_pos_np = np.unique(np.linspace(64, t_len - 1, args.sample_positions, dtype=np.int32))
  head_qpos_np = np.unique(np.linspace(64, t_len - 1, 64, dtype=np.int32))
  sample_pos = jnp.asarray(sample_pos_np)
  head_qpos = jnp.asarray(head_qpos_np)

  jit_obs = jax.jit(
      lambda p, inp, tgt, cat, ps, ss: mj.loom_forward_and_observe(
          p, inp, tgt, sample_pos, head_qpos, cat, ps, ss, ablation="none"
      )
  )

  seq_records = []
  head_stats_list = []
  adj_cos_list = []
  seq_nlls = []

  t_loop = time.monotonic()
  for i in range(n_seq):
    inp_i = inputs_all[i : i + 1]
    tgt_i = targets_all[i : i + 1]
    cat_i = cats_vocab[inp_i]
    ps_i, ss_i = sink_masks_for_inputs(inp_i)
    rec = jax.tree.map(
        np.asarray,
        jax.device_get(jit_obs(
            u_params,
            jnp.asarray(inp_i),
            jnp.asarray(tgt_i),
            jnp.asarray(cat_i),
            jnp.asarray(ps_i),
            jnp.asarray(ss_i),
        )),
    )
    if i == 0:
      abs_diffs = np.abs(ref_xent0 - rec["xent"])
      max_gate_diff = float(np.max(abs_diffs))
      p99_gate_diff = float(np.percentile(abs_diffs, 99))
      med_gate_diff = float(np.median(abs_diffs))
      mean_gate_diff = float(np.abs(np.mean(ref_xent0) - np.mean(rec["xent"])))
      print(
          f"LOOM_GATE_OK seq0_nll={float(np.mean(rec['xent'])):.6f} "
          f"mean_diff={mean_gate_diff:.6e} med_diff={med_gate_diff:.6e} "
          f"p99_diff={p99_gate_diff:.6e} max_diff={max_gate_diff:.6e}",
          flush=True,
      )
      if mean_gate_diff > 5e-3 or p99_gate_diff > 1.5e-1:
        raise RuntimeError(
            f"LoomPythia MaxText equivalence gate failed: mean={mean_gate_diff}, p99={p99_gate_diff}, max={max_gate_diff}"
        )

    seq_nlls.append(float(np.mean(rec["xent"])))
    head_stats_list.append(rec.pop("head_stats"))
    adj_cos_list.append(rec.pop("adj_cos"))
    rec.pop("last_logits")
    rec.pop("xent")
    seq_records.append(rec)
    if (i + 1) in (1, 8, 16, 24, n_seq):
      print(f"LOOM_FORWARD {i + 1}/{n_seq} mean_nll={np.mean(seq_nlls):.6f} elapsed={time.monotonic() - t_loop:.1f}s", flush=True)

  # 3. Whole-network causal loss ablations on the first 8 held-out sequences (16,384 tokens)
  ablation_names = (
      "none",
      "no_dynamic_read",
      "no_static_read",
      "no_m_shift",
      "no_local_o",
      "no_emb_write",
      "no_attn_write",
      "no_mlp_write",
  )
  ablation_results = {}
  n_abl_seq = min(8, n_seq)
  mini_pos = jnp.array([64, 2047], dtype=jnp.int32)
  for abl in ablation_names:
    jit_abl = jax.jit(
        lambda p, inp, tgt, cat, ps, ss, a=abl: mj.loom_forward_and_observe(
            p, inp, tgt, mini_pos, mini_pos, cat, ps, ss, ablation=a
        )["xent"]
    )
    abl_nlls = []
    for i in range(n_abl_seq):
      inp_i = inputs_all[i : i + 1]
      tgt_i = targets_all[i : i + 1]
      cat_i = cats_vocab[inp_i]
      ps_i, ss_i = sink_masks_for_inputs(inp_i)
      xe = np.asarray(jax.device_get(jit_abl(
          u_params, jnp.asarray(inp_i), jnp.asarray(tgt_i),
          jnp.asarray(cat_i), jnp.asarray(ps_i), jnp.asarray(ss_i),
      )))
      abl_nlls.append(float(np.mean(xe)))
    del jit_abl
    mean_nll = float(np.mean(abl_nlls))
    ablation_results[abl] = {
        "sequences": n_abl_seq,
        "per_sequence_nll": abl_nlls,
        "mean_nll": mean_nll,
        "ppl": float(np.exp(min(mean_nll, 20.0))),
    }
    print(f"LOOM_ABLATION {abl:18s}: mean_nll={mean_nll:.5f} (delta={mean_nll - ablation_results['none']['mean_nll']:+.5f})", flush=True)

  # 4. ICL structured micro-circuit prompts
  _, tok = build_pythia_categories(50304)
  icl_summary = run_loom_icl_prompts(u_params, cats_vocab, tok, out_dir)

  del u_params, jit_obs
  gc.collect()

  # 5. Host-side SVD rank, cross-token rewiring, branch/source decomposition, top-k faithfulness
  mech_stats = analyze_loom_records(seq_records, sample_pos_np)
  head_cos = summarize_head_and_cos(
      head_stats_list,
      adj_cos_list,
      ["residual_in", "pre_attn_ln", "local_q", "local_k", "local_v", "local_o"],
      mj.HEAD_STAT_NAMES,
  )

  mean_nll_all = float(np.mean(seq_nlls))
  report = {
      "model": "LoomPythia-2.8B",
      "exp_class": "BamPythia2p8BHUnit300BFastKernelMPreNormQKScale",
      "checkpoint": args.checkpoint,
      "step": 143000,
      "sequence_hashes": seq_hashes,
      "heldout_pile_32x2048": {
          "mean_nll": mean_nll_all,
          "ppl": float(np.exp(mean_nll_all)),
          "per_sequence_nll": seq_nlls,
          "gate_seq0_max_diff": max_gate_diff,
          "gate_seq0_p99_diff": p99_gate_diff,
          "gate_seq0_med_diff": med_gate_diff,
          "gate_seq0_mean_diff": mean_gate_diff,
      },
      "causal_ablations_8x2048": ablation_results,
      "mechanistic": mech_stats,
      "icl_prompts": icl_summary,
      **head_cos,
  }
  out_json = out_dir / "loom_pythia_2p8b_sec53.json"
  out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
  if args.gcs_output:
    subprocess.run(["gsutil", "-q", "cp", str(out_json), args.gcs_output.rstrip("/") + "/loom_pythia_2p8b_sec53.json"], check=True)
    subprocess.run(["gsutil", "-q", "cp", str(out_dir / "loom_icl_circuits.npz"), args.gcs_output.rstrip("/") + "/loom_icl_circuits.npz"], check=True)
  print(f"LOOM_DONE json={out_json} nll={mean_nll_all:.5f} ppl={np.exp(mean_nll_all):.3f}", flush=True)


# ---------------------------------------------------------------------------
# Subcommand 3: mHC-Pythia-2.8B (`Pythia2p8BSerialQKNormMHC300B`)
# ---------------------------------------------------------------------------

def cmd_mhc(args: argparse.Namespace) -> None:
  import jax
  import jax.numpy as jnp
  import models_jax as mj

  out_dir = Path(args.output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  cohort = np.load(args.cohort)
  inputs_all = cohort["inputs"]
  targets_all = cohort["targets"]
  cats_vocab = cohort["categories"]
  seq_hashes = cohort["sequence_hashes"].tolist()
  n_seq, t_len = inputs_all.shape

  raw_params = restore_orbax_params_single_device(args.checkpoint)
  m_params = mj.prepare_mhc_params(raw_params, jnp.bfloat16)
  del raw_params
  gc.collect()

  head_qpos = jnp.asarray(np.unique(np.linspace(64, t_len - 1, 64, dtype=np.int32)))
  jit_mhc = jax.jit(
      lambda p, inp, tgt, cat, ps, ss: mj.mhc_forward_and_observe(
          p, inp, tgt, head_qpos, cat, ps, ss, dtype=jnp.bfloat16
      )
  )

  seq_nlls = []
  head_stats_list = []
  adj_cos_list = []
  t0 = time.monotonic()
  for i in range(n_seq):
    inp_i = inputs_all[i : i + 1]
    tgt_i = targets_all[i : i + 1]
    cat_i = cats_vocab[inp_i]
    ps_i, ss_i = sink_masks_for_inputs(inp_i)
    rec = jax.tree.map(
        np.asarray,
        jax.device_get(jit_mhc(
            m_params, jnp.asarray(inp_i), jnp.asarray(tgt_i),
            jnp.asarray(cat_i), jnp.asarray(ps_i), jnp.asarray(ss_i),
        )),
    )
    seq_nlls.append(float(np.mean(rec["xent"])))
    head_stats_list.append(rec["head_stats"])
    adj_cos_list.append(rec["adj_cos"])
    if (i + 1) in (1, 8, 16, 24, n_seq):
      print(f"MHC_FORWARD {i + 1}/{n_seq} mean_nll={np.mean(seq_nlls):.6f} elapsed={time.monotonic() - t0:.1f}s", flush=True)

  head_cos = summarize_head_and_cos(
      head_stats_list,
      adj_cos_list,
      ["streams_state", "attn_connector_read", "pre_attn_ln"],
      mj.HEAD_STAT_NAMES,
  )
  mean_nll = float(np.mean(seq_nlls))
  report = {
      "model": "mHC-Pythia-2.8B",
      "exp_class": "Pythia2p8BSerialQKNormMHC300B",
      "checkpoint": args.checkpoint,
      "step": 143000,
      "sequence_hashes": seq_hashes,
      "heldout_pile_32x2048": {
          "mean_nll": mean_nll,
          "ppl": float(np.exp(mean_nll)),
          "per_sequence_nll": seq_nlls,
      },
      **head_cos,
  }
  out_json = out_dir / "mhc_pythia_2p8b_sec53.json"
  out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
  if args.gcs_output:
    subprocess.run(["gsutil", "-q", "cp", str(out_json), args.gcs_output.rstrip("/") + "/mhc_pythia_2p8b_sec53.json"], check=True)
  print(f"MHC_DONE json={out_json} nll={mean_nll:.5f} ppl={np.exp(mean_nll):.3f}", flush=True)


# ---------------------------------------------------------------------------
# Subcommand 4: Official Pythia-2.8B (`EleutherAI/pythia-2.8b`)
# ---------------------------------------------------------------------------

def cmd_pythia(args: argparse.Namespace) -> None:
  import jax
  import jax.numpy as jnp
  from huggingface_hub import hf_hub_download
  from safetensors import safe_open
  import models_jax as mj

  out_dir = Path(args.output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  cohort = np.load(args.cohort)
  inputs_all = cohort["inputs"]
  targets_all = cohort["targets"]
  cats_vocab = cohort["categories"]
  seq_hashes = cohort["sequence_hashes"].tolist()
  n_seq, t_len = inputs_all.shape

  t0 = time.monotonic()
  st_path = args.safetensors_path or hf_hub_download("EleutherAI/pythia-2.8b", "model.safetensors")
  hf_tensors = {}
  with safe_open(st_path, framework="np") as f:
    for k in f.keys():
      hf_tensors[k] = f.get_tensor(k)
  sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
  p_params = jax.device_put(mj.prepare_pythia_hf_params(hf_tensors, jnp.bfloat16), sharding)
  del hf_tensors
  gc.collect()
  print(f"PYTHIA_LOAD_OK {st_path} in {time.monotonic() - t0:.2f}s", flush=True)

  head_qpos = jnp.asarray(np.unique(np.linspace(64, t_len - 1, 64, dtype=np.int32)))
  jit_pythia = jax.jit(
      lambda p, inp, tgt, cat, ps, ss: mj.pythia_forward_and_observe(
          p, inp, tgt, head_qpos, cat, ps, ss, dtype=jnp.bfloat16
      )
  )

  seq_nlls = []
  head_stats_list = []
  adj_cos_list = []
  t_loop = time.monotonic()
  for i in range(n_seq):
    inp_i = inputs_all[i : i + 1]
    tgt_i = targets_all[i : i + 1]
    cat_i = cats_vocab[inp_i]
    ps_i, ss_i = sink_masks_for_inputs(inp_i)
    rec = jax.tree.map(
        np.asarray,
        jax.device_get(jit_pythia(
            p_params, jnp.asarray(inp_i), jnp.asarray(tgt_i),
            jnp.asarray(cat_i), jnp.asarray(ps_i), jnp.asarray(ss_i),
        )),
    )
    seq_nlls.append(float(np.mean(rec["xent"])))
    head_stats_list.append(rec["head_stats"])
    adj_cos_list.append(rec["adj_cos"])
    if (i + 1) in (1, 8, 16, 24, n_seq):
      print(f"PYTHIA_FORWARD {i + 1}/{n_seq} mean_nll={np.mean(seq_nlls):.6f} elapsed={time.monotonic() - t_loop:.1f}s", flush=True)

  head_cos = summarize_head_and_cos(
      head_stats_list,
      adj_cos_list,
      ["residual_in", "pre_attn_ln"],
      mj.HEAD_STAT_NAMES,
  )
  mean_nll = float(np.mean(seq_nlls))
  report = {
      "model": "Pythia-2.8B",
      "exp_class": "EleutherAI/pythia-2.8b",
      "checkpoint": "EleutherAI/pythia-2.8b@step143000",
      "step": 143000,
      "sequence_hashes": seq_hashes,
      "heldout_pile_32x2048": {
          "mean_nll": mean_nll,
          "ppl": float(np.exp(mean_nll)),
          "per_sequence_nll": seq_nlls,
      },
      **head_cos,
  }
  out_json = out_dir / "pythia_2p8b_sec53.json"
  out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
  if args.gcs_output:
    subprocess.run(["gsutil", "-q", "cp", str(out_json), args.gcs_output.rstrip("/") + "/pythia_2p8b_sec53.json"], check=True)
  print(f"PYTHIA_DONE json={out_json} nll={mean_nll:.5f} ppl={np.exp(mean_nll):.3f}", flush=True)


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__)
  sub = p.add_subparsers(dest="cmd", required=True)

  sp = sub.add_parser("prepare_cohort")
  sp.add_argument("--tfrecord-gcs", default="gs://newproject-1-common_datasets_us-east5/pythia_model_test/pile_test/val_with_eos.tfrecord")
  sp.add_argument("--sequences", type=int, default=32)
  sp.add_argument("--length", type=int, default=2048)
  sp.add_argument("--seed", type=int, default=20261121)
  sp.add_argument("--output-dir", required=True)
  sp.add_argument("--gcs-output", default="")
  sp.set_defaults(func=cmd_prepare_cohort)

  sl = sub.add_parser("loom")
  sl.add_argument("--checkpoint", default="gs://newproject-1-llm_base_models_us-central1/log/BamPythia2p8BHUnit300BFastKernelMPreNormQKScale/checkpoints/143000/items")
  sl.add_argument("--cohort", required=True)
  sl.add_argument("--sample-positions", type=int, default=32)
  sl.add_argument("--output-dir", required=True)
  sl.add_argument("--gcs-output", default="")
  sl.set_defaults(func=cmd_loom)

  sm = sub.add_parser("mhc")
  sm.add_argument("--checkpoint", default="gs://newproject-1-llm_projects_us-east5/log/Pythia2p8BSerialQKNormMHC300B/checkpoints/143000/items")
  sm.add_argument("--cohort", required=True)
  sm.add_argument("--output-dir", required=True)
  sm.add_argument("--gcs-output", default="")
  sm.set_defaults(func=cmd_mhc)

  spy = sub.add_parser("pythia")
  spy.add_argument("--safetensors-path", default="")
  spy.add_argument("--cohort", required=True)
  spy.add_argument("--output-dir", required=True)
  spy.add_argument("--gcs-output", default="")
  spy.set_defaults(func=cmd_pythia)

  args = p.parse_args()
  args.func(args)


if __name__ == "__main__":
  main()
