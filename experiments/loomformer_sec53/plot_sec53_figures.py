"""Generate ICML-grade vector PDF + PNG preview figures and Markdown/CSV summaries for §5.3.

Reads the real 300B-token 2.8B experimental outputs synced from GCS:
  - `loom_pythia_2p8b_sec53.json`
  - `loom_icl_circuits.npz`
  - `mhc_pythia_2p8b_sec53.json`
  - `pythia_2p8b_sec53.json`

Produces publication figures matching `loomformer_paper/src/plot.py` palette:
  - Pythia-2.8B:     `#222222`
  - mHC-Pythia-2.8B: `#4393C3`
  - LoomPythia-2.8B: `#D62728`
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


C_PYTHIA = "#222222"
C_MHC = "#4393C3"
C_LOOM = "#D62728"
C_PORT = {
    "Q": "#E69F00",  # amber/gold
    "K": "#D55E00",  # vermilion/red
    "V": "#009E73",  # bluish green
    "O": "#0072B2",  # ocean blue
}


def setup_style() -> None:
  plt.rcParams.update({
      "font.family": "sans-serif",
      "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
      "mathtext.fontset": "dejavusans",
      "axes.edgecolor": "#333333",
      "axes.linewidth": 0.9,
      "axes.grid": True,
      "grid.alpha": 0.22,
      "grid.linestyle": "--",
      "grid.linewidth": 0.6,
      "pdf.fonttype": 42,
      "ps.fonttype": 42,
  })


def save_both(fig: plt.Figure, out_dir: Path, stem: str) -> tuple[Path, Path]:
  pdf_p = out_dir / f"{stem}.pdf"
  png_p = out_dir / f"{stem}.png"
  fig.savefig(pdf_p, bbox_inches="tight")
  fig.savefig(png_p, dpi=220, bbox_inches="tight")
  plt.close(fig)
  return pdf_p, png_p


def plot_wiring_rank_and_rewiring(loom: dict, out_dir: Path) -> tuple[Path, Path]:
  """2-panel figure:
  (a) Effective within-token wiring rank vs layer depth (with structural ceilings: 1 for Res/mHC/AttnRes, 3 for MUDD QKV)
  (b) Cross-token circuit rewiring dynamics across sequence positions per port (Q, K, V, O).
  """
  mech = loom["mechanistic"]
  layers = np.arange(1, 33)

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.3))

  # Panel (a): Within-token effective rank (SVD Shannon effective rank)
  er_qkvo = np.array(mech["effective_rank_amplitude"]["cur_qkvo"]["mean"])
  er_qkvo_p10 = np.array(mech["effective_rank_amplitude"]["cur_qkvo"]["p10"])
  er_qkvo_p90 = np.array(mech["effective_rank_amplitude"]["cur_qkvo"]["p90"])

  er_qkv = np.array(mech["effective_rank_amplitude"]["cur_qkv"]["mean"])
  er_attr = np.array(mech["effective_rank_amplitude"]["attr_qkvo"]["mean"])

  ax1.fill_between(layers, er_qkvo_p10, er_qkvo_p90, color=C_LOOM, alpha=0.14, label="Loom QKVO (P10–P90)")
  ax1.plot(layers, er_qkvo, color=C_LOOM, lw=2.3, marker="o", ms=3.8, markevery=2, label="LoomPythia (all ports QKVO)")
  ax1.plot(layers, er_qkv, color="#8C2D04", lw=1.8, ls="-.", label="LoomPythia (QKV ports)")
  ax1.plot(layers, er_attr, color="#6A3D9A", lw=1.7, ls=":", label="LoomPythia (vector attr. $A^{(l)}$)")

  # Structural rank ceilings of layer-centric baselines
  ax1.axhline(3.0, color="#555555", lw=1.3, ls="--", label="MUDDFormer QKV ceiling (rank $\\leq 3$)")
  ax1.axhline(1.0, color=C_MHC, lw=1.3, ls="--", label="Transformer / mHC / AttnRes ceiling (rank $= 1$)")

  ax1.set_xlabel("Layer Depth $l$", fontsize=12)
  ax1.set_ylabel("Within-Token Effective Wiring Rank", fontsize=12)
  ax1.set_title("(a) Non-Rank-1 Within-Token Circuit Topology", fontsize=12.5, fontweight="bold")
  ax1.set_xlim(1, 32)
  ax1.set_xticks([1, 8, 16, 24, 32])
  ax1.legend(fontsize=8.6, loc="upper left", frameon=True)

  # Panel (b): Cross-token rewiring cosine distance per port
  rew = np.array(mech["cross_token_rewiring"]["attr_cosine_dist"])  # [32, 4]
  for p_i, p_name in enumerate(("Q", "K", "V", "O")):
    ax2.plot(layers, rew[:, p_i], color=C_PORT[p_name], lw=2.1, marker="s", ms=3.3, markevery=3, label=f"Port {p_name}")
  ax2.axhline(0.0, color="#666666", lw=1.1, ls="--", label="Static depth weights ($0$ rewiring)")

  ax2.set_xlabel("Layer Depth $l$", fontsize=12)
  ax2.set_ylabel("Cross-Token Attribution Distance $1 - \\cos$", fontsize=12)
  ax2.set_title("(b) Token-Adaptive Circuit Rewiring by Port", fontsize=12.5, fontweight="bold")
  ax2.set_xlim(1, 32)
  ax2.set_xticks([1, 8, 16, 24, 32])
  ax2.set_ylim(-0.03, min(1.08, max(0.65, float(np.nanmax(rew)) * 1.12)))
  ax2.legend(fontsize=9.2, loc="lower right", frameon=True, ncol=2)

  fig.tight_layout(w_pad=2.2)
  return save_both(fig, out_dir, "sec53_wiring_rank_and_rewiring")


def plot_head_vitalization_and_cosine(loom: dict, mhc: dict, pythia: dict, out_dir: Path) -> tuple[Path, Path]:
  """2-panel matched 3-model comparison on the exact same 32 held-out Pile sequences:
  (a) Layer-wise Attention Head Activation Ratio (Pythia vs mHC-Pythia vs LoomPythia)
  (b) Adjacent-layer input cosine similarity (representation diversity vs collapse).
  """
  layers = np.arange(1, 33)
  adj_x = np.arange(2, 33)

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.3))

  # Panel (a): Attention Head Activation Ratio (`active_paper`)
  y_py = 100.0 * np.array(pythia["head_activation"]["active_paper"]["layer_mean"])
  y_mh = 100.0 * np.array(mhc["head_activation"]["active_paper"]["layer_mean"])
  y_lo = 100.0 * np.array(loom["head_activation"]["active_paper"]["layer_mean"])

  ax1.plot(layers, y_py, color=C_PYTHIA, lw=2.1, marker="o", ms=3.5, markevery=2,
           label=f"Pythia-2.8B (avg {np.mean(y_py):.1f}%)")
  ax1.plot(layers, y_mh, color=C_MHC, lw=2.1, marker="^", ms=4.0, markevery=2,
           label=f"mHC-Pythia-2.8B (avg {np.mean(y_mh):.1f}%)")
  ax1.plot(layers, y_lo, color=C_LOOM, lw=2.4, marker="D", ms=3.8, markevery=2,
           label=f"LoomPythia-2.8B (avg {np.mean(y_lo):.1f}%)")

  ax1.set_xlabel("Layer Depth $l$", fontsize=12)
  ax1.set_ylabel("Active Attention Heads (%)", fontsize=12)
  ax1.set_title("(a) Attention Head Vitalization Across Depth", fontsize=12.5, fontweight="bold")
  ax1.set_xlim(1, 32)
  ax1.set_ylim(0, 102)
  ax1.set_xticks([1, 8, 16, 24, 32])
  ax1.legend(fontsize=9.2, loc="best", frameon=True)

  # Panel (b): Adjacent-layer representation cosine similarity
  ax2.plot(adj_x, pythia["adjacent_layer_cosine"]["residual_in"]["layer_mean"],
           color=C_PYTHIA, lw=2.1, label="Pythia-2.8B (residual)")
  ax2.plot(adj_x, mhc["adjacent_layer_cosine"]["attn_connector_read"]["layer_mean"],
           color=C_MHC, lw=2.1, ls="-.", label="mHC-Pythia-2.8B (connector read)")
  ax2.plot(adj_x, loom["adjacent_layer_cosine"]["residual_in"]["layer_mean"],
           color=C_LOOM, lw=1.7, ls="--", alpha=0.75, label="LoomPythia (backbone residual)")
  for p_key, p_lbl in [("local_q", "Q"), ("local_k", "K"), ("local_v", "V"), ("local_o", "O")]:
    ax2.plot(adj_x, loom["adjacent_layer_cosine"][p_key]["layer_mean"],
             color=C_PORT[p_lbl], lw=2.0, label=f"LoomPythia read {p_lbl}")

  ax2.set_xlabel("Layer Depth $l$", fontsize=12)
  ax2.set_ylabel("Adjacent-Layer Cosine Similarity", fontsize=12)
  ax2.set_title("(b) Decoupled Port Readouts Resist Collapse", fontsize=12.5, fontweight="bold")
  ax2.set_xlim(2, 32)
  ax2.set_xticks([2, 8, 16, 24, 32])
  ax2.legend(fontsize=8.3, loc="lower right", frameon=True, ncol=2)

  fig.tight_layout(w_pad=2.2)
  return save_both(fig, out_dir, "sec53_head_vitalization_and_cosine")


def plot_source_decomposition_and_faithfulness(loom: dict, out_dir: Path) -> tuple[Path, Path]:
  """3-panel mechanistic breakdown:
  (a) Writer-type composition (Embedding vs Attention vs Memory-FFN) across depth
  (b) Source-layer -> Target-layer routing heatmap (averaged over QKVO)
  (c) Top-k attribution faithfulness & causal NLL ablations.
  """
  mech = loom["mechanistic"]
  bs = mech["branch_and_source_shares"]
  layers = np.arange(1, 33)

  fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.4, 4.2))

  # (a) Positive attribution share by writer type (averaged over Q, K, V, O)
  wk = 100.0 * np.mean(np.array(bs["writer_kind_posmass_share"]), axis=1)  # [32, 3]
  ax1.stackplot(
      layers,
      wk[:, 0], wk[:, 1], wk[:, 2],
      labels=["Embedding writers (32)", "Attention writers ($32\\times l$)", "Memory-FFN writers ($32\\times l$)"],
      colors=["#7F7F7F", "#377EB8", "#E41A1C"],
      alpha=0.82,
  )
  mshift_share = 100.0 * np.mean(np.array(bs["mshift_prev_abs_share"]), axis=1)
  ax1.plot(layers, mshift_share, color="#111111", lw=2.0, ls="--", label="Prev-token MShift share (%)")
  ax1.set_xlim(1, 32)
  ax1.set_ylim(0, 100)
  ax1.set_xticks([1, 8, 16, 24, 32])
  ax1.set_xlabel("Reader Layer Depth $l$", fontsize=11.5)
  ax1.set_ylabel("Attribution Mass Share (%)", fontsize=11.5)
  ax1.set_title("(a) Upstream Writer Composition", fontsize=12, fontweight="bold")
  ax1.legend(fontsize=8.2, loc="center right", frameon=True)

  # (b) Source-depth -> Target-depth routing matrix [32, 33] (0=Emb, 1..31=Layer 1..31)
  rout = np.mean(np.array(bs["depth_routing_posmass_32x33x4"]), axis=-1)[:, :32]  # [target 0..31, src 0..31]
  # Row-normalize for crisp depth visibility
  rout_disp = rout / np.maximum(np.max(rout, axis=-1, keepdims=True), 1e-12)
  im = ax2.imshow(rout_disp, origin="lower", aspect="auto", cmap="magma_r", vmin=0.0, vmax=1.0)
  ax2.set_xlabel("Upstream Writer Source (0=Emb, 1..31=Layer)", fontsize=11)
  ax2.set_ylabel("Target Reader Layer $l$", fontsize=11.5)
  ax2.set_xticks([0, 8, 16, 24, 31], ["Emb", "L8", "L16", "L24", "L31"])
  ax2.set_yticks([0, 7, 15, 23, 31], ["L1", "L8", "L16", "L24", "L32"])
  ax2.set_title("(b) Depth-to-Depth Routing Map", fontsize=12, fontweight="bold")
  plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label="Rel. Attribution Intensity")

  # (c) Top-k readout reconstruction cosine across representative depths
  ks = np.array(mech["topk_faithfulness"]["ks"])
  f_cos = np.array(mech["topk_faithfulness"]["cosine_similarity"])  # [32, 4, K]
  f_mean_ports = np.mean(f_cos, axis=1)  # [32, K]
  for l_pick, col, lbl in [(7, "#2B8CBE", "Layer 8 (512 writers)"),
                           (15, "#E6550D", "Layer 16 (1024 writers)"),
                           (31, C_LOOM, "Layer 32 (2016 writers)")]:
    ax3.plot(ks, f_mean_ports[l_pick], marker="o", lw=2.1, color=col, label=lbl)
  ax3.set_xscale("log", base=2)
  ax3.set_xticks(ks, [str(k) for k in ks])
  ax3.set_xlabel("Top-$k$ Upstream Writers Retained", fontsize=11.5)
  ax3.set_ylabel("$\\cos(\\hat{u}^{(k)}, u)$", fontsize=11.5)
  ax3.set_ylim(0.2, 1.03)
  ax3.set_title("(c) Sparse Top-$k$ Circuit Faithfulness", fontsize=12, fontweight="bold")
  ax3.legend(fontsize=8.8, loc="lower right", frameon=True)

  fig.tight_layout(w_pad=2.0)
  return save_both(fig, out_dir, "sec53_source_routing_and_faithfulness")


def plot_icl_microcircuit(loom: dict, icl_npz: Path, out_dir: Path) -> tuple[Path, Path]:
  """Render concrete non-rectangular writer x reader submatrix on a real ICL prompt
  at two distinct token roles (e.g. delimiter ':' vs entity 'Germany') to show both:
  (1) Non-rank-1 / non-rectangular within-token circuit structure across heads/ports, and
  (2) Token-dependent circuit rewiring on the exact same layer!
  """
  data = np.load(icl_npz)
  prompts = {p["name"]: p for p in loom["icl_prompts"]["prompts"]}
  p_name = "country_capital" if "country_capital" in prompts else loom["icl_prompts"]["prompts"][0]["name"]
  p_info = prompts[p_name]
  toks = p_info["tokens"]

  w_oc = data[f"{p_name}__writers_o_cur"].astype(np.float32)   # [T, 2016, 80]
  w_kc = data[f"{p_name}__writers_k_cur"].astype(np.float32)   # [T, 2016, 64]
  w_op = data[f"{p_name}__writers_o_prev"].astype(np.float32)
  w_kp = data[f"{p_name}__writers_k_prev"].astype(np.float32)
  z_l = data[f"{p_name}__layer_z"].astype(np.float32)          # [32, T, 64]
  m_rms = data[f"{p_name}__layer_m_rms"].astype(np.float32)    # [32, T]
  q_tot = (data[f"{p_name}__layer_q_dyn"].astype(np.float32)
           + data[f"{p_name}__layer_q_stat"].astype(np.float32))  # [32, T, 4, 32, 64]

  # Pick a deep layer with high rewiring + high effective rank (e.g. Layer 24, index 23)
  l_idx = 23
  w_cnt = 32 + 64 * l_idx
  # Pick final ':' token (`t_b = T - 1`) and the preceding country entity token (`t_a = T - 2`)
  t_b = len(toks) - 1
  t_a = max(1, len(toks) - 2)

  def compute_attr_at(t_pos: int) -> np.ndarray:
    zl = z_l[l_idx, t_pos] if t_pos > 0 else np.zeros((64,), dtype=np.float32)
    kc = w_kc[t_pos, :w_cnt] * (1.0 - zl)[None, :]
    kp = w_kp[t_pos, :w_cnt] * zl[None, :]
    oc = w_oc[t_pos, :w_cnt]
    op = w_op[t_pos, :w_cnt]
    qs = q_tot[l_idx, t_pos] / m_rms[l_idx, t_pos]  # [4, 32, 64]
    cc = np.einsum("snv,wv->wsn", qs, kc)
    cp = np.einsum("snv,wv->wsn", qs, kp)
    u_rec = np.einsum("wsn,wk->snk", cc, oc) + np.einsum("wsn,wk->snk", cp, op)
    un2 = np.maximum(np.sum(u_rec * u_rec, axis=-1), 1e-12)
    ac = (cc * np.einsum("wk,snk->wsn", oc, u_rec)) / un2[None, :, :]
    ap = (cp * np.einsum("wk,snk->wsn", op, u_rec)) / un2[None, :, :]
    return (ac + ap).reshape(w_cnt, 128)  # [W_l, 128]

  A_a = compute_attr_at(t_a)
  A_b = compute_attr_at(t_b)

  # Select top-20 most active writers & top-24 readers (6 per port Q, K, V, O) by joint variance/magnitude
  r_cols = []
  r_labels = []
  for s_i, s_nm in enumerate(("Q", "K", "V", "O")):
    sub_score = np.max(np.abs(A_a[:, s_i * 32:(s_i + 1) * 32]), axis=0) + np.max(np.abs(A_b[:, s_i * 32:(s_i + 1) * 32]), axis=0)
    top_h = np.argsort(-sub_score)[:6]
    for h in np.sort(top_h):
      r_cols.append(s_i * 32 + int(h))
      r_labels.append(f"{s_nm}{int(h)}")

  w_score = np.max(np.abs(A_a[:, r_cols]), axis=1) + np.max(np.abs(A_b[:, r_cols]), axis=1)
  top_w = np.sort(np.argsort(-w_score)[:18])

  def writer_label(w_id: int) -> str:
    if w_id < 32:
      return f"Emb.{w_id}"
    rem = w_id - 32
    sl = rem // 64 + 1
    rem2 = rem % 64
    if rem2 < 32:
      return f"L{sl}.Attn{rem2}"
    return f"L{sl}.FFN{rem2 - 32}"

  w_labels = [writer_label(int(w)) for w in top_w]
  sub_a = A_a[np.ix_(top_w, r_cols)]
  sub_b = A_b[np.ix_(top_w, r_cols)]
  vmax = float(np.percentile(np.abs(np.concatenate([sub_a.ravel(), sub_b.ravel()])), 98.5))
  vmax = max(vmax, 0.05)

  fig = plt.figure(figsize=(12.6, 4.8))
  gs = gridspec.GridSpec(1, 3, width_ratios=[1.0, 1.0, 0.05], wspace=0.22)
  ax_a = fig.add_subplot(gs[0, 0])
  ax_b = fig.add_subplot(gs[0, 1])
  cax = fig.add_subplot(gs[0, 2])

  tok_a_str = repr(toks[t_a]["str"])
  tok_b_str = repr(toks[t_b]["str"])
  im = ax_a.imshow(sub_a, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
  ax_b.imshow(sub_b, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)

  for ax, ttl in [
      (ax_a, f"(a) Layer {l_idx+1} Circuit at Token pos {t_a}: {tok_a_str}"),
      (ax_b, f"(b) Layer {l_idx+1} Rewired at Token pos {t_b}: {tok_b_str}"),
  ]:
    ax.set_xticks(np.arange(len(r_cols)), r_labels, rotation=60, fontsize=7.8)
    ax.set_yticks(np.arange(len(top_w)), w_labels, fontsize=8.0)
    for boundary in (5.5, 11.5, 17.5):
      ax.axvline(boundary, color="#222222", lw=0.9, ls="--")
    ax.set_title(ttl, fontsize=11.0, fontweight="bold")
    ax.set_xlabel("Layer 24 Reader Ports (Q | K | V | O)", fontsize=10.5)
  ax_a.set_ylabel("Upstream Node Writers (Emb / Attn / MemFFN)", fontsize=10.5)

  plt.colorbar(im, cax=cax, label="Exact Attribution Weight $A_{ij}(t)$")
  top1_pred = p_info["top5_next"][0]
  fig.suptitle(
      f"ICL Micro-Circuit Rewiring on {p_name} (Next-token top-1: {repr(top1_pred['token'])}, p={top1_pred['prob']:.2f})",
      fontsize=12.0, fontweight="bold", y=1.02,
  )
  return save_both(fig, out_dir, "sec53_icl_microcircuit_rewiring")


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--data-dir", required=True)
  ap.add_argument("--out-dir", required=True)
  args = ap.parse_args()

  data_dir = Path(args.data_dir)
  out_dir = Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  setup_style()

  loom = json.loads((data_dir / "loom_pythia_2p8b_sec53.json").read_text())
  mhc = json.loads((data_dir / "mhc_pythia_2p8b_sec53.json").read_text())
  pythia = json.loads((data_dir / "pythia_2p8b_sec53.json").read_text())
  icl_npz = data_dir / "loom_icl_circuits.npz"

  for fn in (
      lambda: plot_wiring_rank_and_rewiring(loom, out_dir),
      lambda: plot_head_vitalization_and_cosine(loom, mhc, pythia, out_dir),
      lambda: plot_source_decomposition_and_faithfulness(loom, out_dir),
      lambda: plot_icl_microcircuit(loom, icl_npz, out_dir),
  ):
    pdf_p, png_p = fn()
    print(f"SAVED {pdf_p} & {png_p}", flush=True)


if __name__ == "__main__":
  main()
