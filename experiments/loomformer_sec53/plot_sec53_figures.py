"""Generate ICML 2027 camera-ready vector PDF + PNG figures for §5.3 & Appendix.

Outputs:
  1. `loompythia_analysis_main.pdf` (double-column 1x4 `figure*` for §5.3):
     (a) Within-token effective wiring rank of Q/K/V readers vs. layer depth
     (b) Cross-token circuit rewiring cosine distance by port vs. layer depth
     (c) Fraction of attention heads `on` (%) across depth (Pythia vs. mHC vs. LoomPythia)
     (d) Adjacent-layer input cosine similarity across depth (Pythia vs. mHC vs. LoomPythia)
  2. `loompythia_routing_appendix.pdf` (double-column 1x3 `figure*` for Appendix):
     (a) Upstream writer composition (Embedding, LoomMHA, FFN_M) & MShift share
     (b) 32x32 source-to-target layer routing map
     (c) Sparse top-k writer reconstruction cosine similarity
  3. `loompythia_icl_circuit_appendix.pdf` (double-column 1x2 `figure*` for Appendix):
     Exact Layer-24 writer-to-reader attribution submatrix rewiring from `'Germany'` to `':'`
     on the in-context `country_capital` prompt (top-1 prediction `' Berlin'`, p=0.65).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


C_PYTHIA = "#222222"
C_MHC = "#4393C3"
C_LOOM = "#D62728"
C_PORT = {
    "Q": "#E69F00",  # amber
    "K": "#D55E00",  # vermilion
    "V": "#009E73",  # bluish green
    "O": "#0072B2",  # ocean blue
}


def setup_style() -> None:
  plt.rcParams.update({
      "font.family": "sans-serif",
      "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
      "mathtext.fontset": "dejavusans",
      "axes.edgecolor": "#333333",
      "axes.linewidth": 0.85,
      "axes.grid": True,
      "grid.alpha": 0.22,
      "grid.linestyle": "--",
      "grid.linewidth": 0.55,
      "pdf.fonttype": 42,
      "ps.fonttype": 42,
  })


def save_both(fig: plt.Figure, out_dir: Path, stem: str) -> tuple[Path, Path]:
  pdf_p = out_dir / f"{stem}.pdf"
  png_p = out_dir / f"{stem}.png"
  fig.savefig(pdf_p, bbox_inches="tight", pad_inches=0.03)
  fig.savefig(png_p, dpi=240, bbox_inches="tight", pad_inches=0.03)
  plt.close(fig)
  return pdf_p, png_p


def plot_main_1x4(loom: dict, mhc: dict, pythia: dict, out_dir: Path) -> tuple[Path, Path]:
  """Unified 1x4 double-column strip sharing X-axis = Layer l (1..32)."""
  mech = loom["mechanistic"]
  layers = np.arange(1, 33)
  adj_x = np.arange(2, 33)

  fig, axes = plt.subplots(1, 4, figsize=(13.8, 2.95))
  ax1, ax2, ax3, ax4 = axes

  # --- (a) Within-token effective wiring rank (Q/K/V readers, matching Table 2) ---
  er_qkv = np.array(mech["effective_rank_amplitude"]["cur_qkv"]["mean"])
  er_qkv_p10 = np.array(mech["effective_rank_amplitude"]["cur_qkv"]["p10"])
  er_qkv_p90 = np.array(mech["effective_rank_amplitude"]["cur_qkv"]["p90"])
  er_single = np.mean([
      np.array(mech["effective_rank_amplitude"]["port_Q"]["mean"]),
      np.array(mech["effective_rank_amplitude"]["port_K"]["mean"]),
      np.array(mech["effective_rank_amplitude"]["port_V"]["mean"]),
  ], axis=0)

  ax1.fill_between(layers, er_qkv_p10, er_qkv_p90, color=C_LOOM, alpha=0.14)
  ax1.plot(layers, er_qkv, color=C_LOOM, lw=2.1, marker="o", ms=3.0, markevery=3,
           label="LoomPythia (Q/K/V)")
  ax1.plot(layers, er_single, color="#8C2D04", lw=1.7, ls="-.",
           label="LoomPythia (per Q,K,V)")
  ax1.axhline(3.0, color="#666666", lw=1.2, ls="--", label="MUDDFormer ($\\leq 3$)")
  ax1.axhline(1.0, color=C_MHC, lw=1.2, ls=":", label="Tfm / mHC ($1$)")

  ax1.set_title("(a) Wiring Rank", fontsize=10.8, fontweight="bold", pad=5)
  ax1.set_xlabel("Layer $l$", fontsize=10.2)
  ax1.set_ylabel("Effective Rank", fontsize=10.2)
  ax1.set_xlim(1, 32)
  ax1.set_ylim(0, 56)
  ax1.set_xticks([1, 8, 16, 24, 32])
  ax1.tick_params(labelsize=8.8)
  ax1.legend(fontsize=7.3, loc="upper left", frameon=True, handlelength=1.6, borderpad=0.3)

  # --- (b) Cross-token circuit rewiring cosine distance ---
  rew = np.array(mech["cross_token_rewiring"]["attr_cosine_dist"])  # [32, 4]
  port_disp = [("Q", "$\\tilde{q}$"), ("K", "$\\tilde{k}$"), ("V", "$v$"), ("O", "$o^m$")]
  for p_i, (p_key, p_lbl) in enumerate(port_disp):
    ax2.plot(layers, rew[:, p_i], color=C_PORT[p_key], lw=1.9, marker="s", ms=2.7, markevery=4, label=p_lbl)
  ax2.axhline(0.0, color="#666666", lw=1.1, ls="--", label="Static ($0$)")

  ax2.set_title("(b) Per-Token Rewiring", fontsize=10.8, fontweight="bold", pad=5)
  ax2.set_xlabel("Layer $l$", fontsize=10.2)
  ax2.set_ylabel("Attribution Dist. ($1 - \\cos$)", fontsize=10.2)
  ax2.set_xlim(1, 32)
  ax2.set_ylim(-0.04, 1.02)
  ax2.set_xticks([1, 8, 16, 24, 32])
  ax2.tick_params(labelsize=8.8)
  ax2.legend(fontsize=7.5, loc="lower right", frameon=True, ncol=2, handlelength=1.5, columnspacing=0.7, borderpad=0.3)

  # --- (c) Fraction of heads `on` (%) across depth ---
  y_py = 100.0 * np.array(pythia["head_activation"]["active_paper"]["layer_mean"])
  y_mh = 100.0 * np.array(mhc["head_activation"]["active_paper"]["layer_mean"])
  y_lo = 100.0 * np.array(loom["head_activation"]["active_paper"]["layer_mean"])

  ax3.plot(layers, y_py, color=C_PYTHIA, lw=1.8, marker="o", ms=2.8, markevery=3,
           label=f"Pythia ({np.mean(y_py):.1f}%)")
  ax3.plot(layers, y_mh, color=C_MHC, lw=1.8, marker="^", ms=3.2, markevery=3,
           label=f"mHC-Pythia ({np.mean(y_mh):.1f}%)")
  ax3.plot(layers, y_lo, color=C_LOOM, lw=2.2, marker="D", ms=3.0, markevery=3,
           label=f"LoomPythia ({np.mean(y_lo):.1f}%)")

  ax3.set_title("(c) Heads On", fontsize=10.8, fontweight="bold", pad=5)
  ax3.set_xlabel("Layer $l$", fontsize=10.2)
  ax3.set_ylabel("Heads On (%)", fontsize=10.2)
  ax3.set_xlim(1, 32)
  ax3.set_ylim(0, 102)
  ax3.set_xticks([1, 8, 16, 24, 32])
  ax3.tick_params(labelsize=8.8)
  ax3.legend(fontsize=7.5, loc="center right", frameon=True, handlelength=1.5, borderpad=0.3)

  # --- (d) Adjacent-layer cosine similarity ---
  ax4.plot(adj_x, pythia["adjacent_layer_cosine"]["residual_in"]["layer_mean"],
           color=C_PYTHIA, lw=1.9, label="Pythia ($x$)")
  ax4.plot(adj_x, mhc["adjacent_layer_cosine"]["attn_connector_read"]["layer_mean"],
           color=C_MHC, lw=1.8, ls="-.", label="mHC (read)")
  ax4.plot(adj_x, loom["adjacent_layer_cosine"]["residual_in"]["layer_mean"],
           color=C_LOOM, lw=1.6, ls="--", alpha=0.85, label="Loom ($x$)")
  for p_field, p_key, p_lbl in [
      ("local_q", "Q", "$\\tilde{q}$"),
      ("local_k", "K", "$\\tilde{k}$"),
      ("local_v", "V", "$v$"),
      ("local_o", "O", "$o^m$"),
  ]:
    ax4.plot(adj_x, loom["adjacent_layer_cosine"][p_field]["layer_mean"],
             color=C_PORT[p_key], lw=1.7, label=p_lbl)

  ax4.set_title("(d) Adjacent-Layer Cosine", fontsize=10.8, fontweight="bold", pad=5)
  ax4.set_xlabel("Layer $l$", fontsize=10.2)
  ax4.set_ylabel("Cosine Similarity", fontsize=10.2)
  ax4.set_xlim(1, 32)
  ax4.set_ylim(-0.18, 1.04)
  ax4.set_xticks([1, 8, 16, 24, 32])
  ax4.tick_params(labelsize=8.8)
  ax4.legend(fontsize=7.3, loc="center right", frameon=True, ncol=2, handlelength=1.5, columnspacing=0.7, borderpad=0.3)

  fig.tight_layout(w_pad=1.35)
  return save_both(fig, out_dir, "loompythia_analysis_main")


def plot_appendix_routing_1x3(loom: dict, out_dir: Path) -> tuple[Path, Path]:
  """Appendix 1x3 figure: writer composition, writer-by-reader depth map (matching Fig. 3 axes), top-k faithfulness."""
  mech = loom["mechanistic"]
  bs = mech["branch_and_source_shares"]
  layers = np.arange(1, 33)

  fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.6, 3.65))

  wk = 100.0 * np.mean(np.array(bs["writer_kind_posmass_share"]), axis=1)  # [32, 3]
  ax1.stackplot(
      layers,
      wk[:, 0], wk[:, 1], wk[:, 2],
      labels=["Embedding", "LoomMHA", "$\\mathrm{FFN}_M$"],
      colors=["#7F7F7F", "#377EB8", "#E41A1C"],
      alpha=0.82,
  )
  mshift_share = 100.0 * np.mean(np.array(bs["mshift_prev_abs_share"]), axis=1)
  ax1.plot(layers, mshift_share, color="#111111", lw=1.9, ls="--", label="MShift ($M_{[t-1]}$)")
  ax1.set_xlim(1, 32)
  ax1.set_ylim(0, 100)
  ax1.set_xticks([1, 8, 16, 24, 32])
  ax1.set_xlabel("Reader Layer $l$", fontsize=10.8)
  ax1.set_ylabel("Attribution Share (%)", fontsize=10.8)
  ax1.set_title("(a) Writer Composition", fontsize=11.2, fontweight="bold")
  ax1.legend(fontsize=8.2, loc="center right", frameon=True)

  # Rows (Y, bottom-to-top) = Writer source (Emb, L1..L31);
  # Columns (X, left-to-right) = Reader layer (L1..L32). Causal pairs satisfy writer_row <= reader_col.
  rout_rl_ws = np.mean(np.array(bs["depth_routing_posmass_32x33x4"]), axis=-1)[:, :32]  # [reader 0..31, writer 0..31]
  mat_wr = rout_rl_ws.T  # [writer 0..31, reader 0..31]
  mat_norm = mat_wr / np.maximum(np.max(mat_wr, axis=0, keepdims=True), 1e-12)
  unreachable = np.tril(np.ones_like(mat_norm, dtype=bool), k=-1)  # writer_row > reader_col (upper-left when origin="lower")
  mat_masked = np.ma.array(mat_norm, mask=unreachable)
  cmap = plt.cm.YlOrRd.copy()
  cmap.set_bad(color="#E5E5E5")

  ax2.grid(False)
  im = ax2.imshow(mat_masked, origin="lower", aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
  ax2.set_xlabel("Reader Layer $l$", fontsize=10.8)
  ax2.set_ylabel("Writer Source", fontsize=10.8)
  ax2.set_xticks([0, 7, 15, 23, 31], ["L1", "L8", "L16", "L24", "L32"])
  ax2.set_yticks([0, 8, 16, 24, 31], ["Emb", "L8", "L16", "L24", "L31"])
  ax2.set_title("(b) Depth-to-Depth Routing", fontsize=11.2, fontweight="bold")
  plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label="Col.-Norm. Attribution")

  ks = np.array(mech["topk_faithfulness"]["ks"])
  f_cos = np.mean(np.array(mech["topk_faithfulness"]["cosine_similarity"]), axis=1)
  for l_pick, col, lbl in [
      (7, "#2B8CBE", "Layer 8 ($512$ writers)"),
      (15, "#E6550D", "Layer 16 ($1024$ writers)"),
      (31, C_LOOM, "Layer 32 ($2016$ writers)"),
  ]:
    ax3.plot(ks, f_cos[l_pick], marker="o", ms=4.0, lw=2.0, color=col, label=lbl)
  ax3.set_xscale("log", base=2)
  ax3.set_xticks(ks, [str(k) for k in ks])
  ax3.set_xlabel("Top-$k$ Writers Retained", fontsize=10.8)
  ax3.set_ylabel("$\\cos(\\hat{u}^{(k)}, u)$", fontsize=10.8)
  ax3.set_ylim(0.45, 0.94)
  ax3.set_title("(c) Sparse Top-$k$ Faithfulness", fontsize=11.2, fontweight="bold")
  ax3.legend(fontsize=8.5, loc="lower right", frameon=True)

  fig.tight_layout(w_pad=1.8)
  return save_both(fig, out_dir, "loompythia_routing_appendix")


def plot_appendix_icl_circuit(loom: dict, icl_npz: Path, out_dir: Path) -> tuple[Path, Path]:
  """Appendix 1x2 figure: exact Layer-24 circuit rewiring across adjacent ICL tokens (1-indexed heads)."""
  data = np.load(icl_npz)
  prompts = {p["name"]: p for p in loom["icl_prompts"]["prompts"]}
  p_name = "country_capital"
  p_info = prompts[p_name]
  toks = p_info["tokens"]

  w_oc = data[f"{p_name}__writers_o_cur"].astype(np.float32)
  w_kc = data[f"{p_name}__writers_k_cur"].astype(np.float32)
  w_op = data[f"{p_name}__writers_o_prev"].astype(np.float32)
  w_kp = data[f"{p_name}__writers_k_prev"].astype(np.float32)
  z_l = data[f"{p_name}__layer_z"].astype(np.float32)
  m_rms = data[f"{p_name}__layer_m_rms"].astype(np.float32)
  q_tot = (data[f"{p_name}__layer_q_dyn"].astype(np.float32)
           + data[f"{p_name}__layer_q_stat"].astype(np.float32))

  l_idx = 23  # Layer 24 (1-indexed)
  w_cnt = 32 + 64 * l_idx
  t_b = len(toks) - 1
  t_a = t_b - 1

  def compute_attr_at(t_pos: int) -> np.ndarray:
    zl = z_l[l_idx, t_pos] if t_pos > 0 else np.zeros((64,), dtype=np.float32)
    kc = w_kc[t_pos, :w_cnt] * (1.0 - zl)[None, :]
    kp = w_kp[t_pos, :w_cnt] * zl[None, :]
    oc = w_oc[t_pos, :w_cnt]
    op = w_op[t_pos, :w_cnt]
    qs = q_tot[l_idx, t_pos] / m_rms[l_idx, t_pos]
    cc = np.einsum("snv,wv->wsn", qs, kc)
    cp = np.einsum("snv,wv->wsn", qs, kp)
    u_rec = np.einsum("wsn,wk->snk", cc, oc) + np.einsum("wsn,wk->snk", cp, op)
    un2 = np.maximum(np.sum(u_rec * u_rec, axis=-1), 1e-12)
    ac = (cc * np.einsum("wk,snk->wsn", oc, u_rec)) / un2[None, :, :]
    ap = (cp * np.einsum("wk,snk->wsn", op, u_rec)) / un2[None, :, :]
    return (ac + ap).reshape(w_cnt, 128)

  A_a = compute_attr_at(t_a)
  A_b = compute_attr_at(t_b)

  r_cols, r_labels = [], []
  for s_i, s_nm in enumerate(("Q", "K", "V", "O")):
    sub_score = (np.max(np.abs(A_a[:, s_i * 32:(s_i + 1) * 32]), axis=0)
                 + np.max(np.abs(A_b[:, s_i * 32:(s_i + 1) * 32]), axis=0))
    top_h = np.sort(np.argsort(-sub_score)[:6])
    for h in top_h:
      r_cols.append(s_i * 32 + int(h))
      r_labels.append(f"{s_nm}{int(h) + 1:02d}")  # Fixed-width 2-digit 1-indexed heads

  # Restrict rows to earlier attention-head writers (L{ll}.H{hh}), matching §2 & App. A.7
  head_w_ids = np.array([w for w in range(32, w_cnt) if (w - 32) % 64 < 32], dtype=np.int32)
  w_score = np.max(np.abs(A_a[np.ix_(head_w_ids, r_cols)]), axis=1) + np.max(np.abs(A_b[np.ix_(head_w_ids, r_cols)]), axis=1)
  top_w = head_w_ids[np.sort(np.argsort(-w_score)[:18])]

  def writer_label(w_id: int) -> str:
    rem = w_id - 32
    sl = rem // 64 + 1
    h_idx = rem % 64 + 1
    return f"L{sl:02d}.H{h_idx:02d}"

  w_labels = [writer_label(int(w)) for w in top_w]
  sub_a = A_a[np.ix_(top_w, r_cols)]
  sub_b = A_b[np.ix_(top_w, r_cols)]
  vmax = max(float(np.percentile(np.abs(np.concatenate([sub_a.ravel(), sub_b.ravel()])), 98.5)), 0.05)

  fig = plt.figure(figsize=(13.2, 4.35))
  # Independent horizontal spacers: roomy gap for (b)'s y-tick labels, tight gap before colorbar
  gs = gridspec.GridSpec(1, 5, width_ratios=[1.0, 0.165, 1.0, 0.105, 0.032], wspace=0.0)
  ax_a = fig.add_subplot(gs[0, 0])
  ax_b = fig.add_subplot(gs[0, 2])
  cax = fig.add_subplot(gs[0, 4])

  tok_a_str = toks[t_a]["str"].strip() or toks[t_a]["str"]
  tok_b_str = toks[t_b]["str"]
  for ax in (ax_a, ax_b):
    ax.grid(False)
  im = ax_a.imshow(sub_a, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
  ax_b.imshow(sub_b, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)

  for ax, ttl in [
      (ax_a, f'(a) Layer 24 Wiring at "{tok_a_str}"'),
      (ax_b, f'(b) Layer 24 Rewired at "{tok_b_str}"'),
  ]:
    ax.set_xticks(np.arange(len(r_cols)), r_labels, rotation=55, fontsize=8.0)
    ax.set_yticks(np.arange(len(top_w)), w_labels, fontsize=8.2)
    for boundary in (5.5, 11.5, 17.5):
      ax.axvline(boundary, color="#222222", lw=0.9, ls="--")
    ax.set_title(ttl, fontsize=11.0, fontweight="bold")
    ax.set_xlabel("Layer 24 MRead Outputs ($\\tilde{q}$ | $\\tilde{k}$ | $v$ | $o^m$)", fontsize=10.3)
  ax_a.set_ylabel("Earlier Head Writers", fontsize=10.5)

  plt.colorbar(im, cax=cax, label="Attribution $a_{ij}$")
  return save_both(fig, out_dir, "loompythia_icl_circuit_appendix")


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
      lambda: plot_main_1x4(loom, mhc, pythia, out_dir),
      lambda: plot_appendix_routing_1x3(loom, out_dir),
      lambda: plot_appendix_icl_circuit(loom, icl_npz, out_dir),
  ):
    pdf_p, png_p = fn()
    print(f"SAVED {pdf_p} & {png_p}", flush=True)


if __name__ == "__main__":
  main()
