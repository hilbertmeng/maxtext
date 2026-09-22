"""Unfiltered read-gate / local-O-share / write-gate analysis.

Uses every recorded position with a nonzero local-O vector.  Neither the read-O
gate nor the write gate is used as a selection criterion.
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(sys.argv[1])
out = root / "unfiltered_gate_share"
out.mkdir(exist_ok=True)
meta = json.loads((root / "metadata.json").read_text())
ix = {x: i for i, x in enumerate(meta["fields"])}
valid = np.load(root / "valid.npy")
n = int(meta["n"])
layers = range(1, 24)  # L0 has zero M and is not a meaningful O share.

def q(x):
    x = x[np.isfinite(x)]
    return np.quantile(x, [0.01, .05, .1, .25, .5, .75, .9, .95, .99]).tolist()

summary = {"n": n, "selection": "valid tokens and gram_33>1e-12 only; no read/write gate filtering", "layers": []}
all_read, all_share, all_write, all_layer, all_head = [], [], [], [], []
for l in layers:
    raw = np.stack([np.load(root / f"sample_{i:03d}.npy", mmap_mode="r")[l] for i in range(n)])
    no = np.sqrt(np.maximum(raw[..., ix["gram_33"]], 0))
    norms = np.sqrt(np.maximum(np.stack([raw[..., ix[f"gram_{j}{j}"]] for j in range(4)], -1), 0))
    share = no / np.maximum(norms.sum(-1), 1e-20)
    rg = raw[..., ix["read_o_gate"]]
    wg = raw[..., ix["write_gate"]]
    mask = valid[:n, :, None] & (no > 1e-6) & np.isfinite(rg) & np.isfinite(wg)
    layer = {"layer": l, "tokens": int(mask.sum()), "read_gate": q(rg[mask]), "local_o_share": q(share[mask]), "write_gate": q(wg[mask])}
    byhead = []
    for h in range(16):
        m = mask[:, :, h]
        x, y, z = rg[:, :, h][m], share[:, :, h][m], wg[:, :, h][m]
        if len(x) < 30:
            byhead.append({"head": h, "tokens": int(len(x))})
            continue
        byhead.append({"head": h, "tokens": int(len(x)), "rho_read_share": float(spearmanr(x, y).statistic), "rho_read_write": float(spearmanr(x, z).statistic), "rho_share_write": float(spearmanr(y, z).statistic), "read_gate": q(x), "local_o_share": q(y), "write_gate": q(z)})
        # Keep a bounded, stratified reservoir for pooled plots; do not retain all
        # ~20M positions in RAM.
        keep = min(len(x), 2000)
        all_read.append(x[:keep]); all_share.append(y[:keep]); all_write.append(z[:keep]); all_layer.append(np.full(keep, l)); all_head.append(np.full(keep, h))
    layer["heads"] = byhead
    summary["layers"].append(layer)

R, S, W = np.concatenate(all_read), np.concatenate(all_share), np.concatenate(all_write)
L, H = np.concatenate(all_layer), np.concatenate(all_head)
summary["pooled_L1_L23"] = {"tokens": int(len(R)), "read_gate": q(R), "local_o_share": q(S), "write_gate": q(W), "rho_read_share": float(spearmanr(R, S).statistic), "rho_read_write": float(spearmanr(R, W).statistic), "rho_share_write": float(spearmanr(S, W).statistic)}

# Layer/head demeaned rank correlations: pooled correlations can be driven by depth.
def within_group(a, b, groups):
    ar, br = [], []
    for g in np.unique(groups):
        m = groups == g
        if m.sum() >= 30:
            ar.append(a[m]); br.append(b[m])
    return float(spearmanr(np.concatenate(ar), np.concatenate(br)).statistic)
summary["within_layer"] = {"rho_read_share": within_group(R, S, L), "rho_read_write": within_group(R, W, L), "rho_share_write": within_group(S, W, L)}
summary["within_layer_head"] = {"rho_read_share": within_group(R, S, L * 16 + H), "rho_read_write": within_group(R, W, L * 16 + H), "rho_share_write": within_group(S, W, L * 16 + H)}

# Quantile heatmaps: write gate mean by read-gate decile x local-O-share decile,
# retaining all cells and no write-gate selection.
heat = np.full((24, 16, 10, 10), np.nan)
for l in layers:
    raw = np.stack([np.load(root / f"sample_{i:03d}.npy", mmap_mode="r")[l] for i in range(n)])
    no = np.sqrt(np.maximum(raw[..., ix["gram_33"]], 0)); norms = np.sqrt(np.maximum(np.stack([raw[..., ix[f"gram_{j}{j}"]] for j in range(4)], -1), 0)); share = no / np.maximum(norms.sum(-1), 1e-20)
    rg, wg = raw[..., ix["read_o_gate"]], raw[..., ix["write_gate"]]
    mask = valid[:n, :, None] & (no > 1e-6)
    for h in range(16):
        m = mask[:, :, h]; x, y, z = rg[:, :, h][m], share[:, :, h][m], wg[:, :, h][m]
        if len(x) < 100: continue
        bx, by = np.quantile(x, np.linspace(0, 1, 11)), np.quantile(y, np.linspace(0, 1, 11))
        xi = np.clip(np.searchsorted(bx[1:-1], x, side="right"), 0, 9); yi = np.clip(np.searchsorted(by[1:-1], y, side="right"), 0, 9)
        for a in range(10):
            for b in range(10):
                take = (xi == a) & (yi == b)
                if take.sum() >= 20: heat[l, h, a, b] = z[take].mean()
np.savez_compressed(out / "unfiltered_gate_share.npz", read_gate=R, local_o_share=S, write_gate=W, layer=L, head=H, heat=heat)

fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
ax[0].hexbin(R, S, C=W, reduce_C_function=np.mean, gridsize=50, mincnt=20, cmap="viridis"); ax[0].set(xlabel="local-O read gate", ylabel="local-O norm share", title="Mean write gate")
ax[1].hexbin(R, W, gridsize=50, mincnt=20, cmap="magma"); ax[1].set(xlabel="local-O read gate", ylabel="write gate", title="All nonzero local-O positions")
ax[2].hexbin(S, W, gridsize=50, mincnt=20, cmap="magma"); ax[2].set(xlabel="local-O norm share", ylabel="write gate", title="All nonzero local-O positions")
fig.tight_layout(); fig.savefig(out / "pooled_relationships.png", dpi=180); plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
for name, key, color in [("read→share", "rho_read_share", "#1976b8"), ("read→write", "rho_read_write", "#e56b20"), ("share→write", "rho_share_write", "#32934a")]:
    med = []
    lo = []
    hi = []
    for l in layers:
        vals = [h[key] for h in summary["layers"][l-1]["heads"] if key in h and np.isfinite(h[key])]
        med.append(np.median(vals) if vals else np.nan); lo.append(np.quantile(vals, .1) if vals else np.nan); hi.append(np.quantile(vals, .9) if vals else np.nan)
    ax[0].plot(list(layers), med, "o-", ms=3, color=color, label=name); ax[0].fill_between(list(layers), lo, hi, color=color, alpha=.12)
ax[0].axhline(0, color="gray", lw=.8); ax[0].set(xlabel="Layer", ylabel="Head-wise Spearman rho", title="Per-head relationships"); ax[0].legend(); ax[0].grid(alpha=.2)
for h in [0, 4, 12, 15]:
    vals = np.nanmedian(heat[1:, h], axis=0); im = ax[1].imshow(vals, vmin=.05, vmax=.8, origin="lower", aspect="auto", cmap="magma"); ax[1].set_title(f"Mean write gate, H{h}\n(read decile × O-share decile)"); ax[1].set_xlabel("O-share decile"); ax[1].set_ylabel("read-gate decile")
fig.colorbar(im, ax=ax[1], label="write gate")
ax[2].axis("off"); fig.tight_layout(); fig.savefig(out / "layer_head_relationships.png", dpi=180); plt.close(fig)
(out / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary["pooled_L1_L23"], indent=2))
print(json.dumps(summary["within_layer"], indent=2))
print(json.dumps(summary["within_layer_head"], indent=2))
