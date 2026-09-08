"""Aggregate scalar gate captures, preserving per-sequence sufficient statistics."""
from pathlib import Path
import json
import sys
import hashlib
import numpy as np

LAYERS = [i for i in range(24) if i % 3 != 2]


def plot_summary(root, report):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig, axes = plt.subplots(1,2,figsize=(10,3.6),constrained_layout=True)
  active = [l for l in LAYERS if l > 0]
  for side in ('row','col'):
    axes[0].plot(active,[report['layers'][str(l)][side]['gate_corr_head_centered'] for l in active],
                 'o-',label=f'O vs V: {side}')
  for dst in ('O','V'):
    axes[1].plot(active,[report['layers'][str(l)][f'{dst}_row_col_gate_corr_head_centered'] for l in active],
                 'o-',label=f'Local{dst}: row vs col')
  for ax in axes:
    ax.axhline(0,color='gray',lw=.8)
    ax.set(xlabel='Layer (zero-based)',ylabel='Head-centered gate correlation',ylim=(-.15,.6))
    ax.legend(loc='upper right'); ax.grid(alpha=.2)
  fig.savefig(root/'gate_correlation_depth.png',dpi=160);plt.close(fig)
  fig,axes=plt.subplots(1,4,figsize=(13,3.5),constrained_layout=True)
  # Raw order head/side/destination -> group order O-row/O-col/V-row/V-col.
  order=np.concatenate([np.arange(16)*4+i for i in (0,2,1,3)])
  for ax,l in zip(axes,(3,12,21,22)):
    cc=np.asarray(report['layers'][str(l)]['full_gate_correlation'])
    im=ax.imshow(cc[np.ix_(order,order)],cmap='RdBu_r',vmin=-1,vmax=1)
    ax.set_title(f'L{l}')
    ax.set_xticks([7.5,23.5,39.5,55.5],['O-row','O-col','V-row','V-col'],rotation=45)
    ax.set_yticks([7.5,23.5,39.5,55.5],['O-row','O-col','V-row','V-col'])
    for edge in (15.5,31.5,47.5):
      ax.axhline(edge,color='gray',lw=.5);ax.axvline(edge,color='gray',lw=.5)
  fig.colorbar(im,ax=axes,shrink=.7,label='Signed gate correlation')
  fig.savefig(root/'gate_correlation_matrices.png',dpi=160);plt.close(fig)


def corr(m, centered=False):
  # head-wise moments [weight, wx, wy, wx2, wy2, wxy].
  w, x, y, xx, yy, xy = np.moveaxis(m, -1, 0)
  if centered:
    vx = np.sum(xx - x*x/np.maximum(w, 1e-30))
    vy = np.sum(yy - y*y/np.maximum(w, 1e-30))
    cov = np.sum(xy - x*y/np.maximum(w, 1e-30))
  else:
    w, x, y, xx, yy, xy = m.sum(0)
    vx, vy, cov = xx-x*x/max(w,1e-30), yy-y*y/max(w,1e-30), xy-x*y/max(w,1e-30)
  return float(cov / np.sqrt(vx*vy)) if min(vx, vy) > 1e-20 else None


def main(root):
  root = Path(root)
  metadata = json.loads((root/'metadata.json').read_text())
  stats, loss, baseline, capture_losses, row_col_stats = [], [], [], [], []
  spectrum = {l: [0, np.zeros(64), np.zeros((64,64))] for l in LAYERS}
  gate_spectrum = {l: [0, np.zeros(64), np.zeros((64,64))] for l in LAYERS}
  hists, weighted_hists = [], []
  for file in sorted(root.glob('batch_*.npz')):
    with np.load(file) as data:
      mask = data['mask'].astype(np.float64)
      baseline.append(data['baseline'])
      loss.append(data['losses'])
      capture_losses.append(data['capture_loss'])
      per_layer, hist_layer, whist_layer, rc_layer = [], [], [], []
      for layer in LAYERS:
        logits = data[f'L{layer:02d}_ov_logits'].astype(np.float64)
        energy = data[f'L{layer:02d}_ov_energy'].astype(np.float64)
        gates = 1/(1+np.exp(-np.clip(logits, -80, 80)))
        assert np.isfinite(logits).all() and np.isfinite(energy).all()
        assert (energy >= 0).all()
        rc = []
        for values in (logits, gates):
          x, y = values[...,0,:], values[...,1,:]  # [batch,token,head,destination]
          w = np.broadcast_to(mask[...,None,None], x.shape)
          rc.append(np.stack([a.sum(1) for a in
              (w, w*x, w*y, w*x*x, w*y*y, w*x*y)], -1))
        rc_layer.append(np.stack(rc,1))
        moments = []
        for values, weighted in ((logits, False), (gates, False), (gates, True)):
          x, y = values[...,0], values[...,1]
          w = np.broadcast_to(mask[...,None,None], x.shape)
          if weighted:
            w = w * energy
          moments.append(np.stack([a.sum(1) for a in
              (w, w*x, w*y, w*x*x, w*y*y, w*x*y)], -1))
        per_layer.append(np.stack(moments, 1)) # [batch,3,head,side,6]
        joint = np.zeros((len(mask),2,5,5))
        weighted_joint = np.zeros_like(joint)
        bins = np.minimum((gates*5).astype(int),4)
        for seq in range(len(mask)):
          for side in range(2):
            index = bins[seq,:,:,side,0]*5 + bins[seq,:,:,side,1]
            valid = np.broadcast_to(mask[seq,:,None], index.shape)
            joint[seq,side] = np.bincount(index.ravel(), weights=valid.ravel(), minlength=25).reshape(5,5)
            weighted_joint[seq,side] = np.bincount(index.ravel(), weights=(valid*energy[seq,:,:,side]).ravel(), minlength=25).reshape(5,5)
        hist_layer.append(joint)
        whist_layer.append(weighted_joint)
        # Runtime centered joint-logit spectrum on systematically spaced positions.
        z = logits[:,::8].reshape(-1,64)[mask[:,::8].ravel().astype(bool)]
        spectrum[layer][0] += len(z)
        spectrum[layer][1] += z.sum(0)
        spectrum[layer][2] += z.T @ z
        g = gates[:,::8].reshape(-1,64)[mask[:,::8].ravel().astype(bool)]
        gate_spectrum[layer][0] += len(g)
        gate_spectrum[layer][1] += g.sum(0)
        gate_spectrum[layer][2] += g.T @ g
      stats.append(np.stack(per_layer,1))
      hists.append(np.stack(hist_layer,1))
      weighted_hists.append(np.stack(whist_layer,1))
      row_col_stats.append(np.stack(rc_layer,1))
  stats = np.concatenate(stats)
  hist = np.concatenate(hists)
  whist = np.concatenate(weighted_hists)
  base = np.concatenate(baseline)
  losses = np.concatenate(loss,1)
  captures = np.concatenate(capture_losses)
  rcstats = np.concatenate(row_col_stats)
  assert np.isfinite(base).all() and np.isfinite(losses).all()
  assert len(base) == len(metadata['sequence_hashes']), 'Incomplete cohort'
  expected = [f'batch_{i:03d}.npz' for i in range(0,len(base),metadata['batch_size'])]
  assert [p.name for p in sorted(root.glob('batch_*.npz'))] == expected
  np.savez_compressed(root/'per_sequence_stats.npz', moments=stats, histogram=hist,
      energy_histogram=whist, baseline=base, arm_loss=losses, capture_loss=captures,
      row_col_moments=rcstats, sequence_hashes=metadata['sequence_hashes'])
  report = {'sequences':len(base), 'baseline_loss':float(base.mean()), 'layers':{}, 'ablations':[],
      'analysis_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'capture_drift_mean': float(np.mean(captures-base)),
      'capture_drift_max_abs': float(np.max(np.abs(captures-base))),
      'matrix_token_stride': 8, 'matched_head_moments_token_stride': 1}
  for li, layer in enumerate(LAYERS):
    out = {}
    for side, si in [('row',0),('col',1)]:
      m = stats[:,li,:,:,si,:].sum(0)
      joint = hist[:,li,si].sum(0)
      wjoint = whist[:,li,si].sum(0)
      out[side] = dict(logit_corr=corr(m[0]), logit_corr_head_centered=corr(m[0],True),
          gate_corr=corr(m[1]), gate_corr_head_centered=corr(m[1],True),
          energy_weighted_gate_corr=corr(m[2]),
          energy_weighted_gate_corr_head_centered=corr(m[2],True),
          mean_gate_O=float(m[1,:,1].sum()/m[1,:,0].sum()),
          mean_gate_V=float(m[1,:,2].sum()/m[1,:,0].sum()),
          joint_5x5=(joint/max(joint.sum(),1)).tolist(),
          energy_joint_5x5=(wjoint/max(wjoint.sum(),1e-30)).tolist())
    for destination, di in [('O',0),('V',1)]:
      for name, ti in [('logit',0),('gate',1)]:
        m = rcstats[:,li,ti,:,di,:].sum(0)
        out[f'{destination}_row_col_{name}_corr_head_centered'] = corr(m,True)
    count, su, gram = spectrum[layer]
    cov = gram - np.outer(su,su)/max(count,1)
    eig = np.maximum(np.linalg.eigvalsh(cov)[::-1],0)
    out['runtime_logit_spectrum_energy'] = (eig/max(eig.sum(),1e-30)).tolist()
    # All head x side pairs, not only matched heads. Interleaved destination axis:
    # even indices O, odd V. Preserve signed correlations, including anti-correlation.
    for name, source in [('logit', spectrum), ('gate', gate_spectrum)]:
      nn, ss, gg = source[layer]
      cv = gg - np.outer(ss,ss)/max(nn,1)
      denom = np.sqrt(np.maximum(np.diag(cv),0))
      den = np.outer(denom,denom)
      cc = np.divide(cv, den, out=np.zeros_like(cv), where=den>1e-20)
      out[f'cross_head_side_{name}_corr_O_by_V'] = cc[::2,1::2].tolist()
      out[f'cross_head_side_{name}_valid_O_by_V'] = (den[::2,1::2]>1e-20).tolist()
      out[f'full_{name}_correlation'] = cc.tolist()
      out[f'full_{name}_valid'] = (den>1e-20).tolist()
      # Flattened channel order is head, side(row,col), destination(O,V).
      for destination, di in [('O',0),('V',1)]:
        rows = np.arange(16)*4 + di
        cols = rows + 2
        out[f'{destination}_row_by_col_{name}_correlation'] = cc[np.ix_(rows,cols)].tolist()
        out[f'{destination}_row_by_col_{name}_valid'] = (den[np.ix_(rows,cols)]>1e-20).tolist()
    report['layers'][str(layer)] = out
  for index, arm in enumerate(metadata['arms']):
    delta = losses[index] - base
    report['ablations'].append(dict(layer=arm[0], side=['row','col'][arm[1]],
        replacement=['V<-O','O<-V'][arm[2]], mean=float(delta.mean()),
        ci95=float(1.96*delta.std(ddof=1)/np.sqrt(len(delta)))))
  (root/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
  plot_summary(root,report)
  print(json.dumps({'baseline':report['baseline_loss'],'sequences':len(base),
                    'global_ablations':report['ablations'][:4]},indent=2))


if __name__ == '__main__':
  main(sys.argv[1])
