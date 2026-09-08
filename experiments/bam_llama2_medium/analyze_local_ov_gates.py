"""Aggregate scalar gate captures, preserving per-sequence sufficient statistics."""
from pathlib import Path
import json
import sys
import numpy as np

LAYERS = [i for i in range(24) if i % 3 != 2]


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
  stats, loss, baseline = [], [], []
  spectrum = {l: [0, np.zeros(64), np.zeros((64,64))] for l in LAYERS}
  hists, weighted_hists = [], []
  for file in sorted(root.glob('batch_*.npz')):
    with np.load(file) as data:
      mask = data['mask'].astype(np.float64)
      baseline.append(data['baseline'])
      loss.append(data['losses'])
      per_layer, hist_layer, whist_layer = [], [], []
      for layer in LAYERS:
        logits = data[f'L{layer:02d}_ov_logits'].astype(np.float64)
        energy = data[f'L{layer:02d}_ov_energy'].astype(np.float64)
        gates = 1/(1+np.exp(-np.clip(logits, -80, 80)))
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
      stats.append(np.stack(per_layer,1))
      hists.append(np.stack(hist_layer,1))
      weighted_hists.append(np.stack(whist_layer,1))
  stats = np.concatenate(stats)
  hist = np.concatenate(hists)
  whist = np.concatenate(weighted_hists)
  base = np.concatenate(baseline)
  losses = np.concatenate(loss,1)
  np.savez_compressed(root/'per_sequence_stats.npz', moments=stats, histogram=hist,
      energy_histogram=whist, baseline=base, arm_loss=losses)
  report = {'sequences':len(base), 'baseline_loss':float(base.mean()), 'layers':{}, 'ablations':[]}
  for li, layer in enumerate(LAYERS):
    out = {}
    for side, si in [('row',0),('col',1)]:
      m = stats[:,li,:,:,si,:].sum(0)
      joint = hist[:,li,si].sum(0)
      wjoint = whist[:,li,si].sum(0)
      out[side] = dict(logit_corr=corr(m[0]), logit_corr_head_centered=corr(m[0],True),
          gate_corr=corr(m[1]), gate_corr_head_centered=corr(m[1],True),
          energy_weighted_gate_corr=corr(m[2]),
          mean_gate_O=float(m[1,:,1].sum()/m[1,:,0].sum()),
          mean_gate_V=float(m[1,:,2].sum()/m[1,:,0].sum()),
          joint_5x5=(joint/max(joint.sum(),1)).tolist(),
          energy_joint_5x5=(wjoint/max(wjoint.sum(),1e-30)).tolist())
    count, su, gram = spectrum[layer]
    cov = gram - np.outer(su,su)/max(count,1)
    eig = np.maximum(np.linalg.eigvalsh(cov)[::-1],0)
    out['runtime_logit_spectrum_energy'] = (eig/max(eig.sum(),1e-30)).tolist()
    report['layers'][str(layer)] = out
  for index, arm in enumerate(metadata['arms']):
    delta = losses[index] - base
    report['ablations'].append(dict(layer=arm[0], side=['row','col'][arm[1]],
        replacement=['V<-O','O<-V'][arm[2]], mean=float(delta.mean()),
        ci95=float(1.96*delta.std(ddof=1)/np.sqrt(len(delta)))))
  (root/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
  print(json.dumps({'baseline':report['baseline_loss'],'sequences':len(base),
                    'global_ablations':report['ablations'][:4]},indent=2))


if __name__ == '__main__':
  main(sys.argv[1])
