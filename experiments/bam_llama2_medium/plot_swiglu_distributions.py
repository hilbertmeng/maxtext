"""Plot complete layer/stage u/v distributions and mild SiTU sensitivity curves."""
import argparse
import csv
import json
from pathlib import Path


def absolute_quantile(histogram, edges, q):
  import numpy as np
  h, e = np.asarray(histogram), np.asarray(edges)
  n = h.shape[-1]//2
  assert len(e) == 2*n+1 and e[n] == 0 and np.all(e[1:n] == -e[-2:n:-1])
  absolute = h[..., n:] + h[..., :n][..., ::-1]
  cumulative = np.cumsum(absolute, -1)
  index = np.argmax(cumulative >= q*cumulative[..., -1, None], -1)
  positive_edges = e[n:]
  return positive_edges[index], positive_edges[index+1]


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('results', type=Path)
  p.add_argument('--output-dir', type=Path, required=True)
  args = p.parse_args()
  import numpy as np
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  from matplotlib.colors import LogNorm
  from PIL import Image
  args.output_dir.mkdir(parents=True,exist_ok=True)
  rows = sorted((json.loads(f.read_text()) for f in args.results.glob('*/summary.json')),
                key=lambda j:j['checkpoint']['step'])
  assert rows
  phase, up99, vp99, umax, vmax, entries = [],[],[],[],[],[]
  for row in rows:
    step = row['checkpoint']['step']; phase.append(str(step))
    stats=row['all_sequences']
    with np.load(args.results/str(step)/'statistics.npz') as raw:edges=raw['edges']
    qlo,qhi=absolute_quantile(stats['histogram'],edges,.999)
    u99=np.asarray(stats['signed_quantile_intervals']['0.999']).mean(-1)
    # Use interval midpoint; quantile bins, not exact order statistics.
    up99.append(u99[:,0]); vp99.append((qlo[:,1]+qhi[:,1])/2)
    umax.append(np.asarray(stats['maximum'])[:,0])
    vmax.append(np.maximum(np.abs(stats['minimum']),np.abs(stats['maximum']))[:,1])
    for layer in range(28):
      e=dict(step=step,layer=layer,u_rms=stats['rms'][layer][0],v_rms=stats['rms'][layer][1],
             u_positive_q999=up99[-1][layer],v_abs_q999=vp99[-1][layer],
             u_max=umax[-1][layer],v_abs_max=vmax[-1][layer])
      for t,threshold in enumerate(row['thresholds']):
        e['u_positive_gt_'+str(threshold)]=stats['tails'][layer][0][t]
        e['v_abs_gt_'+str(threshold)]=stats['tails'][layer][2][t]
      entries.append(e)
  csv_path=args.output_dir/'xlprop-v5-uv-by-stage-layer.csv'
  with csv_path.open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(entries[0]));writer.writeheader();writer.writerows(entries)
  fig,axes=plt.subplots(2,2,figsize=(10,6.5),layout='constrained')
  for ax,values,title in zip(axes.ravel(),(up99,vp99,umax,vmax),
      ('Gate u: positive p99.9','Up |v|: p99.9','Gate u: exact maximum','Up |v|: exact maximum')):
    a=np.asarray(values);im=ax.imshow(a,aspect='auto',interpolation='nearest',
        norm=LogNorm(vmin=max(.01,np.nanmin(a)),vmax=max(.02,np.nanmax(a))),cmap='viridis')
    ax.set_title(title);ax.set_xlabel('Ordinary MLP layer');ax.set_ylabel('Checkpoint step')
    ax.set_xticks([0,6,13,20,27]);ax.set_yticks(range(len(phase)),phase)
    fig.colorbar(im,ax=ax,shrink=.8)
  fig.suptitle('XL V5: same 32 Pile sequences, T4096; all 28 ordinary MLPs',fontsize=11)
  distribution=args.output_dir/'xlprop-v5-uv-distributions.png'
  fig.savefig(distribution,dpi=100);plt.close(fig)
  for path in [distribution]:
    with Image.open(path) as image:image.convert('RGB').quantize(colors=128).save(path,optimize=True)
  latest=rows[-1]; soft=latest['all_sequences']['soft']; candidates=latest['candidates']
  fig,axes=plt.subplots(3,1,figsize=(9.5,7),sharex=True,layout='constrained')
  selection=((4.,25.),(4.,8.),(4.,4.),(8.,4.),(2.,2.))
  for candidate in selection:
    i=candidates.index(list(candidate))
    for ax,key in zip(axes,('hidden_relative_change','du_multiplier_rms_ratio','dv_multiplier_rms_ratio')):
      ax.plot(range(28),np.asarray(soft[key])[:,i],label=f'SiTU {candidate[0]:g}/{candidate[1]:g}',linewidth=1.3)
  for ax,label in zip(axes,('Hidden product relative L2 change','RMS of dh/du, relative to SwiGLU','RMS of dh/dv, relative to SwiGLU')):
    ax.set_ylabel(label,fontsize=8);ax.grid(alpha=.2)
  axes[0].legend(ncol=3,fontsize=8);axes[-1].set_xlabel('Ordinary MLP layer')
  fig.suptitle(f'Checkpoint {latest["checkpoint"]["step"]}: fixed-input sensitivity, not a loss/gradient replay',fontsize=10)
  sensitivity=args.output_dir/'xlprop-v5-situ-sensitivity.png'
  fig.savefig(sensitivity,dpi=100);plt.close(fig)
  with Image.open(sensitivity) as image:image.convert('RGB').quantize(colors=128).save(sensitivity,optimize=True)
  summary=dict(checkpoint_steps=[r['checkpoint']['step'] for r in rows],
               source_summaries=[str(args.results/str(r['checkpoint']['step'])/'summary.json') for r in rows],
               distribution_figure=str(distribution),sensitivity_figure=str(sensitivity),csv=str(csv_path))
  (args.output_dir/'xlprop-v5-uv-figures.json').write_text(json.dumps(summary,indent=2)+'\n')
  print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
