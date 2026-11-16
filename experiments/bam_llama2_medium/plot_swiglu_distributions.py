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
  p.add_argument('--sensitivity', action='store_true', help='Opt in to secondary SiTU counterfactual plots')
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
  phase, up99, vp99, umax, vmax, entries, phase_summaries = [],[],[],[],[],[],[]
  for row in rows:
    step = row['checkpoint']['step']; phase.append(str(step))
    stats=row['all_sequences']
    with np.load(args.results/str(step)/'statistics.npz') as raw:
      edges=raw['edges'];counts=raw['moments'][...,0].sum(0)
    qlo,qhi=absolute_quantile(stats['histogram'],edges,.999)
    u99=np.asarray(stats['signed_quantile_intervals']['0.999']).mean(-1)
    # Use interval midpoint; quantile bins, not exact order statistics.
    up99.append(u99[:,0]); vp99.append((qlo[:,1]+qhi[:,1])/2)
    umax.append(np.asarray(stats['maximum'])[:,0])
    vmax.append(np.maximum(np.abs(stats['minimum']),np.abs(stats['maximum']))[:,1])
    tails=np.asarray(stats['tails'])
    phase_summaries.append(dict(step=step,mean_ce=row['mean_ce'],
        median_layer_rms=np.median(stats['rms'],axis=0).tolist(),
        min_over_all_layers=np.min(stats['minimum'],axis=0).tolist(),
        max_over_all_layers=np.max(stats['maximum'],axis=0).tolist(),
        entries_per_branch=counts.sum(0).tolist(),
        positive_u_tail_fractions=(tails[:,0]*counts[:,0,None]).sum(0).tolist(),
        abs_v_tail_fractions=(tails[:,2]*counts[:,1,None]).sum(0).tolist(),
        thresholds=row['thresholds'],
        max_noop_ce_error=max(abs(c['delta']) for c in row['no_op_controls'])))
    for key,branch in (('positive_u_tail_fractions',0),('abs_v_tail_fractions',1)):
      phase_summaries[-1][key]=(np.asarray(phase_summaries[-1][key])/counts[:,branch].sum()).tolist()
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
  raw_figures=[]
  for branch,label in ((0,'u'),(1,'v')):
    columns=2;row_count=(len(rows)+1)//2
    fig,axs=plt.subplots(row_count,columns,figsize=(10,2.05*row_count),sharex=True,sharey=True,
                         squeeze=False,layout='constrained')
    maximum=max(max(abs(x['all_sequences']['minimum'][l][branch]),abs(x['all_sequences']['maximum'][l][branch]))
                for x in rows for l in range(28))
    for ax,row in zip(axs.ravel(),rows):
      s=row['all_sequences'];q=lambda key:np.asarray(s['signed_quantile_intervals'][str(key)])[:,branch].mean(-1)
      x=np.arange(28)
      ax.fill_between(x,q(.001),q(.999),color='C0',alpha=.14,label='p0.1--p99.9')
      ax.fill_between(x,q(.01),q(.99),color='C0',alpha=.35,label='p1--p99')
      ax.plot(x,q(.5),color='black',lw=1,label='median')
      ax.plot(x,np.asarray(s['maximum'])[:,branch],color='.45',lw=.8,label='exact min/max')
      ax.plot(x,np.asarray(s['minimum'])[:,branch],color='.45',lw=.8)
      for amplitude,color in ((4.,'C1'),(7.,'C2'),(10.,'C3')):
        ax.axhline(amplitude,color=color,ls='--',lw=.6,alpha=.75,label=f'{amplitude:g} reference')
        if branch==1:ax.axhline(-amplitude,color=color,ls='--',lw=.6,alpha=.75)
      ax.set_title('Step '+str(row['checkpoint']['step']),fontsize=9)
      ax.set_yscale('symlog',linthresh=1.);ax.set_ylim(-maximum*1.15,maximum*1.15)
      ax.set_xticks([0,6,13,20,27]);ax.grid(alpha=.15)
    for ax in axs.ravel()[len(rows):]:ax.set_visible(False)
    handles, labels = axs[0,0].get_legend_handles_labels()
    if len(rows) < axs.size:
      legend_ax=axs.ravel()[len(rows)];legend_ax.set_visible(True);legend_ax.axis('off')
      legend_ax.legend(handles,labels,fontsize=9,ncol=2,loc='center')
    else:
      fig.legend(handles,labels,fontsize=7,ncol=4,loc='outside lower center')
    for col in range(min(columns,len(rows))):
      last=(len(rows)-1-col)//columns
      axs[last,col].tick_params(labelbottom=True);axs[last,col].set_xlabel('Ordinary MLP layer')
    fig.supylabel(f'Original {label} (symlog; linear within +/-1)')
    fig.suptitle(f'XL V5: original {label} distribution; no activation clipping',fontsize=11)
    path=args.output_dir/f'xlprop-v5-original-{label}-distributions.png';fig.savefig(path,dpi=100);plt.close(fig)
    with Image.open(path) as im:im.convert('RGB').quantize(colors=128).save(path,optimize=True)
    raw_figures.append(str(path))
  summary=dict(checkpoint_steps=[r['checkpoint']['step'] for r in rows],
               phase_summaries=phase_summaries,
               source_summaries=[str(args.results/str(r['checkpoint']['step'])/'summary.json') for r in rows],
               distribution_figure=str(distribution),raw_figures=raw_figures,csv=str(csv_path))
  if not args.sensitivity:
    (args.output_dir/'xlprop-v5-uv-figures.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2));return
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
  summary['sensitivity_figure']=str(sensitivity)
  (args.output_dir/'xlprop-v5-uv-figures.json').write_text(json.dumps(summary,indent=2)+'\n')
  print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
