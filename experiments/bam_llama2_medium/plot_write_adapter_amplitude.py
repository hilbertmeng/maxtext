"""Render activation-measured adapter gains, preserving every layer and head."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def main():
  p=argparse.ArgumentParser(description=__doc__)
  p.add_argument('summary',type=Path)
  p.add_argument('--output',type=Path,required=True)
  args=p.parse_args()
  d=json.loads(args.summary.read_text()); rows=d['per_layer']; layers=np.array([r['layer'] for r in rows])
  head=np.array([r['head_gain'] for r in rows]); gain=np.array([r['aggregate_gain'] for r in rows])
  quant=np.array([r['token_head_gain_p10_p50_p90'] for r in rows]); iso=np.array([r['isotropic_gain'] for r in rows])
  fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
  ax=axes[0,0]; ax.fill_between(layers,head.min(1),head.max(1),color='C0',alpha=.16,label='Head min-max')
  ax.plot(layers,gain,'o-',label='Actual aggregate RMS gain',ms=4)
  ax.plot(layers,np.sqrt(np.mean(iso**2,1)),'s--',label='Isotropic-input prediction',ms=3,color='C2')
  ax.set_title('Measured content gain by layer');ax.set_ylabel('After / before RMS');ax.legend(fontsize=8)
  ax=axes[0,1];ax.fill_between(layers,quant[:,0],quant[:,2],color='C1',alpha=.2,label='Token/head P10-P90')
  ax.plot(layers,quant[:,1],'o-',color='C1',ms=4,label='Token/head median');ax.legend(fontsize=8)
  ax.set_title('Individual-vector amplitude ratios');ax.set_ylabel('After / before vector norm')
  ax=axes[1,0];lo=min(float(head.min())-.01,.99);hi=max(float(head.max())+.01,1.01)
  im=ax.imshow(head.T,origin='lower',aspect='auto',cmap='RdBu_r',norm=TwoSlopeNorm(vmin=lo,vcenter=1,vmax=hi))
  ax.set_title('Actual aggregate gain: every head');ax.set_ylabel('Head');ax.set_yticks(np.arange(head.shape[1]))
  fig.colorbar(im,ax=ax,label='After / before RMS')
  ax=axes[1,1];ax.plot(layers,[r['cosine'] for r in rows],'o-',color='C3',ms=4)
  ax.set_title('Direction retained after adapter');ax.set_ylabel('Mean cosine(before, after)');ax.set_ylim(min(r['cosine'] for r in rows)-.01,1.005)
  for ax in axes.flat:
    ax.set_xlabel('Layer');ax.set_xticks(layers);ax.grid(alpha=.2)
  for ax in axes[0]:ax.axhline(1,color='black',ls=':',lw=1)
  fig.suptitle('Post-RMS write dense adapter: checkpoint 13,500\n32 Pile sequences, T4096; L17 has no attention writer',fontsize=12)
  args.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(args.output,dpi=100);plt.close(fig)
  from PIL import Image
  with Image.open(args.output) as source:
    image=source.convert('RGB').quantize(colors=256);image.save(args.output,optimize=True)
  print(json.dumps(dict(figure=str(args.output),bytes=args.output.stat().st_size)))

if __name__=='__main__':main()
