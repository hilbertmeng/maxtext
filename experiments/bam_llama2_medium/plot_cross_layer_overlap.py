import os
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['OMP_NUM_THREADS']='4'
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import argparse
parser=argparse.ArgumentParser(description='Plot all-layer input-subspace overlaps from restored weights')
parser.add_argument('artifact_root',type=Path)
root=parser.parse_args().artifact_root
fig,axes=plt.subplots(2,2,figsize=(13,11),constrained_layout=True)
report={}
for row,label in enumerate(('pair256','triple384')):
    a=np.load(root/f'{label}-geometry/weights.npz')
    gs=[next(a[n] for n in a.files if '/'+g+'/' in n) for g in ('local_0','local_1','fetch_2')]
    ws=[g[:,b].astype(np.float64) for b in range(6) for g in gs]
    qs=[np.linalg.svd(w,full_matrices=False)[0] for w in ws]
    report[label]={}
    for col,k in enumerate((64,ws[0].shape[1])):
        m=np.array([[np.sum((u[:,:k].T@v[:,:k])**2)/k for v in qs] for u in qs])
        report[label][str(k)]=m.tolist()
        # Off-diagonal values share a fixed scale. Hide trivial self-overlap=1.
        show=m.copy();np.fill_diagonal(show,np.nan)
        ax=axes[row,col];im=ax.imshow(show,vmin=0,vmax=.65,cmap='viridis')
        ax.set_title(f'{label}: top {k} subspace overlap\nrandom-subspace expectation = {k/1200:.3f}')
        ax.set_xticks(range(18));ax.set_xticklabels(range(1,19),fontsize=8)
        ax.set_yticks(range(18));ax.set_yticklabels(range(1,19),fontsize=8)
        ax.set_xlabel('Layer');ax.set_ylabel('Layer')
        for b in (2.5,5.5,8.5,11.5,14.5):
            ax.axhline(b,color='white',lw=.7,alpha=.8);ax.axvline(b,color='white',lw=.7,alpha=.8)
        fig.colorbar(im,ax=ax,shrink=.82)
        def avg(p):return float(np.mean([m[i,j] for i in range(18) for j in range(i+1,18) if p(i,j)]))
        report[label][str(k)+'_summary']={
          'within_block':avg(lambda i,j:i//3==j//3),
          'cross_block_same_role':avg(lambda i,j:i//3!=j//3 and i%3==j%3),
          'cross_block_different_role':avg(lambda i,j:i//3!=j//3 and i%3!=j%3),
          'by_distance':{str(d):avg(lambda i,j:j-i==d) for d in range(1,18)},
          'within_thirds':avg(lambda i,j:i//6==j//6),
          'within_halves':avg(lambda i,j:i//9==j//9),
          'first_block_to_later':avg(lambda i,j:i<3 and j>=3),
          'later_layers':avg(lambda i,j:i>=3),
        }
fig.suptitle('Checkpoint 13500 | shared Down input subspaces | white lines: 3-layer blocks\nOverlap = ||U_i^T U_j||_F^2 / rank; diagonal omitted',fontsize=13)
fig.savefig(root/'layer-overlap-matrices.png',dpi=180)
fig.savefig(root/'layer-overlap-matrices.pdf')
(root/'layer-overlap-matrices.json').write_text(json.dumps(report,indent=2)+'\n')
for label,v in report.items():
    for k in ('64', '256' if label=='pair256' else '384'):
        print(label,k,json.dumps(v[k+'_summary']))
