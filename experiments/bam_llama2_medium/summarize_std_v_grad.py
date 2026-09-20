"""Summarize native standard-V sensitivities without interpreting them as deletions."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def stats(a):
 return dict(mean=float(a.mean()),se=float(a.std(ddof=1)/np.sqrt(len(a))),mean_abs=float(np.abs(a).mean()),rms=float(np.sqrt(np.mean(a*a))),median=float(np.median(a)),positive=int((a>0).sum()),n=len(a))
def run(root):
 files=sorted(root.glob('worker*/grad_???.npz'));assert len(files)==32,len(files)
 ids=[int(p.stem.split('_')[1]) for p in files];assert ids==list(range(32))
 gradients=[];native=[];checks=[];shas={};errors=[]
 meta=json.loads(files[0].with_name('metadata.json').read_text())
 for i,p in enumerate(files):
  a=np.load(p);assert str(a['sequence_hash'])==meta['sequence_hashes'][i]['inputs'];assert a['gradient'].shape==(24,3);assert np.isfinite(a['gradient']).all()
  gradients.append(a['gradient']);native.append(float(a['loss']));errors.append(abs(float(a['ordinary'])-float(a['loss'])))
  for c in a['finite_difference']:checks.append(dict(sample=i,channel=int(c[0]),autodiff=float(c[1]),central_difference=float(c[2]),plus=float(c[3]),minus=float(c[4])))
  shas[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
 g=np.stack(gradients);score=-g # d loss / d removed fraction at the native point
 assert np.max(errors)<=1e-6;assert np.all(g[:,2::3]==0)
 local=[l for l in range(24) if l%3!=2];normal=[l for l in local if l>1]
 rows={str(l):{n:stats(score[:,l,c]) for c,n in enumerate(['front','tail','bam'])} for l in local}
 groups={}
 for name,ls in dict(all_L=local,ordinary_L=normal,early_ordinary_L=[l for l in normal if l<12],late_L=[l for l in normal if l>=12]).items():
  a=score[:,ls,:].sum(axis=1);groups[name]={n:stats(a[:,c]) for c,n in enumerate(['front','tail','bam'])};groups[name]['front_minus_tail']=stats(a[:,0]-a[:,1])
 result=dict(model=meta['model'],checkpoint=meta['checkpoint'],samples=32,native_loss=stats(np.asarray(native)),definition='score = -d loss/d scale, positive means infinitesimal attenuation increases loss; absolute means are samplewise and not additive attribution',layers=rows,groups=groups,finite_difference=checks)
 (root/'summary.json').write_text(json.dumps(result,indent=2));np.savez_compressed(root/'paired_gradients.npz',gradient=g,attenuation_score=score,native_loss=native)
 (root/'verification.json').write_text(json.dumps(dict(raw_count=32,max_native_error=max(errors),sha256=shas),indent=2))
 fig,axes=plt.subplots(2,1,figsize=(11,7),layout='constrained',sharex=True)
 for c,label,color,shift in [(0,'Raw V front 32','#2166ac',-.12),(1,'Raw V tail 32','#d6604d',.12)]:
  a=score[:,normal,c];x=np.asarray(normal)+shift
  axes[0].errorbar(x,a.mean(0),yerr=1.96*a.std(0,ddof=1)/np.sqrt(32),fmt='o-',color=color,label=label,capsize=3)
  axes[1].plot(x,np.abs(a).mean(0),'o-',color=color,label=label)
 axes[0].axhline(0,color='gray',lw=.8);axes[0].set_ylabel('Mean attenuation derivative')
 axes[1].set_ylabel('Mean absolute derivative');axes[1].set_xticks(normal,[f'L{l}' for l in normal]);axes[1].set_xlabel('Local layer (zero-based); L0/L1 omitted from plots, retained in tables')
 for ax in axes:ax.grid(alpha=.2);ax.legend()
 fig.suptitle('Standard V halves at native scale=1; BAM injection preserved\n32 paired Pile sequences; signed mean ± 1.96 SE; units nats/token per unit scale')
 for ext in ['png','pdf']:fig.savefig(root/f'std_v_native_gradient.{ext}',dpi=180)
 lines=['| 层/组 | 前半削弱导数 ±1.96SE | 后半削弱导数 ±1.96SE | 前半绝对值均值 | 后半绝对值均值 | 前半/后半正号样本数 |','|---|---:|---:|---:|---:|---:|']
 for name,r in [(f'L{l}',rows[str(l)]) for l in local]+list(groups.items()):
  f,t=r['front'],r['tail'];lines.append(f"| {name} | {f['mean']:+.6f} ± {1.96*f['se']:.6f} | {t['mean']:+.6f} ± {1.96*t['se']:.6f} | {f['mean_abs']:.6f} | {t['mean_abs']:.6f} | {f['positive']}/32 · {t['positive']}/32 |")
 (root/'gradient_table.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps(groups,indent=2));print('STD_V_GRAD_VERIFIED',len(files),max(errors))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);run(p.parse_args().root)
