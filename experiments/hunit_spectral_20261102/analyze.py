"""Aggregate paired checkpoint geometry; bootstrap texts, never tokens."""
import argparse,json
from pathlib import Path
import numpy as np

R=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--manifest',default=str(R/'arms_gpu.json'));p.add_argument('--output',default=str(R/'results_gpu.json'));args=p.parse_args()
m=json.loads(Path(args.manifest).read_text());D=Path(m['local_DATA'])
rng=np.random.default_rng(20261102);boot=rng.integers(0,32,(5000,32))
def estimate(x,root=False):
 x=np.asarray(x,np.float64);b=x[boot].mean(1);v=x.mean(0)
 if root:v=np.sqrt(v);b=np.sqrt(b)
 return dict(mean=np.asarray(v).tolist(),ci95=np.quantile(b,[.025,.975],axis=0).tolist())
rows={};powers=[];weak=[];frame_means={}
fixed_bases={}
for step in m['steps']:
 with np.load(D/'head_spectra'/f'{step}.npz') as z:fixed_bases[str(step)]=z['centered_basis'][:,:128].copy()
for step in m['steps']:
 f=D/'results'/str(step);s=json.loads((f/'complete.json').read_text());spec=json.loads((D/'head_spectra'/f'{step}.json').read_text())
 q={k:[] for k in ['head_power','raw_power','raw_weak','centered_weak','common_axis','raw_bands','centered_bands','offset_power','logit_power','native_CE','FP32_CE','quantized_CE','center_quantized_CE','centered_first1','centered_first2','centered_first4','fixed33_weak','fixed43_weak']}
 means=[]
 for i in range(32):
  with np.load(f/f'head_{i:03d}.npz') as z:
   for key,src in [('head_power','head_power'),('raw_power','raw_power'),('common_axis','common_axis_power'),('logit_power','centered_logit_power')]:q[key].append(z[src].mean())
   q['raw_weak'].append(z['raw_bands'][:,0].mean());q['centered_weak'].append(z['centered_bands'][:,0].mean())
   for key in ['raw_bands','centered_bands']:q[key].append(z[key].mean(0))
   q['offset_power'].append(np.mean(z['common_logit_offset'].astype(np.float64)**2));q['FP32_CE'].append(z['FP32_CE'].mean())
   q['quantized_CE'].append(z['precision_CE'][1].mean());q['center_quantized_CE'].append(z['precision_CE'][2].mean())
  with np.load(f/f'sequence_{i:03d}.npz') as z:
   q['native_CE'].append(z['native_loss'].mean());h=z['h'][0];means.append(h.mean(0,dtype=np.float64));pc=h@fixed_bases[str(step)]
   for n in [1,2,4]:q['centered_first'+str(n)].append(np.mean(np.sum(pc[:,:n].astype(np.float64)**2,-1)/2560))
   for t,key in [('33000','fixed33_weak'),('43000','fixed43_weak')]:
    pp=h@fixed_bases[t];q[key].append(np.mean(np.sum(pp.astype(np.float64)**2,-1)/2560))
 q={k:np.asarray(v,np.float64) for k,v in q.items()};frame_means[str(step)]=q
 r=dict(summary=s,spectrum=spec)
 for out,key in [('head_RMS','head_power'),('pre_norm_RMS','raw_power'),('raw_weak_RMS','raw_weak'),('centered_weak_RMS','centered_weak'),('common_axis_RMS','common_axis'),('common_logit_offset_RMS','offset_power'),('centered_logit_RMS','logit_power')]:r[out]=estimate(q[key],True)
 r['centered_other_RMS']=estimate(q['head_power']-q['centered_weak'],True)
 for key in ['raw_weak','centered_weak','common_axis']:r[key+'_energy_fraction']=float(q[key].mean()/q['head_power'].mean())
 for key in ['centered_first1','centered_first2','centered_first4','fixed33_weak','fixed43_weak']:r[key+'_energy_fraction']=float(q[key].mean()/q['head_power'].mean())
 mh=np.mean(means,0);r['centered_weak_mean_vector_energy_fraction']=float(np.sum((mh@fixed_bases[str(step)])**2)/2560/q['head_power'].mean())
 for key in ['raw_bands','centered_bands']:r[key+'_energy_fraction']=(q[key].mean(0)/q['head_power'].mean()).tolist()
 for key in ['native_CE','FP32_CE']:r[key]=estimate(q[key])
 r['native_minus_FP32_CE']=estimate(q['native_CE']-q['FP32_CE'])
 r['center_before_quantization_delta_CE']=estimate(q['center_quantized_CE']-q['quantized_CE'])
 g2=s['global_head_gradient_norm']**2
 for out,key in [('raw_weak_gradient_fraction','raw_dark_gradient_energy'),('centered_weak_gradient_fraction','centered_dark_gradient_energy'),('common_axis_gradient_fraction','common_axis_gradient_energy')]:r[out]=s[key]/g2
 r['centered_other_gradient_norm']=float(np.sqrt(g2-s['centered_dark_gradient_energy']))
 rows[str(step)]=r;powers.append(q['head_power']);weak.append(q['centered_weak'])
a,b=[rows[str(t)] for t in m['steps']];ga,gb=[x['summary']['global_head_gradient_norm']**2 for x in [a,b]]
dw=weak[1]-weak[0];dp=powers[1]-powers[0]
backend='Original TPU FastKernel forward' if m.get('backend')=='native_tpu' else 'CUDA pure-JAX reference for native TPU FastKernel'
out=dict(manifest=m,scope=f'Same32 Pile training-source texts, 2048 targets each; {backend}. Fixed frozen-head FP32 gradients; formal training gradients reported separately.',rows=rows,change=dict(head_RMS_ratio=b['head_RMS']['mean']/a['head_RMS']['mean'],head_gradient_norm_ratio=np.sqrt(gb/ga),weak_share_of_head_gradient_squared_norm_change=(b['summary']['centered_dark_gradient_energy']-a['summary']['centered_dark_gradient_energy'])/(gb-ga),weak_share_of_head_input_squared_norm_change=float(dw.mean()/dp.mean()),weak_share_of_input_growth_ci95=np.quantile(dw[boot].mean(1)/dp[boot].mean(1),[.025,.975]).tolist(),paired_native_CE_change=estimate(frame_means['43000']['native_CE']-frame_means['33000']['native_CE'])),formal_training=json.loads((R/'training_gradient_analysis.json').read_text()))
Path(args.output).write_text(json.dumps(out,indent=2)+'\n');print('HUNIT_AGGREGATED',out['change'],flush=True)
