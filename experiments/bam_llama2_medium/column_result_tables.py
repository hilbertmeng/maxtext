"""Generate matched row/column group tables with paired uncertainty."""
import argparse,json
from pathlib import Path
import numpy as np

def fmt(x):return f'{x["mean"]:+.6f} ± {1.96*x["se"]:.6f}'
def run(medium,xl,out):
 ds=[json.loads((r/'summary.json').read_text()) for r in (medium,xl)];out.mkdir(exist_ok=True,parents=True)
 lines=[]
 def table(title,names):
  lines.extend(['## '+title,'','| 场景 | Medium列读Δloss | XL列读Δloss |','|---|---:|---:|'])
  for suffix in names:lines.append('| '+suffix+' | '+' | '.join(fmt(d['metrics']['column/'+suffix]) for d in ds)+' |')
  lines.append('')
 table('整路与半幅',[f'all/coalition_{i:02d}' for i in [1,2,3,4,8,15]]+[f'all/half_{p}' for p in ['Q','K','V','O','all']])
 table('L/F与关键层组合',['targeted/'+x for x in ['O_local_off','O_fetch_off','O_keep_F5_F11','O_keep_L1_F2_F5_F11','O_F5_F11_off','V_keep_L1','combined_sparse_rows']]+['qktype/'+x for x in ['Q_L_off','Q_F_off','K_L_off','K_F_off','QK_L_off','QK_F_off']])
 for stage,p in [('depth','QK'),('vdepth','V'),('odepth','O')]:
  table(p+'深度组',[stage+'/'+p+'_'+s for s in ['all_off','first_half','last_half','first_third','middle_third','last_third']])
 for label,d in zip(['Medium','XL'],ds):
  lines+=['## '+label+'逐层列读','','| Layer | Q | K | V | O |','|---|---:|---:|---:|---:|']
  for l in range(24):
   vals=[fmt(d['metrics'][f'column/all/layer_{l:02d}_{p}']) if not(p=='V' and l%3==2) else '—' for p in 'QKVO'];lines.append(f'| {l}{"F" if l%3==2 else "L"} | '+' | '.join(vals)+' |')
  lines+=['','## '+label+' LLF单元列读','','| Unit/layers | Q | K | V | O |','|---|---:|---:|---:|---:|']
  for u in range(8):lines.append(f'| {u}: {3*u}–{3*u+2} | '+' | '.join(fmt(d['metrics'][f'column/all/unit_{u}_{p}']) for p in 'QKVO')+' |')
  lines.append('')
 (out/'column_tables.md').write_text('\n'.join(lines)+'\n')
 lines=['# 行/列两玩家分配及交互','','其它路径保持原生；mean ±1.96 paired SE。同路行列共同关闭−行单关−列单关为interaction。','']
 for label,d in zip(['Medium','XL'],ds):
  lines += ['## '+label,'','| Scope | Row Δ | Column Δ | Both Δ | Interaction | Row Shapley | Column Shapley |','|---|---:|---:|---:|---:|---:|---:|']
  names=[f'all/coalition_{i:02d}' for i in [1,2,3,4,8,15]]+['targeted/'+s for s in ['O_local_off','O_fetch_off']]+['qktype/QK_L_off','qktype/QK_F_off']
  for name in names:
   s=d['row_column_pairs'][name];lines.append('| '+name+' | '+' | '.join(fmt(s[k]) for k in ['row','column','both','interaction','row_shapley','column_shapley'])+' |')
  lines.append('')
 (out/'row_column_interactions.md').write_text('\n'.join(lines)+'\n')
 # Pair model differences using the same sequence indices, not independent errors.
 arrays=[]
 for r in (medium,xl):
  a=np.load(r/'paired_results.npz');arrays.append({n:a['gap'][:,i] for i,n in enumerate(a['names'])})
 names=[n for n in arrays[0] if n in arrays[1]];delta=np.stack([arrays[1][n]-arrays[0][n] for n in names],axis=1)
 np.savez_compressed(out/'xl_minus_medium_paired.npz',gap=delta,names=names)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('medium',type=Path);p.add_argument('xl',type=Path);p.add_argument('output',type=Path);a=p.parse_args();run(a.medium,a.xl,a.output)
