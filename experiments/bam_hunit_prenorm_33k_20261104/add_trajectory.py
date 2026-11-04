"""Paired three-checkpoint trajectory; keep moving and fixed bases distinct."""
from pathlib import Path
import json,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
steps=[13000,23000,33000]
reports={13000:R.parent/'bam_hunit_prenorm_13k_20261103',23000:R.parent/'bam_hunit_prenorm_23k_20261103',33000:R}
summaries={s:json.loads((r/'results.json').read_text()) for s,r in reports.items()}
D=R.parents[1]/'sites/bam-mudd-head-activation-20261023/dist/shorttrain/no-self-readout/hunit-spectral/write-origin/prenorm-33k'
A=D/'assets';A.mkdir(exist_ok=True,parents=True)
boot=np.random.default_rng(20261103).integers(0,32,(5000,32))
series={};raw={};temporal={}
for arm in ['baseline','prenorm']:
 series[arm]={};raw[arm]={};temporal[arm]={}
 for s in steps:
  m=summaries[s]['models'][arm];assert m['cohort_hashes']==summaries[13000]['models']['baseline']['cohort_hashes']
  vals=[];root=Path(m['local_DATA'])/str(s)
  for i in range(32):
   q=json.loads((root/f'sequence_{i:03d}.json').read_text());assert q['state_step']==s+1 and q['input_hash']==m['cohort_hashes'][i]
   with np.load(root/f'sequence_{i:03d}.npz') as z:vals.append(dict(hp=float(z['own_head_total_power']),p=float(z['baseline_head_power']),mo=float(z['fixed43k_raw_self_power'][0,1]),read=float(z['fixed43k_read_three_power'][0].mean())))
  raw[arm][s]={k:np.array([v[k] for v in vals]) for k in vals[0]}
  row=summaries[s]['rows'][arm]
  series[arm][str(s)]=dict(own_head_fraction=row['own']['head_weak_fraction'],own_raw_weak_RMS=row['own']['raw_weak_RMS'],fixed43k_final_dynamicMo_weak_RMS=row['fixed43k']['raw_self_RMS'][1],fixed43k_final_FFN_weak_RMS=row['fixed43k']['raw_self_RMS'][5],fixed43k_dynamicMo_effective_weak_gain=row['fixed43k']['raw_self_RMS'][1]/row['read_three_RMS'][0],physical_M_RMS=row['memory_RMS'],dynamicMo_preWo_RMS=row['read_three_RMS'][0])
 for sa,sb in [(13000,23000),(23000,33000),(13000,33000)]:
  a,b=raw[arm][sa],raw[arm][sb]
  fd=b['hp'][boot].mean(1)/b['p'][boot].mean(1)-a['hp'][boot].mean(1)/a['p'][boot].mean(1)
  gain=lambda v,ix:np.sqrt(v['mo'][ix].mean(-1)/v['read'][ix].mean(-1))
  g=gain(b,boot)/gain(a,boot)
  temporal[arm][f'{sa}-{sb}']=dict(head_fraction_delta_pp=float((b['hp'].mean()/b['p'].mean()-a['hp'].mean()/a['p'].mean())*100),head_fraction_delta_pp_ci95=(np.quantile(fd,[.025,.975])*100).tolist(),fixed43k_dynamicMo_effective_weak_gain_ratio=float(np.sqrt((b['mo'].mean()/b['read'].mean())/(a['mo'].mean()/a['read'].mean()))),effective_weak_gain_ratio_ci95=np.quantile(g,[.025,.975]).tolist())

weights=json.loads((R/'wo-geometry.json').read_text());lookup={(v['arm'],v['step']):v for v in weights['rows']}
for arm in ['baseline','prenorm']:
 for step in steps:
  v=lookup[arm,step];row=series[arm][str(step)]
  row['Wo_isotropic_weak_gain']=v['fixed43k_isotropic_weak_gain']
  row['read_Wo_alignment_factor']=row['fixed43k_dynamicMo_effective_weak_gain']/v['fixed43k_isotropic_weak_gain']
  row['Wo_weak_projected_top_singular']=v['fixed43k_projected_operator_top_singular']
  row['Wo_weak_projected_stable_rank']=v['fixed43k_projected_stable_rank']
 for sa,sb in [(13000,23000),(23000,33000),(13000,33000)]:
  q=temporal[arm][f'{sa}-{sb}'];ir=lookup[arm,sb]['fixed43k_isotropic_weak_gain']/lookup[arm,sa]['fixed43k_isotropic_weak_gain']
  q['Wo_isotropic_weak_gain_ratio']=ir
  q['read_Wo_alignment_factor_ratio']=q['fixed43k_dynamicMo_effective_weak_gain_ratio']/ir
  q['read_Wo_alignment_factor_ratio_ci95']=(np.array(q['effective_weak_gain_ratio_ci95'])/ir).tolist()
shutil.copy2(R/'wo-geometry.json',A/'wo-geometry.json')

out=dict(series=series,paired_temporal=temporal,scope='Same32x2048 texts. Own contemporaneous Wc bottom128 primary. Old43k fixed basis is retrospective common geometry. Gain=postWo raw weak RMS/preWo dynamic read total RMS; combines Wo and activation direction, not operator singular value or isolated causal effect.')
(R/'cross-step.json').write_text(json.dumps(out,indent=2)+'\n');shutil.copy2(R/'cross-step.json',A/'cross-step.json')
fig,axs=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
specs=[('own_head_fraction','Own Wc bottom5% head energy','%'),('own_raw_weak_RMS','Own weak directions: raw residual','RMS'),('fixed43k_final_dynamicMo_weak_RMS','Final dynamic Mo weak write (fixed old43k)','RMS after Wo'),('fixed43k_final_FFN_weak_RMS','Final FFN weak write (fixed old43k)','RMS'),('fixed43k_dynamicMo_effective_weak_gain','Dynamic Mo effective weak gain (fixed old43k)','post-weak / pre-total RMS'),('physical_M_RMS','Physical final-layer memory','RMS')]
for ax,(k,title,ylabel) in zip(axs.flat,specs):
 for arm,label,color in [('baseline','HUnit FastKernel','#3366a2'),('prenorm','MPreNorm + QKScale','#aa4c00')]:
  values=[series[arm][str(s)][k] for s in steps]
  if k=='own_head_fraction':
   vals=np.array([v['mean']*100 for v in values]);ci=np.array([v['ci95'] for v in values]).T*100
   ax.errorbar([s/1000 for s in steps],vals,yerr=np.stack([vals-ci[0],ci[1]-vals]),marker='o',capsize=4,label=label,color=color)
  else:ax.plot([s/1000 for s in steps],values,'o-',label=label,color=color)
 ax.set(title=title,xlabel='Checkpoint (k)',ylabel=ylabel,xticks=[13,23,33]);ax.grid(alpha=.2);ax.legend(fontsize=8)
for ext in ['png','pdf']:fig.savefig(A/('cross-step.'+ext),dpi=170)
plt.close(fig)
n=temporal['prenorm']['23000-33000'];s=series['prenorm']
frac=' → '.join(f"{s[str(t)]['own_head_fraction']['mean']*100:.2f}%" for t in steps)
oldfrac=' → '.join(f"{series['baseline'][str(t)]['own_head_fraction']['mean']*100:.2f}%" for t in steps)
health_rows=[]
for step in steps:
 h=json.loads((reports[step]/'tb_health.json').read_text())
 cells=[]
 for arm in ['baseline','prenorm']:
  row=summaries[step]['rows'][arm];hm=h[summaries[step]['models'][arm]['run_name']]['metrics']
  cells.extend([f"{row['head_parameters']['sigma_min']:.3f}",f"{row['head_parameters']['gamma_RMS']:.3f}",f"{row['own_head_fraction_over_gamma_isotropic']:.2f}",f"{hm['learning/raw_grad_norm']['mean']:.4f}",f"{hm['raw_grads/decoder/lm_head/logits_dense']['mean']:.4f}"])
 health_rows.append('<tr><td>'+str(step)+'</td>'+''.join('<td>'+c+'</td>' for c in cells)+'</tr>')
health_table='<div class="scroll"><table><tr><th>step</th>'+''.join('<th>'+model+' '+metric+'</th>' for model in ['旧','新'] for metric in ['最小奇异值','LN γ RMS','弱占比/γ对照','raw_grad_norm','lm_head raw grad norm'])+'</tr>'+''.join(health_rows)+'</table></div>'
text=f'''<section><h2>13k→23k→33k：弱能量是否继续累积？</h2><img src="assets/cross-step.png"><p>同32条文本。新模型当期最弱5% head能量占比：{frac}；原模型：{oldfrac}。新模型23k→33k的配对变化为{n['head_fraction_delta_pp']:+.2f}个百分点，95%区间[{n['head_fraction_delta_pp_ci95'][0]:+.2f},{n['head_fraction_delta_pp_ci95'][1]:+.2f}]。</p><p>公共固定43k方向上，新模型动态Mo的有效弱方向增益（post-Wo弱RMS / pre-Wo动态总RMS）23k→33k变为{n['fixed43k_dynamicMo_effective_weak_gain_ratio']:.3f}×，配对95%区间[{n['effective_weak_gain_ratio_ci95'][0]:.3f},{n['effective_weak_gain_ratio_ci95'][1]:.3f}]。这个比值同时包含W_o和读出方向，不是单独W_o谱范数；各自当期弱空间的总占比与固定方向的局部增益是不同观测。</p>{health_table}<p>梯度取每个checkpoint前100步窗口的实际训练日志，各10个记录；不同于诊断文本的梯度，也不等于弱方向梯度分解。</p><p><a href="assets/cross-step.json">跨步数完整数值与区间</a> · <a href="assets/cross-step.pdf">PDF</a>。原始残差RMS的上升不自动代表最终LN后的弱能量占比上升，也不能直接等同训练梯度范数的变化。</p></section>'''
p=D/'index.html';p.write_text(p.read_text().replace('<section><h2>归因边界与保真核验</h2>',text+'<section><h2>归因边界与保真核验</h2>'))
print(json.dumps(temporal,ensure_ascii=False))


fig,axes=plt.subplots(1,3,figsize=(15,4.4),constrained_layout=True)
for ax,(key,title,unit) in zip(axes,[('Wo_isotropic_weak_gain','Wo isotropic weak-output gain','RMS gain'),('read_Wo_alignment_factor','Dynamic read / Wo joint alignment','actual gain / isotropic gain'),('Wo_weak_projected_stable_rank','Weak-projected Wo: stable rank','Frobenius^2 / top singular^2')]):
 for arm,label,color in [('baseline','HUnit FastKernel','#3366a2'),('prenorm','MPreNorm + QKScale','#aa4c00')]:
  ax.plot([13,23,33],[series[arm][str(step)][key] for step in steps],'o-',label=label,color=color)
 ax.set(title=title,xlabel='Checkpoint (k)',ylabel=unit,xticks=[13,23,33]);ax.grid(alpha=.2);ax.legend(fontsize=8)
for ext in ['png','pdf']:fig.savefig(A/('wo-decomposition.'+ext),dpi=170)
plt.close(fig)
wnew=temporal['prenorm']['23000-33000'];wold=temporal['baseline']['23000-33000']
weight_section=f"""<section><h2>W_o平均幅度、方向匹配与集中度</h2><img src="assets/wo-decomposition.png"><p>使用同一固定旧43k弱空间U，P=C_D U与原始弱写入口径一致。定义g_iso=‖W_o P‖F/√2560：输入为单位RMS各向同性向量时，弱输出RMS的期望增益；实际g=动态Mo在W_o后的弱RMS / W_o前动态Mo总RMS。方向匹配因子a=g/g_iso，因此时间变化满足g比值=g_iso比值×a比值。这是二阶几何分解，a同时依赖读出协方差和W_o的谱形状/方向，不能视为纯读出变化或独立因果效应。</p><p>23k→33k：新模型实际弱增益{wnew['fixed43k_dynamicMo_effective_weak_gain_ratio']:.3f}× = W_o平均弱增益{wnew['Wo_isotropic_weak_gain_ratio']:.3f}× × 方向匹配{wnew['read_Wo_alignment_factor_ratio']:.3f}×；旧模型分别为{wold['fixed43k_dynamicMo_effective_weak_gain_ratio']:.3f}× = {wold['Wo_isotropic_weak_gain_ratio']:.3f}× × {wold['read_Wo_alignment_factor_ratio']:.3f}×。方向匹配比值的配对区间见JSON。</p><p>stable rank=‖W_o P‖F²/σ_max(W_o P)²。较低表示该算子更集中在少数强输入方向，不是实际矩阵rank。这个投影在最终LN前计量；不能直接等同经过LN后的head能量占比，也不能从权重谱单独判断实际激活写入。<a href="assets/wo-geometry.json">权重几何</a> · <a href="assets/wo-decomposition.pdf">PDF</a>。</p></section>"""
p=D/'index.html';p.write_text(p.read_text().replace('<section><h2>归因边界与保真核验</h2>',weight_section+'<section><h2>归因边界与保真核验</h2>'))
