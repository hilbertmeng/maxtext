"""Publish new43k and explicitly reused cached history, after cleanup verification."""
from pathlib import Path
import html,json,shutil,time
from datetime import datetime,timezone
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
d=json.loads((R/'results.json').read_text());audit=json.loads((R/'resource-audit.json').read_text());assert d['complete'] and audit['all_absent']
assert d['measured_only']==dict(arm='prenorm',steps=[43000],sequences=32)
steps=[13000,23000,33000,43000];series={s:d['previous'][str(s)]['rows'] for s in steps[:-1]};series[43000]=d['rows']
D=R.parents[1]/'sites/bam-mudd-head-activation-20261023/dist/shorttrain/no-self-readout/hunit-spectral/write-origin/prenorm-43k';A=D/'assets';A.mkdir(exist_ok=True,parents=True)
labels={'baseline':'HUnit FastKernel (cached)','prenorm':'MPreNorm + QKScale'};colors={'baseline':'#3366a2','prenorm':'#aa4c00'}
fig,axs=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
for ax,arms in zip(axs,[['baseline','prenorm'],['prenorm']]):
 for arm in arms:
  vals=np.array([series[s][arm]['own']['head_weak_fraction']['mean'] for s in steps])*100;ci=np.array([series[s][arm]['own']['head_weak_fraction']['ci95'] for s in steps]).T*100
  ax.errorbar([13,23,33,43],vals,yerr=np.stack([vals-ci[0],ci[1]-vals]),marker='o',capsize=4,label=labels[arm],color=colors[arm])
  for x,y in zip([13,23,33,43],vals):ax.annotate(f'{y:.2f}%',(x,y),xytext=(0,8),textcoords='offset points',ha='center',fontsize=9)
 ax.set(xlabel='Checkpoint (k)',ylabel='% of final-LN head-input energy',xticks=[13,23,33,43]);ax.grid(alpha=.2);ax.legend(fontsize=8)
axs[0].set_title('Own centered Wc bottom5%: both models');axs[0].set_ylim(bottom=0)
axs[1].set_title('New model: magnified trajectory')
for ext in ['png','pdf']:fig.savefig(A/('head-energy.'+ext),dpi=170)
plt.close(fig)

def weakraw(r):return r['own'].get('raw_weak_RMS',float(np.sqrt(r['own']['raw_weak_power']['mean'])))
def weakgain(r):return r['fixed43k']['raw_self_RMS'][1]/r['read_three_RMS'][0]
fig,axs=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
specs=[('Raw residual before final LN','RMS',lambda r:r['raw_RMS']),('Raw residual in own weak directions','RMS / full D',weakraw),('Final dynamic Mo: own weak write','RMS after Wo',lambda r:r['own']['raw_self_RMS'][1]),('Final FFN: own weak write','RMS',lambda r:r['own']['raw_self_RMS'][5]),('Dynamic Mo gain in fixed old43k space','post-weak / pre-total RMS',weakgain),('Physical M entering final read','RMS',lambda r:r['memory_RMS'])]
for ax,(title,unit,fn) in zip(axs.flat,specs):
 for arm in labels:ax.plot([13,23,33,43],[fn(series[s][arm]) for s in steps],'o-',label=labels[arm],color=colors[arm])
 ax.set(title=title,xlabel='Checkpoint (k)',ylabel=unit,xticks=[13,23,33,43]);ax.grid(alpha=.2);ax.legend(fontsize=8)
for ext in ['png','pdf']:fig.savefig(A/('writes.'+ext),dpi=170)
plt.close(fig)
weights=json.loads((R.parent/'bam_hunit_prenorm_33k_20261104/wo-geometry.json').read_text())['rows']+json.loads((R/'wo-geometry.json').read_text())['rows'];wl={(x['arm'],x['step']):x for x in weights}
fig,axs=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
specs=[('sigma_min','Minimum singular value of Wc','Singular value'),('gamma_RMS','Final LayerNorm scale','RMS'),('fixed43k_isotropic_weak_gain','Wo average weak gain (fixed old43k)','RMS gain'),('alignment','Dynamic read / Wo joint alignment','actual gain / isotropic gain'),('fixed43k_projected_stable_rank','Weak-projected Wo stable rank','Frobenius^2 / top singular^2'),('grad','New model training raw_grad_norm','100-step window mean')]
for ax,(key,title,unit) in zip(axs.flat,specs):
 for arm in labels:
  vals=[]
  for s in steps:
   row=series[s][arm]
   if key in ['sigma_min','gamma_RMS']:v=row['head_parameters'][key]
   elif key=='grad':
    if arm=='baseline':v=np.nan
    else:
     report=R if s==43000 else Path(d['previous'][str(s)]['report']);h=json.loads((report/'tb_health.json').read_text());v=h['BamPythia2p8BHUnit300BFastKernelMPreNormQKScale']['metrics']['learning/raw_grad_norm']['mean']
   elif (arm,s) not in wl:v=np.nan
   elif key=='alignment':v=weakgain(row)/wl[arm,s]['fixed43k_isotropic_weak_gain']
   else:v=wl[arm,s][key]
   vals.append(v)
  if np.isfinite(vals).any():ax.plot([13,23,33,43],vals,'o-',label=labels[arm],color=colors[arm])
 ax.set(title=title,xlabel='Checkpoint (k)',ylabel=unit,xticks=[13,23,33,43]);ax.grid(alpha=.2);ax.legend(fontsize=8)
for ext in ['png','pdf']:fig.savefig(A/('readout-health.'+ext),dpi=170)
plt.close(fig)

def table(headers,body):return '<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in body)+'</table></div>'
a,b=d['rows']['baseline'],d['rows']['prenorm'];past=series[33000]['prenorm'];t=d['temporal']['33000-43000'];ci=t['head_fraction_delta_pp_ci95'];p=d['paired']['own_head_fraction_delta'];bf=b['own']['head_weak_fraction']['mean'];af=a['own']['head_weak_fraction']['mean']
trend='继续下降' if ci[1]<0 else ('出现回升' if ci[0]>0 else '变化区间包含0，尚无明确方向')
headline=f'43k：新模型弱能量占比{bf*100:.2f}%，33k→43k{trend}'
history=table(['模型','13k','23k','33k','43k'],[[labels[arm]]+[f"{series[s][arm]['own']['head_weak_fraction']['mean']*100:.2f}%" for s in steps] for arm in labels])
metrics=[('当期最弱5% head能量占比',lambda r:f"{r['own']['head_weak_fraction']['mean']*100:.2f}%"),('最终LN前残差总RMS',lambda r:f"{r['raw_RMS']:.3f}"),('当期弱方向原始残差RMS',lambda r:f"{weakraw(r):.3f}"),('最终LN后head总RMS',lambda r:f"{r['head_RMS']:.3f}"),('Wc最小奇异值',lambda r:f"{r['head_parameters']['sigma_min']:.3f}"),('LN γ RMS',lambda r:f"{r['head_parameters']['gamma_RMS']:.3f}"),('末层物理M RMS',lambda r:f"{r['memory_RMS']:.3f}"),('前Wo动态Mo总RMS',lambda r:f"{r['read_three_RMS'][0]:.3f}"),('末层动态Mo当期弱写入RMS',lambda r:f"{r['own']['raw_self_RMS'][1]:.3f}"),('末层FFN当期弱写入RMS',lambda r:f"{r['own']['raw_self_RMS'][5]:.3f}"),('移除末层整路Mo ΔCE',lambda r:f"{r['causal']['remove_mo']['delta_CE']['mean']:+.5f}"),('移除末层整路MHA ΔCE',lambda r:f"{r['causal']['remove_mha']['delta_CE']['mean']:+.5f}")]
summary=table(['指标','新33k（复用）','新43k（新增）','旧43k（复用）'],[[name,get(past),get(b),get(a)] for name,get in metrics])
components=table(['末层组分','新33k：弱写入RMS','新43k：弱写入RMS','旧43k：弱写入RMS','新43k：head弱功率有符号分摊'],[[name,f"{past['own']['raw_self_RMS'][i]:.4f}",f"{b['own']['raw_self_RMS'][i]:.4f}",f"{a['own']['raw_self_RMS'][i]:.4f}",f"{b['own']['head_share']['mean'][i]*100:.2f}%"] for i,name in enumerate(d['components'])])
wg=wl['prenorm',43000]['fixed43k_isotropic_weak_gain']/wl['prenorm',33000]['fixed43k_isotropic_weak_gain'];ag=t['fixed43k_dynamicMo_effective_weak_gain_ratio'];align=ag/wg;lc=t['common_fp32_CE_delta']
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HUnit MPreNormQKScale 43k弱能量分析</title><style>body{font:16px/1.7 system-ui;max-width:1250px;margin:auto;padding:24px;background:#f4f7fa;color:#172b3a}section{background:white;padding:24px;margin:20px 0;border-radius:12px}img{width:100%;height:auto}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:9px;border-bottom:1px solid #ddd;text-align:left;white-space:nowrap}a{color:#006c9a}code{overflow-wrap:anywhere}</style><a href="../prenorm-33k/">← 33k分析</a><h1>'''+headline+'''</h1><section><p><strong>本次仅新增运行新模型43k的32条×2048-token前传与末层整路消融。13k/23k/33k和旧模型43k全部复用已验证结果。</strong></p>'''+f'<p>各自当期Wc最弱128/2560方向的最终LN后head激活能量占比：新模型43k为{bf*100:.2f}%，对照旧模型43k为{af*100:.2f}%。新模型33k→43k变化{t["head_fraction_delta_pp"]:+.2f}个百分点，配对95%区间[{ci[0]:+.2f},{ci[1]:+.2f}]；{trend}。新43k−旧43k差{p["mean"]*100:+.2f}个百分点，配对95%区间[{p["ci95"][0]*100:+.2f},{p["ci95"][1]*100:+.2f}]。</p>'+history+'''<img src="assets/head-energy.png"></section><section><h2>弱方向幅度与末层写入</h2>'''+f'<p>新模型33k→43k原始残差总RMS {past["raw_RMS"]:.3f}→{b["raw_RMS"]:.3f}，当期弱方向原始RMS {weakraw(past):.3f}→{weakraw(b):.3f}；末层动态Mo弱写入 {past["own"]["raw_self_RMS"][1]:.3f}→{b["own"]["raw_self_RMS"][1]:.3f}，FFN弱写入 {past["own"]["raw_self_RMS"][5]:.3f}→{b["own"]["raw_self_RMS"][5]:.3f}。LN前绝对幅度和LN后能量占比是不同指标，需要分开判断。</p>'+'''<img src="assets/writes.png">'''+summary+components+'''<p>attention各路在W_o后计量，FFN是直接向量残差写入；RMS分母为完整D=2560。有符号分摊包含交叉项，可负或超过100%，不能当作独立非负能量或因果贡献。消融覆盖末层整路的所有head，并重算后续FFN/LN，不是只移除弱方向。</p></section><section><h2>读出几何、方向增益与训练梯度</h2><img src="assets/readout-health.png">'''+f'<p>同一固定旧43k弱空间上，新模型动态Mo有效弱增益33k→43k变为{ag:.3f}× = W_o各向同性平均弱增益{wg:.3f}× × 读出—W_o联合方向匹配{align:.3f}×。方向匹配同时依赖读出协方差及W_o谱形状/方向，不是纯激活变化或独立因果效应。stable rank为Frobenius范数平方除以最大奇异值平方，反映算子集中度，不是矩阵实际rank。旧模型43k的W_o算子诊断未在此图补算，相关曲线截止33k。</p>'+f'<p>新模型统一FP32 head CE从33k的{past["common_fp32_CE"]["mean"]:.5f}到43k的{b["common_fp32_CE"]["mean"]:.5f}，配对变化{lc["mean"]:+.5f}，95%区间[{lc["ci95"][0]:+.5f},{lc["ci95"][1]:+.5f}]。旧43k档案记录的是原生精度CE，未新增统一FP32 head计算，因此不据此声称匹配精度的跨模型loss差。</p>'+'''<p>训练梯度取各checkpoint前100-step窗口的10个日志点，是训练监控参考，不是本32条诊断文本的梯度，更不是弱方向梯度能量分解。</p></section><section><h2>口径与核验</h2>'''+f'<p>新模型完整名为 <code>{d["models"]["prenorm"]["experiment"]}</code>，checkpoint标签43000、内部step43001；原模型为 <code>BamPythia2p8BHUnit300BFastKernel</code>，实际RUN <code>BamPythia2p8BHUnit300B</code>。所有结果使用同一组32条固定Pile来源文本及原生v5p单芯片环境。Wc=W−mean_vocab(W)，取实际BF16有效参数的最弱5%方向；主指标是最终LN后head激活能量占比，原始残差投影复用这些head坐标，是几何对照，不是原始空间有效算子的最弱子空间。</p>'+f'<p>新43k全部32条通过原阈值：最大观察/原生H相对差{d["verification"]["prenorm"]["max_H_relative"]:.3g}，最大单文本CE差{d["verification"]["prenorm"]["max_CE_difference"]:.3g}，最大弱功率相对差{d["verification"]["prenorm"]["max_weak_power_relative"]:.3g}；能量账本闭合，不丢弃不利样本。旧模型43k直接复用原有核验记录。所有本任务分析节点/队列已确认删除；未干预正式训练。</p>'+'''<p>新配置同时包含M读前归一化、QK读scale和训练logits精度差异；结果不能单独因果归因于preNorm。弱激活能量与梯度能量、训练收益不同。本批32条文本不代表完整验证集。</p><p><a href="assets/results.json">完整数值与区间</a> · <a href="assets/verification.json">核验</a> · <a href="assets/resource-audit.json">资源回收</a> · <a href="assets/head-energy.pdf">主指标PDF</a> · <a href="assets/writes.pdf">写入图PDF</a> · <a href="assets/readout-health.pdf">健康图PDF</a></p></section></html>'''
(D/'index.html').write_text(page)
for n in ['results.json','verification.json','resource-audit.json','wo-geometry.json']:shutil.copy2(R/n,A/n)
parent=D.parent
for pth in [parent/'index.html',parent/'prenorm-33k/index.html']:
 text=pth.read_text()
 if 'prenorm-43k/' not in text:pth.write_text(text.replace('</h1>','</h1><p><a href="'+('prenorm-43k/' if pth.parent==parent else '../prenorm-43k/')+'">新增：43k弱能量分析（历史结果全部复用）</a></p>',1))
url=json.loads((R/'task.json').read_text())['report_url']
message=(f'新模型43k弱能量分析完成：仅新增运行新43k，旧模型和之前步数全部复用。\n'
 f'主指标轨迹13k→23k→33k→43k：\n新模型 '+ ' → '.join(f"{series[s]['prenorm']['own']['head_weak_fraction']['mean']*100:.2f}%" for s in steps)+'\n旧模型 '+ ' → '.join(f"{series[s]['baseline']['own']['head_weak_fraction']['mean']*100:.2f}%" for s in steps)+f'\n新模型33k→43k变化{t["head_fraction_delta_pp"]:+.2f}个百分点，配对95%区间[{ci[0]:+.2f},{ci[1]:+.2f}]，{trend}。\n末层动态Mo弱写入RMS {past["own"]["raw_self_RMS"][1]:.3f}→{b["own"]["raw_self_RMS"][1]:.3f}，FFN {past["own"]["raw_self_RMS"][5]:.3f}→{b["own"]["raw_self_RMS"][5]:.3f}。\n全部32条通过核验，分析TPU和队列已删除。激活弱能量不等同梯度弱能量。\n图表与完整分析：{url}')
(R/'final-message.txt').write_text(message)
(R/'completion.json').write_text(json.dumps(dict(complete=True,checked_utc=datetime.now(timezone.utc).isoformat(),measured_only=d['measured_only'],reused='all prior steps and baseline43k',resources_released=True,url=url),indent=2)+'\n')
print(message)
