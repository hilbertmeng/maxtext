"""Publish only validated carry-boundary/final-only attribution as primary results."""
from pathlib import Path
import json,html,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;j=json.loads((R/'results.json').read_text());assert j['complete'];D=R.parents[1]/'sites/bam-mudd-head-activation-20261023/dist/shorttrain/no-self-readout/hunit-spectral/write-origin';A=D/'assets';A.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False});steps=['33000','43000'];x=np.arange(1,33)
def save(fig,name):fig.tight_layout();fig.savefig(A/(name+'.png'),dpi=160,bbox_inches='tight');fig.savefig(A/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
def table(headers,rows):return '<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</table></div>'
fig,axs=plt.subplots(2,2,figsize=(14,8))
for t,c in zip(steps,['#0072b2','#d55e00']):
 r=j['rows'][t]['layer_boundaries']
 for ax,key,title in [(axs[0,0],'head_share','Mapped through actual final LN'),(axs[0,1],'raw_share','Centered residual before final LN')]:
  if r['accepted']:ax.plot(x,100*np.array(r[key]),'o-',ms=3,c=c,label=t[:-3]+'k')
  else:ax.text(.5,.5,'Boundary observation failed fidelity gate',ha='center',transform=ax.transAxes)
  ax.set(title=title,xlabel='Layer (1-based)',ylabel='Signed net layer allocation of final weak energy (%)');ax.axhline(0,c='#999',lw=.7);ax.grid(alpha=.2)
 for ax,key,title in [(axs[1,0],'head_share','Layers1-31: mapped through final LN'),(axs[1,1],'raw_share','Layers1-31: before final LN')]:
  if r['accepted']:ax.plot(x[:-1],100*np.array(r[key])[:-1],'o-',ms=3,c=c,label=t[:-3]+'k')
  ax.set(title=title,xlabel='Layer',ylabel='Signed net allocation (%)');ax.axhline(0,c='#999',lw=.7);ax.grid(alpha=.2)
for ax in axs.flat:ax.legend(fontsize=9)
save(fig,'layer_net_allocations')
fig,axs=plt.subplots(1,2,figsize=(14,4.5));ii=[0,1,2,5];labels=['MHA','Dynamic M_o','Static M_o','FFN'];xx=np.arange(4)
for t,c,shift in [('33000','#0072b2',-.16),('43000','#d55e00',.16)]:
 q=j['rows'][t]['focused_last_layer'];axs[0].bar(xx+shift,np.array(q['head_share'])[ii]*100,.3,color=c,label=t[:-3]+'k');axs[1].bar(xx+shift,np.array(q['raw_share'])[ii]*100,.3,color=c,label=t[:-3]+'k')
axs[0].set(title='Independent final-only observation: aligned allocation',ylabel='Share of final weak energy (%)');axs[1].set(title='Independent final-only observation: before final LN',ylabel='Share of centered residual weak energy (%)')
for ax in axs:ax.set_xticks(xx,labels);ax.legend();ax.grid(axis='y',alpha=.2)
save(fig,'final_layer_components')
fig,axs=plt.subplots(1,2,figsize=(14,4.5))
for ax,key,title in [(axs[0],'focused_last_layer_raw_growth_share','Before LN: sources of weak-energy growth'),(axs[1],'focused_last_layer_head_growth_share','After LN: sources of weak-energy growth')]:
 vals=np.array(j['growth'][key])[ii]*100;ax.bar(np.arange(4),vals,color=['#0072b2','#7b2cbf','#009e73','#d55e00']);ax.set_xticks(np.arange(4),labels);ax.set(title=title,ylabel='Allocation of33k to43k weak-energy growth (%)');ax.axhline(0,c='#777',lw=.8);ax.grid(axis='y',alpha=.2)
save(fig,'weak_energy_growth_sources')
fig,axs=plt.subplots(1,2,figsize=(13,4.5))
for shift,(name,label,c) in zip([-.16,.16],[('remove_mo','Remove final M_o','#7b2cbf'),('remove_mha','Remove final MHA','#0072b2')]):
 rr=[j['rows'][t]['last_layer_causal_controls'][name] for t in steps];y=np.array([r['delta_CE']['mean'] for r in rr]);ci=np.array([r['delta_CE']['ci95'] for r in rr]);pos=np.arange(2)+shift
 axs[0].bar(pos,y,.3,color=c,label=label);axs[0].errorbar(pos,y,yerr=np.maximum(0,np.vstack([y-ci[:,0],ci[:,1]-y])),fmt='none',ecolor='#222',capsize=3);axs[1].bar(pos,[r['weak_power_ratio_to_keep'] for r in rr],.3,color=c,label=label)
axs[0].set(title='Recompute final FFN and LN: loss effect',ylabel='Paired FP32 CE change vs keep');axs[1].set(title='Recompute final FFN and LN: weak energy',ylabel='Head weak power / keep');axs[0].axhline(0,c='#777',lw=.8);axs[1].axhline(1,c='#777',lw=.8)
for ax in axs:ax.set_xticks([0,1],['33k','43k']);ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
save(fig,'last_layer_controls')
component_labels=['MHA','动态M_o','静态M_o','Attention输出bias','Attention舍入项','向量FFN','残差相加舍入项'];body=[]
for k,label in enumerate(component_labels):body.append([label,*[f"{j['rows'][t]['focused_last_layer']['head_share'][k]:+.2%}" for t in steps],*[f"{j['rows'][t]['focused_last_layer']['raw_share'][k]:+.2%}" for t in steps]])
body.append(['M_o合计',*[f"{sum(j['rows'][t]['focused_last_layer']['head_share'][1:3]):+.2%}" for t in steps],*[f"{sum(j['rows'][t]['focused_last_layer']['raw_share'][1:3]):+.2%}" for t in steps]])
for k,label in enumerate(['前31层及Embedding的累计净项','最终LN bias','最终LN舍入项']):body.append([label,*[f"{j['rows'][t]['focused_last_layer']['prefix_and_LN_head_share'][k]:+.2%}" for t in steps],'—','—'])
rms=[]
for k,label in enumerate(component_labels):
 if k not in [0,1,2,5]:continue
 rms.append([label,*[f"{j['rows'][t]['focused_last_layer']['raw_self_RMS'][k]:.4f}" for t in steps],*[f"{j['rows'][t]['focused_last_layer']['head_self_RMS'][k]:.4f}" for t in steps]])
rms_section='<section><h2>各路自身的弱投影写入RMS</h2>'+table(['末层组分','33k LN前','43k LN前','33k映射后','43k映射后'],rms)+'<p>attention各路均在W_o之后测量，RMS以完整D=2560为分母。各路不是正交的，RMS平方不能直接相加；上面的可加分摊包含交叉项。</p></section>'
net=[]
for l in range(32):net.append([l+1,*[f"{j['rows'][t]['layer_boundaries']['head_share'][l]:+.3%}" if j['rows'][t]['layer_boundaries']['accepted'] else '未通过保真' for t in steps],*[f"{j['rows'][t]['layer_boundaries']['raw_share'][l]:+.3%}" if j['rows'][t]['layer_boundaries']['accepted'] else '未通过保真' for t in steps]])
causal=[]
for t in steps:
 for name,label in [('keep','保留'),('remove_mo','移除末层M_o'),('remove_mha','移除末层MHA')]:
  r=j['rows'][t]['last_layer_causal_controls'][name];v=r['delta_CE'];causal.append([t[:-3]+'k',label,f"{r['CE']['mean']:.6f}",f"{v['mean']:+.6f} [{v['ci95'][0]:+.6f},{v['ci95'][1]:+.6f}]",f"{r['weak_power_ratio_to_keep']:.3f}",f"{r['head_RMS']:.4f}",f"{r['raw_RMS']:.4f}"])
verification=[]
for t in steps:
 r=j['rows'][t];f=r['focused_last_layer'];b=r['layer_boundaries'];verification.append([t[:-3]+'k',f"{f['max_head_relative_RMS']:.4%}",f"{f['mean_CE_replay_difference']:+.7f}",f"{b['max_head_relative_RMS']:.4%}",str(b['accepted']),f"{f['full_vs_focus_max_absolute_share_difference']*100:.4f}"])
for f in ['results.json','verification.json','arms.json']:shutil.copyfile(R/f,A/('manifest.json' if f=='arms.json' else f))
conclusion=(R/'conclusion.html').read_text()
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HUnit BAM弱读出方向写入归因</title><style>body{font:16px/1.7 system-ui;max-width:1260px;margin:auto;padding:24px;background:#f4f7fa;color:#172b3a}section{background:white;padding:22px;margin:20px 0;border-radius:12px}img{width:100%}.scroll{overflow:auto}table{border-collapse:collapse;font-size:14px;width:100%}td,th{padding:8px;text-align:left;border-bottom:1px solid #ddd;white-space:nowrap}code{overflow-wrap:anywhere}a{color:#006c9a}</style><a href="../">← HUnit输出层谱分析</a><h1>HUnit BAM：哪些组分向弱读出方向写入最多？</h1><section>'''+conclusion+'''</section><section><h2>主测量：逐层净写入</h2><img src="assets/layer_net_allocations.png"><p>只观察每层的残差流边界，不给层内attention/FFN增加额外消费者。原始前传保真通过后，按实际最终LN映射分摊最终弱方向能量。每层为attention+FFN+舍入的净项；净项小不等于层无用，也不排除层内两路互相抵消。</p><details><summary>展开32层数值</summary>'''+table(['层','33k head弱能量','43k head弱能量','33k LN前弱能量','43k LN前弱能量'],net)+'''</details></section><section><h2>主测量：独立拆解最后一层</h2><img src="assets/final_layer_components.png">'''+table(['末层写入端','33k head弱能量','43k head弱能量','33k LN前弱能量','43k LN前弱能量'],body)+'''<p>前31层保持原执行路径，只在最后一层拆分MHA、动态/静态M_o和FFN。份额为有符号的同向分摊，含交叉项，可为负或超过100%；自身RMS与净同向贡献分开报告。前31层累计净项仍含所有早层M及向量计算的影响，不能解释为这些层不重要。</p></section>'''+rms_section+'''<section><h2>33k到43k的弱能量增长来自哪里</h2><img src="assets/weak_energy_growth_sources.png"><p>固定同一个43k Wc基底，按各路同向分摊的绝对能量差分，再除以总弱能量增量。LN前与经过最终LN后的口径分别显示，不能混在一起相加；这是观察轨迹的分摊，不是重新训练的因果证明。</p></section><section><h2>真实末层消融：重新计算FFN和最终LN</h2><img src="assets/last_layer_controls.png">'''+table(['checkpoint','干预','FP32 CE','ΔCE及95%文本区间','head弱能量/保留','head RMS','LN前残差RMS'],causal)+'''<p>最后一层W_o输入从MHA+M_o改为只保留其中一路；参数、Q/K/V和attention map保持原值，随后末层FFN与最终LN正常重算。三种模式共用同一个编译图、同32条2048-token文本，以文本bootstrap5000次。删除整路同时影响强/弱方向，因此其loss代价不能单独归给弱分量，也不等于重新训练收益。</p></section><section><h2>保真与独立复核</h2>'''+table(['checkpoint','末层观察最大H误差','末层观察平均CE差','层边界最大H误差','层边界是否接受','末层两种观察图最大差（百分点）'],verification)+'''<p>原始的完整层内重型观察图会改变BF16编译融合的数值路径，部分文本超过H保真门槛，未用于以上主图和主结论。主测量换成独立“只观察层边界”和“只观察末层”两条路径，沿用每条H相对误差1.5%、CE差0.005、cohort平均CE差0.001、弱能量差1.5%的门槛；实际误差见表。原重型图只作为探索性诊断留在JSON中，不筛掉不利文本来平均。</p></section><section><h2>定义、来源与范围</h2><p>W是lm_head的原生BF16有效权重，Wc=W−mean_vocab(W)，U取43k最小128个左奇异向量，两个checkpoint共用同一基底。z=hU，每项份额为E⟨δh_iU,z⟩/E‖z‖²，其中δh_i=[γ⊙(δx_i−mean_dimδx_i)]/r_t。r_t冻结为当前实际最终LN分母，所以这是直接写入记账，不是删除该项的非线性效果；真实消融另列。LN前的中心化残差另作同一U坐标下的几何对照，它不是对LN前完整有效读出矩阵另做SVD；判断读出影响以映射后的head口径为主。</p><p>静态M_o用同一原生Pallas读取、只将动态O scale设0得到；动态项为实际总M_o减静态项，保留该读出中的舍入差异。各路使用FP32线性的W_o，实际bias及BF16相加/投影舍入单列。整数RNE落实BF16边界，所有统计matmul显式HIGHEST；小模型核验原始/观察loss一致、扫描层间连续、能量账本闭合。</p><p>MHA与M_o都消费M中的信息；这里定位向量残差的直接写入端，未追溯M中信息来自哪次历史写入，也不等于训练原因。模型 <code>BamPythia2p8BHUnit300BFastKernel</code>，实际RUN <code>BamPythia2p8BHUnit300B</code>。标签33k/43k对应内部33001/43001。相同32条Pile训练来源文本，不是held-out验证集排名。</p><p>设备 <code>'''+html.escape(j['device'])+'''</code>；训练源码 <code>'''+j['manifest']['training_commit']+'''</code>；分析源码 <code>'''+j['manifest']['commit']+'''</code>。<a href="assets/results.json">完整数值与区间</a> · <a href="assets/verification.json">验证</a> · <a href="assets/manifest.json">来源</a>。各PNG有同名PDF。</p></section></html>'''
(D/'index.html').write_text(page);print('PRIMARY_WRITE_ORIGIN_REPORT_PUBLISHED',D)
