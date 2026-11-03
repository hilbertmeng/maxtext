"""Publish early trajectory alongside the existing write-origin analysis."""
from pathlib import Path
import json,html,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;j=json.loads((R/'results.json').read_text());steps=j['manifest']['steps'];x=np.array(steps)/1000;rows=[j['rows'][str(t)] for t in steps]
D=R.parents[1]/'sites/bam-mudd-head-activation-20261023/dist/shorttrain/no-self-readout/hunit-spectral/write-origin/early-trajectory';A=D/'assets';A.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
def save(fig,name):
 fig.tight_layout();fig.savefig(A/(name+'.png'),bbox_inches='tight');fig.savefig(A/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
def axis(ax):ax.set_xlabel('Checkpoint (k steps)');ax.set_xticks(x);ax.grid(alpha=.2)
cs=[(0,'MHA','#009e73'),(1,'Dynamic M_o','#0072b2'),(2,'Static M_o','#777777'),(5,'Vector FFN','#cc79a7')]
fig,axs=plt.subplots(2,2,figsize=(13,8))
for b,c in [('fixed43k','#0072b2'),('own','#d55e00')]:
 axs[0,0].plot(x,[r[b]['head_weak_power']['mean'] for r in rows],'-o',color=c,label=b)
 axs[0,1].plot(x,[100*r[b]['head_weak_fraction']['mean'] for r in rows],'-o',color=c,label=b)
axs[0,1].plot(x,[100*r['gamma_isotropic_control']['centered_isotropic_input_expected_weak_fraction'] for r in rows],'--',c='#777777',label='Own: isotropic-input LN control')
for k,name,c in cs:
 axs[1,0].plot(x,[r['fixed43k']['raw_self_RMS'][k] for r in rows],'-o',c=c,label=name)
 axs[1,1].plot(x,[r['fixed43k']['head_self_RMS'][k] for r in rows],'-o',c=c,label=name)
for ax in axs.flat:axis(ax);ax.legend(fontsize=8)
axs[0,0].set(title='Final head weak-coordinate power',ylabel='E ||h U||² / D');axs[0,1].set(title='Weak power / total head power',ylabel='Percent (%)');axs[1,0].set(title='Final-layer writes: raw same-U, fixed43k',ylabel='Own weak RMS after W_o');axs[1,1].set(title='Final-layer writes: mapped through final LN',ylabel='Own weak RMS after LN map')
save(fig,'weak_trajectory')
fig,axs=plt.subplots(2,2,figsize=(13,8))
for col,b in enumerate(['fixed43k','own']):
 for k,name,c in cs:
  for row,space in enumerate(['raw','head']):
   axs[row,col].plot(x,[100*r[b][space+'_share']['mean'][k] for r in rows],'-o',c=c,label=name)
 for row,space in enumerate(['raw','head']):axs[row,col].set(title=f'{b}: {space} signed allocation',ylabel='Share of final weak energy (%)');axis(axs[row,col]);axs[row,col].legend(fontsize=8)
save(fig,'component_shares')
fig,axs=plt.subplots(2,2,figsize=(13,8))
for k,name,c in [(None,'Memory entering final read','#0072b2'),('raw_RMS','Pre-LN vector residual','#cc79a7')]:axs[0,0].plot(x,[r['memory_RMS'] if k is None else r[k] for r in rows],'-o',c=c,label=name)
axs[0,0].set(title='Physical memory and residual amplitude',ylabel='RMS')
for idx,name,c in [(0,'Dynamic M_o','#0072b2'),(1,'Static M_o','#777777'),(2,'Total M_o','#cc79a7')]:axs[0,1].plot(x,[r['read_three_RMS'][idx] for r in rows],'-o',c=c,label=name)
axs[0,1].plot(x,[r['mha_Mo_RMS'][0] for r in rows],'-o',c='#009e73',label='MHA');axs[0,1].set(title='Native final read before W_o',ylabel='RMS')
axs[1,0].plot(x,[r['head_parameters']['gamma_RMS'] for r in rows],'-o',label='LN gamma',c='#0072b2');axs[1,0].plot(x,[r['head_parameters']['bias_RMS'] for r in rows],'-o',label='LN beta',c='#cc79a7');axs[1,0].set(title='Actual final-LN parameters',ylabel='RMS')
axs[1,1].plot(x,[100*r['own_fixed43k_subspace_overlap'] for r in rows],'-o',c='#0072b2',label='Own vs fixed43k basis');axs[1,1].set(title='Weak-subspace overlap',ylabel='||U_own^T U_43k||_F² / 128 (%)')
for ax in axs.flat:axis(ax);ax.legend(fontsize=8)
save(fig,'memory_and_scales')
fig,axs=plt.subplots(1,2,figsize=(13,4))
for k,name,c in [('remove_mo','Remove final M_o','#0072b2'),('remove_mha','Remove final MHA','#009e73')]:
 v=np.array([r['causal'][k]['delta_CE']['mean'] for r in rows]);ci=np.array([r['causal'][k]['delta_CE']['ci95'] for r in rows]);axs[0].errorbar(x,v,yerr=np.stack([v-ci[:,0],ci[:,1]-v]),fmt='-o',color=c,label=name)
 axs[1].plot(x,[r['causal'][k]['weak_power_ratio_to_keep'] for r in rows],'-o',c=c,label=name)
axs[0].set(title='Final whole-branch removal: loss cost',ylabel='Paired CE difference, 95% text CI');axs[1].set(title='Removal: fixed43k weak energy left',ylabel='Weak power / keep');axs[1].axhline(1,color='#888',lw=.7)
for ax in axs:axis(ax);ax.legend(fontsize=8)
save(fig,'causal_trajectory')
def table(headers,rs):return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rs)+'</tbody></table></div>'
summary=[];control=[];verification=[]
for t,r,q in zip(steps,rows,j['verification']):
 summary.append([str(t//1000)+'k',f"{r['CE']['mean']:.5f}",f"{r['memory_RMS']:.3f}",f"{r['raw_RMS']:.3f}",f"{100*r['own']['head_weak_fraction']['mean']:.2f}%",f"{r['fixed43k']['raw_self_RMS'][1]:.3f}",f"{100*r['fixed43k']['raw_share']['mean'][1]:.2f}%",f"{100*r['fixed43k']['head_share']['mean'][1]:.2f}%"])
 control.append([str(t//1000)+'k',*[f"{r['causal'][name]['delta_CE']['mean']:+.6f}" for name in ['remove_mo','remove_mha']]])
 verification.append([t,f"{q['max_H_relative']:.4%}",f"{q['max_CE_difference']:.7f}",f"{q['max_weak_power_difference']:.4%}"])
for name in ['results.json','verification.json','arms.json']:shutil.copyfile(R/name,A/name)
conclusion=(R/'conclusion.html').read_text() if (R/'conclusion.html').exists() else '<p>数值已完成，解释整理中。</p>'
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HUnit BAM 3k至43k弱读出轨迹</title><style>body{font:16px/1.7 system-ui;max-width:1260px;margin:auto;padding:24px;background:#f4f7fa;color:#172b3a}section{background:#fff;padding:22px;margin:20px 0;border-radius:12px}img{width:100%;height:auto}.scroll{overflow:auto}table{border-collapse:collapse;font-size:14px;width:100%}td,th{padding:8px;text-align:left;border-bottom:1px solid #ddd;white-space:nowrap}code{overflow-wrap:anywhere}a{color:#006c9a}</style><a href="../">← 末层写入归因</a><h1>HUnit BAM：3k、13k、23k、33k、43k 的弱读出轨迹</h1><section>'''+conclusion+'''</section><section><h2>弱方向能量与末层各路写入</h2><img src="assets/weak_trajectory.png">'''+table(['checkpoint','同文本CE','末层读前M RMS','最终LN前x RMS','当期弱方向/head总能量','动态Mo原始弱RMS','动态Mo原始分摊','动态Mo映射分摊'],summary)+'''<p>底部两图使用固定43k的Wc最弱128方向，W_o之后测量，RMS分母为完整D=2560。原始弱投影与经过实际最终LN的映射分开。上方对照各checkpoint自己的最弱128方向，以防固定未来基底带来错误的早期预警结论。灰虚线是参数解析对照：令LN输入为零均值各向同性、忽略β，仅计算当前γ产生的预期弱能量比例；它不是实际文本激活，也不是干预因果实验。自己基底下的能量占比并非loss或梯度占比。</p></section><section><h2>动态Mo、静态Mo、MHA、FFN的有符号分摊</h2><img src="assets/component_shares.png"><p>份额按E⟨分量,最终弱投影⟩/E‖最终弱投影‖²分配交叉项，可负、可超过100%。没有显示的前31层累计项、LN bias和舍入项仍参与完整闭合检验。它衡量直接写入端；FFN输入已受M读取影响，不能由FFN归因排除M的上游作用。</p></section><section><h2>M幅度、读出幅度和最终LN缩放</h2><img src="assets/memory_and_scales.png"><p>M为末层原生read输入的未shift物理矩阵，并非把shift混合后的读视图另存一份；M_o/MHA为原生read/splash得到的实际前W_o输出。动态Mo=实际总Mo−同一原生读取但动态scale为0的静态Mo。M和x的RMS可比较数值尺度，不能解释为相同向量空间里的可加份额。子空间重合度1表示两组128维方向相同，0表示正交。</p></section><section><h2>逐checkpoint真实末层消融</h2><img src="assets/causal_trajectory.png">'''+table(['checkpoint','移除全部Mo的ΔCE','移除MHA的ΔCE'],control)+'''<p>末层W_o输入去掉其中整路，然后重新计算末层FFN与最终LN；其余参数与attention map保持原值。每点使用同32条文本，区间按文本bootstrap5000次。删除整路同时去掉强弱分量，不能用这个loss成本断言弱分量有益，也不是归一化后重新训练的结果。</p></section><section><h2>推理保真与口径</h2>'''+table(['checkpoint','最大H误差','最大单文本CE差','最大弱功率差'],verification)+'''<p>每条文本先运行无拆解的原生模型，再核验末层拆解的H、CE与弱投影功率；固定与当期两种基底都检查，所有160条都保留。所有统计matmul用HIGHEST，BF16整数RNE落实物理边界，原始/映射能量账本闭合。统一使用FastKernel推理；3k/13k/23k实际上在26k切换FastKernel前训练，因此这些点不是历史后端逐比特回放。此处固定的是架构数学和同一推理后端。</p><p>模型配置BamPythia2p8BHUnit300BFastKernel，实际RUN BamPythia2p8BHUnit300B；checkpoint内部step均比标签大1。固定32条Pile训练来源文本×2048预测位置，不是held-out指标或新训练实验。Wc=W−mean_vocab(W)，使用实际BF16有效head权重。固定43k方向用于事后追踪，早期实验的可观测判据应使用当期谱和相同方向内基线。</p><p>分析源码 <code>'''+j['manifest']['commit']+'''</code>。<a href="assets/results.json">完整数值与区间</a> · <a href="assets/verification.json">核验</a> · <a href="assets/arms.json">配置与来源</a>。PNG均有同名PDF。</p></section></html>'''
(D/'index.html').write_text(page)
# Idempotent attachment to the existing analysis page, also retained by its publisher.
parent=D.parent/'index.html';text=parent.read_text();start='<!-- EARLY_TRAJECTORY_LINK_START -->';end='<!-- EARLY_TRAJECTORY_LINK_END -->';link=start+'<section><h2>新增：3k至43k的训练轨迹</h2><p><a href="early-trajectory/">查看3k、13k、23k、33k、43k的弱方向写入、M RMS和逐checkpoint消融</a></p></section>'+end
if start in text:text=text[:text.index(start)]+text[text.index(end)+len(end):]
text=text.replace('</h1>','</h1>'+link,1);parent.write_text(text);(R/'parent_link.html').write_text(link);(R.parent/'bam_hunit_early_weak_20261103/parent_link.html').write_text(link)
print('EARLY_TRAJECTORY_PUBLISHED',D)
