"""Publish the paired comparison beside the existing HUnit spectral report."""
from pathlib import Path
import html, json, shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

R=Path(__file__).resolve().parent
j=json.loads((R/'results.json').read_text())
D=R.parents[1]/'sites/bam-mudd-head-activation-20261023/dist/shorttrain/no-self-readout/hunit-spectral/write-origin/prenorm-13k'
A=D/'assets';A.mkdir(parents=True,exist_ok=True)
arms=['baseline','prenorm'];labels=['HUnit FastKernel','MPreNorm + QKScale'];colors=['#3366a2','#aa4c00']
rows=[j['rows'][a] for a in arms]
fig,axs=plt.subplots(2,3,figsize=(15,8.5),constrained_layout=True)
for a in axs.flat:a.grid(axis='y',alpha=.2);a.set_axisbelow(True)
def bar(ax,values,title,ylabel):
    ax.bar(labels,values,color=colors);ax.set_title(title);ax.set_ylabel(ylabel)
    for i,v in enumerate(values):ax.text(i,v,f'{v:.3g}',ha='center',va='bottom')
bar(axs[0,0],[r['own']['head_weak_fraction']['mean']*100 for r in rows],'Own Wc bottom 5%: head energy','% of final-LN output energy')
bar(axs[0,1],[r['raw_RMS'] for r in rows],'Residual RMS before final LN','RMS')
bar(axs[0,2],[r['own']['raw_weak_RMS'] for r in rows],'Own weak directions: raw residual','RMS (full D denominator)')
bar(axs[1,0],[r['head_parameters']['sigma_min'] for r in rows],'Centered effective head: min singular value','Singular value')
x=np.arange(4)
for ai,r in enumerate(rows):
    vals=np.asarray(r['own']['raw_self_RMS'])[[0,1,2,5]]
    axs[1,1].bar(x+(-.2 if ai==0 else .2),vals,.38,label=labels[ai],color=colors[ai])
axs[1,1].set_xticks(x,['MHA','Dynamic Mo','Static Mo','FFN']);axs[1,1].set_title('Final layer: own-weak residual writes');axs[1,1].set_ylabel('RMS after Wo (full D denominator)');axs[1,1].legend(fontsize=8)
bar(axs[1,2],[r['common_fp32_CE']['mean'] for r in rows],'Same texts, common FP32 output-head CE','Nats/token')
for ext in ['png','pdf']:fig.savefig(A/('comparison.'+ext),dpi=170)
plt.close(fig)
def table(headers,body):
    return '<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in body)+'</table></div>'
metrics=[('同文本、统一FP32 head CE',lambda r:f"{r['common_fp32_CE']['mean']:.5f}"),
 ('原生配置CE',lambda r:f"{r['native_CE']['mean']:.5f}"),
 ('最终LN前残差RMS',lambda r:f"{r['raw_RMS']:.3f}"),
 ('最终LN后head输入RMS',lambda r:f"{r['head_RMS']:.3f}"),
 ('当期最弱5%占head能量',lambda r:f"{100*r['own']['head_weak_fraction']['mean']:.2f}%"),
 ('当期最弱5%占零均值原始残差能量',lambda r:f"{100*r['own']['raw_centered_weak_fraction']['mean']:.2f}%"),
 ('当期弱投影原始RMS',lambda r:f"{r['own']['raw_weak_RMS']:.3f}"),
 ('最小奇异值',lambda r:f"{r['head_parameters']['sigma_min']:.3f}"),
 ('γ各向同性解析对照',lambda r:f"{100*r['head_parameters']['gamma_isotropic_control']:.2f}%"),
 ('实际弱占比/γ解析对照',lambda r:f"{r['own_head_fraction_over_gamma_isotropic']:.2f}×"),
 ('末层物理M RMS',lambda r:f"{r['memory_RMS']:.3f}"),
 ('移除末层整路Mo ΔCE',lambda r:f"{r['causal']['remove_mo']['delta_CE']['mean']:+.5f}"),
 ('移除末层整路MHA ΔCE',lambda r:f"{r['causal']['remove_mha']['delta_CE']['mean']:+.5f}")]
summary=table(['指标@13k']+labels,[[name]+[get(r) for r in rows] for name,get in metrics])
components=[]
for i,name in enumerate(j['components']):
    components.append([name]+[f"{r['own']['raw_self_RMS'][i]:.4f}" for r in rows]+[f"{r['own']['head_share']['mean'][i]*100:.2f}%" for r in rows])
vtable=table(['模型','文本数','最大H相对差','最大单文本CE差','最大弱功率相对差'],
 [[labels[i],j['verification'][a]['sequences'],f"{j['verification'][a]['max_H_relative']:.3g}",f"{j['verification'][a]['max_CE_difference']:.3g}",f"{j['verification'][a]['max_weak_power_relative']:.3g}"] for i,a in enumerate(arms)])
conclusion=(R/'conclusion.html').read_text() if (R/'conclusion.html').exists() else '<p>完整配对数据已计算；下表使用所有32条文本。</p>'
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HUnit MPreNorm 13k 弱方向能量比较</title><style>body{font:16px/1.7 system-ui;max-width:1250px;margin:auto;padding:24px;background:#f4f7fa;color:#172b3a}section{background:white;padding:24px;margin:20px 0;border-radius:12px}img{width:100%;height:auto}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:9px;border-bottom:1px solid #ddd;text-align:left;white-space:nowrap}code{overflow-wrap:anywhere}a{color:#006c9a}</style><a href="../early-trajectory/">← 原模型3k–43k轨迹</a><h1>HUnit：M读前归一化 + QKScale，在13k是否改善弱方向能量？</h1><section>'''+conclusion+'''</section><section><h2>同32条文本的配对结果</h2><img src="assets/comparison.png">'''+summary+'''<p>两模型全名为 <code>BamPythia2p8BHUnit300BFastKernel</code>（实际RUN <code>BamPythia2p8BHUnit300B</code>）和 <code>BamPythia2p8BHUnit300BFastKernelMPreNormQKScale</code>，都取checkpoint标签13000、内部step13001。同一台spot v5p-8，四个独立单芯片batch1进程，固定32条Pile训练来源文本，每条2048预测位置。本表不代表完整验证集或训练batch1024的梯度。</p><p>Wc=W−mean_vocab(W)，对实际BF16有效lm_head参数做谱分解。主指标取各模型自己的最弱128/2560方向，弱占比按总能量之比汇总；95%区间与配对差异以文本bootstrap5000次，见完整JSON。两个模型的弱子空间重合度为 '''+f"{j['own_subspace_overlap']:.3f}"+'''。固定旧43k方向的几何对照也保留在JSON中，不替代各自当期弱方向指标。</p></section><section><h2>末层各组分：弱方向上的写入</h2>'''+table(['组分','旧：原始弱RMS','新：原始弱RMS','旧：head弱功率分摊','新：head弱功率分摊'],components)+'''<p>attention各路在W_o后计量，FFN为直接残差写入。RMS分母为完整D=2560；各自弱子空间用于匹配读出难度，不代表相同物理方向。映射份额为E⟨分量,最终弱投影⟩/E‖最终弱投影‖²，包括交叉项，可负或超过100%，不能当作独立非负能量。前31层累计项、LN bias及舍入也进入闭合账本。末层消融移除整路所有head，并重算下游FFN/LN；不是只删除弱方向。</p></section><section><h2>归因边界与保真核验</h2>'''+vtable+'''<p>新模型同时增加M读前RMS归一化、local QK共同scale、FP32 logits输出；因此不能把模型间loss差全部归因于M归一化。统一FP32 head CE使用各自原生隐藏状态和BF16有效权重，消除评估时logits量化口径差，但不消除训练期间精度差的影响。弱方向激活能量与lm_head梯度能量不同；单个13k点不能证明中后期raw_grad_norm已稳定。</p><p>每个模型每条文本的末层观察与原生前传都通过原阈值：H和弱功率相对差&lt;1.5%，单文本CE差&lt;.005；代数账本闭合，不删除不利样本。旧模型同硬件历史复测差异见完整核验JSON。所有checkpoint只读，未跑训练。</p><p><a href="assets/results.json">完整数值、配对差异和区间</a> · <a href="assets/verification.json">核验</a> · <a href="assets/comparison.pdf">PDF图</a> · <a href="assets/resource-audit.json">资源回收状态</a></p></section></html>'''
D.joinpath('index.html').write_text(page)
for name in ['results.json','verification.json','resource-audit.json']:
    if (R/name).exists():shutil.copy2(R/name,A/name)
print('http://172.29.142.228:8873/shorttrain/no-self-readout/hunit-spectral/write-origin/prenorm-13k/')
