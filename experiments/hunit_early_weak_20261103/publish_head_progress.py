"""Publish completed head-only evidence while native inference waits for capacity."""
from pathlib import Path
import json,html,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;m=json.loads((R/'arms.json').read_text());H=Path(m['local_DATA']).parent/'head_spectra';D=R.parents[1]/'sites/bam-mudd-head-activation-20261023/dist/shorttrain/no-self-readout/hunit-spectral/write-origin/early-trajectory';A=D/'assets';A.mkdir(exist_ok=True,parents=True);steps=m['steps'];control=json.loads((R/'gamma_isotropic_control.json').read_text());uf=np.load(H/'43000.npy');rows=[]
for t in steps:
 q=json.loads((H/f'{t}.json').read_text());u=np.load(H/f'{t}.npy');q['future_basis_overlap']=float(np.sum((u.T@uf)**2)/128);q.update(control[str(t)]);rows.append(q)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False});fig,axs=plt.subplots(2,2,figsize=(12,7));x=np.array(steps)/1000
for ax,key,title in zip(axs.flat,['sigma_min','gamma_RMS','future_basis_overlap','centered_isotropic_input_expected_weak_fraction'],['Smallest singular value of Wc','Final LayerNorm gamma RMS','Own / 43k weak subspace overlap','Isotropic-input expected weak-energy fraction']):
 ax.plot(x,[r[key] for r in rows],'-o',c='#0072b2');ax.set(title=title,xlabel='Checkpoint (k steps)',xticks=x);ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(A/'head_parameter_trajectory.png',dpi=160);fig.savefig(A/'head_parameter_trajectory.pdf');plt.close(fig)
(A/'head_parameter_trajectory.json').write_text(json.dumps(dict(status='head_only_complete_native_activations_pending',rows=rows),indent=2)+'\n')
for f in ['head_verification.json','cuda_fallback_fidelity.json']:shutil.copyfile(R/f,A/f)
trs=''.join('<tr>'+''.join('<td>'+v+'</td>' for v in [str(r['label_step']//1000)+'k',f"{r['sigma_min']:.3f}",f"{r['gamma_RMS']:.3f}",f"{r['bias_RMS']:.3f}",f"{r['future_basis_overlap']:.1%}",f"{r['centered_isotropic_input_expected_weak_fraction']:.1%}"])+'</tr>' for r in rows)
state=json.loads((R/'state.json').read_text()) if (R/'state.json').exists() else {}
status_note='真实激活、末层各组分写入和消融仍在等待Spot容量；本页暂时不包含这些未完成的结果。两台欧洲区诊断Spot在准备阶段被抢占，美国区备选队列继续等待。'
if state.get('phase','').startswith('failed'):
 status_note='真实激活与消融本次未完成；本页只包含已完成的输出层参数分析。诊断运行已终止，资源回收状态见本页下方状态链接。需要资源可用后重新启动，不能把本页解释为已完成激活分析。'
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HUnit BAM早期轨迹：参数分析已完成</title><style>body{font:16px/1.7 system-ui;max-width:1200px;margin:auto;padding:24px;background:#f4f7fa;color:#172b3a}section{background:#fff;padding:22px;margin:20px 0;border-radius:12px}img{width:100%;height:auto}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left;white-space:nowrap}a{color:#006c9a}</style><a href="../">← 已完成的33k/43k末层写入归因</a><h1>HUnit BAM：3k至43k的早期轨迹</h1><section><h2>当前状态</h2><p>五个checkpoint的输出层谱分析已完成。'''+status_note+'''无训练实验或checkpoint写入。</p></section><section><h2>已经测得的参数轨迹</h2><img src="assets/head_parameter_trajectory.png"><div class="scroll"><table><tr><th>checkpoint</th><th>Wc最小奇异值</th><th>LN γ RMS</th><th>LN β RMS</th><th>当期/43k弱子空间重合度</th><th>各向同性输入的预期弱能量占比</th></tr>'''+trs+'''</table></div><p>W为原生BF16有效lm_head权重，Wc=W−mean_vocab(W)，每点独立分解。取最弱128/2560方向；子空间重合度为‖U当期ᵀU43k‖²F/128，是平均主夹角余弦平方，不是“相同方向个数”。最小奇异值下降说明读出逐渐变弱，但不能单独证明网络在向这些方向写入更多。</p><p>3k与43k的弱子空间重合度约14.4%，因此固定43k基底只能用来事后追踪。判断新归一化实验能否提早看出作用，需要同时比较当期弱谱与相同数据上的激活。</p><p>最后一列是参数解析对照：假设LN输入是零均值各向同性，忽略β，计算‖CΓU‖²F / [(1−1/D)Σγ²]，C=I−11ᵀ/D。不是实际文本激活，也不是归一化干预结果。43k约10.6%，须与尚待测量的真实激活分布区分。</p></section><section><h2>验证与下一步</h2><p>33k/43k的独立谱重算与前次归档完全一致；五组U的正交误差均小于5×10⁻⁷。新增M统计通过原生小模型前传一致性、能量闭合与末层消融检查。已有CUDA数学参考与TPU原生H的最大偏差在33k约5.1%、43k约6.0%，超过原1.5%门槛，故没有将其替换成原生激活结果。</p><p>待完成：同32条Pile来源文本×2048预测位置；统一FastKernel推理五个checkpoint；固定43k/当期两种基底；末层MHA、动态Mo、静态Mo、FFN的直接写入及整路移除loss变化；读前M RMS。3k/13k/23k实际训练早于26k的FastKernel切换，此处会统一推理实现来比较参数轨迹。</p><p>现有代码的读前M归一化是RMSNorm，当前FastKernel尚不支持直接开启。M归一化是否有效仍未测试：需要保持有用读出和loss，观察弱方向增长是否降低，同时检查读scale、W_o和最终LN是否补偿；norm变小本身不足以证明有效。</p><p><a href="assets/head_parameter_trajectory.json">参数轨迹数据</a> · <a href="assets/head_verification.json">谱验证</a> · <a href="assets/cuda_fallback_fidelity.json">CUDA替代路径误差</a>。图有同名PDF。</p></section></html>'''
(A/'run_state.json').write_text(json.dumps(state,indent=2)+'\n');page=page.replace('</html>','<section><p><a href="assets/run_state.json">最近运行状态</a></p></section></html>');(D/'index.html').write_text(page)
start='<!-- EARLY_TRAJECTORY_LINK_START -->';end='<!-- EARLY_TRAJECTORY_LINK_END -->';link=start+'<section><h2>新增：3k至43k的训练轨迹</h2><p><a href="early-trajectory/">已完成五点输出层谱；真实激活与消融等待Spot容量</a></p></section>'+end
if state.get('phase','').startswith('failed'):link=link.replace('已完成五点输出层谱；真实激活与消融等待Spot容量','已完成五点输出层谱；本次激活运行未完成')
parent=D.parent/'index.html';s=parent.read_text()
if start in s:s=s[:s.index(start)]+s[s.index(end)+len(end):]
parent.write_text(s.replace('</h1>','</h1>'+link,1));(R/'parent_link.html').write_text(link);print('HEAD_ONLY_PROGRESS_PUBLISHED',D)
