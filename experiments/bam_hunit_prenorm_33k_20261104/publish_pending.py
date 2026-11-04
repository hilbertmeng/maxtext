"""Publish only completed weight/log measurements while native activation analysis waits."""
from pathlib import Path
import json,html,os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
assert not (R/'results.json').exists()
D=R.parents[1]/'sites/bam-mudd-head-activation-20261023/dist/shorttrain/no-self-readout/hunit-spectral/write-origin/prenorm-33k'
A=D/'assets';A.mkdir(exist_ok=True,parents=True)
w=json.loads((R/'wo-geometry.json').read_text());lookup={(x['arm'],x['step']):x for x in w['rows']}
steps=[13000,23000,33000];rows={}
for step in steps:
 report=R if step==33000 else R.parent/f'bam_hunit_prenorm_{step//1000}k_20261103'
 h=json.loads((report/'tb_health.json').read_text())
 for arm in ['baseline','prenorm']:
  m=json.loads((report/(arm+'.json')).read_text())
  hp=json.loads((Path(m['local_DATA'])/'head_spectra'/f'{step}.json').read_text())
  rows[arm,step]=dict(hp,grad=h[m['run_name']]['metrics']['learning/raw_grad_norm']['mean'],**lookup[arm,step])
fig,axs=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
for ax,(key,title,ylabel) in zip(axs.flat,[('sigma_min','Centered effective lm_head: minimum singular value','Singular value'),('gamma_RMS','Final LayerNorm scale','RMS'),('fixed43k_isotropic_weak_gain','Final Wo: weak-output gain for isotropic input','RMS gain'),('fixed43k_projected_stable_rank','Final Wo projected to fixed old43k weak space','Stable rank')]):
 for arm,label,color in [('baseline','HUnit FastKernel','#3366a2'),('prenorm','MPreNorm + QKScale','#aa4c00')]:ax.plot([13,23,33],[rows[arm,s][key] for s in steps],'o-',label=label,color=color)
 ax.set(title=title,xlabel='Checkpoint (k)',ylabel=ylabel,xticks=[13,23,33]);ax.grid(alpha=.2);ax.legend(fontsize=8)
for ext in ['png','pdf']:fig.savefig(A/('weight-only.'+ext),dpi=170)
plt.close(fig)
trs=[]
for step in steps:
 for arm,label in [('baseline','旧FastKernel'),('prenorm','MPreNormQKScale')]:
  q=rows[arm,step];vals=[str(step),label,f"{q['sigma_min']:.3f}",f"{q['gamma_RMS']:.3f}",f"{q['grad']:.4f}",f"{q['fixed43k_isotropic_weak_gain']:.3f}",f"{q['fixed43k_projected_stable_rank']:.2f}"]
  trs.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in vals)+'</tr>')
headers=['step','模型','head最小奇异值','LN scale RMS','训练raw_grad_norm','Wo弱方向平均增益','Wo弱投影stable rank']
past=json.loads((R/'wo-past-decomposition.json').read_text())
text='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="60"><title>33k弱能量分析：权重结果已出，激活等待spot</title><style>body{font:16px/1.7 system-ui;max-width:1200px;margin:auto;padding:24px;background:#f4f7fa;color:#172b3a}section{background:white;padding:24px;margin:20px 0;border-radius:12px}img{width:100%;height:auto}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:9px;border-bottom:1px solid #ddd;text-align:left;white-space:nowrap}a{color:#006c9a}code{overflow-wrap:anywhere}</style><a href="../prenorm-23k/">← 已完成的23k分析</a><h1>33k弱能量分析：权重和训练日志已完成</h1><section><p><strong>前传激活能量、末层Mo/MHA消融仍待spot容量。</strong>以下只展示已实际算完的权重谱、W_o几何和训练日志，不将它们冒充33k激活弱能量占比。</p><p>当前任务：<span id="state">读取状态…</span>。一次性流程将在资源就绪后自动完成同32条×2048-token分析、核验、出图、发群和资源回收。本页每分钟刷新，最终结果会替换此临时页。</p></section><section><h2>13k→23k→33k：已完成的读出几何与梯度观测</h2><img src="assets/weight-only.png"><div class="scroll"><table><tr>'''+''.join('<th>'+x+'</th>' for x in headers)+'''</tr>'''+''.join(trs)+'''</table></div><p>训练梯度取各checkpoint之前100-step窗口的10个日志点，不是诊断32条文本的梯度，也不是弱方向梯度能量。lm_head谱使用BF16有效权重Wc=W−mean_vocab(W)，最终LN的scale为实际checkpoint参数。</p></section><section><h2>目前能判断什么？</h2><p>新模型33k的最小奇异值2.087与23k的2.119接近；raw_grad_norm为0.1112，与23k的0.1130接近。目前这两项未显示旧模型式的恶化，但还不能据此断言33k激活弱能量占比持平。</p><p>末层W_o在公共旧43k弱空间上的各向同性输入平均增益，新模型23k→33k为1.127倍，与W_o整体幅度增长接近。新模型33k投影算子的stable rank约23.80，旧模型为5.63：旧算子更集中于少数强输入方向。stable rank是Frobenius范数平方除以最大奇异值平方，不是矩阵的实际rank。</p><p>补查13k→23k：旧模型实际动态Mo弱增益2.205倍 = W_o平均弱增益1.277倍 × 读出—W_o方向匹配1.727倍；新模型1.422倍 = 1.281倍 × 1.111倍。方向匹配因子由实际增益除以各向同性输入增益得到，同时依赖读出协方差与W_o的谱形状/方向，不能视为纯激活变化或独立因果归因。</p><p>W_o统计投影到同一固定旧43k弱空间，且在最终LN前计量；不能直接代替各自当期Wc最弱5%上的LN后激活能量。新配置还包含QK读scale和训练logits精度差异，不能单独归因为M preNorm。</p><p><a href="assets/weight-only.pdf">权重图PDF</a></p></section><script>fetch('status.json?'+Date.now()).then(r=>r.json()).then(d=>{document.getElementById('state').textContent=d.phase+'（'+new Date(d.utc*1000).toLocaleString()+'）'}).catch(()=>document.getElementById('state').textContent='状态暂不可读；计算结果不受影响')</script></html>'''
link=D/'status.json'
if not link.exists():link.symlink_to(R.resolve()/'pipeline-status.json')
assert not (R/'results.json').exists()
p=D/'index.pending';p.write_text(text);p.replace(D/'index.html')
print('PENDING_WEIGHT_REPORT_PUBLISHED')
