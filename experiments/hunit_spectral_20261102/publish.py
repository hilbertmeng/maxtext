"""Publish computed checkpoint geometry, formal training gradients and scope."""
from pathlib import Path
import argparse,html,json,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--results',default=str(R/'results_gpu.json'));p.add_argument('--verification',default=str(R/'verified_gpu.json'));p.add_argument('--controls',default=str(R/'head_precision_controls.json'));args=p.parse_args()
j=json.loads(Path(args.results).read_text());rows=j['rows'];ss=[33000,43000];X=np.array(ss)/1000
D=R.parents[1]/'sites/bam-mudd-head-activation-20261023/dist/shorttrain/no-self-readout/hunit-spectral';A=D/'assets';A.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
def save(fig,name):
 fig.tight_layout();fig.savefig(A/(name+'.png'),dpi=160,bbox_inches='tight');fig.savefig(A/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
def values(key):return np.array([rows[str(t)][key]['mean'] for t in ss])
fig,axs=plt.subplots(2,3,figsize=(15,8))
for key,label in [('head_RMS','Head input'),('centered_weak_RMS','Wc bottom5%'),('centered_other_RMS','Other95%')]:axs[0,0].plot(X,values(key),'o-',label=label)
axs[0,0].set(ylabel='RMS',title='Actual post-LN head input')
for key,label in [('raw_weak_energy_fraction','Raw W'),('centered_weak_energy_fraction','Centered Wc')]:axs[0,1].plot(X,[100*rows[str(t)][key] for t in ss],'o-',label=label)
axs[0,1].set(ylabel='Input energy share (%)',title='Bottom5% input energy')
for key,label in [('global_head_gradient_norm','All directions'),('centered_dark_gradient_energy','Wc bottom5%')]:
 y=np.array([rows[str(t)]['summary'][key] for t in ss]);y=np.sqrt(y) if key.endswith('energy') else y
 axs[0,2].plot(X,y,'o-',label=label)
axs[0,2].plot(X,[rows[str(t)]['centered_other_gradient_norm'] for t in ss],'o-',label='Other95%');axs[0,2].set(ylabel='Gradient L2 norm',title='Same32-text FP32 head gradient')
for key,label in [('raw_weak_gradient_fraction','Raw W'),('centered_weak_gradient_fraction','Centered Wc')]:axs[1,0].plot(X,[100*rows[str(t)][key] for t in ss],'o-',label=label)
axs[1,0].set(ylabel='Gradient energy share (%)',title='Bottom5% share of head gradient')
axs[1,1].plot(X,values('common_logit_offset_RMS'),'o-',label='Common logit offset');axs[1,1].plot(X,values('centered_logit_RMS'),'o-',label='Centered logits');axs[1,1].set(ylabel='RMS',title='Common vs discriminative logits')
for key,label in [('gamma_RMS','Final LN scale'),('bias_RMS','Final LN bias')]:axs[1,2].plot(X,[rows[str(t)]['spectrum'][key] for t in ss],'o-',label=label)
axs[1,2].set(ylabel='RMS',title='Learned final LayerNorm parameters')
for ax in axs.flat:ax.set_xticks(X);ax.set_xlabel('Checkpoint label (k steps)');ax.legend(fontsize=8);ax.grid(alpha=.2)
save(fig,'geometry')
fig,axs=plt.subplots(1,2,figsize=(12,4.5));percent=np.arange(2.5,100,5)
for i,key in enumerate(['raw_bands_energy_fraction','centered_bands_energy_fraction']):
 for t,c in zip(ss,['#0072b2','#d55e00']):axs[i].plot(percent,100*np.array(rows[str(t)][key]),'o-',c=c,label=f'{t//1000}k')
 axs[i].set(yscale='log',xlabel='Singular-value percentile (low to high)',ylabel='Input energy per5% band (%)',title=['Raw W basis','Vocabulary-centered Wc basis'][i]);axs[i].axvspan(0,5,color='#999',alpha=.15);axs[i].grid(alpha=.2);axs[i].legend()
save(fig,'spectral_bands')
formal=j['formal_training'];bins=formal['bins'];fig,axs=plt.subplots(1,2,figsize=(12,4.5));x=[(v['interval'][0]+v['interval'][1])/2000 for v in bins]
for key,label in [('global_raw_gradient_RMS','All parameters'),('head_gradient_RMS','lm_head matrix'),('other_parameters_combined_RMS','Other parameters')]:axs[0].plot(x,[v[key] for v in bins],'o-',label=label)
axs[0].set(xlabel='Training step (k)',ylabel='RMS of recorded gradient L2 norm',title='Formal training, non-overlapping1k-step windows');axs[0].legend();axs[0].grid(alpha=.2)
axs[1].plot(x,[100*v['head_gradient_energy_fraction'] for v in bins],'o-',c='#0072b2');axs[1].set(xlabel='Training step (k)',ylabel='Head / all gradient energy (%)',title='Recorded head share');axs[1].grid(alpha=.2)
save(fig,'formal_gradients')
def fmt(v):return f'{v:.5g}'
def table(headers,body):return '<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in body)+'</table></div>'
metrics=[]
for name,key,pct in [('Head输入RMS','head_RMS',False),('LN前残差RMS','pre_norm_RMS',False),('Wc最低5%输入RMS','centered_weak_RMS',False),('其余95%输入RMS','centered_other_RMS',False),('Wc最低5%输入能量占比','centered_weak_energy_fraction',True),('原始W最低5%输入能量占比','raw_weak_energy_fraction',True),('Wc最低5% head梯度能量占比','centered_weak_gradient_fraction',True),('原始W最低5% head梯度能量占比','raw_weak_gradient_fraction',True),('公共logit偏移RMS','common_logit_offset_RMS',False),('中心化logits RMS','centered_logit_RMS',False),('BF16 logits＋FP32 CE','native_CE',False),('冻结h的FP32读出CE','FP32_CE',False),('先减公共偏移再BF16舍入的CE变化','center_before_quantization_delta_CE',False)]:
 vals=[]
 for t in ss:
  v=rows[str(t)][key];v=v['mean'] if isinstance(v,dict) else v;vals.append(f'{v:.2%}' if pct else fmt(v))
 metrics.append([name,*vals])
for name,key in [('Head梯度norm','global_head_gradient_norm'),('Wc最低5%梯度平方能量','centered_dark_gradient_energy')]:metrics.append([name,*[fmt(rows[str(t)]['summary'][key]) for t in ss]])
for name,key in [('Final LN scale RMS','gamma_RMS'),('Final LN bias RMS','bias_RMS'),('W公共偏移的权重能量占比','common_offset_head_energy_fraction')]:metrics.append([name,*[fmt(rows[str(t)]['spectrum'][key]) for t in ss]])
for source,n in [(Path(args.results),'results.json'),(Path(args.verification),'verification.json'),(R/'training_gradient_analysis.json','formal_training_gradients.json')]:shutil.copyfile(source,A/n)
(A/'manifest.json').write_text(json.dumps(j['manifest'],indent=2)+'\n')
controls=json.loads(Path(args.controls).read_text());shutil.copyfile(args.controls,A/'head_precision_controls.json')
control_rows=[];fig,axs=plt.subplots(1,2,figsize=(12,4.5))
for mode,label,color in [('training','Original CE backward','#0072b2'),('FP32_CE_BF16_logits','FP32 CE / BF16 logits','#009e73'),('FP32_readout','FP32 readout','#d55e00')]:
 rr=[controls['checkpoints'][str(t)][mode] for t in ss]
 axs[0].plot(X,[q['gradient_norm'] for q in rr],'o-',c=color,label=label);axs[1].plot(X,[100*q['centered_dark_fraction'] for q in rr],'o-',c=color,label=label)
 for t,q in zip(ss,rr):control_rows.append([str(t//1000)+'k',label,fmt(q['loss']),fmt(q['gradient_norm']),f"{q['centered_dark_fraction']:.2%}",f"{q['centered_first_ranks_gradient_fraction']['1']:.2%}",f"{q['vocabulary_constant_gradient_fraction']:.4%}"])
for ax,title,ylabel in zip(axs,['Frozen-H head precision controls','Wc bottom5% in each backward'],['Head gradient L2 norm','Gradient energy share (%)']):ax.set(title=title,xlabel='Checkpoint label (k steps)',ylabel=ylabel);ax.set_xticks(X);ax.grid(alpha=.2);ax.legend(fontsize=8)
save(fig,'head_precision_controls')
precision_section='<section><h2>训练CE反向的核对</h2><img src="assets/head_precision_controls.png">'+table(['步数','读出/反向','CE','Head梯度norm','最低5%梯度占比','最弱单方向梯度占比','词表公共平移梯度占比'],control_rows)+'<p>保留同一批实际h，用原始4096-token配置（本批2048长度只有一个分块）、BF16投影和原始CE自定义VJP，在CUDA上回放head反向；另测仅把CE改为FP32、以及投影与CE均为FP32。CPU上已验证跨两个分块与真实OutputHead的CE及每项权重梯度完全一致。这里仍不是正式训练batch1024的TPU全网络反传；词表公共平移是输出词表轴的梯度，不等于输入空间的Wc弱方向，两者不能混淆。<a href="assets/head_precision_controls.json">完整控制结果</a>。</p></section>'
shutil.copyfile(R/'ln_swap.json',A/'ln_swap.json')
conclusion=(R/'conclusion.html').read_text()
body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HUnit BAM 33k→43k：lm_head弱方向与梯度</title><style>body{{font:16px/1.7 system-ui,sans-serif;max-width:1200px;margin:auto;padding:24px;background:#f4f7fa;color:#172b3a}}section{{background:white;padding:22px;border-radius:10px;margin:20px 0}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border-bottom:1px solid #ddd;padding:8px;text-align:left;white-space:nowrap}}.scroll{{overflow:auto}}code{{overflow-wrap:anywhere}}a{{color:#006c9a}}</style><a href="../">← 分析目录</a><h1>HUnit BAM：33k→43k 的输出层弱方向与梯度</h1><section>{conclusion}</section><section><h2>正式训练记录</h2><img src="assets/formal_gradients.png"><p>每个点为1k步窗口内记录的norm平方平均再开根号；33k/43k采用[32k,33k)与[42k,43k)，各100帧。1918个参数梯度的平方和与全局raw_grad_norm²一致，最大相对误差6.01×10⁻⁷。</p></section><section><h2>相同文本下的残差与输出层梯度</h2><img src="assets/geometry.png">{table(['指标','33k','43k'],metrics)}</section><section><h2>完整谱分布</h2><img src="assets/spectral_bands.png"><p>按各checkpoint自己的奇异值从小到大排列，每段占128/2560=5%的输入维度。Wc=W−mean_vocab(W)，其投影能量不能与原始W投影混用；公共均值方向与Wc弱方向也不能重复相加。</p></section><section><h2>口径与复现</h2><p>正式实验配置 <code>{html.escape(j['manifest']['experiment'])}</code>，实际bucket RUN为 <code>{html.escape(j['manifest']['run_name'])}</code>。两点都属于26k后FastKernel运行阶段，训练源码 <code>{j['manifest']['training_commit']}</code>；checkpoint标签33k/43k，内部state.step分别33001/43001。</p><p>固定32条Pile训练来源文本，各2048个预测位置，共65536个token。这里不是held-out valid，也不是正式训练batch1024。使用物理GPU3上的CUDA纯JAX数学参考路径，替代TPU专用Pallas/Flash kernel；七项后端差异逐项记在结果中，参数、结构和其他关键标量均核对。因而不能声称复现TPU FastKernel逐比特前传。head原生CE回放、final LN、文本身份、Parseval和CPU逐文本投影已验证；首条文本重复前传结果一致。</p><p>梯度为冻结网络输入h后，对同一FP32读出CE计算 <code>G=Hᵀ(P−Y)/N</code>，对同一32文本取平均后再求norm。弱方向来自各checkpoint自己的谱。输入能量增量是观测，G的正交投影是该诊断梯度的精确分解；两者均不等于对训练过程或loss退化的干预因果证明。</p><p>推理源码 <code>{j['manifest']['commit']}</code>。<a href="assets/results.json">完整数值与文本bootstrap区间</a> · <a href="assets/verification.json">验证记录</a> · <a href="assets/formal_training_gradients.json">正式梯度统计</a> · <a href="assets/manifest.json">checkpoint与来源</a> · <a href="../mudd-spectral/">新旧MUDD对照</a>。图旁PNG同名PDF可下载。</p></section></html>'''
body=body.replace('<section><h2>完整谱分布</h2>',precision_section+'<section><h2>完整谱分布</h2>')
if j['manifest'].get('backend')=='native_tpu':
 body=body.replace('使用物理GPU3上的CUDA纯JAX数学参考路径，替代TPU专用Pallas/Flash kernel；七项后端差异逐项记在结果中，参数、结构和其他关键标量均核对。因而不能声称复现TPU FastKernel逐比特前传。','此页主表采用分析专用Spot v6e-1上的原生TPU FastKernel前传，保留Pallas、融合QKNorm和Splash，参数、结构及关键标量均核对。CUDA纯JAX参考结果另存作跨路径对照；它不是此主表的数据来源。')
body=body.replace('推理源码 <code>', 'head分析源码 <code>')
body=body.replace('CUDA纯JAX参考结果另存作跨路径对照；它不是此主表的数据来源。', 'H与原生CE来自TPU捕获源码aa12ff2e；谱指标与FP32 head读出采用c5fc8130，在CUDA上用位运算强制落实BF16有效权重后计算。此前CUDA可消除BF16往返转换的版本已弃用；当前FP32读出与独立head控制严格复现。CUDA纯JAX全网络结果另存对照。')
body=body.replace('输入能量增量是观测', '各正交子空间RMS均以完整2560维为分母，故其平方可直接相加。输入能量增量是观测')
(D/'index.html').write_text(body);print('HUNIT_REPORT_PUBLISHED',D)
