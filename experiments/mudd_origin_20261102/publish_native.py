"""Publish paired M-route interventions, separately from exact final-router CPU split."""
from pathlib import Path
import json,shutil,html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;m=json.loads((R/'arms_native.json').read_text());F=Path(m['local_DATA']);s=json.loads((F/'complete.json').read_text());verification=json.loads((R/'verified_native.json').read_text());assert verification['complete']
D=R.parents[2]/'sites/bam-mudd-head-activation-20261023/dist/shorttrain/no-self-readout/mudd-spectral/gradient-origin';A=D/'assets';A.mkdir(parents=True,exist_ok=True)
base=s['arms']['keep'];rng=np.random.default_rng(20261102);boot=rng.integers(0,32,(5000,32));results={}
for name,v in s['arms'].items():
 delta=np.array([q['FP32_CE'] for q in v['sequence']])-np.array([q['FP32_CE'] for q in base['sequence']])
 results[name]=dict(**v,paired_fixed_error_aligned_reduction=v['fixed_original_error_aligned_reduction']-base['fixed_original_error_aligned_reduction'],delta_FP32_CE=float(delta.mean()),delta_FP32_CE_ci95=np.quantile(delta[boot].mean(1),[.025,.975]).tolist(),dark_gradient_norm=float(np.sqrt(v['dark_gradient_energy'])))
functional={}
for step in [33000,53000]:functional[str(step)]=json.loads((R.parent/f'gradient_origin_serial_{step}.json').read_text())
out=dict(manifest=m,baseline=base,results=results,selection=s['selection_rule'],selected_group=s['selected_group'],exact_archival_CPU_final_router=functional,paired_definition='For every intervention subtract keep aligned score, so native replay-versus-archival baseline offset is canceled. Original archival prediction errors and original dark-gradient direction held fixed. Different interventions are not additive.')
(R/'results_native.json').write_text(json.dumps(out,indent=2)+'\n')
groups=['all_m','all_body','body_02_08','body_09_16','body_17_24','body_25_32','final_only'];singles=[n for n in results if n.startswith('single_')]
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,2,figsize=(15,10))
for col,names in enumerate([groups,singles]):
 x=np.arange(len(names));labels=[n.replace('single_before_block_','before L').replace('body_','L').replace('_','-') for n in names];v=[results[n] for n in names]
 axs[0,col].bar(x,[q['paired_fixed_error_aligned_reduction']*100 for q in v],color=['#0072b2' if q['paired_fixed_error_aligned_reduction']>=0 else '#d55e00' for q in v]);axs[0,col].axhline(0,c='#777',lw=.8)
 y=np.array([q['delta_FP32_CE'] for q in v]);ci=np.array([q['delta_FP32_CE_ci95'] for q in v]);axs[1,col].errorbar(x,y,yerr=np.maximum(0,np.vstack([y-ci[:,0],ci[:,1]-y])),fmt='o',c='#0072b2');axs[1,col].axhline(0,c='#777',lw=.8)
 for row in range(2):axs[row,col].set_xticks(x,labels,rotation=35,ha='right');axs[row,col].grid(axis='y',alpha=.2)
 axs[0,col].set_title(['Grouped M-route resets','Individual resets in selected group'][col]);axs[0,col].set_ylabel('Removed aligned component / original full head G² (pp)');axs[1,col].set_ylabel('FP32 readout CE change vs native TPU keep')
fig.tight_layout();fig.savefig(A/'route_ablations.png',dpi=160,bbox_inches='tight');fig.savefig(A/'route_ablations.pdf',bbox_inches='tight');plt.close(fig)
def table(headers,rows):return '<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</table></div>'
tab=[]
for name,v in results.items():
 ci=v['delta_FP32_CE_ci95'];tab.append([name,','.join(str(i+1) if i<32 else 'final' for i in v['targets']),f"{v['paired_fixed_error_aligned_reduction']*100:+.3f}",f"{v['delta_FP32_CE']:+.5f} [{ci[0]:+.5f},{ci[1]:+.5f}]",f"{v['head_RMS']:.4f}",f"{v['dark_RMS']:.4f}",f"{v['global_head_gradient_norm']:.4f}",f"{v['dark_gradient_fraction']:.2%}"])
for source,target in [(R/'results_native.json','results.json'),(R/'verified_native.json','verification.json'),(R/'arms_native.json','manifest.json')]:shutil.copyfile(source,A/target)
conclusion=(R/'conclusion_native.html').read_text()
page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MUDD53k：弱方向梯度的来源</title><style>body{{font:16px/1.7 system-ui,sans-serif;max-width:1250px;margin:auto;padding:24px;background:#f4f7fa;color:#172b3a}}section{{background:white;margin:20px 0;padding:22px;border-radius:12px}}img{{width:100%;height:auto}}.scroll{{overflow:auto}}table{{border-collapse:collapse;font-size:13px;width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left;white-space:nowrap}}code{{overflow-wrap:anywhere}}a{{color:#006c9a}}</style><a href="../">← 新旧MUDD完整分析</a><h1>Serial MUDD53k：55.3%弱方向梯度来自哪里</h1><section>{conclusion}</section><section><h2>先区分最终router与早层影响</h2><p>对原始不可变TPU捕获做CPU分解，固定原预测误差，<code>h=LN(last block)+[h_actual−LN(last block)]</code>。将各分量的梯度与原弱梯度取内积，除以全部原head梯度平方范数，33k得到19.95−1.65=18.30个百分点，53k得到62.46−7.21=55.25个百分点。最终router的额外作用抵消弱梯度。这里的last block仍包含更早的MUDD和已学到的LN，不能叫成纯标准Transformer。</p><p>若冻结原LN分母，53k会得到standard+23.76、final-extra+42.70、bias−11.21个百分点。这种原始信号记账把LN会消去的径向放大也归给extra，因此不适合直接判断最终MUDD的功能性作用；它与上面的LN-aware分解不能混用。</p></section><section><h2>分组与单层参数消融</h2><img src="assets/route_ablations.png">{table(['干预','读取前层号','去除的同向分量 / 原总G² (pp)','ΔCE及文本bootstrap95%区间','h RMS','弱方向RMS','新head G norm','新弱梯度占比'],tab)}<p>每次只把指定m路的动态末级kernel置0、静态bias设为current-only；Q/K/V路、FFN、head和LN参数保持该checkpoint的实际值。L2–L8表示第2至8个block之前的m混合点；final为末层之后的最终混合。</p><p>上图数值为 <code>⟨G_dark,keep−G_dark,ablation(fixed E_orig), G_dark,orig⟩/‖G_full,orig‖²</code>，是有符号的同向分量分摊；不是新loss下梯度平方能量的直接减少量。以同一原生TPU的keep为基准减去偏移，避免把重新回放与原档案间的基线误差算给层。正值代表去除的部分原本支持原弱梯度，负值代表原本抵消它。各消融存在非线性交互，不能把单层或分组数值直接相加。</p><p>实际新梯度重新计算预测误差，与固定原误差的分量分摊分开记录。图下方loss对照使用BF16网络h的FP32读出CE；区间以32条文本为单位bootstrap，而非把65536个token当独立样本。梯度投影只有该固定cohort的点估计。</p></section><section><h2>验证与范围</h2><p>checkpoint为53k，state.step53001，2,784,962,589参数。原始谱与cohort沿用上一项分析，32×2048条预测；保留原生TPU前传与原始a03观测图，所有变体复用同一图，仅变动参数值。keep的32条h与原档案逐项相同；实际偏差与验证阈值记录在验证文件中。完整实际偏差见验证文件。</p><p>先扫描4段，再在固定误差对齐量绝对值最大的一段逐层测量；不是对全部31个m路的穷举排名，也没有移除全部QKV动态连接。固定参数的current-only消融描述当前模型依赖，不能代替重新训练的架构因果比较。</p><p>源码 <code>{m['commit']}</code>，分支 <code>{m['branch']}</code>。<a href="assets/results.json">完整结果</a> · <a href="assets/verification.json">逐帧验证</a> · <a href="assets/manifest.json">身份与来源</a> · <a href="assets/route_ablations.pdf">PDF</a> · <a href="../../hunit-spectral/">HUnit BAM33k→43k对照</a>。</p></section></html>'''
(D/'index.html').write_text(page);print('MUDD_ORIGIN_REPORT_PUBLISHED',D)
