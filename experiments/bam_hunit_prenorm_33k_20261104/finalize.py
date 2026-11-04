"""Publish only after all paired inference, fidelity and owned-resource cleanup pass."""
from pathlib import Path
import json,subprocess,html
from datetime import datetime,timezone
R=Path(__file__).resolve().parent
URL='http://172.29.142.228:8873/shorttrain/no-self-readout/hunit-spectral/write-origin/prenorm-33k/'
d=json.loads((R/'results.json').read_text());assert d['complete']
a,b=(d['rows'][k] for k in ['baseline','prenorm'])
audit=json.loads((R/'resource-audit.json').read_text());assert audit['all_absent'] and audit['resources']
assert all(d['verification'][k]['sequences']==32 for k in ['baseline','prenorm'])
afrac,bfrac=(r['own']['head_weak_fraction']['mean'] for r in [a,b]);p=d['paired']['own_head_fraction_delta'];loss=d['paired']['common_fp32_CE'];lci=loss['ci95']
lossnote='配对95%区间包含0，尚无明确loss差异' if lci[0]<=0<=lci[1] else ('本批文本中新模型loss更低' if lci[1]<0 else '本批文本中新模型loss更高')
paras=[f'<h2>33k：当期最弱5%方向的head能量占比，{afrac*100:.2f}% → {bfrac*100:.2f}%</h2><p>旧HUnit FastKernel → MPreNormQKScale，同32条×2048-token文本；相对变化{(bfrac/afrac-1)*100:+.1f}%，配对差{p["mean"]*100:+.2f}个百分点，95%区间[{p["ci95"][0]*100:+.2f},{p["ci95"][1]*100:+.2f}]。这是最终LN后head输入的激活能量占比，不是梯度能量占比。</p>',
f'<p>最终LN前原始残差总RMS {a["raw_RMS"]:.3f} → {b["raw_RMS"]:.3f}，投到各自当期弱方向的原始RMS {a["own"]["raw_weak_RMS"]:.3f} → {b["own"]["raw_weak_RMS"]:.3f}；最终LN后head总RMS {a["head_RMS"]:.3f} → {b["head_RMS"]:.3f}。相对于各自γ各向同性解析对照，弱能量集中倍数 {a["own_head_fraction_over_gamma_isotropic"]:.2f} → {b["own_head_fraction_over_gamma_isotropic"]:.2f}。</p>',
f'<p>末层动态Mo投到各自弱方向的直接写入RMS {a["own"]["raw_self_RMS"][1]:.3f} → {b["own"]["raw_self_RMS"][1]:.3f}，末层FFN {a["own"]["raw_self_RMS"][5]:.3f} → {b["own"]["raw_self_RMS"][5]:.3f}；物理M RMS {a["memory_RMS"]:.3f} → {b["memory_RMS"]:.3f}。M的整体大小、读出幅度和读出方向需要分开看。</p>',
f'<p>统一FP32 head CE {a["common_fp32_CE"]["mean"]:.5f} → {b["common_fp32_CE"]["mean"]:.5f}；配对差{loss["mean"]:+.5f}，95%区间[{lci[0]:+.5f},{lci[1]:+.5f}]，{lossnote}。末层移除整路Mo的ΔCE {a["causal"]["remove_mo"]["delta_CE"]["mean"]:+.5f} → {b["causal"]["remove_mo"]["delta_CE"]["mean"]:+.5f}；移除整路MHA的ΔCE {a["causal"]["remove_mha"]["delta_CE"]["mean"]:+.5f} → {b["causal"]["remove_mha"]["delta_CE"]["mean"]:+.5f}。整路消融覆盖末层全部head，重算其后FFN和LN，不是只消融弱方向。</p>',
'<p>新配置同时包含M读前RMS归一化、QK读scale和训练logits精度差，配对结果不能单独归因于preNorm。所有64次模型文本评估通过原生/观察前传保真、能量账本和历史同硬件复测核验；全部独立分析TPU及队列已确认删除，正式训练未更改。</p>']
(R/'conclusion.html').write_text('\n'.join(paras))
subprocess.run(['/home/mengqy/miniconda3/bin/python',str(R/'publish.py')],check=True)
subprocess.run(['node',str(R/'check_page.cjs')],check=True)
site=R.parents[1]/'sites/bam-mudd-head-activation-20261023/dist/shorttrain/no-self-readout/hunit-spectral/write-origin'
for pth in [site/'index.html',site/'early-trajectory/index.html',site/'prenorm-13k/index.html',site/'prenorm-23k/index.html']:
 s=pth.read_text()
 if 'prenorm-33k/' not in s:
  href='prenorm-33k/' if pth.parent==site else '../prenorm-33k/'
  pth.write_text(s.replace('</h1>','</h1><p><a href="'+href+'">新增：33k MPreNormQKScale弱能量及13k→23k→33k轨迹</a></p>',1))
cross=json.loads((R/'cross-step.json').read_text());series=cross['series'];temporal=cross['paired_temporal']['prenorm']['23000-33000']
frac=lambda arm:' → '.join(f"{series[arm][str(s)]['own_head_fraction']['mean']*100:.2f}%" for s in [13000,23000,33000])
text=(f'33k弱能量复查完成：MPreNormQKScale vs 原HUnit FastKernel\n'
 f'各自当期Wc最弱5%方向head能量占比：旧{afrac*100:.2f}% → 新{bfrac*100:.2f}%（相对{(bfrac/afrac-1)*100:+.1f}%）。\n'
 f'13k→23k→33k：新 {frac("prenorm")}；旧 {frac("baseline")}。\n'
 f'新模型23k→33k变化{temporal["head_fraction_delta_pp"]:+.2f}个百分点，配对95%区间[{temporal["head_fraction_delta_pp_ci95"][0]:+.2f},{temporal["head_fraction_delta_pp_ci95"][1]:+.2f}]。\n'
 f'末层动态Mo自身弱方向写入RMS：旧{a["own"]["raw_self_RMS"][1]:.3f} → 新{b["own"]["raw_self_RMS"][1]:.3f}；末层FFN：{a["own"]["raw_self_RMS"][5]:.3f} → {b["own"]["raw_self_RMS"][5]:.3f}。\n'
 f'统一FP32 head CE差{loss["mean"]:+.5f}，{lossnote}。\n'
 f'这是激活弱能量，不是梯度弱能量；64次评估通过核验，分析TPU已删除。\n图表与完整分析：{URL}')
(R/'final-message.txt').write_text(text)
completion=dict(complete=True,checked_utc=datetime.now(timezone.utc).isoformat(),cohort_pairs=32,model_text_evaluations=64,checkpoints=[d['models'][k]['checkpoints']['33000'] for k in ['baseline','prenorm']],url=URL,resources_released=True,numerical_evidence='results.json; verification.json; cross-step.json',plot_evidence='page_checks.json')
(R/'completion.json').write_text(json.dumps(completion,indent=2)+'\n')
print(text)
