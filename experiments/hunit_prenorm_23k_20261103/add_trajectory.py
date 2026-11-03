"""Add the measured13k->23k comparison; keep raw growth distinct from head fraction."""
from pathlib import Path
import json, shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path(__file__).resolve().parent
reports={13000:R.parent/'bam_hunit_prenorm_13k_20261103',23000:R}
summaries={t:json.loads((p/'results.json').read_text()) for t,p in reports.items()}
D=R.parents[1]/'sites/bam-mudd-head-activation-20261023/dist/shorttrain/no-self-readout/hunit-spectral/write-origin/prenorm-23k'
A=D/'assets';A.mkdir(exist_ok=True,parents=True)
boot=np.random.default_rng(20261103).integers(0,32,(5000,32))
series={};raw={};temporal={}
for arm in ['baseline','prenorm']:
    series[arm]={};raw[arm]={}
    for step in [13000,23000]:
        m=summaries[step]['models'][arm]
        assert m['cohort_hashes']==summaries[13000]['models']['baseline']['cohort_hashes']
        root=Path(m['local_DATA'])/str(step);v=[]
        for i in range(32):
            meta=json.loads((root/f'sequence_{i:03d}.json').read_text())
            assert meta['state_step']==step+1 and meta['input_hash']==m['cohort_hashes'][i]
            with np.load(root/f'sequence_{i:03d}.npz') as z:
                v.append(dict(hp=float(z['own_head_total_power']),p=float(z['baseline_head_power']),
                    mo=float(z['fixed43k_raw_self_power'][0,1]),read=float(z['fixed43k_read_three_power'][0].mean())))
        v={k:np.asarray([r[k] for r in v]) for k in v[0]};raw[arm][step]=v
        row=summaries[step]['rows'][arm]
        series[arm][str(step)]=dict(own_head_fraction=row['own']['head_weak_fraction'],
            own_raw_weak_RMS=row['own']['raw_weak_RMS'],
            fixed43k_final_dynamicMo_weak_RMS=row['fixed43k']['raw_self_RMS'][1],
            fixed43k_final_FFN_weak_RMS=row['fixed43k']['raw_self_RMS'][5],
            fixed43k_dynamicMo_effective_weak_gain=row['fixed43k']['raw_self_RMS'][1]/row['read_three_RMS'][0],
            physical_M_RMS=row['memory_RMS'])
    a,b=raw[arm][13000],raw[arm][23000]
    fd=b['hp'][boot].mean(1)/b['p'][boot].mean(1)-a['hp'][boot].mean(1)/a['p'][boot].mean(1)
    gain=lambda v,ix:np.sqrt(v['mo'][ix].mean(-1)/v['read'][ix].mean(-1))
    g=gain(b,boot)/gain(a,boot)
    temporal[arm]=dict(head_fraction_delta_pp=float((b['hp'].mean()/b['p'].mean()-a['hp'].mean()/a['p'].mean())*100),
        head_fraction_delta_pp_ci95=(np.quantile(fd,[.025,.975])*100).tolist(),
        fixed43k_dynamicMo_effective_weak_gain_ratio=float(np.sqrt((b['mo'].mean()/b['read'].mean())/(a['mo'].mean()/a['read'].mean()))),
        effective_weak_gain_ratio_ci95=np.quantile(g,[.025,.975]).tolist())
out=dict(series=series,paired_temporal=temporal,
    scope='Same32 texts, own contemporaneous centered effective BF16 Wc bottom128; fixed old43k basis is a common retrospective geometric control. Effective weak gain=postWo raw weak RMS/preWo dynamic read total RMS, combining learned Wo and activation direction; not an operator singular value or an isolated causal effect.')
(R/'cross-step.json').write_text(json.dumps(out,indent=2)+'\n');shutil.copy2(R/'cross-step.json',A/'cross-step.json')
fig,axs=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
specs=[('own_head_fraction','Own Wc bottom5% head energy','%'),
       ('own_raw_weak_RMS','Own weak directions: raw residual','RMS'),
       ('fixed43k_final_dynamicMo_weak_RMS','Final dynamic Mo weak write (fixed old43k)','RMS after Wo'),
       ('fixed43k_final_FFN_weak_RMS','Final FFN weak write (fixed old43k)','RMS'),
       ('fixed43k_dynamicMo_effective_weak_gain','Dynamic Mo effective weak gain (fixed old43k)','post-weak / pre-total RMS'),
       ('physical_M_RMS','Physical final-layer memory','RMS')]
for ax,(key,title,ylabel) in zip(axs.flat,specs):
    for arm,label,color in [('baseline','HUnit FastKernel','#3366a2'),('prenorm','MPreNorm + QKScale','#aa4c00')]:
        values=[series[arm][str(s)][key] for s in [13000,23000]]
        if key=='own_head_fraction':
            vals=np.array([v['mean']*100 for v in values]);ci=np.array([v['ci95'] for v in values]).T*100
            ax.errorbar([13,23],vals,yerr=np.stack([vals-ci[0],ci[1]-vals]),marker='o',capsize=4,label=label,color=color)
        else:ax.plot([13,23],values,'o-',label=label,color=color)
    ax.set(title=title,xlabel='Checkpoint (k)',ylabel=ylabel,xticks=[13,23]);ax.grid(alpha=.2);ax.legend(fontsize=8)
for ext in ['png','pdf']:fig.savefig(A/('cross-step.'+ext),dpi=170)
plt.close(fig)
n=temporal['prenorm'];s=series['prenorm'];gain=n['fixed43k_dynamicMo_effective_weak_gain_ratio']
mo_growth=s['23000']['fixed43k_final_dynamicMo_weak_RMS']/s['13000']['fixed43k_final_dynamicMo_weak_RMS']
supp=f'''<section id="cross-step-13-23"><h2>13k→23k：改善是否维持，哪些问题仍在增长？</h2><img src="assets/cross-step.png">
<p>同32条文本、每条2048预测位置；原模型的当期弱方向head能量占比20.58%→34.77%，新模型14.43%→14.32%。新模型的配对时间差为{n['head_fraction_delta_pp']:+.2f}个百分点，95%区间[{n['head_fraction_delta_pp_ci95'][0]:+.2f},{n['head_fraction_delta_pp_ci95'][1]:+.2f}]，与基本持平相符。相对于各自γ各向同性对照，原模型的超额集中倍数4.19→6.55，新模型2.95→2.94。这里的倍数是弱能量集中度，不能当作训练loss收益倍数。</p>
<p><strong>仍需注意动态Mo的弱写入增益在上升。</strong>在同一固定旧43k方向中，新模型末层动态Mo的原始弱RMS从0.261升至0.329（+{(mo_growth-1)*100:.1f}%），而pre-Wo动态读出总RMS从0.662降至0.587。两者比值形成的有效弱方向增益从0.394升至0.561，即{gain:.3f}×（配对95%区间{n['effective_weak_gain_ratio_ci95'][0]:.3f}–{n['effective_weak_gain_ratio_ci95'][1]:.3f}×）。物理M RMS约19.26→19.17、末层公共VO scalar约1.050→1.032，这个增长不能只用整体M幅度或公共scalar增大解释。</p>
<p>这个增益同时包含W_o和动态读出方向的变化，不是单独的W_o谱范数或重训练因果归因。当前证据支持新配置明显减轻了弱能量集中，并在13k–23k维持了主要健康指标；动态Mo的方向放大仍值得在33k继续检查。新模型原始弱RMS3.455→4.267仍在增长，因此不能把最终LN后占比持平解释成所有空间的弱分量都停止增长。<a href="assets/cross-step.json">时间配对数值与区间</a> · <a href="assets/cross-step.pdf">PDF图</a>。</p></section>'''
p=D/'index.html';text=p.read_text()
if 'id="cross-step-13-23"' not in text:
    text=text.replace('<section><h2>归因边界与保真核验</h2>',supp+'<section><h2>归因边界与保真核验</h2>')
    p.write_text(text)
print(json.dumps(out['paired_temporal'],ensure_ascii=False))
