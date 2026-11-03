"""After committed23k: sealed paired spot analysis, verification, cleanup, report."""
from pathlib import Path
import fcntl, json, os, shutil, subprocess, sys, time

R=Path(__file__).resolve().parent
PREVIOUS=R.parent/'bam_hunit_prenorm_13k_20261103'
PY='/home/mengqy/.cache/bam-diag-cpu-py312/bin/python'
RUN='BamPythia2p8BHUnit300BFastKernelMPreNormQKScale'
TRAINING_COMMIT='5a48faaa36989deb1c24747c6314b49b0d8b649f'
NATIVE_COMMIT='e85828935d446aaf7b6def302fbab3cdd1bcec74'
DATA=Path('/data0/mengqy/bam_diagnostics')/R.name
GCS='gs://newproject-1-llm_projects_us-east5/log/diagnostics/mengqy/'+R.name+'/native_v1'
URL='http://172.29.142.228:8873/shorttrain/no-self-readout/hunit-spectral/write-origin/prenorm-23k/'
ENV=dict(os.environ,JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',TF_CPP_MIN_LOG_LEVEL='3')

def progress(phase,**kw):
    p=R/'goal-progress.tmp';p.write_text(json.dumps(dict(phase=phase,utc=time.time(),**kw),ensure_ascii=False,indent=2)+'\n');p.replace(R/'goal-progress.json')

def derive(s):
    return s.replace('13001','23001').replace('13000','23000').replace('12901','22901').replace('12910','22910').replace('hunpre13','hunpre23').replace('13k','23k')

def prepare(snap):
    assert snap['run_name']==RUN and snap['checkpoint_committed']
    assert snap['code_commit']==TRAINING_COMMIT,('Training runtime changed; inspect before analysis',snap['code_commit'])
    for name in ['controller.py','worker_jobs.py','collect_analyze.py','head_inventory.py','install_stream_diag.sh','delete_tpu_xd.sh','publish.py','check_page.cjs']:
        text=derive((PREVIOUS/name).read_text())
        if name=='publish.py':
            text=text.replace('旧模型重新计算的弱功率与历史v5p归档最大相对差0.040%，通过原门槛。','旧模型与历史v5p归档的复测差异见核验JSON。')
        (R/name).write_text(text)
    # Shared diagnostics source accepts arbitrary checkpoint steps; only wrapper paths/labels change.
    for arm in ['baseline','prenorm']:
        m=json.loads((PREVIOUS/(arm+'.json')).read_text())
        m.update(task=R.name,steps=[23000],execution_order=[23000],checkpoint_state_steps={'23000':23001},
            commit=NATIVE_COMMIT,gcs_root=GCS+'/'+arm,own_basis_root=GCS+'/'+arm+'/head_spectra',
            local_DATA=str(DATA/arm),scope='Same32x2048 Pile-source cohort, both23k checkpoints on one native spot v5p-8; own Wc bottom128 and fixed old43k; final write allocation/whole-branch ablation; no training. Compare with completed13k.',
            checkpoints={'23000':snap['checkpoint'] if arm=='prenorm' else 'gs://newproject-1-llm_projects_us-east5/log/BamPythia2p8BHUnit300B/checkpoints/23000/items'})
        if arm=='baseline':m['checkpoint_provenance_note']=m.get('checkpoint_provenance_note','').replace('13k','23k')
        (R/(arm+'.json')).write_text(json.dumps(m,indent=2)+'\n')
    if not (R/'resources.json').exists():(R/'resources.json').write_text('[]\n')
    assert json.loads((R/'resources.json').read_text())==[],'Existing resource ledger requires review, no duplicate acquisition'
    shutil.copy2(PREVIOUS/'baseline.train.config.txt',R/'baseline.train.config.txt')
    source=snap['checkpoint'].split('/checkpoints/')[0]+'/train.config.txt'
    subprocess.run(['gsutil','-q','cp',source,str(R/'prenorm.train.config.txt')],check=True,timeout=180)
    # Try the actual saved TB prefix, which is independent of checkpoint geography.
    prefix=None
    for line in (R/'prenorm.train.config.txt').read_text().splitlines():
        if line.startswith('tensorboard_dir: '):prefix=line.split(': ',1)[1].strip();break
    tb=DATA/'training_logs';tb.mkdir(parents=True,exist_ok=True)
    synced=False
    if prefix and prefix.startswith('gs://'):
        q=subprocess.run(['gsutil','-q','-m','rsync','-r',prefix,str(tb)],capture_output=True,text=True,timeout=300)
        synced=q.returncode==0
        (R/'training_log_sync.json').write_text(json.dumps(dict(prefix=prefix,local=str(tb),success=synced,utc=time.time()),indent=2)+'\n')
    s=derive((PREVIOUS/'tb_health.py').read_text())
    if synced:
        s=s.replace("files=sorted(Path('/data0/xd/tensorboard_logs',name).glob('events*'));index=[]",
            "root=Path("+repr(str(tb))+") if name=="+repr(RUN)+" else Path('/data0/xd/tensorboard_logs',name)\n files=sorted(root.glob('events*'));index=[]")
    (R/'tb_health.py').write_text(s)
    with (R/'tb_health.log').open('w') as log:subprocess.run([PY,str(R/'tb_health.py')],env=ENV,stdout=log,stderr=subprocess.STDOUT,check=True)

def conclude():
    d=json.loads((R/'results.json').read_text());old=json.loads((PREVIOUS/'results.json').read_text())
    a,b=(d['rows'][k] for k in ['baseline','prenorm'])
    oa,ob=(old['rows'][k] for k in ['baseline','prenorm'])
    af,bf=(r['own']['head_weak_fraction']['mean'] for r in [a,b]);delta=d['paired']['own_head_fraction_delta']
    ci=delta['ci95'];better=bf<af;word='下降' if better else '上升'
    effect=100*(bf/af-1);loss=d['paired']['common_fp32_CE'];lci=loss['ci95']
    evidence='配对区间支持新模型弱占比更低' if ci[1]<0 else ('配对区间支持新模型弱占比更高' if ci[0]>0 else '有限文本上的配对区间包含0，暂不判定显著方向')
    loss_note='区间包含0，尚不足以判定loss有显著差异' if lci[0]<=0<=lci[1] else ('本批文本中新模型loss更低' if lci[1]<0 else '本批文本中新模型loss更高')
    ratio13=ob['own']['head_weak_fraction']['mean']/oa['own']['head_weak_fraction']['mean'];ratio23=bf/af
    heading='23k弱方向能量较原模型更低' if better else '23k弱方向能量未较原模型降低'
    def f(x):return f'{x:.4f}'
    parts=[]
    parts.append(f'<h2>{heading}</h2><p>各自当期Wc最弱128/2560方向：原模型{af*100:.2f}% → 新模型{bf*100:.2f}%，相对变化{effect:+.1f}%；配对差{delta["mean"]*100:+.2f}个百分点，95%区间[{ci[0]*100:+.2f},{ci[1]*100:+.2f}]个百分点。{evidence}。最终LN整体head输入RMS为{a["head_RMS"]:.3f}→{b["head_RMS"]:.3f}；原始弱RMS为{a["own"]["raw_weak_RMS"]:.3f}→{b["own"]["raw_weak_RMS"]:.3f}。</p>')
    parts.append(f'<p>13k的旧/新弱占比为{oa["own"]["head_weak_fraction"]["mean"]*100:.2f}%/{ob["own"]["head_weak_fraction"]["mean"]*100:.2f}%；新/旧比例由13k的{ratio13:.3f}变成23k的{ratio23:.3f}。比例更低代表相对原模型的改善更大。这是同文本匹配比较，不证明33k之后也会保持。</p>')
    parts.append(f'<p>末层动态Mo原始弱方向写入RMS {f(a["own"]["raw_self_RMS"][1])}→{f(b["own"]["raw_self_RMS"][1])}，末层FFN {f(a["own"]["raw_self_RMS"][5])}→{f(b["own"]["raw_self_RMS"][5])}。相同口径的全部组分、公共固定43k方向及有符号分摊见下表/JSON；不能只由总RMS变化判断哪个上游机制造成差异。</p>')
    parts.append(f'<p>物理M RMS {a["memory_RMS"]:.3f}→{b["memory_RMS"]:.3f}；前Wo的总Mo RMS {a["mha_Mo_RMS"][1]:.3f}→{b["mha_Mo_RMS"][1]:.3f}。弱占比相对当前γ各向同性解析对照的倍数为{a["own_head_fraction_over_gamma_isotropic"]:.2f}→{b["own_head_fraction_over_gamma_isotropic"]:.2f}；它是方向集中度对照，不是梯度能量占比。</p>')
    parts.append(f'<p>统一FP32 head CE {a["common_fp32_CE"]["mean"]:.5f}→{b["common_fp32_CE"]["mean"]:.5f}，配对差{loss["mean"]:+.5f}，95%区间[{lci[0]:+.5f},{lci[1]:+.5f}]；{loss_note}。末层移除整路Mo的ΔCE {a["causal"]["remove_mo"]["delta_CE"]["mean"]:+.5f}→{b["causal"]["remove_mo"]["delta_CE"]["mean"]:+.5f}；移除整路MHA的ΔCE {a["causal"]["remove_mha"]["delta_CE"]["mean"]:+.5f}→{b["causal"]["remove_mha"]["delta_CE"]["mean"]:+.5f}。弱能量/幅度、整路消融必要性、训练收益是不同观测，不能互相替代。</p>')
    parts.append('<p>两模型同一spot v5p-8、同32条2048-token文本，所有样本均保留；先核验原生/观察前传，再检查能量闭合；全部分析TPU和队列已确认删除。模型差异包含M读前归一化、QK读scale和训练logits精度，不能单独归因于M归一化。</p>')
    (R/'conclusion.html').write_text('\n'.join(parts))
    text=(f'23k弱能量对比完成：MPreNormQKScale vs 原HUnit FastKernel\n'
        f'各自Wc最弱5%方向的head能量占比：{af*100:.2f}% → {bf*100:.2f}%（相对{effect:+.1f}%）。\n'
        f'原始弱RMS：{a["own"]["raw_weak_RMS"]:.3f} → {b["own"]["raw_weak_RMS"]:.3f}；'
        f'末层动态Mo：{a["own"]["raw_self_RMS"][1]:.3f} → {b["own"]["raw_self_RMS"][1]:.3f}；'
        f'末层FFN：{a["own"]["raw_self_RMS"][5]:.3f} → {b["own"]["raw_self_RMS"][5]:.3f}。\n'
        f'13k新/旧弱占比比值{ratio13:.3f}，23k为{ratio23:.3f}。\n'
        f'统一FP32 loss差{loss["mean"]:+.5f}；{loss_note}。\n'
        f'全部64个配对样本通过核验，分析TPU已删除。\n图表与完整分析：{URL}')
    return text

def main():
    with (R/'analysis.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            state=json.loads((R/'watch-state.json').read_text());snap=state['snapshot']
            progress('preparing');prepare(snap)
            progress('head_spectra')
            with (R/'head_inventory.log').open('w') as log:subprocess.run([PY,str(R/'head_inventory.py')],env=ENV,stdout=log,stderr=subprocess.STDOUT,check=True)
            with (R/'controller.log').open('w') as log:
                p=subprocess.Popen([sys.executable,str(R/'controller.py')],stdout=log,stderr=subprocess.STDOUT)
                while p.poll() is None:
                    if (R/'state.json').exists():
                        d=json.loads((R/'state.json').read_text());phase=d.get('phase')
                        if phase=='inference':phase='inference_'+d.get('worker',{}).get('arm','starting')
                        progress(phase,controller=d)
                    time.sleep(20)
                assert p.wait()==0,('Native controller failed',str(R/'controller.log'))
            audit=json.loads((R/'resource-audit.json').read_text());assert audit['all_absent']
            text=conclude();progress('publishing')
            subprocess.run(['/home/mengqy/miniconda3/bin/python',str(R/'publish.py')],check=True)
            subprocess.run(['node',str(R/'check_page.cjs')],check=True)
            site=R.parents[1]/'sites/bam-mudd-head-activation-20261023/dist/shorttrain/no-self-readout/hunit-spectral/write-origin'
            for p in [site/'index.html',site/'early-trajectory/index.html',site/'prenorm-13k/index.html']:
                s=p.read_text()
                if 'prenorm-23k/' not in s:
                    href='prenorm-23k/' if p.parent==site else '../prenorm-23k/'
                    s=s.replace('</h1>','</h1><p><a href="'+href+'">新增：23k MPreNormQKScale 配对弱能量分析</a></p>',1);p.write_text(s)
            subprocess.run(['gsutil','-q','cp',str(R/'results.json'),str(R/'verification.json'),str(R/'resource-audit.json'),GCS+'/report/'],check=True,timeout=180)
            progress('complete',url=URL,feishu_text=text,resources_released=True)
        except BaseException as e:
            progress('failed',error=str(e));raise

if __name__=='__main__':main()
