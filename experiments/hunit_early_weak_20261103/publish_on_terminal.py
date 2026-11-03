"""Publish only independently verified inference, or mark the real terminal failure."""
from pathlib import Path
import json,os,subprocess,time,tarfile
R=Path(__file__).resolve().parent;env=dict(os.environ,OPENBLAS_NUM_THREADS='4');end=time.time()+100*60
while time.time()<end:
 try:s=json.loads((R/'state.json').read_text())
 except (FileNotFoundError,json.JSONDecodeError):time.sleep(20);continue
 if s.get('resources_released') and s['phase'] in ['verified_resources_released','failed_resources_released']:break
 time.sleep(20)
else:raise TimeoutError('No verified resource closeout; refusing to claim success')
if s['phase']=='verified_resources_released':
 j=json.loads((R/'results.json').read_text());assert j['complete'];steps=j['manifest']['steps'];rows=[j['rows'][str(t)] for t in steps]
 if not (R/'conclusion.html').exists():
  a,b=rows[0],rows[-1]
  text='<h2>五个checkpoint的原生分析已完成</h2><p>相同32条2048-token文本；当期与固定43k弱基底均核验，160条全部保留。以下是已测量的轨迹，不是M归一化训练实验。</p>'
  text+=f"<p>3k→43k：末层动态M_o在固定43k坐标中的LN前弱投影RMS从{a['fixed43k']['raw_self_RMS'][1]:.3f}变为{b['fixed43k']['raw_self_RMS'][1]:.3f}；当期弱方向的head能量占比从{a['own']['head_weak_fraction']['mean']:.1%}变为{b['own']['head_weak_fraction']['mean']:.1%}。读前物理M RMS从{a['memory_RMS']:.3f}变为{b['memory_RMS']:.3f}。完整五点曲线和各路贡献见下方。</p>"
  text+='<p>判断M读前归一化是否更早击中机制，应对照3k/13k/23k的当期弱方向占比、动态M_o的W_o后弱写入，以及相同文本的loss和MHA功能贡献。固定43k方向属于事后追踪。仅M RMS或梯度norm变小还不足以证明有益：读scale、W_o和最终LN可能补偿，整路Mo本身也可能仍有降loss价值。这里没有运行归一化训练，所以尚不能证明该改动有效。</p>'
  (R/'conclusion.html').write_text(text)
 subprocess.run(['python3',str(R/'publish.py')],env=env,check=True)
else:subprocess.run(['python3',str(R/'publish_head_progress.py')],env=env,check=True)
subprocess.run(['node',str(R/'check_page.cjs')],env=env,check=True)
m=json.loads((R/'arms.json').read_text());D=R.parents[1]/'sites/bam-mudd-head-activation-20261023/dist/shorttrain/no-self-readout/hunit-spectral/write-origin/early-trajectory'
for p in [R/'state.json',R/'resource-audit.json']:
 if p.exists():(D/'assets'/p.name).write_bytes(p.read_bytes())
archive=R/'report_bundle.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
 tar.add(D,arcname='early-trajectory')
 for name in ['arms.json','head_inventory.py','collect_analyze.py','publish.py','publish_head_progress.py','publish_on_terminal.py','head_verification.json','gamma_isotropic_control.json','resource-audit.json','state.json','page_checks.json']:
  if (R/name).exists():tar.add(R/name,arcname='run/'+name)
subprocess.run(['/home/mengqy/google-cloud-sdk/bin/gsutil','-q','cp',str(archive),m['gcs_root']+'/report_bundle.tar.gz'],check=True)
(R/'publication_state.json').write_text(json.dumps(dict(utc=time.time(),phase=s['phase'],native_analysis_complete=s['phase']=='verified_resources_released',page=str(D/'index.html'),url='http://172.29.142.228:8873/shorttrain/no-self-readout/hunit-spectral/write-origin/early-trajectory/',checks_passed=True),indent=2)+'\n');print('TERMINAL_REPORT_PUBLISHED',s['phase'],flush=True)
