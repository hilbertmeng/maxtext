"""New43k only; reuse historical results, detect durable outputs before provisioning."""
from pathlib import Path
import fcntl,json,os,re,shutil,subprocess,sys,time
import support as o
R=Path(__file__).resolve().parent
TASK=json.loads((R/'task.json').read_text())

def ready():
 m=json.loads((R/'prenorm.json').read_text());p=subprocess.run(['gsutil','-q','stat',m['gcs_root']+'/complete.json'],capture_output=True,text=True,timeout=60);return p.returncode==0

def collect():
 env=dict(os.environ,JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
 with (R/'collect.log').open('w') as log:subprocess.run(['/home/mengqy/.cache/bam-diag-cpu-py312/bin/python',str(R/'collect_analyze.py')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 assert json.loads((R/'results.json').read_text())['measured_only']==dict(arm='prenorm',steps=[43000],sequences=32)

def transient(error):
 return error=='Diagnostic spot TPU lost; complete per-text GCS chunks retained' or bool(re.search(r'This TPU has state "(?:DELETING|PREEMPTED|STOPPED|TERMINATED)"',error or ''))

def attempt_run(i):
 old=json.loads((R/'resource-audit.json').read_text());assert old['all_absent']
 remaining=int(TASK['acquisition_deadline']-time.time());assert remaining>0,'Spot acquisition deadline expired'
 folder=R/f'attempt-{i}';folder.mkdir(exist_ok=False)
 for n in ['controller.py','state.json','resources.json','resource-audit.json']:
  if (R/n).exists():shutil.copy2(R/n,folder/n)
 (R/'resources.json').write_text('[]\n')
 p=R/'controller.py';s=p.read_text();prefix=time.strftime('xd-v5p8-mqy-hunpre43-%m%d%H%M-',time.gmtime())+str(i)+'-'
 s=re.sub(r"^PREFIX='[^']*'",'PREFIX='+repr(prefix),s,flags=re.M);s=re.sub(r'time.time\(\)-started<\d+',f'time.time()-started<{remaining}',s);p.write_text(s)
 (R/'state.json').write_text(json.dumps(dict(phase='preparing',utc=time.time(),attempt=i))+'\n')
 with (R/'controller.log').open('w') as log:
  o.child=subprocess.Popen([sys.executable,str(p)],stdout=log,stderr=subprocess.STDOUT)
  while o.child.poll() is None:o.progress();time.sleep(20)
  rc=o.child.wait();o.child=None
 audit=json.loads((R/'resource-audit.json').read_text());audit['resources']=old['resources']+audit['resources'];audit['all_absent']=old['all_absent'] and audit['all_absent'];(R/'resource-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
 assert audit['all_absent'],'Owned diagnostic resource cleanup incomplete'
 return rc,json.loads((R/'state.json').read_text())

with (R/'oneshot.lock').open('a+b') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  assert json.loads((R/'resource-audit.json').read_text())['all_absent']
  o.notify('start','开始新模型43k弱能量分析：BamPythia2p8BHUnit300BFastKernelMPreNormQKScale。只补算43k，同32条×2048-token文本、当期Wc最弱5%及末层写入/消融；13k/23k/33k和原模型43k全部复用。只用独立spot v5p-8，完成后发群、更新图表并删除TPU。')
  done=False
  if ready():collect();done=True
  start=max([int(p.name.rsplit('-',1)[1]) for p in R.glob('attempt-*') if p.is_dir()]+[0])+1
  for i in ([] if done else range(start,TASK['max_attempts']+1)):
   o.checkpoint()
   if ready():collect();done=True;break
   rc,state=attempt_run(i)
   if rc==0:done=True;break
   # A preempted worker may finish its upload before the VM disappears.
   if ready():collect();done=True;break
   if not transient(state.get('error')):raise RuntimeError(state.get('error') or 'Controller failed')
   o.notify('preempt-'+str(i),'新模型43k分析遇到Spot抢占，已确认清理本次资源；检查GCS后保留已完成的固定样本及编译缓存继续续算，不重跑历史对照。')
  assert done,'Spot retry limit reached or acquisition deadline expired'
  assert json.loads((R/'resource-audit.json').read_text())['all_absent']
  o.checkpoint();o.save('publishing',resources_released=True)
  subprocess.run(['/home/mengqy/miniconda3/bin/python',str(R/'publish.py')],check=True)
  subprocess.run(['node',str(R/'check_page.cjs')],check=True)
  o.notify('final',(R/'final-message.txt').read_text())
  m=json.loads((R/'prenorm.json').read_text());names=['results.json','verification.json','resource-audit.json','completion.json','pipeline-receipts.json','wo-geometry.json']
  subprocess.run(['gsutil','-q','cp']+[str(R/n) for n in names]+[m['gcs_root'].rsplit('/',1)[0]+'/report/'],check=True,timeout=180)
  e=Path(m['worktree'])/'experiments'/R.name;e.mkdir(exist_ok=True)
  for name in names:shutil.copy2(R/name,e/name)
  o.save('complete',url=TASK['report_url'],resources_released=True)
 except BaseException as e:
  if o.child and o.child.poll() is None:o.interrupt_once(o.child.pid)
  absent=o.await_cleanup();o.save('failed',error=str(e),resources_released=absent)
  try:o.notify('failed','新模型43k弱能量分析尚未完成：'+str(e)[:500]+'。资源'+('已确认回收' if absent else '回收状态尚未确认')+'；未把缺失或未通过核验的结果写成结论。')
  except Exception:pass
  raise
