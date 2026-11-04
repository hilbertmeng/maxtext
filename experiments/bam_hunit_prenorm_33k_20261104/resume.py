"""Resume fixed-cohort native analysis after spot loss, with bounded preemption retries."""
from pathlib import Path
import fcntl,json,os,re,shutil,subprocess,sys,time
import oneshot as o
R=Path(__file__).resolve().parent
MAX_ATTEMPTS=3
acquisition_deadline=json.loads((R/'pipeline-receipts.json').read_text())['capacity-wait']['utc']+21600

def transient_spot_loss(error):
 return error=='Diagnostic spot TPU lost; complete per-text GCS chunks retained' or bool(re.search(r'This TPU has state "(?:DELETING|PREEMPTED|STOPPED|TERMINATED)"',error or ''))

def run_attempt(attempt):
 oldaudit=json.loads((R/'resource-audit.json').read_text());assert oldaudit['all_absent']
 folder=R/f'resume-before-{attempt}';folder.mkdir(exist_ok=False)
 for n in ['controller.py','controller.log','state.json','resources.json','resource-audit.json']:
  if (R/n).exists():shutil.copy2(R/n,folder/n)
 remaining=int(acquisition_deadline-time.time());assert remaining>0,'Original six-hour spot acquisition deadline expired'
 (R/'resources.json').write_text('[]\n')
 p=R/'controller.py';s=p.read_text();prefix=time.strftime('xd-v5p8-mqy-hunpre33-r%m%d%H%M-',time.gmtime())+str(attempt)+'-'
 s=re.sub(r"^PREFIX='[^']*'",'PREFIX='+repr(prefix),s,flags=re.M)
 s=re.sub(r'time.time\(\)-started<\d+',f'time.time()-started<{remaining}',s)
 s=s.replace('No spot v5p-8 ready within 6 hours','No spot v5p-8 ready before original acquisition deadline')
 compile(s,str(p),'exec');p.write_text(s)
 o.save('resuming',attempt=attempt,acquisition_deadline=acquisition_deadline)
 o.notify('resume-'+str(attempt),'恢复33k弱能量分析，第'+str(attempt)+'次续算：上次Spot被抢占，原模型32条前传已保存在GCS，新模型尚未完成。保留全部既定样本，继续使用相同v5p单芯片环境、原阈值核验；不再因一次抢占直接结束，最多重试3次，资源申请仍受原6小时截止约束。完成后更新网页并发群，所有分析TPU用完删除。')
 with (R/'controller.log').open('w') as log:
  o.child=subprocess.Popen([sys.executable,str(p)],stdout=log,stderr=subprocess.STDOUT)
  while o.child.poll() is None:
   o.progress();time.sleep(20)
  rc=o.child.wait();o.child=None
 audit=json.loads((R/'resource-audit.json').read_text());audit['previous_attempts']=oldaudit;audit['resources']=oldaudit['resources']+audit['resources'];audit['all_absent']=oldaudit['all_absent'] and audit['all_absent'];(R/'resource-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
 return rc,json.loads((R/'state.json').read_text())

with (R/'oneshot.lock').open('a+b') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  assert json.loads((R/'resource-audit.json').read_text())['all_absent']
  done=False
  start=max([int(p.name.rsplit('-',1)[1]) for p in R.glob('resume-before-*')]+[0])+1
  assert start<=MAX_ATTEMPTS,'Preemption retry limit reached'
  for attempt in range(start,MAX_ATTEMPTS+1):
   o.checkpoint()
   rc,state=run_attempt(attempt)
   if rc==0:
    assert state['verified'] and state['resources_released'];done=True;break
   assert state.get('resources_released'),('Cleanup failed',state)
   if not transient_spot_loss(state.get('error')):raise RuntimeError(state.get('error') or 'Controller failure; see logs')
   o.notify('resume-preempt-'+str(attempt),'33k续算再次遇到Spot抢占，已确认清理本次资源；GCS中已完成的固定样本保留，'+('继续自动续算。' if attempt<MAX_ATTEMPTS else '达到本轮重试上限，尚未得到完整结果。'))
  assert done,'Preemption retry limit reached'
  o.save('publishing')
  if not (R/'completion.json').exists():subprocess.run([sys.executable,str(R/'finalize.py')],check=True)
  o.notify('final',(R/'final-message.txt').read_text())
  m=json.loads((R/'prenorm.json').read_text());gcs=m['gcs_root'].rsplit('/',1)[0]+'/report/'
  names=['results.json','verification.json','cross-step.json','wo-geometry.json','resource-audit.json','completion.json','pipeline-receipts.json']
  subprocess.run(['gsutil','-q','cp']+[str(R/n) for n in names]+[gcs],check=True,timeout=180)
  exp=Path(m['worktree'])/'experiments'/R.name
  for name in names:
   p=exp/name
   if p.exists() and name not in ['wo-geometry.json']:assert p.read_bytes()==(R/name).read_bytes(),('Existing artifact differs',p)
   else:shutil.copy2(R/name,p)
  o.save('complete',url=json.loads((R/'completion.json').read_text())['url'],resources_released=True)
 except BaseException as e:
  if o.child and o.child.poll() is None:o.interrupt_once(o.child.pid)
  absent=o.await_cleanup();o.save('failed',error=str(e),resources_released=absent)
  try:o.notify('resume-failed','33k续算尚未完成：'+str(e)[:500]+'。分析资源'+('已确认回收。' if absent else '回收状态尚未确认，需检查。')+'不把缺失的新模型弱能量结果写成结论。')
  except Exception:pass
  raise
