"""Own one Spot inference client, seal source, and resume chunks after physical preemption."""
from pathlib import Path
import concurrent.futures,fcntl,json,os,shlex,subprocess,time
R=Path(__file__).resolve().parent;TASK='bam_hunit_early_weak_20261103';PREFIX='xd-v6e1-mqy-hunearly-1103-'
ENV=dict(os.environ,CLOUDSDK_ACTIVE_CONFIG_NAME='mengqy-tpu',TPU_AG_PROJECTS_DIR=str(R));PROJECT='newproject-1-451205'
def write(name,d):
 p=R/(name+'.tmp');p.write_text(json.dumps(d,indent=2)+'\n');p.replace(R/name)
def state(phase,status='running',**kw):write('controller-state.json',dict(phase=phase,status=status,utc=time.time(),**kw))
def node(x):
 p=subprocess.run(['gcloud','compute','tpus','tpu-vm','describe',x['name'],'--zone='+x['zone'],'--project='+PROJECT,'--format=value(state)'],capture_output=True,text=True,env=ENV,timeout=45)
 if p.returncode==0:return p.stdout.strip()
 if 'NOT_FOUND' in p.stderr:return 'ABSENT'
 raise RuntimeError(p.stderr[-700:])
def release(x):
 assert x['name'].startswith(PREFIX) and x['owner_task']==TASK
 with (R/'release-controller.log').open('a') as f:subprocess.run(['bash',str(R/'delete_tpu_xd.sh'),x['name'],x['zone'],PROJECT],env=ENV,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=900)
def ssh(x,c,t=120):return subprocess.run(['gcloud','compute','tpus','tpu-vm','ssh',x['name'],'--zone='+x['zone'],'--project='+PROJECT,'--worker=0','--command='+c],env=ENV,capture_output=True,text=True,timeout=t)
def complete(m):
 for t in m['steps']:
  p=subprocess.run(['gsutil','cat',m['gcs_root']+'/'+str(t)+'/complete.json'],env=ENV,capture_output=True,text=True,timeout=90)
  if p.returncode:return False
  d=json.loads(p.stdout);assert d['source_commit']==m['commit'] and d['sequences']==32 and d['checkpoint']==m['checkpoints'][str(t)]
 p=subprocess.run(['gsutil','cat',m['gcs_root']+'/complete.json'],env=ENV,capture_output=True,text=True,timeout=90)
 if p.returncode:return False
 d=json.loads(p.stdout);assert d['source_commit']==m['commit'] and d['cases']==32*len(m['steps'])
 return True
class Preempted(Exception):pass
def acquire(candidates,deadline):
 until=min(deadline,time.time()+1800)
 while time.time()<until:
  with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:states=list(pool.map(node,candidates))
  state('acquire',candidates=[dict(x,node_state=s) for x,s in zip(candidates,states)])
  for x,s in zip(candidates,states):
   if s=='READY':return x
  for i,(x,s) in enumerate(zip(candidates,states)):
   if s not in ['PREEMPTED','STOPPED']:continue
   release(x);candidates[i]=renew_candidate(x);write('candidates.json',candidates)
  time.sleep(15)
 raise TimeoutError('No owned Spot candidate became ready within bounded acquisition')
def renew_candidate(x):
 resources=json.loads((R/'resources.json').read_text());gen=1+max(int(q['name'].rsplit('-r',1)[1]) for q in resources)
 if gen>8:raise RuntimeError('Eight diagnostic generations exhausted')
 new=dict(x,name=x['name'].rsplit('-r',1)[0]+'-r'+str(gen),created_utc=time.time());resources.append(new);write('resources.json',resources)
 p=subprocess.run(['gcloud','compute','tpus','queued-resources','create',new['name'],'--node-id='+new['name'],'--project='+PROJECT,'--zone='+new['zone'],'--accelerator-type=v6e-1','--runtime-version=v2-alpha-tpuv6e','--service-account=626151558586-compute@developer.gserviceaccount.com','--spot','--async'],env=ENV,capture_output=True,text=True,timeout=90)
 if p.returncode:raise RuntimeError(p.stderr[-700:])
 return new
def submit(gen):
 old=json.loads((R/'resources.json').read_text());gen=max(gen,1+max((int(x['name'].rsplit('-r',1)[1]) for x in old),default=0))
 candidates=[dict(name=PREFIX+suffix+'-r'+str(gen),zone=zone,owner_task=TASK,created_utc=time.time(),type='v6e-1') for suffix,zone in [('ew4a','europe-west4-a'),('uc1a','us-central1-a'),('ue5a','us-east5-a')]]
 resources=json.loads((R/'resources.json').read_text());resources.extend(candidates);write('resources.json',resources)
 out=[]
 for x in candidates:
  p=subprocess.run(['gcloud','compute','tpus','queued-resources','create',x['name'],'--node-id='+x['name'],'--project='+PROJECT,'--zone='+x['zone'],'--accelerator-type=v6e-1','--runtime-version=v2-alpha-tpuv6e','--service-account=626151558586-compute@developer.gserviceaccount.com','--spot','--async'],env=ENV,capture_output=True,text=True,timeout=90)
  if p.returncode:print('SUBMIT_FAILURE',x['name'],p.stderr[-500:],flush=True)
  else:out.append(x)
 assert out,'All owned candidate submissions failed';write('candidates.json',out);return out
with (R/'.controller.lock').open('a+b') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);m=json.loads((R/'arms.json').read_text());candidates=json.loads((R/'candidates.json').read_text());deadline=m.get('deadline_utc',time.time()+3*3600)
 try:
  for gen in range(m.get('generation_start',1),m.get('generation_start',1)+6):
   x=acquire(candidates,deadline);m.update(tpu=x['name'],zone=x['zone']);write('arms.json',m);write('install_manifest.json',dict(name=x['name'],zone=x['zone'],commit=m['commit']))
   subprocess.run(['gsutil','-q','cp',str(R/'arms.json'),m['gcs_root']+'/arms.json'],env=ENV,check=True,timeout=120)
   state('install',**x)
   try:
    if m.get('adopt_installed') and gen==m.get('generation_start',1):
     p=ssh(x,'python3 /home/lishengping/mengqy/hunit_early_weak_worker_jobs.py status',120)
     lines=[a for a in p.stdout.splitlines() if a.startswith('JOB_STATUS_JSON=')];assert lines
     d=json.loads(lines[-1].split('=',1)[1]);assert d['source_commit']==m['commit'] and d['tpu']==x['name'] and d['status'] in ['running','complete']
     state('adopt_existing_inference',**x)
    else:
     with (R/'install.log').open('a') as log:subprocess.run(['python3',str(R/'install_guard.py'),x['name'],x['zone']],env=ENV,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=2400)
   except (subprocess.CalledProcessError,subprocess.TimeoutExpired):
    if node(x) not in ['PREEMPTED','STOPPED','DELETING','ABSENT']:raise
    release(x);candidates=[o for o in candidates if o['name']!=x['name']]+[renew_candidate(x)];write('candidates.json',candidates);continue
   launch='gsutil -q cp '+shlex.quote(m['gcs_root']+'/worker_jobs.py')+' /home/lishengping/mengqy/hunit_early_weak_worker_jobs.py && python3 /home/lishengping/mengqy/hunit_early_weak_worker_jobs.py start '+shlex.quote(m['gcs_root']+'/arms.json')
   try:p=ssh(x,launch,240)
   except subprocess.TimeoutExpired:p=None
   if p is not None and p.returncode:print('LAUNCH_TRANSPORT_RETRY_SAFE',p.stderr[-800:],flush=True)
   start=time.time();backups_released=False;unknown=0;failure_since=None
   try:
    while time.time()<deadline and time.time()-start<2*3600:
     s=node(x)
     if s in ['PREEMPTED','STOPPED','DELETING','ABSENT']:
      if complete(m):break
      state('preempted_resume',**x);raise Preempted()
     try:p=ssh(x,'python3 /home/lishengping/mengqy/hunit_early_weak_worker_jobs.py status',90)
     except subprocess.TimeoutExpired:p=None
     lines=[] if p is None else [a for a in p.stdout.splitlines() if a.startswith('JOB_STATUS_JSON=')]
     if lines:
      d=json.loads(lines[-1].split('=',1)[1]);unknown=0;state('probe_'+d.get('family','all'),worker=d,**x)
      if d['status']=='failed':
       failure_since=failure_since or time.time();state('bounded_probe_diagnosis',worker=d,inspection_deadline=failure_since+600,**x)
       if time.time()-failure_since>600:raise RuntimeError('Detached inference failed after bounded inspection: '+str(d))
       time.sleep(20);continue
      failure_since=None
      if d['status']=='complete':assert complete(m);break
      if not backups_released and any(v in d.get('tail','') for v in ['WRITE_ORIGIN_CASE_COMPLETE','WRITE_ORIGIN_CHECKPOINT_COMPLETE']):
       for other in candidates:
        if other['name']!=x['name']:release(other)
       backups_released=True
     else:
      unknown+=1
      if unknown>=5:ssh(x,launch,240);unknown=0
     time.sleep(20)
    else:raise TimeoutError('Inference deadline exceeded')
    state('all_arms',status_detail='complete_awaiting_local_verification',**x);break
   except Preempted:
    release(x)
    candidates=([] if backups_released else [o for o in candidates if o['name']!=x['name']])+[renew_candidate(x)]
    write('candidates.json',candidates)
    if time.time()>deadline:raise TimeoutError('Bounded Spot recovery deadline')
  else:raise RuntimeError('Six Spot leases preempted; raw GCS chunks retained')
 except BaseException as e:
  state('terminal_failure',status='failed',error=str(e))
  for other in candidates:
   try:release(other)
   except Exception as problem:print('CLEANUP_NEEDS_RETRY',other['name'],str(problem),flush=True)
  raise
