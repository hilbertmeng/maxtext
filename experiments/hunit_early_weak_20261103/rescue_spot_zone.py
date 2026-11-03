"""Exclusive task-owned Spot fallback; record every request and cancel retired queues."""
from pathlib import Path
import json,fcntl,os,signal,subprocess,sys,time
R=Path(__file__).resolve().parent;env=dict(os.environ,CLOUDSDK_ACTIVE_CONFIG_NAME='mengqy-tpu');task='bam_hunit_early_weak_20261103';prefix='xd-v6e1-mqy-hunearly-1103-'
zone=sys.argv[1] if len(sys.argv)>1 else 'us-east1-d';suffix={'us-east1-d':'ue1d','asia-southeast1-b':'sea1b','asia-east1-c':'ae1c'}[zone]
current=json.loads((R/'controller-state.json').read_text())
if current['phase']!='acquire':
 print('SKIP_FALLBACK_ACTIVE_CONTROLLER_PHASE',current['phase'],flush=True);raise SystemExit(0)
launch=json.loads((R/'launch.json').read_text());pid=launch['pid'];proc=Path(f'/proc/{pid}/cmdline')
if proc.exists() and str(R/'controller.py').encode() in proc.read_bytes():
 os.kill(pid,signal.SIGTERM)
 for _ in range(30):
  if not proc.exists() or not proc.read_bytes():break
  time.sleep(.2)
with (R/'.controller.lock').open('a+b') as lock:
 for attempt in range(150):
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
  except BlockingIOError:time.sleep(.2)
 else:raise RuntimeError('Task controller lock still owned after bounded stop')
 resources=json.loads((R/'resources.json').read_text());candidates=json.loads((R/'candidates.json').read_text());kept=[];retired=[]
 for x in candidates:
  if x['zone']!='europe-west4-a':kept.append(x);continue
  p=subprocess.run(['gcloud','compute','tpus','tpu-vm','describe',x['name'],'--zone='+x['zone'],'--project=newproject-1-451205','--format=value(state)'],env=env,capture_output=True,text=True,timeout=60)
  state=p.stdout.strip() if p.returncode==0 else ('ABSENT' if 'NOT_FOUND' in p.stderr else 'UNKNOWN')
  if state not in ['PREEMPTED','STOPPED','DELETING','ABSENT']:kept.append(x)
  else:retired.append(x)
 if not any(x['zone']==zone for x in kept):
  gen=1+max(int(x['name'].rsplit('-r',1)[1]) for x in resources);x=dict(name=prefix+suffix+'-r'+str(gen),zone=zone,owner_task=task,created_utc=time.time(),type='v6e-1',provisioning='spot');assert gen<=8
  resources.append(x);kept.append(x)
  (R/'resources.json').write_text(json.dumps(resources,indent=2)+'\n');(R/'candidates.json').write_text(json.dumps(kept,indent=2)+'\n')
  p=subprocess.run(['gcloud','compute','tpus','queued-resources','create',x['name'],'--node-id='+x['name'],'--project=newproject-1-451205','--zone='+zone,'--accelerator-type=v6e-1','--runtime-version=v2-alpha-tpuv6e','--service-account=626151558586-compute@developer.gserviceaccount.com','--spot','--async'],env=env,capture_output=True,text=True,timeout=90)
  print(p.stdout.strip(),flush=True)
  if p.returncode:
   print('FALLBACK_REQUEST_REJECTED',x['name'],p.stderr[-800:],flush=True);kept=[q for q in kept if q['name']!=x['name']];(R/'candidates.json').write_text(json.dumps(kept,indent=2)+'\n')
for x in retired:
 assert x['name'].startswith(prefix) and x['owner_task']==task
 with (R/(x['name']+'-retired-release.log')).open('a') as f:subprocess.Popen(['bash',str(R/'delete_tpu_xd.sh'),x['name'],x['zone'],'newproject-1-451205'],env=dict(env,TPU_AG_PROJECTS_DIR=str(R)),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
with (R/'controller.log').open('a') as f:p=subprocess.Popen(['python3',str(R/'controller.py')],env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
(R/'launch.json').write_text(json.dumps(dict(pid=p.pid,utc=time.time()))+'\n');print('EXCLUSIVE_SPOT_FALLBACK_RELAUNCHED',p.pid,flush=True)
