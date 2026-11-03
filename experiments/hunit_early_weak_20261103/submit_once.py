from pathlib import Path
import concurrent.futures,fcntl,json,os,subprocess,time
R=Path(__file__).resolve().parent;ENV=dict(os.environ,CLOUDSDK_ACTIVE_CONFIG_NAME='mengqy-tpu')
with (R/'.submit.lock').open('a+b') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert json.loads((R/'resources.json').read_text())==[]
 m=json.loads((R/'arms.json').read_text());gen=m['generation_start']
 candidates=[dict(name='xd-v5p8-mqy-hunearly-1103-ue5a-r'+str(gen),zone='us-east5-a',owner_task='bam_hunit_early_weak_20261103',attempt_id=m['attempt_id'],created_utc=time.time(),type='v5p-8',provisioning='spot')]
 (R/'resources.json').write_text(json.dumps(candidates,indent=2)+'\n');(R/'candidates.json').write_text(json.dumps(candidates,indent=2)+'\n')
 def submit(x):
  p=subprocess.run(['gcloud','compute','tpus','queued-resources','create',x['name'],'--node-id='+x['name'],'--project=newproject-1-451205','--zone='+x['zone'],'--accelerator-type=v5p-8','--runtime-version=tpu-ubuntu2204-base','--service-account=626151558586-compute@developer.gserviceaccount.com','--spot','--async'],env=ENV,capture_output=True,text=True,timeout=90)
  assert p.returncode==0,p.stderr[-500:]
  print('SUBMITTED',x['name'],flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  for result in pool.map(submit,candidates):pass
