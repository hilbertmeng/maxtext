"""Collect, verify, then release the exact shared task-owned ledger."""
from pathlib import Path
from run_identity import same_attempt,owned_resources
import concurrent.futures,json,os,subprocess,time
R=Path(__file__).resolve().parent;P=R;M=R
REMOTE='/home/lishengping/mengqy/projects/bam_tpu/bam_hunit_early_weak_20261103';CHILD=REMOTE;SSH=['ssh','-S','/tmp/ssh-tpu-ag-mengqy-readscale.sock','tpu-ag']
m=json.loads((R/'arms.json').read_text());failure=None;collected=[]
def state(phase,**kw):(R/'state.json').write_text(json.dumps(dict(phase=phase,utc=time.time(),attempt_id=m['attempt_id'],source_commit=m['commit'],**kw),indent=2)+'\n')
try:
 end=time.time()+90*60
 while time.time()<end:
  q=subprocess.run(SSH+['cat '+CHILD+'/controller-state.json'],capture_output=True,text=True,check=True,timeout=60);d=json.loads(q.stdout)
  if not same_attempt(d,m):
   state('waiting_for_matching_controller',observed_attempt=d.get('attempt_id'));time.sleep(5);continue
  state(d['phase'],remote=d)
  if d['status']=='failed':raise RuntimeError(d)
  if d['phase']=='all_arms':break
  time.sleep(20)
 else:raise TimeoutError('Frozen-head probe deadline')
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OPENBLAS_NUM_THREADS='4')
 py='/home/mengqy/.cache/bam-diag-cpu-py312/bin/python'
 subprocess.run([py,str(R/'collect_analyze.py')],env=env,check=True)
 state('verified_releasing_owned_resources')
except BaseException as e:failure=str(e);state('failed_releasing_owned_resources',error=failure)
finally:
 # Freeze the sole resource writer before fetching the final authoritative ledger.
 stop="python3 - <<'PY'\nimport json,os,signal\nfrom pathlib import Path\nr=Path('"+CHILD+"');p=r/'launch.json'\nif p.exists():\n d=json.loads(p.read_text());pid=d.get('pid');q=Path(f'/proc/{pid}/cmdline')\n if d.get('attempt_id')==EXPECTED and q.exists() and str(r/'controller.py').encode() in q.read_bytes():os.kill(pid,signal.SIGTERM)\nPY"
 stop=stop.replace('EXPECTED',repr(m['attempt_id']))
 subprocess.run(SSH+[stop],check=True,timeout=60)
 q=subprocess.run(SSH+['cat '+REMOTE+'/resources.json'],capture_output=True,text=True,check=True,timeout=60);resources=owned_resources(json.loads(q.stdout),m);(P/'resources.json').write_text(json.dumps(resources,indent=2)+'\n')
 def release(x):
  assert x['name'].startswith('xd-v5p8-mqy-hunearly-1103-') and x['owner_task']=='bam_hunit_early_weak_20261103'
  cmd='env CLOUDSDK_ACTIVE_CONFIG_NAME=mengqy-tpu TPU_AG_PROJECTS_DIR='+REMOTE+' bash '+REMOTE+'/delete_tpu_xd.sh '+x['name']+' '+x['zone']+' newproject-1-451205'
  with (R/(x['name']+'-release.log')).open('w') as log:subprocess.run(SSH+[cmd],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(release,resources))
 q=subprocess.run(SSH+['python3 '+REMOTE+'/audit_resources.py'],capture_output=True,text=True,check=True,timeout=240);audit=json.loads(q.stdout);assert audit['all_absent'];(P/'resource-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
 state('failed_resources_released' if failure else 'verified_resources_released',error=failure,resources_released=True,resource_count=len(resources))
if failure:raise RuntimeError(failure)
print('NATIVE_WRITE_ORIGIN_VERIFIED_ALL_OWNED_RESOURCES_RELEASED',flush=True)
