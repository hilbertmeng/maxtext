"""Four disjoint single-chip inference processes; no training or shared device ownership."""
from pathlib import Path
import fcntl,json,os,subprocess,sys,time,concurrent.futures
R=Path('/tmp/hunit_early_weak_v5p_job');R.mkdir(exist_ok=True);STATE=R/'state.json'
def save(**kw):
 p=R/'state.tmp';p.write_text(json.dumps(dict(utc=time.time(),**kw))+'\n');p.replace(STATE)
def status():
 d=json.loads(STATE.read_text()) if STATE.exists() else dict(status='starting');log=R/'origin.log'
 if log.exists():d['tail']=log.read_text(errors='replace')[-2400:]
 print('JOB_STATUS_JSON='+json.dumps(d),flush=True)
mode=sys.argv[1]
if mode=='status':status();raise SystemExit()
if mode=='start':
 if STATE.exists():
  d=json.loads(STATE.read_text());p=Path('/proc',str(d.get('pid',-1)),'cmdline')
  if d.get('status') in ['complete','failed'] or (p.exists() and b'hunit_early_weak_worker_jobs.py' in p.read_bytes()):status();raise SystemExit()
 with (R/'wrapper.log').open('a') as f:p=subprocess.Popen(['python3',str(Path(__file__).resolve()),'run',sys.argv[2]],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 print('ORIGIN_JOB_LAUNCHED',p.pid,flush=True);raise SystemExit()
with (R/'job.lock').open('a+b') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);m=None;pid=os.getpid();children=[];files=[];completed=False;save(status='starting',pid=pid)
 try:
  subprocess.run(['gsutil','-q','cp',sys.argv[2],str(R/'arms.json')],check=True);m=json.loads((R/'arms.json').read_text());assert m['provisioning']=='spot' and m['accelerator_type']=='v5p-8' and m['num_shards']==4
  root=Path('/home/lishengping/mengqy/hunit_early_weak_diag');work=root/'xd/projects/maxtext';py=str(root/'miniconda3/bin/python')
  assert subprocess.check_output(['git','-C',str(work),'rev-parse','HEAD'],text=True).strip()==m['commit']
  subprocess.run(['gsutil','-q','cp',m['gcs_root']+'/serial.train.config.txt',str(R/'serial.train.config.txt')],check=True);Path('/tmp/hunit_origin/hunit_origin').mkdir(parents=True,exist_ok=True)
  base=dict(os.environ,TENSORSTORE_GCS_REQUEST_CONCURRENCY='128',JAX_PLATFORMS='tpu',PYTHONPATH='MaxText:'+str(root/'native_deps')+':'+str(root/'xd/projects/praxis'),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TF_NUM_INTEROP_THREADS='2',TF_NUM_INTRAOP_THREADS='2',TF_CPP_MIN_LOG_LEVEL='3',CUDA_VISIBLE_DEVICES='',TPU_CHIPS_PER_HOST_BOUNDS='1,1,1',TPU_HOST_BOUNDS='1,1,1',ALLOW_MULTIPLE_LIBTPU_LOAD='true')
  save(status='running',pid=pid,family='hunit',source_commit=m['commit'],tpu=m['tpu'],zone=m['zone'])
  with (R/'origin.log').open('a',buffering=1) as log:
   preflight="import jax,json; print(json.dumps(dict(devices=[str(x) for x in jax.devices()],count=jax.device_count())),flush=True); assert jax.device_count()==1"
   def check(i):
    p=subprocess.run([py,'-c',preflight],cwd=work,env=dict(base,TPU_VISIBLE_CHIPS=str(i)),capture_output=True,text=True,timeout=180);(R/f'chip_{i}_preflight.log').write_text(p.stdout+p.stderr);assert p.returncode==0,(i,p.stderr[-1500:]);return p.stdout.strip()
   with concurrent.futures.ThreadPoolExecutor(4) as pool:
    for i,out in enumerate(pool.map(check,range(4))):print('CHIP_ISOLATION_PASS',i,out,file=log,flush=True)
   for i in range(4):
    f=(R/f'rank_{i}.log').open('a');files.append(f)
    p=subprocess.Popen([py,'MaxText/hunit_early_weak_v5p_probe.py','--manifest',str(R/'arms.json'),'--output',f'/tmp/hunit_early_weak/rank{i}','--training-config',str(R/'serial.train.config.txt'),'--shard-index',str(i),'--num-shards','4'],cwd=work,env=dict(base,TPU_VISIBLE_CHIPS=str(i)),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT);children.append(p)
   useful=False
   while any(p.poll() is None for p in children):
    failed=[(i,p.returncode) for i,p in enumerate(children) if p.poll() not in [None,0]]
    if failed:
     for i,rc in failed:print('RANK_FAILURE',i,rc,(R/f'rank_{i}.log').read_text(errors='replace')[-5000:],file=log,flush=True)
     raise RuntimeError('Single-chip inference failed: '+str(failed))
    tails=[(R/f'rank_{i}.log').read_text(errors='replace')[-2500:] for i in range(4)]
    if not useful and any('WRITE_ORIGIN_CASE_COMPLETE' in t for t in tails):useful=True;print('WRITE_ORIGIN_CASE_COMPLETE first validated v5p text',file=log,flush=True)
    print('RANK_PROGRESS',[(i,t.splitlines()[-1] if t.splitlines() else '') for i,t in enumerate(tails)],file=log,flush=True);time.sleep(10)
   assert all(p.wait()==0 for p in children)
   merge=r'''
import io,json
from pathlib import Path
import numpy as np
from hunit_early_weak_v5p_probe import blob,jblob
m=json.loads(Path(__import__('sys').argv[1]).read_text());steps={}
for t in m['steps']:
 shards=[jblob(m['gcs_root']+f'/{t}/complete_shard_{i}.json') for i in range(4)]
 assert all(q and q['source_commit']==m['commit'] and q['sequences']==8 and q['shard_index']==i for i,q in enumerate(shards))
 rows=sorted([r for q in shards for r in q['sequence']],key=lambda r:r['sequence']);assert [r['sequence'] for r in rows]==list(range(32));assert [r['input_hash'] for r in rows]==m['cohort_hashes']
 assert all(q['final_small_parameters']==shards[0]['final_small_parameters'] for q in shards)
 q=dict(shards[0],sequences=32,sequence_hashes=m['cohort_hashes'],sequence=rows,num_shards=4);q.pop('shard_index');blob(m['gcs_root']+f'/{t}/complete.json',(json.dumps(q,indent=2)+'\n').encode());steps[str(t)]=q
rank=jblob(m['gcs_root']+'/complete_shard_0.json');q=dict(source_commit=m['commit'],device_kind=rank['device_kind'],jax_version=rank['jax_version'],steps=m['steps'],cases=160,checkpoints=steps,scope=m['scope'],num_shards=4,accelerator_type='v5p-8');blob(m['gcs_root']+'/complete.json',(json.dumps(q,indent=2)+'\n').encode());print('WRITE_ORIGIN_ALL_COMPLETE')
'''
   subprocess.run([py,'-c',merge,str(R/'arms.json')],cwd=work,env=dict(base,TPU_VISIBLE_CHIPS='0'),stdout=log,stderr=subprocess.STDOUT,check=True)
  completed=True
 except BaseException as e:
  for p in children:
   if p.poll() is None:p.terminate()
  for p in children:
   try:p.wait(timeout=15)
   except subprocess.TimeoutExpired:p.kill();p.wait()
  save(status='failed',pid=pid,family='hunit',error=str(e),source_commit=m.get('commit') if m else None);raise
 finally:
  for f in files:f.close()
  if m:
   for p in R.glob('*.log'):
    try:subprocess.run(['gsutil','-q','cp',str(p),m['gcs_root']+'/worker_logs/'+m['tpu']+'-'+p.name],timeout=90,check=True)
    except Exception as e:print('LOG_UPLOAD_RETRY_NEEDED',p.name,str(e),flush=True)

  if completed:save(status='complete',pid=pid,family='hunit',source_commit=m['commit'],tpu=m['tpu'],zone=m['zone'])
