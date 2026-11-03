"""Detached exclusive readonly origin probe on one owned diagnostic worker."""
from pathlib import Path
import fcntl,json,os,subprocess,sys,time
R=Path('/tmp/hunit_early_weak_job');R.mkdir(exist_ok=True);STATE=R/'state.json'
def save(**kw):
 p=R/'state.tmp';p.write_text(json.dumps(dict(utc=time.time(),**kw))+'\n');p.replace(STATE)
def status():
 d=json.loads(STATE.read_text()) if STATE.exists() else dict(status='starting')
 log=R/'origin.log'
 if log.exists():d['tail']=log.read_text(errors='replace')[-1600:]
 print('JOB_STATUS_JSON='+json.dumps(d),flush=True)
mode=sys.argv[1]
if mode=='status':status();raise SystemExit()
if mode=='start':
 if STATE.exists():
  d=json.loads(STATE.read_text());p=Path('/proc',str(d.get('pid',-1)),'cmdline')
  if d.get('status') in ['complete','failed'] or (p.exists() and b'hunit_early_weak_worker_jobs.py' in p.read_bytes()):status();raise SystemExit()
 with (R/'wrapper.log').open('a') as f:p=subprocess.Popen(['python3',str(Path(__file__).resolve()),'run',sys.argv[2]],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 print('ORIGIN_JOB_LAUNCHED',p.pid,flush=True);raise SystemExit()
with (R/'job.lock').open('a+b') as f:
 fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);m=None;pid=os.getpid();save(status='starting',pid=pid)
 try:
  subprocess.run(['gsutil','-q','cp',sys.argv[2],str(R/'arms.json')],check=True);m=json.loads((R/'arms.json').read_text())
  root=Path('/home/lishengping/mengqy/hunit_early_weak_diag');work=root/'xd/projects/maxtext'
  assert subprocess.check_output(['git','-C',str(work),'rev-parse','HEAD'],text=True).strip()==m['commit']
  subprocess.run(['gsutil','-q','cp',m['gcs_root']+'/serial.train.config.txt',str(R/'serial.train.config.txt')],check=True)
  Path('/tmp/hunit_origin/hunit_origin').mkdir(parents=True,exist_ok=True)
  env=dict(os.environ,TENSORSTORE_GCS_REQUEST_CONCURRENCY='256',JAX_PLATFORMS='tpu',PYTHONPATH='MaxText:'+str(root/'native_deps')+':'+str(root/'xd/projects/praxis'),OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',TF_NUM_INTEROP_THREADS='2',TF_NUM_INTRAOP_THREADS='2',TF_CPP_MIN_LOG_LEVEL='3',CUDA_VISIBLE_DEVICES='')
  save(status='running',pid=pid,family='hunit',source_commit=m['commit'],tpu=m['tpu'],zone=m['zone'])
  with (R/'origin.log').open('a') as log:subprocess.run([str(root/'miniconda3/bin/python'),'MaxText/hunit_early_weak_probe.py','--manifest',str(R/'arms.json'),'--output','/tmp/hunit_early_weak','--training-config',str(R/'serial.train.config.txt')],cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  subprocess.run(['gsutil','-q','cp',str(R/'origin.log'),m['gcs_root']+'/worker_logs/'+m['tpu']+'-origin.log'],check=True)
  save(status='complete',pid=pid,family='hunit',source_commit=m['commit'],tpu=m['tpu'],zone=m['zone'])
 except BaseException as e:
  if m:
   for p in R.glob('*.log'):subprocess.run(['gsutil','-q','cp',str(p),m['gcs_root']+'/worker_logs/'+m['tpu']+'-'+p.name],timeout=60)
  save(status='failed',pid=pid,family='hunit',error=str(e),source_commit=m.get('commit') if m else None);raise
