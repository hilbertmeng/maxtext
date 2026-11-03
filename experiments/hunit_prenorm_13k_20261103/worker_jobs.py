"""One spot v5p-8, four isolated inference chips, two read-only 13k arms."""
from pathlib import Path
import concurrent.futures, fcntl, json, os, subprocess, sys, time

R = Path('/tmp/hunit_prenorm_13k_job')
R.mkdir(exist_ok=True)
STATE = R/'state.json'

def save(**kw):
    p=R/'state.tmp';p.write_text(json.dumps(dict(utc=time.time(), **kw))+'\n');p.replace(STATE)

def status():
    d=json.loads(STATE.read_text()) if STATE.exists() else dict(status='starting')
    for arm in ['baseline', 'prenorm']:
        d[arm+'_tails'] = [(R/f'{arm}_rank_{i}.log').read_text(errors='replace')[-1800:]
                          if (R/f'{arm}_rank_{i}.log').exists() else '' for i in range(4)]
    print('JOB_STATUS_JSON='+json.dumps(d), flush=True)

if sys.argv[1]=='status':
    status();raise SystemExit()
if sys.argv[1]=='start':
    if STATE.exists():
        d=json.loads(STATE.read_text());p=Path('/proc',str(d.get('pid',-1)),'cmdline')
        if d.get('status') in ['complete','failed'] or (p.exists() and b'hunit_prenorm_worker_jobs.py' in p.read_bytes()):
            status();raise SystemExit()
    with (R/'wrapper.log').open('a') as f:
        p=subprocess.Popen(['python3',str(Path(__file__).resolve()),'run',sys.argv[2]],
            stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    print('JOB_STARTED',p.pid,flush=True);raise SystemExit()

with (R/'job.lock').open('a+b') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pid=os.getpid();children=[];files=[];m=None;save(status='starting',pid=pid)
    try:
        subprocess.run(['gsutil','-q','cp',sys.argv[2],str(R/'job.json')],check=True)
        m=json.loads((R/'job.json').read_text())
        root=Path('/home/lishengping/mengqy/hunit_prenorm_13k_diag')
        work=root/'xd/projects/maxtext';py=str(root/'miniconda3/bin/python')
        assert subprocess.check_output(['git','-C',str(work),'rev-parse','HEAD'],text=True).strip()==m['commit']
        Path('/tmp/hunit_origin/hunit_origin').mkdir(parents=True,exist_ok=True)
        base=dict(os.environ,TENSORSTORE_GCS_REQUEST_CONCURRENCY='128',JAX_PLATFORMS='tpu',
            PYTHONPATH='MaxText:'+str(root/'native_deps')+':'+str(root/'xd/projects/praxis'),
            OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TF_NUM_INTEROP_THREADS='2',TF_NUM_INTRAOP_THREADS='2',
            TF_CPP_MIN_LOG_LEVEL='3',CUDA_VISIBLE_DEVICES='',TPU_CHIPS_PER_HOST_BOUNDS='1,1,1',
            TPU_HOST_BOUNDS='1,1,1',ALLOW_MULTIPLE_LIBTPU_LOAD='true')
        def check(i):
            p=subprocess.run([py,'-c','import jax;print(jax.devices());assert jax.device_count()==1'],
                cwd=work,env=dict(base,TPU_VISIBLE_CHIPS=str(i)),capture_output=True,text=True,timeout=180)
            assert p.returncode==0,p.stderr[-2000:]
        with concurrent.futures.ThreadPoolExecutor(4) as pool:list(pool.map(check,range(4)))
        for arm in ['baseline','prenorm']:
            manifest=R/(arm+'.json');config=R/(arm+'.train.config.txt')
            subprocess.run(['gsutil','-q','cp',m['gcs_root']+'/'+arm+'.json',str(manifest)],check=True)
            subprocess.run(['gsutil','-q','cp',m['gcs_root']+'/'+arm+'.train.config.txt',str(config)],check=True)
            save(status='running',pid=pid,arm=arm,commit=m['commit'])
            children=[]
            for i in range(4):
                f=(R/f'{arm}_rank_{i}.log').open('a');files.append(f)
                p=subprocess.Popen([py,'MaxText/hunit_prenorm_compare_probe.py','--manifest',str(manifest),
                    '--training-config',str(config),'--output',f'/tmp/hunit_prenorm_13k/{arm}/rank{i}',
                    '--shard-index',str(i),'--num-shards','4'],cwd=work,env=dict(base,TPU_VISIBLE_CHIPS=str(i)),
                    stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
                children.append(p)
            while any(p.poll() is None for p in children):
                failed=[(i,p.returncode) for i,p in enumerate(children) if p.poll() not in [None,0]]
                if failed:raise RuntimeError('Inference rank failed '+str(failed))
                time.sleep(5)
            assert all(p.wait()==0 for p in children)
            merge='''
import json,sys
from pathlib import Path
from hunit_prenorm_compare_probe import blob,jblob
m=json.loads(Path(sys.argv[1]).read_text());parts=[jblob(m['gcs_root']+f'/13000/complete_shard_{i}.json') for i in range(4)]
assert all(p['source_commit']==m['commit'] and p['sequences']==8 and p['shard_index']==i for i,p in enumerate(parts))
rows=sorted([r for p in parts for r in p['sequence']],key=lambda q:q['sequence'])
assert [q['sequence'] for q in rows]==list(range(32))
assert [q['input_hash'] for q in rows]==m['cohort_hashes']
assert all(p['final_small_parameters']==parts[0]['final_small_parameters'] for p in parts)
q=dict(parts[0],sequences=32,sequence=rows,sequence_hashes=m['cohort_hashes'],num_shards=4);q.pop('shard_index')
blob(m['gcs_root']+'/13000/complete.json',(json.dumps(q,indent=2)+'\\n').encode())
blob(m['gcs_root']+'/complete.json',(json.dumps(dict(source_commit=m['commit'],cases=32,steps=[13000],checkpoints={'13000':q}))+'\\n').encode())
print('ARM_COMPLETE',m['arm'],flush=True)
'''
            subprocess.run([py,'-c',merge,str(manifest)],cwd=work,env=dict(base,TPU_VISIBLE_CHIPS='0'),check=True)
        save(status='complete',pid=pid,commit=m['commit'])
    except BaseException as e:
        for p in children:
            if p.poll() is None:p.terminate()
        for p in children:
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        save(status='failed',pid=pid,error=str(e));raise
    finally:
        for f in files:f.close()
        if m:
            for p in R.glob('*.log'):
                subprocess.run(['gsutil','-q','cp',str(p),m['gcs_root']+'/worker_logs/'+p.name],timeout=90)
