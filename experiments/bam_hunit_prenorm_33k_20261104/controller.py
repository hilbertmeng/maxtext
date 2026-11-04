"""Bounded spot inference lifecycle; only this task's exact resources are deleted."""
from pathlib import Path
import concurrent.futures, fcntl, json, os, shlex, subprocess, time

R=Path(__file__).resolve().parent
SSH=['ssh','-S','/tmp/ssh-tpu-ag-mengqy-readscale.sock','tpu-ag']
REMOTE='/home/lishengping/mengqy/projects/bam_tpu/'+R.name
GCS='gs://newproject-1-llm_projects_us-east5/log/diagnostics/mengqy/'+R.name+'/native_v1'
PROJECT='newproject-1-451205'
PREFIX='xd-v5p8-mqy-hunpre33-1104d-'
resources=[]
started=time.time()
def save(phase,**kw):
    p=R/'state.tmp';p.write_text(json.dumps(dict(phase=phase,utc=time.time(),**kw),indent=2)+'\n');p.replace(R/'state.json')
def remote(args,timeout=120,check=True):
    command='env CLOUDSDK_ACTIVE_CONFIG_NAME=mengqy-tpu '+shlex.join(args)
    p=subprocess.run(SSH+[command],capture_output=True,text=True,timeout=timeout)
    if check and p.returncode:raise RuntimeError(p.stderr[-3000:]+'\n'+p.stdout[-3000:])
    return p
def node(x):
    p=remote(['gcloud','compute','tpus','tpu-vm','describe',x['name'],'--project='+PROJECT,
              '--zone='+x['zone'],'--format=value(state)'],timeout=60,check=False)
    if not p.returncode:return p.stdout.strip()
    if 'NOT_FOUND' in p.stderr:return 'ABSENT'
    raise RuntimeError(p.stderr[-1000:])
def submit(zone,suffix):
    x=dict(name=PREFIX+suffix+'-r1',zone=zone,owner_task=R.name,created_utc=time.time(),
           type='v5p-8',provisioning='spot')
    resources.append(x);(R/'resources.json').write_text(json.dumps(resources,indent=2)+'\n')
    remote(['gcloud','compute','tpus','queued-resources','create',x['name'],'--node-id='+x['name'],
            '--project='+PROJECT,'--zone='+zone,'--accelerator-type=v5p-8','--runtime-version=v2-alpha-tpuv5',
            '--service-account=626151558586-compute@developer.gserviceaccount.com','--spot','--async'])
    return x
def release(x):
    assert x['name'].startswith(PREFIX) and x['owner_task']==R.name
    p=remote(['env','TPU_AG_PROJECTS_DIR='+REMOTE,'bash',REMOTE+'/delete_tpu_xd.sh',
               x['name'],x['zone'],PROJECT],timeout=900)
    (R/(x['name']+'-release.log')).write_text(p.stdout+p.stderr)
def worker(x,command,timeout=120):
    return remote(['gcloud','compute','tpus','tpu-vm','ssh',x['name'],'--project='+PROJECT,
            '--zone='+x['zone'],'--worker=0','--command='+command],timeout=timeout)

with (R/'.controller.lock').open('a+b') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert json.loads((R/'resources.json').read_text())==[]
    failure=None;verified=False
    try:
        for arm in ['baseline','prenorm']:
            m=json.loads((R/(arm+'.json')).read_text())
            subprocess.run(['gsutil','-q','cp',str(R/(arm+'.json')),GCS+'/'+arm+'.json'],check=True)
            subprocess.run(['gsutil','-q','cp',str(R/(arm+'.train.config.txt')),GCS+'/'+arm+'.train.config.txt'],check=True)
        job=dict(commit=m['commit'],gcs_root=GCS,owner_task=R.name)
        (R/'job.json').write_text(json.dumps(job,indent=2)+'\n')
        subprocess.run(['gsutil','-q','cp',str(R/'job.json'),str(R/'worker_jobs.py'),GCS+'/'],check=True)
        remote(['mkdir','-p',REMOTE])
        subprocess.run(['scp','-o','ControlPath=/tmp/ssh-tpu-ag-mengqy-readscale.sock',
            str(R/'install_stream_diag.sh'),str(R/'delete_tpu_xd.sh'),'tpu-ag:'+REMOTE+'/'],check=True)
        candidates=[submit('europe-west4-b','ew4b'),submit('us-central1-a','uc1a'),submit('us-east5-a','ue5a')];chosen=None
        while time.time()-started<21600:
            if time.time()-started>300 and len(candidates)==1:
                candidates.extend([submit('us-central1-a','uc1a'),submit('us-east5-a','ue5a')])
            states=[node(x) for x in candidates];save('acquiring',candidates=[dict(x,state=s) for x,s in zip(candidates,states)])
            ready=[x for x,s in zip(candidates,states) if s=='READY']
            if ready:chosen=ready[0];break
            time.sleep(15)
        assert chosen,'No spot v5p-8 ready within 6 hours'
        save('installing',resource=chosen)
        worker(chosen,'mkdir -p /home/lishengping/mengqy')
        remote(['gcloud','compute','tpus','tpu-vm','scp',REMOTE+'/install_stream_diag.sh',
            chosen['name']+':/home/lishengping/mengqy/install_hunit_prenorm.sh',
            '--zone='+chosen['zone'],'--project='+PROJECT,'--worker=0'],timeout=180)
        p=worker(chosen,shlex.join(['bash','/home/lishengping/mengqy/install_hunit_prenorm.sh',
            chosen['zone'].rsplit('-',1)[0],m['commit']]),timeout=1800)
        (R/'install.log').write_text(p.stdout+p.stderr)
        for x in candidates:
            if x!=chosen:release(x)
        worker(chosen,'gsutil -q cp '+GCS+'/worker_jobs.py /home/lishengping/mengqy/hunit_prenorm_worker_jobs.py && python3 /home/lishengping/mengqy/hunit_prenorm_worker_jobs.py start '+GCS+'/job.json')
        deadline=time.time()+5400;d={}
        while time.time()<deadline:
            if node(chosen)!='READY':raise RuntimeError('Diagnostic spot TPU lost; complete per-text GCS chunks retained')
            p=worker(chosen,'python3 /home/lishengping/mengqy/hunit_prenorm_worker_jobs.py status')
            lines=[s for s in p.stdout.splitlines() if s.startswith('JOB_STATUS_JSON=')]
            if not lines:raise RuntimeError('No worker status '+p.stderr[-1000:])
            d=json.loads(lines[-1].split('=',1)[1]);save('inference',resource=chosen,worker=d)
            if d['status']=='failed':raise RuntimeError(json.dumps(d))
            if d['status']=='complete':break
            time.sleep(20)
        assert d.get('status')=='complete','Inference deadline exceeded'
        save('collecting',resource=chosen)
        env=dict(os.environ,JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='4')
        subprocess.run(['/home/mengqy/.cache/bam-diag-cpu-py312/bin/python',str(R/'collect_analyze.py')],env=env,check=True)
        verified=True
    except BaseException as e:
        failure=str(e);save('failed_releasing',error=failure)
    finally:
        errors=[]
        for x in resources:
            try:release(x)
            except Exception as e:errors.append(str(e))
        audit=[]
        for x in resources:
            try:
                ns=node(x)
                p=remote(['gcloud','compute','tpus','queued-resources','describe',x['name'],
                    '--zone='+x['zone'],'--project='+PROJECT,'--format=value(state)'],check=False)
                qs='ABSENT' if p.returncode and 'NOT_FOUND' in p.stderr else p.stdout.strip() or 'UNKNOWN'
                audit.append(dict(x,node_state=ns,queue_state=qs))
            except Exception as e:audit.append(dict(x,error=str(e)))
        absent=all(x.get('node_state')=='ABSENT' and x.get('queue_state')=='ABSENT' for x in audit)
        (R/'resource-audit.json').write_text(json.dumps(dict(all_absent=absent,resources=audit,errors=errors),indent=2)+'\n')
        save('complete' if verified and absent else 'failed',verified=verified,resources_released=absent,error=failure,release_errors=errors)
    if failure:raise RuntimeError(failure)
    assert verified and absent
