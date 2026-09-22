"""Task-specific rolling dispatch on tpu-ag. Retain all VMs at completion."""
import concurrent.futures,json,shlex,subprocess,sys,time
from pathlib import Path
ROOT=Path('/home/lishengping/xd/projects');PROJECT='newproject-1-451205'
GCS='gs://newproject-1-llm_projects_europe-west4/log/diagnostics/alllocal-o-writeback-0922'
ZONES=['us-central1-a','europe-west4-a','us-east5-a']

def run(args,timeout=60):return subprocess.run(args,capture_output=True,text=True,timeout=timeout)
def inventory(zone):
    r=run(['gcloud','compute','tpus','tpu-vm','list',f'--zone={zone}',f'--project={PROJECT}','--format=json(name,state)'])
    if r.returncode:raise RuntimeError(r.stderr[-1000:])
    return zone,{x['name'].rsplit('/',1)[-1]:x['state'] for x in json.loads(r.stdout) if x['name'].rsplit('/',1)[-1].startswith(('xd-v6e-alllocal-','xd-v6e-localow-'))}
def complete(i):return i,run(['gsutil','-q','stat',f'{GCS}/shard{i}/worker/DONE']).returncode==0
def launch(i,task,name,zone,commit):
    out=f'/tmp/localo-writeback-shard{i}-0922';source=f'{GCS}/shard{i}'
    job=f'''set -e
cd {ROOT}/maxtext
gsutil -m rsync -r {source}/worker {out} || true
gsutil cp {GCS}/protocol.json {source}/protocol.json
export JAX_DEFAULT_MATMUL_PRECISION=highest BET_OUTPUT={out} BET_GCS={source}
export BET_RUNNER=experiments/bam_llama2_medium/localo_writeback.py LOCALO_MODE={task['mode']} BET_START={task['start']} BET_STOP={task['stop']}
set +e
bash experiments/bam_llama2_medium/run_write_gate_bets.sh dtype=float32 matmul_precision=highest
rc=$?
echo "$rc" > {out}/EXIT
gsutil cp {out}/EXIT {source}/worker/EXIT
exit "$rc"
'''
    command=f'''set -e
cd {ROOT}/maxtext
if pgrep -f '^/home/lishengping/miniconda3/bin/python .*experiments/bam_llama2_medium/' >/dev/null; then echo WORKER_BUSY; exit 2; fi
git fetch origin codex/alllocal-write-geometry
git checkout --detach {commit}
mkdir -p {out}
printf %s {shlex.quote(job)} > {out}/dispatch_job.sh
nohup bash {out}/dispatch_job.sh > {out}/dispatch_launcher.log 2>&1 </dev/null &
echo LAUNCH_PID=$!
'''
    r=run(['gcloud','compute','tpus','tpu-vm','ssh','--internal-ip',name,f'--zone={zone}',f'--project={PROJECT}','--worker=0','--command',command],90)
    if r.returncode or 'LAUNCH_PID=' not in r.stdout:raise RuntimeError((r.stdout+r.stderr)[-2000:])
    return r.stdout.strip()
def main(path):
    state=json.loads(path.read_text())
    while not (path.parent/'STOP_LOCALO_DISPATCH').exists():
        try:
            with concurrent.futures.ThreadPoolExecutor(7) as pool:
                inv=dict(pool.map(inventory,ZONES));done=dict(pool.map(complete,range(len(state['tasks']))))
            for i,task in enumerate(state['tasks']):
                k=str(i);a=state['shards'].get(k)
                if done[i]:state['completed'][k]=True;continue
                if a and inv[a['zone']].get(a['name']) in ['PREEMPTED','DELETING','TERMINATED',None]:
                    state.setdefault('preemptions',[]).append(dict(shard=i,**a,time=time.time()));state['shards'][k]=None
            if all(done.values()):
                state['status']='complete_workers_retained';path.write_text(json.dumps(state,indent=2));return
            occupied={v['name'] for k,v in state['shards'].items() if v and not done[int(k)]}
            ready=[]
            for zone,nodes in inv.items():
                for name,status in nodes.items():
                    if status!='READY' or name in occupied:continue
                    log=ROOT/'logs'/f'{name}-create.log'
                    installed=name=='xd-v6e-alllocal-bets-5-0922' or (log.exists() and f'INSTALL_OK TPU={name}' in log.read_text())
                    if installed:ready.append((name,zone))
            for i,task in enumerate(state['tasks']):
                k=str(i)
                if done[i] or state['shards'].get(k) or not ready:continue
                name,zone=ready.pop(0);result=launch(i,task,name,zone,state['runtime'])
                state['shards'][k]=dict(name=name,zone=zone,launched=time.time(),result=result);path.write_text(json.dumps(state,indent=2));print('DISPATCH',i,name,result,flush=True)
            # Maintain at most two pending EW4 requests while work remains.
            r=run(['gcloud','alpha','compute','tpus','queued-resources','list','--zone=europe-west4-a',f'--project={PROJECT}','--format=json(name,state)'])
            if r.returncode:raise RuntimeError(r.stderr[-1000:])
            pending=sum(x['name'].rsplit('/',1)[-1].startswith('xd-v6e-localow-') and x['state'].get('state') in ['WAITING_FOR_RESOURCES','PROVISIONING','ACCEPTED','CREATING'] for x in json.loads(r.stdout))
            target=min(2,sum(not d for d in done.values()))
            if pending<target:
                state['generation']+=1;name=f'xd-v6e-localow-ew4-0922-{state["generation"]}'
                path.write_text(json.dumps(state,indent=2))
                r=run([str(ROOT/'start_standalone_tpu.sh'),name,'v6e-1','europe-west4-a','install_xd_maxtext_jax081.sh',state['runtime']])
                if r.returncode:raise RuntimeError(r.stderr[-1000:])
                print('REFILL',r.stdout.strip(),flush=True)
            state['status']='monitoring';state['last_check']=time.time();path.write_text(json.dumps(state,indent=2))
        except Exception as e:
            state['last_error']=repr(e);path.write_text(json.dumps(state,indent=2));print('ERROR',repr(e),flush=True)
        time.sleep(30)
if __name__=='__main__':main(Path(sys.argv[1]))
