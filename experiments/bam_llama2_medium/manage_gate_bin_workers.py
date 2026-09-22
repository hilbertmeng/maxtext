"""Task-owned rolling TPU dispatch; run on tpu-ag, retain workers and queues.

Only handles this task's three fixed shards. Never deletes resources or retries
a failed experiment blindly. Stops dispatching after all shard DONE objects exist.
"""
import concurrent.futures
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

PROJECT='newproject-1-451205'
ROOT=Path('/home/lishengping/xd/projects')
COMMIT='3215eb881f2685201b758ed3d1ddc7235e80a78f'
GCS='gs://newproject-1-llm_projects_europe-west4/log/diagnostics/alllocal-write-gate-bets-0922/negative_binned/fp32'
ZONES=['us-central1-a','europe-west4-a','us-east5-a']
PREFIX='xd-v6e-alllocal-'


def run(args,timeout=60):
    return subprocess.run(args,capture_output=True,text=True,timeout=timeout)


def inventory(zone):
    result={}
    for kind,args in [('nodes',['compute','tpus','tpu-vm']),('queues',['alpha','compute','tpus','queued-resources'])]:
        r=run(['gcloud',*args,'list',f'--zone={zone}',f'--project={PROJECT}','--format=json(name,state)'])
        if r.returncode:raise RuntimeError(r.stderr[-1000:])
        result[kind]={x['name'].rsplit('/',1)[-1]:x['state'] for x in json.loads(r.stdout)
                      if x['name'].rsplit('/',1)[-1].startswith(PREFIX)}
    return zone,result


def done(i):
    return i,run(['gsutil','-q','stat',f'{GCS}/shard{i}/worker/DONE']).returncode==0


def launch(i,name,zone):
    out=f'/tmp/alllocal-negative-binned-shard{i}-0922'
    source=f'{GCS}/shard{i}'
    job=f'''set -e
cd {ROOT}/maxtext
gsutil -m rsync -r {source}/worker {out} || true
export JAX_DEFAULT_MATMUL_PRECISION=highest BET_OUTPUT={out} BET_GCS={source}
export BET_GRADIENT_CACHE_GCS={GCS}/gradient_cache.npz BET_START={32+32*i} BET_STOP={64+32*i}
unset BET_ZERO_CUTOFF
set +e
bash experiments/bam_llama2_medium/run_write_gate_bets.sh dtype=float32 matmul_precision=highest
rc=$?
echo "$rc" > {out}/EXIT
gsutil cp {out}/EXIT {source}/worker/EXIT
exit "$rc"
'''
    command=f'''set -e
cd {ROOT}/maxtext
git fetch origin codex/alllocal-write-geometry
git checkout --detach {COMMIT}
mkdir -p {out}
printf %s {shlex.quote(job)} > {out}/dispatch_job.sh
nohup bash {out}/dispatch_job.sh >{out}/dispatch_launcher.log 2>&1 </dev/null &
echo LAUNCH_PID=$!
'''
    r=run(['gcloud','compute','tpus','tpu-vm','ssh','--internal-ip',name,f'--zone={zone}',
           f'--project={PROJECT}','--worker=0','--command',command],90)
    if r.returncode or 'LAUNCH_PID=' not in r.stdout:raise RuntimeError((r.stdout+r.stderr)[-2000:])
    return r.stdout.strip()


def main(path):
    state=json.loads(path.read_text())
    while not (path.parent/'STOP_GATE_BIN_DISPATCH').exists():
        try:
            with concurrent.futures.ThreadPoolExecutor(6) as pool:
                inv=dict(pool.map(inventory,ZONES))
                completed=dict(pool.map(done,range(3)))
            for i in range(3):
                key=str(i);assignment=state['shards'].get(key)
                if completed[i]:
                    state['completed'][key]=True
                    continue
                if assignment and inv[assignment['zone']]['nodes'].get(assignment['name']) in ['PREEMPTED','DELETING','TERMINATED']:
                    state.setdefault('preemptions',[]).append(dict(shard=i,**assignment,time=time.time()))
                    state['shards'][key]=None
            if all(completed.values()):
                state['status']='all_shards_done_workers_retained'
                path.write_text(json.dumps(state,indent=2));print(state['status'],flush=True);return
            occupied={x['name'] for k,x in state['shards'].items() if x and not completed[int(k)]}
            ready=[]
            for zone,data in inv.items():
                for name,status in data['nodes'].items():
                    log=ROOT/'logs'/f'{name}-create.log'
                    if status=='READY' and name not in occupied and log.exists() and f'INSTALL_OK TPU={name}' in log.read_text():
                        ready.append((name,zone))
            for i in range(3):
                key=str(i)
                if completed[i] or state['shards'].get(key) or not ready:continue
                name,zone=ready.pop(0)
                state['shards'][key]=dict(name=name,zone=zone,launch_pending=time.time())
                path.write_text(json.dumps(state,indent=2))
                result=launch(i,name,zone)
                state['shards'][key]=dict(name=name,zone=zone,launched=time.time(),result=result)
                path.write_text(json.dumps(state,indent=2));print('DISPATCH',i,name,result,flush=True)
            ew=inv['europe-west4-a']
            pending=sum(v.get('state') in ['WAITING_FOR_RESOURCES','PROVISIONING','ACCEPTED','CREATING'] for v in ew['queues'].values())
            target_pending=min(2,sum(not value for value in completed.values()))
            if pending < target_pending and not ready:
                state['generation']=state.get('generation',0)+1
                name=f'{PREFIX}auto-ew4-0922-{state["generation"]}'
                path.write_text(json.dumps(state,indent=2))
                r=run([str(ROOT/'start_standalone_tpu.sh'),name,'v6e-1','europe-west4-a','install_xd_maxtext_jax081.sh',COMMIT])
                if r.returncode:raise RuntimeError(r.stderr[-1000:])
                print('REFILL',r.stdout.strip(),flush=True)
            state['last_check']=time.time();state['status']='monitoring'
            path.write_text(json.dumps(state,indent=2))
        except Exception as e:
            state['last_error']=str(e);path.write_text(json.dumps(state,indent=2));print('CHECK_ERROR',repr(e),flush=True)
        time.sleep(30)


if __name__=='__main__':main(Path(sys.argv[1]))
