import json,os,signal,subprocess,sys,time
from pathlib import Path
name,zone=sys.argv[1:3]
root=Path('/home/lishengping/mengqy/projects/bam_tpu/bam_hunit_early_weak_v5p_20261103')
manifest=json.loads((root/'install_manifest.json').read_text())
assert manifest['name']==name and manifest['zone']==zone
commit=manifest['commit']
if len(sys.argv)>3 and sys.argv[3]!=commit:raise ValueError('CLI commit differs from locally generated install manifest')
state=root/(name+'-install-state.json')
env=dict(os.environ,CLOUDSDK_ACTIVE_CONFIG_NAME='mengqy-tpu')
base=['gcloud','--project=newproject-1-451205','compute','tpus','tpu-vm']
def save(status,**kwargs):state.write_text(json.dumps(dict(name=name,zone=zone,status=status,utc=time.time(),**kwargs),indent=2)+'\n')
def node():
 try:r=subprocess.run(base+['describe',name,'--zone='+zone,'--format=value(state)'],capture_output=True,text=True,env=env,timeout=45)
 except subprocess.TimeoutExpired:return 'UNKNOWN'
 return r.stdout.strip() if r.returncode==0 else ('ABSENT' if 'NOT_FOUND' in r.stderr else 'UNKNOWN')
def guarded(command,phase,timeout):
 with (root/(name+'-'+phase+'.log')).open('a') as log:
  p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
  start=time.time();save(phase,pid=p.pid)
  while p.poll() is None:
   time.sleep(15)
   status=node()
   if status in ('PREEMPTED','DELETING','STOPPED','ABSENT') or time.time()-start>timeout:
    os.killpg(p.pid,signal.SIGTERM);save('failed',phase=phase,node_state=status)
    raise RuntimeError('Lost diagnostic TPU or timed out: '+status)
  if p.returncode:raise RuntimeError(phase+' failed '+str(p.returncode))
end=time.time()+1200
while True:
 current=node()
 if current=='READY':break
 if current in ('PREEMPTED','DELETING','STOPPED'):
  save('failed',phase='acquire',node_state=current)
  raise RuntimeError('Diagnostic resource terminated before installation: '+current)
 save('waiting_for_ready',node_state=current)
 if time.time()>end:
  save('failed',phase='acquire',node_state=current)
  raise RuntimeError('Resource acquisition timeout')
 time.sleep(20)
installer=root/'install_stream_diag.sh'
guarded(base+['ssh',name,'--zone='+zone,'--worker=0','--command=mkdir -p /home/lishengping/mengqy'],'mkdir-worker-root',120)
guarded(base+['scp',str(installer),name+':/home/lishengping/mengqy/install_stream_diag.sh','--zone='+zone,'--worker=0'],'scp-installer',180)
guarded(base+['ssh',name,'--zone='+zone,'--worker=0','--command=bash /home/lishengping/mengqy/install_stream_diag.sh '+zone.rsplit('-',1)[0]+' '+commit],'install',900)
save('installed',commit=commit);print('INSTALL_COMPLETE',name,zone,flush=True)
