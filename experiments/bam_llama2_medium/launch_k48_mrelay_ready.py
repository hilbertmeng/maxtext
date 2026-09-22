import json,pathlib,shlex,subprocess,sys,time
root=pathlib.Path('/home/lishengping/xd/projects');commit,suffix=sys.argv[1:]
base='BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayer'
assert suffix in ('MRelayM3','MRelayM3QKVO','MRelayM3LearnedScale')
run=base+suffix;ident='k48-mrelay-m3'+('-qkvo' if suffix.endswith('QKVO') else '-scale' if suffix.endswith('LearnedScale') else '')
bases=([base+'MRelayM3',base] if suffix.endswith('LearnedScale') else [base]+([base+'MRelayM3'] if suffix.endswith('QKVO') else []))
for attempt in range(480):
 reg=root/'run_registry'/(run+'.json')
 if reg.exists():
  d=json.loads(reg.read_text());assert d['code_commit']==commit
  print('REGISTERED',run,flush=True);break
 ready=[]
 for p in (root/'aot_runs').glob(commit[:7]+'-*.json'):
  d=json.loads(p.read_text())
  if d.get('exp')==run and d.get('commit')==commit and d.get('status')=='ready':
   assert not d.get('cleanup_failures');ready.append(d)
 if not ready:time.sleep(30);continue
 env=dict(EXP=run,ID=ident,MODE='install+train',PRIMARY_ZONE='us-east5-a',BACKUP_ZONES='us-central1-a,europe-west4-b',BRANCH='codex/llf-parameter-matched',CODE_COMMIT=commit,COMPARE_RUNS=','.join(bases),TPU_TYPE='v5p-16',PLANNED_STEPS='13500',LOSS_REPORT_INTERVAL='200',COMPILED_TRAINSTEP_GCS=ready[-1]['artifact'])
 command=shlex.join(['env',*[k+'='+v for k,v in env.items()],'bash',str(root/'run_exp_xd.sh')])
 subprocess.run(['tmux','new-session','-d','-s',ident+'-train',command],check=True)
 print('LAUNCH_SUBMITTED',run,ready[-1]['artifact'],flush=True);break
else:raise TimeoutError(run)
