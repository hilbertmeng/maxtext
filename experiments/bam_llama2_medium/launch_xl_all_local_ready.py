"""Launch XL all-local experiment after exact-runtime AOT readiness."""
import json,pathlib,shlex,subprocess,sys,time
root=pathlib.Path('/home/lishengping/xd/projects');commit=sys.argv[1]
parent='BamXLSharedBasisQKConcatStaticLocalVOSharedC8IndependentGatesK96QK96SharedRank4MLPPerLayer'
exp=parent+'AllLocal';ident='xl-k96-rank4-all-local'
deadline=time.monotonic()+14400
while time.monotonic()<deadline:
 reg=root/'run_registry'/(exp+'.json')
 if reg.exists():
  assert json.loads(reg.read_text())['code_commit']==commit
  print('REGISTERED',exp,flush=True);break
 states=[]
 for path in (root/'aot_runs').glob(commit[:7]+'-*.json'):
  state=json.loads(path.read_text())
  if state.get('exp')==exp and state.get('commit')==commit and state.get('status')=='ready':
   assert not state.get('cleanup_failures'),state
   states.append(state)
 if not states:time.sleep(30);continue
 state=states[-1]
 env=dict(EXP=exp,ID=ident,MODE='install+train',PRIMARY_ZONE='us-east5-a',BACKUP_ZONES='us-central1-a,europe-west4-b',BRANCH='codex/llf-parameter-matched',CODE_COMMIT=commit,COMPARE_RUNS=parent,TPU_TYPE='v5p-32',PLANNED_STEPS='50000',LOSS_REPORT_INTERVAL='500',COMPILED_TRAINSTEP_GCS=state['artifact'])
 command=shlex.join(['env',*[k+'='+v for k,v in env.items()],'bash',str(root/'run_exp_xd.sh')])
 subprocess.run(['tmux','new-session','-d','-s',exp+'-TPU'+ident+'-xd',command],check=True)
 print('TRAIN_LAUNCH_SUBMITTED',exp,state['artifact'],flush=True);break
else:raise TimeoutError(exp)
