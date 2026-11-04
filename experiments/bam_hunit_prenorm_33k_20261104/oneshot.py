"""Finish one authorized33k analysis when spot capacity appears, then release everything."""
from pathlib import Path
import fcntl,json,os,shlex,shutil,signal,subprocess,time,sys
os.environ['PATH']='/home/mengqy/miniconda3/bin:/home/mengqy/google-cloud-sdk/bin:/usr/local/bin:/usr/bin:/bin'
R=Path(__file__).resolve().parent
EXTERNAL_DRIVER=4145011
EXTERNAL_CONTROLLER=4145071
GROUP='oc_42bfd5b59295381ff58efd811fe73317'
CLI='/home/mengqy/Projects/weather/radarbot/node_modules/.bin/lark-cli'
child=None;cancel=False;interrupted=set()

def save(phase,**kw):
 p=R/'pipeline-status.tmp';p.write_text(json.dumps(dict(phase=phase,utc=time.time(),pid=os.getpid(),**kw),indent=2)+'\n');p.replace(R/'pipeline-status.json')
def alive(pid,needle):
 p=Path('/proc',str(pid),'cmdline')
 try:return needle.encode() in p.read_bytes()
 except FileNotFoundError:return False
def notify(key,text):
 p=R/'pipeline-receipts.json';d=json.loads(p.read_text()) if p.exists() else {}
 if key in d:return
 q=subprocess.run([CLI,'--profile','cli_aa800b01d4b95be9','im','+messages-send','--as','bot','--chat-id',GROUP,'--text',text,'--idempotency-key','hunit-prenorm-33k-20261104-auto-'+key,'--format','json'],capture_output=True,text=True,timeout=90)
 assert q.returncode==0,'Feishu CLI failed; see service log'
 x=json.loads(q.stdout);assert x.get('ok') is True and x['data']['message_id']
 d[key]=dict(message_id=x['data']['message_id'],utc=time.time());p.write_text(json.dumps(d,indent=2)+'\n')
def interrupt_once(pid):
 if pid not in interrupted and alive(pid,'bam_hunit_prenorm_33k_20261104/controller.py'):
  interrupted.add(pid);os.kill(pid,signal.SIGINT)
def handle(sig,frame):
 global cancel
 cancel=True
 for pid in [child.pid if child else EXTERNAL_CONTROLLER]:
  interrupt_once(pid)
for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,handle)
def checkpoint():
 if cancel:raise InterruptedError('Analysis canceled; waiting for owned-resource cleanup')
def progress():
 checkpoint()
 p=R/'state.json';d=json.loads(p.read_text()) if p.exists() else {}
 phase=d.get('phase','waiting')
 if phase=='inference':phase+='_'+d.get('worker',{}).get('arm','starting')
 save(phase,controller=d)
 messages={'installing':'33k弱能量分析已获得独立spot '+d.get('resource',{}).get('type','v5p')+'，正在准备固定诊断环境。正式训练未更改。','inference_baseline':'33k弱能量分析：正在计算原HUnit FastKernel，同32条×2048-token文本，含原生前传保真、弱方向写入和末层整路消融。','inference_prenorm':'33k弱能量分析：原模型计算完成，正在计算MPreNormQKScale；随后核验全部32对样本并对比13k/23k/33k。','collecting':'33k两组前传已完成，正在汇总配对区间、检查历史复测与能量闭合；通过后删除分析TPU并发布图表。'}
 if phase in messages:
  try:notify(phase,messages[phase])
  except Exception as e:print('PROGRESS_NOTIFICATION_RETRY',type(e).__name__,flush=True)

def regional_wait():
 audit=json.loads((R/'resource-audit.json').read_text());assert audit['all_absent']
 for name in ['controller.py','controller.log','state.json','resources.json','resource-audit.json']:
  p=R/name;shutil.copy2(p,p.with_name(p.stem+'-attempt3'+p.suffix))
 (R/'resources.json').write_text('[]\n')
 p=R/'controller.py';s=p.read_text().replace('xd-v5p16-mqy-hunpre33-1104c-','xd-v5p8-mqy-hunpre33-1104d-').replace('v5p-16','v5p-8').replace('time.time()-started<900','time.time()-started<21600').replace('within 15 minutes','within 6 hours');p.write_text(s)
 for arm in ['baseline','prenorm']:
  p=R/(arm+'.json');m=json.loads(p.read_text());m['accelerator_type']='v5p-8';m['scope']=m['scope'].replace('v5p-16 (worker0 only, four isolated single-v5p-chip processes)','v5p-8');m['execution_note']='Bounded6h spot acquisition across EW4b/UC1a/UE5a after unavailable v5p8/16 short queues; unchanged isolated single-v5p-chip arithmetic and fidelity gates.';p.write_text(json.dumps(m,indent=2)+'\n')
 p=R/'publish.py';p.write_text(p.read_text().replace('spot v5p-16（仅worker0，四个独立单v5p芯片进程）','spot v5p-8'))
 try:notify('capacity-wait','33k只读分析因spot容量不足延后：v5p-8及v5p-16的短队列均未分配，均已确认清理。已启动一次性v5p-8任务，在EW4b/UC1a/UE5a等待最多6小时；获得资源后自动计算、严格核验、出图发群并删除TPU。排队不运行推理，正式训练不受影响。已有谱分解和训练日志结果已缓存。')
 except Exception as e:print('CAPACITY_NOTIFICATION_FAILED',type(e).__name__,flush=True)
 global child
 with (R/'controller.log').open('w') as log:
  child=subprocess.Popen([sys.executable,str(R/'controller.py')],stdout=log,stderr=subprocess.STDOUT)
  while child.poll() is None:progress();time.sleep(20)
  rc=child.wait();child=None
 q=json.loads((R/'resource-audit.json').read_text());q['previous_attempts']=audit;q['resources']=audit['resources']+q['resources'];q['all_absent']=audit['all_absent'] and q['all_absent'];(R/'resource-audit.json').write_text(json.dumps(q,indent=2)+'\n')
 return rc

def await_cleanup():
 deadline=time.time()+1800
 while time.time()<deadline and (alive(EXTERNAL_DRIVER,'retry_slice.py') or alive(EXTERNAL_CONTROLLER,'bam_hunit_prenorm_33k_20261104/controller.py') or (child and child.poll() is None)):
  time.sleep(5)
 # Controller finally owns deletion. If it has exited, require its actual audit.
 q=json.loads((R/'resource-audit.json').read_text())
 return q.get('all_absent') is True and not alive(EXTERNAL_CONTROLLER,'bam_hunit_prenorm_33k_20261104/controller.py') and (not child or child.poll() is not None)

if __name__=='__main__':
 with (R/'oneshot.lock').open('a+b') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   save('attached_to_spot_v5p16',driver_pid=EXTERNAL_DRIVER)
   while alive(EXTERNAL_DRIVER,'retry_slice.py'):progress();time.sleep(20)
   checkpoint();state=json.loads((R/'state.json').read_text())
   if state.get('phase')!='complete':
    assert state.get('error')=='No spot v5p-16 ready within 15 minutes',state
    assert regional_wait()==0,'Long spot acquisition/inference failed; see controller state'
   checkpoint();state=json.loads((R/'state.json').read_text());assert state['verified'] and state['resources_released']
   save('publishing')
   if not (R/'completion.json').exists():
    subprocess.run([sys.executable,str(R/'finalize.py')],check=True)
   else:
    assert json.loads((R/'completion.json').read_text())['complete'] and json.loads((R/'resource-audit.json').read_text())['all_absent']
   for attempt in range(3):
    try:notify('final',(R/'final-message.txt').read_text());break
    except Exception:
     if attempt==2:raise
     time.sleep(10)
   m=json.loads((R/'prenorm.json').read_text());gcs=m['gcs_root'].rsplit('/',1)[0]+'/report/'
   subprocess.run(['gsutil','-q','cp']+[str(R/n) for n in ['results.json','verification.json','cross-step.json','resource-audit.json','completion.json','pipeline-receipts.json']]+[gcs],check=True,timeout=180)
   exp=Path(m['worktree'])/'experiments'/R.name;exp.mkdir(exist_ok=True)
   for n in ['results.json','verification.json','cross-step.json','resource-audit.json','completion.json','pipeline-receipts.json']:
    p=exp/n
    if p.exists():assert p.read_bytes()==(R/n).read_bytes(),('Refuse overwrite of existing result',p)
    else:shutil.copy2(R/n,p)
   save('complete',url=json.loads((R/'completion.json').read_text())['url'],resources_released=True)
  except BaseException as e:
   if child and child.poll() is None:interrupt_once(child.pid)
   interrupt_once(EXTERNAL_CONTROLLER)
   absent=await_cleanup()
   save('failed',error=str(e),resources_released=absent)
   try:notify('failed','33k弱能量分析尚未完成：'+str(e)[:500]+'。分析资源回收核验：'+('已确认删除' if absent else '尚未确认，需检查resource-audit.json')+'。已有谱分解/训练梯度结果仍保留；未将不完整前传写成结论。')
   except Exception:pass
   raise
