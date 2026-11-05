"""Finish one authorized43k analysis when spot capacity appears, then release everything."""
from pathlib import Path
import fcntl,json,os,shlex,shutil,signal,subprocess,time,sys
os.environ['PATH']='/home/mengqy/miniconda3/bin:/home/mengqy/google-cloud-sdk/bin:/usr/local/bin:/usr/bin:/bin'
R=Path(__file__).resolve().parent
EXTERNAL_DRIVER=0
EXTERNAL_CONTROLLER=0
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
 q=subprocess.run([CLI,'--profile','cli_aa800b01d4b95be9','im','+messages-send','--as','bot','--chat-id',GROUP,'--text',text,'--idempotency-key','hunit-prenorm-43k-20261105-auto-'+key,'--format','json'],capture_output=True,text=True,timeout=90)
 assert q.returncode==0,'Feishu CLI failed; see service log'
 x=json.loads(q.stdout);assert x.get('ok') is True and x['data']['message_id']
 d[key]=dict(message_id=x['data']['message_id'],utc=time.time());p.write_text(json.dumps(d,indent=2)+'\n')
def interrupt_once(pid):
 if pid not in interrupted and alive(pid,'bam_hunit_prenorm_43k_20261105/controller.py'):
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
 messages={'installing':'43k弱能量分析已获得独立spot '+d.get('resource',{}).get('type','v5p')+'，正在准备固定诊断环境。正式训练未更改。','inference_baseline':'43k弱能量分析：正在计算原HUnit FastKernel，同32条×2048-token文本，含原生前传保真、弱方向写入和末层整路消融。','inference_prenorm':'43k弱能量分析：只补算新MPreNormQKScale的32条固定文本；之前步数和原模型数据全部复用。','collecting':'新模型43k前传已完成（原模型数据复用），正在汇总配对区间、检查历史复测与能量闭合；通过后删除分析TPU并发布图表。'}
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
 try:notify('capacity-wait','43k只读分析因spot容量不足延后：v5p-8及v5p-16的短队列均未分配，均已确认清理。已启动一次性v5p-8任务，在EW4b/UC1a/UE5a等待最多6小时；获得资源后自动计算、严格核验、出图发群并删除TPU。排队不运行推理，正式训练不受影响。已有谱分解和训练日志结果已缓存。')
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
 while time.time()<deadline and (alive(EXTERNAL_DRIVER,'retry_slice.py') or alive(EXTERNAL_CONTROLLER,'bam_hunit_prenorm_43k_20261105/controller.py') or (child and child.poll() is None)):
  time.sleep(5)
 # Controller finally owns deletion. If it has exited, require its actual audit.
 q=json.loads((R/'resource-audit.json').read_text())
 return q.get('all_absent') is True and not alive(EXTERNAL_CONTROLLER,'bam_hunit_prenorm_43k_20261105/controller.py') and (not child or child.poll() is not None)

