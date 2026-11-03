"""Check one registered training RUN every600s; notify and analyze committed23k."""
from pathlib import Path
import fcntl, hashlib, json, os, shlex, subprocess, sys, time

R=Path(__file__).resolve().parent
RUN='BamPythia2p8BHUnit300BFastKernelMPreNormQKScale'
CHAT='oc_42bfd5b59295381ff58efd811fe73317'
CLI='/home/mengqy/Projects/weather/radarbot/node_modules/.bin/lark-cli'
PROFILE='cli_aa800b01d4b95be9'
INTERVAL=600
SSH=['ssh','-S','/tmp/ssh-tpu-ag-mengqy-readscale.sock','-o','BatchMode=yes','-o','ConnectTimeout=10','tpu-ag']
R.mkdir(parents=True,exist_ok=True)
S=R/'watch-state.json'

def save(d):
    p=S.with_suffix('.tmp');p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n');p.replace(S)

def notify(state,key,text):
    if key in state.setdefault('delivered',{}):return True
    digest=hashlib.sha256((CHAT+RUN+key).encode()).hexdigest()[:40]
    try:
        p=subprocess.run([CLI,'--profile',PROFILE,'im','+messages-send','--as','bot','--chat-id',CHAT,
            '--text',text,'--idempotency-key','pre23-'+digest,'--format','json'],capture_output=True,text=True,timeout=60)
        d=json.loads(p.stdout)
        assert p.returncode==0 and d.get('ok') is True and d.get('data',{}).get('message_id')
        state['delivered'][key]=dict(message_id=d['data']['message_id'],utc=time.time())
        save(state);return True
    except Exception as e:
        state['notification_error']=dict(key=key,error=str(e)[:500],utc=time.time());save(state);return False

def snapshot():
    code='''
import json,pathlib,subprocess
run="BamPythia2p8BHUnit300BFastKernelMPreNormQKScale"
p=pathlib.Path('/home/lishengping/xd/projects/run_registry')/(run+'.json');d=json.loads(p.read_text())
keys=['run_name','code_commit','base_output_directory','last_progress_step','last_progress_utc','last_committed_checkpoint_step','last_checkpoint_observed_utc','declared_status','zone','tpu','planned_steps','observed_node_state','preemption_active']
q={k:d.get(k) for k in keys};q['checkpoint']=d['base_output_directory'].rstrip('/')+'/'+run+'/checkpoints/23000/items'
q['checkpoint_committed']=False
if max(d.get('last_progress_step') or 0,d.get('last_committed_checkpoint_step') or 0)>=23000:
 p=subprocess.run(['gsutil','-q','stat']+[q['checkpoint']+'/'+x for x in ['commit_success.txt','_METADATA','manifest.ocdbt']],capture_output=True,text=True,timeout=90)
 q['checkpoint_committed']=p.returncode==0
 if p.returncode and not any(x in p.stderr for x in ['No URLs matched','matched no objects','404']):q['checkpoint_probe_error']=p.stderr[-500:]
print(json.dumps(q))
'''
    cmd='env CLOUDSDK_ACTIVE_CONFIG_NAME=mengqy-tpu '+shlex.join(['python3','-c',code])
    p=subprocess.run(SSH+[cmd],capture_output=True,text=True,timeout=115)
    assert p.returncode==0,p.stderr[-500:]
    d=json.loads(p.stdout);assert d['run_name']==RUN;return d

PHASES={
 'head_spectra':'23k分析：正在计算新旧模型输出头的弱读出子空间。',
 'acquiring':'23k分析：参数谱已准备，正在申请只用于分析的spot v5p-8。',
 'installing':'23k分析：spot v5p-8已就绪，正在准备原生推理环境。',
 'inference_baseline':'23k分析：正在复测原HUnit FastKernel，使用与13k相同的32×2048文本。',
 'inference_prenorm':'23k分析：原模型已完成，正在计算MPreNormQKScale的弱能量和末层消融。',
 'collecting':'23k分析：新旧模型前传已完成，正在取回结果并核验统计口径、能量闭合。',
 'publishing':'23k分析：配对计算与核验完成，分析TPU已回收，正在整理图表与结论。'
}

def main():
    with (R/'watch.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=json.loads(S.read_text()) if S.exists() else dict(delivered={})
        state.update(pid=os.getpid(),run_name=RUN,chat_id=CHAT,interval_seconds=INTERVAL)
        next_check=0
        while True:
            if time.time()>=next_check and state.get('status')!='complete':
                tick=time.time();next_check=tick+INTERVAL
                try:
                    snap=snapshot();state.update(snapshot=snap,last_check_utc=tick,next_check_utc=next_check,consecutive_errors=0)
                    ready=snap['checkpoint_committed'];state['status']='analyzing' if ready else 'monitoring'
                    save(state)
                    notify(state,'monitor-start',f'已启用23k checkpoint监测：{RUN}\n当前step：{snap.get("last_progress_step")}；最近checkpoint：{snap.get("last_committed_checkpoint_step")}。\n每10分钟检查；确认23k checkpoint完整提交后立即通知，并开始与原FastKernel的配对弱能量分析。')
                    if ready:
                        notify(state,'checkpoint-ready',f'检测到{RUN}已完成23k，checkpoint完整提交到bucket。\n{snap["checkpoint"]}\n开始与原FastKernel @23k分析对比；使用同32条2048-token文本、当期及固定弱子空间，并逐阶段同步进展。')
                        child=state.get('analysis_pid');cmdline=Path(f'/proc/{child}/cmdline')
                        active=cmdline.exists() and b'run_analysis.py' in cmdline.read_bytes()
                        if not active and not (R/'goal-progress.json').exists():
                            assert (R/'run_analysis.py').exists()
                            with (R/'analysis.log').open('a') as log:
                                p=subprocess.Popen([sys.executable,str(R/'run_analysis.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                            state['analysis_pid']=p.pid;save(state)
                except Exception as e:
                    state['consecutive_errors']=state.get('consecutive_errors',0)+1
                    state.update(last_check_utc=tick,next_check_utc=next_check,last_error=str(e)[:1000]);save(state)
                    if state['consecutive_errors']>=3:notify(state,'monitor-error',f'23k监测连续3次未能取得可靠状态：{str(e)[:350]}。这不代表训练停止，正在保留原训练并继续检查。')
            # A transient Feishu failure must not silently lose the required checkpoint
            # notice or let a final result close the watcher before that notice is confirmed.
            cached=state.get('snapshot',{})
            if cached.get('checkpoint_committed') and 'checkpoint-ready' not in state.get('delivered',{}):
                notify(state,'checkpoint-ready',f'检测到{RUN}已完成23k，checkpoint完整提交到bucket。\n{cached["checkpoint"]}\n正在执行与原FastKernel @23k的配对弱能量分析，后续同步进展和结果。')
            progress=R/'goal-progress.json'
            if progress.exists():
                d=json.loads(progress.read_text());phase=d.get('phase')
                if phase in PHASES:notify(state,'analysis-'+phase,PHASES[phase])
                if phase=='complete':
                    if 'checkpoint-ready' in state.get('delivered',{}) and notify(state,'analysis-final',d['feishu_text']):
                        state.update(status='complete',completed_utc=time.time(),result=d);save(state);return
                elif phase=='failed':
                    notify(state,'analysis-failed','23k分析遇到问题，需要继续排查：'+d.get('error','')[:600]+'\n已保留原始结果；分析TPU回收状态见任务记录。')
                    state['status']='analysis_needs_attention';save(state)
            time.sleep(20 if state.get('status')=='analyzing' else min(30,max(1,next_check-time.time())))

if __name__=='__main__':main()
