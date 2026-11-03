"""Synthetic retry/receipt contract: no network, TPU, or real Feishu delivery."""
from pathlib import Path
import importlib.util, json, subprocess, tempfile
from unittest import mock

source=Path(__file__).with_name('watch23.py')
spec=importlib.util.spec_from_file_location('watch23_delivery_test',source)
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
with tempfile.TemporaryDirectory(prefix='watch23-retry-') as temp:
    w.R=Path(temp);w.S=w.R/'watch-state.json'
    (w.R/'goal-progress.json').write_text(json.dumps(dict(phase='complete',feishu_text='SYNTHETIC FINAL')))
    snap=dict(run_name=w.RUN,last_progress_step=23020,last_committed_checkpoint_step=23000,
              checkpoint_committed=True,checkpoint='gs://synthetic/checkpoints/23000/items')
    attempts=[];ready_attempts=0
    def fake_run(args,**kwargs):
        global ready_attempts
        assert args[0]==w.CLI and args[args.index('--chat-id')+1]==w.CHAT
        text=args[args.index('--text')+1];key=args[args.index('--idempotency-key')+1]
        kind='ready' if text.startswith('检测到') else ('final' if text=='SYNTHETIC FINAL' else 'start')
        ok=True
        if kind=='ready':ready_attempts+=1;ok=ready_attempts>2
        attempts.append(dict(kind=kind,key=key,ok=ok))
        return subprocess.CompletedProcess(args,0,json.dumps(dict(ok=ok,data={'message_id':'synthetic-'+kind} if ok else {})),'')
    with mock.patch.object(w,'snapshot',return_value=snap),mock.patch.object(w.subprocess,'run',side_effect=fake_run),mock.patch.object(w.time,'sleep',return_value=None):
        w.main()
    state=json.loads(w.S.read_text())
    assert state['status']=='complete'
    assert len({a['key'] for a in attempts if a['kind']=='ready'})==1
    assert ready_attempts==3
    assert [a['kind'] for a in attempts]==['start','ready','ready','ready','final']
    assert state['delivered']['checkpoint-ready']['message_id']=='synthetic-ready'
    assert state['delivered']['analysis-final']['message_id']=='synthetic-final'
    # Re-reading a confirmed cursor cannot resend even if called again.
    with mock.patch.object(w.subprocess,'run',side_effect=AssertionError('duplicate delivery')):
        assert w.notify(state,'checkpoint-ready','ignored')
print('PASS: checkpoint notice retries share an idempotency key; final waits for confirmed notice; confirmed receipts do not resend. No real messages or cloud requests.')
