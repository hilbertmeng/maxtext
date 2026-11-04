"""Keep preemption retry narrow; mirror only finished cache payloads."""
import ast,re,tempfile,os,time
from pathlib import Path
from types import SimpleNamespace
R=Path(__file__).resolve().parent
src=ast.parse((R/'resume.py').read_text());node=next(n for n in src.body if isinstance(n,ast.FunctionDef) and n.name=='transient_spot_loss');ctx={'re':re};exec(compile(ast.Module(body=[node],type_ignores=[]),'resume-fn','exec'),ctx);f=ctx['transient_spot_loss']
assert f('Diagnostic spot TPU lost; complete per-text GCS chunks retained')
assert f('ERROR: This TPU has state "DELETING", so it cannot be currently connected to.')
assert f('This TPU has state "PREEMPTED"')
assert not f('Native weak power fidelity failed')
assert not f('Permission denied (publickey)')
node=next(n for n in ast.parse((R/'worker_jobs.py').read_text()).body if isinstance(n,ast.ClassDef) and n.name=='CacheMirror')
calls=[];ctx={'time':time,'subprocess':SimpleNamespace(run=lambda args,**kw:(calls.append(args) or SimpleNamespace(returncode=0)))};exec(compile(ast.Module(body=[node],type_ignores=[]),'cache-class','exec'),ctx)
with tempfile.TemporaryDirectory() as td:
 root=Path(td);a=root/'complete-cache';a.write_bytes(b'finished');os.utime(a,(time.time()-10,time.time()-10));(root/'writing-cache').write_bytes(b'in progress');(root/'meta-atime').write_bytes(b'time')
 c=ctx['CacheMirror'].__new__(ctx['CacheMirror']);c.paths=[root];c.uri='gs://test-only-not-contacted';c.sent={};c.sync();assert len(calls)==1 and str(a) in calls[0] and not any('writing-cache' in x or 'meta-atime' in x for x in calls[0]);c.sync();assert len(calls)==1
print('RESUME_CHECKS_PASSED: deleting-state retry, numeric/auth failures stop, stable-only cache mirror, dedup')
# Ready GCS outputs must pass collection before they can skip resource acquisition.
import json
fn=next(n for n in ast.parse((R/'resume.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='collect_if_ready')
with tempfile.TemporaryDirectory() as td:
 root=Path(td)
 for arm in ['baseline','prenorm']:(root/(arm+'.json')).write_text(json.dumps({'gcs_root':'gs://fake-not-contacted/'+arm}))
 calls=[];ready=[False];valid=[True]
 def run(args,**kw):
  calls.append(args)
  if args[0]=='gsutil':return SimpleNamespace(returncode=0 if ready[0] else 1)
  assert kw['env']['JAX_PLATFORMS']=='cpu'
  (root/'results.json').write_text(json.dumps({'complete':valid[0],'verification':{arm:{'sequences':32} for arm in ['baseline','prenorm']}}))
  return SimpleNamespace(returncode=0)
 ctx={'R':root,'json':json,'subprocess':SimpleNamespace(run=run),'os':os}
 exec(compile(ast.Module(body=[fn],type_ignores=[]),'recovery-fn','exec'),ctx)
 assert ctx['collect_if_ready']() is False and len(calls)==1
 ready[0]=True;assert ctx['collect_if_ready']() is True and len(calls)==3
 valid[0]=False
 try:ctx['collect_if_ready']()
 except AssertionError:pass
 else:raise AssertionError('Invalid collection accepted')
print('RESULT_RECOVERY_CHECKS_PASSED: missing outputs wait, ready outputs collect on CPU, invalid collection stops')
