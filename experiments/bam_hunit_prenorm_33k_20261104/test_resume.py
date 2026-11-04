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
