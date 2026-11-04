"""Check cancellation and stale-audit behavior without contacting GCP."""
import ast,json,signal,tempfile
from pathlib import Path
from types import SimpleNamespace
src=Path(__file__).with_name('oneshot.py').read_text();tree=ast.parse(src)
names={'interrupt_once','handle','await_cleanup'}
code=compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),'oneshot-functions','exec')
calls=[]
ctx=dict(interrupted=set(),alive=lambda p,n:p in [21,22],os=SimpleNamespace(kill=lambda *a:calls.append(a)),signal=signal,child=SimpleNamespace(pid=21),EXTERNAL_CONTROLLER=22,EXTERNAL_DRIVER=23,cancel=False)
exec(code,ctx)
ctx['handle'](signal.SIGINT,None);ctx['handle'](signal.SIGINT,None);ctx['interrupt_once'](21)
assert calls==[(21,signal.SIGINT)],calls
ctx['child']=None;ctx['handle'](signal.SIGTERM,None);ctx['child']=SimpleNamespace(pid=21);ctx['interrupt_once'](21)
assert calls==[(21,signal.SIGINT),(22,signal.SIGINT)],calls
with tempfile.TemporaryDirectory() as td:
 Path(td,'resource-audit.json').write_text(json.dumps({'all_absent':True}))
 ctx.update(R=Path(td),json=json,child=None,time=SimpleNamespace(time=iter([0,2000]).__next__,sleep=lambda _:None))
 assert ctx['await_cleanup']() is False,'Old all_absent audit must not hide a still-running controller'
 ctx.update(alive=lambda p,n:False,time=SimpleNamespace(time=iter([0,2000]).__next__,sleep=lambda _:None))
 assert ctx['await_cleanup']() is True
print('CLEANUP_CHECKS_PASSED: repeated signal, controller transition, stale-audit rejection, terminal cleanup')
