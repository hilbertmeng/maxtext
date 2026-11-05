"""Guard the user's no-rerun constraint and artifact-first recovery."""
import ast,json
from pathlib import Path
R=Path(__file__).resolve().parent
m=json.loads((R/'prenorm.json').read_text());assert m['arm']=='prenorm' and m['steps']==[43000] and list(m['checkpoints'])==['43000']
assert m['checkpoints']['43000'].endswith('/43000/items')
for filename in ['controller.py','worker_jobs.py','head_inventory.py']:
 tree=ast.parse((R/filename).read_text())
 loops=[n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='arm']
 assert loops and all(ast.literal_eval(n.iter)==['prenorm'] for n in loops),filename
w=ast.parse((R/'wo_geometry.py').read_text())
reports=next(n for n in w.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='reports' for t in n.targets));assert [ast.literal_eval(k) for k in reports.value.keys]==[43000]
for n in ast.walk(w):
 if isinstance(n,ast.comprehension) and isinstance(n.target,ast.Name) and n.target.id=='arm':assert ast.literal_eval(n.iter)==['prenorm']
s=(R/'controller.py').read_text();loop=s[s.index('while time.time()<deadline:'):];assert loop.index('if artifact_ready()')<loop.index('if node(chosen)')
assert "if ready():collect();done=True" in (R/'run.py').read_text()
print('SCOPE_CHECKS_PASSED: new43k only on TPU/CPU parameter restore; old cached records only; GCS checked before node status')
