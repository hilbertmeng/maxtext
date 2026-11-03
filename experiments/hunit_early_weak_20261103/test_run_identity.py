import unittest,ast,time
from pathlib import Path
from run_identity import same_attempt,owned_resources
class RunIdentityTest(unittest.TestCase):
 def setUp(self):self.m=dict(attempt_id='v5p-second',commit='runtime',task='probe')
 def test_previous_terminal_failure_does_not_fail_new_run(self):
  for old in [dict(phase='terminal_failure',status='failed'),dict(phase='terminal_failure',status='failed',attempt_id='v6e-first',source_commit='runtime')]:self.assertFalse(same_attempt(old,self.m))
 def test_current_success_and_failure_are_accepted(self):
  for phase in ['acquire','all_arms','terminal_failure']:self.assertTrue(same_attempt(dict(phase=phase,attempt_id='v5p-second',source_commit='runtime'),self.m))
 def test_wrong_source_and_missing_identity_rejected(self):
  self.assertFalse(same_attempt(dict(attempt_id='v5p-second',source_commit='other'),self.m))
  with self.assertRaises(ValueError):same_attempt({},dict(commit='runtime'))
 def test_old_finalizer_cannot_select_new_or_foreign_resources(self):
  rs=[dict(name='old',attempt_id='v6e-first',owner_task='probe'),dict(name='mine',attempt_id='v5p-second',owner_task='probe'),dict(name='foreign',attempt_id='v5p-second',owner_task='other')]
  self.assertEqual([x['name'] for x in owned_resources(rs,self.m)],['mine'])
 def test_controller_accepts_current_resource_and_rejects_foreign_batch(self):
  source=ast.parse((Path(__file__).parent/'controller.py').read_text());fn=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='state');calls=[];ns=dict(m=self.m,time=time,write=lambda name,data:calls.append((name,data)))
  exec(compile(ast.Module(body=[fn],type_ignores=[]),'<actual controller state>','exec'),ns)
  ns['state']('install',name='owned',attempt_id='v5p-second');self.assertEqual(calls[-1][1]['attempt_id'],'v5p-second');self.assertEqual(calls[-1][1]['phase'],'install')
  with self.assertRaises(AssertionError):ns['state']('install',name='foreign',attempt_id='another-batch')
  self.assertEqual(len(calls),1)
if __name__=='__main__':unittest.main()
