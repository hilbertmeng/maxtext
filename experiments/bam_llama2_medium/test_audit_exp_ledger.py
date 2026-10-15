"""Coverage checks accept natural Chinese conclusions without English marker words."""
import unittest

from audit_exp_ledger import audit


class AuditLedgerTest(unittest.TestCase):
  def check_record(self, comment, status='complete', commit='1234567abcdef'):
    source = f'class Example:\n  # code_commit: 1234567\n  # {comment}\n  model_name = "Example"\n'
    record = dict(run_name='Example', code_commit=commit, declared_status=status)
    return audit(source, [record], ['Example'])

  def test_chinese_completed_conclusion(self):
    self.assertEqual(self.check_record('完成13500；结论：收益稳定保留。'), [])

  def test_chinese_early_stop(self):
    self.assertEqual(self.check_record('用户停于5000；结论：收益持续收窄。', 'stopped'), [])

  def test_chinese_pause_is_provisional(self):
    self.assertEqual(self.check_record('用户暂停于5000；暂时结论：接近持平。', 'paused'), [])

  def test_not_completed_is_not_completion(self):
    errors = self.check_record('未完成13500；gap -.01。')
    self.assertTrue(any('terminal/pause' in error for error in errors))

  def test_english_records_remain_supported(self):
    self.assertEqual(self.check_record('Completed 13500; vs parent gap -.01.'), [])

  def test_runtime_mismatch_is_not_hidden_by_conclusion(self):
    errors = self.check_record('完成13500；结论：收益稳定。', commit='7654321abcdef')
    self.assertTrue(any('registered runtime' in error for error in errors))

  def test_status_alone_is_not_an_outcome(self):
    errors = self.check_record('完成13500；TB同步。')
    self.assertTrue(any('outcome' in error for error in errors))


if __name__ == '__main__':
  unittest.main()
