"""Protect scanned-layer identity and bounded intervention plans."""
import unittest
from xlprop_loss_localization_probe import selected_parameter, case_plans


class SelectionTest(unittest.TestCase):
  def test_scan_axis_and_final_layer_are_disjoint(self):
    family = lambda names: 'attention_local_v' if names[-1] == 'V' else 'read_scales'
    groups = {'attention_local_v'}
    for layer in range(28):
      self.assertEqual(selected_parameter(('decoder', 'layers', 'V'), (120, 27, 20, 1), family, groups, layer),
                       (layer < 27, 1))
      self.assertEqual(selected_parameter(('decoder', 'final_local_layer', 'V'), (120, 20, 1), family, groups, layer),
                       (layer == 27, None))
    self.assertEqual(selected_parameter(('decoder', 'layers', 'scale'), (27,), family, groups, 6), (False, None))
    with self.assertRaises(AssertionError):
      selected_parameter(('decoder', 'layers', 'V'), (120, 28, 20, 1), family, groups, 6)

  def test_controls_and_combinations_precede_layer_scan(self):
    plans = case_plans()
    self.assertEqual(len(plans), 37)
    self.assertEqual(plans[0]['families'], [])
    self.assertEqual((plans[1]['families'], plans[1]['direction']), (['ALL'], 'revert'))
    self.assertEqual({p['layer'] for p in plans if 'layer' in p}, set(range(28)))
    for name in ('V_WO', 'O_WO', 'V_O_WO'):
      a, b = [p for p in plans if p['name'] in ('revert_' + name, 'introduce_' + name)]
      self.assertEqual(a['families'], b['families'])
      self.assertEqual({a['direction'], b['direction']}, {'revert', 'introduce'})


if __name__ == '__main__':
  unittest.main()
