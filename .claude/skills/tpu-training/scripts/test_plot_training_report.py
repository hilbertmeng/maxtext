"""Keep registry arithmetic and unequal reporting horizons intact when rendering."""
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import plot_training_report as plot


def example():
  return {'runs': [{'run_name': 'Run', 'pending_report': {
      'reports': [{'run': 'Run', 'base': 'Parent', 'series': [
          {'step': 400, 'gap': -.4}, {'step': 600, 'gap': -.03},
          {'step': 800, 'gap': -.02}, {'step': 1000, 'gap': -.01}]}],
      'gain_ratios': [{'run': 'Run', 'comparator': 'Parent', 'reference': 'MHA',
          'series': [{'step': 600, 'ratio': 7.25, 'status': 'ok'},
                     {'step': 800, 'ratio': None, 'status': 'near_zero_denominator'}]}]
  }}, {'run_name': 'Slow', 'pending_report': {
      'reports': [{'run': 'Slow', 'base': 'Parent',
                   'series': [{'step': 600, 'gap': .01}]}], 'gain_ratios': []}},
      {'run_name': 'Unchanged', 'pending_report': None}]}


class PlotRegistryReportTest(unittest.TestCase):
  def test_same_native_series_from_batch_and_single_event(self):
    payload = example()
    a = plot.collect([payload], runs=['Run'])
    b = plot.collect([payload['runs'][0]['pending_report']])
    self.assertEqual(a, b)
    gaps, ratios = a
    self.assertEqual(sorted(gaps['Run', 'Parent']), [600, 800, 1000])
    # This deliberately unusual native ratio must never be recalculated from gaps.
    self.assertEqual(ratios['Run', 'Parent', 'MHA'][600]['ratio'], 7.25)
    self.assertEqual(plot.collect([payload], bases=['Other']), ({}, {}))

  def test_undefined_and_missing_ratios_stay_blank(self):
    self.assertTrue(math.isnan(plot.ratio_value({})))
    self.assertTrue(math.isnan(plot.ratio_value({'ratio': None, 'status': 'ok'})))
    self.assertTrue(math.isnan(plot.ratio_value(
        {'ratio': 1., 'status': 'nonpositive_denominator'})))
    self.assertEqual(plot.ratio_value({'ratio': .98, 'status': 'ok'}), .98)

  def test_real_png_and_provenance_preserve_unequal_horizons_and_identities(self):
    with tempfile.TemporaryDirectory() as directory:
      result = plot.render([example()], directory, labels={'Run': 'Experiment'})
      image = Path(result['figure'])
      self.assertTrue(image.is_absolute())
      self.assertEqual(image.read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
      meta = json.loads(Path(result['provenance']).read_text())
      shown = {row['run']: row for row in meta['shown']}
      self.assertEqual(shown['Run']['gap_through'], 1000)
      self.assertEqual(shown['Run']['gain_through']['MHA'], 600)
      self.assertEqual(shown['Slow']['gap_through'], 600)
      self.assertEqual(meta['gain_ratios'][0]['series'][0]['ratio'], 7.25)
      self.assertIsNone(meta['gain_ratios'][0]['series'][1]['ratio'])
      self.assertTrue(meta['warnings'])
      again = plot.render([example()], directory)
      self.assertNotEqual(again['figure'], result['figure'])
      self.assertTrue(image.exists())

  def test_no_mature_data_does_not_publish_an_empty_or_stale_figure(self):
    with tempfile.TemporaryDirectory() as directory:
      with self.assertRaisesRegex(ValueError, 'No selected cumulative points'):
        plot.render([example()], directory, start=2000)
      self.assertEqual(list(Path(directory).iterdir()), [])

  def test_gain_only_preserves_native_ratio_and_its_own_horizon(self):
    import matplotlib.pyplot as plt
    captured = {}
    real_subplots = plt.subplots
    def capture(*args, **kwargs):
      fig, grid = real_subplots(*args, **kwargs)
      captured['grid'] = grid
      return fig, grid
    with tempfile.TemporaryDirectory() as directory, patch.object(plt, 'subplots', capture):
      result = plot.render([example()], directory, runs=['Run'], panels=('gain',), independent_x=True)
      meta = json.loads(Path(result['provenance']).read_text())
    grid = captured['grid']
    self.assertEqual(grid.shape, (1, 1))
    self.assertEqual(grid[0, 0].lines[0].get_ydata()[0], 7.25)
    self.assertTrue(math.isnan(grid[0, 0].lines[0].get_ydata()[1]))
    self.assertEqual(meta['panels'], ['gain'])
    self.assertEqual(meta['panel_end_steps'], [800])

  def test_comparator_columns_keep_curves_and_limits_matched(self):
    import matplotlib.pyplot as plt
    payload = example()
    other = {'reports': [{'run': 'OtherRun', 'base': 'OtherParent',
                         'series': [{'step': 600, 'gap': -.07}]}],
             'gain_ratios': [{'run': 'OtherRun', 'comparator': 'OtherParent',
                             'reference': 'MHA',
                             'series': [{'step': 600, 'ratio': 1.23}]}]}
    real_subplots = plt.subplots
    captured = {}
    def capture(*args, **kwargs):
      fig, grid = real_subplots(*args, **kwargs)
      captured['grid'] = grid
      return fig, grid
    with tempfile.TemporaryDirectory() as directory, patch.object(plt, 'subplots', capture):
      result = plot.render([payload, other], directory, facet_by_base=True,
                           bases=['OtherParent', 'Parent'])
      meta = json.loads(Path(result['provenance']).read_text())
    grid = captured['grid']
    self.assertEqual(grid.shape, (2, 2))
    self.assertEqual(grid[0, 0].get_legend_handles_labels()[1], ['OtherRun'])
    self.assertEqual(grid[0, 1].get_legend_handles_labels()[1], ['Run', 'Slow'])
    self.assertEqual(grid[0, 0].get_ylim(), grid[0, 1].get_ylim())
    self.assertEqual(grid[1, 0].get_ylim(), grid[1, 1].get_ylim())
    self.assertEqual(list(grid[1, 0].lines[0].get_ydata()), [1.23])
    self.assertEqual(len(meta['shown']), 3)
    self.assertEqual(meta['column_bases'], ['OtherParent', 'Parent'])


if __name__ == '__main__':
  unittest.main()
