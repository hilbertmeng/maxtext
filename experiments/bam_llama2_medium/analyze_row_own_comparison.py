"""Pair all-origin cross/self/whole consumer screens, including graph controls."""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze_row_mediation import stats
from analyze_row_token_worlds import analyze


def load(root):
  validated = analyze(root)
  values = []
  for path in sorted(Path(root).glob('batch_*.npz')):
    with np.load(path, allow_pickle=False) as batch:
      values.append(batch['loss'].astype(np.float64))
  loss = np.concatenate(values)
  return validated, dict(zip(validated['metadata']['arms'], loss.T))


def compare(root, prefix, terminal_prefix=None):
  loaded = {}
  reference_meta = None
  clean_reference = None
  controls = []
  for component, suffix in [('cross', ''), ('self', '-rowself'), ('both', '-rowboth')]:
    merged = {}
    for group in ('joint', 'individual'):
      path = Path(root) / (prefix.format(group=group) + suffix)
      checked, values = load(path)
      meta = checked['metadata']
      assert meta['source_component'] == component
      if reference_meta is None:
        reference_meta = meta
        clean_reference = values['clean']
      for key in ('base_config_class', 'checkpoint', 'trainer_commit',
                  'diagnostic_commit', 'cohort_sha256', 'source_layer', 'requested_sequences'):
        assert meta[key] == reference_meta[key], key
      np.testing.assert_array_equal(values['clean'], clean_reference)
      for name in merged.keys() & values.keys():
        np.testing.assert_array_equal(merged[name], values[name])
      merged.update(values)
      controls.append(dict(root=str(path), n=checked['n'], exact_controls=True))
    np.testing.assert_array_equal(merged['cut_L11_attention'], merged['own_origin_only_deleted'])
    if terminal_prefix:
      path = Path(root) / (terminal_prefix + suffix)
      checked, values = load(path)
      meta = checked['metadata']
      assert meta['source_component'] == component
      for key in ('base_config_class', 'checkpoint', 'trainer_commit',
                  'cohort_sha256', 'source_layer', 'requested_sequences'):
        assert meta[key] == reference_meta[key], key
      shared = merged.keys() & values.keys()
      assert {'clean', 'cut_L11_attention', 'cut_L22_mlp'} <= shared
      for name in shared:
        np.testing.assert_array_equal(merged[name], values[name])
      merged.update(values)
      controls.append(dict(root=str(path), n=checked['n'], exact_controls=True,
                           diagnostic_commit=meta['diagnostic_commit'],
                           exact_shared_arms=sorted(shared)))
    loaded[component] = merged
  names = list(loaded['cross'])
  for component in ('self', 'both'):
    assert set(loaded[component]) == set(names)
  rows = []
  for name in names:
    if name == 'clean':
      continue
    deltas = {c: x[name] - x['clean'] for c, x in loaded.items()}
    rows.append(dict(arm=name, **{c: stats(d) for c, d in deltas.items()},
        joint_minus_individual=stats(deltas['both']-deltas['cross']-deltas['self'])))
  return dict(metadata={k: reference_meta[k] for k in (
      'base_config_class', 'checkpoint', 'trainer_commit', 'diagnostic_commit',
      'cohort_sha256', 'source_layer', 'requested_sequences')},
      controls=controls, exact_shared_arms=True, immediate_cut_matches_deletion=True,
      rows=rows, interpretation='All effects are paired original-position loss changes; '
      'joint-minus-individual is nonadditivity, not an additive exported-benefit share.')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('root')
  parser.add_argument('--prefix', default='bam-row-mediation-xl-L11-own_consumers-{group}-968f84c')
  parser.add_argument('--output', required=True)
  parser.add_argument('--terminal-prefix')
  args = parser.parse_args()
  result = compare(args.root, args.prefix, args.terminal_prefix)
  Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
  print(f"PAIRED_OK: all {len(result['controls'])} complete cohorts, controls, shared arms and immediate cuts exact")
  for row in result['rows']:
    print(row['arm'], ' '.join(f"{c}={row[c]['mean']:+.7f}±{row[c]['ci95']:.7f}"
                               for c in ('cross','self','both')))
