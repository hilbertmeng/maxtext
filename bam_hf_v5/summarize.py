"""Median LoomFormer / MHA throughput ratios over the alternating runs of bench_pair.sh.

  python bam_hf_v5/summarize.py OUT_PREFIX
"""
import glob
import json
import statistics
import sys


def load(prefix, kind):
  runs = [json.load(open(f)) for f in sorted(glob.glob(f'{prefix}-{kind}-*.json'))]
  keys = [k for k in runs[0] if isinstance(runs[0][k], dict)]
  return {k: statistics.median(r[k]['tok_per_s'] for r in runs) for k in keys}, len(runs)


def main():
  prefix = sys.argv[1]
  loom, nl = load(prefix, 'loom')
  mha, nm = load(prefix, 'mha')
  out = {k: {'loom_tok_per_s': loom[k], 'mha_tok_per_s': mha[k], 'ratio': loom[k] / mha[k]} for k in loom if k in mha}
  print(json.dumps({'runs': [nl, nm], 'results': out}, indent=1))
  with open(prefix + '-summary.json', 'w') as f:
    json.dump({'runs': [nl, nm], 'results': out}, f, indent=1)


if __name__ == '__main__':
  main()
