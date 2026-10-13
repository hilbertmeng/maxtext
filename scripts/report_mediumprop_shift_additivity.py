#!/usr/bin/env python3
"""Experiment-family-only M-shift plus KV-shift additivity report.

Refresh loss caches with the unchanged run_registry loss-report before invoking.
A measures total additive gain; C measures weaker-arm retention. No registry edits.
"""
import argparse
import importlib.util
import json
from pathlib import Path

JOINT = 'BamMediumPropSharedR256GeneralReadMShiftReadKVShiftG050TruePile'
SINGLE = 'BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile'
KV = 'BamMediumPropSharedR256GeneralReadKVShiftG050TruePile'
SOTA = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile'

def additivity_series(joint, arm_a, arm_b, baseline, interval, radius,
                      sample_period, through_step=None):
    """Fraction of the weaker arm's gain added beyond the stronger arm.

    Compute from four identical complete window means, never from point ratios.
    C needs both individual gains positive; A needs their sum positive.
    For C, 0=best arm, 1=fully additive.
    """
    caches = (joint, arm_a, arm_b, baseline)
    if any(not cache for cache in caches):
        return []
    mature = max(0, min(map(max, caches)) - radius)
    if through_step is not None:
        mature = min(mature, through_step)
    series = []
    for center in range(interval, mature + 1, interval):
        points = [s for s in range(center-radius, center+radius+1) if s % sample_period == 0]
        if not points or not all(s in cache for cache in caches for s in points):
            continue
        means = [sum(cache[s] for s in points)/len(points) for cache in caches]
        joint_gain, a_gain, b_gain = (means[3] - value for value in means[:3])
        weaker, stronger = min(a_gain, b_gain), max(a_gain, b_gain)
        total_gain = a_gain + b_gain
        total_status = ('nonpositive_total_gain' if total_gain < -1e-8 else
                        'near_zero_total_gain' if total_gain <= 1e-8 else 'ok')
        status = ('nonpositive_individual_gain' if weaker < -1e-8 else
                  'near_zero_individual_gain' if weaker <= 1e-8 else 'ok')
        series.append(dict(step=center, retention=(joint_gain-stronger)/weaker if status == 'ok' else None,
                           total_fraction=joint_gain/total_gain if total_status == 'ok' else None,
                           total_status=total_status,
                           joint_gain=joint_gain, arm_a_gain=a_gain, arm_b_gain=b_gain,
                           weaker_gain=weaker, stronger_gain=stronger,
                           samples=len(points), status=status))
    return series

def print_additivity_series(report, columns):
    print(f"ADDITIVITY RUN={report['run']} ARM_A={report['arm_a']} ARM_B={report['arm_b']} "
          f"BASELINE={report['baseline']} A=G_joint/(G_a+G_b) C=(G_joint-max(G_a,G_b))/min(G_a,G_b)")
    series = report['series']
    if not series:
        print('PENDING_ADDITIVITY: no complete four-way window')
    for start in range(0, len(series), columns):
        block = series[start:start+columns]
        print('     step: ' + '  '.join(f"{p['step']:>10}" for p in block))
        print('  total_A: ' + '  '.join(f"{100*p['total_fraction']:9.2f}%" if p['total_fraction'] is not None
                                     else f"{'undefined':>10}" for p in block))
        print('retention: ' + '  '.join(f"{100*p['retention']:9.2f}%" if p['retention'] is not None
                                     else f"{'undefined':>10}" for p in block))
        for p in block:
            if p['total_status'] != 'ok':
                print(f"ADDITIVITY_A_UNDEFINED step={p['step']} reason={p['total_status']}")
            if p['status'] != 'ok':
                print(f"ADDITIVITY_C_UNDEFINED step={p['step']} reason={p['status']} weaker_gain={p['weaker_gain']:.9g}")
    recent = series[-5:]
    totals = [p['total_fraction'] for p in recent if p['total_fraction'] is not None]
    if totals:
        print(f'additivity_trend_A: last{len(recent)} valid={len(totals)}/{len(recent)} '
              f'mean={100*sum(totals)/len(totals):.2f}% range={100*min(totals):.2f}%..{100*max(totals):.2f}% '
              '(mean of window fractions)')
    valid = [p['retention'] for p in recent if p['retention'] is not None]
    if valid:
        print(f'additivity_trend: last{len(recent)} valid={len(valid)}/{len(recent)} '
              f'mean={100*sum(valid)/len(valid):.2f}% range={100*min(valid):.2f}%..{100*max(valid):.2f}% '
              '(mean of window fractions)')
    print()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--through-step', type=int)
    parser.add_argument('--columns', type=int, default=20)
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--registry-script', type=Path,
                        default=Path('/home/lishengping/xd/projects/run_registry.py'))
    args = parser.parse_args()
    if args.columns <= 0:
        parser.error('columns must be positive')
    spec = importlib.util.spec_from_file_location('registry_loss_cache', args.registry_script)
    registry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(registry)
    names = (JOINT, SINGLE, KV, SOTA)
    meta = registry.load_run(JOINT)
    report = dict(run=JOINT, arm_a=SINGLE, arm_b=KV, baseline=SOTA,
                  series=additivity_series(*(registry.load_loss_points(name) for name in names),
                    int(meta.get('loss_interval', 200)), int(meta.get('loss_window', 25)),
                    10, args.through_step))
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_additivity_series(report, args.columns)


if __name__ == '__main__':
    main()
