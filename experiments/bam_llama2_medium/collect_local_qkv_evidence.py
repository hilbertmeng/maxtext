"""Read-only paired loss export; run on tpu-ag with existing registry caches.

ssh -S /tmp/ssh-tpu-ag-xd.sock tpu-ag python3 - < this_file.py
No refresh, registry mutation, TPU action, or baseline substitution.
"""
import datetime
import json
import sys
sys.path.insert(0, '/home/lishengping/xd/projects')
import run_registry as registry

P = 'BamMediumIndependentLLF'
X = 'BamXLIndependentLLF'
legacy = P + 'RoutingLegacy'
r4 = legacy + 'LocalVRank4'
b = P + 'LocalVRank4RoutingB'
aligned = b + 'AlignedRow'
historical = 'BamLlama2MediumV2C256LocalFetchC8LocalVLLFScan'
pairs = [
    (r4, legacy, 200, 2800),
    (b, r4, 200, 2800),
    (b, legacy, 200, 10400),
    (aligned, b, 200, 13400),
    (aligned, historical, 200, 13400),
    (aligned, 'BamLlama2MediumV2C256ScanAotCleanControl', 200, 13400),
    (aligned, 'BamLlama2MediumV2', 200, 13400),
    (P + 'LocalVRank4RoutingCFp32', b, 200, 13400),
    (P + 'LocalVRank4RoutingA', b, 200, 5600),
    (P + 'AlignedRowLocalVRowRank2', aligned, 200, 13400),
    (P + 'LocalVRank2RoutingBAlignedRow', aligned, 200, 3400),
    (P + 'LocalVRank2RoutingBAlignedRow', legacy, 200, 3400),
    (P + 'LocalVRank4RoutingCFp32NoBias', P + 'LocalVRank4RoutingCFp32', 200, 5000),
    (X + 'LocalVRank4CFp32AlignedRow',
     'BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2LocalFetchC8LocalVLLF', 500, 21000),
    (X + 'LocalQKVCFp32AlignedRow', X + 'LocalVRank4CFp32AlignedRow', 500, 29500),
    (X + 'LocalQKRank4CFp32AlignedRow', X + 'LocalQKVCFp32AlignedRow', 500, 22000),
]
output = {'generated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'method': 'RUN-BASE; exact-common raw step multiples of 10; +/-25 window; cached records only',
          'pairs': []}
for run, base, interval, end in pairs:
    m = dict(registry.load_run(run))
    m.update(compare_runs=[base], loss_interval=interval)
    try:
        reports = registry.build_loss_reports(m, end, 10, False)
        for report in reports:
            series = [{'step': p['step'], 'gap': p['gap'], 'samples': p['samples']}
                      for p in report['series']]
            output['pairs'].append(dict(run=run, base=base, interval=interval,
                                        requested_end=end, series=series))
    except Exception as exc:
        output['pairs'].append(dict(run=run, base=base, error=str(exc)))
print(json.dumps(output, indent=2))
