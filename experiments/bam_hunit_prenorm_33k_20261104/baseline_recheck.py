"""Verify all32 retained baseline cases; never invent a missing new-model result."""
from pathlib import Path
import json
R=Path(__file__).resolve().parent
source=(R/'collect_analyze.py').read_text().split("assert manifests['baseline']['cohort_hashes']")[0]
source=source.replace("for arm in ['baseline','prenorm']:","for arm in ['baseline']:")
ctx={'__file__':str(R/'collect_analyze.py'),'__name__':'baseline_only_recheck'}
exec(compile(source,str(R/'collect_analyze.py'),'exec'),ctx)
np=ctx['np'];v=ctx['data']['baseline'];m=ctx['manifests']['baseline'];legacy=R.parent/'bam_hunit_early_weak_v5p_20261103/arms.json';oldroot=Path(json.loads(legacy.read_text())['local_DATA'])/'33000';checks=[]
for i in range(32):
 q=json.loads((Path(m['local_DATA'])/'33000'/f'sequence_{i:03d}.json').read_text());assert q['device_kind']=='TPU v5'
 with np.load(oldroot/f'sequence_{i:03d}.npz') as z:checks.append({'CE_delta':float(v['baseline_CE'][i].mean()-z['baseline_CE'].mean()),'weak_power_relative':float(v['own_head_total_power'][i]/z['own_head_total_power']-1)})
assert max(abs(x['CE_delta']) for x in checks)<.005
assert max(abs(x['weak_power_relative']) for x in checks)<.015
out={'step':33000,'baseline32_complete':True,'prenorm33k_not_measured':True,'row':ctx['rows']['baseline'],'verification':ctx['verification']['baseline'],'legacy_max_CE_difference':max(abs(x['CE_delta']) for x in checks),'legacy_max_weak_power_relative':max(abs(x['weak_power_relative']) for x in checks)}
(R/'baseline-only.json').write_text(json.dumps(out,indent=2)+'\n')
print('BASELINE33K_VERIFIED',json.dumps(out['row']['own']['head_weak_fraction']),out['legacy_max_weak_power_relative'])
