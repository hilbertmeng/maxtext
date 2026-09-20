"""Report concat health using the training skill's incremental TensorBoard cache."""
import argparse
import importlib.util
import json
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('run')
parser.add_argument('--steps', required=True)
parser.add_argument('--reader', default='/home/xd/projects/maxtext/.claude/skills/tpu-training/scripts/report_bam_read_health.py')
parser.add_argument('--tb-root', default='/data0/xd/tensorboard_logs')
args=parser.parse_args()
spec=importlib.util.spec_from_file_location('bam_health_reader',args.reader)
reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
steps=[int(x) for x in args.steps.split(',')]
scalars=reader.Scalars(Path(args.tb_root)/args.run, steps)
metrics={}
for arm in ('local_q','local_k','local_v','local_o','fetched_o'):
  for stat in ('mean','std','frac_lt_005','frac_gt_050','frac_gt_095'):
    metrics[arm+'/gate_'+stat]=f'bam/concat/{arm}_gate/layer_{{layer:03d}}/{stat}'
  metrics[arm+'/bam_over_standard']=f'bam/concat/{arm}_amplitude/layer_{{layer:03d}}/bam_over_standard'
metrics['qk_scores/bam_over_standard']='bam/concat/qk_scores/layer_{layer:03d}/bam_over_standard'
for arm in ('static_q', 'static_k', 'static_v', 'static_o'):
  metrics[arm+'/over_dynamic']=f'bam/concat/{arm}_amplitude/layer_{{layer:03d}}/bam_over_standard'
for stat in ('mean','std','frac_lt_005','frac_gt_050','frac_gt_095'):
  metrics['write_mix/gate_'+stat]=f'bam/concat/write_mix_gate/layer_{{layer:03d}}/{stat}'
for stat in ('mean_abs_diff', 'rms_diff', 'correlation'):
  metrics['vo_gate_pair/'+stat]=f'bam/concat/vo_gate_pair/layer_{{layer:03d}}/{stat}'
bands={'L0':range(1),'L1-7':range(1,8),'L8-15':range(8,16),'L16-23':range(16,24),'L24':range(24,25)}
result={}
for name,template in metrics.items():
  values={band:[scalars.band_mean(template,step,layers) for step in steps]
          for band,layers in bands.items()}
  if any(v is not None for series in values.values() for v in series):
    result[name]=values
print(json.dumps(dict(run=args.run,steps=steps,band_means=result),indent=2))
