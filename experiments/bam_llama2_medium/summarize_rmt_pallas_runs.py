#!/usr/bin/env python3
"""Summarize identical post-profile windows from collected worker train logs."""
import argparse
import json
from pathlib import Path
import re
import statistics


def summarize(path, start=20, stop=49):
  text=path.read_text()
  records={int(step):(float(speed),float(loss)) for step,speed,loss in
           re.findall(r'completed step: (\d+), steps/s: ([\d.]+).*?loss: ([\d.]+)',text)}
  if not all(step in records for step in range(start,stop+1)):
    return None
  name=re.search(r'_(RMT\w+)\.log$',path.name)
  commit=re.search(r'Profile([0-9a-f]{7})_',path.name)
  values=[records[step] for step in range(start,stop+1)]
  keys=('base_num_decoder_layers','rmt_mlp_dim_by_block','per_device_batch_size',
        'global_batch_size_to_train_on','max_target_length','learning_rate_schedule_steps',
        'opt_type','dtype','rmt_record_dynamic_health','record_training_health_metrics',
        'record_internal_nn_metrics','rmt_block_scan','rmt_remat_policy','attention',
        'query_chunk_size','scan_layers','base_emb_dim','base_mlp_dim','head_dim')
  resolved={key:re.findall(r'^Config param '+key+r': (.+)$',text,re.M)[-1]
            for key in keys if re.search(r'^Config param '+key+r': ',text,re.M)}
  return dict(configuration=name[1] if name else path.stem,
              runtime=commit[1] if commit else None,log=str(path),
              first_step=start,last_step=stop,count=len(values),
              steps_per_second=1/statistics.mean(1/v[0] for v in values),
              min_step_s=min(v[0] for v in values),max_step_s=max(v[0] for v in values),
              loss_at_last_step=values[-1][1],
              loaded_aot='Loaded compiled function!' in text,resolved=resolved)


def main():
  p=argparse.ArgumentParser(description=__doc__)
  p.add_argument('roots',nargs='+',type=Path)
  p.add_argument('--output',type=Path,required=True)
  args=p.parse_args()
  rows=[]
  for path in sorted({x for root in args.roots for x in root.rglob('train_Profile*.log')}):
    if (row:=summarize(path)) is not None:rows.append(row)
  args.output.parent.mkdir(parents=True,exist_ok=True)
  args.output.write_text(json.dumps(rows,indent=2)+'\n')
  for row in rows:
    print(f"{row['runtime']} {row['configuration']} {row['steps_per_second']:.6f} AOT={row['loaded_aot']}")


if __name__=='__main__':main()
