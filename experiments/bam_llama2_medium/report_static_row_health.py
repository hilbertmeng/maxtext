"""Read-only matched-window TB analysis; reuses the incremental scalar cache."""
import argparse
import importlib.util
import json
import struct
from pathlib import Path
import numpy as np

RUNS = {
    'Norm': 'BamMediumIndependentLLFBAlignedRowLocalOStaticDynamicRow',
    'NoNorm': 'BamMediumIndependentLLFBAlignedRowLocalOStaticDynamicRowNoNorm',
    'OldRank4': 'BamMediumIndependentLLFBAlignedRowORowRank4CFp32',
}

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--tb-root', type=Path, default=Path('/data0/xd/tensorboard_logs'))
    parser.add_argument('--reader', type=Path, default=Path('/home/xd/projects/maxtext/.claude/skills/tpu-training/scripts/report_bam_read_health.py'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('health_reader', args.reader)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    sources = {label: reader.Scalars(args.tb_root / run, list(range(0, 13601, 200)))
               for label, run in RUNS.items()}
    latest = {label: max(s._points['learning/raw_grad_norm']) for label,s in sources.items()}
    common = (min(latest.values())-25)//200*200
    steps = [0] + list(range(200,common+1,200))
    last_new = max((latest[k]-25)//200*200 for k in ('Norm','NoNorm'))
    losses={}
    for label,run in RUNS.items():
        points={}
        for file in sorted((args.tb_root/run).glob('events.out.tfevents.*')):
            with file.open('rb') as f:
                while True:
                    header=f.read(12)
                    if len(header)!=12: break
                    size=struct.unpack('<Q',header[:8])[0]
                    data=f.read(size); footer=f.read(4)
                    if len(data)!=size or len(footer)!=4: break
                    event=reader.event_pb2.Event.FromString(data)
                    for value in event.summary.value:
                        if value.tag=='learning/loss':
                            points[event.step]=reader._scalar_value(value)
        losses[label]=points

    def mean(s, tag, step):
        data=s._points.get(tag,{})
        selected=([data[0]] if step==0 and 0 in data else
                  [v for t,v in data.items() if abs(t-step)<=25 and t%10==0])
        return float(np.mean(selected)) if selected else None

    def branch(s, prefix, step):
        names=('static_ms','dynamic_ms','cross','total_ms','gate_mean',
               'static_pre_gate_ms','dynamic_pre_gate_ms','a','a_over_a0')
        row={n:mean(s,f'{prefix}/{n}',step) for n in names}
        es,ed,cross,total=(row[n] for n in names[:4])
        if es is not None:
            denom=max(es+ed,1e-30)
            row.update(static_share=es/denom, interference=cross/denom,
                       pooled_cosine=cross/(2*np.sqrt(es*ed)) if es*ed>0 else None,
                       total_rms=float(np.sqrt(total)),
                       static_rms=float(np.sqrt(es)), dynamic_rms=float(np.sqrt(ed)),
                       closure_error=(total-es-ed-cross)/denom)
        return row

    report=dict(runs=RUNS, latest_tb_steps=latest, steps=steps,
                window='step0 exact; others +/-25, stride10, mean; latest event file wins',
                runtime={'Norm':'b600adf','NoNorm':'b600adf','OldRank4':'d437020'},data={})
    for label,s in sources.items():
        rows=[]
        own_steps=[0]+list(range(200,min(last_new,(latest[label]-25)//200*200)+1,200))
        for step in own_steps:
            grad=s._points['learning/raw_grad_norm']
            vals=[v for t,v in grad.items() if (t==0 if step==0 else abs(t-step)<=25) and t%10==0]
            row=dict(step=step, raw_grad=float(np.mean(vals)),
                     clip_fraction=float(np.mean(np.asarray(vals)>1)),
                     clip_scale=float(np.mean(np.minimum(1,1/np.maximum(vals,1e-30)))))
            if label!='OldRank4':
                common_loss_steps=sorted(t for t in losses[label] if t in losses['OldRank4']
                                        and (t==0 if step==0 else abs(t-step)<=25) and t%10==0)
                row['gap_vs_old_rank4']=float(np.mean([losses[label][t]-losses['OldRank4'][t] for t in common_loss_steps]))
                row['loss_window_steps']=common_loss_steps
            if label!='OldRank4':
                row['all_local']=branch(s,'bam/local_o_row_branches/all_local',step)
                row['layers']={str(l):branch(s,f'bam/local_o_row_branches/layer_{l:03d}',step)
                               for l in range(24) if l%3!=2}
            rows.append(row)
        report['data'][label]=rows
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('latest',latest,'common',common)
    for label,rows in report['data'].items():
        print(label)
        for row in rows:
            b=row.get('all_local',{})
            a=[v['a_over_a0'] for v in row.get('layers',{}).values() if v['a_over_a0'] is not None]
            print(row['step'], 'grad',round(row['raw_grad'],4),
                  'gap',row.get('gap_vs_old_rank4'),
                  'clip',row['clip_fraction'],
                  {k:round(b[k],5) if b.get(k) is not None else None
                   for k in ('static_share','interference','pooled_cosine','gate_mean','total_rms')},
                  'a/a0 mean/min/max', [round(f(a),4) for f in (np.mean,np.min,np.max)] if a else None)

if __name__=='__main__':
    main()
