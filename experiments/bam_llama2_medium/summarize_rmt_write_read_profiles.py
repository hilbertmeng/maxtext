"""Summarize chunk sweeps from local primary traces and orchestration text logs."""
import argparse,json,pathlib,re,statistics
p=argparse.ArgumentParser();p.add_argument('root',type=pathlib.Path);args=p.parse_args()
rows=[]
for analysis in sorted((args.root/'profiles').rglob('analysis.json')):
 a=json.loads(analysis.read_text())['arms'][0]
 run=next(x for x in analysis.parts if x.startswith('Profile'))
 name=run.split('_0_')[-1]; repeat='-repeat_' in run
 log=args.root/'logs'/('control-repeat-profile.log' if repeat else name+'-profile.log')
 text=log.read_text() if log.exists() else ''
 speeds=[float(x) for x in re.findall(r'completed step: \d+, steps/s: ([\d.]+)',text)]
 row=dict(config=name,repeat=repeat,run=run,device_ms=a['mean_device_step_ms'],
          steps_s=statistics.mean(speeds) if speeds else None,
          first_core_ms=a['first_core_step_ms'],leaf_ms=a['first_core_leaf_ms'],
          unattributed_ms=a.get('first_core_unattributed_ms',a['first_core_step_ms']-a['first_core_leaf_ms']),
          categories_ms=a['categories_ms'])
 rows.append(row)
bases=[r for r in rows if r['config']=='RMTWriteReadControlProfile']
base=next(r for r in bases if not r['repeat'])
for r in rows:
 r['device_throughput_change_pct']=100*(base['device_ms']/r['device_ms']-1)
 r['log_throughput_change_pct']=100*(r['steps_s']/base['steps_s']-1) if r['steps_s'] and base['steps_s'] else None
print('| Configuration | Device step ms | Log step/s | Device throughput vs control | Unattributed first-core ms |')
print('|---|---:|---:|---:|---:|')
for r in rows:
 speed=f"{r['steps_s']:.4f}" if r['steps_s'] else 'pending'
 print(f"| `{r['config']}`{' repeat' if r['repeat'] else ''} | {r['device_ms']:.2f} | {speed} | {r['device_throughput_change_pct']:+.2f}% | {r['unattributed_ms']:.2f} |")
(args.root/'summary.json').write_text(json.dumps(rows,indent=2))

# A compact scientific plot makes the chunk-size tradeoff explicit.
try:
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(8,4.5))
 extra=args.root/'unroll/summary.json'
 plot_rows=rows+(json.loads(extra.read_text()) if extra.exists() else [])
 for kind,marker in [('Split','o'),('Merged','s'),('Unrolled','^')]:
  points=[]
  for r in plot_rows:
   match=re.fullmatch(r'RMTWriteRead'+kind+r'C(\d+)Profile',r['config'])
   if match:points.append((int(match[1]),r['device_throughput_change_pct']))
  if points:
   points.sort();ax.plot(*zip(*points),marker=marker,label=kind)
 ax.axhline(0,color='black',lw=1,label='Original control')
 merged=next((r for r in rows if r['config']=='RMTWriteReadMergedProfile'),None)
 if merged:ax.axhline(merged['device_throughput_change_pct'],color='gray',ls='--',label='Merged, no chunk')
 ax.set_xscale('log',base=2);ax.set_xticks([64,128,256,512,1024,2048],labels=[64,128,256,512,1024,2048])
 ax.set(xlabel='Tokens per write/read chunk',ylabel='Full-step throughput change (%)',title='RMT 18-layer write/read scheduling, v5p-16')
 ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(args.root/'throughput.png',dpi=180);plt.close(fig)
except ImportError:
 pass
