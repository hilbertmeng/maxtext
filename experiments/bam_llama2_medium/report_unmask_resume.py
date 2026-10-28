#!/usr/bin/env python3
"""Paired intervention trajectory, including the first pre-update training loss.

All gap/window arithmetic is delegated to run_registry.loss_gap_series. Run names,
checkpoint provenance and review intent come from the task's state.json.
"""
import argparse,datetime,hashlib,json,math,shlex,subprocess
from pathlib import Path


def collect(state, through=None):
  names=state['runs']
  code='''import sys,json,concurrent.futures
sys.path.insert(0,"/home/lishengping/xd/projects")
import run_registry as reg
names=%r;through=%r
registered={n:reg.load_run(n) for n in names}
live=[n for n in names if registered[n].get('declared_status','running') == 'running']
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 list(pool.map(lambda n:reg.collect_live_loss(registered[n]),live))
caches={n:reg.load_loss_points(n) for n in names}
if any(not p for p in caches.values()):
 print(json.dumps({'ready':False,'horizons':{n:max(p) if p else None for n,p in caches.items()}}));raise SystemExit(0)
firsts={n:min(p) for n,p in caches.items()};assert len(set(firsts.values()))==1,firsts
start=next(iter(firsts.values()));assert 21557<=start<=21558,start
points={n:{step-start:v for step,v in p.items()} for n,p in caches.items()}
common=sorted(set(points[names[0]])&set(points[names[1]]))
if through is not None:common=[s for s in common if s<=through]
if not common:print(json.dumps({'ready':False}));raise SystemExit(0)
horizon=common[-1]
# Shift by one only because the standard positive-center iterator starts at1.
# This retains the actual first intervention sample at plotted update0.
shifted=[{s+1:v for s,v in points[n].items()} for n in names]
raw=reg.loss_gap_series(shifted[1],shifted[0],1,0,1,horizon+1)
for row in raw:row['step']-=1
windows=reg.loss_gap_series(points[names[1]],points[names[0]],20,5,1,horizon)
print(json.dumps({'ready':True,'runs':names,'absolute_start':start,'through_update':horizon,
 'raw_pair':raw,'windows':windows,'window':{'interval':20,'radius':5,'sample_period':1},
 'cache_horizons':{n:max(p) for n,p in caches.items()},'generated_utc':reg.now_utc()},indent=2))
'''%(names,through)
  result=subprocess.run(['ssh','-S','/tmp/ssh-tpu-ag-xd.sock','tpu-ag','python3 -'],input=code,
      text=True,capture_output=True,timeout=240)
  if result.returncode:raise RuntimeError(result.stdout+result.stderr)
  return json.loads(result.stdout)


def render(data, directory, source, checkpoint):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  from matplotlib.font_manager import FontProperties
  plt.rcParams.update({'font.family':FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').get_name(),'axes.unicode_minus':False})
  raw=data['raw_pair'];means=data['windows'];fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True)
  styles=[('baseline_mean','保留预掩码','#2479b5'),('experiment_mean','直接解封','#cd4646')]
  for key,label,color in styles:
    axes[0].plot([r['step'] for r in raw],[r[key] for r in raw],color=color,alpha=.4,lw=.9,label=label)
    axes[0].plot([r['step'] for r in means],[r[key] for r in means],color=color,lw=2.0,marker='o',markersize=3)
  axes[1].plot([r['step'] for r in raw],[r['gap'] for r in raw],color='#888888',alpha=.35,lw=.8,label='逐步配对差')
  axes[1].plot([r['step'] for r in means],[r['gap'] for r in means],color='#111111',lw=2,marker='o',markersize=3,label='20步中心、±5步共同样本均值')
  axes[1].scatter([0],[raw[0]['gap']],color='#cd4646',s=30,zorder=4,label='首个 batch：更新前的干预')
  axes[1].axhline(0,color='#777777',ls=':',lw=1)
  if len(raw)>=15:
    zoom_start=max(10,(raw[-1]['step']//100-2)*100)
    tail=[r for r in raw if r['step']>=zoom_start]
    tail_means=[r for r in means if r['step']>=zoom_start]
    zoom=axes[1].inset_axes([.38,.35,.60,.57])
    zoom.plot([r['step'] for r in tail],[r['gap'] for r in tail],color='#888888',alpha=.5,lw=.9)
    zoom.plot([r['step'] for r in tail_means],[r['gap'] for r in tail_means],color='#111111',lw=1.5,marker='o',markersize=2)
    zoom.axhline(0,color='#777777',ls=':',lw=.8);zoom.grid(alpha=.15)
    zoom.set_title(f'更新{zoom_start}以后放大',fontsize=8);zoom.tick_params(labelsize=7)
  axes[0].set_ylabel('同批训练 loss');axes[1].set_ylabel('解封 − 保留掩码（负值更好）');axes[1].set_xlabel('恢复后的更新数（0为首个更新前的loss）')
  for axis in axes:axis.grid(alpha=.2);axis.legend(frameon=False,fontsize=9)
  axes[1].get_legend().remove()
  fig.suptitle('组合 checkpoint：直接解除预掩码 vs 保留预掩码')
  fig.text(.04,.015,'同源完整参数／Adam状态／Pile游标；global1024、T2048、原28k LR轨迹。\n淡线保留全部逐步loss，粗线为原生配对窗口均值；本实验保留初始瞬态。',fontsize=9,color='#555555')
  fig.tight_layout(rect=(0,.065,1,.96));directory.mkdir(parents=True,exist_ok=True)
  name='unmask-resume-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d-%H%M%S-%f');path=directory/(name+'.png');fig.savefig(path,dpi=160,bbox_inches='tight');plt.close(fig)
  path.with_suffix('.json').write_text(json.dumps({'figure':str(path),'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'checkpoint':checkpoint,'runs':data['runs'],'absolute_start':data['absolute_start'],'through_update':data['through_update'],'renderer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
  return str(path)


def main():
  p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task-dir',type=Path,required=True);p.add_argument('--through-update',type=int);p.add_argument('--label',default='latest');a=p.parse_args();s=json.loads((a.task_dir/'state.json').read_text());data=collect(s,a.through_update);out=a.task_dir/'reports';out.mkdir(exist_ok=True);source=out/('paired-'+a.label+'.json');source.write_text(json.dumps(data,indent=2)+'\n')
  if not data['ready']:print(json.dumps(data));return
  figure=render(data,a.task_dir/'figures',source,s['source_checkpoint']);r=data['raw_pair'];w=data['windows'];summary={'ready':True,'absolute_start':data['absolute_start'],'through_update':data['through_update'],'first_pair':r[0],'latest_pair':r[-1],'last_windows':w[-5:],'figure':figure,'source':str(source)};(out/('summary-'+a.label+'.json')).write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
