"""Read the nine compact training tags; no event synchronization or loss math."""
import argparse
import json
from pathlib import Path

from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
from tensorboard.util import tensor_util


def main():
  ap=argparse.ArgumentParser(description=__doc__)
  ap.add_argument('runs',nargs='+')
  ap.add_argument('--root',type=Path,default=Path('/data0/xd/tensorboard_logs'))
  ap.add_argument('--through-step',type=int)
  ap.add_argument('--output',type=Path)
  args=ap.parse_args();reports=[]
  for run in args.runs:
    points={};tags=set()
    for file in sorted((args.root/run).glob('**/events.out.tfevents.*')):
      for event in EventFileLoader(str(file)).Load():
        if args.through_step is not None and event.step>args.through_step:
          continue
        for value in event.summary.value:
          if not value.tag.startswith('embedding_stem/'):
            continue
          tags.add(value.tag)
          scalar=float(value.simple_value if value.HasField('simple_value') else tensor_util.make_ndarray(value.tensor))
          points.setdefault(event.step,{})[value.tag.removeprefix('embedding_stem/')]=scalar
    reports.append(dict(run=run,tag_count=len(tags),tags=sorted(tags),
                        rows=[dict(step=k,**points[k]) for k in sorted(points)]))
  result=json.dumps(reports,indent=2)+'\n'
  if args.output:args.output.write_text(result)
  else:print(result,end='')


if __name__=='__main__':main()
