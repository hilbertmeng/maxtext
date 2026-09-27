#!/usr/bin/env python3
"""Attribute matched RMT train-step traces without counting scan containers twice."""
import argparse
import collections
import gzip
import json
import re
from pathlib import Path
import statistics
import importlib.util
import os


def read_xplane_events(path):
  # Load the protobuf definitions without importing the TensorFlow runtime.
  proto = Path(os.environ.get('XPLANE_PROTO',
      '/data0/xd/conda/envs/maxtext-cpu/lib/python3.12/site-packages/'
      'tensorflow/tsl/profiler/protobuf/xplane_pb2.py'))
  spec = importlib.util.spec_from_file_location('rmt_xplane_pb2', proto)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  space = module.XSpace()
  space.ParseFromString(path.read_bytes())
  if space.errors:
    raise ValueError(f'XPlane errors: {list(space.errors)}')
  planes = [p for p in space.planes if p.name.startswith('/device:TPU:')]
  base_ns = min(line.timestamp_ns for p in planes for line in p.lines)
  events = []
  for pid, plane in enumerate(planes):
    events.append(dict(ph='M', name='process_name', pid=pid, args=dict(name=plane.name)))
    def decode(stats):
      result = {}
      for stat in stats:
        kind = stat.WhichOneof('value')
        value = getattr(stat, kind) if kind else None
        if kind == 'ref_value':
          value = plane.stat_metadata[value].name
        if kind != 'bytes_value':
          result[plane.stat_metadata[stat.metadata_id].name] = value
      return result
    # Preserve full enclosing steps on every core, leaves on first core only.
    for line in plane.lines:
      if line.name != 'XLA Modules':
        continue
      for event in line.events:
        md = plane.event_metadata[event.metadata_id]
        name = md.display_name or md.name
        if name.startswith('jit_train_step('):
          events.append(dict(ph='X', pid=pid, name=name,
              ts=((line.timestamp_ns-base_ns)*1000+event.offset_ps)/1e6,
              dur=event.duration_ps/1e6, args={}))
    if pid != 0:
      continue
    first = min((e for e in events if e['pid']==0 and e['ph']=='X'), key=lambda e:e['ts'])
    metadata = {}
    for line in plane.lines:
      if line.name != 'XLA Ops':
        continue
      for event in line.events:
        ts=((line.timestamp_ns-base_ns)*1000+event.offset_ps)/1e6
        dur=event.duration_ps/1e6
        if ts < first['ts'] or ts+dur > first['ts']+first['dur']+1:
          continue
        md=plane.event_metadata[event.metadata_id]
        if event.metadata_id not in metadata:
          args=decode(md.stats); args['long_name']=md.name
          metadata[event.metadata_id]=args
        events.append(dict(ph='X',pid=pid,name=md.display_name or md.name,
                           ts=ts,dur=dur,args=metadata[event.metadata_id]))
  return events


def read_trace(path):
  pb = path if path.suffix=='.pb' else path.with_name(path.name.replace('.trace.json.gz','.xplane.pb'))
  if pb.exists():
    events = read_xplane_events(pb)
    source_format = 'raw_xplane'
  else:
    with gzip.open(path, 'rt') as stream:
      events = json.load(stream)['traceEvents']
    source_format = 'trace_json'
    if len(events) >= 1000000:
      raise ValueError(f'Trace JSON may be truncated; supply raw XPlane: {path}')
  device_pids = [e['pid'] for e in events
                 if e.get('name') == 'process_name'
                 and str(e.get('args', {}).get('name', '')).startswith('/device:TPU:')]
  spans = [e for e in events if e.get('pid') in device_pids
           and e.get('ph') == 'X' and e.get('name', '').startswith('jit_train_step(')]
  if not spans:
    raise ValueError(f'No device train step: {path}')
  # Different TPU execution cores execute different subsets of the kernels.
  # Average enclosing wall times, but attribute leaves on one complete core only.
  step = min((e for e in spans if e['pid'] == device_pids[0]), key=lambda e: e['ts'])
  categories = collections.defaultdict(float)
  sources = collections.defaultdict(float)
  ops = collections.defaultdict(float)
  examples = {}
  copies = {}
  leaf_intervals = []
  for event in events:
    if event.get('pid') != step['pid'] or event.get('ph') != 'X':
      continue
    name, args = event.get('name', ''), event.get('args', {})
    if (name.isdigit() or name.startswith(('jit_train_step(', 'while.'))
        or args.get('hlo_category') == 'while'):
      continue
    if not (step['ts'] <= event['ts']
            and event['ts'] + event['dur'] <= step['ts'] + step['dur'] + 1):
      continue
    ms = event['dur'] / 1000
    leaf_intervals.append((event['ts'], event['ts'] + event['dur']))
    category = args.get('hlo_category', 'unclassified')
    source = args.get('source', '').split('/MaxText/')[-1]
    op = args.get('tf_op') or f'UNSCOPED/{source}/{category}'
    categories[category] += ms
    sources[source] += ms
    ops[op] += ms
    if category == 'data formatting' and name.startswith('copy.'):
      shape = args.get('shape_with_layout', '')
      match = re.search(r'copy\(([^%]+) %', args.get('long_name', ''))
      input_shape = match.group(1).strip() if match else 'unknown'
      copy_key = f'{input_shape} -> {shape}'
      item = copies.setdefault(copy_key, dict(ms=0., count=0, kernels={},
                                             input=input_shape, output=shape))
      item['ms'] += ms
      item['count'] += 1
      kernel = item['kernels'].setdefault(name, dict(ms=0., count=0, op=op))
      kernel['ms'] += ms
      kernel['count'] += 1
    examples.setdefault(op, dict(name=name, shape=args.get('shape_with_layout'),
                                source=source, category=category))
  wall_ms = step['dur'] / 1000
  leaf_ms = sum(categories.values())
  # Chunk loops can contain time without a leaf event. Preserve this gap instead
  # of silently assigning it to compute or rejecting the enclosing step time.
  covered_us = 0.
  end = step['ts']
  for lo, hi in sorted(leaf_intervals):
    covered_us += max(0., hi - max(lo, end))
    end = max(end, hi)
  overlap_ms = leaf_ms - covered_us / 1000
  if overlap_ms > max(2., wall_ms * .002):
    raise ValueError(f'Overlapping leaves: {overlap_ms} ms: {path}')
  return dict(trace=str(path), source_format=source_format, device_step_ms=[e['dur']/1000 for e in spans],
              mean_device_step_ms=statistics.mean(e['dur']/1000 for e in spans),
              first_core_step_ms=wall_ms, first_core_leaf_ms=leaf_ms,
              first_core_unattributed_ms=wall_ms-covered_us/1000,
              first_core_leaf_overlap_ms=overlap_ms,
              categories_ms=dict(categories), sources_ms=dict(sources),
              ops_ms=dict(ops), examples=examples, copies_by_layout=copies)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('traces', nargs='+', type=Path, help='Baseline first, then comparison arms')
  parser.add_argument('--output', required=True, type=Path)
  args = parser.parse_args()
  arms = [read_trace(path) for path in args.traces]
  base = arms[0]
  comparisons = []
  for arm in arms[1:]:
    deltas = {}
    for field in ('categories_ms', 'sources_ms', 'ops_ms'):
      a, b = base[field], arm[field]
      deltas[field] = dict(sorted(((k, b.get(k, 0.) - a.get(k, 0.))
                                  for k in a.keys() | b.keys()), key=lambda pair: -abs(pair[1])))
    comparisons.append(dict(trace=arm['trace'],
                            wall_delta_ms=arm['mean_device_step_ms']-base['mean_device_step_ms'],
                            first_core_leaf_delta_ms=arm['first_core_leaf_ms']-base['first_core_leaf_ms'],
                            deltas=deltas))
    print(f"{arm['mean_device_step_ms']:.3f} ms; delta {comparisons[-1]['wall_delta_ms']:+.3f} ms")
    for category, delta in deltas['categories_ms'].items():
      if abs(delta) > .1:
        print(f'  {category}: {delta:+.3f} ms')
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(dict(arms=arms, comparisons=comparisons), indent=2))


if __name__ == '__main__':
  main()
