"""Account for logical and compiler-padded Pallas ABI buffers.

Input is Mosaic's post-infer-memref-layout dump. This is an ABI residency
estimate, NOT the allocator's total: SSA temporaries, register spills, internal
scratch and allocator fragmentation must be compared separately. Invariant
weights/shared gradients need one resident copy; streamed blocks are reported
with both one and two copies so unspecified compiler pipelining is not hidden.
"""
import argparse
import json
import math
import re
from pathlib import Path


def audit(path):
  text = Path(path).read_text()
  signature = next(line for line in text.splitlines() if 'func.func @main(' in line)
  physical = re.findall(r'%arg\d+: memref<([0-9x]+)(bf16|f32|i32),', signature)
  windows = re.findall(r'\{([^{}]*window_bounds = array<i64: [^>]+>[^{}]*)\}', signature)
  if len(physical) != len(windows):
    raise ValueError(f'Expected one window per ABI buffer: {len(physical)} != {len(windows)}')
  buffers = []
  for i, ((shape, dtype), window) in enumerate(zip(physical, windows)):
    logical = tuple(map(int, re.search(r'window_bounds = array<i64: ([^>]+)>', window)[1].split(',')))
    padded = tuple(map(int, shape.rstrip('x').split('x')))
    size = 2 if dtype == 'bf16' else 4
    fixed = 'synchronous' in window
    buffers.append(dict(index=i, logical_shape=logical, physical_shape=padded,
                        dtype=dtype, logical_bytes=math.prod(logical)*size,
                        physical_bytes=math.prod(padded)*size,
                        explicit_single_buffer=fixed))
  one = sum(b['physical_bytes'] for b in buffers)
  two = sum(b['physical_bytes']*(1 if b['explicit_single_buffer'] else 2) for b in buffers)
  return dict(source=str(path), buffers=buffers,
              logical_one_copy_bytes=sum(b['logical_bytes'] for b in buffers),
              padded_one_copy_bytes=one, padded_stream_double_buffer_bytes=two,
              caveat='ABI only; unspecified pipeline residency is bracketed, not assumed. '
                     'Add peak overlapping temporaries/spills/scratch, account for aliases '
                     'and compiler buffer reuse, and compare with actual allocation.')


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('dump')
  p.add_argument('--output', required=True)
  a = p.parse_args()
  result = audit(a.dump)
  Path(a.output).write_text(json.dumps(result, indent=2)+'\n')
  print(json.dumps({k:v for k,v in result.items() if k != 'buffers'}, indent=2))
