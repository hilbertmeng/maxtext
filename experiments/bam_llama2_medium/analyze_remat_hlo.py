#!/usr/bin/env python3
"""Summarize optimized remat HLOs; static instruction counts are not execution times.

Input: a local copy of compile_qknorm_remat_hlo.py's GCS prefix, excluding AOT
pickles. Retains computation/source names and shapes for inspecting changed tapes,
copies and collectives. Sizes describe instruction outputs, not unique/peak buffers.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import re

INSTRUCTION = re.compile(r'^\s*(?:ROOT )?%([^ ]+) = (.*?) ([a-zA-Z][\w-]*)\(')
ARRAY = re.compile(r'\b(pred|bf16|f16|f32|f64|s8|u8|s16|u16|s32|u32|s64|u64)\[([\d,]*)\]')
WIDTH = dict(pred=1, bf16=2, f16=2, f32=4, f64=8,
             s8=1, u8=1, s16=2, u16=2, s32=4, u32=4, s64=8, u64=8)
WATCH = {'copy', 'transpose', 'dynamic-update-slice', 'all-gather', 'all-gather-start',
         'all-reduce', 'all-reduce-start', 'reduce-scatter', 'collective-permute', 'while'}


def summarize(path):
    text = gzip.open(path, 'rt').read()
    counts = Counter()
    watched = []
    computation = None
    for line in text.splitlines():
        if line.startswith(('ENTRY %', '%')) and line.endswith('{'):
            computation = line.split('%', 1)[1].split(' ', 1)[0].split('(', 1)[0]
        match = INSTRUCTION.match(line)
        if not match:
            continue
        name, shape, op = match.groups()
        counts[op] += 1
        if op not in WATCH:
            continue
        sizes = [WIDTH[dt] * math.prod(map(int, dims.split(','))) if dims else WIDTH[dt]
                 for dt, dims in ARRAY.findall(shape)]
        source = re.search(r'op_name="([^"]*)"', line)
        watched.append(dict(name=name, computation=computation, op=op, shape=shape,
                            largest_output_array_bytes=max(sizes, default=0),
                            source=source.group(1) if source else None))
    analysis_path = path.with_name('analysis.json')
    return dict(path=str(path), sha256=hashlib.sha256(text.encode()).hexdigest(),
                static_instruction_counts=dict(counts),
                largest_watched_outputs=sorted(watched, key=lambda v:v['largest_output_array_bytes'],
                                               reverse=True)[:50],
                watched_instructions=watched,
                compiler_analysis=json.loads(analysis_path.read_text()) if analysis_path.exists() else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = {path.parent.name:summarize(path) for path in sorted(args.root.glob('*/optimized-hlo.txt.gz'))}
    if not result:
        raise SystemExit('No HLO artifacts found')
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    for arm, data in result.items():
        print(arm, data['compiler_analysis'])
        print({op:n for op,n in data['static_instruction_counts'].items() if op in WATCH or op in ('dot','fusion')})


if __name__ == '__main__':
    main()
