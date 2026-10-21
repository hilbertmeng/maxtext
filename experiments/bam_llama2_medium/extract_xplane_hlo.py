#!/usr/bin/env python3
"""Extract the executed HLO and compiler buffer assignment from a raw TPU XPlane.

Run locally with the pinned CPU environment, after pulling the worker's artifact.
No model initialization, recompilation or TPU is required. Color0 is HBM on TPU;
allocation sizes are compiler heap reservations, not post-step live-buffer stats.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import os
from pathlib import Path


def extract(trace, output):
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    from tensorflow.tsl.profiler.protobuf import xplane_pb2
    from tensorflow.compiler.xla.service import hlo_pb2
    from jaxlib import xla_client

    payload=trace.read_bytes()
    space=xplane_pb2.XSpace();space.ParseFromString(payload)
    if space.errors:raise ValueError(list(space.errors))
    found=[]
    for plane in space.planes:
        for md in plane.event_metadata.values():
            if not md.name.startswith('jit_train_step('):continue
            for stat in md.stats:
                if plane.stat_metadata[stat.metadata_id].name=='Hlo Proto':
                    found.append((md.name,stat.bytes_value))
    if len(found)!=1:raise ValueError(f'Expected one train-step HLO, found {len(found)}')
    name,raw=found[0]
    proto=hlo_pb2.HloProto();proto.ParseFromString(raw)
    module_bytes=proto.hlo_module.SerializeToString()
    module=xla_client._xla.HloModule.from_serialized_hlo_module_proto(module_bytes)
    output.mkdir(parents=True,exist_ok=True)
    (output/'hlo-proto.pb').write_bytes(raw)
    (output/'hlo-module.pb').write_bytes(module_bytes)
    with gzip.open(output/'optimized-hlo.txt.gz','wt') as stream:stream.write(module.to_string())
    colors=defaultdict(lambda:dict(allocated_bytes=0,nonparameter_nonconstant_heap_bytes=0,
                                   parameter_bytes=0,allocation_count=0))
    for allocation in proto.buffer_assignment.buffer_allocations:
        row=colors[allocation.color];row['allocated_bytes']+=allocation.size;row['allocation_count']+=1
        if allocation.is_entry_computation_parameter:row['parameter_bytes']+=allocation.size
        if not(allocation.is_entry_computation_parameter or allocation.is_constant or allocation.is_thread_local):
            row['nonparameter_nonconstant_heap_bytes']+=allocation.size
    result=dict(trace=str(trace.resolve()),trace_sha256=hashlib.sha256(payload).hexdigest(),
                program=name,hlo_proto_sha256=hashlib.sha256(raw).hexdigest(),
                compiler_buffer_assignment_by_color=dict(colors),
                largest_allocations=[dict(index=a.index,size=a.size,color=a.color,
                                          parameter=a.is_entry_computation_parameter,
                                          thread_local=a.is_thread_local,constant=a.is_constant)
                                     for a in sorted(proto.buffer_assignment.buffer_allocations,
                                                     key=lambda a:a.size,reverse=True)[:20]])
    (output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(output=str(output),program=name,allocation_colors=dict(colors))))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('xplane',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    extract(args.xplane,args.output)


if __name__=='__main__':main()
