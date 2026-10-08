#!/usr/bin/env python3
"""Raw XPlane -> per-step leaf-op aggregates (ms, model TF, GB) on one TPU device.
Excludes while/scan wrappers and numeric module markers; averages over fully covered train steps."""
import importlib.util, json, sys, collections
from pathlib import Path
PROTO='/data0/xd/conda/envs/maxtext-cpu/lib/python3.12/site-packages/tensorflow/tsl/profiler/protobuf/xplane_pb2.py'
spec=importlib.util.spec_from_file_location('xp',PROTO); xp=importlib.util.module_from_spec(spec); spec.loader.exec_module(xp)
def load(path):
    s=xp.XSpace(); s.ParseFromString(Path(path).read_bytes())
    planes=[p for p in s.planes if p.name.startswith('/device:TPU:')]
    p=planes[0]
    def stats(md):
        r={}
        for st in md.stats:
            k=st.WhichOneof('value'); v=getattr(st,k) if k else None
            if k=='ref_value': v=p.stat_metadata[v].name
            if k!='bytes_value': r[p.stat_metadata[st.metadata_id].name]=v
        return r
    steps=[];ops=[]
    for line in p.lines:
        base=line.timestamp_ns*1000
        if line.name=='XLA Modules':
            for e in line.events:
                md=p.event_metadata[e.metadata_id]; n=md.display_name or md.name
                if n.startswith('jit_train_step'): steps.append((base+e.offset_ps, base+e.offset_ps+e.duration_ps))
        if line.name=='XLA Ops':
            cache={}
            for e in line.events:
                if e.metadata_id not in cache:
                    md=p.event_metadata[e.metadata_id]; a=stats(md); a['hlo']=md.display_name or md.name; cache[e.metadata_id]=a
                ops.append((base+e.offset_ps, e.duration_ps, cache[e.metadata_id]))
    return p.name, steps, ops
def main(path,out):
    dev,steps,ops=load(path)
    agg=collections.defaultdict(lambda:[0.0,0.0,0.0,0]); covered=[]
    for (a,b) in steps:
        inside=sorted([(t,d,m) for t,d,m in ops if t>=a and t+d<=b+1e6 and not m['hlo'].lower().startswith('while') and not m['hlo'].isdigit()], key=lambda x:(x[0],-x[1]))
        # exclusive time: subtract directly nested children from their enclosing event
        excl=[d for _,d,_ in inside]; stack=[]
        for i,(t,d,m) in enumerate(inside):
            while stack and inside[stack[-1]][0]+inside[stack[-1]][1] <= t: stack.pop()
            if stack and t+d <= inside[stack[-1]][0]+inside[stack[-1]][1]+1:
                excl[stack[-1]]-=d
            stack.append(i)
        cov=0.0
        for (t,d,m),x_ in zip(inside,excl):
            x_=max(x_,0); cov+=x_
            key=(str(m.get('tf_op','')), m['hlo'].split('.')[0])
            x=agg[key]; x[0]+=x_/1e9; x[1]+=float(m.get('model_flops',0) or 0)/1e12; x[2]+=float(m.get('bytes_accessed',0) or 0)/1e9; x[3]+=1
        covered.append(cov/(b-a))
    n=len(steps)
    res={'device':dev,'steps':n,'step_ms':[(b-a)/1e9 for a,b in steps],'coverage':covered,
         'ops':[dict(tf_op=k[0],hlo=k[1],ms=v[0]/n,tf=v[1]/n,gb=v[2]/n,count=v[3]/n) for k,v in sorted(agg.items(),key=lambda kv:-kv[1][0])]}
    json.dump(res,open(out,'w'))
    print(dev,'steps',n,'step_ms',[round(x,2) for x in res['step_ms']],'coverage',[round(c,4) for c in covered])
    print('exclusive ms/step',round(sum(o['ms'] for o in res['ops']),2),'TF',round(sum(o['tf'] for o in res['ops']),3),'GB',round(sum(o['gb'] for o in res['ops']),2))
if __name__=='__main__': main(sys.argv[1],sys.argv[2])
