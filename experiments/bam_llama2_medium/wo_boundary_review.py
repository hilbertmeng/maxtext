"""Global review for frozen boundary artifacts. No TPU lifecycle actions."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from wo_boundary_decomposition import merge_sums,describe

EXPECTED={35635:(2.0112438201904297,.21486762166023254),35636:(1.924464225769043,2.1978654861450195)}

def require(condition,message):
    if not condition:raise ValueError(message)

def review_layer(partitions,actual,plain):
    merged=merge_sums(partitions,range(128))
    require(merged['parts'].shape[:2]==(3,3),'Missing scope or branch')
    require(merged['amplitude'].shape==(3,6),'Invalid amplitude shape')
    require(all(np.isfinite(merged[k]).all() for k in ('parts','amplitude')),'Nonfinite partials')
    actual,plain=np.asarray(actual,np.float64),np.asarray(plain,np.float64)
    require(actual.shape==merged['parts'].shape[2:]==plain.shape,'W_O shape mismatch')
    denom=max(np.linalg.norm(actual),1e-30)
    identity=float(np.linalg.norm(actual-plain)/max(np.linalg.norm(plain),1e-30))
    require(identity<.01,'W_O capture/plain gate')
    closure_h=float(np.linalg.norm(merged['parts'][0,2]-actual)/denom)
    closure_branches=float(np.linalg.norm(merged['parts'][0,:2].sum(0)-actual)/denom)
    require(closure_h<.015,'Frozen H^T G/actual W_O gate')
    require(closure_branches<.015,'Frozen AV+O/actual W_O gate')
    scopes=describe(merged)
    for i,s in enumerate(scopes):
        a2,o2,h2,cross,error,count=merged['amplitude'][i]
        require(count>0 and min(a2,o2,h2,error)>=0,'Empty/invalid amplitude')
        require(s['sum_forward_relative_squared_error']<1e-4,'Forward H=AV+O gate')
        s.update(AV_rms=float(np.sqrt(a2/count)),O_rms=float(np.sqrt(o2/count)),H_rms=float(np.sqrt(h2/count)))
        if i==2:
            ga=s['AV_gradient_norm']/max(np.sqrt(a2),1e-30)
            go=s['O_gradient_norm']/max(np.sqrt(o2),1e-30)
            require(ga>0 and go>0 and abs(ga/go-1)<.01,'Same-G single-token rank-one gate')
            s.update(output_G_norm_from_AV=float(ga),output_G_norm_from_O=float(go))
        require(all(np.isfinite(v)for v in s.values()if isinstance(v,(int,float))),'Nonfinite derived metric')
    return dict(W_O_leaf_relative_error=identity,H_gradient_closure=closure_h,split_relative_closure=closure_branches,scopes=scopes)


def review_workers(directory,runner_files,input_reference):
    workers=sorted(p for p in Path(directory).iterdir()if p.is_dir())
    require(len(workers)==4,'Require exactly four workers')
    reports={step:[]for step in EXPECTED}
    for worker in workers:
        require((worker/'exit-code.txt').read_text().strip()=='0','Worker failed')
        for name,sha in runner_files.items():
            require(hashlib.sha256((worker/'runner'/name).read_bytes()).hexdigest()==sha,'Runner version mismatch')
        for step,(loss,raw)in EXPECTED.items():
            p=worker/'full'/f'step{step}';r=json.loads((p/'wo-split.json').read_text())
            require(r['step']==step and r['complete'] and r['capture_complete'],'Incomplete/wrong-step capture')
            require(r['scientific_review_complete'] is False,'Capture mislabeled as reviewed')
            require(abs(r['native_control']['learning/loss']-loss)<1e-6 and abs(r['native_control']['learning/raw_grad_norm']/raw-1)<1e-6,'Native replay gate')
            require(abs(r['plain_loss']-loss)<1e-3 and abs(r['plain_raw']/raw-1)<.01,'Plain/native gate')
            require(abs(r['instrumented_loss']-r['plain_loss'])<1e-6 and r['instrumented_vector_relative_error']<.01,'Full gradient identity gate')
            require(len(r['W_O_leaf_relative_errors'])==6 and all(0<=x<.01 for x in r['W_O_leaf_relative_errors']),'Reported W_O identity gate')
            meta=json.loads((p/'boundary-manifests.json').read_text())
            for layer in range(6):
                rows=sum((x['metadata']['global_rows']for x in meta if x['metadata']['layer']==layer),[])
                require(sorted(rows)==sorted(r['local_rows']) and len(set(rows))==len(rows),'Boundary capture coverage mismatch')
                require(all(x['metadata']['step']==step for x in meta),'Boundary step mismatch')
            reports[step].append((p,r))
    result={}
    for step,items in reports.items():
        rows=sum((r['local_rows']for _,r in items),[])
        require(sorted(rows)==list(range(128)),'Global row identity mismatch')
        shards=sum((r['input_shards']for _,r in items),[])
        require(len(shards)==16 and len({x['index']for x in shards})==16,'Missing/duplicate input shard')
        if step==35636:
            key=lambda xs:Counter((x['index'],x['sha256'])for x in xs)
            require(key(shards)==key(input_reference),'First-spike input16 mismatch')
        actual=[np.load(p/'actual-wo.npy',allow_pickle=False)for p,_ in items]
        plain=[np.load(p/'plain-wo.npy',allow_pickle=False)for p,_ in items]
        require(all(np.array_equal(x,actual[0])for x in actual[1:]),'Cross-worker actual W_O mismatch')
        require(all(np.array_equal(x,plain[0])for x in plain[1:]),'Cross-worker plain W_O mismatch')
        require(actual[0].shape[1]==6 and np.isfinite(actual[0]).all() and np.isfinite(plain[0]).all(),'Invalid W_O arrays')
        layers=[]
        for layer in range(6):
            partitions=[]
            for p,r in items:
                with np.load(p/f'layer-{layer:02d}-partial.npz',allow_pickle=False)as f:part={k:f[k]for k in ('parts','amplitude','global_rows')}
                require(sorted(part['global_rows'].tolist())==sorted(r['local_rows']),'Partial row identity mismatch')
                partitions.append(part)
            layers.append(dict(layer=layer,**review_layer(partitions,actual[0][:,layer],plain[0][:,layer])))
        result[str(step)]=layers
    comparisons=[]
    for layer,(normal,spike)in enumerate(zip(result['35635'],result['35636'])):
        for n,s in zip(normal['scopes'],spike['scopes']):
            keys=['AV_gradient_norm','O_gradient_norm','sum_gradient_norm','AV_rms','O_rms','H_rms']
            if s['scope']=='token9_1211':keys+=['output_G_norm_from_AV']
            comparisons.append(dict(layer=layer,scope=s['scope'],normal=n,spike=s,spike_over_normal={k:s[k]/n[k]if n[k]else None for k in keys}))
    return dict(passed=True,scientific_review_complete=True,workers=4,first_spike_input16_exact=True,layers=result,rows=comparisons,
        scope='Actual adjacent normal35635/spike35636 batches, shared G per boundary. Norms and signed projections are not additive causal shares or fixed-batch H/G counterfactuals.')
