"""Require split-execution provenance before the unchanged global closure review."""
import json
from pathlib import Path
import numpy as np
from wo_boundary_review import review_workers,require,EXPECTED


def review_split_workers(directory,runner_files,input_reference):
    workers=sorted(p for p in Path(directory).iterdir() if p.is_dir())
    require(len(workers)==4,'Require exactly four workers')
    for worker in workers:
        for step in EXPECTED:
            p=worker/'full'/f'step{step}'
            r=json.loads((p/'wo-split.json').read_text())
            require(r.get('capture_method')=='split_execution_v1','Missing split-execution method')
            require(abs(r['forward_snapshot_loss']-r['plain_loss'])<1e-6,'Independent forward loss gate')
            if r.get('primal_execution')=='value_and_grad':
                require(r['forward_vector_relative_error']<.01,'Snapshot full gradient gate')
                require(len(r['forward_W_O_leaf_relative_errors'])==6 and all(0<=e<.01 for e in r['forward_W_O_leaf_relative_errors']),'Snapshot reported leaf gate')
                fwd=np.load(p/'forward-wo.npy',allow_pickle=False)
                plain=np.load(p/'plain-wo.npy',allow_pickle=False)
                require(fwd.shape==plain.shape and np.isfinite(fwd).all(),'Snapshot gradient shape/finiteness')
                for layer in range(6):
                    ref=plain[:,layer].astype(np.float64)
                    err=np.linalg.norm(fwd[:,layer].astype(np.float64)-ref)/max(np.linalg.norm(ref),1e-30)
                    require(err<.01,'Snapshot actual leaf gate')
            context=r['snapshot_context']
            require(all(isinstance(context.get(k),str) and context[k] for k in ('run','state','input','rng')),'Missing split context')
            require(len(context['input'])==64 and len(context['rng'])==64,'Invalid context fingerprints')
            records=json.loads((p/'boundary-manifests.json').read_text())
            for record in records:
                a,g=record['metadata'],record['cotangent_metadata']
                require(a['phase']=='primal' and g['phase']=='cotangent','Wrong snapshot phases')
                require(a['context']==g['context']==context,'Different split snapshot context')
                require(a['step']==g['step']==step and a['layer']==g['layer'],'Split step/layer mismatch')
                require(a['global_rows']==g['global_rows'],'Split row mismatch')
                require(set(a['files'])=={'AV','localO','H','global_rows','valid'} and set(g['files'])=={'G','global_rows','valid'},'Missing split payload identities')
                for key in ('global_rows','valid'):
                    require(a['files'][key]==g['files'][key],'Split row/mask file mismatch')
                for name in ('AV','localO','H'):
                    require(a['files'][name]['dtype']=='bfloat16','Unexpected primal dtype')
                require(g['files']['G']['dtype']=='bfloat16','Unexpected cotangent dtype')
            if step==35635:
                diag=json.loads((p/'fidelity-diagnostic.json').read_text())
                require(diag['native_moment_consistency']<1e-5,'Native moment consistency gate')
                native=np.load(p/'native-aot-wo.npy',allow_pickle=False)
                require(np.isfinite(native).all(),'Nonfinite native reference')
                for name in ('actual-wo.npy','plain-wo.npy'):
                    v=np.load(p/name,allow_pickle=False)
                    require(v.shape==native.shape and v.shape[1]==6,'Native leaf shape mismatch')
                    for layer in range(6):
                        ref=native[:,layer].astype(np.float64)
                        err=np.linalg.norm(v[:,layer].astype(np.float64)-ref)/max(np.linalg.norm(ref),1e-30)
                        require(err<.01,'Normal native per-leaf gate')
    # Performs all original4worker/native/input16/1%/global1.5%/forward/rank1 gates.
    result=review_workers(directory,runner_files,input_reference)
    result['capture_method']='split_execution_v1'
    result['independent_forward_provenance_verified']=True
    return result
