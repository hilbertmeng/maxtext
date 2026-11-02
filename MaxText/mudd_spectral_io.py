"""Read only selected parameter arrays from committed historical OCDBT checkpoints."""
import ast,concurrent.futures,json,os,subprocess
from pathlib import Path
import numpy as np
import tensorstore as ts

def metadata(root):
    return json.loads(subprocess.check_output(['gsutil','cat',root.rstrip('/')+'/_METADATA'],text=True))

def entries(meta,prefix):
    result={}
    for key,value in meta['tree_metadata'].items():
        path=ast.literal_eval(key)
        if path[:len(prefix)]==tuple(prefix) and not value['value_metadata'].get('skip_deserialize',False):result[path[len(prefix):]]=path
    return result

def read(root,meta,path):
    driver='zarr3' if meta.get('use_zarr3',False) else 'zarr'
    a=ts.open({'driver':driver,'kvstore':{'driver':'ocdbt','base':root.rstrip('/')+'/'},'path':'.'.join(path)},open=True).result()
    return np.asarray(a.read().result())

def params(root,family,head_only=False,workers=8):
    meta=metadata(root);prefix=('mdl_vars','params','lm') if family=='old' else ('params','params')
    wanted=entries(meta,prefix)
    if head_only:
        wanted={k:v for k,v in wanted.items() if k[:1] in [('final_ln',),('softmax',)]} if family=='old' else {k:v for k,v in wanted.items() if k[:2]==('decoder','lm_head')}
    assert len(wanted)==(3 if head_only else 800),(family,len(wanted))
    actual_step=int(read(root,meta,('step',)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        arrays=dict(zip(wanted,pool.map(lambda p:read(root,meta,p),wanted.values())))
    return arrays,dict(checkpoint=root,step=actual_step,keys=len(arrays),parameter_count=sum(x.size for x in arrays.values()),driver='zarr3' if meta.get('use_zarr3') else 'zarr')

def flatten_head(flat,family):
    if family=='old':return flat[('softmax','logits_ffn','linear','w')],flat[('final_ln','scale')]+1,flat[('final_ln','bias')]
    return flat[('decoder','lm_head','logits_dense')],flat[('decoder','lm_head','decoder_norm','scale')],flat[('decoder','lm_head','decoder_norm','bias')]

def head_inventory(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    import jax.numpy as jnp
    from scipy.linalg import eigh
    for family in ['old','serial']:
        for step in [33000,43000,53000]:
            path=output/f'{family}_{step}.npz';marker=path.with_suffix('.json')
            if marker.exists():continue
            root=(f'gs://newproject-1-llm_projects_europe-west4/log/PileMUDDPythia3BPlusOcdbtv6e/checkpoints/checkpoint_{step:08d}/state' if family=='old' else f'gs://newproject-1-llm_projects_us-east5/log/MuddPythia3BSerialMLP/checkpoints/{step}/items')
            flat,meta=params(root,family,True)
            assert meta['step']==step+(family=='serial'),meta
            meta['checkpoint_label_step']=step
            meta['state_step_offset_from_label']=int(meta['step']-step)
            w,g,b=flatten_head(flat,family);assert w.shape==(2560,50432)
            # Native forwards cast the vocabulary projection to BF16. Geometry is
            # analyzed in FP64 using those same representable effective weights.
            wf=np.asarray(jnp.asarray(w).astype(jnp.bfloat16),np.float64)
            gram=wf@wf.T
            eig,basis=eigh(gram,driver='evd',check_finite=False);assert eig.min()>0
            eff=g.astype(np.float64)[:,None]*gram*g.astype(np.float64)[None,:]
            ee,eb=eigh(eff,driver='evd',check_finite=False);assert ee.min()>0
            singular=np.sqrt(eig);es=np.sqrt(ee)
            meta.update(family=family,analysis_weight='BF16 forward-effective lm_head weights; FP64 Gram/eigh',dimension=2560,
                sigma_min=float(singular[0]),sigma_max=float(singular[-1]),sigma_min_over_max=float(singular[0]/singular[-1]),
                exact_null_dimension=int((singular<singular[-1]*1e-10).sum()),gamma_min=float(g.min()),gamma_max=float(g.max()),gamma_rms=float(np.sqrt(np.mean(g*g))),bias_rms=float(np.sqrt(np.mean(b*b))))
            np.savez_compressed(path,singular=singular,basis=basis.astype(np.float32),effective_singular=es,effective_basis=eb.astype(np.float32),gamma=g,bias=b)
            marker.write_text(json.dumps(meta,indent=2)+'\n');print('HEAD_SPECTRUM_COMPLETE',family,step,json.dumps(meta),flush=True)

if __name__=='__main__':
    import sys
    # Short-lived token stays only in this process; do not persist credentials.
    if not os.getenv('GOOGLE_AUTH_TOKEN_FOR_TESTING'):
        os.environ['GOOGLE_AUTH_TOKEN_FOR_TESTING']=subprocess.check_output(['/home/mengqy/google-cloud-sdk/bin/gcloud','auth','print-access-token'],text=True).strip()
    head_inventory(sys.argv[1])
