"""Actual d(loss)/d(write probability), per layer/token/head at the baseline."""
import contextlib
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import write_geometry as w
from write_gate_bets import intervene
from write_geometry import jax,jnp,partitioning,train,max_utils


def run(cfg):
    out=Path(os.environ['BET_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
    path=Path('/tmp/pile_eval_cohort.npz')
    with np.load(path) as f:cohort={k:f[k] for k in w.KEYS}
    protocol=json.loads(Path(os.environ['BET_PROTOCOL']).read_text())
    meta=dict(model=w.BASE,checkpoint=cfg.load_parameters_path,dtype=str(cfg.dtype),matmul_precision=str(cfg.matmul_precision),
              protocol=protocol,runtime=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              cohort_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),gradient='d per-sequence mean loss / d actual write-gate probability; no h weighting')
    (out/'metadata.json').write_text(json.dumps(meta,indent=2))
    rng,writer,manager,mesh,model,_,tx=train.setup_mesh_and_model(cfg)
    state,_,_,_=max_utils.setup_training_state(model,SimpleNamespace(meta_dict={'checkpoint_step':None}),tx,cfg,rng,mesh,manager)
    length=cohort['inputs'].shape[-1]
    zeros=jnp.zeros((24,1,length,16),jnp.float32)
    def forward(params,b,delta):
        with intervene(-1,0,0.,0.,probability_delta=delta) if delta is not None else contextlib.nullcontext():
            result,inter=model.apply(params,b['inputs'],b['inputs_position'],decoder_segment_ids=b['inputs_segmentation'],
                                     decoder_target_mask=b['targets_segmentation'],decoder_target_tokens=b['targets'],enable_dropout=False,
                                     rngs={'params':rng,'dropout':rng},mutable=['intermediates'])
        valid=b['targets_segmentation']!=0;tokens=jnp.where(valid,result[0],0.)
        loss=tokens.sum()/jnp.maximum(valid.sum(),1)
        if delta is None:return tokens
        context=jnp.concatenate([jnp.asarray(v[0]).reshape(-1,length,16,3) for k,v in w.flatten_dict(inter).items() if k[-1]=='gate_gradient_context'],axis=0)
        return loss,(tokens,context)
    gradfn=jax.jit(jax.value_and_grad(forward,argnums=2,has_aux=True))
    native=jax.jit(lambda p,b:forward(p,b,None));start=time.perf_counter()
    with mesh,partitioning.axis_rules(cfg.logical_axis_rules):
        for i in range(32,128):
            dest=out/f'gradient_{i:03d}.npz'
            if dest.exists():continue
            b={k:jnp.asarray(v[i:i+1]) for k,v in cohort.items()}
            original=np.asarray(native(state.params,b))
            (_, (tokens,ctx)),derivative=jax.device_get(gradfn(state.params,b,zeros))
            assert np.isfinite(derivative).all()
            layers=ctx[:,0,0,2].astype(int);assert sorted(layers.tolist())==list(range(24)),layers
            ctx=ctx[np.argsort(layers)];derivative=derivative[:,0]
            assert np.count_nonzero(derivative[23])==0, 'Terminal M write must not affect the loss'
            valid=cohort['targets_segmentation'][i]!=0;count=int(valid.sum())
            drift=float((tokens.astype(np.float64)-original).sum()/count)
            assert abs(drift)<1e-7,drift
            # Cross-check each head against the earlier h-weighted direction.
            reductions={}
            for h in protocol['heads']:
                l,n=h['layer'],h['head'];g=ctx[l,:,n,0];r=ctx[l,:,n,1]
                selected=(r>=h['threshold'])&valid
                step=np.minimum(.01,np.minimum(.1*g,.1*(1-g)))
                reductions[h['id']]=float(np.sum(derivative[l,:,n].astype(np.float64)*step*selected))
            pending=dest.with_suffix('.npz.tmp')
            with pending.open('wb') as f:np.savez_compressed(f,gradient=derivative,write_gate=ctx[...,0],read_gate=ctx[...,1],valid=valid,positions=cohort['inputs_position'][i])
            pending.replace(dest)
            (out/f'check_{i:03d}.json').write_text(json.dumps(dict(sequence=i,primal_delta=drift,directional_reconstruction=reductions),indent=2))
            print('GATE_GRADIENT_SEQUENCE_DONE',i,'elapsed',time.perf_counter()-start,flush=True)
    if writer:writer.flush()
    (out/'DONE').write_text('96 per-position gate gradient sequences complete\n')
    print('GATE_GRADIENT_DONE',flush=True)


if __name__=='__main__':w.app.run(lambda argv:run(w.pyconfig.initialize(argv)))
