"""Matched-cohort, checkpoint-only BAM stream measurements and M-write ablations."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile
from types import SimpleNamespace

from flax import linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict
import jax
import jax.numpy as jnp
import numpy as np

import vo_similarity_capture as capture
import checkpointing
import exp
import max_utils
import pyconfig
from layers.models import Transformer


from vo_similarity_probe import config_for,cohort,hash_sequence,plain,observed

def setup(a):
    cfg=config_for(a)
    expected={}
    for line in Path(a.training_config).read_text().splitlines():
        key,sep,val=line.partition(': ')
        if sep:expected[key]=val
    critical={'qk_norm','normalization_layer_epsilon','normalization_layer_type','direct_scale','decoder_block','vocab_size','head_dim','num_decoder_layers','emb_dim','num_query_heads','num_kv_heads','mlp_dim','mlp_dim_by_block','mlp_activations','dynamic_mlp_dim','scan_layers','scan_use_mudd','dtype','weight_dtype','attention','attention_type','logits_via_embedding','normalize_embedding_logits'}
    checked=[]
    for key,raw in expected.items():
        if key not in cfg.get_keys() or not(key in critical or key.startswith(('bam_','mudd_','pythia_'))):continue
        try:want=ast.literal_eval(raw)
        except (ValueError,SyntaxError):want=raw
        value=getattr(cfg,key)
        if isinstance(value,type):value=getattr(value,'__name__',str(value))
        if isinstance(value,tuple):value=list(value)
        if isinstance(want,tuple):want=list(want)
        if value!=want:raise ValueError(('Model config mismatch',key,want,str(value)))
        checked.append(key)
    assert not (getattr(cfg,'bam_standard_value_heads',0) or 0)
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes);model=Transformer(cfg,mesh,quant=None);rng=jax.random.key(a.seed)
    with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
        native_unbox=max_utils.unbox_logicallypartioned
        max_utils.unbox_logicallypartioned=lambda tree:jax.tree.map(lambda v:v.unbox(apply_constraint=False) if isinstance(v,nn.spmd.LogicallyPartitioned) else v,tree,is_leaf=lambda v:isinstance(v,nn.spmd.LogicallyPartitioned))
        try:abstract,_,_=max_utils.get_abstract_state(model,None,cfg,rng,mesh,is_training=False)
        finally:max_utils.unbox_logicallypartioned=native_unbox
    return cfg,critical,checked,mesh,model,rng,abstract

def main():
    p=argparse.ArgumentParser()
    for key in ['manifest','training-config','cohort-reference','output','gcs-output']:p.add_argument('--'+key,required=True)
    a=p.parse_args();manifest=json.loads(Path(a.manifest).read_text());arms=manifest['probe_arms']
    a.experiment=manifest['experiment'];a.expected_parameters=manifest['parameter_count'];a.seed=20261021;a.batch_size=4;a.sequences=32;a.length=2048;a.label='trajectory';a.cohort='/tmp/bam_stream_probe/trajectory_cohort32.npz';a.cohort_source=manifest['cohort_source']
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();assert all(x['commit']==source for x in arms)
    assert [x['step'] for x in arms]==sorted(set(x['step'] for x in arms))
    for arm in arms:assert arm['checkpoint']==manifest['checkpoint_root']+'/'+str(arm['step'])+'/items'
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True);Path('/tmp/bam_stream_probe/trajectory').mkdir(parents=True,exist_ok=True)
    cfg,critical,checked,mesh,model,rng,abstract=setup(a)
    data=cohort(a.cohort,a.cohort_source,a.seed,32);hashes=[hash_sequence(x[:2049]) for x in data['tokens']]
    reference=json.loads(Path(a.cohort_reference).read_text());assert hashes==reference['sequence_hashes']
    assert np.all(data['labels'][:,1:2049]>=0)
    batches=[]
    for i in range(0,32,4):
        tokens=data['tokens'][i:i+4];shape=(4,2048)
        batches.append(dict(inputs=jnp.asarray(tokens[:,:2048]),targets=jnp.asarray(tokens[:,1:2049]),mask=jnp.ones(shape,jnp.int32),segments=jnp.ones(shape,jnp.int32),positions=jnp.broadcast_to(jnp.arange(2048),shape)))
    capture.ACTIVE=False;native_fn=jax.jit(lambda p,b:plain(model,p,b,rng));observed_fn=None;previous_losses=None
    destination=a.gcs_output
    for arm in arms:
        a.label=arm['label'];a.checkpoint=arm['checkpoint'];a.gcs_output=destination+'/'+a.label;out=root/a.label;out.mkdir(parents=True,exist_ok=True);start=time.time()
        capture.ACTIVE=False
        with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):params=checkpointing.load_params_from_path(a.checkpoint,abstract.params)
        count=sum(v.size for v in jax.tree.leaves(params));assert count==a.expected_parameters
        baseline=[]
        for b in batches:
            with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):baseline.extend(np.asarray(jax.device_get(native_fn(params,b))).tolist())
        baseline=np.asarray(baseline);assert baseline.shape==(32,) and np.isfinite(baseline).all()
        if previous_losses is not None:assert not np.array_equal(baseline,previous_losses),'Different checkpoint produced identical native losses: reject stale parameter reuse'
        previous_losses=baseline.copy()
        if observed_fn is None:
            capture.install();observed_fn=jax.jit(lambda p,b:observed(model,p,b,rng))
        capture.ACTIVE=True;losses=[];arrays={};indices={};first=None
        for i,b in enumerate(batches):
            with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):loss,raw=jax.device_get(observed_fn(params,b))
            loss=np.asarray(loss);assert loss.shape==(4,) and np.isfinite(loss).all();losses.extend(loss.tolist())
            err=float(np.max(np.abs(loss-baseline[i*4:i*4+4])))
            if err>.005:raise ValueError(('Observer changed native loss',i,err))
            selected={}
            for name,value in raw.items():
                if not name.endswith(('stream_vo_moments','stream_vo_attention','stream_vo_errors','stream_vo_layer_index')):continue
                assert isinstance(value,(tuple,list)) and len(value)==1
                value=np.asarray(value[0]);assert np.isfinite(value).all();name=name.replace('/','__');selected[name]=value
                if name.endswith('stream_vo_layer_index'):
                    if name in indices:np.testing.assert_array_equal(indices[name],value)
                    indices[name]=value
                else:arrays.setdefault(name,[]).append(value)
            if i==0:first=(loss,selected)
            print('VO_BATCH_COMPLETE',a.label,i+1,'native_CE_drift',err,flush=True)
        merged={name:np.concatenate(values,axis=indices[name.rsplit('__',1)[0]+'__stream_vo_layer_index'].ndim) for name,values in arrays.items()};merged.update(indices)
        L=int(cfg.num_decoder_layers)
        def layer_array(suffix):
            result={}
            for name,value in merged.items():
                if not name.endswith('__stream_vo_'+suffix):continue
                index=merged[name.rsplit('__',1)[0]+'__stream_vo_layer_index']
                for pos in np.ndindex(index.shape):
                    idx=int(index[pos]);assert idx not in result;result[idx]=value[pos]
            assert set(result)==set(range(L));return np.stack([result[i] for i in range(L)],axis=1)
        m=layer_array('moments');errors=layer_array('errors');attn=layer_array('attention')
        assert m.shape==(32,L,64,cfg.num_query_heads,5,3) and attn.shape==(32,L,64,cfg.num_query_heads,2) and errors.shape==(32,L,4)
        closures={}
        for key,i in [('V',0),('O',2)]:
            ratio=np.sqrt(errors[...,i]/np.maximum(errors[...,i+1],1e-30));closures[key]=float(ratio.max())
            if closures[key]>.02:raise ValueError(('Read decomposition failed',key,closures[key]))
        with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):again,repeat=jax.device_get(observed_fn(params,batches[0]))
        np.testing.assert_array_equal(again,first[0])
        for name,value in repeat.items():
            key=name.replace('/','__')
            if key in first[1]:np.testing.assert_array_equal(value[0],first[1][key])
        capture.ACTIVE=False
        np.savez_compressed(out/'observations.npz',inputs=data['tokens'][:,:2048],targets=data['tokens'][:,1:2049],moments=m,attention=attn,closure_errors=errors,native_losses=baseline,observed_losses=np.asarray(losses))
        meta=dict(step=arm['step'],compiled_forward_reused=True,label=a.label,experiment=a.experiment,checkpoint=a.checkpoint,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),parameter_count=count,sequences=32,length=2048,batch_size=4,sequence_hashes=hashes,query_positions=capture.positions(2048).tolist(),pair_names=list(capture.PAIR_NAMES),pair_dimensions=[int(cfg.head_dim)]*4+[int(cfg.bam_v)],moment_names=list(capture.MOMENT_NAMES),attention_names=['self_probability','argmax_self'],closure_relative_RMS_max=closures,native_observer_max_CE_drift=float(np.max(np.abs(np.asarray(losses)-baseline))),repeat_first_batch_exact=True,config={key:value for key,value in cfg.get_keys().items() if key in critical or key in ['bam_v'] or key.startswith(('bam_','pythia_','mudd_'))},training_config_verified_keys=checked,elapsed_seconds=time.time()-start,jax_version=jax.__version__,devices=[str(d) for d in jax.devices()],definition='Native frozen forward. Same-token/layer/head V/O moments at64fixedqueries>=64, beforeW_o. Dynamic/static parts via native shadow read or actual XLA component sites; nativeoutputs unchanged. Zeroenergy vectors excluded on host. Keypair is actual dynamicaddress includingprebias, beforeRMS normalization (positive scalar does not changeangle). Separate V/O slab order VthenO. MHA/O pair is different from rawV/O. No optimizer/update/checkpointwrite.')
        (out/'metadata.json').write_text(json.dumps(meta,indent=2,default=str)+'\n');subprocess.run(['gsutil','-m','rsync','-r',str(out),a.gcs_output],check=True)
        (out/'complete.json').write_text(json.dumps(dict(label=a.label,sequences=32))+'\n');subprocess.run(['gsutil','-q','cp',str(out/'complete.json'),a.gcs_output+'/complete.json'],check=True);print('VO_SIMILARITY_COMPLETE',a.label,flush=True)
        del params,merged,m,errors,attn,arrays,first,raw,repeat
    print('ALL_TRAJECTORY_CHECKPOINTS_COMPLETE',len(arms),flush=True)
if __name__=='__main__':main()
