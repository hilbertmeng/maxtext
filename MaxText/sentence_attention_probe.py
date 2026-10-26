"""Frozen native all-head attention for a literal prompt, with observer controls."""
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

import sentence_attention_capture as capture
import checkpointing,exp,max_utils,pyconfig
from layers.models import Transformer

def config_for(args):
    parent=getattr(exp,args.experiment)
    attrs=dict(model_name=args.experiment,only_eval=True,enable_checkpointing=False,
        per_device_batch_size=1.0,eval_per_device_batch_size=1.0,target_train_batch_size=args.batch_size,
        max_target_length=args.length,max_prefill_predict_length=args.length,
        dataset_type='synthetic',log_config=False,jax_cache_dir='',enable_goodput_recording=False,
        monitor_goodput=False,record_training_health_metrics=False,record_internal_nn_metrics=False,
        tensorboard_dir='/tmp/bam_stream_probe/tensorboard/',base_output_directory='/tmp/bam_stream_probe/')
    exp.StreamProbeConfig=type('StreamProbeConfig',(parent,),attrs)
    return pyconfig.initialize(['bam_stream_probe','MaxText/configs/base.yml'],exp_class='StreamProbeConfig',
        run_name=args.label,base_output_directory='/tmp/bam_stream_probe/',enable_checkpointing=False,
        only_eval=True)


def plain(model,params,batch,rng):
    result=model.apply(params,batch['inputs'],batch['positions'],
        decoder_target_tokens=batch['targets'],decoder_segment_ids=batch['segments'],
        decoder_target_mask=batch['mask'],enable_dropout=False,rngs={'dropout':rng,'params':rng,'aqt':rng})
    loss=result[0]
    weights=batch['mask']
    return jnp.sum(loss*weights,axis=-1)/jnp.maximum(jnp.sum(weights,axis=-1),1)


def main():
    p=argparse.ArgumentParser()
    for name in ['experiment','checkpoint','training-config','training-commit','label','output','gcs-output','request']:p.add_argument('--'+name,required=True)
    p.add_argument('--expected-parameters',type=int,required=True)
    a=p.parse_args();a.batch_size=4;a.length=2048;a.seed=20261026
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);Path('/tmp/bam_stream_probe',a.label).mkdir(parents=True,exist_ok=True)
    request=json.loads(Path(a.request).read_text());ids=np.asarray(request['token_ids'],np.int32);n=len(ids);assert n==55 and n<2048
    assert hashlib.sha256(request['text'].encode()).hexdigest()==request['text_sha256']
    capture.PREFIX=n;cfg=config_for(a)
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
        params=checkpointing.load_params_from_path(a.checkpoint,abstract.params)
    count=sum(v.size for v in jax.tree.leaves(params));assert count==a.expected_parameters

    assert cfg.num_decoder_layers==cfg.num_query_heads==32 and cfg.head_dim==80
    assert bool(getattr(cfg,'bam_enabled',False)) == ('SeparateVOKeys' in a.experiment)
    tokens=np.zeros((4,2049),np.int32);tokens[:,:n]=ids
    batch=dict(inputs=jnp.asarray(tokens[:,:2048]),targets=jnp.asarray(tokens[:,1:2049]),mask=jnp.broadcast_to(jnp.arange(2048)<n-1,(4,2048)).astype(jnp.int32),positions=jnp.broadcast_to(jnp.arange(2048),(4,2048)),segments=jnp.ones((4,2048),jnp.int32))
    capture.ACTIVE=False;native_fn=jax.jit(lambda p,b:plain(model,p,b,rng))
    with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):baseline=np.asarray(native_fn(params,batch))
    assert np.isfinite(baseline).all();capture.install()
    def observe(p,b):
        capture.ACTIVE=True
        try:
            result,collection=model.apply(p,b['inputs'],b['positions'],decoder_target_tokens=b['targets'],decoder_segment_ids=b['segments'],decoder_target_mask=b['mask'],enable_dropout=False,rngs={'params':rng,'dropout':rng,'aqt':rng},mutable=['intermediates'])
            loss=jnp.sum(result[0]*b['mask'],axis=-1)/jnp.sum(b['mask'],axis=-1)
            return loss,{'/'.join(k):v for k,v in flatten_dict(collection['intermediates']).items() if k[-1].startswith('sentence_')}
        finally:capture.ACTIVE=False
    observed_fn=jax.jit(observe)
    with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):loss,raw=jax.device_get(observed_fn(params,batch))
    drift=float(np.max(np.abs(loss-baseline)));assert drift<.005 and np.isfinite(loss).all(),('Native CE drift',drift)
    def unpack(raw):
        flat={name:np.asarray(value[0]) for name,value in raw.items()};layers={}
        for name,index in flat.items():
            if not name.endswith('/sentence_layer_index'):continue
            prefix=name.rsplit('/',1)[0]+'/'
            for pos in np.ndindex(index.shape):
                idx=int(index[pos]);assert idx not in layers;layers[idx]={key[len(prefix)+9:]:value[pos] for key,value in flat.items() if key.startswith(prefix) and key[len(prefix):].startswith('sentence_')}
        assert set(layers)==set(range(32));return {key:np.stack([layers[i][key] for i in range(32)]) for key in layers[0] if key!='layer_index'}
    arrays=unpack(raw);alpha=arrays['alpha'];assert alpha.shape==(32,32,n,n) and np.isfinite(alpha).all()
    assert (alpha>=0).all() and (alpha<=1).all() and np.max(np.abs(alpha.sum(-1)-1))<2e-6
    assert np.max(np.abs(np.triu(alpha,1)))==0
    errors=arrays['error'];closure=float(np.sqrt(errors[...,0]/np.maximum(errors[...,1],1e-30)).max());assert closure<.02,('Native AV reconstruction',closure)
    with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):again,repeat=jax.device_get(observed_fn(params,batch))
    np.testing.assert_array_equal(again,loss)
    for k,v in raw.items():np.testing.assert_array_equal(v[0],repeat[k][0])
    changed=tokens.copy();changed[:,n:]=2;future=dict(batch,inputs=jnp.asarray(changed[:,:2048]),targets=jnp.asarray(changed[:,1:2049]))
    with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):_,other=jax.device_get(observed_fn(params,future))
    future_arrays=unpack(other)
    for key in ['alpha','query','key','value','native_output']:np.testing.assert_array_equal(arrays[key],future_arrays[key])
    np.savez_compressed(out/'attention.npz',**arrays,token_ids=ids)
    meta=dict(data_kind='model_attention',experiment=a.experiment,checkpoint=a.checkpoint,step=16000,training_commit=a.training_commit,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),parameter_count=count,request=request,layers=32,heads=32,head_dim=80,native_length=2048,batch_size=4,unique_prompts=1,batch_repeats=4,observer_native_CE_max_difference=drift,native_AV_reconstruction_max_relative_RMS=closure,repeat_exact=True,future_padding_invariance_exact=True,observed_loss=loss.tolist(),native_loss=baseline.tolist(),devices=[str(x) for x in jax.devices()],jax_version=jax.__version__,verified_config_keys=checked,config={key:value for key,value in cfg.get_keys().items() if key in critical or key in ['bam_enabled','bam_splash_attention','bam_pallas_core','bam_pallas_vmem_mib']},definition='All32layers/32heads, full55x55 literal-token maps, reconstructed FP32 softmax from actual post-norm/pre-scaled Q/K passed to nativeattention. Native AV unchanged and verified by reconstruction. Causal no BOS/EOS. Padding follows prompt and is excluded from exported maps; future tokens varied to verify prefix invariance. Batch repeated only for native TPU mesh; export one example. No optimizer/update/checkpointwrite.')
    (out/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2,default=str)+'\n')
    subprocess.run(['gsutil','-m','rsync','-r',str(out),a.gcs_output],check=True)
    (out/'complete.json').write_text(json.dumps(dict(label=a.label,text_sha256=request['text_sha256'],source_commit=meta['source_commit']))+'\n');subprocess.run(['gsutil','-q','cp',str(out/'complete.json'),a.gcs_output+'/complete.json'],check=True)
    print('REAL_SENTENCE_ATTENTION_COMPLETE',a.label,'CE_drift',drift,'AV_relative_RMS',closure,flush=True)
if __name__=='__main__':main()
