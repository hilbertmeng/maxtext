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


def cohort(path, source, seed, count):
    path = Path(path)
    if path.exists():
        with np.load(path) as d:return {k:d[k] for k in d.files}
    if str(source).endswith('.npz'):
        path.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['gsutil','-q','cp',source,str(path)],check=True)
        with np.load(path) as d:return {k:d[k] for k in d.files}
    import tensorflow as tf
    try:
        tf.config.threading.set_inter_op_parallelism_threads(2)
        tf.config.threading.set_intra_op_parallelism_threads(2)
    except RuntimeError:
        pass  # The process-level TF thread limits still apply after TF context initialization.
    pool=[];labels=[]
    for record in tf.data.TFRecordDataset([source]).take(512):
        example=tf.train.Example.FromString(record.numpy())
        tokens=np.array(example.features.feature['input_ids'].int64_list.value,dtype=np.int32)
        if len(tokens)<4097:continue
        label=np.array(example.features.feature['labels'].int64_list.value,dtype=np.int32)
        if len(label)!=len(tokens):label=tokens.copy()
        pool.append(tokens[:4097]);labels.append(label[:4097])
    if len(pool)<count:raise RuntimeError('Not enough full-length sequences')
    chosen=np.random.default_rng(seed).permutation(len(pool))[:count]
    result={'tokens':np.stack([pool[i] for i in chosen]),'labels':np.stack([labels[i] for i in chosen]),'source_indices':chosen}
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**result)
    return result


def hash_sequence(x):return hashlib.sha256(np.asarray(x,dtype=np.int32).tobytes()).hexdigest()


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


def observed(model,params,batch,rng):
    capture.TOKENS = batch['inputs']
    result,collections=model.apply(params,batch['inputs'],batch['positions'],
        decoder_target_tokens=batch['targets'],decoder_segment_ids=batch['segments'],
        decoder_target_mask=batch['mask'],enable_dropout=False,rngs={'dropout':rng,'params':rng,'aqt':rng},
        mutable=['intermediates'])
    weights=batch['mask'];loss=jnp.sum(result[0]*weights,axis=-1)/jnp.maximum(jnp.sum(weights,axis=-1),1)
    # All values have already been sliced on device by the observers.
    flat=flatten_dict(collections['intermediates'])
    selected={'/'.join(k):v for k,v in flat.items() if k[-1].startswith('stream_')}
    return loss,selected


def unwrap(x):
    while isinstance(x,(list,tuple)) and len(x)==1:x=x[0]
    return np.asarray(x,dtype=np.float32)


def save_capture(path,arrays):
    # Low compression avoids spending most of a short TPU probe in CPU deflate.
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as archive:
        for name,value in arrays.items():
            with archive.open(name+'.npy','w',force_zip64=True) as item:
                np.lib.format.write_array(item,np.asarray(value),allow_pickle=False)


def upload(output,destination):
    if destination:
        subprocess.run(['gsutil','-m','rsync','-r',str(output),destination],check=True)


def main():
    p=argparse.ArgumentParser()
    for key in ['experiment','checkpoint','training-config','label','output','gcs-output','cohort-reference']:p.add_argument('--'+key,required=True)
    p.add_argument('--expected-parameters',required=True,type=int)
    p.add_argument('--length',type=int,default=2048);p.add_argument('--batch-size',type=int,default=4);p.add_argument('--sequences',type=int,default=32);p.add_argument('--seed',type=int,default=20261021)
    p.add_argument('--cohort',default='/tmp/bam_stream_probe/vo_cohort32.npz');p.add_argument('--cohort-source',default='gs://newproject-1-llm_projects_us-east5/log/diagnostics/mengqy/cross_scale_20261025/read_gradient/cohort32.npz')
    a=p.parse_args();assert a.sequences==32 and a.length==2048 and a.batch_size==4
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);Path('/tmp/bam_stream_probe',a.label).mkdir(parents=True,exist_ok=True);start=time.time();cfg=config_for(a)
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
    data=cohort(a.cohort,a.cohort_source,a.seed,32);hashes=[hash_sequence(x[:2049]) for x in data['tokens']]
    reference=json.loads(Path(a.cohort_reference).read_text());assert hashes==reference['sequence_hashes']
    assert np.all(data['labels'][:,1:2049]>=0)
    batches=[]
    for i in range(0,32,4):
        tokens=data['tokens'][i:i+4];shape=(4,2048)
        batches.append(dict(inputs=jnp.asarray(tokens[:,:2048]),targets=jnp.asarray(tokens[:,1:2049]),mask=jnp.ones(shape,jnp.int32),segments=jnp.ones(shape,jnp.int32),positions=jnp.broadcast_to(jnp.arange(2048),shape)))
    capture.ACTIVE=False;native_fn=jax.jit(lambda p,b:plain(model,p,b,rng));baseline=[]
    for b in batches:
        with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):baseline.extend(np.asarray(jax.device_get(native_fn(params,b))).tolist())
    baseline=np.asarray(baseline);assert baseline.shape==(32,) and np.isfinite(baseline).all()
    capture.install();capture.ACTIVE=True;observed_fn=jax.jit(lambda p,b:observed(model,p,b,rng));losses=[];arrays={};indices={};first=None
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
    meta=dict(label=a.label,experiment=a.experiment,checkpoint=a.checkpoint,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),parameter_count=count,sequences=32,length=2048,batch_size=4,sequence_hashes=hashes,query_positions=capture.positions(2048).tolist(),pair_names=list(capture.PAIR_NAMES),pair_dimensions=[int(cfg.head_dim)]*4+[int(cfg.bam_v)],moment_names=list(capture.MOMENT_NAMES),attention_names=['self_probability','argmax_self'],closure_relative_RMS_max=closures,native_observer_max_CE_drift=float(np.max(np.abs(np.asarray(losses)-baseline))),repeat_first_batch_exact=True,config={key:value for key,value in cfg.get_keys().items() if key in critical or key in ['bam_v'] or key.startswith(('bam_','pythia_','mudd_'))},training_config_verified_keys=checked,elapsed_seconds=time.time()-start,jax_version=jax.__version__,devices=[str(d) for d in jax.devices()],definition='Native frozen forward. Same-token/layer/head V/O moments at64fixedqueries>=64, beforeW_o. Dynamic/static parts via native shadow read or actual XLA component sites; nativeoutputs unchanged. Zeroenergy vectors excluded on host. Keypair is actual dynamicaddress includingprebias, beforeRMS normalization (positive scalar does not changeangle). Separate V/O slab order VthenO. MHA/O pair is different from rawV/O. No optimizer/update/checkpointwrite.')
    (out/'metadata.json').write_text(json.dumps(meta,indent=2,default=str)+'\n');subprocess.run(['gsutil','-m','rsync','-r',str(out),a.gcs_output],check=True)
    (out/'complete.json').write_text(json.dumps(dict(label=a.label,sequences=32))+'\n');subprocess.run(['gsutil','-q','cp',str(out/'complete.json'),a.gcs_output+'/complete.json'],check=True);print('VO_SIMILARITY_COMPLETE',a.label,flush=True)
if __name__=='__main__':main()
