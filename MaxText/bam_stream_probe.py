"""Matched-cohort, checkpoint-only BAM stream measurements and M-write ablations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

from flax import linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict
import jax
import jax.numpy as jnp
import numpy as np

import bam_stream_capture as capture
import checkpointing
import exp
import max_utils
import pyconfig
from layers.models import Transformer


def cohort(path, source, seed, count):
    path = Path(path)
    if path.exists():
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
    result,collections=model.apply(params,batch['inputs'],batch['positions'],
        decoder_target_tokens=batch['targets'],decoder_segment_ids=batch['segments'],
        decoder_target_mask=batch['mask'],enable_dropout=False,rngs={'dropout':rng,'params':rng,'aqt':rng},
        mutable=['intermediates'])
    weights=batch['mask'];loss=jnp.sum(result[0]*weights,axis=-1)/jnp.maximum(jnp.sum(weights,axis=-1),1)
    # All values have already been sliced on device by the observers.
    flat=flatten_dict(collections['intermediates'])
    selected={'/'.join(k):v for k,v in flat.items() if k[-1].startswith('stream_')}
    return loss,selected


def variant(params,name):
    flat=flatten_dict(params);changed={};count=0
    for path,value in flat.items():
        leaf=path[-1]
        apply=(name=='no_embedding_M_write' and leaf=='emb_gw_b0'
            or name=='no_attention_M_write' and leaf=='gw_b0'
            or name=='no_mlp_M_write' and leaf=='mlp_write_gate_bias')
        read_bias=name=='no_M_read' and leaf in ('W_lq_gate_b0','W_lk_gate_b0','W_R_gate_b0','W_lv_gate_b0')
        read_static=name=='no_M_read' and leaf in ('static_q_key','static_k_key','static_v_key','static_o_key')
        if apply or read_bias:changed[path]=jnp.full_like(value,-80);count+=1
        elif read_static:changed[path]=jnp.zeros_like(value);count+=1
        else:changed[path]=value
    if count==0:raise RuntimeError('Ablation matched no parameters: '+name)
    return unflatten_dict(changed),count


def unwrap(x):
    while isinstance(x,(list,tuple)) and len(x)==1:x=x[0]
    return np.asarray(x,dtype=np.float32)


def upload(output,destination):
    if destination:
        subprocess.run(['gsutil','-m','rsync','-r',str(output),destination],check=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--experiment',required=True);parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--label',required=True);parser.add_argument('--length',type=int,default=2048)
    parser.add_argument('--sequences',type=int,default=32);parser.add_argument('--seed',type=int,default=20261021)
    parser.add_argument('--cohort',default='/tmp/bam_stream_probe/cohort.npz')
    parser.add_argument('--cohort-source',default='gs://newproject-1-common_datasets_us-east5/pythia_pile_idxmaps_tfrecord_4096/pile.tfrecord.b0')
    parser.add_argument('--output',required=True);parser.add_argument('--gcs-output',default='')
    parser.add_argument('--no-ablations',action='store_true');parser.add_argument('--skip-raw',action='store_true')
    parser.add_argument('--batch-size',type=int,default=1)
    args=parser.parse_args()
    if args.sequences % args.batch_size:
        raise ValueError('Sequence count must be divisible by diagnostic batch size')
    Path('/tmp/bam_stream_probe',args.label).mkdir(parents=True,exist_ok=True)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    start=time.time();capture.install();capture.ACTIVE=False
    cfg=config_for(args)
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes)
    model=Transformer(cfg,mesh,quant=None);rng=jax.random.key(20261021)
    with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
        # Old MaxText unboxes ShapeDtypeStruct under active axis rules. Flax's
        # default unbox applies a real sharding primitive to these abstract leaves.
        original_unbox=max_utils.unbox_logicallypartioned
        max_utils.unbox_logicallypartioned=lambda tree: jax.tree.map(
            lambda leaf: leaf.unbox(apply_constraint=False) if isinstance(leaf,nn.spmd.LogicallyPartitioned) else leaf,
            tree,is_leaf=lambda leaf:isinstance(leaf,nn.spmd.LogicallyPartitioned))
        try:
            abstract,_,_=max_utils.get_abstract_state(model,None,cfg,rng,mesh,is_training=False)
        finally:
            max_utils.unbox_logicallypartioned=original_unbox
        params=checkpointing.load_params_from_path(args.checkpoint,abstract.params)
    shapes={'/'.join(k):list(v.shape) for k,v in flatten_dict(params).items()}
    metadata=dict(experiment=args.experiment,checkpoint=args.checkpoint,label=args.label,length=args.length,
        seed=args.seed,cohort_source=args.cohort_source,sequences=args.sequences,
        config={k:('<redacted>' if any(marker in k.lower() for marker in ('access_token','api_key','password','credential','secret')) else v) for k,v in cfg.get_keys().items()},
        parameter_count=sum(v.size for v in jax.tree.leaves(params)),parameter_shapes=shapes,
        setup_seconds=time.time()-start,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        token_positions=capture.positions(args.length).tolist(),devices=[str(d) for d in jax.devices()],
        jax_version=jax.__version__,device_batch=args.batch_size)
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2,default=str)+'\n')
    data=cohort(args.cohort,args.cohort_source,args.seed,args.sequences)
    plain_fn=jax.jit(lambda p,b:plain(model,p,b,rng))
    batches=[];hashes=[];losses=[];measurements=[]
    for i in range(0,args.sequences,args.batch_size):
        seq=data['tokens'][i:i+args.batch_size,:args.length+1]
        labels=data['labels'][i:i+args.batch_size,1:args.length+1]
        batch=dict(inputs=jnp.asarray(seq[:,:-1]),targets=jnp.asarray(seq[:,1:]),
            positions=jnp.broadcast_to(jnp.arange(args.length),(args.batch_size,args.length)),segments=jnp.ones((args.batch_size,args.length),jnp.int32),
            mask=jnp.asarray(labels>=0,dtype=jnp.int32))
        if int(seq.max())>=cfg.vocab_size:raise RuntimeError('Out-of-vocabulary shared cohort')
        batches.append(batch);hashes.extend(hash_sequence(row) for row in seq)
    with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
        baseline=np.asarray(jax.device_get(plain_fn(params,batches[0])))
    capture.ACTIVE=True
    observed_fn=jax.jit(lambda p,b:observed(model,p,b,rng))
    for i,batch in enumerate(batches):
        t=time.time()
        with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
            loss,raw=observed_fn(params,batch)
            loss,raw=jax.device_get((loss,raw))
        if i==0:
            difference=float(np.max(np.abs(np.asarray(loss)-baseline)))
            if difference>2e-3:raise RuntimeError(f'Instrumentation changed loss by {difference}')
            metadata['instrumentation_loss_abs_difference']=difference
        for within in range(args.batch_size):
            sequence_index=i*args.batch_size+within
            arrays={}
            for key,value in raw.items():
                arr=unwrap(value);metric=key.split('/')[-1]
                static=(metric in ('stream_mlp_write_bias','stream_embedding_gate_bias','stream_m_shift_bias',
                        'stream_read_key_epsilon','stream_read_key_scale')
                        or metric.startswith('stream_static_') and metric.endswith('_key'))
                if not static:
                    axis=1 if 'layers' in key.split('/') else 0
                    if arr.shape[axis]!=args.batch_size:raise RuntimeError((key,arr.shape,args.batch_size))
                    arr=np.take(arr,[within],axis=axis)
                arrays[key.replace('/','__')]=arr
            arrays['inputs']=np.asarray(batch['inputs'][within:within+1])
            arrays['mask']=np.asarray(batch['mask'][within:within+1])
            np.savez_compressed(output/f'sequence_{sequence_index:03d}.npz',**arrays)
            losses.append(float(loss[within]))
            measurements.append(dict(sequence=sequence_index,loss=losses[-1],hash=hashes[sequence_index],seconds=time.time()-t))
        (output/'progress.json').write_text(json.dumps(dict(completed_sequences=len(losses),losses=losses,measurements=measurements),indent=2)+'\n')
        print('PROBE_SEQUENCE',args.label,'through',len(losses),'loss',float(np.mean(loss)),'arrays',len(arrays),'seconds',time.time()-t,flush=True)
    capture.ACTIVE=False
    variants={}
    if not args.no_ablations:
        for name in ['no_embedding_M_write','no_attention_M_write','no_mlp_M_write','no_M_read']:
            changed,changed_count=variant(params,name);values=[]
            for i,batch in enumerate(batches):
                with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
                    loss=np.asarray(jax.device_get(plain_fn(changed,batch)))
                values.extend(float(value) for value in loss)
            delta=np.array(values)-np.array(losses)
            variants[name]=dict(losses=values,paired_delta=delta.tolist(),mean_delta=float(delta.mean()),
                sem=float(delta.std(ddof=1)/np.sqrt(len(delta))),changed_tensors=changed_count)
            print('ABLATION',args.label,name,variants[name]['mean_delta'],flush=True)
    metadata['sequence_hashes']=hashes;metadata['losses']=losses;metadata['ablations']=variants
    metadata['elapsed_seconds']=time.time()-start
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2,default=str)+'\n')
    upload(output,args.gcs_output)
    (output/'complete.json').write_text(json.dumps(dict(label=args.label,sequences=len(losses),gcs_output=args.gcs_output))+'\n')
    if args.gcs_output:subprocess.run(['gsutil','cp',str(output/'complete.json'),args.gcs_output+'/complete.json'],check=True)
    print('PROBE_COMPLETE',args.label,output,flush=True)

if __name__=='__main__':main()
