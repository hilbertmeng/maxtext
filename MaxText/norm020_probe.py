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

import checkpointing,exp,max_utils,pyconfig
from layers.models import Transformer
import norm020_capture as capture
from norm020_cases import plan,digest

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

def save_npz(path,**arrays):
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for key,value in arrays.items():
            with z.open(key+'.npy','w',force_zip64=True) as f:np.lib.format.write_array(f,np.asarray(value),allow_pickle=False)

def read_remote(uri):
    p=subprocess.run(['gsutil','cat',uri],capture_output=True,text=True,timeout=60)
    if p.returncode==0:return json.loads(p.stdout)
    if any(x in p.stderr for x in ['No URLs matched','matched no objects','NotFound','404']):return None
    raise RuntimeError('Cannot inspect completed artifact: '+p.stderr[-500:])

def publish_pair(array_file,marker,metadata,destination):
    subprocess.run(['gsutil','-q','cp',str(array_file),destination+'/'+array_file.name],check=True)
    marker.write_text(json.dumps(metadata,indent=2,default=str)+'\n')
    subprocess.run(['gsutil','-q','cp',str(marker),destination+'/'+marker.name],check=True)

def unpack(collection,layers):
    flat={name:np.asarray(value[0]) for name,value in collection.items()};result={}
    for name,index in flat.items():
        if not name.endswith('/readout_layer_index'):continue
        prefix=name.rsplit('/',1)[0]+'/'
        for pos in np.ndindex(index.shape):
            i=int(index[pos]);assert i not in result
            result[i]={k[len(prefix)+8:]:v[pos] for k,v in flat.items() if k.startswith(prefix) and k[len(prefix):].startswith('readout_')}
    assert set(result)==set(range(layers))
    return {key:np.stack([result[i][key] for i in range(layers)],axis=1) for key in result[0] if key!='layer_index'}

def main():
    p=argparse.ArgumentParser()
    for key in ['manifest','family','training-config','cohort-reference','output','gcs-output']:p.add_argument('--'+key,required=True)
    a=p.parse_args();manifest=json.loads(Path(a.manifest).read_text());model_info=manifest['models'][a.family]
    a.experiment=model_info['experiment'];a.expected_parameters=model_info['parameter_count'];a.label='norm020_'+a.family;a.length=2048;a.batch_size=4;a.seed=20261021
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();assert source==manifest['commit'];outroot=Path(a.output);outroot.mkdir(parents=True,exist_ok=True);Path('/tmp/bam_stream_probe',a.label).mkdir(parents=True,exist_ok=True)
    cfg,critical,checked,mesh,model,rng,abstract=setup(a)
    assert cfg.bam_attention_negative_diagonal and cfg.bam_pallas_core and cfg.bam_pallas_fused_inputs and cfg.bam_mlp_write_every==1
    assert cfg.num_decoder_layers==cfg.num_query_heads==32 and cfg.head_dim==80
    data=cohort('/tmp/bam_stream_probe/noself_cohort32.npz',manifest['cohort_source'],a.seed,32);hashes=[hash_sequence(x[:2049]) for x in data['tokens']];assert hashes==manifest['sequence_hashes']==json.loads(Path(a.cohort_reference).read_text())['sequence_hashes']
    batches=[]
    for i in range(0,32,4):
        tokens=data['tokens'][i:i+4];shape=(4,2048);batches.append(dict(inputs=jnp.asarray(tokens[:,:2048]),targets=jnp.asarray(tokens[:,1:2049]),mask=jnp.ones(shape,jnp.int32),segments=jnp.ones(shape,jnp.int32),positions=jnp.broadcast_to(jnp.arange(2048),shape)))
    def native(p,b):return model.apply(p,b['inputs'],b['positions'],decoder_target_tokens=b['targets'],decoder_target_mask=b['mask'],decoder_segment_ids=b['segments'],enable_dropout=False,rngs={'params':rng,'dropout':rng,'aqt':rng})[0]
    native_fn=jax.jit(native)
    def controlled(p,b,controls,observe):
        capture.ACTIVE=True;capture.CONTROLS=controls;capture.OBSERVE=observe
        try:
            result=model.apply(p,b['inputs'],b['positions'],decoder_target_tokens=b['targets'],decoder_target_mask=b['mask'],decoder_segment_ids=b['segments'],enable_dropout=False,rngs={'params':rng,'dropout':rng,'aqt':rng},**({'mutable':['intermediates']} if observe else {}))
            if not observe:return result[0]
            output,collections=result;raw={'/'.join(k):v for k,v in flatten_dict(collections['intermediates']).items() if k[-1].startswith('readout_')}
            return output[0],raw
        finally:capture.ACTIVE=False;capture.OBSERVE=False;capture.CONTROLS=None
    ones=jnp.ones((32,4),jnp.float32);ablate_fn=jax.jit(lambda p,b,c:controlled(p,b,c,False));observe_fn=jax.jit(lambda p,b:controlled(p,b,ones,True));installed=False
    def loss_batches(p,controls=None):
        result=[]
        for b in batches:
            with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):value=native_fn(p,b) if controls is None else ablate_fn(p,b,controls)
            result.append(np.asarray(jax.device_get(value),np.float32))
        value=np.concatenate(result);assert value.shape==(32,2048) and np.isfinite(value).all();return value
    def validate(marker,common):
        for key in ['source_commit','checkpoint','sequence_hashes','plan_digest','experiment']:assert marker[key]==common[key],('Completed artifact mismatch',key)
    previous_native=None
    for step in model_info['steps']:
        cases=plan(step,manifest['main_step']);destination=a.gcs_output+'/'+a.family+'/'+str(step);folder=outroot/a.family/str(step);folder.mkdir(parents=True,exist_ok=True)
        checkpoint=manifest['checkpoint_roots'][a.family]+'/'+str(step)+'/items'
        common=dict(source_commit=source,training_commit=model_info['training_commit'],experiment=a.experiment,checkpoint=checkpoint,step=step,family=a.family,sequence_hashes=hashes,plan_digest=digest(cases),parameter_count=a.expected_parameters,layers=32,heads=32,head_dim=80,query_positions=capture.positions(2048).tolist(),gate_names=list(capture.GATE_NAMES))
        completed=read_remote(destination+'/complete.json')
        if completed is not None:validate(completed,common);print('NOSELF_PREVIOUSLY_COMPLETE',a.family,step,flush=True);continue
        with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):params=checkpointing.load_params_from_path(checkpoint,abstract.params)
        assert sum(v.size for v in jax.tree.leaves(params))==a.expected_parameters
        gamma_leaves=[v for k,v in flatten_dict(params).items() if k[-1]=='local_o_dynamic_norm_scale']
        assert sum(v.size for v in gamma_leaves)==(65536 if a.family=='norm020' else 0)
        native_loss=loss_batches(params)
        if a.family=='full' and step==12000:
            reference=Path('/tmp/bam_stream_probe/historical_full12k.npz')
            if not reference.exists():subprocess.run(['gsutil','-q','cp',manifest['historical_capture'],str(reference)],check=True)
            with np.load(reference) as z:
                np.testing.assert_array_equal(z['inputs'],data['tokens'][:,:2048])
                drift=float(np.max(np.abs(native_loss.mean(1)-z['native_loss'].mean(1))))
                assert drift<.005,('Historical Full12k runtime parity failed',drift)
                common['historical_native_CE_max_sequence_difference']=drift
        if previous_native is not None:assert not np.array_equal(native_loss,previous_native),"Different checkpoints produced identical losses: inspect parameter reuse"
        previous_native=native_loss.copy()
        if not installed:capture.install();installed=True
        neutral=loss_batches(params,ones);neutral_drift=float(np.max(np.abs(neutral.mean(1)-native_loss.mean(1))));assert neutral_drift<.005,('Neutral changed native CE',neutral_drift)
        common['neutral_native_CE_max_difference']=neutral_drift
        captured=read_remote(destination+'/capture.json')
        if captured is None:
            arrays={};observed=[];first=None
            for i,b in enumerate(batches):
                with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):loss,raw=jax.device_get(observe_fn(params,b))
                observed.append(np.asarray(loss));group=unpack(raw,32)
                for key,value in group.items():arrays.setdefault(key,[]).append(value)
                if i==0:first=(loss,raw)
            merged={key:np.concatenate(value,axis=0) for key,value in arrays.items()};observed=np.concatenate(observed);drift=float(np.max(np.abs(observed.mean(1)-native_loss.mean(1))));assert drift<.005
            assert merged['pre_heads'].shape==(32,32,32,5) and merged['gates'].shape==(32,32,64,32,6)
            assert all(np.isfinite(x).all() for x in merged.values()) and merged['empty_history_mha_max'].max()==0
            assert merged['gates'][...,:4].min()>=0 and merged['gates'][...,:4].max()<=1
            closure=merged['closure'];post_error=float(np.sqrt(closure[...,0]/np.maximum(closure[...,1],1e-30)).max());assert post_error<.02
            with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):again,repeat=jax.device_get(observe_fn(params,batches[0]))
            np.testing.assert_array_equal(first[0],again)
            for key,value in raw.items():assert len(value)==1
            for key,value in first[1].items():np.testing.assert_array_equal(value[0],repeat[key][0])
            file=folder/'capture.npz';save_npz(file,**merged,inputs=data['tokens'][:,:2048],targets=data['tokens'][:,1:2049],native_loss=native_loss,neutral_loss=neutral,observed_loss=observed)
            meta=dict(**common,observer_native_CE_max_difference=drift,projection_max_relative_RMS=post_error,repeat_exact=True,config={key:value for key,value in cfg.get_keys().items() if key in critical or key.startswith(('bam_','pythia_','mudd_'))},verified_config_keys=checked,jax_version=jax.__version__,devices=[str(x) for x in jax.devices()],gate_definition='First4 are native sigmoid outputs; last2 are actual1.1*static_sigmoid/.99 multipliers. Samples64queries>=64; not binary gating.',projection_definition='PreW_o head moments actual nativeA/Mo. Posthead energy uses per-head W_o Gram; aggregate post-W_o projects allheads together. FP32 linear decomposition from nativeBF16 operands/kernel; bias separate. Closure against native output checked.',ablation_scope='Combined: independent vector projection and attention-write content controls both0. Vector-only preserves each current call computed directattentionwrite content, not the unperturbed whole-model trace; downstream/MLP writes remain live. Write-only preserves current attentionvector. Outputbias retained once.')
            publish_pair(file,folder/'capture.json',meta,destination);del merged,arrays,observed,first,raw,repeat
            print('NOSELF_CAPTURE_COMPLETE',a.family,step,'CEdrift',drift,'projection_relative_RMS',post_error,flush=True)
        else:validate(captured,common)
        # Every chunk carries its own same-graph neutral reference, including across retries.
        for chunk,start in enumerate(range(0,len(cases),8)):
            selected=cases[start:start+8];stem=f'cases_{chunk:03d}';old=read_remote(destination+'/'+stem+'.json')
            if old is not None:validate(old,common);assert old['cases']==selected;continue
            losses=[]
            for case in selected:
                values=loss_batches(params,jnp.asarray(case['controls'],jnp.float32));losses.append(values)
                print('NOSELF_CASE',a.family,step,case['name'],'deltaCE',float((values-neutral).mean()),flush=True)
            with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):again=np.asarray(ablate_fn(params,batches[0],ones))
            np.testing.assert_array_equal(again,neutral[:4])
            file=folder/(stem+'.npz');save_npz(file,losses=np.stack(losses),neutral=neutral)
            publish_pair(file,folder/(stem+'.json'),dict(**common,cases=selected,repeat_neutral_exact=True),destination)
            print('NOSELF_CHUNK_COMPLETE',a.family,step,chunk,len(selected),flush=True)
        (folder/'complete.json').write_text(json.dumps(dict(**common,total_cases=len(cases)),indent=2)+'\n');subprocess.run(['gsutil','-q','cp',str(folder/'complete.json'),destination+'/complete.json'],check=True)
        print('NOSELF_CHECKPOINT_COMPLETE',a.family,step,len(cases),flush=True);del params,neutral,native_loss
    print('NOSELF_MODEL_COMPLETE',a.family,flush=True)
if __name__=='__main__':main()
