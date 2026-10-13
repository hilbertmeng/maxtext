"""Fixed200-step local teacher fit of three R256 banks into shared R384.

Uses already captured32-sequence cohort: first16 fit, last16 held out.
Checkpoint is read-only. No full-model training or validation-step selection.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

helper=Path(os.environ.get('R384_JOINT_RUNNER',str(Path(__file__).with_name('diagnose_mediumprop_joint_r384.py'))))
spec=importlib.util.spec_from_file_location('joint_r384',helper)
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
from absl import app
from flax import linen as nn
from flax.traverse_util import flatten_dict,unflatten_dict
from flax.linen import partitioning as nn_partitioning
import pyconfig,train,max_utils
from input_pipeline.input_pipeline_interface import create_data_iterator

def run(config):
    if not config.only_eval:raise ValueError('Read-only checkpoint probe')
    config.get_keys()['enable_checkpointing']=False
    source=Path(os.environ.get('R384_CAPTURE_DIR','/tmp/xd-joint-r384-sota13500'))
    output=g.t.output;output.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    previous=json.loads((source/'joint-r384-report.json').read_text())
    rng,_,manager,mesh,model,_,tx=train.setup_mesh_and_model(config)
    tf.random.set_seed(int(config.data_shuffle_seed))
    data,eval_data=create_data_iterator(config,mesh)
    state,_,_,_=max_utils.setup_training_state(model,data,tx,config,rng,mesh,manager)
    flat=flatten_dict(state.params);batches=[];hashes=[]
    for _ in range(8):
        b=next(eval_data);batches.append(b)
        hashes.extend(hashlib.sha256(x.tobytes()).hexdigest() for x in np.asarray(jax.device_get(b['inputs'])))
    if hashes!=previous['sequence_hashes']:raise ValueError('Cohort hash mismatch')
    cap=np.load(source/'joint-r384-capture.npz');saved=np.load(source/'joint-r384-variants.npz')
    vf={tuple(k.split('/',1)[1].split('/')):jnp.asarray(saved[k]) for k in saved.files if k.startswith('triple_select384/')}
    block='decoder/layers/local_1/block';attn=block+'/self_attention'
    rd=('params',*attn.split('/'),'full_m_read_down','kernel')
    ad=('params',*attn.split('/'),'P_loc_down','kernel')
    md=('params',*block.split('/'),'mlp_address_down','kernel')
    for path in (ad,md):np.testing.assert_array_equal(vf[rd[1:]],vf[path[1:]])
    xpre=cap[attn+'/full_m_read_down/r256_x'].reshape(6,-1,1200)
    xpost=cap[block+'/mlp_address_down/r256_x'].reshape(6,-1,1200)
    valid=cap['valid'];nfit=16*valid.shape[1]
    ports=[(attn,k) for k in g.READ_UPS]+[(attn,'P_loc_up'),(block,'mlp_address_up')]
    kernels=[];biases=[];raw=[];bias_offsets=[]
    hidden=jax.jit(lambda x,w:nn.gelu(jnp.asarray(x,config.dtype)@jnp.asarray(w,config.dtype)).astype(jnp.float32))
    initial_down=np.moveaxis(np.asarray(vf[rd[1:]]),1,0)
    for head,(scope,up) in enumerate(ports):
        kp=('params',*scope.split('/'),up,'kernel')
        kernels.append(np.moveaxis(np.asarray(vf[kp[1:]]),1,0).reshape(6,384,16,32))
        raw.append(cap[scope+'/'+up+'/r256_y'].reshape(6,-1,16,32))
        if head<3:
            bp=('params',*attn.split('/'),'general_'+g.READ_UPS[up]+'_pre_bias')
            old=np.moveaxis(np.asarray(flat[bp]),1,0).reshape(6,16,32)
            new=[]
            for scan in range(6):
                h=np.asarray(jax.device_get(hidden(xpre[scan].reshape(32,-1,1200),initial_down[scan])))
                target=cap[scope+'/'+up+'/r256_y'][scan]
                k,b=g.affine(h,target,valid)
                np.testing.assert_allclose(k.reshape(384,16,32),kernels[-1][scan],rtol=1e-6,atol=1e-6)
                new.append(old[scan]+b.reshape(16,32))
            biases.append(np.array(new));bias_offsets.append(old)
        else:
            bp=kp[:-1]+('bias',)
            biases.append(np.moveaxis(np.asarray(vf[bp[1:]]),1,0));bias_offsets.append(np.zeros((6,16,32),np.float32))
    p={'down':jnp.asarray(initial_down),'up':jnp.asarray(np.stack(kernels,axis=2)),
       'bias':jnp.asarray(np.stack(biases,axis=1))}
    scale=np.ones((6,5,16,32),np.float32);norm_bias=np.zeros_like(scale)
    ns=('params',*attn.split('/'),'write_address_norm')
    if ns+('scale',) in flat:scale[:,3:]=1+np.moveaxis(np.asarray(flat[ns+('scale',)]),1,0)[:,None]
    if ns+('bias',) in flat:norm_bias[:,3:]=np.moveaxis(np.asarray(flat[ns+('bias',)]),1,0)[:,None]
    readstats=jnp.float32 if config.bam_read_rms_statistics_dtype=='float32' else config.dtype
    writestats=jnp.float32 if config.bam_write_rms_statistics_dtype=='float32' else config.dtype
    eps=float(config.normalization_layer_epsilon)
    def normalized(y,d):
        def rms(a,dtype):
            a=a.astype(dtype)
            return (a*jax.lax.rsqrt(jnp.mean(a*a,axis=-1,keepdims=True)+eps)).astype(config.dtype)
        y=jnp.concatenate((rms(y[:,:,:3],readstats),rms(y[:,:,3:],writestats)),axis=2)
        return (y*d['scale'][:,None]+d['norm_bias'][:,None]).astype(jnp.float32)
    d={'pre':jnp.asarray(xpre),'post':jnp.asarray(xpost),'scale':jnp.asarray(scale,config.dtype),
       'norm_bias':jnp.asarray(norm_bias,config.dtype)}
    raw=jnp.asarray(np.stack(raw,axis=2),config.dtype)+jnp.asarray(np.stack(bias_offsets,axis=1),config.dtype)[:,None]
    d['target']=normalized(raw,d)
    d['content_a']=jnp.asarray(cap[attn+'/r256_attn_content'].reshape(6,-1,16,75))
    d['content_m']=jnp.asarray(cap[attn+'/r256_mlp_content'].reshape(6,-1,16,75))
    d['dm_a']=jnp.einsum('lbhn,lbhv->lbnv',d['content_a'],d['target'][:,:,3])
    d['dm_m']=jnp.einsum('lbhn,lbhv->lbnv',d['content_m'],d['target'][:,:,4])
    d['energy_a']=jnp.mean(d['dm_a'][:,:nfit]**2,axis=(1,2,3))
    d['energy_m']=jnp.mean(d['dm_m'][:,:nfit]**2,axis=(1,2,3))
    def loss(p,d,indices):
        w=p['down'].astype(config.dtype);k=p['up'].astype(config.dtype)
        a=nn.gelu(jnp.einsum('lbd,ldr->lbr',d['pre'][:,indices].astype(config.dtype),w))
        b=nn.gelu(jnp.einsum('lbd,ldr->lbr',d['post'][:,indices].astype(config.dtype),w))
        ya=jnp.einsum('lbr,lrhnv->lbhnv',a,k[:,:,:4])
        ym=jnp.einsum('lbr,lrhnv->lbhnv',b,k[:,:,4:])
        y=normalized(jnp.concatenate((ya,ym),axis=2)+p['bias'].astype(config.dtype)[:,None],d)
        read=jnp.mean((y[:,:,:3]-d['target'][:,indices,:3])**2)
        dma=jnp.einsum('lbhn,lbhv->lbnv',d['content_a'][:,indices],y[:,:,3])
        dmm=jnp.einsum('lbhn,lbhv->lbnv',d['content_m'][:,indices],y[:,:,4])
        la=jnp.mean(jnp.mean((dma-d['dm_a'][:,indices])**2,axis=(1,2,3))/jnp.maximum(d['energy_a'],1e-12))
        lm=jnp.mean(jnp.mean((dmm-d['dm_m'][:,indices])**2,axis=(1,2,3))/jnp.maximum(d['energy_m'],1e-12))
        return (read+la+lm)/3
    optimizer=optax.multi_transform({'down':optax.adam(1e-4),'up':optax.adam(3e-4)},
                                   {'down':'down','up':'up','bias':'up'})
    opt_state=optimizer.init(p)
    @jax.jit
    def step(p,opt_state,d,indices):
        value,grad=jax.value_and_grad(loss)(p,d,indices)
        updates,opt_state=optimizer.update(grad,opt_state,p)
        return optax.apply_updates(p,updates),opt_state,value
    score=jax.jit(loss)
    fit_indices=np.flatnonzero(valid[:16].reshape(-1));held_indices=np.flatnonzero(valid[16:].reshape(-1))+nfit
    initial={name:float(score(p,d,jnp.asarray(idx))) for name,idx in [('fit',fit_indices),('heldout',held_indices)]}
    initial_params=p;generator=np.random.default_rng(20261013)
    for i in range(200):
        indices=jnp.asarray(generator.choice(fit_indices,128,replace=True))
        p,opt_state,value=step(p,opt_state,d,indices)
        if i%50==0:print('TRIPLE_REFIT_STEP',i,float(value),flush=True)
    final={name:float(score(p,d,jnp.asarray(idx))) for name,idx in [('fit',fit_indices),('heldout',held_indices)]}
    def variant(pp):
        pf=dict(flat);vv=dict(vf);down=np.asarray(pp['down']);up=np.asarray(pp['up']);bias=np.asarray(pp['bias'])
        for scan in range(6):
            for dp in (rd,ad,md):vv[dp[1:]]=vv[dp[1:]].at[:,scan].set(down[scan])
            for head,(scope,name) in enumerate(ports):
                kp=('params',*scope.split('/'),name,'kernel');shape=vv[kp[1:]][:,scan].shape
                vv[kp[1:]]=vv[kp[1:]].at[:,scan].set(up[scan,:,head].reshape(shape))
                if head<3:
                    bp=('params',*attn.split('/'),'general_'+g.READ_UPS[name]+'_pre_bias')
                    pf[bp]=pf[bp].at[:,scan].set(bias[scan,head].reshape(pf[bp][:,scan].shape))
                else:
                    bp=kp[:-1]+('bias',);vv[bp[1:]]=vv[bp[1:]].at[:,scan].set(bias[scan,head])
        for path,value in vv.items():pf[('params',)+path[:-1]+('r384_'+path[-1],)]=value
        return unflatten_dict(pf)
    baseline=dict(flat)
    for path,value in g.padded_ports(state.params).items():baseline[('params',)+path[:-1]+('r384_'+path[-1],)]=value
    forward=jax.jit(lambda params,batch:g.t.forward(model,params,batch,rng,False)[0])
    outcomes={}
    for name,params in [('baseline',unflatten_dict(baseline)),('initial',variant(initial_params)),('refit200',variant(p))]:
        metrics=[]
        for batch in batches[4:]:
            with mesh,nn_partitioning.axis_rules(config.logical_axis_rules):metrics.append(jax.device_get(forward(params,batch)))
        seq=np.concatenate([m['sequence_loss'] for m in metrics]);weights=np.concatenate([m['sequence_weights'] for m in metrics])
        outcomes[name]={'loss':float(np.average(seq,weights=weights)),'sequence_loss':seq.tolist()}
        if name!='baseline':
            delta=seq-np.asarray(outcomes['baseline']['sequence_loss'])
            outcomes[name].update(delta=float(np.average(delta,weights=weights)),paired_sem=float(delta.std(ddof=1)/np.sqrt(len(delta))))
        print('TRIPLE_REFIT_LOSS',name,json.dumps(outcomes[name]),flush=True)
    report={'checkpoint':config.load_parameters_path,'sequence_hashes':hashes,'steps':200,
            'lr_down':1e-4,'lr_up_bias':3e-4,'batch_positions':128,'fit_sequences':16,'heldout_sequences':16,
            'objective':'Equal three-stream mean: normalized read directions and relative attention/MLP dM errors',
            'teacher_forced_initial':initial,'teacher_forced_final':final,'losses':outcomes,
            'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'seconds':time.perf_counter()-start}
    (output/'triple-refit200-report.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(output/'triple-refit200-ports.npz',**{name:np.asarray(value) for name,value in p.items()})
    print('TRIPLE_REFIT_COMPLETE',report['seconds'],flush=True)

def main(argv):run(pyconfig.initialize(argv))
if __name__=='__main__':app.run(main)
