"""Compress two R256 feature banks into one shared R384 on held-out data.

Uses the sibling frozen-source capture runner. Virtual projection kernels allow
rank changes without editing historical model code. All loss variants use one
compiled graph, and the original checkpoint remains untouched.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

argv=list(sys.argv)
helper=Path(os.environ.get('R256_TEACHER_RUNNER',str(Path(__file__).with_name('diagnose_mediumprop_r256_functional.py'))))
spec=importlib.util.spec_from_file_location('r256_teacher',helper)
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg
import tensorflow as tf
from absl import app
from flax import linen as nn
from flax.traverse_util import flatten_dict,unflatten_dict
from flax.linen import partitioning as nn_partitioning
from layers import linears,normalizations
import max_utils,pyconfig,train
from input_pipeline.input_pipeline_interface import create_data_iterator

RANK=384
METHODS=tuple(os.environ.get('R384_METHODS','select,pca').split(','))
SEQUENCES=int(os.environ.get('R384_SEQUENCES','32'))
FIT=int(os.environ.get('R384_FIT_SEQUENCES','16'))
GROUPS=tuple(os.environ.get('R384_GROUPS','local_0,local_1,fetch_2').split(','))
READ_UPS={'W_lq_c8_up':'q','W_lk_c8_up':'k','W_R_up':'vo'}
t.UPS.update(READ_UPS)
capturing_dense=linears.DenseGeneral.__call__
def virtual_dense(self, inputs):
    kernel=self.get_variable('params','r384_kernel')
    if kernel is None:return capturing_dense(self,inputs)
    x=jnp.asarray(inputs,self.dtype);w=jnp.asarray(kernel,self.dtype)
    y=jax.lax.dot_general(x,w,(((x.ndim-1,),(0,)),((),())),
                         precision=jax.lax.Precision(self.matmul_precision))
    if self.use_bias:
        bias=self.get_variable('params','r384_bias')
        if bias is None:bias=nn.unbox(self.variables['params']['bias'])
        y=y+jnp.asarray(bias,self.dtype)
    return y
linears.DenseGeneral.__call__=virtual_dense

def padded_ports(params,rank=RANK):
    """Zero padding is an identity control; real sharing is set separately."""
    virtual={}
    for path,value in flatten_dict(params).items():
        if path[-1]!='kernel' or path[-2] not in t.DOWNS | t.UPS:continue
        axis=-1 if path[-2] in t.DOWNS else 0
        padding=[(0,0)]*value.ndim;padding[axis]=(0,rank-value.shape[axis])
        virtual[path[1:]]=jnp.pad(value,padding)
        bp=path[:-1]+('bias',)
        flat=flatten_dict(params)
        if bp in flat:virtual[bp[1:]]=flat[bp]
    return virtual

def affine(h,target,valid):
    a=h[:FIT][valid[:FIT]].astype(np.float64)
    b=target[:FIT][valid[:FIT]].astype(np.float64).reshape(len(a),-1)
    ma,mb=a.mean(0),b.mean(0);a=a-ma;b=b-mb
    gram=a.T@a;ridge=1e-4*np.trace(gram)/len(gram)
    k=np.linalg.solve(gram+ridge*np.eye(len(gram)),a.T@b);bias=mb-ma@k
    shape=target.shape[2:]
    return k.astype(np.float32).reshape((h.shape[-1],)+shape),bias.astype(np.float32).reshape(shape)

def importance(up,target,valid):
    """Output sensitivity normalized by actual per-head pre-RMS energy."""
    y=target[:FIT][valid[:FIT]].astype(np.float64)
    energy=np.mean(y*y,axis=(0,-1),keepdims=False)
    sensitivity=np.sum(np.asarray(up,np.float64)**2,axis=-1)
    score=np.sum((sensitivity/np.maximum(energy,1e-8)).reshape(up.shape[0],-1),axis=1)
    return score/np.maximum(score.sum(),1e-12)

def joint_bank(wa,wb,xs,hs,score,valid,method,rank=RANK,third=None):
    """Selection preserves GELU units; PCA recombines important preactivations."""
    banks=np.concatenate((wa,wb) if third is None else (wa,wb,third),axis=1).astype(np.float64)
    scale=np.sqrt(np.maximum(score,1e-12))
    if method=='select':
        features=np.concatenate([h[:FIT][valid[:FIT]] for h in hs],axis=0).astype(np.float64)
        features-=features.mean(0)
        _,_,pivot=scipy.linalg.qr(features*scale,mode='economic',pivoting=True,
                                 overwrite_a=True,check_finite=False)
        chosen=np.sort(pivot[:rank])
        return banks[:,chosen].astype(np.float32),{'units_from_a':int(np.sum(chosen<wa.shape[1])),
            'units_from_b':int(np.sum((chosen>=wa.shape[1])&(chosen<wa.shape[1]+wb.shape[1]))),
            'units_from_c':int(np.sum(chosen>=wa.shape[1]+wb.shape[1])),
            'selected_columns':chosen.tolist()}
    inputs=np.concatenate([x[:FIT][valid[:FIT]] for x in xs],axis=0).astype(np.float64)
    z=inputs@banks;weighted=(z-z.mean(0))*scale
    gram=weighted.T@weighted
    eigen,vectors=np.linalg.eigh(gram);vectors=vectors[:,-rank:]
    joint=(banks*scale)@vectors
    # Give each new coordinate a native-scale preactivation before GELU.
    rms=np.sqrt(np.mean((inputs@joint)**2))
    native_rms=np.sqrt(np.mean(z*z))
    joint*=native_rms/max(float(rms),1e-8)
    return joint.astype(np.float32),{'weighted_preactivation_energy_retained':float(eigen[-rank:].sum()/eigen.sum()),
                                  'native_preactivation_rms':float(native_rms)}

def run(config):
    if not config.only_eval:raise ValueError('Inference-only probe')
    config.get_keys()['enable_checkpointing']=False
    out=t.output;out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    rng,_,manager,mesh,model,_,tx=train.setup_mesh_and_model(config)
    tf.random.set_seed(int(config.data_shuffle_seed))
    data,evaluation=create_data_iterator(config,mesh)
    state,_,_,_=max_utils.setup_training_state(model,data,tx,config,rng,mesh,manager)
    params=state.params;flat=flatten_dict(params)
    batches=[];hashes=[];chunks={};masks=[]
    reuse=Path(os.environ['R384_REUSE_CAPTURE_DIR']) if os.environ.get('R384_REUSE_CAPTURE_DIR') else None
    def capture_groups(p,b):
        metrics,values=t.forward(model,p,b,rng,True)
        return metrics,{k:v for k,v in values.items() if any('/'+group+'/' in k for group in GROUPS)}
    capture=jax.jit(capture_groups)
    if FIT%4 or SEQUENCES%4 or not 0<FIT<SEQUENCES:raise ValueError('Use complete batches of4')
    for i in range(SEQUENCES//4):
        batch=next(evaluation);batches.append(batch)
        hashes.extend(hashlib.sha256(row.tobytes()).hexdigest() for row in np.asarray(jax.device_get(batch['inputs'])))
        masks.append(np.asarray(jax.device_get(batch['targets_segmentation'][:,::t.STRIDE]))!=0)
        if reuse is None:
            with mesh,nn_partitioning.axis_rules(config.logical_axis_rules):
                _,values=jax.device_get(capture(params,batch))
            for name,value in values.items():chunks.setdefault(name,[]).append(np.asarray(value))
        print('JOINT_CAPTURE',i,'sequences',len(hashes),flush=True)
    if len(hashes)!=SEQUENCES or len(set(hashes))!=SEQUENCES:raise ValueError('Cohort must have unique sequences')
    valid=np.concatenate(masks)
    if reuse is None:
        captured={k:np.concatenate(v,axis=1) for k,v in chunks.items()}
    else:
        prior=json.loads((reuse/'joint-r384-report.json').read_text())
        if hashes!=prior['sequence_hashes'] or config.load_parameters_path!=prior['checkpoint']:
            raise ValueError('Reusable capture checkpoint/cohort mismatch')
        with np.load(reuse/'joint-r384-capture.npz') as saved:
            np.testing.assert_array_equal(valid,saved['valid'])
            captured={k:saved[k] for k in saved.files if k!='valid'}
    base_virtual=padded_ports(params)
    variants={name:(dict(flat),dict(base_virtual)) for name in
              ('read_attn_select384','read_attn_pca384','attn_mlp_select384','attn_mlp_pca384',
               'triple_select384','triple_pca384') if any('_'+m+'384' in name for m in METHODS)}
    report={'checkpoint':config.load_parameters_path,'runtime_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'cohort_seed':config.data_shuffle_seed,'sequence_hashes':hashes,'fit_sequences':FIT,'heldout_sequences':SEQUENCES-FIT,
        'groups':GROUPS,
        'rank':RANK,'sampling_stride':t.STRIDE,'banks':[],'reconstruction':[],'losses':{},
        'parameter_delta_direct_heads':{'read_attn_active17':17*128*(3*16*32+16*32-1200),
                                         'read_attn_formal18':18*128*(3*16*32+16*32-1200),
                                         'attn_mlp':6*128*(2*16*32-1200),
                                         'triple':6*(384*(1200+5*16*32)-256*(3*1200+5*16*32))},
        'scope':'Representation retrofit, no MLP budget compensation and no runtime speed claim.'}
    hidden=jax.jit(lambda x,w:nn.gelu(jnp.asarray(x,config.dtype)@jnp.asarray(w,config.dtype)).astype(jnp.float32))
    native_dot=jax.jit(lambda h,k:jax.lax.dot_general(jnp.asarray(h,config.dtype),jnp.asarray(k,config.dtype),
              (((h.ndim-1,),(0,)),((),())),precision=jax.lax.Precision(config.matmul_precision)))
    for group,offset in [('local_0',0),('local_1',1),('fetch_2',2)]:
        if group not in GROUPS:continue
        block=f'decoder/layers/{group}/block';attn=block+'/self_attention'
        rd=('params',*attn.split('/'),'full_m_read_down','kernel')
        ad=('params',*attn.split('/'),'P_loc_down','kernel')
        md=('params',*block.split('/'),'mlp_address_down','kernel')
        for scan in range(6):
            # Original final M has no consumer; don't sacrifice its active
            # reader to preserve an untrained, unused final writer.
            if 3*scan+offset==17:continue
            xpre=captured[attn+'/P_loc_down/r256_x'][scan]
            hread=captured[attn+'/full_m_read_down/r256_h'][scan]
            hatt=captured[attn+'/P_loc_down/r256_h'][scan]
            reader_heads=[(attn,name,'read') for name in READ_UPS]
            attention_head=(attn,'P_loc_up','write')
            a_score=sum(importance(np.asarray(flat[('params',*attn.split('/'),name,'kernel')])[:,scan],
                       captured[attn+'/'+name+'/r256_y'][scan],valid) for name in READ_UPS)/3
            b_score=importance(np.asarray(flat[('params',*attn.split('/'),'P_loc_up','kernel')])[:,scan],
                      captured[attn+'/P_loc_up/r256_y'][scan],valid)
            pairs=[('read_attn',rd,ad,[xpre],[np.concatenate((hread,hatt),-1)],
                    np.concatenate((a_score,b_score)),reader_heads+[attention_head],None)]
            if group=='local_1':
                xpost=captured[block+'/mlp_address_down/r256_x'][scan]
                wa=np.asarray(flat[ad])[:,scan];wm=np.asarray(flat[md])[:,scan]
                hpre=np.asarray(jax.device_get(hidden(xpre,np.concatenate((wa,wm),1))))
                hpost=np.asarray(jax.device_get(hidden(xpost,np.concatenate((wa,wm),1))))
                m_score=importance(np.asarray(flat[('params',*block.split('/'),'mlp_address_up','kernel')])[:,scan],
                        captured[block+'/mlp_address_up/r256_y'][scan],valid)
                pairs.append(('attn_mlp',ad,md,[xpre,xpost],[hpre,hpost],
                              np.concatenate((b_score,m_score)),[attention_head,(block,'mlp_address_up','write')],None))
                wr=np.asarray(flat[rd])[:,scan]
                triple_weights=np.concatenate((wr,wa,wm),1)
                triple_pre=np.asarray(jax.device_get(hidden(xpre,triple_weights)))
                triple_post=np.asarray(jax.device_get(hidden(xpost,triple_weights)))
                pairs.append(('triple',rd,ad,[xpre,xpost],[triple_pre,triple_post],
                      np.concatenate((a_score,b_score,m_score)),
                      reader_heads+[attention_head,(block,'mlp_address_up','write')],md))
            for pair,downa,downb,xs,hs,score,heads,downc in pairs:
                for method in METHODS:
                    name=pair+'_'+method+'384';pf,vf=variants[name]
                    joint,details=joint_bank(np.asarray(flat[downa])[:,scan],np.asarray(flat[downb])[:,scan],xs,hs,score,valid,method,
                                            third=None if downc is None else np.asarray(flat[downc])[:,scan])
                    vf[downa[1:]]=vf[downa[1:]].at[:,scan].set(joint)
                    vf[downb[1:]]=vf[downb[1:]].at[:,scan].set(joint)
                    if downc is not None:vf[downc[1:]]=vf[downc[1:]].at[:,scan].set(joint)
                    bank={'case':name,'layer':3*scan+offset,**details};report['banks'].append(bank)
                    for scope,up,kind in heads:
                        x=xpost if up=='mlp_address_up' else xpre
                        h=np.asarray(jax.device_get(hidden(x,joint)))
                        target=captured[scope+'/'+up+'/r256_y'][scan]
                        k,b=affine(h,target,valid)
                        kp=('params',*scope.split('/'),up,'kernel')
                        vf[kp[1:]]=vf[kp[1:]].at[:,scan].set(k)
                        if kind=='write':
                            bp=kp[:-1]+('bias',);vf[bp[1:]]=vf[bp[1:]].at[:,scan].set(b)
                            original_bias=0
                        else:
                            bp=('params',*attn.split('/'),'general_'+READ_UPS[up]+'_pre_bias')
                            original_bias=np.asarray(flat[bp])[:,scan]
                            pf[bp]=pf[bp].at[:,scan].set(jnp.asarray(original_bias)+b)
                        pred64=(h.astype(np.float64)@k.reshape(RANK,-1)+b.reshape(-1)).reshape(target.shape).astype(np.float32)
                        dot=native_dot(h,k)
                        pred=np.asarray(jax.device_get(dot+jnp.asarray(original_bias+b,config.dtype)),np.float32)
                        target_full=np.asarray(jax.device_get(jnp.asarray(target,config.dtype)+jnp.asarray(original_bias,config.dtype)),np.float32)
                        if kind=='write':
                            ns=('params',*attn.split('/'),'write_address_norm')
                            scale=flat.get(ns+('scale',));bias=flat.get(ns+('bias',))
                            def norm(y):return t.normalize_address(y,None if scale is None else scale[:,scan],None if bias is None else bias[:,scan],config)
                        else:
                            def norm(y):return np.asarray(jax.device_get(normalizations.rms_norm(jnp.asarray(y,config.dtype),dtype=config.dtype,
                              epsilon=config.normalization_layer_epsilon,statistics_dtype=jnp.float32 if config.bam_read_rms_statistics_dtype=='float32' else config.dtype)),np.float32)
                        native,prednorm=norm(target_full),norm(pred)
                        hold=valid[FIT:];a=native[FIT:][hold];bpred=prednorm[FIT:][hold]
                        fit_mask=valid[:FIT]
                        train_a=native[:FIT][fit_mask];train_b=prednorm[:FIT][fit_mask]
                        cosine=np.sum(a*bpred,-1)/np.maximum(np.linalg.norm(a,axis=-1)*np.linalg.norm(bpred,axis=-1),1e-12)
                        row={'case':name,'layer':3*scan+offset,'head':up,'address_cosine':float(cosine.mean()),
                              'projection_bf16_relative_rmse':float(np.sqrt(np.sum((pred-(pred64+original_bias))**2)/np.sum((pred64+original_bias)**2))),
                              'fit_normalized_relative_rmse':float(np.sqrt(np.sum((train_b-train_a)**2)/np.sum(train_a*train_a))),
                              'normalized_relative_rmse':float(np.sqrt(np.sum((bpred-a)**2)/np.sum(a*a)))}
                        if kind=='write':
                            content=captured[attn+('/r256_mlp_content' if up=='mlp_address_up' else '/r256_attn_content')][scan,FIT:][hold]
                            dm=np.einsum('nhk,nhv->nkv',content,a);dd=np.einsum('nhk,nhv->nkv',content,bpred-a)
                            row['write_relative_rmse']=float(np.sqrt(np.sum(dd*dd)/np.sum(dm*dm)))
                        report['reconstruction'].append(row)
                    print('JOINT_BANK',json.dumps(bank),flush=True)
    # Match the six MLP-write layers when comparing pair versus triple sharing.
    for method in METHODS:
        full_pf,full_vf=variants['read_attn_'+method+'384']
        sparse_pf,sparse_vf=dict(flat),dict(base_virtual)
        sparse_pf.update({p:v for p,v in full_pf.items() if 'local_1' in p})
        sparse_vf.update({p:v for p,v in full_vf.items() if 'local_1' in p})
        variants['read_attn_sparse_'+method+'384']=(sparse_pf,sparse_vf)
    plain=jax.jit(lambda p,b:t.forward(model,p,b,rng,False)[0])
    def evaluate(p):
        metrics=[]
        for batch in batches[FIT//4:]:
            with mesh,nn_partitioning.axis_rules(config.logical_axis_rules):metrics.append(jax.device_get(plain(p,batch)))
        return np.concatenate([x['sequence_loss'] for x in metrics]),np.concatenate([x['sequence_weights'] for x in metrics])
    original,_=evaluate(params)
    def variables(pf,vf):
        augmented=dict(pf)
        for path,value in vf.items():
            augmented[('params',)+path[:-1]+('r384_'+path[-1],)]=value
        return unflatten_dict(augmented)
    baseline,weights=evaluate(variables(dict(flat),base_virtual))
    report['identity_padding']={'loss_delta':float(np.average(baseline-original,weights=weights)),
                                'max_sequence_delta':float(np.max(np.abs(baseline-original)))}
    report['losses']['baseline']={'loss':float(np.average(baseline,weights=weights)),'sequence_loss':baseline.tolist()}
    for name,(pf,vf) in variants.items():
        loss,_=evaluate(variables(pf,vf));delta=loss-baseline
        report['losses'][name]={'loss':float(np.average(loss,weights=weights)),
             'delta':float(np.average(delta,weights=weights)),'paired_sem':float(delta.std(ddof=1)/np.sqrt(len(delta))),
             'paired_delta':delta.tolist()}
        print('JOINT_LOSS',name,json.dumps(report['losses'][name]),flush=True)
    report['seconds']=time.perf_counter()-start
    (out/'joint-r384-report.json').write_text(json.dumps(report,indent=2)+'\n')
    if reuse is None:np.savez_compressed(out/'joint-r384-capture.npz',**captured,valid=valid)
    np.savez_compressed(out/'joint-r384-variants.npz',**{name+'/'+'/'.join(p):np.asarray(v) for name,(_,vf) in variants.items() for p,v in vf.items()})
    np.savez_compressed(out/'joint-r384-reader-biases.npz',**{name+'/'+'/'.join(p):np.asarray(v)
        for name,(pf,_) in variants.items() for p,v in pf.items() if p[-1] in ('general_q_pre_bias','general_k_pre_bias','general_vo_pre_bias')})
    print('JOINT_R384_COMPLETE',report['seconds'],flush=True)

def main(argv):run(pyconfig.initialize(argv))
if __name__=='__main__':app.run(main)
