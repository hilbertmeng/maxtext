"""Matched-cohort native MUDD features, spectral energies and head-only probes."""
import argparse,contextlib,gc,hashlib,json,subprocess,time,zipfile
from pathlib import Path
import numpy as np
import jax,jax.numpy as jnp
from flax import linen as nn
from flax.traverse_util import unflatten_dict
import mudd_spectral_capture as cap
import mudd_spectral_io as io

VARIANTS=['keep','remove_head_dark5','remove_head_random5','remove_final_mudd','remove_final_mudd_dark5']
SUMMARY=['CE','gradient_norm2','gradient_dark_norm2','probability_error_norm2','head_input_mean_square','entropy','centered_logit_mean_square']

def save_npz(path,**arrays):
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
        for key,array in arrays.items():
            with archive.open(key+'.npy','w',force_zip64=True) as f:np.lib.format.write_array(f,np.asarray(array),allow_pickle=False)

def gs_json(uri):
    p=subprocess.run(['gsutil','cat',uri],capture_output=True,text=True,timeout=90)
    if p.returncode==0:return json.loads(p.stdout)
    if any(x in p.stderr for x in ['No URLs matched','matched no objects','NotFound','404']):return None
    raise RuntimeError(p.stderr[-600:])

def publish(path,metadata,destination):
    subprocess.run(['gsutil','-q','cp',str(path),destination+'/'+path.name],check=True)
    marker=path.with_suffix('.json');marker.write_text(json.dumps(metadata,indent=2)+'\n')
    subprocess.run(['gsutil','-q','cp',str(marker),destination+'/'+marker.name],check=True)

def validate(marker,common):
    for key in ['source_commit','checkpoint','family','label_step','sequence_hashes']:
        assert marker[key]==common[key],('Artifact identity differs',key)

def main():
    parser=argparse.ArgumentParser()
    for name in ['manifest','family','output','training-config']:parser.add_argument('--'+name,required=True)
    a=parser.parse_args();manifest=json.loads(Path(a.manifest).read_text());info=manifest['models'][a.family]
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();assert source==manifest['commit']
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    cohort=out/'cohort32.npz'
    if not cohort.exists():subprocess.run(['gsutil','-q','cp',manifest['cohort_source'],str(cohort)],check=True)
    with np.load(cohort) as z:tokens=z['tokens'][:,:2049].copy()
    hashes=[hashlib.sha256(np.asarray(x,np.int32).tobytes()).hexdigest() for x in tokens]
    assert hashes==manifest['sequence_hashes'] and tokens.shape==(32,2049)
    layers=32;length=2048;count=32;mesh=None;cfg=None
    if a.family=='old':
        import mudd_spectral_praxis as old
        from absl import flags
        if not flags.FLAGS.is_parsed():flags.FLAGS(['mudd_spectral_native'])
        p=old.configuration();model=old.base_layer.instantiate(p)
        p.skip_compute_loss=True;feature_model=old.base_layer.instantiate(p)
        def native(params,x,y):
            params=jax.tree.map(lambda v:v.astype(jnp.bfloat16),params)
            labels=old.py_utils.NestedMap(class_ids=y,class_weights=jnp.ones(y.shape,jnp.bfloat16))
            with old.base_layer.JaxContext.new_context(hparams=old.base_layer.JaxContext.HParams(do_eval=True)):
                result=model.apply(params,x,jnp.zeros(x.shape,jnp.bfloat16),labels=labels,segment_ids=jnp.ones_like(x),segment_pos=jnp.broadcast_to(jnp.arange(length),x.shape))
            return result.per_example_xent.astype(jnp.float32)
        def observed(params,x,y,dark):
            params=jax.tree.map(lambda v:v.astype(jnp.bfloat16),params)
            observer=cap.Capture('old',dark,length,layers)
            with old.base_layer.JaxContext.new_context(hparams=old.base_layer.JaxContext.HParams(do_eval=True)),nn.intercept_methods(observer.interceptor):
                feature_model.apply(params,x,jnp.zeros(x.shape,jnp.bfloat16),segment_ids=jnp.ones_like(x),segment_pos=jnp.broadcast_to(jnp.arange(length),x.shape))
            return observer.result()
    else:
        import max_utils
        from mudd_spectral_model import serial_setup
        cfg,checked,mesh,model,rng,abstract=serial_setup(a.training_config)
        max_utils.cross_entropy_with_logits=cap.precise_cross_entropy
        def native(params,x,y):
            with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
                return model.apply(params,x,jnp.broadcast_to(jnp.arange(length),x.shape),decoder_target_tokens=y,decoder_target_mask=jnp.ones_like(x),decoder_segment_ids=jnp.ones_like(x),enable_dropout=False,rngs={'params':rng,'dropout':rng,'aqt':rng})[0]
        def observed(params,x,y,dark):
            observer=cap.Capture('serial',dark,length,layers)
            with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules),nn.intercept_methods(observer.interceptor):
                _,collection=model.apply(params,x,jnp.broadcast_to(jnp.arange(length),x.shape),decoder_target_tokens=y,decoder_target_mask=jnp.ones_like(x),decoder_segment_ids=jnp.ones_like(x),enable_dropout=False,rngs={'params':rng,'dropout':rng,'aqt':rng},mutable=['intermediates'])
            return observer.result(collection['intermediates'])
    native_fn=jax.jit(native);feature_fn=jax.jit(observed);native_ce=jax.jit(cap.native_head_ce)
    grad_fn=jax.jit(cap.readout_gradient);stage_fn=jax.jit(cap.stage_stats)
    norm_fn=jax.jit(lambda x,s,b:cap.final_norm(x,s,b,a.family))
    random_basis=jnp.asarray(np.linalg.qr(np.random.default_rng(20261102).normal(size=(2560,128)))[0],jnp.float32)
    project=jax.jit(lambda x,v:jnp.einsum('btd,dk,ek->bte',x.astype(jnp.float32),v,v,precision=jax.lax.Precision.HIGHEST))
    for step in info['steps']:
        destination=manifest['gcs_root']+f'/{a.family}/{step}';folder=out/a.family/str(step);folder.mkdir(parents=True,exist_ok=True)
        checkpoint=info['checkpoints'][str(step)]
        common=dict(source_commit=source,praxis_commit=manifest['praxis_commit'],family=a.family,experiment=info['experiment'],checkpoint=checkpoint,label_step=step,sequence_hashes=hashes,sequence_count=count,length=length,variants=VARIANTS,head_summary_fields=SUMMARY,pair_metric_fields=cap.PAIR_METRICS,stage_names=cap.STAGES,composition_channels=['q','k','v','m'],composition_sites='1..31 before block2..32; site32 is final readout, m channel only')
        complete=gs_json(destination+'/complete.json')
        if complete is not None:
            validate(complete,common);assert complete['sequences']==count
            print('SPECTRAL_REUSED_COMPLETE',a.family,step,flush=True);continue
        listed=subprocess.run(['gsutil','ls',destination+'/'],capture_output=True,text=True)
        if listed.returncode==0:subprocess.run(['gsutil','-q','-m','rsync','-r',destination,str(folder)],check=True)
        elif not any(x in listed.stderr for x in ['No URLs matched','matched no objects','404','NotFound']):raise RuntimeError(listed.stderr[-500:])
        spectrum=folder/'spectrum.npz'
        subprocess.run(['gsutil','-q','cp',manifest['gcs_root']+f'/head_spectra/{a.family}_{step}.npz',str(spectrum)],check=True)
        with np.load(spectrum) as z:basis=jnp.asarray(z['basis']);eff_basis=jnp.asarray(z['effective_basis']);svals=z['singular'].copy()
        dark=basis[:,:128];dark_eff=eff_basis[:,:128]
        captures=all((folder/f'sequence_{i:03d}.json').exists() for i in range(count))
        flat,identity=io.params(checkpoint,a.family,head_only=captures,workers=16)
        assert identity['step']==step+(a.family=='serial'),identity
        common['state_step']=identity['step'];common['state_step_offset_from_label']=identity['step']-step
        if not captures:assert identity['parameter_count']==info['parameter_count']
        w,gamma,bias_affine=io.flatten_head(flat,a.family)
        raw_scale=flat[('final_ln','scale')] if a.family=='old' else flat[('decoder','lm_head','decoder_norm','scale')]
        w=jnp.asarray(w);scale=jnp.asarray(raw_scale);bias_affine=jnp.asarray(bias_affine)
        trace=float(jnp.sum(w.astype(jnp.bfloat16).astype(jnp.float32)**2));assert abs(trace/np.sum(svals*svals)-1)<1e-5
        if not captures:
            tree=unflatten_dict(flat);params=jax.device_put({'params':tree});del tree,flat;gc.collect()
            for i in range(count):
                path=folder/f'sequence_{i:03d}.npz';marker=path.with_suffix('.json')
                if marker.exists():validate(json.loads(marker.read_text()),common);continue
                x=jnp.asarray(tokens[i:i+1,:length]);y=jnp.asarray(tokens[i:i+1,1:length+1])
                native_loss=native_fn(params,x,y)
                features=feature_fn(params,x,y,dark_eff)
                replay=native_ce(features['h'],w,y)
                drift=float(jnp.max(jnp.abs(native_loss.mean(1)-replay.mean(1))));assert drift<.005,('Native vs captured readout CE',a.family,step,i,drift)
                norm_h=norm_fn(features['raw'],scale,bias_affine)
                relative=float(jnp.sqrt(jnp.mean((norm_h.astype(jnp.float32)-features['h'].astype(jnp.float32))**2)/jnp.mean(features['h'].astype(jnp.float32)**2)))
                assert relative<.015,('Native final norm replay',relative)
                if i==0:
                    again=feature_fn(params,x,y,dark_eff)
                    for k in ['raw','base','h','composition']:np.testing.assert_array_equal(np.asarray(features[k]),np.asarray(again[k]))
                arrays={k:np.asarray(v,dtype=np.float32) for k,v in features.items()}
                arrays.update(inputs=np.asarray(x),targets=np.asarray(y),native_loss=np.asarray(native_loss),captured_native_loss=np.asarray(replay))
                dtypes={k:str(features[k].dtype) for k in ['raw','base','h']}
                metadata=dict(common,sequence=i,input_hash=hashes[i],original_dtypes=dtypes,native_CE_mean_difference=drift,norm_relative_RMS=relative)
                save_npz(path,**arrays);publish(path,metadata,destination)
                print('SPECTRAL_CAPTURE_COMPLETE',a.family,step,i,float(native_loss.mean()),'CEdrift',drift,'norm',relative,flush=True)
            del params;gc.collect()
        else:del flat
        # Replay only the small vocabulary readout. Aggregate gradients use the
        # whole identical32-text cohort, distinct from the training batch1024.
        grad_sums={};all_summaries=[];max_fp32_drift=0.
        for i in range(count):
            path=folder/f'sequence_{i:03d}.npz';metadata=json.loads(path.with_suffix('.json').read_text());validate(metadata,common)
            with np.load(path) as z:
                f={k:jnp.asarray(z[k],jnp.bfloat16 if metadata['original_dtypes'][k]=='bfloat16' else jnp.float32) for k in ['raw','base','h']}
                y=jnp.asarray(z['targets']);native_loss=z['native_loss'].copy()
            stages=stage_fn(f['raw'],f['base'],f['h'],basis,eff_basis)
            parseval=float(jnp.max(jnp.abs(stages['spectral_energy'].sum(-1)-stages['stage_total2'])/jnp.maximum(stages['stage_total2'],1e-20)));assert parseval<1e-4,parseval
            h=f['h'].astype(jnp.float32);extra=f['raw'].astype(jnp.float32)-f['base'].astype(jnp.float32)
            remove_extra_dark=(f['raw'].astype(jnp.float32)-project(extra,dark_eff)).astype(f['raw'].dtype)
            hs=[h,h-project(h,dark),h-project(h,random_basis),norm_fn(f['base'],scale,bias_affine).astype(jnp.float32),norm_fn(remove_extra_dark,scale,bias_affine).astype(jnp.float32)]
            hs=[hv.astype(f['h'].dtype) for hv in hs]
            summaries=[];ces=[];native_ces=[]
            for name,hv in zip(VARIANTS,hs):
                g,ce,summary=grad_fn(hv,w,y,dark)
                grad_sums[name]=g/count if name not in grad_sums else grad_sums[name]+g/count
                summaries.append(np.asarray(summary));ces.append(np.asarray(ce).reshape(1,length))
                native_ces.append(np.asarray(native_ce(hv,w,y)))
            drift=float(abs(ces[0].mean()-native_loss.mean()));assert drift<.005,('FP32 head replay vs native',drift)
            max_fp32_drift=max(max_fp32_drift,drift);all_summaries.append(np.stack(summaries))
            small={k:np.asarray(v) for k,v in stages.items()};small.update(variant_CE=np.stack(ces),variant_native_CE=np.stack(native_ces),head_summary=np.stack(summaries))
            hp=folder/f'head_{i:03d}.npz';save_npz(hp,**small)
            publish(hp,dict(common,sequence=i,input_hash=hashes[i],parseval_relative_error=parseval,FP32_head_native_mean_CE_difference=drift),destination)
            print('SPECTRAL_HEAD_CASE',a.family,step,i,'keep/remove-dark/random/remove-extra/remove-extra-dark',np.stack(summaries)[:,:3].tolist(),flush=True)
        global_stats={}
        for name,g in grad_sums.items():
            gd=jnp.einsum('dk,dv->kv',dark,g,precision=jax.lax.Precision.HIGHEST)
            total=float(jnp.sum(g*g));dark_energy=float(jnp.sum(gd*gd));assert dark_energy<=total*1.0001
            global_stats[name]=dict(gradient_norm=float(np.sqrt(total)),dark_gradient_energy=dark_energy,dark_gradient_fraction=dark_energy/total,mean_vocabulary_direction_norm=float(jnp.linalg.norm(jnp.mean(g,axis=1))))
        marker=dict(common,sequences=count,global_head_gradients=global_stats,head_per_sequence_summary_mean=np.mean(all_summaries,axis=0).tolist(),max_FP32_head_native_mean_CE_difference=max_fp32_drift)
        cp=folder/'complete.json';cp.write_text(json.dumps(marker,indent=2)+'\n');subprocess.run(['gsutil','-q','cp',str(cp),destination+'/complete.json'],check=True)
        print('SPECTRAL_CHECKPOINT_COMPLETE',a.family,step,json.dumps(global_stats),flush=True)
        del grad_sums,w,scale,bias_affine;gc.collect()

if __name__=='__main__':main()
