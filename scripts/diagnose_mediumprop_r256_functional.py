"""Held-out functional R256 sharing probe on immutable original model source.

Arguments: SEALED_REPO OUTPUT_DIR [normal MaxText config arguments...].
Only eval is permitted; checkpoint and original parameter tree remain read-only.
Instrumentation samples inside forward, before device-to-host transfer.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

repository, output = Path(sys.argv[1]), Path(sys.argv[2])
sys.argv = [sys.argv[0], *sys.argv[3:]]
os.chdir(repository)
sys.path.insert(0, str(repository / 'MaxText'))
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from absl import app
from flax import linen as nn
from flax.core import FrozenDict, freeze
from flax.linen import partitioning as nn_partitioning
from flax.traverse_util import flatten_dict, unflatten_dict
import max_utils
import pyconfig
import train
from input_pipeline.input_pipeline_interface import create_data_iterator
from layers import attentions, linears, normalizations

STRIDE = 64
DOWNS = {'full_m_read_down', 'P_loc_down', 'mlp_address_down'}
UPS = {'P_loc_up', 'mlp_address_up'}

def recording(module):
    return module.is_mutable_collection('intermediates') and not module.is_initializing()

original_dense = linears.DenseGeneral.__call__
def dense_capture(self, inputs):
    result = original_dense(self, inputs)
    if recording(self):
        if self.name in DOWNS:
            self.sow('intermediates', 'r256_x', inputs[:, ::STRIDE].astype(jnp.float32))
            self.sow('intermediates', 'r256_h', nn.gelu(result)[:, ::STRIDE].astype(jnp.float32))
        elif self.name in UPS:
            self.sow('intermediates', 'r256_y', result[:, ::STRIDE].astype(jnp.float32))
    return result
linears.DenseGeneral.__call__ = dense_capture

original_write = attentions.BamAttention._write
def write_capture(self, head, x, matrix):
    result = original_write(self, head, x, matrix)
    if recording(self):
        content = self.write_data_norm(head[..., :self.bam_k]) if self._write_data_rms else head[..., :self.bam_k]
        scale = self.num_query_heads ** -.5 if self.config.bam_sqrt_n_scale else 1.
        self.sow('intermediates', 'r256_attn_content', (content * result[1][..., None] * scale)[:, ::STRIDE].astype(jnp.float32))
    return result
attentions.BamAttention._write = write_capture

original_deferred = attentions.BamAttention._deferred_write_factors
def deferred_capture(self, head, x):
    result = original_deferred(self, head, x)
    if recording(self):
        self.sow('intermediates', 'r256_attn_content', result[0][:, ::STRIDE].astype(jnp.float32))
    return result
attentions.BamAttention._deferred_write_factors = deferred_capture

original_merge = attentions.BamAttention.merge_mlp_write
def merge_capture(self, head, gate, factors, matrix, independent_address=None, **kwargs):
    result = original_merge(self, head, gate, factors, matrix, independent_address, **kwargs)
    if recording(self):
        content = self.write_data_norm(head) if self._write_data_rms else head
        scale = self.num_query_heads ** -.5 if self.config.bam_sqrt_n_scale else 1.
        self.sow('intermediates', 'r256_mlp_content', (content * gate[..., None] * scale)[:, ::STRIDE].astype(jnp.float32))
    return result
attentions.BamAttention.merge_mlp_write = merge_capture

def forward(model, params, batch, rng, capture=False):
    rng1, rng2 = jax.random.split(rng)
    kwargs = dict(decoder_segment_ids=batch['inputs_segmentation'],
        decoder_target_mask=batch['targets_segmentation'], decoder_target_tokens=batch['targets'],
        enable_dropout=False, rngs={'dropout': rng1, 'params': rng2})
    result = model.apply(params, batch['inputs'], batch['inputs_position'],
                         mutable=['intermediates'] if capture else False, **kwargs)
    sampled = {}
    if capture:
        result, collections = result
        for path, value in flatten_dict(collections['intermediates']).items():
            if path[-1].startswith('r256_'):
                while isinstance(value, (tuple, list)) and len(value) == 1:
                    value = value[0]
                if isinstance(value, (tuple, list)):
                    raise ValueError(f'Duplicate projection evaluation: {path}')
                sampled['/'.join(path)] = value
    xent, _, _ = result
    mask = batch['targets_segmentation'] != 0
    counts = jnp.sum(mask, axis=-1)
    return {'sequence_loss': jnp.sum(xent * mask, axis=-1) / jnp.maximum(counts, 1),
            'sequence_weights': counts}, sampled

def fit_affine(h, target, valid, train_sequences=16):
    """Fit solely on first16 sequences; preserve per-head affine output shape."""
    a = h[:train_sequences][valid[:train_sequences]].astype(np.float64)
    b = target[:train_sequences][valid[:train_sequences]].astype(np.float64)
    b = b.reshape(len(b), -1)
    ma, mb = a.mean(0), b.mean(0)
    ac, bc = a - ma, b - mb
    gram = ac.T @ ac
    ridge = 1e-4 * np.trace(gram) / len(gram)
    kernel = np.linalg.solve(gram + ridge * np.eye(len(gram)), ac.T @ bc)
    bias = mb - ma @ kernel
    prediction = (h.astype(np.float64) @ kernel + bias).reshape(target.shape)
    return kernel.astype(np.float32).reshape((h.shape[-1],) + target.shape[-2:]), bias.astype(np.float32).reshape(target.shape[-2:]), prediction.astype(np.float32), float(ridge)

def normalize_address(a, scale, bias, config):
    mode=config.bam_write_rms_statistics_dtype
    if mode not in ('float32','activation'):raise ValueError(mode)
    y=normalizations.rms_norm(jnp.asarray(a,config.dtype),dtype=config.dtype,
         epsilon=config.normalization_layer_epsilon,
         statistics_dtype=jnp.float32 if mode=='float32' else config.dtype)
    if scale is not None:y=y*(jnp.asarray(scale,config.dtype)+1)
    if bias is not None:y=y+jnp.asarray(bias,config.dtype)
    return np.asarray(jax.device_get(y),dtype=np.float32)

def run(config):
    if not config.only_eval or config.get_keys().get('bam_diagnostics', False):
        raise ValueError('Require only_eval=True and raw BAM diagnostics disabled')
    # Historical pyconfig requires this flag for any load_parameters_path.
    # After validation, disable the manager entirely: restore is read-only.
    config.get_keys()['enable_checkpointing'] = False
    output.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    subprocess.run(['git', 'diff', '--quiet'], check=True)
    t0 = time.perf_counter()
    rng, _, manager, mesh, model, _, tx = train.setup_mesh_and_model(config)
    tf.random.set_seed(int(config.data_shuffle_seed))
    data, evaluation = create_data_iterator(config, mesh)
    state, _, _, _ = max_utils.setup_training_state(model, data, tx, config, rng, mesh, manager)
    params = state.params
    flat = flatten_dict(params)
    batches, hashes, original_metrics, parts, masks = [], [], [], {}, []
    capture_forward = jax.jit(lambda p,b,r: forward(model,p,b,r,True))
    plain_forward = jax.jit(lambda p,b,r: forward(model,p,b,r,False)[0])
    for index in range(8):
        batch = next(evaluation)
        batches.append(batch)
        host_tokens = np.asarray(jax.device_get(batch['inputs']))
        hashes.extend(hashlib.sha256(row.tobytes()).hexdigest() for row in host_tokens)
        masks.append(np.asarray(jax.device_get(batch['targets_segmentation'][:, ::STRIDE])) != 0)
        with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
            metrics, captured = jax.device_get(capture_forward(params, batch, rng))
        original_metrics.append(metrics)
        for name, value in captured.items():
            parts.setdefault(name, []).append(np.asarray(value))
        print('CAPTURE_BATCH',index,'sequences',len(hashes),'seconds',time.perf_counter()-t0,flush=True)
    if len(hashes) != 32 or len(set(hashes)) != 32:
        raise ValueError(f'Cohort must be32 unique sequences: {len(hashes)}, unique={len(set(hashes))}')
    captured = {name: np.concatenate(values, axis=1) for name, values in parts.items()}
    valid = np.concatenate(masks, axis=0)
    projections = {name: np.asarray(value) for name,value in flat.items()
                   if len(name)>1 and name[-2] in DOWNS | UPS}
    report = {'checkpoint':config.load_parameters_path, 'runtime_commit':commit,
       'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       'cohort_seed':config.data_shuffle_seed,'shuffle_buffer':config.eval_shuffle_buffer_size,
       'sequence_hashes':hashes,'train_sequences':16,'heldout_sequences':16,
       'stride':STRIDE,'device_count':jax.device_count(),'projection_shapes':{
           '/'.join(p):list(v.shape) for p,v in projections.items()}, 'fits':[], 'losses':{}}
    variants = {name: dict(flat) for name in ('native_refit','read_attn','attn_mlp','all_three')}
    hidden_fn = jax.jit(lambda x,w: nn.gelu(jnp.asarray(x,config.dtype) @ jnp.asarray(w,config.dtype)).astype(jnp.float32))
    groups = [('local_0',0),('local_1',1),('fetch_2',2)]
    for group, offset in groups:
        block = f'decoder/layers/{group}/block'
        attn = block + '/self_attention'
        read_key = attn + '/full_m_read_down/r256_h'
        attention_down = ('params', *attn.split('/'), 'P_loc_down', 'kernel')
        read_down = ('params', *attn.split('/'), 'full_m_read_down', 'kernel')
        assert projections[read_down].shape == (1200,6,256)
        for scan in range(6):
            cases = [('native_attn',attn,'P_loc_up',captured[attn+'/P_loc_down/r256_h'][scan],attention_down,'r256_attn_content'),
                     ('read_attn',attn,'P_loc_up',captured[read_key][scan],read_down,'r256_attn_content')]
            if group == 'local_1':
                x_mlp = captured[block+'/mlp_address_down/r256_x'][scan]
                aw = jax.device_get(hidden_fn(x_mlp, projections[attention_down][:,scan]))
                rw = jax.device_get(hidden_fn(x_mlp, projections[read_down][:,scan]))
                mlp_down = ('params', *block.split('/'), 'mlp_address_down', 'kernel')
                cases += [('native_mlp',block,'mlp_address_up',captured[block+'/mlp_address_down/r256_h'][scan],mlp_down,'r256_mlp_content'),
                          ('attn_mlp',block,'mlp_address_up',aw,attention_down,'r256_mlp_content'),
                          ('read_mlp',block,'mlp_address_up',rw,read_down,'r256_mlp_content')]
            for case, scope, up, h, source_path, content_name in cases:
                target = captured[scope+'/'+up+'/r256_y'][scan]
                kernel,bias,prediction,ridge = fit_affine(h,target,valid)
                kp = ('params',*scope.split('/'),up,'kernel')
                bp = ('params',*scope.split('/'),up,'bias')
                target_down = ('params',*scope.split('/'),'P_loc_down' if up=='P_loc_up' else 'mlp_address_down','kernel')
                destinations = (('native_refit',) if case.startswith('native_') else
                    (('read_attn','all_three') if case=='read_attn' else (('attn_mlp',) if case=='attn_mlp' else ('all_three',))))
                for destination in destinations:
                    updates = variants[destination]
                    updates[target_down] = updates[target_down].at[:,scan].set(flat[source_path][:,scan])
                    updates[kp] = updates[kp].at[:,scan].set(kernel)
                    updates[bp] = updates[bp].at[:,scan].set(bias)
                norm_scope = ('params',*attn.split('/'),'write_address_norm')
                scale_value = flat.get(norm_scope+('scale',))
                bias_value = flat.get(norm_scope+('bias',))
                def normalized(a):
                    return normalize_address(a,None if scale_value is None else scale_value[:,scan],
                        None if bias_value is None else bias_value[:,scan],config)
                native, shared = normalized(target), normalized(prediction)
                hold = valid[16:]
                a,b = native[16:][hold],shared[16:][hold]
                cosine = np.sum(a*b,-1) / np.maximum(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1),1e-12)
                content = captured[attn+'/'+content_name][scan,16:][hold]
                dm0=np.einsum('nhk,nhv->nkv',content,a)
                dmd=np.einsum('nhk,nhv->nkv',content,b-a)
                raw_a,raw_b=target[16:][hold],prediction[16:][hold]
                item={'case':case,'layer':3*scan+offset,'ridge':ridge,
                      'raw_relative_rmse':float(np.sqrt(np.sum((raw_b-raw_a)**2)/np.sum(raw_a**2))),
                      'address_cosine_mean':float(cosine.mean()),'address_cosine_p10':float(np.quantile(cosine,.1)),
                      'write_relative_rmse':float(np.sqrt(np.sum(dmd**2)/np.sum(dm0**2)))}
                report['fits'].append(item)
                print('FIT',json.dumps(item),flush=True)
    def losses(p):
        values=[]
        for batch in batches[4:]:
            with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
                values.append(jax.device_get(plain_forward(p,batch,rng)))
        return values
    baseline = losses(params)
    # Extra outputs can change BF16 compiler fusion and rounding. All causal
    # loss comparisons below use the same capture-free executable. Quantify
    # capture roundoff, and use a native-feature refit as the fitting control.
    capture_delta=np.concatenate([a['sequence_loss']-b['sequence_loss']
                                for a,b in zip(baseline,original_metrics[4:])])
    report['capture_vs_plain']={'max_abs':float(np.max(np.abs(capture_delta))),
        'mean':float(capture_delta.mean()),'paired_delta':capture_delta.tolist()}
    if report['capture_vs_plain']['max_abs']>2e-3:
        raise ValueError(f'Unexpected BF16 capture drift: {report["capture_vs_plain"]}')
    base_loss=np.concatenate([a['sequence_loss'] for a in baseline])
    weights=np.concatenate([a['sequence_weights'] for a in baseline])
    report['losses']['baseline']={'loss':float(np.average(base_loss,weights=weights)), 'sequence_loss':base_loss.tolist()}
    for name, updated in variants.items():
        tree=unflatten_dict(updated)
        if isinstance(params,FrozenDict):tree=freeze(tree)
        result=losses(tree)
        loss=np.concatenate([a['sequence_loss'] for a in result])
        delta=loss-base_loss
        report['losses'][name]={'loss':float(np.average(loss,weights=weights)),
            'delta':float(np.average(delta,weights=weights)),'paired_delta':delta.tolist(),
            'paired_sem':float(delta.std(ddof=1)/np.sqrt(len(delta))), 'sequence_loss':loss.tolist()}
        print('LOSS',name,json.dumps(report['losses'][name]),flush=True)
    report['seconds']=time.perf_counter()-t0
    (output/'functional-report.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(output/'sampled-ports.npz', **captured, valid=valid)
    np.savez_compressed(output/'projection-parameters.npz', **{'/'.join(p):v for p,v in projections.items()})
    print('R256_FUNCTIONAL_COMPLETE',output,'seconds',report['seconds'],flush=True)

def main(argv):
    config = pyconfig.initialize(argv)
    run(config)

if __name__ == '__main__':
    app.run(main)
