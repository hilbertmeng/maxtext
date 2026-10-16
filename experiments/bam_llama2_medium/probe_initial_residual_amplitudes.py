"""Full-depth initial branch amplitudes with analytic initialization checks.

CPU-only, no training/checkpoint reads. Keep formal scan/layout/widths; bypass only
the vocabulary output head, which cannot affect the residual layers. Reductions
are inside JIT; retain per-sequence moments, not full activations.
"""
import argparse
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict

from pythia_test import PythiaTest


ARMS = {
    'mha': 'PythiaMediumPropT2048SerialMLP',
    'mudd': 'MuddPythiaMediumPropT2048SerialMLP',
    'unfixed': 'BamPythiaMediumPropT2048MShiftReadG050ReadAttentionSharedR256SerialMLP',
    'bam_fix': 'BamPythiaMediumPropT2048SerialMLPBamInitFix',
    'all_original': 'BamPythiaMediumPropT2048SerialMLPOriginalInit',
    'large_comp': 'BamPythiaMediumPropT2048SerialMLPPythiaEmbeddingScaledWrite',
    'large_uncomp': 'BamPythiaMediumPropT2048SerialMLPBamInitFix',
}


def gelu_second_moment(q):
    """E[(Z Phi(Z))**2], Z~N(0,q); not the centered variance."""
    return q * (.25 + np.arcsin(q / (1 + q)) / (2 * np.pi)) + q*q / (
        np.pi * (1 + q) * np.sqrt(1 + 2*q))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arm', choices=ARMS, required=True)
    ap.add_argument('--cohort', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--sequences', type=int, default=4)
    ap.add_argument('--length', type=int, default=2048)
    ap.add_argument('--seed', type=int, default=19)
    args = ap.parse_args()
    helper = PythiaTest(); helper.setUp()
    t0 = time.monotonic()
    with np.load(args.cohort, allow_pickle=False) as z:
        def load(a, b):
            return np.asarray(z[a if a in z else b])[:args.sequences, :args.length].astype(np.int32)
        ids = load('inputs', 'batch__inputs')
        pos = load('inputs_position', 'batch__positions')
        seg = load('inputs_segmentation', 'batch__segments')
    assert ids.shape == (args.sequences, args.length)
    assert np.all(seg > 0), 'This cohort probe requires all selected tokens valid.'
    # Uniform-prefix comparison below assumes an unpacked, single segment.
    assert all(len(np.unique(row)) == 1 for row in seg)
    overrides = dict(
        exp_class=ARMS[args.arm], base_emb_dim=1200,
        base_num_decoder_layers=18, base_num_query_heads=16,
        base_num_kv_heads=16, head_dim=75, pythia_rotary_dim=18,
        vocab_size=50432, max_target_length=args.length,
        max_prefill_predict_length=args.length, attention='dot_product',
        bam_splash_attention=False, loss_chunk_size=64, remat_policy='none',
        record_internal_nn_metrics=False)
    if args.arm == 'large_uncomp':
        overrides.update(token_embedding_init_method='', bam_embedding_write_input_scale=1.)
    with contextlib.redirect_stdout(io.StringIO()):
        c = helper.config(**overrides)
        model, mesh = helper.model(c)
    inputs = tuple(jnp.asarray(v) for v in (ids, pos, ids, np.ones_like(ids), seg))

    def rms(x):
        return jnp.sqrt(jnp.mean(jnp.asarray(x, jnp.float32)**2, axis=tuple(range(1, x.ndim))))

    def dot(x, y):
        return jnp.mean(x.astype(jnp.float32)*y.astype(jnp.float32), axis=tuple(range(1, x.ndim)))

    def hook(record):
        cache = {}
        def intercept(next_fun, a, kw, ctx):
            mod = ctx.module
            if mod.__class__.__name__ == 'OutputHead' and ctx.method_name == '__call__':
                return (jnp.array(0.), jnp.array(0.), jnp.array(0.))
            y = next_fun(*a, **kw)
            if not record or mod.is_initializing():
                return y
            path = mod.scope.path_text
            parent = path.rsplit('/', 1)[0]
            def put(name, value):
                mod.sow('intermediates', 'initial_amplitude_' + name, value)
            if ctx.method_name == '__call__':
                if mod.name == 'token_embedder':
                    put('embedding', rms(y))
                if mod.name in ('input_layernorm', 'pre_self_attention_layer_norm',
                                'pre_self_attention_layer_norm_q'):
                    put('norm_input', jnp.stack((rms(a[0]), rms(y)), -1))
                    cache.setdefault(parent, {})['attn_norm'] = y
                if mod.name in ('post_attention_layernorm', 'post_self_attention_layer_norm'):
                    cache.setdefault(parent, {})['post_attn'] = a[0]
                    cache[parent]['mlp_norm'] = y
                if mod.name == 'self_attention':
                    cache.setdefault(parent, {})['attn'] = y[0] if isinstance(y, tuple) else y
                if mod.name == 'mlp':
                    cache.setdefault(parent, {})['mlp'] = y
                    cache[parent]['mlp_norm'] = a[0]
                if mod.name == 'wi' and '/mlp/' in path:
                    put('wi', rms(y))
                if mod.name == 'wo' and '/mlp/' in path:
                    put('gelu', rms(a[0]))
                if mod.name == 'out' and '/self_attention/' in path:
                    put('head', rms(a[0]))
                if mod.name == 'value' and '/self_attention/' in path:
                    put('value', rms(y))
                if mod.name == 'attention_op':
                    v = a[2]
                    uniform = jnp.cumsum(v, axis=1) / jnp.arange(1, v.shape[1]+1)[None, :, None, None]
                    put('attention_aggregate', jnp.stack((rms(v), rms(uniform), rms(y)), -1))
                if mod.__class__.__name__ in ('SubDecoderLayer', 'PythiaDecoderLayer'):
                    saved = cache[path]
                    att, mlp = saved['attn'], saved['mlp']
                    post = saved['post_attn']
                    before = post - att
                    after = y[0] if isinstance(y, tuple) else y
                    fields = (rms(before), rms(att), rms(mlp), rms(post), rms(after),
                              dot(before, att), dot(post, mlp), rms(saved['mlp_norm']),
                              rms(jnp.mean(before, axis=1)), rms(jnp.mean(att, axis=1)),
                              rms(jnp.mean(mlp, axis=1)))
                    put('block', jnp.stack(fields, -1))
            if ctx.method_name == '_attention_block':
                v = a[2]
                uniform = jnp.cumsum(v, axis=1) / jnp.arange(1, v.shape[1]+1)[None, :, None, None]
                put('attention_aggregate', jnp.stack((rms(v), rms(uniform), rms(y[0])), -1))
            return y
        return intercept

    try:
        with mesh, nn.partitioning.axis_rules(c.logical_axis_rules), contextlib.redirect_stdout(io.StringIO()):
            init_inputs = tuple(a[:, :16] for a in inputs)
            with nn.intercept_methods(hook(False)):
                params = jax.jit(lambda key: model.init(key, *init_inputs, enable_dropout=False)['params'])(jax.random.key(args.seed))
            jax.block_until_ready(params)
            print('INITIALIZED', args.arm, 'seconds', time.monotonic()-t0, file=__import__('sys').stderr, flush=True)
            def forward(p, batch):
                with nn.intercept_methods(hook(True)):
                    _, inter = model.apply({'params': p}, *batch, enable_dropout=False, mutable=['intermediates'])
                return inter
            inter = jax.jit(forward)(params, inputs)
            inter = jax.device_get(inter)
        flat = flatten_dict(inter)
        records = {}
        for path, value in flat.items():
            if not path[-1].startswith('initial_amplitude_'):
                continue
            assert len(value) == 1
            records['/'.join(path)] = np.asarray(value[0]).tolist()
        cfgkeys = ['emb_dim','num_decoder_layers','num_query_heads','head_dim','mlp_dim',
                   'mlp_dim_by_layer','mlp_dim_by_block','dynamic_mlp_dim','scan_layers',
                   'param_scan_axis','bam_pair_scan','bam_local_fetch_block_size',
                   'init_method','bam_init_method','token_embedding_init_method',
                   'normalization_layer_epsilon','bam_write_rms_epsilon',
                   'bam_embedding_write_input_scale','pythia_parallel_residual',
                   'mudd_prenorm','mudd_full_history','vocab_size']
        out = dict(arm=args.arm, configuration=ARMS[args.arm], seed=args.seed,
                   source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   input_ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
                   cohort=args.cohort, sequences=args.sequences, length=args.length,
                   scope='Untrained full-depth fp32 CPU dense attention; formal scan/widths, head bypassed; paired token identities. Init traced at T16, parameters independent of T.',
                   resolved={k:c.get_keys().get(k) for k in cfgkeys},
                   block_fields=['residual_in','attention','mlp','post_attention','residual_out',
                                 'cross_residual_attention','cross_postatt_mlp','mlp_norm_input',
                                 'token_mean_residual','token_mean_attention','token_mean_mlp'],
                   records=records, elapsed_seconds=time.monotonic()-t0)
        Path(args.output).write_text(json.dumps(out, indent=2, default=str)+'\n')
        print(json.dumps({'arm':args.arm,'output':args.output,'seconds':out['elapsed_seconds'],'records':len(records)}),flush=True)
    finally:
        helper.tearDown()


if __name__ == '__main__':
    main()
