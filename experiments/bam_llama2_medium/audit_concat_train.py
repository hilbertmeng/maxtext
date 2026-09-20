"""Trace the actual train step and check health export for both concat layouts."""
import functools
import sys
import tempfile
from pathlib import Path
import jax
import max_utils
import pyconfig
import train
import train_compile

RUNS = ('BamMediumIndependentLLFMLPPerLayerColOnlyNoPE32PartialRoPE',
        'BamMediumIndependentLLFColOnlyVConcatMLPPerLayer',
        'BamMediumIndependentLLFColOnlyQKConcatSharedRank4MLPPerLayer',
        'BamMediumIndependentLLFColOnlyVConcatStaticVOWriteMixMLPPerLayer',
        'BamMediumIndependentLLFColOnlyQKConcatSharedRank4StaticMLPPerLayer')
for exp in (sys.argv[1:] or RUNS):
  with tempfile.TemporaryDirectory() as out:
    Path(out, 'audit').mkdir()
    cfg = pyconfig.initialize(
        [None, 'MaxText/configs/base.yml'], exp_class=exp, run_name='audit',
        enable_checkpointing=False, base_output_directory=out+'/', jax_cache_dir='',
        log_config=False, dataset_type='synthetic', max_target_length=16,
        max_prefill_predict_length=16, query_chunk_size=4, per_device_batch_size=1.)
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
    args, kwargs, shardings, model = train_compile.get_shaped_inputs(mesh, cfg)
    result = jax.eval_shape(functools.partial(train.train_step, model, cfg, shardings),
                            *args, **kwargs)
    metrics = result[1]['scalar']
    keys = [k for k in metrics if k.startswith('bam/concat/')]
    assert all(metrics[k].shape == () for k in keys)
    for layer in range(cfg.num_decoder_layers):
      assert f'bam/concat/local_q_gate/layer_{layer:03d}/mean' in keys
      if cfg.bam_concat_qk:
        assert f'bam/concat/qk_scores/layer_{layer:03d}/bam_over_standard' in keys
    if cfg.bam_concat_v:
      assert not any('/local_v_' in k and '/layer_000/' in k for k in keys)
      assert not any('/local_o_' in k and '/layer_000/' in k for k in keys)
      assert 'bam/concat/local_v_amplitude/layer_001/bam_over_standard' in keys
    assert 'bam/concat/fetched_o_amplitude/layer_023/bam_over_standard' in keys
    if getattr(cfg, 'bam_extra_final_local_layer', False):
      assert 'bam/concat/local_v_amplitude/layer_024/bam_over_standard' in keys
      assert 'bam/concat/local_o_amplitude/layer_024/bam_over_standard' in keys
      assert not any('/fetched_o_' in k and '/layer_024/' in k for k in keys)
    print('CONCAT_TRAIN_TRACE_OK', exp, len(keys), 'scalar read-health metrics', flush=True)
