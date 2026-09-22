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
    modes = cfg.bam_layer_modes
    for layer in range(cfg.num_decoder_layers):
      mode = modes[layer] if isinstance(modes, list) else modes
      fetch_key = f'bam/concat/fetched_o_amplitude/layer_{layer:03d}/bam_over_standard'
      local_key = f'bam/concat/local_o_amplitude/layer_{layer:03d}/bam_over_standard'
      if 'full' in mode:
        assert fetch_key in keys and local_key not in keys
      elif 'local_o' in mode and not (cfg.bam_concat_v_full_first_layer and layer == 0):
        assert local_key in keys and fetch_key not in keys
        assert f'bam/concat/local_v_amplitude/layer_{layer:03d}/bam_over_standard' in keys
    if getattr(cfg, 'bam_extra_final_local_layer', False):
      assert 'bam/concat/local_v_amplitude/layer_024/bam_over_standard' in keys
      assert 'bam/concat/local_o_amplitude/layer_024/bam_over_standard' in keys
      assert not any('/fetched_o_' in k and '/layer_024/' in k for k in keys)
    if getattr(cfg, 'bam_m_relay_anchor', 0):
      relay = [k for k in metrics if k.startswith('bam/m_relay/')]
      arms = {'all': ('all',), 'qk_vo': ('qk', 'vo'), 'vo_only': ('vo',)}[cfg.bam_m_relay_reads]
      stat_count = 10 if getattr(cfg, 'bam_m_relay_learned_scale', False) else 7
      assert len(relay) == (cfg.num_decoder_layers-3)*stat_count*len(arms)
      assert all(metrics[k].shape == () for k in relay)
      for layer in range(3, cfg.num_decoder_layers):
        for arm in arms:
          prefix = 'bam/m_relay' if arm == 'all' else f'bam/m_relay/{arm}'
          assert f'{prefix}/layer_{layer:03d}/scale_mean' in relay
      print('RELAY_TRAIN_TRACE_OK', exp, len(relay), flush=True)
    print('CONCAT_TRAIN_TRACE_OK', exp, len(keys), 'scalar read-health metrics', flush=True)
