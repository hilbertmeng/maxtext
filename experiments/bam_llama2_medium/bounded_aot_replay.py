"""Replay a short, isolated diagnostic trajectory with the original train-step AOT.

The only source interposition bounds the host loop and corrects its final metric
label. Model/optimizer executable, schedule, RNG folding, data and saves remain
the runtime's implementations. Never point this runner at a formal RUN prefix.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys


def bound_host_loop(source, end_step):
  original = 'for step in np.arange(start_step, config.steps):'
  assert source.count(original) == 1, 'Unexpected runtime loop; refusing unbounded replay'
  source = source.replace(original, f'for step in np.arange(start_step, min(config.steps, {end_step})):')
  final = 'metrics, config.steps - 1, config)  # final step metrics'
  assert source.count(final) == 1
  return source.replace(final, 'metrics, int(get_first_step(state)) - 1, config)  # final diagnostic metrics')


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--runtime-dir', type=Path, required=True)
  parser.add_argument('--runtime-commit', required=True)
  parser.add_argument('--start-step', type=int, required=True)
  parser.add_argument('--end-step', type=int, required=True, help='Exclusive host-loop endpoint')
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--isolate-step', type=int,
                      help='Keep this pre-update state/batch, then binary-search loss contributors with the same AOT')
  args, overrides = parser.parse_known_args()
  assert 0 < args.end_step - args.start_step <= 200, 'Diagnostic replay must be bounded to at most200 updates'
  assert overrides and all('=' in x for x in overrides)
  values = dict(x.split('=', 1) for x in overrides)
  assert '/diagnostics/' in values['base_output_directory']
  assert values['run_name'].startswith('ASpikeReplay')
  assert values['compiled_trainstep_file']
  actual = subprocess.check_output(
      ['git', '-C', str(args.runtime_dir), 'rev-parse', 'HEAD'], text=True).strip()
  assert actual == args.runtime_commit
  os.chdir(args.runtime_dir)
  sys.path.insert(0, str(args.runtime_dir / 'MaxText'))
  import train
  import jax
  import jax.numpy as jnp
  import numpy as np

  source = inspect.getsource(train.train_loop)
  bounded = bound_host_loop(source, args.end_step)
  exec(compile(bounded, str(Path(__file__)), 'exec'), train.__dict__)
  original_get_first_step = train.get_first_step
  verified = False

  def checked_first_step(state):
    nonlocal verified
    step = original_get_first_step(state)
    if not verified:
      assert step == args.start_step, (step, args.start_step)
      verified = True
      print('DIAGNOSTIC_RESTORE_START_VERIFIED', step, flush=True)
    return step

  train.get_first_step = checked_first_step
  captured = {}
  if args.isolate_step is not None:
    assert args.start_step <= args.isolate_step < args.end_step
    original_load = train.maxtext_utils.load_compiled

    def copy_tree(tree):
      return jax.tree.map(lambda x: x.copy() if hasattr(x, 'copy') else x, tree)

    def select_metrics(metrics):
      keep = {k: v for k, v in metrics['scalar'].items()
              if k in ('learning/loss', 'learning/raw_grad_norm', 'learning/total_weights')
              or (k.startswith('raw_grads/decoder/') and '/block/self_attention/' in k
                  and any('/' + x + '/' in k for x in
                          ('query', 'key', 'W_lv_c8_up', 'W_R_up', 'out')))}
      return {k: float(np.asarray(v)) for k, v in jax.device_get(keep).items()}

    def load(config, *a, **kw):
      assert not config.mix_attn
      compiled = original_load(config, *a, **kw)
      counter = args.start_step
      captured['compiled'] = compiled

      def call(state, data, rng):
        nonlocal counter
        step = counter
        counter += 1
        if step == args.isolate_step:
          assert original_get_first_step(state) == step
          captured.update(state=copy_tree(state), data=copy_tree(data), rng=rng.copy())
        result, metrics = compiled(state, data, rng)
        if step == args.isolate_step:
          captured['observed'] = select_metrics(metrics)
          print('ISOLATION_STATE_CAPTURED', step, captured['observed']['learning/raw_grad_norm'], flush=True)
        return result, metrics

      return call

    train.maxtext_utils.load_compiled = load
  args.output.mkdir(parents=True, exist_ok=True)
  metadata = dict(runtime_commit=actual, start_step=args.start_step,
      end_step_exclusive=args.end_step, overrides=values,
      runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      original_loop_sha256=hashlib.sha256(source.encode()).hexdigest(),
      bounded_loop_sha256=hashlib.sha256(bounded.encode()).hexdigest(),
      scope='Diagnostic checkpoint copy; original serialized train-step and schedule unchanged.')
  (args.output / 'invocation.json').write_text(json.dumps(metadata, indent=2) + '\n')
  train.main([None, 'MaxText/configs/base.yml', *overrides])
  print('BOUNDED_AOT_REPLAY_COMPLETE', flush=True)
  if args.isolate_step is not None:
    assert 'observed' in captured
    data = captured['data']
    batch, length = data['targets_segmentation'].shape
    records = []

    @jax.jit
    def masked_segments(segments, lo, hi, token_lo, token_hi):
      rows = jnp.arange(batch)[:, None]
      tokens = jnp.arange(length)[None, :]
      mask = (rows >= lo) & (rows < hi) & (tokens >= token_lo) & (tokens < token_hi)
      return jnp.where(mask, segments, jnp.zeros_like(segments))

    def probe(lo, hi, token_lo=0, token_hi=length):
      masked = dict(data)
      masked['targets_segmentation'] = masked_segments(data['targets_segmentation'], lo, hi, token_lo, token_hi)
      updated, metrics = captured['compiled'](copy_tree(captured['state']), masked, captured['rng'])
      measured = select_metrics(metrics)
      del updated
      record = dict(batch_range=[lo, hi], target_position_range=[token_lo, token_hi],
                    mean_loss_normalization='selected valid target tokens', metrics=measured)
      records.append(record)
      if jax.process_index() == 0:
        (args.output / 'isolation.json').write_text(json.dumps(records, indent=2) + '\n')
        print('LOSS_MASK_PROBE', lo, hi, token_lo, token_hi,
              measured['learning/raw_grad_norm'], measured['learning/loss'], flush=True)
      return measured['learning/raw_grad_norm']

    control = probe(0, batch)
    expected = captured['observed']['learning/raw_grad_norm']
    assert abs(control - expected) <= 1e-5 * max(1., abs(expected)), (control, expected)
    lo, hi = 0, batch
    while hi - lo > 1:
      mid = (lo + hi) // 2
      left, right = probe(lo, mid), probe(mid, hi)
      if left >= right:
        hi = mid
      else:
        lo = mid
    bad_sequence = lo
    token_lo, token_hi = 0, length
    while token_hi - token_lo > 1:
      mid = (token_lo + token_hi) // 2
      left, right = probe(lo, hi, token_lo, mid), probe(lo, hi, mid, token_hi)
      if left >= right:
        token_hi = mid
      else:
        token_lo = mid
    # All processes execute the gather; only process0 writes the small cohort.
    sample = {}
    for name, array in data.items():
      replicated = jax.sharding.NamedSharding(array.sharding.mesh, jax.sharding.PartitionSpec())
      single = jax.jit(lambda x: x[bad_sequence:bad_sequence + 1], out_shardings=replicated)(array)
      sample[name] = np.asarray(jax.device_get(single))
    if jax.process_index() == 0:
      np.savez_compressed(args.output / 'isolated-sequence.npz', **sample)
      (args.output / 'isolation-result.json').write_text(json.dumps(dict(
          pre_update_step=args.isolate_step, batch_sequence_index=bad_sequence,
          target_position=token_lo, selection='greedy largest mean-CE gradient in each binary split',
          note='Target-loss masks only; model inputs and original AOT unchanged. Each case restores the frozen state.',
          input_sha256=hashlib.sha256(sample['inputs'].tobytes()).hexdigest()), indent=2) + '\n')
    print('LOSS_MASK_ISOLATION_COMPLETE', bad_sequence, token_lo, flush=True)


if __name__ == '__main__':
  main()
