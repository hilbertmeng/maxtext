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


def bound_host_loop(source, end_step, force_save_steps=()):
  original = 'for step in np.arange(start_step, config.steps):'
  assert source.count(original) == 1, 'Unexpected runtime loop; refusing unbounded replay'
  source = source.replace(original, f'for step in np.arange(start_step, min(config.steps, {end_step})):')
  final = 'metrics, config.steps - 1, config)  # final step metrics'
  assert source.count(final) == 1
  source = source.replace(final, 'metrics, int(get_first_step(state)) - 1, config)  # final diagnostic metrics')
  if force_save_steps:
    original = 'if checkpoint_saver.save(int(step), save_current_checkpoint, save_current_cursor):'
    assert source.count(original) == 1, 'Unexpected checkpoint host protocol'
    source = source.replace(original, 'if checkpoint_saver.save(int(step), save_current_checkpoint, '
        f'save_current_cursor, force=int(step) in {tuple(force_save_steps)!r}):')
  return source


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--runtime-dir', type=Path, required=True)
  parser.add_argument('--runtime-commit', required=True)
  parser.add_argument('--start-step', type=int, required=True)
  parser.add_argument('--end-step', type=int, required=True, help='Exclusive host-loop endpoint')
  parser.add_argument('--max-updates', type=int, default=200,
                      help='Explicit bounded replay budget; at most2000, default200')
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--isolate-step', type=int,
                      help='Keep this pre-update state/batch, then binary-search loss contributors with the same AOT')
  parser.add_argument('--capture-first-raw-above', type=float,
                      help='Capture the first pre-update state whose original AOT raw norm exceeds this threshold')
  parser.add_argument('--capture-from-step', type=int,
                      help='Earliest eligible onset capture; defaults to the replay start')
  parser.add_argument('--record-step-metrics', action='store_true')
  parser.add_argument('--all-parameter-metrics', action='store_true')
  parser.add_argument('--force-save-steps', default='',
                      help='Comma-separated extra host checkpoint saves; model/AOT/schedule stay unchanged')
  parser.add_argument('--frozen-batch-steps', default='',
                      help='Retain these original future states/batches for frozen-pre-onset-parameter comparisons')
  parser.add_argument('--skip-loss-isolation', action='store_true',
                      help='After frozen controls, omit the already-completed binary loss-mask search')
  parser.add_argument('--tensor-probe-script', type=Path,
                      help='Instead of binary search, run this read-only probe on the captured formal state/mesh')
  parser.add_argument('--probe-sequence', type=int)
  parser.add_argument('--probe-token', type=int)
  parser.add_argument('--probe-layer', type=int, default=15)
  args, overrides = parser.parse_known_args()
  assert not (args.isolate_step is not None and args.capture_first_raw_above is not None)
  if args.capture_first_raw_above is not None:
    assert args.capture_first_raw_above > 0
  force_save_steps = tuple(sorted({int(x) for x in args.force_save_steps.split(',') if x}))
  assert all(args.start_step <= x < args.end_step for x in force_save_steps)
  frozen_batch_steps = tuple(sorted({int(x) for x in args.frozen_batch_steps.split(',') if x}))
  assert len(frozen_batch_steps) <= 12, 'Bound retained full states and device memory'
  assert all(args.start_step <= x < args.end_step for x in frozen_batch_steps)
  assert not frozen_batch_steps or args.isolate_step is not None or args.capture_first_raw_above is not None
  capture_from = args.start_step if args.capture_from_step is None else args.capture_from_step
  assert args.start_step <= capture_from < args.end_step
  if args.tensor_probe_script:
    assert args.isolate_step is not None
    assert args.probe_sequence is not None and args.probe_token is not None
  assert 0 < args.end_step - args.start_step <= args.max_updates <= 2000, 'Replay exceeds explicit bounded budget'
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
  bounded = bound_host_loop(source, args.end_step, force_save_steps)
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
  needs_capture = args.isolate_step is not None or args.capture_first_raw_above is not None
  if needs_capture or args.record_step_metrics:
    if args.isolate_step is not None:
      assert args.start_step <= args.isolate_step < args.end_step
    original_load = train.maxtext_utils.load_compiled

    def copy_tree(tree):
      return jax.tree.map(lambda x: x.copy() if hasattr(x, 'copy') else x, tree)

    def select_metrics(metrics):
      keep = {k: v for k, v in metrics['scalar'].items()
              if k in ('learning/loss', 'learning/raw_grad_norm', 'learning/grad_norm', 'learning/total_weights')
              or (args.all_parameter_metrics and k.startswith('raw_grads/'))
              or (k.startswith('raw_grads/decoder/') and '/block/self_attention/' in k
                  and any('/' + x + '/' in k for x in
                          ('query', 'key', 'W_lv_c8_up', 'W_R_up', 'out')))}
      return {k: float(np.asarray(v)) for k, v in jax.device_get(keep).items()}

    def load(config, *a, **kw):
      assert not config.mix_attn
      compiled = original_load(config, *a, **kw)
      counter = args.start_step
      captured['compiled'] = compiled
      captured.update(config=config, functional_train=a[0], mesh=a[2])

      def call(state, data, rng):
        nonlocal counter
        step = counter
        counter += 1
        eligible = (step == args.isolate_step or (args.capture_first_raw_above is not None
                    and step >= capture_from and 'observed' not in captured))
        candidate = None
        if eligible:
          assert original_get_first_step(state) == step
          candidate = dict(state=copy_tree(state), data=copy_tree(data), rng=rng.copy())
        frozen_batch = None
        if step in frozen_batch_steps:
          frozen_batch = dict(state=copy_tree(state), data=copy_tree(data), rng=rng.copy(), step=step)
        result, metrics = compiled(state, data, rng)
        raw_norm = float(np.asarray(jax.device_get(metrics['scalar']['learning/raw_grad_norm'])))
        if args.record_step_metrics and jax.process_index() == 0:
          row = dict(step=step, **{k: float(np.asarray(v)) for k, v in jax.device_get(
              {k: v for k, v in metrics['scalar'].items() if k in (
                  'learning/loss', 'learning/raw_grad_norm', 'learning/grad_norm', 'learning/total_weights')}).items()})
          with (args.output / 'replay-metrics.jsonl').open('a') as handle:
            handle.write(json.dumps(row) + '\n')
        if candidate is not None and (step == args.isolate_step or raw_norm > args.capture_first_raw_above):
          captured.update(candidate, step=step)
          captured['observed'] = select_metrics(metrics)
          print('ISOLATION_STATE_CAPTURED', step, captured['observed']['learning/raw_grad_norm'], flush=True)
        if frozen_batch is not None:
          frozen_batch['observed'] = select_metrics(metrics)
          captured.setdefault('frozen_batches', []).append(frozen_batch)
        del candidate
        return result, metrics

      class CapturingCompiled:
        def __call__(self, state, data, rng):
          return call(state, data, rng)

        def __getattr__(self, name):
          return getattr(compiled, name)

      return CapturingCompiled()

    train.maxtext_utils.load_compiled = load
  args.output.mkdir(parents=True, exist_ok=True)
  metadata = dict(runtime_commit=actual, start_step=args.start_step,
      end_step_exclusive=args.end_step, overrides=values,
      runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      original_loop_sha256=hashlib.sha256(source.encode()).hexdigest(),
      bounded_loop_sha256=hashlib.sha256(bounded.encode()).hexdigest(),
      isolate_step=args.isolate_step, capture_first_raw_above=args.capture_first_raw_above,
      capture_from_step=capture_from, force_save_steps=force_save_steps,
      frozen_batch_steps=frozen_batch_steps, max_updates=args.max_updates,
      scope='Diagnostic checkpoint copy; original serialized train-step and schedule unchanged.')
  (args.output / 'invocation.json').write_text(json.dumps(metadata, indent=2) + '\n')
  train.main([None, 'MaxText/configs/base.yml', *overrides])
  print('BOUNDED_AOT_REPLAY_COMPLETE', flush=True)
  if needs_capture:
    assert 'observed' in captured, 'Expected onset was not reproduced; inspect replay metrics before interpreting'
    captured_step = captured['step']
    if jax.process_index() == 0:
      (args.output / 'captured-step.json').write_text(json.dumps(dict(
          step=captured_step, trigger_threshold=args.capture_first_raw_above,
          metrics=captured['observed']), indent=2) + '\n')
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
    if frozen_batch_steps:
      comparisons = []
      assert len(captured.get('frozen_batches', [])) == len(frozen_batch_steps)
      for sample in captured['frozen_batches']:
        repeated, metrics = captured['compiled'](copy_tree(sample['state']), sample['data'], sample['rng'])
        actual = select_metrics(metrics)
        del repeated
        for key in ('learning/raw_grad_norm', 'learning/loss'):
          ref = sample['observed'][key]
          assert abs(actual[key] - ref) <= 1e-5 * max(1., abs(ref)), (sample['step'], key, actual[key], ref)
        # Keep the original future optimizer moments, counters and RNG; only
        # replace parameters. Ignore the returned update: this is read-only.
        frozen = copy_tree(sample['state']).replace(params=copy_tree(captured['state'].params))
        updated, metrics = captured['compiled'](frozen, sample['data'], sample['rng'])
        counterfactual = select_metrics(metrics)
        del updated
        comparisons.append(dict(step=sample['step'], frozen_parameters_before_step=captured_step,
            actual=actual, frozen_parameters=counterfactual,
            raw_ratio=counterfactual['learning/raw_grad_norm'] / max(actual['learning/raw_grad_norm'], 1e-30),
            loss_difference=counterfactual['learning/loss'] - actual['learning/loss']))
        if jax.process_index() == 0:
          (args.output / 'frozen-batch-comparison.json').write_text(json.dumps(comparisons, indent=2) + '\n')
          print('FROZEN_PARAMETER_BATCH_CHECK', sample['step'], comparisons[-1]['raw_ratio'], flush=True)
    if args.skip_loss_isolation:
      return
    if args.tensor_probe_script:
      assert 0 <= args.probe_sequence < batch and 0 <= args.probe_token < length
      probe(args.probe_sequence, args.probe_sequence + 1, args.probe_token, args.probe_token + 1)
      captured['probe_token'] = args.probe_token
      import runpy
      namespace = runpy.run_path(str(args.tensor_probe_script.resolve()))
      namespace['run_probe'](train, captured, records[-1]['metrics'], args.output,
                             args.probe_sequence, args.probe_layer)
      return
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
          pre_update_step=captured_step, batch_sequence_index=bad_sequence,
          target_position=token_lo, selection='greedy largest mean-CE gradient in each binary split',
          note='Target-loss masks only; model inputs and original AOT unchanged. Each case restores the frozen state.',
          input_sha256=hashlib.sha256(sample['inputs'].tobytes()).hexdigest()), indent=2) + '\n')
    print('LOSS_MASK_ISOLATION_COMPLETE', bad_sequence, token_lo, flush=True)


if __name__ == '__main__':
  main()
