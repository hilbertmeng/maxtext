"""Validate a tapped SiTU MLP at its own frozen inputs/cotangents in FP32.

This does not assert equivalence between a tapped graph and an untapped full
network. Their separate fidelity metadata is retained alongside local errors.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--runtime-dir', type=Path, required=True)
  p.add_argument('--runtime-commit', required=True)
  p.add_argument('--checkpoint', required=True)
  p.add_argument('--capture-root', required=True)
  p.add_argument('--layer', type=int, required=True)
  p.add_argument('--output', type=Path, required=True)
  p.add_argument('--gate-limit', type=float, default=4.)
  p.add_argument('--up-limit', type=float, default=7.)
  args = p.parse_args()
  assert subprocess.check_output(['git', '-C', str(args.runtime_dir), 'rev-parse', 'HEAD'], text=True).strip() == args.runtime_commit
  os.chdir(args.runtime_dir)
  sys.path.insert(0, str(args.runtime_dir / 'MaxText'))
  import jax
  import jax.numpy as jnp
  import numpy as np
  import orbax.checkpoint as ocp
  import tensorflow as tf
  from flax import linen as nn
  from flax.core import unfreeze
  from swiglu_local_jacobian import analyze

  args.output.mkdir(exist_ok=True, parents=True)
  with tf.io.gfile.GFile(args.capture_root.rstrip('/') + '/identity-tap-fidelity.json') as f:
    graph_fidelity = json.load(f)
  assert graph_fidelity['target_layer'] == args.layer
  cp = ocp.PyTreeCheckpointer()
  metadata = cp.metadata(args.checkpoint).item_metadata.tree['params']['params']['decoder']['layers']['block']['mlp']
  sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
  abstract = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=sharding), metadata)
  target = {'params': {'params': {'decoder': {'layers': {'block': {'mlp': abstract}}}}}}
  restored = cp.restore(args.checkpoint, item=target, transforms={},
      restore_args=ocp.checkpoint_utils.construct_restore_args(target))
  cp.close()
  stacked = unfreeze(nn.unbox(restored['params']['params']['decoder']['layers']['block']['mlp']))
  assert all(x.shape[1] == 27 for x in jax.tree.leaves(stacked))
  params = jax.tree.map(lambda x: jnp.take(x, args.layer, axis=1), stacked)
  jax.block_until_ready(params)
  del restored, stacked, target, abstract

  def load(site, direction):
    with tf.io.gfile.GFile(args.capture_root.rstrip('/') + f'/{site}-{direction}.npy', 'rb') as f:
      return jnp.asarray(np.load(f), jnp.float32)

  x, y = load('mlp_input', 'forward'), load('mlp_output', 'forward')
  dx, dy = load('mlp_input', 'backward'), load('mlp_output', 'backward')
  token = int(jnp.argmax(jnp.linalg.norm(dx.reshape((-1, dx.shape[-1])), axis=-1)))
  result = analyze(params, x, dy, dx, y, token,
                   gate_limit=args.gate_limit, value_limit=args.up_limit)
  result.update(layer=args.layer, selected_token=token, checkpoint=args.checkpoint,
                runtime_commit=args.runtime_commit, graph_fidelity=graph_fidelity,
                interpretation='Local derivatives of the tapped function only; graph fidelity is a separate question')
  result['local_reference_passed'] = all(
      arm['output_relative_error'] <= .02 and arm['input_vjp_relative_error'] <= .02
      for arm in result['arms'])
  (args.output / 'local-mlp-vjp.json').write_text(json.dumps(result, indent=2) + '\n')
  print('TAPPED_MLP_LOCAL_RESULT', json.dumps(dict(layer=args.layer, token=token,
      passed=result['local_reference_passed'], arms=[dict(
          bf16=a['effective_bf16_weights'], output_error=a['output_relative_error'],
          vjp_error=a['input_vjp_relative_error'], gain=a['global_directional_gain'])
          for a in result['arms']])), flush=True)


if __name__ == '__main__':
  main()
