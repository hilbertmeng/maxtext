"""Reproduce the AOT/JIT optimizer construction mismatch without TPU training.

Run with the fixed MaxText CPU environment and PYTHONPATH=MaxText. This tests
the actual optimizer and rule builder, not just regex matching a fake name.
The live compiler at bef8312 omits wd_tree; normal train.setup_mesh_and_model
constructs and passes it. No training state or source is modified here.
"""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import optax
import exp
import optimizers
import train


def main():
  cfg = SimpleNamespace(
      wd_mults=exp.BamLlama2MediumV2C256RmsGeluAlphaMix.wd_mults,
      adam_weight_decay=.1, adam_b1=.9, adam_b2=.95, adam_eps=1e-8,
      adam_eps_root=0., opt_type='adam_pax')
  params = {'params': {'decoder': {'layers': {'block': {'self_attention': {
      'fetch_mix_scale': jnp.full((1,24), .25)}}}}}}
  grads = jax.tree.map(jnp.zeros_like, params)
  wd_tree = train.get_wd_tree(cfg, params)
  outputs = {}
  for label, rules in [('jit_rules', wd_tree), ('aot_omitted_rules', None)]:
    tx = optimizers.get_optimizer(cfg, lambda step: jnp.asarray(3e-4), rules)
    updates, _ = tx.update(grads, tx.init(params), params)
    result = optax.apply_updates(params, updates)
    outputs[label] = float(result['params']['decoder']['layers']['block']
                          ['self_attention']['fetch_mix_scale'][0,0])
  assert outputs['jit_rules'] == .25
  assert outputs['aot_omitted_rules'] < .25
  print(outputs)


if __name__ == '__main__':
  main()
