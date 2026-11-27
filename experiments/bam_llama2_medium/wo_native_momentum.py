"""Recover an unclipped Adam-Pax gradient reference from original AOT moments.

For diagnostics only. Does not execute an optimizer or change model state.
Do not use for a clipped step without separately accounting for actual clipping.
"""

def find_wo_adam_state(opt_state):
    import jax
    candidates=[]
    for state in jax.tree.leaves(opt_state,is_leaf=lambda x:all(hasattr(x,k)for k in ('mu','nu','count'))):
        if not all(hasattr(state,k)for k in ('mu','nu','count')):continue
        try:
            state.mu['params']['decoder']['layers']['block']['self_attention']['out']['kernel']
        except (KeyError,TypeError):continue
        candidates.append(state)
    if len(candidates)!=1:raise ValueError('Require one unambiguous Adam-Pax W_O state')
    return candidates[0]


def recover_unclipped(before,after,beta1,beta2):
    import jax.numpy as jnp
    # Match d54 Adam-Pax: bias-corrected decay uses PRE-update count in FP32.
    count=before.count.astype(jnp.float32)
    def decay(beta):
        return beta*(1-jnp.power(beta,count))/(1-jnp.power(beta,count+1))
    def wo(tree):return tree['params']['decoder']['layers']['block']['self_attention']['out']['kernel'][:,:6]
    old_mu,new_mu=wo(before.mu),wo(after.mu)
    old_nu,new_nu=wo(before.nu),wo(after.nu)
    b1=decay(beta1).astype(new_mu.dtype);b2=decay(beta2).astype(new_nu.dtype)
    gradient=(new_mu-b1*old_mu)/(1-b1)
    expected_nu=(1-b2)*gradient**2+b2*old_nu
    return gradient,new_nu,expected_nu,after.count-before.count
