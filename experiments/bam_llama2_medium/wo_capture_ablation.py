"""Diagnostic taps isolating callback placement from custom-VJP boundaries.

These are ablations, not replacements for the gated scientific capture. Forward
callbacks may repeat under rematerialization; consumers must not count them as
independent samples. No mode manufactures a gradient or corrects its scale.
"""
def make_ablation_tap(writer, mode, mesh, batch_axes):
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P
    if mode not in ('identity', 'forward', 'cotangent'):
        raise ValueError('Unknown capture ablation: '+mode)
    axes = tuple(batch_axes) if isinstance(batch_axes, list) else batch_axes

    def local_forward(layer, a, o, h, rows, valid):
        jax.debug.callback(writer, layer, a, o, h, rows, valid,
                           partitioned=False, ordered=False)
        return jnp.int32(0)

    def local_backward(layer, g, rows, valid):
        jax.debug.callback(writer, layer, g, rows, valid,
                           partitioned=False, ordered=False)
        return jnp.int32(0)

    forward_capture = jax.shard_map(local_forward, mesh=mesh,
        in_specs=(P(), P(axes), P(axes), P(axes), P(axes), P(axes)),
        out_specs=P(), check_vma=False)
    backward_capture = jax.shard_map(local_backward, mesh=mesh,
        in_specs=(P(), P(axes), P(axes), P(axes)),
        out_specs=P(), check_vma=False)

    @jax.custom_vjp
    def identity(y):
        return y

    def identity_fwd(y):
        return y, ()

    def identity_bwd(saved, g):
        return (g,)

    identity.defvjp(identity_fwd, identity_bwd)

    @jax.custom_vjp
    def cotangent(y, rows, valid, layer):
        return y

    def cotangent_fwd(y, rows, valid, layer):
        # Deliberately retain no AV/O/H tensor; isolate true G observation.
        return y, (rows, valid, layer)

    def cotangent_bwd(saved, g):
        rows, valid, layer = saved
        backward_capture(layer, g, rows, valid)
        return g, None, None, None

    cotangent.defvjp(cotangent_fwd, cotangent_bwd)

    def tap(y, a, o, h, rows, valid, layer):
        if mode == 'identity':
            return identity(y)
        if mode == 'forward':
            # Ordinary forward callback: no custom reverse-mode rule at all.
            forward_capture(layer, a, o, h, rows, valid)
            return y
        return cotangent(y, rows, valid, layer)

    return tap
