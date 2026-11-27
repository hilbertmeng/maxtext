"""Diagnostic taps isolating callback placement from custom-VJP boundaries.

These are ablations, not replacements for the gated scientific capture. Forward
callbacks may repeat under rematerialization; consumers must not count them as
independent samples. No mode manufactures a gradient or corrects its scale.
"""
def restore_producer_primal_layout(writer):
    """Undo wire transposes on host without casts or arithmetic.

    AV/O arrive BNTH, H arrives BNHT; the recorder receives canonical BTNH.
    This changes logical callback operands, not a guarantee of physical layout
    or TPU fusion preservation. Only compiled HLO and native gates test that.
    """
    def receive(layer, a, o, h, rows, valid):
        import numpy as np
        a,o,h,rows,valid=map(np.asarray,(a,o,h,rows,valid))
        if rows.ndim!=2 or a.ndim!=4 or a.shape!=o.shape or h.ndim!=4:
            raise ValueError('Invalid producer-layout ranks or shapes')
        b,t=rows.shape
        if a.shape[0]!=b or a.shape[2]!=t or h.shape!=(b,a.shape[1],a.shape[3],t):
            raise ValueError('Producer BNTH/BNHT layout mismatch')
        if a.dtype!=o.dtype or a.dtype!=h.dtype:
            raise ValueError('Producer operand dtype mismatch')
        writer(layer,a.transpose(0,2,1,3),o.transpose(0,2,1,3),h.transpose(0,3,1,2),rows,valid)
    return receive


def make_ablation_tap(writer, mode, mesh, batch_axes, layer_limit=None, primal_layout='canonical'):
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P
    if mode not in ('identity', 'forward', 'primal', 'cotangent'):
        raise ValueError('Unknown capture ablation: '+mode)
    if primal_layout not in ('canonical','producer'):
        raise ValueError('Unknown primal wire layout')
    if primal_layout!='canonical' and mode not in ('forward','primal'):
        raise ValueError('Primal layout only applies to primal observations')
    forward_writer=restore_producer_primal_layout(writer) if primal_layout=='producer' else writer
    axes = tuple(batch_axes) if isinstance(batch_axes, list) else batch_axes

    def local_forward(layer, a, o, h, rows, valid):
        if primal_layout=='producer':
            a,o,h=a.transpose(0,2,1,3),o.transpose(0,2,1,3),h.transpose(0,2,3,1)
        jax.debug.callback(forward_writer, layer, a, o, h, rows, valid,
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
        observe(layer, lambda: backward_capture(layer, g, rows, valid))
        return g, None, None, None

    cotangent.defvjp(cotangent_fwd, cotangent_bwd)

    @jax.custom_vjp
    def primal(y, a, o, h, rows, valid, layer):
        return y

    def primal_fwd(y, a, o, h, rows, valid, layer):
        return y, (a, o, h, rows, valid, layer)

    def primal_bwd(saved, g):
        a, o, h, rows, valid, layer = saved
        # Observe retained primal operands during backward, never observe G.
        # TPU forward callback/remat channel failure is a separate limitation.
        observe(layer, lambda: forward_capture(layer, a, o, h, rows, valid))
        return g, None, None, None, None, None, None

    primal.defvjp(primal_fwd, primal_bwd)

    def observe(layer, callback):
        def emit(_):
            callback()
            return jnp.int32(0)
        if layer_limit is None:
            emit(None)
        else:
            jax.lax.cond((layer >= 0) & (layer < layer_limit), emit,
                         lambda _: jnp.int32(0), None)

    def tap(y, a, o, h, rows, valid, layer):
        if mode == 'identity':
            return identity(y)
        if mode == 'primal':
            return primal(y, a, o, h, rows, valid, layer)
        if mode == 'forward':
            # Ordinary forward callback: no custom reverse-mode rule at all.
            observe(layer, lambda: forward_capture(layer, a, o, h, rows, valid))
            return y
        return cotangent(y, rows, valid, layer)

    return tap
