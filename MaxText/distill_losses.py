
import jax
import jax.numpy as jnp


def distillation_loss(student_logits, teacher_logits, mask=None, temperature=2.0):
    # 维度保持 [batch, seq, vocab]
    student_log_probs = jax.nn.log_softmax(student_logits / temperature, axis=-1)
    teacher_probs = jax.nn.softmax(teacher_logits / temperature, axis=-1)
    distill_xent = optax.losses.kl_divergence(student_log_probs, teacher_probs)
    return distill_xent


def forward_kld_topk(logits, teacher_logits, k: int):
    logits = jnp.asarray(logits, dtype=jnp.bfloat16)
    teacher_logits = jnp.asarray(teacher_logits, dtype=jnp.bfloat16)

    if logits.shape != teacher_logits.shape:
        raise ValueError(
            f"Student logits shape {logits.shape} and "
            f"teacher logits shape {teacher_logits.shape} must be identical."
        )
    if logits.ndim < 1:
        raise ValueError("Logits must have at least one dimension (num_classes).")

    num_classes = teacher_logits.shape[-1]
    k_val = max(1, k)
    teacher_probs = jax.nn.softmax(teacher_logits, axis=-1)
    student_logprobs = jax.nn.log_softmax(logits, axis=-1)
    _, top_k_indices = jax.lax.top_k(teacher_logits, k=k_val)
    top_k_mask = jnp.sum(jax.nn.one_hot(top_k_indices, num_classes, dtype=jnp.bfloat16), axis=-2)
    effective_teacher_probs = teacher_probs * top_k_mask
    # safe_student_logprobs = jnp.where(
    #     effective_teacher_probs == 0.0, 
    #     0.0,                           
    #     student_logprobs              
    # )
    prod_probs = student_logprobs * effective_teacher_probs
    distill_xent = -jnp.sum(prod_probs, axis=-1)
    return distill_xent


def reverse_kld_topk(logits, teacher_logits, k: int):
    logits = jnp.asarray(logits, dtype=jnp.bfloat16)
    teacher_logits = jnp.asarray(teacher_logits, dtype=jnp.bfloat16)

    if logits.shape != teacher_logits.shape:
        raise ValueError(
            f"Student logits shape {logits.shape} and "
            f"teacher logits shape {teacher_logits.shape} must be identical."
        )
    if logits.ndim < 1:
        raise ValueError("Logits must have at least one dimension (num_classes).")

    num_classes = teacher_logits.shape[-1]
    k_val = max(1, k)

    teacher_logprobs = jax.nn.log_softmax(teacher_logits, axis=-1)
    student_probs = jax.nn.softmax(logits, axis=-1)

    _, top_k_indices = jax.lax.top_k(teacher_logits, k=k_val)
    top_k_mask = jnp.sum(jax.nn.one_hot(top_k_indices, num_classes, dtype=jnp.bfloat16), axis=-2)
    effective_teacher_logprobs = teacher_logprobs * top_k_mask
    # safe_student_probs = jnp.where(
    #     effective_teacher_logprobs == 0.0, 
    #     0.0,                           
    #     student_probs              
    # )
    prod_probs = student_probs * effective_teacher_logprobs
    distill_xent = -jnp.sum(prod_probs, axis=-1)
    return distill_xent


def bi_kld(logits, teacher_logits):
    logits = jnp.asarray(logits, dtype=jnp.bfloat16)
    teacher_logits = jnp.asarray(teacher_logits, dtype=jnp.bfloat16)
    inf_mask = jnp.isinf(logits) | jnp.isinf(teacher_logits)
    teacher_probs = jax.nn.softmax(teacher_logits, axis=-1)
    student_logprobs = jax.nn.log_softmax(logits, axis=-1)
    prod_probs = jnp.where(inf_mask, 0.0, student_logprobs * teacher_probs)
    distill_xent = -jnp.sum(prod_probs, axis=-1)
    return distill_xent
            

def skewed_forward_kl(logits, teacher_logits, lam=0.1):
    logits = jnp.asarray(logits, dtype=jnp.bfloat16)
    teacher_logits = jnp.asarray(teacher_logits, dtype=jnp.bfloat16)
    teacher_probs = jax.nn.softmax(teacher_logits, axis=-1)
    student_probs = jax.nn.softmax(logits, axis=-1)
    mixed_probs = lam * teacher_probs + (1-lam) * student_probs
    mixed_logprobs = jnp.log(mixed_probs)
    inf_mask = jnp.isinf(logits) | jnp.isinf(teacher_logits)
    prod_probs = jnp.where(inf_mask, 0.0, teacher_probs * mixed_logprobs)
    distill_xent = -jnp.sum(prod_probs, axis=-1)
    return distill_xent


def skewed_reverse_kl(logits, teacher_logits, lam=0.1):
    logits = jnp.asarray(logits, dtype=jnp.bfloat16)
    teacher_logits = jnp.asarray(teacher_logits)
    teacher_probs = jax.nn.softmax(teacher_logits, axis=-1)
    student_probs = jax.nn.softmax(logits, axis=-1)
    mixed_probs = (1-lam) * teacher_probs + lam * student_probs
    student_logprobs = jax.nn.log_softmax(logits, axis=-1)
    mixed_logprobs = jnp.log(mixed_probs)
    inf_mask = jnp.isinf(logits) | jnp.isinf(teacher_logits)
    prod_probs = jnp.where(inf_mask, 0.0, student_probs * mixed_logprobs)
    prod_probs -= jnp.where(inf_mask, 0.0, student_probs * student_logprobs)
    # prod_probs： b*seq
    distill_xent = -jnp.sum(prod_probs, axis=-1)
    return distill_xent
    

def compute_distill_loss(config, logits, teacher_logits):
    # distill_loss: b x seq
    print(f'distill_loss_method: {config.distill_loss_method}')
    if config.distill_loss_method == 'srkl':
      distill_xent = skewed_reverse_kl(logits, teacher_logits, lam=config.lam)
    elif config.distill_loss_method == 'skl':
      distill_xent = skewed_forward_kl(logits, teacher_logits, lam=config.lam)
    elif config.distill_loss_method == 'kl':
      distill_xent = bi_kld(logits, teacher_logits)
    elif config.distill_loss_method == 'rkl':
      distill_xent = bi_kld(teacher_logits, logits)
    elif config.distill_loss_method == 'topk_kl':
      assert config.distill_topk > 0
      distill_xent = forward_kld_topk(logits, teacher_logits, k=int(config.distill_topk))
      assert config.distill_topk > 0
    elif config.distill_loss_method == 'topk_rkl':
      distill_xent = reverse_kld_topk(logits, teacher_logits, k=int(config.distill_topk))
    else:
      distill_xent = distillation_loss(logits, teacher_logits, temperature=config.distill_temperature)
    return distill_xent
