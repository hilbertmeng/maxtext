import jax
import jax.numpy as jnp

# -----------------------------------------------------------------
# 1. 定义一个带有自定义VJP（反向传播）的新函数
# -----------------------------------------------------------------
@jax.custom_vjp
def tpu_friendly_gather(inputs, indices):
  """
  与 jnp.take(inputs, indices, axis=0) 等价, 但有 TPU 优化的反向传播。
  """
  # 前向传播与 jnp.take 完全相同
  return jnp.take(inputs, indices, axis=0, unique_indices=False, indices_are_sorted=False)

# -----------------------------------------------------------------
# 2. 定义前向传播函数 (_fwd)
#    它需要返回输出结果和为了反向传播需要保存的中间变量 (residuals)
# -----------------------------------------------------------------
def _tpu_friendly_gather_fwd(inputs, indices):
  # 执行前向操作
  outputs = tpu_friendly_gather(inputs, indices)
  # 保存 inputs 的形状和 indices，供反向传播使用
  residuals = (inputs.shape, indices)
  return outputs, residuals

# -----------------------------------------------------------------
# 3. 定义反向传播函数 (_bwd)
#    这是优化的核心
# -----------------------------------------------------------------
def _tpu_friendly_gather_bwd(residuals, grad_outputs):
  """
  使用 argsort + segment_sum 实现高效的 scatter_add。
  """
  input_shape, indices = residuals
  
  # grad_outputs 是上游传来的梯度，对应于 gather 操作的输出
  # 我们需要计算相对于 gather 操作的输入 (inputs) 的梯度

  # 传统的、在 TPU 上较慢的实现方式：
  # grad_inputs = jnp.zeros(input_shape, dtype=grad_outputs.dtype)
  # grad_inputs = grad_inputs.at[indices].add(grad_outputs)

  # TPU 优化的实现方式：
  # 1. 获取对 indices 进行排序的排列，这会将相同的 index 值分组
  inverse_permutation = jnp.argsort(indices)
  
  # 2. 使用这个排列来重排梯度和索引
  #    现在，与同一个输入位置相关的梯度在 grad_outputs_permuted 中是相邻的
  grad_outputs_permuted = grad_outputs[inverse_permutation] # (BTK)D
  grad_inputs = grad_outputs_permuted.reshape(input_shape[0], -1, input_shape[-1]).sum(-2)
  # indices_sorted = indices[inverse_permutation]
  
  # # 3. 使用 segment_sum 进行分段求和
  # #    num_segments 是原始输入张量的大小
  # #    segment_ids 是排序后的索引
  # grad_inputs = jax.ops.segment_sum(
  #       data=grad_outputs_permuted,
  #       segment_ids=indices_sorted,
  #       num_segments=input_shape[0]
  #   )
  
  # gather 操作的梯度只与第一个参数 (inputs) 有关，与 indices 无关
  return (grad_inputs, None)

# -----------------------------------------------------------------
# 4. 将前向和反向传播函数注册到 tpu_friendly_gather
# -----------------------------------------------------------------
tpu_friendly_gather.defvjp(_tpu_friendly_gather_fwd, _tpu_friendly_gather_bwd)



@jax.custom_vjp
def tpu_gather_by_permutation(inputs, permutation_indices):
  """
  与 jnp.take(inputs, permutation_indices, axis=0) 等价, 但有 TPU 优化的反向传播。
  适用于索引 (permutation_indices) 是一个不重复的排列 (permutation) 的情况。
  """
  # 前向传播与 jnp.take 完全相同
  return jnp.take(inputs, permutation_indices, axis=0)

def _tpu_gather_by_permutation_fwd(inputs, permutation_indices):
  outputs = tpu_gather_by_permutation(inputs, permutation_indices)
  # 保存排列索引以计算其逆排列
  residuals = permutation_indices
  return outputs, residuals

def _tpu_gather_by_permutation_bwd(residuals, grad_outputs):
  permutation_indices = residuals
  
  # 计算逆排列
  inverse_permutation = jnp.argsort(permutation_indices)
  
  # 使用 gather (take) 操作来代替 scatter_add
  grad_inputs = jnp.take(grad_outputs, inverse_permutation, axis=0)
  
  return (grad_inputs, None)

tpu_gather_by_permutation.defvjp(_tpu_gather_by_permutation_fwd, _tpu_gather_by_permutation_bwd)