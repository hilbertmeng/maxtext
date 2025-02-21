
# noam
import jax.numpy as jnp

class NoamScheduler:
    def __init__(self, d_model, warmup_steps):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.scale = d_model ** -0.5

    def __call__(self, step_num):
        step_num = np.maximum(step_num, 1)  # 防止 step_num 为 0
        lr = self.scale * np.minimum(
            step_num ** -0.5,
            step_num * self.warmup_steps ** -1.5
        )
        return lr


d_model = 512  # 模型维度
warmup_steps = 100

noam_scheduler = NoamScheduler(d_model, warmup_steps)


# 测试学习率调度器
steps = np.arange(1, 5000)
learning_rates = np.array([noam_scheduler(step) for step in steps])

import matplotlib.pyplot as plt
plt.plot(steps, learning_rates)
plt.title("Noam Learning Rate Schedule")
plt.xlabel("Step")
plt.ylabel("Learning Rate")
plt.show()




# cosine

import functools
import time
import socket
import subprocess


import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
import orbax.checkpoint as ocp


import json
import yaml
import flax
from flax.training import train_state
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning

import optax
import os
from typing import Tuple



def create_learning_rate_schedule(config):
  if config.use_noam_schedule:
    return create_noam_learning_rate_schedule(config)

  def make_cos_schedule(init_lr, final_lr, len_steps):
    def schedule(step):
      pct = (step) / len_steps
      a = 0.5 * (jnp.cos(jnp.pi * pct) + 1)
      lr = init_lr * a + final_lr * (1 - a)
      return lr

    return schedule

  lr = config.learning_rate
  cos_final_lr = lr * config.cosine_learning_rate_final_fraction

  warmup_steps = int(
      config.learning_rate_schedule_steps * config.warmup_steps_fraction
  )
  cos_steps = config.learning_rate_schedule_steps - warmup_steps
  constant_zero_steps = config.steps - config.learning_rate_schedule_steps

  warmup_schedule = optax.linear_schedule(
      init_value=0.0, end_value=lr, transition_steps=warmup_steps
  )
  cos_schedule = make_cos_schedule(lr, cos_final_lr, cos_steps)
  constant_schedule = optax.constant_schedule(cos_final_lr)  # lsp: cos end lr, 0.0 -> cos_final_lr

  pieces = [warmup_schedule, cos_schedule]
  boundaries = [
      warmup_steps,
      warmup_steps + cos_steps,
  ]

  if constant_zero_steps > 0:
    pieces.append(constant_schedule)
    boundaries.append(warmup_steps + cos_steps + constant_zero_steps)
  # scheduler, 结束的步数
  return optax.join_schedules(pieces, boundaries)


class Config():
    use_noam_schedule = False
    cosine_learning_rate_final_fraction = 0.333
    learning_rate_schedule_steps = 10000
    warmup_steps_fraction = 0.01
    steps = 200000
    learning_rate = 1e-4
config = Config()
scheudle = create_learning_rate_schedule(config)
lrs = [scheudle(s) for s in range(1, 1000)]


# 测试学习率调度器
steps = np.arange(1, 1000)

import matplotlib.pyplot as plt
plt.plot(steps, lrs)
plt.title("Noam Learning Rate Schedule")
plt.xlabel("Step")
plt.ylabel("Learning Rate")
plt.show()

