"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Create an Orbax CheckpointManager with specified (Async or not) Checkpointer."""
import os
from typing import Optional, Union
from etils import epath
import orbax.checkpoint
from orbax.checkpoint.logging import abstract_logger, cloud_logger, standard_logger, composite_logger
from orbax.checkpoint import pytree_checkpoint_handler, type_handlers
from orbax.checkpoint.checkpoint_manager import CheckpointManager, CheckpointManagerOptions, PyTree
import orbax.checkpoint.experimental.emergency.checkpoint_manager as emergency_checkpoint_manager
import jax
import numpy as np
import grain.python as grain
from flax.traverse_util import flatten_dict, unflatten_dict

import max_logging
from multihost_dataloading import MultiHostDataLoadIterator
from flax.training import orbax_utils, train_state

PyTreeCheckpointHandler = pytree_checkpoint_handler.PyTreeCheckpointHandler
LocalCheckpointOptions = emergency_checkpoint_manager.LocalCheckpointOptions
PersistentCheckpointOptions = (
    emergency_checkpoint_manager.PersistentCheckpointOptions
)


def create_orbax_checkpoint_manager(config):
  """Returns specified Orbax (async or not) CheckpointManager or None if checkpointing is disabled."""
  if not config.enable_checkpointing:
    max_logging.log("Checkpointing disabled, not creating checkpoint manager.")
    return None
  max_logging.log("Creating checkpoint manager...")
  p = epath.Path(config.checkpoint_dir)

  if config.dataset_type=='c4-array_record':
    item_names = ('state', 'iter')
  else:
    item_names = ('state',)

  items = {
        "state": orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler(use_ocdbt=False)), # lsp
    }
  save_on_steps = getattr(config, 'save_on_steps', None)
  if save_on_steps:
    save_on_steps = list(range(*save_on_steps))
  mngr = CheckpointManager(
      p,
      items,
      # item_names = item_names,
      options = CheckpointManagerOptions(
          enable_background_delete=getattr(config, 'enable_background_delete', True), # lsp
          max_to_keep=getattr(config, 'max_to_keep', None), # lsp
          save_on_steps=save_on_steps, # lsp
          create=True,
          save_interval_steps=config.save_interval_steps,
          enable_async_checkpointing=config.use_async,
      )
  )
  max_logging.log("Checkpoint manager created!")
  return mngr


def create_orbax_emergency_checkpoint_manager(
    local_checkpoint_dir: str,
    persistent_checkpoint_dir: str,
    global_mesh: jax.sharding.Mesh,
    abstract_state: PyTree,
    local_save_interval_steps: int,
    persistent_save_interval_steps: int,
):
  """Returns an emergency checkpoint."""
  max_logging.log("Creating emergency checkpoint manager...")

  local_registry = type_handlers.create_type_handler_registry(
      (
          jax.Array,
          type_handlers.ArrayHandler(primary_host=None, replica_id=None),
      ),
  )

  local_checkpoint_handler = PyTreeCheckpointHandler(
      use_ocdbt=True,
      use_zarr3=True,
      primary_host=None,
      type_handler_registry=local_registry,
  )

  options = emergency_checkpoint_manager.CheckpointManagerOptions(
      local=LocalCheckpointOptions(
          save_interval_steps=local_save_interval_steps
      ),
      persistent=PersistentCheckpointOptions(
          save_interval_steps=persistent_save_interval_steps
      ),
  )

  emergency_mngr = emergency_checkpoint_manager.CheckpointManager(
      local_checkpoint_dir,
      epath.Path(persistent_checkpoint_dir),
      global_mesh=global_mesh,
      abstract_state=abstract_state,
      options=options,
      local_state_handler=local_checkpoint_handler,
  )

  max_logging.log("Emergency checkpoint manager created!")
  return emergency_mngr


def _find_idx(array: np.ndarray, replica_axis_idx: int):
  """Returns the index along given dimension that the current host belongs to."""
  idx = None
  for idx, val in np.ndenumerate(array):
    if val.process_index == jax.process_index():
      break
  return idx[replica_axis_idx]


def _replica_devices(device_array: np.ndarray, replica_axis_idx: int):
  """Returns the devices from the replica that current host belongs to.

  Replicas are assumed to be restricted to the first axis.

  Args:
    device_array: devices of the mesh that can be obtained by mesh.devices()
    replica_axis_idx: axis dimension along which replica is taken

  Returns:
    devices inside the replica that current host is in
  """
  idx = _find_idx(device_array, replica_axis_idx)
  replica_result = np.take(device_array, idx, axis=replica_axis_idx)
  return np.expand_dims(replica_result, axis=replica_axis_idx)


def load_state_if_possible(checkpoint_manager: CheckpointManager,
                           data_iterator: Union[MultiHostDataLoadIterator, None],
                           load_parameters_path: str,
                           load_full_state_path: str,
                           abstract_unboxed_pre_state: train_state.TrainState,
                           enable_single_replica_ckpt_restoring: Optional[bool] = False,
                           dataset_type: Optional[str] = "tfds",
                           checkpoint_dir: Optional[str] = None,
                           ):
  job_dir = epath.Path(checkpoint_dir)
  max_logging.log(f'job_dir: {job_dir}')
  meta_dict = data_iterator.meta_dict
  checkpoint_step = meta_dict.get('checkpoint_step', None) # 如果存在meta dict则自动加载最新模型
  if load_full_state_path:
    checkpoint_dir = epath.Path(load_full_state_path)
    max_logging.log(f"restoring state from {load_full_state_path=}")
    load_step = os.path.basename(load_full_state_path)
    try:
      load_step = int(load_step)
    except Exception as error:
      error = f'Error: {error}, please check whether ‘load_parameters_path’ endswith step number'
      raise ValueError(error)
    print(f'abstract_unboxed_pre_state: \n\n{abstract_unboxed_pre_state}\n\n')
    state = checkpoint_manager.restore(load_step, items={"state": abstract_unboxed_pre_state})
    return state, None

  elif load_parameters_path:
    checkpoint_dir = epath.Path(load_parameters_path)
    max_logging.log(f"restoring params from {load_parameters_path=}")
    load_step = os.path.basename(load_parameters_path)
    try:
      load_step = int(load_step)
    except Exception as error:
      error = f'Error: {error}, please check whether ‘load_parameters_path’ endswith step number'
      raise ValueError(error)
    params_shapedtype = abstract_unboxed_pre_state['params'] if isinstance(abstract_unboxed_pre_state, dict) else abstract_unboxed_pre_state.params
    state = checkpoint_manager.restore(load_step, items={"state": {"params": params_shapedtype}})
    restored = state['state']
    return None, restored['params']

  elif checkpoint_step is not None:
    max_logging.log(f"restoring params from ’{job_dir}‘ checkpoint_step: {checkpoint_step}")
    checkpoint_dir = job_dir / str(checkpoint_step) / 'state'
    ckptr = orbax.checkpoint.StandardCheckpointer()
    restored = ckptr.restore(checkpoint_dir, args=orbax.checkpoint.args.StandardRestore(abstract_unboxed_pre_state))
    return  {'state': restored}, None
    
  else:
    max_logging.log("No existing checkpoints found, not restoring checkpoint.")
    return None, None


def setup_checkpoint_logger(config) -> composite_logger.CompositeLogger | None:
  """Setup checkpoint logger.
  Args:
    config
  Returns:
    CompositeLogger
  """
  orbax_cloud_logger = None
  orbax_standard_logger = None
  max_logging.log("Setting up checkpoint logger...")
  if config.enable_checkpoint_cloud_logger:
    logger_name = f"checkpoint_{config.run_name}"
    options = cloud_logger.CloudLoggerOptions(
        job_name=config.run_name, logger_name=logger_name
    )
    orbax_cloud_logger = cloud_logger.CloudLogger(options=options)
    max_logging.log("Successfully set up checkpoint cloud logger.")

  if config.enable_checkpoint_standard_logger:
    orbax_standard_logger = standard_logger.StandardLogger()
    max_logging.log("Successfully set up checkpoint standard logger.")

  orbax_logger = None
  if orbax_cloud_logger is not None and orbax_standard_logger is not None:
    orbax_logger = composite_logger.CompositeLogger(
        orbax_cloud_logger, orbax_standard_logger
    )
    max_logging.log("Successfully set up checkpoint composite logger.")

  return orbax_logger


def load_params_from_path(load_parameters_from_path, abstract_unboxed_params):
  """Load decode params from checkpoint at specified path."""
  assert load_parameters_from_path, "load_parameters_from_path is not defined."
  max_logging.log(f"restoring params from {load_parameters_from_path}")
  ckpt = epath.Path(load_parameters_from_path)
  ckptr = orbax.checkpoint.PyTreeCheckpointer()
  # This is a memory optimization. We don't want to restore the entire checkpoint - only the params.
  # Rather than pass the entire abstract state, which could unnecessarily restore opt_state and such and waste
  # memory, we instead specify here that we are just restoring the params field of the checkpoint
  # (which itself may be a dictionary containing a key named 'params').
  restore_args = orbax.checkpoint.checkpoint_utils.construct_restore_args(abstract_unboxed_params)
  restored = ckptr.restore(
    ckpt,
    item={"params": abstract_unboxed_params},
    transforms={},
    restore_args={"params": restore_args}
    )
  return restored["params"]


def save_params_to_path(checkpoint_dir, params):
  """Save decode params in checkpoint at specified path."""
  assert checkpoint_dir, "checkpoint_dir is not defined."
  orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
  save_args = orbax_utils.save_args_from_target({"params":params})
  orbax_checkpointer.save(
    checkpoint_dir,
    {"params":params},
    save_args=save_args,
    force=True
    )
  print(f"Quantized params checkpoint saved at: {checkpoint_dir}")