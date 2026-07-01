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

from typing import Any, Optional, Union
from absl import flags
from etils import epath
import flax
from flax.training import train_state
import grain.python as grain
import jax
import jax.numpy as jnp
import max_logging
from multihost_dataloading import MultiHostDataLoadIterator
import numpy as np
import orbax.checkpoint as ocp
import orbax.checkpoint.experimental.emergency.checkpoint_manager as emergency_checkpoint_manager
import orbax.checkpoint.experimental.emergency.replicator_checkpoint_manager as emergency_replicator_checkpoint_manager

# pylint: disable=too-many-positional-arguments

CheckpointManager = ocp.CheckpointManager
CheckpointManagerOptions = ocp.CheckpointManagerOptions
PyTreeCheckpointHandler = ocp.PyTreeCheckpointHandler
LocalCheckpointOptions = emergency_checkpoint_manager.LocalCheckpointOptions
PersistentCheckpointOptions = emergency_checkpoint_manager.PersistentCheckpointOptions


def create_orbax_checkpoint_manager(
    checkpoint_dir: str,
    enable_checkpointing: bool,
    use_async: bool,
    save_interval_steps: int,
    dataset_type: Optional[str] = "tfds",
    orbax_logger: Any = None,  # pytype: disable=attribute-error
    use_ocdbt: bool = True,
    use_zarr3: bool = True,
    config = None,
):
  """Returns specified Orbax (async or not) CheckpointManager or None if checkpointing is disabled."""
  if not enable_checkpointing:
    max_logging.log("Checkpointing disabled, not creating checkpoint manager.")
    return None
  max_logging.log("Creating checkpoint manager...")
  p = epath.Path(checkpoint_dir)

  if dataset_type == "grain":
    item_names = ("items", "iter")
  else:
    item_names = ("items",)

  # local storage checkpoint needs parent directory created
  p.mkdir(exist_ok=True, parents=True)
  # we need to use ocdbt and zarr3 to control max file size in the checkpoint
  # omitting `iter` uses default handler for `iter`
  item_handlers = {"items": PyTreeCheckpointHandler(use_ocdbt=use_ocdbt, use_zarr3=use_zarr3)}
  mngr = CheckpointManager(
      p,
      item_names=item_names,
      item_handlers=item_handlers,
      options=CheckpointManagerOptions(
          create=True,
          save_interval_steps=save_interval_steps,
          enable_async_checkpointing=use_async,
          max_to_keep=config.max_to_keep, # lsp: max save checkpoint nums nearby
          keep_period=config.keep_period, # lsp: step / keep_period would not be deleted
      ),
      logger=orbax_logger,
  )
  max_logging.log("Checkpoint manager created!")
  return mngr


def create_orbax_emergency_checkpoint_manager(
    local_checkpoint_dir: str,
    persistent_checkpoint_dir: str,
    global_mesh: jax.sharding.Mesh,
    abstract_state: Any,
    local_save_interval_steps: int,
    persistent_save_interval_steps: int,
    orbax_logger: Any = None,  # pytype: disable=attribute-error
):
  """Returns an emergency checkpoint manager."""
  flags.FLAGS.experimental_orbax_use_distributed_process_id = True
  max_logging.log("Creating emergency checkpoint manager...")

  options = emergency_checkpoint_manager.CheckpointManagerOptions(
      local=LocalCheckpointOptions(save_interval_steps=local_save_interval_steps),
      persistent=PersistentCheckpointOptions(save_interval_steps=persistent_save_interval_steps),
  )
  manager = emergency_checkpoint_manager.CheckpointManager(
      local_checkpoint_dir,
      epath.Path(persistent_checkpoint_dir),
      global_mesh=global_mesh,
      abstract_state=abstract_state,
      options=options,
      logger=orbax_logger,
  )

  max_logging.log("Emergency checkpoint manager created!")
  return manager


def create_orbax_emergency_replicator_checkpoint_manager(
    local_checkpoint_dir: str,
    save_interval_steps: int,
    global_mesh: jax.sharding.Mesh,
):
  """Returns an emergency replicator checkpoint manager."""
  flags.FLAGS.experimental_orbax_use_distributed_process_id = True
  max_logging.log("Creating emergency replicator checkpoint manager...")

  options = emergency_replicator_checkpoint_manager.ReplicatorCheckpointManagerOptions(
      save_interval_steps=save_interval_steps,
  )
  manager = emergency_replicator_checkpoint_manager.ReplicatorCheckpointManager(
      epath.Path(local_checkpoint_dir),
      options,
      global_mesh=global_mesh,
  )

  max_logging.log("Emergency replicator checkpoint manager created!")
  return manager


def print_save_message(step, async_checkpointing):
  if async_checkpointing:
    max_logging.log(f"Started an asynchronous checkpoint save for step {step}")
  else:
    max_logging.log(f"Saved a checkpoint at step {step}.")


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


def load_state_if_possible(
    checkpoint_manager: Union[CheckpointManager, None],
    data_iterator: Union[MultiHostDataLoadIterator, None],
    load_parameters_from_path: str,
    load_full_state_from_path: str,
    abstract_unboxed_pre_state: train_state.TrainState,
    enable_single_replica_ckpt_restoring: Optional[bool] = False,
    dataset_type: Optional[str] = "tfds",
    step: int = -1,  # -1 means latest
    config = False, # lsp
    load_params_skip_paths: Optional[tuple[tuple[str, ...], ...]] = None,
):
  """Loads TrainState as possible from the inputs.

  Args:
    checkpoint_manager: if the checkpoint_manager has a valid checkpoint, return
      that TrainState. This enables a full reload of a run in progress.
    load_parameters_from_path: if there is no checkpoint in the checkpoint manager,
      load parameters from a parameter only checkpoint at this path.
    load_full_state_from_path: if there is no checkpoint in the checkpoint manager,
      load full state from a full state checkpoint at this path.
    abstract_unboxed_pre_state: an unboxed, abstract TrainState that Orbax
      matches type against.
    enable_single_replica_ckpt_restoring: bool flag for restoring checkpoitn
      with SingleReplicaArrayHandler

  Returns:
    A tuple of (train_state, train_state_params) where full_train_state captures
     a full reload and train_state_params just the params for a partial reload.
     At most one will be non-None. Both can be None if neither checkpoint is
     set.
  """

  if checkpoint_manager is not None:
    max_logging.log("checkpoint manager exists so trying to load this run's existing checkpoint")

    # step = checkpoint_manager.latest_step() if step < 0 else step
    step = data_iterator.meta_dict.get('checkpoint_step') # lsp
    if config.only_eval:
      if not load_parameters_from_path:
        if config.eval_model_step >= 0:
          step = config.eval_model_step
        load_parameters_from_path = epath.Path(config.checkpoint_dir) / str(step) / 'items'
      step = None
      print(f'Only eval mode, load_parameters_from_path: {load_parameters_from_path} step: {step}')

    if step is not None:
      max_logging.log(f"restoring from this run's directory step {step}")

      def map_to_pspec(data):
        if not enable_single_replica_ckpt_restoring:
          return ocp.type_handlers.ArrayRestoreArgs(sharding=data.sharding)
        pspec = data.sharding.spec
        mesh = data.sharding.mesh
        replica_axis_index = 0
        replica_devices = _replica_devices(mesh.devices, replica_axis_index)
        replica_mesh = jax.sharding.Mesh(replica_devices, mesh.axis_names)
        single_replica_sharding = jax.sharding.NamedSharding(replica_mesh, pspec)

        return ocp.type_handlers.SingleReplicaArrayRestoreArgs(
            sharding=jax.sharding.NamedSharding(mesh, pspec),
            single_replica_sharding=single_replica_sharding,
            global_shape=data.shape,
            dtype=data.dtype,
        )

      if enable_single_replica_ckpt_restoring:
        array_handler = ocp.type_handlers.SingleReplicaArrayHandler(
            replica_axis_index=0,
            broadcast_memory_limit_bytes=1024 * 1024 * 1000,  # 1000 MB limit
        )
        ocp.type_handlers.register_type_handler(jax.Array, array_handler, override=True)

      restore_args = jax.tree_util.tree_map(
          map_to_pspec,
          abstract_unboxed_pre_state,
      )

      if isinstance(
          checkpoint_manager,
          (
              emergency_checkpoint_manager.CheckpointManager,
              emergency_replicator_checkpoint_manager.ReplicatorCheckpointManager,
          ),
      ):
        return (
            checkpoint_manager.restore(
                step,
                args=ocp.args.PyTreeRestore(item=abstract_unboxed_pre_state, restore_args=restore_args),
            ),
            None,
        )
      if (
          dataset_type == "grain"
          and data_iterator is not None
          and (checkpoint_manager.directory / str(step) / "iter").exists()
      ):
        return (
            checkpoint_manager.restore(
                step,
                args=ocp.args.Composite(
                    items=ocp.args.PyTreeRestore(
                        item=abstract_unboxed_pre_state,
                        restore_args=restore_args,
                    ),
                    iter=grain.PyGrainCheckpointRestore(data_iterator.local_iterator),
                ),
            ),
            None,
        )
      else:
        return (
            checkpoint_manager.restore(
                step,
                args=ocp.args.Composite(
                    items=ocp.args.PyTreeRestore(
                        item=abstract_unboxed_pre_state,
                        restore_args=restore_args,
                    )
                ),
            ),
            None,
        )

  if load_parameters_from_path != "":
    restored_params = load_params_from_path(
        load_parameters_from_path,
        abstract_unboxed_pre_state.params,
        load_params_skip_paths,
        unroll_scanned_layers=bool(config and getattr(config, "train_unroll_loaded_scanned_layers", False)),
        num_decoder_layers=getattr(config, "num_decoder_layers", None) if config else None,
        param_scan_axis=getattr(config, "param_scan_axis", 0) if config else 0,
    )
    return None, restored_params
  elif load_full_state_from_path != "":
    max_logging.log(f"restoring full state from {load_full_state_from_path=}")
    p = epath.Path(load_full_state_from_path)
    ckptr = ocp.StandardCheckpointer()
    restored = ckptr.restore(p, abstract_unboxed_pre_state)
    return {"items": restored}, None

  else:
    max_logging.log("No existing checkpoints found, not restoring checkpoint.")
    return None, None


def setup_checkpoint_logger(config) -> Any | None:  # pytype: disable=attribute-error
  """Setup checkpoint logger.
  Args:
    config
  Returns:
    CloudLogger
  """
  orbax_cloud_logger = None
  max_logging.log("Setting up checkpoint logger...")
  if config.enable_checkpoint_cloud_logger:
    logger_name = f"goodput_{config.run_name}"
    orbax_cloud_logger = ocp.logging.CloudLogger(
        options=ocp.logging.CloudLoggerOptions(job_name=config.run_name, logger_name=logger_name)
    )
    max_logging.log("Successfully set up checkpoint cloud logger.")

  return orbax_cloud_logger


def _without_nested_paths(tree, paths):
  tree_was_frozen = isinstance(tree, flax.core.FrozenDict)
  result = flax.core.unfreeze(tree)
  for path in paths or ():
    cur = result
    parents = []
    for key in path[:-1]:
      if key not in cur:
        cur = None
        break
      parents.append((cur, key))
      cur = cur[key]
    if cur is not None:
      cur.pop(path[-1], None)
      for parent, key in reversed(parents):
        child = parent.get(key)
        if isinstance(child, (dict, flax.core.FrozenDict)) and not child:
          parent.pop(key, None)
        else:
          break
  return flax.core.freeze(result) if tree_was_frozen else result


def _filter_tree_to_structure(tree, structure):
  tree_was_frozen = isinstance(tree, flax.core.FrozenDict)
  source = flax.core.unfreeze(tree)

  def filter_node(node, structure_node):
    if isinstance(node, (dict, flax.core.FrozenDict)) and isinstance(structure_node, (dict, flax.core.FrozenDict)):
      return {
          key: filter_node(value, structure_node[key])
          for key, value in node.items()
          if key in structure_node
      }
    return node

  filtered = filter_node(source, structure)
  return flax.core.freeze(filtered) if tree_was_frozen else filtered


def _checkpoint_metadata_tree(checkpointer, checkpoint_path):
  metadata = checkpointer.metadata(checkpoint_path)
  item_metadata = getattr(metadata, "item_metadata", None)
  return getattr(item_metadata, "tree", None)


def _tree_structure_matches(tree, structure):
  tree_is_mapping = isinstance(tree, (dict, flax.core.FrozenDict))
  structure_is_mapping = isinstance(structure, (dict, flax.core.FrozenDict))
  if tree_is_mapping or structure_is_mapping:
    if not tree_is_mapping or not structure_is_mapping:
      return False
    if set(tree.keys()) != set(structure.keys()):
      return False
    return all(_tree_structure_matches(tree[key], structure[key]) for key in tree)
  return True


def _layers_i_keys(decoder):
  return sorted(
      (key for key in decoder if isinstance(key, str) and key.startswith("layers_") and key[7:].isdigit()),
      key=lambda key: int(key.split("_")[-1]),
  )


def _with_inserted_scan_axis(x, num_decoder_layers, param_scan_axis):
  axis = min(param_scan_axis, len(x.shape))
  shape = x.shape[:axis] + (num_decoder_layers,) + x.shape[axis:]
  kwargs = {}
  sharding = getattr(x, "sharding", None)
  if isinstance(sharding, jax.sharding.NamedSharding):
    spec = list(sharding.spec)
    spec.insert(min(param_scan_axis, len(spec)), None)
    kwargs["sharding"] = jax.sharding.NamedSharding(sharding.mesh, jax.sharding.PartitionSpec(*spec))
  return jax.ShapeDtypeStruct(shape=shape, dtype=x.dtype, **kwargs)


def _with_removed_scan_axis(x, param_scan_axis):
  if not isinstance(x, jax.ShapeDtypeStruct):
    return jnp.take(x, 0, axis=param_scan_axis)
  axis = min(param_scan_axis, len(x.shape) - 1)
  shape = x.shape[:axis] + x.shape[axis + 1 :]
  kwargs = {}
  sharding = getattr(x, "sharding", None)
  if isinstance(sharding, jax.sharding.NamedSharding):
    spec = list(sharding.spec)
    if axis < len(spec):
      spec.pop(axis)
    kwargs["sharding"] = jax.sharding.NamedSharding(sharding.mesh, jax.sharding.PartitionSpec(*spec))
  return jax.ShapeDtypeStruct(shape=shape, dtype=x.dtype, **kwargs)


def _restore_target_with_scanned_layers(abstract_unboxed_params, num_decoder_layers, param_scan_axis):
  params_was_frozen = isinstance(abstract_unboxed_params, flax.core.FrozenDict)
  target = flax.core.unfreeze(abstract_unboxed_params)
  decoder = target.get("params", {}).get("decoder", {})
  layer_keys = _layers_i_keys(decoder)
  if not layer_keys:
    return abstract_unboxed_params
  if len(layer_keys) != num_decoder_layers:
    max_logging.log(
        f"Expected {num_decoder_layers} unscanned decoder layers but found {len(layer_keys)} layer keys."
    )
  layer0 = decoder[layer_keys[0]]
  decoder["layers"] = jax.tree_util.tree_map(
      lambda x: _with_inserted_scan_axis(x, len(layer_keys), param_scan_axis),
      layer0,
  )
  for key in layer_keys:
    decoder.pop(key, None)
  return flax.core.freeze(target) if params_was_frozen else target


def _unroll_restored_scanned_layers(restored_params, num_decoder_layers, param_scan_axis):
  params_was_frozen = isinstance(restored_params, flax.core.FrozenDict)
  params = flax.core.unfreeze(restored_params)
  decoder = params.get("params", {}).get("decoder", {})
  if "layers" not in decoder:
    return restored_params
  scanned_layers = decoder.pop("layers")
  for layer_idx in range(num_decoder_layers):
    decoder[f"layers_{layer_idx}"] = jax.tree_util.tree_map(
      lambda x, idx=layer_idx: (
          _with_removed_scan_axis(x, param_scan_axis)
          if isinstance(x, jax.ShapeDtypeStruct)
          else np.take(x, idx, axis=param_scan_axis)
          if isinstance(x, np.ndarray)
          else jnp.take(x, idx, axis=param_scan_axis)
      ),
      scanned_layers,
    )
  return flax.core.freeze(params) if params_was_frozen else params


def load_params_from_path(
    load_parameters_from_path,
    abstract_unboxed_params,
    skip_paths=None,
    unroll_scanned_layers=False,
    num_decoder_layers=None,
    param_scan_axis=0,
):
  """Load decode params from checkpoint at specified path."""
  assert load_parameters_from_path, "load_parameters_from_path is not defined."
  max_logging.log(f"restoring params from {load_parameters_from_path}")
  ckpt = epath.Path(load_parameters_from_path)
  ckptr = ocp.PyTreeCheckpointer()
  # This is a memory optimization. We don't want to restore the entire checkpoint - only the params.
  # Rather than pass the entire abstract state, which could unnecessarily restore opt_state and such and waste
  # memory, we instead specify here that we are just restoring the params field of the checkpoint
  # (which itself may be a dictionary containing a key named 'params').
  if skip_paths:
    max_logging.log(f"Skipping parameter restore for: {', '.join('/'.join(path) for path in skip_paths)}")
    abstract_unboxed_params = _without_nested_paths(abstract_unboxed_params, skip_paths)
  if unroll_scanned_layers:
    assert num_decoder_layers is not None, "num_decoder_layers is required when unrolling scanned layers."
    max_logging.log(
        f"Restoring scanned decoder/layers and unrolling to layers_0..layers_{num_decoder_layers - 1}."
    )
    abstract_unboxed_params = _restore_target_with_scanned_layers(
        abstract_unboxed_params, num_decoder_layers, param_scan_axis
    )

  restore_item = {"params": abstract_unboxed_params}
  if unroll_scanned_layers:
    checkpoint_tree = _checkpoint_metadata_tree(ckptr, ckpt)
    if checkpoint_tree is not None:
      restore_item = _filter_tree_to_structure(restore_item, checkpoint_tree)
      abstract_unboxed_params = restore_item["params"]
      if not _tree_structure_matches(restore_item, checkpoint_tree):
        max_logging.log(
            "Scanned restore target does not match checkpoint metadata; restoring full params before unroll."
        )
        restored = ckptr.restore(ckpt)
        restored_params = restored["params"]
        if skip_paths:
          restored_params = _without_nested_paths(restored_params, skip_paths)
        return _unroll_restored_scanned_layers(restored_params, num_decoder_layers, param_scan_axis)
    else:
      max_logging.log("Checkpoint metadata tree unavailable; restoring with unfiltered target.")

  restore_args = ocp.checkpoint_utils.construct_restore_args(abstract_unboxed_params)
  restore_kwargs = {"item": restore_item, "restore_args": {"params": restore_args}}
  # We intentionally pass only the params subtree here, while full training
  # checkpoints may also contain opt_state and step. Tell Orbax this partial
  # tree mismatch is expected.
  if not (unroll_scanned_layers and checkpoint_tree is not None):
    restore_kwargs["partial_restore"] = True
  restored = ckptr.restore(ckpt, args=ocp.args.PyTreeRestore(**restore_kwargs))
  restored_params = restored["params"]
  if unroll_scanned_layers:
    restored_params = _unroll_restored_scanned_layers(restored_params, num_decoder_layers, param_scan_axis)
  return restored_params


def save_params_to_path(checkpoint_dir, params):
  """Save decode params in checkpoint at specified path."""
  assert checkpoint_dir, "checkpoint_dir is not defined."
  orbax_checkpointer = ocp.PyTreeCheckpointer()
  orbax_checkpointer.save(checkpoint_dir, {"params": params}, force=True)
  print(f"Quantized params checkpoint saved at: {checkpoint_dir}")
