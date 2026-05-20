import math
import json
import os
import random
from typing import Dict, List, Optional
import copy

import numpy as np
import max_logging
import tensorflow as tf
import jax
from jax import numpy as jnp
import multihost_dataloading
from google.cloud import storage
from etils import epath
from collections import defaultdict


class PileDatasets():
    def __init__(self,
                mesh: str = None,
                name: str = 'pile',
                path: Optional[str] = None,
                num_infeed_hosts: int = 0,
                reset_for_eval: bool = False,
                batch_size: int = 8,
                seq_len: int = 2048,
                repeat: int = 1,
                seed: int = 9876,
                task_features: Optional[dict] = None,
                shuffle_buffer_size: Optional[int] = None,
                pad_id: int = 0,
                drop_remainder: bool = True,
                iter_file_nums: int = 2, # 100  500 steps/file,
                meta_dict: Optional[dict] = None,
                num_batches_to_skip: Optional[int] = None,
                only_eval: bool = False,
                zero_loss: bool = True,
                mix_attn: bool = False,
                shift: bool = True,
                arc_grid_positions: bool = False,
                arc_data_processing: bool = False,
                arc_select_demo_pairs: bool = True,
                arc_loss_on_all_outputs: bool = False,
                arc_remove_output_padding: bool = False,
                strictly_follow_nvarc_tokenizer: bool = False,
                ):
        self.mesh = mesh
        self.name = name
        self.path = path
        self.num_infeed_hosts = num_infeed_hosts
        self.reset_for_eval = reset_for_eval
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.repeat = repeat
        self.seed = seed
        self.task_features = task_features
        self.shuffle_buffer_size = shuffle_buffer_size
        self.pad_id = pad_id
        self.drop_remainder = drop_remainder
        self.iter_file_nums = iter_file_nums
        self.meta_dict = meta_dict
        self.num_batches_to_skip = num_batches_to_skip
        self.only_eval = only_eval
        self.zero_loss = zero_loss
        self.batch_padding_size = 0
        self.mix_attn = mix_attn
        self.shift = shift
        self.arc_grid_positions = arc_grid_positions
        self.arc_data_processing = arc_grid_positions or arc_data_processing
        self.arc_compress = self.arc_data_processing
        self.arc_select_demo_pairs = arc_select_demo_pairs
        self.arc_loss_on_all_outputs = arc_loss_on_all_outputs
        self.arc_remove_output_padding = arc_remove_output_padding
        self.strictly_follow_nvarc_tokenizer = strictly_follow_nvarc_tokenizer
        
        self.__post_init__()
        
    def __post_init__(self):
        if self.num_infeed_hosts == 0:
            self.num_infeed_hosts = jax.process_count()
        self.global_batch_size = self.batch_size * self.num_infeed_hosts

        if not self.meta_dict or self.only_eval:
            self.meta_dict = {}
            self.init_meta()
        else:
            if self.meta_dict["file_in_data"] != 0:
                assert self.meta_dict["iter_file_nums"] == self.iter_file_nums, print(
                    f'iter_file_nums in meta_dict is not equal to cur args. => {self.meta_dict["iter_file_nums"]}≠'
                    f" {self.iter_file_nums}"
                )
            saved_global_batch_size = self.meta_dict.get("global_batch_size")
            if saved_global_batch_size is not None:
                assert saved_global_batch_size == self.global_batch_size, print(
                    f"global_batch_size in meta_dict is not equal to cur args. => {saved_global_batch_size}≠"
                    f" {self.global_batch_size}"
                )
            self.step_in_file = self.meta_dict.get('step_in_file')  # XD fix
            self.meta_dict["global_batch_size"] = self.global_batch_size
            self.meta_dict["num_infeed_hosts"] = self.num_infeed_hosts

        print(f'meta_dict: {self.meta_dict}')
        self.seed = self.meta_dict['seed']
        self.dataset = self.load_tfrecord_dataset(fnames=self.path)
        self._peek = None
        self._state_before_peek = None

    def init_meta(self):
        self.meta_dict = {
                "seed": self.seed,
                "cur_files": self.meta_dict.get('cur_files', []),
                "file_in_data": 0,
                "step_in_file": 0,
                "iter_file_nums": self.iter_file_nums,
                "checkpoint_step": self.meta_dict.get('checkpoint_step', None),
                "global_batch_size": self.global_batch_size,
                "num_infeed_hosts": self.num_infeed_hosts,
            }
        self.step_in_file = 0

 #   def peek_padded(self):
  #      return self.get_next_padded()

    def reset(self):
        self.init_meta()
        self.dataset = self.load_tfrecord_dataset(fnames=self.path)

    def __iter__(self):
        return self.get_next_padded()
    
    def __next__(self):
        return self.get_next_padded()

    def get_next_padded(self):
        if self._peek is not None:
          output = self._peek
          self._peek = None
          self._state_before_peek = None
          return output
        unpadded = next(self.dataset)
        pad_size = int(self.batch_padding_size)
        if pad_size == 0:
            return unpadded
        return jax.tree_util.tree_map(
            lambda x: np.pad(x, [[0, pad_size]] + [[0, 0]] * (x.ndim - 1)),
            unpadded,
        )

    def get_global_batch_size(self, train_input):
        return self.batch_size * self.num_infeed_hosts

    def _slice_global_batch_for_host(self, data):
        process_index = jax.process_index()
        start = process_index * self.batch_size
        end = start + self.batch_size
        return {key: value[start:end] for key, value in data.items()}

    def _parse_function(self, example_proto):
        feature_desc = {key: tf.io.VarLenFeature(tf.int64) for key in self.task_features}
        example = tf.io.parse_single_example(example_proto, feature_desc)
        read_len = 8 * 2048 + 2 if self.arc_data_processing else self.seq_len + 1
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.cast(t, dtype=tf.int32)
            example[name] = tf.sparse.to_dense(t, default_value=0)[:read_len]
        return example

    def _select_arc_pairs(self, example):
        """Per-sample: if >4 pairs, randomly select 3 demos + last; otherwise keep all."""
        PAIR_TOKENS = 2048
        BOS, EOS = 75, 76
        MAX_DEMOS = 3

        feat_key = self.task_features[0]
        tokens = example[feat_key]

        inner = tokens[1:]  # skip BOS
        inner = tf.where(tf.equal(inner, EOS), 0, inner)

        content_len = tf.reduce_sum(tf.cast(inner != 0, tf.int32))
        n_pairs = content_len // PAIR_TOKENS
        n_demos = n_pairs - 1

        pairs = tf.reshape(inner[:n_pairs * PAIR_TOKENS], [n_pairs, PAIR_TOKENS])
        last_pair = pairs[-1:]
        demo_pairs = pairs[:-1]

        # Always select min(n_demos, 3) demos -- avoids tf.cond shape mismatch
        n_select = tf.minimum(n_demos, MAX_DEMOS)
        indices = tf.sort(tf.random.shuffle(tf.range(n_demos))[:n_select])
        selected = tf.gather(demo_pairs, indices)

        result = tf.concat([[BOS], tf.reshape(selected, [-1]), tf.reshape(last_pair, [-1]), [EOS]], axis=0)
        example[feat_key] = result
        return example

    def build_attn_mask(self):
        if not self.mix_attn:
            return tf.ones([self.batch_size, self.seq_len], dtype=tf.int32)
        p = 0.4                         
        body = tf.ones([self.batch_size, self.seq_len - 1], dtype=tf.int32)
        mask  = tf.random.uniform([self.batch_size, 1]) < p
        last_column  = tf.where(mask,
                                tf.zeros([self.batch_size, 1], dtype=tf.int32),   # 选中 → 0
                                tf.ones ([self.batch_size, 1], dtype=tf.int32))   # 未选 → 1
        inputs_segmentation = tf.concat([body, last_column], axis=1)
        return inputs_segmentation
    
    def _build_arc_position_ids(self, tokens):
        """Build grid-aligned position IDs for ARC sequences.

        Handles BOS/EOS: treats them as markers with reserved positions.
        Assigns position IDs so grid content aligns with Golden Gate RoPE
        4D grid: d0=pair, d1=input/output, d2=row, d3=column.

        Returns (tokens, position_ids, loss_mask).
        """
        PAIR_TOKENS = 2048
        OUTPUT_OFFSET = 1025
        BOS, EOS = 75, 76
        MAX_LEN = 8 * PAIR_TOKENS + 2  # max 8 pairs + BOS + EOS

        tokens = tokens[:, :MAX_LEN]
        L = tf.shape(tokens)[1]
        seq_pos = tf.range(L)

        # adj_pos = seq_pos - 1 to skip BOS at position 0
        adj_pos = seq_pos - 1
        p_in_pair = adj_pos % PAIR_TOKENS

        # Markers: BOS, EOS, I (p_in_pair==0), O (p_in_pair==1024)
        is_bos_eos = tf.equal(seq_pos, 0)  # BOS at pos 0
        # EOS detected by token value (per-sequence, broadcast)
        is_eos = tf.reduce_any(tf.equal(tokens, EOS), axis=0)  # approximate: just use token value below
        is_io = tf.equal(p_in_pair, 0) | tf.equal(p_in_pair, 1024)
        is_marker = is_bos_eos | is_io

        marker_pos = L - 1
        grid_pos = (adj_pos // PAIR_TOKENS) * PAIR_TOKENS + p_in_pair - 1
        pos = tf.where(is_marker, marker_pos, grid_pos)

        # Also mark BOS/EOS tokens as markers per-sequence (for position broadcast)
        tokens_is_bos_eos = tf.equal(tokens, BOS) | tf.equal(tokens, EOS)
        pos_b = tf.broadcast_to(pos[tf.newaxis], tf.shape(tokens))
        pos_b = tf.where(tokens_is_bos_eos, marker_pos, pos_b)

        # Loss mask: last pair's output only by default (counting from content after BOS).
        non_special = tf.cast((tokens != self.pad_id) & ~tokens_is_bos_eos, tf.int32)
        content_len = tf.reduce_sum(non_special, axis=1)
        n_pairs = content_len // PAIR_TOKENS

        adj_pos_b = tf.broadcast_to(adj_pos[tf.newaxis], tf.shape(tokens))
        output_pos = (p_in_pair >= OUTPUT_OFFSET) & (p_in_pair < PAIR_TOKENS)
        output_pos_b = tf.broadcast_to(output_pos[tf.newaxis], tf.shape(tokens))
        pair_idx_b = adj_pos_b // PAIR_TOKENS

        if self.arc_loss_on_all_outputs:
            loss_region = output_pos_b & (pair_idx_b >= 0) & (pair_idx_b < n_pairs[:, tf.newaxis])
        else:
            # Last output in adj_pos space: (n_pairs-1)*2048 + 1025 .. n_pairs*2048
            last_out_start = (n_pairs - 1) * PAIR_TOKENS + OUTPUT_OFFSET
            last_out_end = n_pairs * PAIR_TOKENS
            loss_region = (
                (adj_pos_b >= last_out_start[:, tf.newaxis]) &
                (adj_pos_b < last_out_end[:, tf.newaxis])
            )

        loss_mask = tf.cast(
            loss_region & ~tokens_is_bos_eos,
            tf.int32,
        )

        return tokens, pos_b, loss_mask

    def _compress_arc(self, tokens, pos, loss_mask, seq_len=None):
        """Remove dot padding tokens (id=5) and compact each ARC sequence.

        By default, keeps dots under the loss mask to preserve old last-output
        behavior. arc_remove_output_padding=True removes those dots as well.
        Uses argsort trick for vectorized per-sequence compaction.
        Zeros out positions beyond the kept count to prevent dot leakage.
        """
        if seq_len is None:
            seq_len = self.seq_len
        DOT_TOKEN = 5
        is_loss_output = tf.cast(loss_mask, tf.bool)
        if self.arc_remove_output_padding:
            keep = tf.not_equal(tokens, DOT_TOKEN)
        else:
            keep = tf.not_equal(tokens, DOT_TOKEN) | is_loss_output
        keep = keep & tf.not_equal(tokens, self.pad_id)

        indices = tf.argsort(tf.cast(~keep, tf.int32), axis=1, stable=True)
        tokens = tf.gather(tokens, indices, batch_dims=1)
        pos = tf.gather(pos, indices, batch_dims=1)
        loss_mask = tf.gather(loss_mask, indices, batch_dims=1)

        tokens = tokens[:, :seq_len]
        pos = pos[:, :seq_len]
        loss_mask = loss_mask[:, :seq_len]

        # Zero out positions beyond kept count (removed dots are non-zero token 5)
        num_kept = tf.reduce_sum(tf.cast(keep, tf.int32), axis=1)  # [B]
        valid = tf.range(seq_len)[tf.newaxis] < num_kept[:, tf.newaxis]
        tokens = tf.where(valid, tokens, 0)
        pos = tf.where(valid, pos, 0)
        loss_mask = tf.where(valid, loss_mask, 0)

        return tokens, pos, loss_mask

    @staticmethod
    def _old_arc_grid_to_nvarc_compact(grid_tokens):
        """Decode one old 32x31 bordered ARC grid into compact NVARC token ids."""
        rows = []
        for row_idx in range(32):
            start = row_idx * 32
            line = grid_tokens[start : start + 31]
            if len(line) < 31:
                break
            plus_positions = np.where(line == 2)[0]
            if plus_positions.size:
                break
            bar_positions = np.where(line == 73)[0]
            if not bar_positions.size:
                continue
            raw = line[: bar_positions[0]]
            digits = raw[(raw >= 7) & (raw <= 16)] - 7
            if digits.size:
                rows.append(digits.astype(np.int32))

        if not rows:
            return []

        pieces = []
        for i, row in enumerate(rows):
            if i:
                pieces.append(np.asarray([10], dtype=np.int32))
            pieces.append(row)
        return np.concatenate(pieces).astype(np.int32).tolist()

    def _arc_to_nvarc_compact_numpy(self, batch_tokens, output_len):
        """Convert old 86-token ARC records to compact 16-token NVARC chat records."""
        old_bos, old_eos = 75, 76
        pair_tokens = 2048
        im_start, im_end, eot = 14, 15, 13
        user, assistant, newline = 11, 12, 10

        batch_tokens = np.asarray(batch_tokens, dtype=np.int32)
        output_len = int(output_len)
        out_tokens = np.full((batch_tokens.shape[0], output_len), eot, dtype=np.int32)
        out_valid = np.zeros((batch_tokens.shape[0], output_len), dtype=np.int32)
        out_loss = np.zeros((batch_tokens.shape[0], output_len), dtype=np.int32)

        for batch_idx, sample in enumerate(batch_tokens):
            eos_positions = np.where(sample == old_eos)[0]
            eos_pos = int(eos_positions[0]) if eos_positions.size else len(sample)
            start = 1 if len(sample) and sample[0] == old_bos else 0
            inner = sample[start:eos_pos]
            num_pairs = len(inner) // pair_tokens

            compact = []
            loss = []
            for pair_idx in range(num_pairs):
                pair = inner[pair_idx * pair_tokens : (pair_idx + 1) * pair_tokens]
                input_grid = self._old_arc_grid_to_nvarc_compact(pair[1:1024])
                output_grid = self._old_arc_grid_to_nvarc_compact(pair[1025:2048])
                supervise_output = self.arc_loss_on_all_outputs or pair_idx == num_pairs - 1

                user_prefix = [im_start, user, newline]
                assistant_prefix = [im_end, im_start, assistant, newline]
                compact.extend(user_prefix)
                loss.extend([0] * len(user_prefix))
                compact.extend(input_grid)
                loss.extend([0] * len(input_grid))
                compact.extend(assistant_prefix)
                loss.extend([0] * len(assistant_prefix))
                compact.extend(output_grid)
                loss.extend([1 if supervise_output else 0] * len(output_grid))
                compact.append(im_end)
                loss.append(0)

            compact.append(eot)
            loss.append(0)

            n = min(output_len, len(compact))
            if n:
                out_tokens[batch_idx, :n] = np.asarray(compact[:n], dtype=np.int32)
                out_valid[batch_idx, :n] = 1
                out_loss[batch_idx, :n] = np.asarray(loss[:n], dtype=np.int32)

        return out_tokens, out_valid, out_loss

    def _build_nvarc_compact_tokens(self, tokens, output_len):
        out_tokens, out_valid, out_loss = tf.py_function(
            func=lambda x: self._arc_to_nvarc_compact_numpy(x, output_len),
            inp=[tokens],
            Tout=[tf.int32, tf.int32, tf.int32],
        )
        out_tokens.set_shape([self.batch_size, output_len])
        out_valid.set_shape([self.batch_size, output_len])
        out_loss.set_shape([self.batch_size, output_len])
        return out_tokens, out_valid, out_loss

    def convert(self, data):
        seq_len = self.seq_len
        feat_key = self.task_features[0] if self.task_features[0] in data else 'input_ids'
        model_needed_inputs = {}
        if self.shift:
            if self.arc_data_processing:
                if self.strictly_follow_nvarc_tokenizer:
                    tokens, valid, loss_mask = self._build_nvarc_compact_tokens(data[feat_key], seq_len + 1)
                    pos = tf.broadcast_to(tf.range(seq_len + 1)[tf.newaxis], tf.shape(tokens))
                    inputs = tokens[:, : seq_len]
                    input_seg = valid[:, : seq_len]
                    model_needed_inputs['inputs'] = inputs
                    model_needed_inputs['targets'] = tokens[:, 1: seq_len + 1]
                    model_needed_inputs['targets_segmentation'] = tf.cast(loss_mask[:, 1: seq_len + 1], dtype=tf.int32)
                    model_needed_inputs['inputs_segmentation'] = input_seg
                    model_needed_inputs['inputs_position'] = input_seg * pos[:, : seq_len]
                    model_needed_inputs['targets_position'] = valid[:, 1: seq_len + 1] * pos[:, 1: seq_len + 1]
                    return model_needed_inputs
                tokens, pos, loss_mask = self._build_arc_position_ids(data[feat_key])
                if self.arc_compress:
                    tokens, pos, loss_mask = self._compress_arc(tokens, pos, loss_mask, seq_len + 1)
                if not self.arc_grid_positions:
                    pos = tf.broadcast_to(tf.range(tf.shape(tokens)[1])[tf.newaxis], tf.shape(tokens))
                inputs = tokens[:, : seq_len]
                input_seg = tf.cast(inputs != self.pad_id, tf.int32)
                model_needed_inputs['inputs'] = inputs
                model_needed_inputs['targets'] = tokens[:, 1: seq_len + 1]
                model_needed_inputs['targets_segmentation'] = tf.cast(loss_mask[:, 1: seq_len + 1], dtype=tf.int32)
                model_needed_inputs['inputs_segmentation'] = input_seg
                model_needed_inputs['inputs_position'] = input_seg * pos[:, : seq_len]
                model_needed_inputs['targets_position'] = (
                    tf.cast(loss_mask[:, 1: seq_len + 1] != 0, tf.int32) * pos[:, 1: seq_len + 1]
                )
                return model_needed_inputs
            model_needed_inputs['inputs'] = data[feat_key][:, : seq_len]
            model_needed_inputs['targets'] = data[feat_key][:, 1: seq_len + 1]
            key = 'labels' if "labels" in data else feat_key
            weights = data[key] != self.pad_id
            model_needed_inputs['targets_segmentation'] = tf.cast(weights[:, 1: seq_len + 1], dtype=tf.int32)
        else:
            if self.arc_data_processing:
                if self.strictly_follow_nvarc_tokenizer:
                    tokens, valid, loss_mask = self._build_nvarc_compact_tokens(data[feat_key], seq_len)
                    pos = tf.broadcast_to(tf.range(seq_len)[tf.newaxis], tf.shape(tokens))
                    model_needed_inputs['inputs'] = tokens
                    model_needed_inputs['targets'] = tokens
                    model_needed_inputs['targets_segmentation'] = loss_mask
                    model_needed_inputs['inputs_segmentation'] = valid
                    model_needed_inputs['inputs_position'] = valid * pos
                    model_needed_inputs['targets_position'] = valid * pos
                    return model_needed_inputs
                tokens, pos, loss_mask = self._build_arc_position_ids(data[feat_key])
                if getattr(self, 'arc_compress', False):
                    tokens, pos, loss_mask = self._compress_arc(tokens, pos, loss_mask)
                if not self.arc_grid_positions:
                    pos = tf.broadcast_to(tf.range(tf.shape(tokens)[1])[tf.newaxis], tf.shape(tokens))
                model_needed_inputs['inputs'] = tokens
                model_needed_inputs['targets'] = tokens
                model_needed_inputs['targets_segmentation'] = loss_mask
                seg = tf.cast(tokens != self.pad_id, tf.int32)
                model_needed_inputs['inputs_segmentation'] = seg
                model_needed_inputs['inputs_position'] = seg * pos
                model_needed_inputs['targets_position'] = seg * pos
                return model_needed_inputs
            model_needed_inputs['inputs'] = data[feat_key][:, : seq_len]
            model_needed_inputs['targets'] = data[feat_key][:, : seq_len]
            weights = data[feat_key] != self.pad_id
            model_needed_inputs['targets_segmentation'] = tf.cast(weights[:, : seq_len], dtype=tf.int32)
        model_needed_inputs['inputs_segmentation'] = self.build_attn_mask()
        pos = tf.range(seq_len)
        model_needed_inputs['inputs_position'] = model_needed_inputs['inputs_segmentation'] * pos
        model_needed_inputs['targets_position'] = model_needed_inputs['inputs_segmentation'] * pos
        return model_needed_inputs

    def _load_file_dataset(self, fname):
        tf.random.set_seed(self.seed)
        ds = tf.data.Dataset.from_tensor_slices(fname)
        ds = ds.apply(tf.data.TFRecordDataset)
        ds = ds.map(self._parse_function, num_parallel_calls=tf.data.AUTOTUNE)
        if self.arc_data_processing and self.arc_select_demo_pairs and 'eval' not in self.name:
            ds = ds.map(self._select_arc_pairs, num_parallel_calls=tf.data.AUTOTUNE)
        print(f'shuffle_buffer_size: {self.shuffle_buffer_size}')
        if self.shuffle_buffer_size is not None:
            ds = ds.shuffle(buffer_size=self.shuffle_buffer_size)

        pad_len = 8 * 2048 + 2 if self.arc_data_processing else self.seq_len + 1
        padded_shapes = {key: pad_len for key in self.task_features}
        padding_values = {key: self.pad_id for key in self.task_features}
        ds = ds.padded_batch(
            batch_size=self.global_batch_size,
            padded_shapes=padded_shapes,
            padding_values=padding_values,
            drop_remainder=True,
        )
        if self.shuffle_buffer_size is not None:
            # batch化之后继续进行shuffle，让batch之间shuffle更加彻底
            ds = ds.shuffle(buffer_size=max(1, self.shuffle_buffer_size // self.global_batch_size))
        if self.step_in_file:
            ds = ds.skip(self.step_in_file)  # step_in_file is now the number of global batches already consumed
        # Build a process-count-independent global batch stream, then slice each host's local batch from it.
        ds = ds.map(self._slice_global_batch_for_host, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(self.convert, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        # local data to global data
        ds = multihost_dataloading.MultiHostDataLoadIterator(ds, self.mesh)

        return ds

    def load_tfrecord_dataset(self, fnames):
        tf.random.set_seed(self.seed)
        assert isinstance(fnames, list)
        import random as _random
        shuffled = list(fnames)
        _random.Random(self.seed).shuffle(shuffled)
        repeat_fnames = shuffled * self.repeat
        N = math.ceil(len(repeat_fnames) / self.iter_file_nums)
        file_in_data = self.meta_dict["file_in_data"]
        print(f'file_in_data: {file_in_data} N: {N}')
        for n in range(file_in_data, N, 1):
            fname = repeat_fnames[n * self.iter_file_nums : (n + 1) * self.iter_file_nums]
            self.meta_dict["cur_files"] = fname
            ds = self._load_file_dataset(fname)
            # ds = ds.as_numpy_iterator()
            for batch in ds:
                self.meta_dict["step_in_file"] += 1
                self.step_in_file += 1
                yield batch
            self.meta_dict["file_in_data"] += 1
            self.meta_dict["step_in_file"] = 0
            self.step_in_file = 0


SKIP_STEP_NAME = 'skip_file_and_step.json'
def record_file_and_step(step, config, train_input):  # lsp
    save_dir = epath.Path(config.checkpoint_dir)
    save_path = save_dir / str(step) / SKIP_STEP_NAME
    save_newest_path = save_dir / SKIP_STEP_NAME

    if not hasattr(train_input, 'meta_dict'):
        return
    meta_dict = train_input.meta_dict
    meta_dict['checkpoint_step'] = int(step)

    print(f'save_newest_path: {save_newest_path}')
    print(f'save_path: {save_path}')
    print(f'meta_dict: {meta_dict}')
    for k, v in meta_dict.items():
      print(k, type(v))

    if jax.process_index() == 0:
      try:
        with save_newest_path.open('w') as f1:
            json.dump(meta_dict, f1)

        with save_path.open('w') as f2:
            json.dump(meta_dict, f2)
      except Exception as error:
        print(f'Write meta dict error: {error}')

    print(f'Save skip_file_and_step successful... file_in_data: {meta_dict["file_in_data"]} || step_in_file: {meta_dict["step_in_file"]}')  # XD


def extract_pythia_datapath(dataset_path, eval_split):  # lsp
    if not dataset_path:
      return []
    client = storage.Client()
    path = dataset_path.replace('gs://', '')
    path_parts = path.split('/')
    bucket_name = path_parts[0]
    directory_path = '/'.join(path_parts[1:])
    directory_path = directory_path if directory_path.endswith('/') else directory_path + '/'
    print(f'bucket_name = {bucket_name}, directory_path = {directory_path}')
    step_map_path = {}
    eval_pathes = []
    rerank = 0
    for blob in client.list_blobs(bucket_name, prefix=directory_path):
        if ".tfrecord" not in blob.name: continue
        try:
            step = int(blob.name.rsplit("pile.tfrecord.b", maxsplit=1)[-1])
        except:
            step = rerank
            rerank += 1
        path = f'gs://{os.path.join(bucket_name, blob.name)}'

        if eval_split in path:
            print(f'eval path: {path}')
            eval_pathes.append(path)
            continue
        step_map_path[step] = path

    if not eval_pathes:
        eval_pathes = ['gs://newproject-1-common_datasets_europe-west4/pythia_model_test/pile_test/val_with_eos.tfrecord']
        
    sorted_step_path = sorted(step_map_path.items(), key=lambda x: x[0])
    steps, pathes = zip(*sorted_step_path)
    if not isinstance(pathes, list):
        pathes = list(pathes)
    max_logging.log(f'pathes: {len(pathes)} eval_pathes: {eval_pathes}')
    return pathes, eval_pathes


def extract_v3p5_longdata_files(dataset_path, eval_split=None):  # lsp
    random.seed(9876)
    client = storage.Client()
    #v3: us-east1-d -> common_datasets, v4: us-central2-b -> common_datasets_us-central2-b
    path = dataset_path.replace('gs://', '')
    path_parts = path.split('/')
    bucket_name = path_parts[0]
    directory_path = '/'.join(path_parts[1:])
    directory_path = directory_path if directory_path.endswith('/') else directory_path + '/'
    train_files, valid_files = [], []
    train_long_files, train_short_files = [], []
    for blob in client.list_blobs(bucket_name, prefix=directory_path):
        path = f'gs://{os.path.join(bucket_name, blob.name)}'
        if 'valid' in path:
            valid_files.append(path)
        else:
            if '.long' in path:
                train_long_files.append(path)
            else:
                train_short_files.append(path)
    # file size short：long = 1.5: 1, 为了保证short的token: long = 3: 7, 因此 short 取 (1 / 1.5) * (3 / 7) = 2 / 7
    short_k = min(3 * len(train_long_files) // 14, len(train_short_files))
    selected_short_files = random.sample(train_short_files, k=short_k)
    train_files = selected_short_files + train_long_files
    print(f'selected_short_files: {len(selected_short_files)} train_long_files: {len(train_long_files)}')
    random.shuffle(train_files)
    print(f'first 10 train files: {train_files[:10]}')
    valid_files = sorted(valid_files)
    print(f'valid_files: {valid_files}')
    return train_files, valid_files


def extract_v3p5_data_files(dataset_path, eval_split):
    random.seed(9876)
    client = storage.Client()
    path = dataset_path.replace('gs://', '')
    path_parts = path.split('/')
    bucket_name = path_parts[0]
    directory_path = '/'.join(path_parts[1:])
    directory_path = directory_path if directory_path.endswith('/') else directory_path + '/'
    print(f'bucket_name = {bucket_name}, directory_path = {directory_path}')
    train_files, valid_files = [], []
    for blob in client.list_blobs(bucket_name, prefix=directory_path):
        path = f'gs://{os.path.join(bucket_name, blob.name)}'
        if eval_split in path:
            valid_files.append(path)
        else:
            train_files.append(path)
    # train_files = sorted(train_files)
    # valid_files = sorted(valid_files)
    random.shuffle(train_files)
    print(f'Train file: {len(train_files)},  test file: {len(valid_files)}')
    print(f'first 10 train files: {train_files[:10]}')
    print(f'valid_files: {valid_files}')
    return train_files, valid_files


def extract_v3p5mini_data_files_qwen(dataset_path, eval_split, train_stage):

    random.seed(9876)
    client = storage.Client()
    path = dataset_path.replace('gs://', '')
    path_parts = path.split('/')
    bucket_name = path_parts[0]
    directory_path = '/'.join(path_parts[1:])
    directory_path1 = directory_path + 'B0-20/' if directory_path.endswith('/') else directory_path + '/B0-20/'
    directory_path2 = directory_path + 'B20-40/' if directory_path.endswith('/') else directory_path + '/B20-40/'
    directory_path3 = directory_path + 'B0-40-last/' if directory_path.endswith('/') else directory_path + '/B0-40-last/'
    valid_directory_path = directory_path + 'validation/' if directory_path.endswith('/') else directory_path + '/validation/'

    print(f'directory_path1: {directory_path1} 2: {directory_path2} 3: {directory_path3} valid_directory_path: {valid_directory_path}')

    rank_last_path = epath.Path(os.path.join(dataset_path, 'last_files.json'))
    with rank_last_path.open('r') as f:
        rank_last_files = json.load(f)['last_files']

    train_files, valid_files = [], []
    for directory_path in [directory_path1, directory_path2, directory_path3, valid_directory_path]:
        print(f'bucket_name = {bucket_name}, directory_path = {directory_path}')
        for blob in client.list_blobs(bucket_name, prefix=directory_path):
            path = f'gs://{os.path.join(bucket_name, blob.name)}'
            if path in rank_last_files:
                print(f'filter last file: {path}')
                continue
            if eval_split in path:
                valid_files.append(path)
            else:
                train_files.append(path)

    random.shuffle(train_files)
    print(f'Total train file: {len(train_files)},  test file: {len(valid_files)}')

    epoch = 2
    shuffled_train_files = copy.deepcopy(train_files)
    for e in range(epoch - 1):
        temp_train_files = copy.deepcopy(train_files)
        random.shuffle(temp_train_files)
        shuffled_train_files.extend(temp_train_files)
    train_files = shuffled_train_files

    if train_stage == 1:
        train_files = train_files[:1376 + 1] # +1是为了超出后不会报错
    elif train_stage == 2:
        train_files = train_files[1376: 1376*2 + 1]
    elif train_stage == 3:
        train_files = train_files[1376*2 :1376*6 + 1]
    else:
        # last_f = os.path.join(dataset_path, 'R051.000076')
        train_files = train_files[1376*6:]

    print(f'[S{train_stage}]Train file: {len(train_files)},  test file: {len(valid_files)}')
    print(f'[S{train_stage}]First 10 train files: {train_files[:10]}')
    print(f'[S{train_stage}]Valid_files: {valid_files}')
 
    return train_files, valid_files

# unigram
def extract_v3p5mini_data_files(dataset_path, eval_split, train_stage):

    random.seed(9876)
    client = storage.Client()
    path = dataset_path.replace('gs://', '')
    path_parts = path.split('/')
    bucket_name = path_parts[0]
    directory_path = '/'.join(path_parts[1:])
    directory_path1 = directory_path + 'B0-40/' if directory_path.endswith('/') else directory_path + '/B0-40/'
    directory_path2 = directory_path + 'B0-40-last/' if directory_path.endswith('/') else directory_path + '/B0-40-last/'
    valid_directory_path = directory_path + 'validation/' if directory_path.endswith('/') else directory_path + '/validation/'
    print(f'directory_path1: {directory_path1} 2: {directory_path2} valid_directory_path: {valid_directory_path}')
    
    if train_stage < 5:
        rank_last_path = epath.Path(os.path.join(dataset_path, 'last_files.json'))
        with rank_last_path.open('r') as f:
            rank_last_files = json.load(f)['last_files']
    else:
        rank_last_files = []

    train_files, valid_files = [], []
    for directory_path in [directory_path1, directory_path2, valid_directory_path]:
        print(f'bucket_name = {bucket_name}, directory_path = {directory_path}')
        for blob in client.list_blobs(bucket_name, prefix=directory_path):
            path = f'gs://{os.path.join(bucket_name, blob.name)}'
            if path in rank_last_files:
                print(f'filter last file: {path}')
                continue
            if eval_split in path:
                valid_files.append(path)
            else:
                train_files.append(path)

    random.shuffle(train_files)
    print(f'Total train file: {len(train_files)},  test file: {len(valid_files)}')
    epoch = 2 if train_stage != 5 else 1 # 第5阶段为32k训练，新的数据
    shuffled_train_files = copy.deepcopy(train_files)
    for e in range(epoch - 1):
        temp_train_files = copy.deepcopy(train_files)
        random.shuffle(temp_train_files)
        shuffled_train_files.extend(temp_train_files)
    train_files = shuffled_train_files
    print(f'Total repeat:{epoch} train file: {len(train_files)},  test file: {len(valid_files)}')

    if train_stage == 1:
        train_files = train_files[:1536 + 1] + train_files[-191:]
    elif train_stage == 2:
        train_files = train_files[1536: 1536*2 + 1] + train_files[:191]
    elif train_stage == 3:
        train_files = train_files[1536*2 :1536*6 + 1] + train_files[:191]
    elif train_stage == 4:
        # last_f = os.path.join(dataset_path, 'R051.000076')
        train_files = train_files[1536*6: ] + train_files[:100]

    print(f'[S{train_stage}]Train file: {len(train_files)},  test file: {len(valid_files)}')
    print(f'[S{train_stage}]First 10 train files: {train_files[:10]}')
    print(f'[S{train_stage}]Valid_files: {valid_files}')
 
    return train_files, valid_files


def extract_v4p5_1p5B_data_files2(dataset_path, eval_split):
    random.seed(9876)
    client = storage.Client()
    path = dataset_path.replace('gs://', '')
    path_parts = path.split('/')
    bucket_name = path_parts[0]
    directory_path = '/'.join(path_parts[1:])
    directory_path = directory_path if directory_path.endswith('/') else directory_path + '/'
    print(f'bucket_name = {bucket_name}, directory_path = {directory_path}')
    train_files = defaultdict(list)
    error_pathes = []
    for blob in client.list_blobs(bucket_name, prefix=directory_path):
        path = f'gs://{os.path.join(bucket_name, blob.name)}'
        # print(f'path: {path}')
        if 'packed' in path or '4k' in path:
            flag = False
            for dataset_name in ['algebraic-stack', 'arxiv', 'dclm', 'open-web-math', 'pes2o', 'starcoder', 'wiki']:
                if dataset_name in path:
                    flag = True
                    train_files[dataset_name].append(path) # 全量数据，因此之后需要shuffle 1/10数据
            if not flag:
                error_pathes.append(path)
                
    print(f'error_pathes: {len(error_pathes)} first 10 error_pathes: {error_pathes[:10]}')
    total_train_files, total_valid_files = [], []
    for dataset_name, pathes in train_files.items():
        random.shuffle(pathes)
        if dataset_name == 'dclm':
            sample_pathes = pathes[: -2]
            total_valid_files.extend(pathes[-2:]) # add last file as valid_files
        else:
            sample_pathes = pathes[: math.ceil(len(pathes) / 10)]
            total_valid_files.append(pathes[-1]) # add last file as valid_files

        print(f'dataset_name: {dataset_name}, pathes: {len(pathes)} sample_pathes: {len(sample_pathes)}')
        total_train_files.extend(sample_pathes) # add 1/10 data into total_train_files

    random.shuffle(total_train_files)
    random.shuffle(total_valid_files)

    print(f'Train file: {len(total_train_files)},  test file: {len(total_valid_files)}')
    print(f'first 10 train files: {total_train_files[:10]}')
    print(f'valid_files: {total_valid_files}')
    return total_train_files, total_valid_files


def extract_v4p5_1p5B_data_files(dataset_path, eval_split):
    random.seed(9876)
    client = storage.Client()
    path = dataset_path.replace('gs://', '')
    path_parts = path.split('/')
    bucket_name = path_parts[0]
    directory_path = '/'.join(path_parts[1:])
    directory_path = directory_path if directory_path.endswith('/') else directory_path + '/'
    print(f'bucket_name = {bucket_name}, directory_path = {directory_path}')
    total_valid_files = []
    total_train_files = []
    for blob in client.list_blobs(bucket_name, prefix=directory_path):
        path = f'gs://{os.path.join(bucket_name, blob.name)}'
        if eval_split in path:
            total_valid_files.append(path)
        else:
            total_train_files.append(path)
    # total_train_files.sort()
    random.shuffle(total_train_files)
    # random.shuffle(total_valid_files)
    # total_valid_files = total_valid_files + total_train_files[-6:] # add last 6 files as valid_files
    # total_train_files = total_train_files[:-6]
    print(f'Train file: {len(total_train_files)},  test file: {len(total_valid_files)}')

    total_train_files = total_train_files[1768: ] # remove first 2M batch trained 1000 files
    print(f'Train file2: {len(total_train_files)},  test file: {len(total_valid_files)}')

    print(f'first 10 train files: {total_train_files[:10]}')
    print(f'valid_files: {total_valid_files}')
    return total_train_files, total_valid_files


def extract_v4p5_1p5B_data_files_sec_stage(dataset_path, eval_split):
    random.seed(9876)
    client = storage.Client()
    path = dataset_path.replace('gs://', '')
    path_parts = path.split('/')
    bucket_name = path_parts[0]
    directory_path = '/'.join(path_parts[1:])
    directory_path = directory_path if directory_path.endswith('/') else directory_path + '/'
    print(f'bucket_name = {bucket_name}, directory_path = {directory_path}')
    total_valid_files = []
    total_train_files = []
    for blob in client.list_blobs(bucket_name, prefix=directory_path):
        path = f'gs://{os.path.join(bucket_name, blob.name)}'
        if eval_split in path:
            total_valid_files.append(path)
        else:
            total_train_files.append(path)
    random.shuffle(total_train_files)
    print(f'Train file: {len(total_train_files)},  test file: {len(total_valid_files)}')
    print(f'first 10 train files: {total_train_files[:10]}')
    print(f'valid_files: {total_valid_files}')
    return total_train_files, total_valid_files


def extract_train_skip_step(model_dir, step, only_eval=False):  # lsp
    if model_dir is None:
        return {}
    if step is not None:
        skip_file_and_step_path = model_dir / str(step) / SKIP_STEP_NAME
    else:
        skip_file_and_step_path = model_dir / SKIP_STEP_NAME
    print(f"model_dir: {model_dir}")
    try:
        with skip_file_and_step_path.open('r') as f:
            meta_dict = json.load(f)
        print(f"Load skip_file_and_step_path: ’{skip_file_and_step_path}‘ Finished.......")
    except:
        print(f"skip_file_and_step_path: ’{skip_file_and_step_path}‘ is not existed.......")
        meta_dict = {}

    if jax.process_index() == 0:
        mode = 'train_break_steps' if not only_eval else 'eval_metric_steps'
        back_meta_dict_dir = epath.Path(os.path.dirname(model_dir)) / mode # lsp
        if 'gs:' not in str(back_meta_dict_dir):
          os.makedirs(back_meta_dict_dir, exist_ok=True)
        back_meta_dict_path = back_meta_dict_dir /f'{meta_dict.get("checkpoint_step", None)}.json'
        with back_meta_dict_path.open('w') as f1:
            json.dump(meta_dict, f1)
    return meta_dict


def make_pile_train_iterator(config, mesh):  # lsp
  train_name = f'{config.dataset_type}.train'
  eval_name = f'{config.dataset_type}.eval'
  if config.dataset_type == 'pile':
    dataset_paths = [p.strip() for p in config.dataset_path.split(',')]
    train_pathes, eval_pathes = [], []
    for dp in dataset_paths:
      tp, ep = extract_pythia_datapath(dp, config.eval_split)
      train_pathes.extend(tp)
      eval_pathes.extend(ep)
    eval_dataset_path = getattr(config, 'eval_dataset_path', '')
    if eval_dataset_path:
      eval_pathes, _ = extract_pythia_datapath(eval_dataset_path, '__none__')
  elif config.dataset_type == 'novel_4_32k':
    train_pathes, eval_pathes = extract_v3p5_longdata_files(config.dataset_path, config.eval_split)
  elif config.dataset_type == 'pretrain_4k':
    train_pathes, eval_pathes = extract_v3p5_data_files(config.dataset_path, config.eval_split)
  elif config.dataset_type == 'xm3.5mini':
    train_pathes, eval_pathes = extract_v3p5mini_data_files(config.dataset_path, config.eval_split, config.train_stage)
  elif config.dataset_type == 'v4.5_1.5B':
     train_pathes, eval_pathes = extract_v4p5_1p5B_data_files(config.dataset_path, config.eval_split)
  elif config.dataset_type == 'v4.5_1.5B_sec_stage':
     train_pathes, eval_pathes = extract_v4p5_1p5B_data_files_sec_stage(config.dataset_path, config.eval_split)
  else:
    raise ValueError(f'Unknow ‘config.datase_dtype’={config.datase_dtype}')

  num_local_devices = jax.local_device_count()

  job_dir = epath.Path(config.checkpoint_dir)
  try:
    only_eval = config.only_eval
  except:
    only_eval = False
  meta_dict = extract_train_skip_step(job_dir,  step=config.training_num_batches_to_skip, only_eval=only_eval)
  # load_full_state_path
  print(f'meta_dict: {meta_dict}')

  task_features = config.task_features
  train_dataloader = PileDatasets(
                            mesh=mesh,
                            name=train_name, 
                            path=train_pathes, 
                            meta_dict=meta_dict,
                            batch_size=int(config.per_device_batch_size * num_local_devices),
                            seq_len=config.max_target_length,
                            repeat=config.epoch,
                            seed=config.data_shuffle_seed,
                            task_features=task_features,
                            shuffle_buffer_size=config.train_shuffle_buffer_size,
                            num_batches_to_skip=None,
                            only_eval=False,
                            zero_loss=config.zero_loss,
                            iter_file_nums=config.iter_file_nums,
                            mix_attn=config.mix_attn,
                            pad_id=config.pad_id,
                            shift=config.decoder_block != "llada",
                            arc_grid_positions=getattr(config, 'arc_grid_positions', False),
                            arc_data_processing=getattr(config, 'arc_data_processing', False),
                            arc_select_demo_pairs=getattr(config, 'arc_select_demo_pairs', True),
                            arc_loss_on_all_outputs=getattr(config, 'arc_loss_on_all_outputs', False),
                            arc_remove_output_padding=getattr(config, 'arc_remove_output_padding', False),
                            strictly_follow_nvarc_tokenizer=getattr(config, 'strictly_follow_nvarc_tokenizer', False),
                            )
  eval_dataloader = None
  if eval_pathes:
    eval_dataloader = PileDatasets(
                            mesh=mesh,
                            name=eval_name, 
                            path=eval_pathes, 
                            meta_dict={},
                            batch_size=int(config.eval_per_device_batch_size * num_local_devices),
                            seq_len=getattr(config, 'eval_max_target_length', config.max_target_length),
                            repeat=config.epoch,
                            seed=config.data_shuffle_seed,
                            task_features=task_features,
                            shuffle_buffer_size=config.eval_shuffle_buffer_size,
                            num_batches_to_skip=None,
                            only_eval=False,
                            zero_loss=config.zero_loss,
                            iter_file_nums=config.iter_file_nums,
                            mix_attn=config.mix_attn,
                            pad_id=config.pad_id,
                            shift=config.decoder_block != "llada",
                            arc_grid_positions=getattr(config, 'arc_grid_positions', False),
                            arc_data_processing=getattr(config, 'arc_data_processing', False),
                            arc_select_demo_pairs=getattr(config, 'arc_select_demo_pairs', True),
                            arc_loss_on_all_outputs=getattr(config, 'arc_loss_on_all_outputs', False),
                            arc_remove_output_padding=getattr(config, 'arc_remove_output_padding', False),
                            strictly_follow_nvarc_tokenizer=getattr(config, 'strictly_follow_nvarc_tokenizer', False),
                            )
  def train_dataloader_fn():
    return train_dataloader

  def eval_dataloader_fn():
    return eval_dataloader
  return train_dataloader_fn, eval_dataloader_fn
