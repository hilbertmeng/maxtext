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
                mix_attn: bool = False
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
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.cast(t, dtype=tf.int32)
            example[name] = tf.sparse.to_dense(t, default_value=0)[:self.seq_len + 1]
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
    
    def convert(self, data):
        seq_len = self.seq_len
        model_needed_inputs = {}
        model_needed_inputs['inputs'] = data["input_ids"][:, : seq_len]
        model_needed_inputs['targets'] = data["input_ids"][:, 1: seq_len + 1]
        key = 'labels' if "labels" in data else 'input_ids'
        # weights = data[key] >= 0 if self.zero_loss else data[key] > 0
        weights = data[key] != self.pad_id
        # label loss mask, origin bool type, but due the complie is int32
        model_needed_inputs['targets_segmentation'] = tf.cast(weights[:, 1: seq_len + 1], dtype=tf.int32) 
        model_needed_inputs['inputs_segmentation'] = self.build_attn_mask()
        pos = tf.range(seq_len)
        model_needed_inputs['inputs_position'] = model_needed_inputs['inputs_segmentation'] * pos # rotary position, mtp use shift position
        model_needed_inputs['targets_position'] = model_needed_inputs['inputs_segmentation'] * pos  # no use, but complie have this key
        return model_needed_inputs

    def _load_file_dataset(self, fname):
        tf.random.set_seed(self.seed)
        ds = tf.data.Dataset.from_tensor_slices(fname)
        ds = ds.apply(tf.data.TFRecordDataset)
        ds = ds.map(self._parse_function, num_parallel_calls=tf.data.AUTOTUNE) # 取 seq_len + 1
        print(f'shuffle_buffer_size: {self.shuffle_buffer_size}')
        if self.shuffle_buffer_size is not None:
            ds = ds.shuffle(buffer_size=self.shuffle_buffer_size)

        padded_shapes = {key: self.seq_len + 1 for key in self.task_features}
        padding_values = {key: self.pad_id if key == 'input_ids' else -100 for key in self.task_features}
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
        repeat_fnames = fnames * self.repeat
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
    train_pathes, eval_pathes = extract_pythia_datapath(config.dataset_path, config.eval_split)
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
                            )
  eval_dataloader = None
  if eval_pathes:
    eval_dataloader = PileDatasets(
                            mesh=mesh,
                            name=eval_name, 
                            path=eval_pathes, 
                            meta_dict={},
                            batch_size=int(config.eval_per_device_batch_size * num_local_devices),
                            seq_len=config.max_target_length,
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
                            )
  def train_dataloader_fn():
    return train_dataloader

  def eval_dataloader_fn():
    return eval_dataloader
  return train_dataloader_fn, eval_dataloader_fn