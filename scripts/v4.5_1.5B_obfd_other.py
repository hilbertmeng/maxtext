import os
import tensorflow as tf
import numpy as np
from google.cloud import storage
from transformers import AutoTokenizer
import multiprocessing
from collections import defaultdict, deque
import time
import gc

# =================配置区域=================
BUCKET_NAME = "newproject-1-data-xm4d5"
PROJECT_ROOT = "gs://newproject-1-llm_base_models_us-east5/data/v4.5-1.5B/olmo-mix-1124-r0.1"

# 待处理的数据集 (可以一次跑一个，也可以跑列表)
DATASETS_TO_PROCESS = ['algebraic-stack', 'arxiv', 'open-web-math', 'pes2o', 'starcoder', 'wiki']

MAX_SEQ_LENGTH = 4096
FILES_PER_GROUP = 120  
# 根据内存调整，越大碎片越少但内存占用越高 
# pes2o: 4个进程，100个文件/进程,
#  wiki: 1个进程，130个文件/进程, 
# pes2o: 3个进程，300个文件/进程, 
# starcoder: 13个进程，130个文件/进程
# other: 1个进程，120个文件/进程, 
# =========================================

# --- 线段树与OBFD算法 (复用即可) ---
class _SegmentTree:
    def __init__(self, maxval: int):
        self.maxval = maxval
        self.tree_size = 1 << (maxval - 1).bit_length()
        self.tree = [0] * (2 * self.tree_size)

    def add(self, val):
        if val <= 0 or val > self.maxval: return
        i = self.tree_size + val - 1
        self.tree[i] = val
        while i > 1:
            i >>= 1
            left, right = self.tree[i << 1], self.tree[(i << 1) + 1]
            self.tree[i] = left if left >= right else right

    def remove(self, val):
        if val <= 0 or val > self.maxval: return
        i = self.tree_size + val - 1
        self.tree[i] = 0
        while i > 1:
            i >>= 1
            left, right = self.tree[i << 1], self.tree[(i << 1) + 1]
            self.tree[i] = left if left >= right else right

    def search(self, val):
        if val > self.maxval: return 0
        if self.tree[1] < val: return 0
        i = 1
        while i < self.tree_size:
            if self.tree[i << 1] >= val: i = i << 1
            else: i = (i << 1) + 1
        return self.tree[i]

class StreamingOBFD:
    def __init__(self, max_seq_length=4096, flush_threshold=50):
        self.max_seq_length = max_seq_length
        self.flush_threshold = flush_threshold
        self.segment_tree = _SegmentTree(max_seq_length)
        self.segment_tree.add(max_seq_length)
        self.space_to_bin = defaultdict(deque)

    def add_sequence(self, input_ids):
        total_len = len(input_ids)
        start_idx = 0
        # 1. 切分满块
        while total_len - start_idx >= self.max_seq_length:
            yield input_ids[start_idx : start_idx + self.max_seq_length]
            start_idx += self.max_seq_length
        if start_idx == total_len: return

        # 2. 处理尾部
        current_ids = input_ids[start_idx:]
        seq_len = len(current_ids)
        space = self.segment_tree.search(seq_len)
        
        if space < self.max_seq_length:
            target_bin = self.space_to_bin[space].popleft()
            if not self.space_to_bin[space]: self.segment_tree.remove(space)
        else:
            target_bin = []

        target_bin.extend(current_ids)
        remaining_space = self.max_seq_length - len(target_bin)
        
        if remaining_space < self.flush_threshold:
            yield target_bin
        else:
            self.space_to_bin[remaining_space].append(target_bin)
            self.segment_tree.add(remaining_space)
            
    def flush(self):
        for _, bins in self.space_to_bin.items():
            while bins: yield bins.popleft()
        self.space_to_bin.clear()

# --- TFRecord IO ---
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def write_to_tfrecord(writer, input_ids):
    feature = {"input_ids": _int64_feature(input_ids)}
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())

def parse_tfrecord_fn(example_proto):
    feature_description = {'input_ids': tf.io.VarLenFeature(tf.int64)}
    example = tf.io.parse_single_example(example_proto, feature_description)
    return tf.sparse.to_dense(example['input_ids'])

# --- Worker ---
def worker_process(output_dir, file_list, worker_id):
    # 避免 OOM，不一次加载所有文件，而是分组处理
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B", use_fast=True)
    bos_id = tokenizer.bos_token_id
    
    writer_idx = 0
    total_saved = 0
    writer = None
    
    def get_writer():
        nonlocal writer_idx
        fname = f"Rank{worker_id:03d}.{writer_idx:04d}.tfrecord"
        fpath = os.path.join(output_dir, fname)
        writer_idx += 1
        return tf.io.TFRecordWriter(fpath)

    # 分组读取，控制内存
    chunks = [file_list[i:i + FILES_PER_GROUP] for i in range(0, len(file_list), FILES_PER_GROUP)]
    
    for chunk_idx, chunk in enumerate(chunks):
        packer = StreamingOBFD(max_seq_length=MAX_SEQ_LENGTH, flush_threshold=50)
        
        ds = tf.data.TFRecordDataset(chunk, num_parallel_reads=tf.data.AUTOTUNE)
        ds = ds.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        start_time = time.time()
        for tensor_ids in ds.as_numpy_iterator():
            input_ids = tensor_ids.tolist()
            
            for packed_seq in packer.add_sequence(input_ids):
                if writer is None: writer = get_writer()
                
                # 重要：OBFD 拼好后，前面加 BOS
                final_ids = [bos_id] + packed_seq
                if len(final_ids) > MAX_SEQ_LENGTH: 
                    final_ids = final_ids[:MAX_SEQ_LENGTH+1] # 允许4097(包含bos)或者硬截断，通常模型需要一致长度，这里截断到4096+1可能需要看你模型config，通常是4096。

                write_to_tfrecord(writer, final_ids)
                total_saved += 1
                if total_saved % 10000 == 0:
                    writer.close()
                    writer = get_writer()
                if total_saved % 1000 == 0:
                    elapsed = (time.time() - start_time) / 60
                    print(f"  [Worker {worker_id}] Processing group {chunk_idx+1}/{len(chunks)} ({len(chunk)} files) take: {elapsed:.1f}m...")
        
        # Flush
        for packed_seq in packer.flush():
            if writer is None: writer = get_writer()
            final_ids = [bos_id] + packed_seq
            if len(final_ids) > MAX_SEQ_LENGTH: 
                final_ids = final_ids[:MAX_SEQ_LENGTH+1]
            write_to_tfrecord(writer, final_ids)
            total_saved += 1
            if total_saved % 10000 == 0:
                writer.close()
                writer = get_writer()
            
        del packer
        del ds
        gc.collect()

    if writer: writer.close()
    print(f"[Worker {worker_id}] Done. Saved {total_saved} sequences.")

def main():
    num_processes = 1 # 或 multiprocessing.cpu_count()
    s, e = 1, 2

    for ds_name in DATASETS_TO_PROCESS[s:e]:
        print(f"\n{'='*20} Packing Dataset: {ds_name} {'='*20}")
        
        # 输入：来自第一步的 obfd 文件夹
        input_obfd_path = f"{PROJECT_ROOT}/{ds_name}/obfd/"
        # 输出：打包好的文件夹
        output_packed_path = f"{PROJECT_ROOT}/{ds_name}/obfd_packed/"
        
        print(f"Reading from: {input_obfd_path}")
        print(f"Writing to: {output_packed_path}")
        
        # 获取输入文件列表
        path_parts = input_obfd_path.replace("gs://", "").split("/")
        bucket_name = path_parts[0]
        prefix = "/".join(path_parts[1:])
        
        client = storage.Client()
        blobs = client.list_blobs(bucket_name, prefix=prefix)
        all_files = [f"gs://{bucket_name}/{b.name}" for b in blobs if b.name.endswith(".tfrecord")]
        
        print(f"Found {len(all_files)} obfd shards.")
        if not all_files: continue
        
        chunk_size = len(all_files) // num_processes + 1
        file_shards = [all_files[i:i + chunk_size] for i in range(0, len(all_files), chunk_size)]
        
        process_args = []
        for i, shard in enumerate(file_shards):
            if shard:
                process_args.append((output_packed_path, shard, i))
        
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.starmap(worker_process, process_args)

    print("\nAll Datasets Packed (Step 2).")

if __name__ == "__main__":
    main()