import os
import tensorflow as tf
import numpy as np
from google.cloud import storage
from transformers import AutoTokenizer
import multiprocessing
from collections import defaultdict, deque
import time
import gc
from typing import Iterator, List, Dict, Any

# ==============================================================================
# 1. 基础配置与路径 (根据你的环境修改)
# ==============================================================================

# 输入路径：这是上一轮脚本生成的 obfd 文件夹
INPUT_OBFD_DIR = 'gs://newproject-1-llm_base_models_us-east5/data/v4.5-1.5B/olmo-mix-1124-r0.1/dclm/obfd/'

# 输出路径：处理完的打包数据存放处
OUTPUT_PACKED_DIR = 'gs://newproject-1-llm_base_models_us-east5/data/v4.5-1.5B/olmo-mix-1124-r0.1/dclm/obfd_packed_all/'

BUCKET_NAME = "newproject-1-data-xm4d5" # 仅用于 list_blobs，读取走 tf.data (GCS path)

# 批处理大小：每多少个文件重置一次 OBFD 算法。
# 值越大：拼凑得越满（碎片利用率高），内存占用越高。
# 值越小：内存越省，但可能产生较多填不满的尾部。
# 建议：20-50 之间。
FILES_PER_GROUP = 120 
MAX_SEQ_LENGTH = 4096

# ==============================================================================
# 2. 核心算法: SegmentTree & StreamingOBFD
# ==============================================================================

class _SegmentTree:
    """线段树，用于高效查找剩余空间"""
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
            if self.tree[i << 1] >= val:
                i = i << 1
            else:
                i = (i << 1) + 1
        return self.tree[i]

class StreamingOBFD:
    def __init__(self, max_seq_length: int = 4096, flush_threshold: int = 50):
        """
        Args:
            max_seq_length: 目标最大长度 (例如 4096)
            flush_threshold: 剩余空间小于此值时，视为已满，立即输出 (例如 50)
        """
        self.max_seq_length = max_seq_length
        self.flush_threshold = flush_threshold
        
        # 初始化线段树
        self.segment_tree = _SegmentTree(max_seq_length)
        # 初始化时，虽然没有实际的bin，但逻辑上我们总是可以创建一个全新的、满空间的bin
        # 因此我们在树中标记 max_seq_length 是可用的。
        self.segment_tree.add(max_seq_length)
        
        # 存储当前正在等待填充的 bins
        # key: 剩余空间 (int), value: bin列表 (deque)
        self.space_to_bin = defaultdict(deque)


    def add_sequence(self, input_ids: List[int]) -> Iterator[List[int]]:
        """
        处理输入序列：
        1. 循环切出完整的 max_seq_length 块，直接 yield (保存)。
        2. 将最后不足 max_seq_length 的 '尾部'，送入 OBFD 算法进行拼接。
        """
        total_len = len(input_ids)
        start_idx = 0
        
        # --- 阶段 1: 处理满块 (Full Chunks) ---
        # 只要剩余数据够切一个完整的 4096，就切出来直接保存
        # 这些块不需要进入 OBFD，因为它们已经没有空间拼其他数据了
        while total_len - start_idx >= self.max_seq_length:
            full_chunk = input_ids[start_idx : start_idx + self.max_seq_length]
            yield full_chunk
            start_idx += self.max_seq_length

        # 如果恰好整除（没有尾部），则直接结束
        if start_idx == total_len:
            return

        # --- 阶段 2: 提取尾部 (Tail) ---
        # 这里的 seq_len 一定是 < 4096 的
        current_ids = input_ids[start_idx:]
        seq_len = len(current_ids)

        # --- 阶段 3: 尾部进入 OBFD 算法 (关键修正点) ---
        # 拿着这个短尾巴，去线段树里找空位
        space = self.segment_tree.search(seq_len)
        
        target_bin = None
        
        # 情况 A: 找到了现有的、没填满的 bin
        if space < self.max_seq_length:
            target_bin = self.space_to_bin[space].popleft()
            # 如果该空间的 bin 取完了，从树中移除该标记
            if not self.space_to_bin[space]:
                self.segment_tree.remove(space)
        # 情况 B: 没找到合适的位置，或者 best fit 就是开个新桶
        else:
            target_bin = []

        # --- 拼接数据 ---
        target_bin.extend(current_ids)
        
        # 计算拼接后的剩余空间
        current_bin_len = len(target_bin)
        remaining_space = self.max_seq_length - current_bin_len
        
        # --- 阶段 4: 决定是保存还是等待 ---
        if remaining_space < self.flush_threshold:
            # 1. 如果拼完后空间所剩无几 (例如 < 50)，视为“已满”
            # 直接输出保存，并释放内存
            yield target_bin
        else:
            # 2. 如果拼完后还有大量空间 (例如还剩 2000)
            # 将其放入池子 (space_to_bin) 等待下一条数据来填充
            self.space_to_bin[remaining_space].append(target_bin)
            self.segment_tree.add(remaining_space)
            
    def flush(self) -> Iterator[List[int]]:
        """
        处理完所有数据后调用。
        将所有还在内存中等待填充的 bin 全部输出。
        """
        # 遍历所有剩余的 bin
        # space_to_bin 的 key 是剩余空间
        for space, bins in self.space_to_bin.items():
            while bins:
                yield bins.popleft()
        
        # 清理状态
        self.space_to_bin.clear()
        # 重置树（可选）

# ==============================================================================
# 3. TFRecord 读写辅助函数
# ==============================================================================

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def write_to_tfrecord(writer, input_ids):
    feature = {"input_ids": _int64_feature(input_ids)}
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())

def parse_tfrecord_fn(example_proto):
    feature_description = {
        'input_ids': tf.io.VarLenFeature(tf.int64),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    # 转换为 dense tensor 并转为 int32 (节省显存/内存)
    return tf.sparse.to_dense(example['input_ids'])

def get_dataset_from_files(file_paths):
    """构建高效的 tf.data pipeline"""
    # 自动交错读取多个文件，提高 I/O 吞吐
    dataset = tf.data.TFRecordDataset(file_paths, num_parallel_reads=tf.data.AUTOTUNE)
    dataset = dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

# ==============================================================================
# 4. Worker 进程逻辑
# ==============================================================================

def worker_process(file_list, worker_id):
    """
    Worker 处理函数：
    1. 接收一组文件列表
    2. 将其按 FILES_PER_GROUP 分组
    3. 每组运行一次 OBFD，处理完 flush，释放内存
    """
    print(f"🚀 [Worker {worker_id}] Started. Assigned {len(file_list)} files.")
    
    # 初始化 Tokenizer (仅用于获取 BOS id)
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B", use_fast=True)
    bos_id = tokenizer.bos_token_id
    
    total_saved = 0
    writer_idx = 0
    writer = None
    
    # 创建输出目录的 Writer 工厂方法 (带 worker_id 防止冲突)
    def get_next_writer():
        nonlocal writer_idx
        # 文件名: worker_0_batch_00001.tfrecord
        fname = f"Rank{worker_id:03d}.{writer_idx:04d}.tfrecord"
        fpath = os.path.join(OUTPUT_PACKED_DIR, fname)
        writer_idx += 1
        return tf.io.TFRecordWriter(fpath)

    # 将文件列表切分为小批次 (Batching)
    chunks = [file_list[i:i + FILES_PER_GROUP] for i in range(0, len(file_list), FILES_PER_GROUP)]
    
    start_time = time.time()
    
    for chunk_idx, file_chunk in enumerate(chunks):
        
        # 1. 初始化新的 OBFD 打包器 (每组文件一个新的 packer，防止内存无限增长)
        packer = StreamingOBFD(max_seq_length=MAX_SEQ_LENGTH, flush_threshold=50)
        
        # 2. 构建 Dataset 读取器
        ds = get_dataset_from_files(file_chunk)
        # 3. 处理数据
        # 使用 as_numpy_iterator 避免 TensorFlow 图模式下的内存泄漏，并将 Tensor 转为 numpy array
        for tensor_ids in ds.as_numpy_iterator():
            # tensor_ids 是 numpy array (int64)
            input_ids = tensor_ids.tolist() 
            
            # 喂给 OBFD
            adds = 0
            for packed_seq in packer.add_sequence(input_ids):
                adds += 1
                if writer is None: writer = get_next_writer()
                
                # ！！！关键点：OBFD 拼好的是一段文本，保存时前面加 BOS ！！！
                final_ids = [bos_id] + packed_seq
                
                # 安全截断 (防止加上 BOS 后溢出，虽然理论上 add_sequence 控制了长度)
                if len(final_ids) > MAX_SEQ_LENGTH:
                    final_ids = final_ids[:MAX_SEQ_LENGTH+1] # 4096 + 1 for bos id
                    
                write_to_tfrecord(writer, final_ids)
                total_saved += 1
                
                # 每 10000 条换一个输出文件，避免单文件过大
                if total_saved % 10000 == 0:
                    writer.close()
                    writer = get_next_writer()
        
            if total_saved % 10000 == 0:
                elapsed = (time.time() - start_time) / 60
                print(f"  [Worker {worker_id}] Processing group {chunk_idx+1}/{len(chunks)} ({len(file_chunk)} files) take: {elapsed:.1f}m...")

        # 4. Flush 当前组的剩余数据
        for packed_seq in packer.flush():
            if writer is None: writer = get_next_writer()
            final_ids = [bos_id] + packed_seq
            if len(final_ids) > MAX_SEQ_LENGTH: final_ids = final_ids[:MAX_SEQ_LENGTH+1]
            write_to_tfrecord(writer, final_ids)
            total_saved += 1
        
        # 5. 清理内存
        del packer
        del ds
        gc.collect() # 强制垃圾回收
        
        elapsed = (time.time() - start_time) / 60
        print(f"  [Worker {worker_id}] Finished group {chunk_idx+1}/{len(chunks)}. Total Saved: {total_saved}. Time: {elapsed:.1f}m")

    if writer:
        writer.close()
    
    print(f"✅ [Worker {worker_id}] Done.")


# ==============================================================================
# 5. 主程序
# ==============================================================================

def main():
    # 1. 获取输入文件列表
    # 注意：这里我们列出 INPUT_OBFD_DIR 下所有的 tfrecord
    print(f"🔍 Listing files from {INPUT_OBFD_DIR} ...")
    
    # 解析 Bucket 和 Prefix
    if not INPUT_OBFD_DIR.startswith("gs://"):
        raise ValueError("INPUT_OBFD_DIR must start with gs://")
    
    path_parts = INPUT_OBFD_DIR.replace("gs://", "").split("/")
    bucket_name = path_parts[0]
    prefix = "/".join(path_parts[1:])
    
    client = storage.Client()
    blobs = client.list_blobs(bucket_name, prefix=prefix)
    
    # 构造完整的 gs:// 路径列表
    all_files = []
    for blob in blobs:
        if blob.name.endswith(".tfrecord"):
            all_files.append(f"gs://{bucket_name}/{blob.name}")
            
    # start = 0
    # end = 3000
    # all_files = all_files[start:end] # 一共6000
    print(f"📦 Found {len(all_files)} files to process.")
    
    if len(all_files) == 0:
        print("No files found. Exiting.")
        return

    # 2. 规划多进程
    # 使用所有可用核心，或根据内存情况适当减少
    num_processes = multiprocessing.cpu_count()
    # 如果内存紧张，可以手动设置为 cpu_count() // 2
    num_processes = 50
    
    print(f"⚙️ Using {num_processes} processes.")
    
    # 3. 任务分发 (Sharding)
    # 将文件列表均匀切分
    chunk_size = len(all_files) // num_processes + 1
    file_shards = [all_files[i:i + chunk_size] for i in range(0, len(all_files), chunk_size)]
    
    # 过滤空分片
    file_shards = [s for s in file_shards if len(s) > 0]
    
    process_args = []
    for i, shard in enumerate(file_shards):
        process_args.append((shard, i))
        
    # 4. 启动进程池
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.starmap(worker_process, process_args)
        
    print("🎉 All processing finished.")

if __name__ == "__main__":
    main()


# 记录：50进程，每个进程处理120个90M的文件，需要内存：1T，v6e-8 cpu 需要25min