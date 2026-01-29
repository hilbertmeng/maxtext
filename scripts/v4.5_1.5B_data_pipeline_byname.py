"""
统一数据处理流水线：Tokenize -> OBFD Packing -> Shuffle

流程说明：
  Step 1 (Tokenize): 将原始文本 tokenize，长文本切分为 4k 块，短文本存入 obfd
  Step 2 (OBFD):     将所有数据集的短序列混合打包成 4k 块
  Step 3 (Shuffle):  将 4k 数据和 packed 数据分别 shuffle

用法:
    # 1. Tokenize 单个数据集
    python v4.5_1.5B_data_pipeline.py --step tokenize --dataset dclm
    
    # 2. Tokenize 多个数据集 (混合处理，充分利用多进程)
    python v4.5_1.5B_data_pipeline.py --step tokenize --dataset dclm,wiki,math
    
    # 3. Tokenize 所有数据集
    python v4.5_1.5B_data_pipeline.py --step tokenize --dataset all
    
    # 4. OBFD Packing (混合所有数据集的 obfd 数据)
    python v4.5_1.5B_data_pipeline.py --step obfd
    
    # 5. Shuffle (混合 shuffle 所有数据)
    python v4.5_1.5B_data_pipeline.py --step shuffle
    
    # 6. 完整流水线 (tokenize -> obfd -> shuffle)
    python v4.5_1.5B_data_pipeline.py --step all --dataset all
    
    # 7. 控制 OBFD 每次处理的文件数 (内存不足时减小此值)
    python v4.5_1.5B_data_pipeline.py --step obfd --files-per-group 50
    
    # 8. 控制 Shuffle 缓冲区大小 (内存不足时减小此值)
    python v4.5_1.5B_data_pipeline.py --step shuffle --shuffle-buffer-size 50000
"""

import os
import io
import time
import random
import argparse
import orjson
import zstandard as zstd
import smart_open
import tensorflow as tf
import numpy as np
from google.cloud import storage
from transformers import AutoTokenizer
import multiprocessing
from collections import defaultdict, deque
import gc
from typing import Iterator, List, Dict, Any, Optional, Tuple

# ==============================================================================
# 配置区域
# ==============================================================================
DATASET_NAME = "olmo-mix-1124/"
DATASET_NAME = "dolmino-mix-1124"

class Config:
    """全局配置"""
    # GCS 配置
    BUCKET_NAME = "newproject-1-data-xm4d5"
    PROJECT_ROOT = f"gs://newproject-1-llm_base_models_us-east5/data/v4.5-1.5B/{DATASET_NAME}-r0.1"
    
    # 模型配置
    TOKENIZER_NAME = "allenai/OLMo-2-0425-1B"
    MAX_SEQ_LENGTH = 4096
    LENGTH_THRESHOLD = 4050  # 长度阈值，大于此值才切分
    
    # 数据集配置
    # 文件后缀说明:
    #   dclm: .json.zst
    #   flan: .json.gz
    #   math: .jsonl, .jsonl.gz, .jsonl.zst (混合)
    #   pes2o: .json.gz
    #   stackexchange: .json.gz
    #   wiki: .json.gz
    ALL_DATASETS = ['dclm', 'flan', 'math', 'pes2o', 'stackexchange', 'wiki']
    
    # 数据集的输入前缀映射
    INPUT_PREFIX_MAP = {
        'dclm': f"datasets/{DATASET_NAME}/data/dclm",
        'flan': f"datasets/{DATASET_NAME}/data/flan",
        'math': f"datasets/{DATASET_NAME}/data/math",
        'pes2o': f"datasets/{DATASET_NAME}/data/pes2o",
        'stackexchange': f"datasets/{DATASET_NAME}/data/stackexchange",
        'wiki': f"datasets/{DATASET_NAME}/data/wiki"
    }
    
    # 支持的文件后缀 (用于过滤文件)
    SUPPORTED_SUFFIXES = ('.json.zst', '.jsonl.zst', '.json.gz', '.jsonl.gz', '.jsonl', '.json')
    
    # Tokenize 步骤配置
    DATA_4K_NUM_PER_FILE = 10000
    DATA_OBFD_NUM_PER_FILE = 50000
    
    # OBFD Packing 步骤配置
    # FILES_PER_GROUP: 每次 OBFD 处理的文件数量
    # - 值越大：拼接效果越好（碎片利用率高），但内存占用越高
    # - 值越小：内存占用越低，但可能产生较多填不满的尾部
    # - 建议范围：20-120，根据内存情况调整
    FILES_PER_GROUP = 120
    OBFD_FLUSH_THRESHOLD = 50
    PACKED_SAMPLES_PER_FILE = 10000
    
    # 混合 OBFD 输出目录 (所有数据集的 obfd 混合打包后输出到这里)
    MIXED_OBFD_PACKED_DIR = f"gs://newproject-1-llm_base_models_us-east5/data/v4.5-1.5B/{DATASET_NAME}-r0.1/mixed_obfd_packed/"
    
    # Shuffle 步骤配置
    # SHUFFLE_BUFFER_SIZE: Shuffle 缓冲区大小
    # - 值越大：Shuffle 效果越好（更随机），但内存占用越高
    # - 值越小：内存占用越低，但 Shuffle 效果有限
    # - 建议范围：10000-500000，根据内存情况调整
    SHUFFLE_BUFFER_SIZE = 500000
    SHUFFLE_OUTPUT_SAMPLES_PER_FILE = 10000  # 每个输出文件的样本数
    SHUFFLE_SEED = 1234  # 随机种子，保证可复现性
    
    # 混合 Shuffle 输出目录 (4k + packed 混合后的最终输出)
    SHUFFLED_OUTPUT_DIR = f"gs://newproject-1-llm_base_models_us-east5/data/v4.5-1.5B/{DATASET_NAME}-r0.1/shuffled/"
    
    # 进程配置，第一阶段50，第二阶段10
    NUM_PROCESSES = 10


# ==============================================================================
# 工具函数
# ==============================================================================

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

def list_gcs_files(bucket_name: str, prefix: str, suffix: str = "") -> List[str]:
    """列出 GCS 上的文件"""
    client = storage.Client()
    blobs = client.list_blobs(bucket_name, prefix=prefix)
    files = []
    for blob in blobs:
        if suffix and not blob.name.endswith(suffix):
            continue
        files.append(f"gs://{bucket_name}/{blob.name}")
    return files

def parse_gcs_path(gcs_path: str) -> Tuple[str, str]:
    """解析 gs:// 路径为 bucket 和 prefix"""
    if not gcs_path.startswith("gs://"):
        raise ValueError(f"Path must start with gs://: {gcs_path}")
    parts = gcs_path.replace("gs://", "").split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket, prefix

def get_output_paths(dataset_name: str) -> Dict[str, str]:
    """获取数据集的输出路径"""
    base = Config.PROJECT_ROOT
    return {
        '4k': f"{base}/{dataset_name}/4k/",
        'obfd': f"{base}/{dataset_name}/obfd/",
        'obfd_packed': f"{base}/{dataset_name}/obfd_packed/",
        'shuffled': f"{base}/{dataset_name}/shuffled/",
    }


def read_jsonl_file(file_path: str, bucket_cache: Dict = None) -> Iterator[Dict]:
    """
    根据文件后缀自动选择读取方式，返回解析后的 JSON 对象迭代器
    
    支持的格式:
    - .json.zst / .jsonl.zst: zstandard 压缩
    - .json.gz / .jsonl.gz: gzip 压缩 (smart_open 自动处理)
    - .json / .jsonl: 无压缩
    
    Args:
        file_path: GCS 文件路径 (gs://bucket/path)
        bucket_cache: 可选的 bucket 对象缓存，避免重复创建
    
    Yields:
        解析后的 JSON 对象
    """
    file_lower = file_path.lower()
    
    if file_lower.endswith('.zst'):
        # zstandard 压缩：需要特殊处理
        bucket_name, relative_path = parse_gcs_path(file_path)
        
        if bucket_cache is not None and bucket_name in bucket_cache:
            bucket = bucket_cache[bucket_name]
        else:
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            if bucket_cache is not None:
                bucket_cache[bucket_name] = bucket
        
        blob = bucket.blob(relative_path)
        dctx = zstd.ZstdDecompressor()
        
        with blob.open("rb") as compressed_stream:
            with dctx.stream_reader(compressed_stream) as reader:
                text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                for line in text_stream:
                    line = line.strip()
                    if line:
                        try:
                            yield orjson.loads(line)
                        except Exception:
                            continue
    else:
        # .gz 或无压缩：smart_open 可以自动处理
        with smart_open.open(file_path, "rb") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield orjson.loads(line)
                    except Exception:
                        continue


def is_supported_file(file_name: str) -> bool:
    """检查文件是否是支持的格式"""
    file_lower = file_name.lower()
    return any(file_lower.endswith(suffix) for suffix in Config.SUPPORTED_SUFFIXES)


# ==============================================================================
# Step 1: Tokenize
# ==============================================================================

class TokenizeWorker:
    """Tokenize 工作器"""
    
    def __init__(self, dataset_name: str, output_4k: str, output_obfd: str, worker_id: int):
        self.dataset_name = dataset_name
        self.output_4k = output_4k
        self.output_obfd = output_obfd
        self.worker_id = worker_id
        
        # 初始化 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_NAME, use_fast=True)
        self.bos_id = self.tokenizer.bos_token_id
        self.eos_id = self.tokenizer.eos_token_id
        
        # 计数器和 writer
        self.count_4k = 0
        self.count_obfd = 0
        self.writer_4k = None
        self.writer_obfd = None
    
    def _get_writer(self, output_dir: str, count: int, samples_per_file: int) -> tf.io.TFRecordWriter:
        """获取或创建 writer"""
        file_idx = count // samples_per_file
        fname = f'Rank{self.worker_id:03}.{file_idx:04}.tfrecord'
        fpath = os.path.join(output_dir, fname)
        return tf.io.TFRecordWriter(fpath)
    
    def process_text(self, text: str):
        """处理单条文本"""
        if not text:
            return
        
        input_ids = self.tokenizer.encode(text)
        
        # 处理长文本 -> 4k
        while len(input_ids) >= Config.LENGTH_THRESHOLD:
            # [BOS] + text[:4096]
            save_ids = [self.bos_id] + input_ids[:Config.MAX_SEQ_LENGTH]
            
            if self.count_4k % Config.DATA_4K_NUM_PER_FILE == 0:
                if self.writer_4k:
                    self.writer_4k.close()
                self.writer_4k = self._get_writer(
                    self.output_4k, self.count_4k, Config.DATA_4K_NUM_PER_FILE
                )
            
            write_to_tfrecord(self.writer_4k, save_ids)
            input_ids = input_ids[Config.MAX_SEQ_LENGTH:]
            self.count_4k += 1
        
        # 处理剩余/短文本 -> obfd
        if input_ids:
            input_ids = input_ids + [self.eos_id]  # 添加 EOS
            
            if self.count_obfd % Config.DATA_OBFD_NUM_PER_FILE == 0:
                if self.writer_obfd:
                    self.writer_obfd.close()
                self.writer_obfd = self._get_writer(
                    self.output_obfd, self.count_obfd, Config.DATA_OBFD_NUM_PER_FILE
                )
            
            write_to_tfrecord(self.writer_obfd, input_ids)
            self.count_obfd += 1
    
    def close(self):
        """关闭所有 writer"""
        if self.writer_4k:
            self.writer_4k.close()
        if self.writer_obfd:
            self.writer_obfd.close()


class MultiDatasetTokenizeWorker:
    """
    多数据集 Tokenize 工作器
    可以处理来自不同数据集的文件，每个数据集有独立的输出 writer
    """
    
    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        
        # 初始化 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_NAME, use_fast=True)
        self.bos_id = self.tokenizer.bos_token_id
        self.eos_id = self.tokenizer.eos_token_id
        
        # 为每个数据集维护独立的计数器和 writer
        # {dataset_name: {'count_4k': int, 'count_obfd': int, 'writer_4k': writer, 'writer_obfd': writer}}
        self.dataset_state = {}
    
    def _get_or_create_state(self, dataset_name: str) -> Dict:
        """获取或创建数据集的状态"""
        if dataset_name not in self.dataset_state:
            paths = get_output_paths(dataset_name)
            self.dataset_state[dataset_name] = {
                'output_4k': paths['4k'],
                'output_obfd': paths['obfd'],
                'count_4k': 0,
                'count_obfd': 0,
                'writer_4k': None,
                'writer_obfd': None,
            }
        return self.dataset_state[dataset_name]
    
    def _get_writer(self, output_dir: str, count: int, samples_per_file: int) -> tf.io.TFRecordWriter:
        """创建新的 writer"""
        file_idx = count // samples_per_file
        fname = f'Rank{self.worker_id:03}.{file_idx:04}.tfrecord'
        fpath = os.path.join(output_dir, fname)
        return tf.io.TFRecordWriter(fpath)
    
    def process_text(self, text: str, dataset_name: str):
        """处理单条文本，输出到对应数据集的目录"""
        if not text:
            return
        
        state = self._get_or_create_state(dataset_name)
        input_ids = self.tokenizer.encode(text)
        
        # 处理长文本 -> 4k
        while len(input_ids) >= Config.LENGTH_THRESHOLD:
            save_ids = [self.bos_id] + input_ids[:Config.MAX_SEQ_LENGTH]
            
            if state['count_4k'] % Config.DATA_4K_NUM_PER_FILE == 0:
                if state['writer_4k']:
                    state['writer_4k'].close()
                state['writer_4k'] = self._get_writer(
                    state['output_4k'], state['count_4k'], Config.DATA_4K_NUM_PER_FILE
                )
            
            write_to_tfrecord(state['writer_4k'], save_ids)
            input_ids = input_ids[Config.MAX_SEQ_LENGTH:]
            state['count_4k'] += 1
        
        # 处理剩余/短文本 -> obfd
        if input_ids:
            input_ids = input_ids + [self.eos_id]
            
            if state['count_obfd'] % Config.DATA_OBFD_NUM_PER_FILE == 0:
                if state['writer_obfd']:
                    state['writer_obfd'].close()
                state['writer_obfd'] = self._get_writer(
                    state['output_obfd'], state['count_obfd'], Config.DATA_OBFD_NUM_PER_FILE
                )
            
            write_to_tfrecord(state['writer_obfd'], input_ids)
            state['count_obfd'] += 1
    
    def close(self):
        """关闭所有 writer"""
        for ds_name, state in self.dataset_state.items():
            if state['writer_4k']:
                state['writer_4k'].close()
            if state['writer_obfd']:
                state['writer_obfd'].close()
    
    def get_stats(self) -> Dict[str, Tuple[int, int]]:
        """获取各数据集的统计信息"""
        return {ds: (state['count_4k'], state['count_obfd']) 
                for ds, state in self.dataset_state.items()}


def tokenize_queue_worker(task_queue, worker_id: int, global_counter, counter_lock, total_files: int):
    """
    使用任务队列的 tokenize worker
    从队列中动态获取任务，实现负载均衡
    
    Args:
        task_queue: 共享的任务队列
        worker_id: Worker ID
        global_counter: 共享的全局计数器
        counter_lock: 计数器的锁
        total_files: 总文件数
    """
    print(f"[Worker {worker_id}] Started.")
    
    worker = MultiDatasetTokenizeWorker(worker_id)
    
    # 缓存 bucket 对象，避免重复创建
    bucket_cache = {}
    
    start_time = time.time()
    processed_files = 0
    total_lines = 0
    
    try:
        while True:
            try:
                # 从队列获取任务，超时 5 秒
                task = task_queue.get(timeout=5)
                
                if task is None:  # None 是结束信号
                    break
                    
                file_path, dataset_name = task
                
                # 根据文件后缀自动选择读取方式
                for data in read_jsonl_file(file_path, bucket_cache):
                    text = data.get('text', '')
                    worker.process_text(text, dataset_name)
                    total_lines += 1
                
                processed_files += 1
                
                # 更新全局计数器
                with counter_lock:
                    global_counter.value += 1
                    global_done = global_counter.value
                
                if processed_files % 1 == 0:
                    elapsed = (time.time() - start_time) / 60
                    stats = worker.get_stats()
                    total_4k = sum(s[0] for s in stats.values())
                    total_obfd = sum(s[1] for s in stats.values())
                    print(f"[Worker {worker_id}] Files: {processed_files} | "
                          f"Global: {global_done}/{total_files} ({100*global_done/total_files:.1f}%) | "
                          f"Lines: {total_lines} | 4k: {total_4k} | obfd: {total_obfd} | Time: {elapsed:.1f}m")
                    
            except Exception as e:
                # 队列为空时会抛出 Empty 异常，这是正常的
                if 'Empty' in str(type(e).__name__):
                    # 队列为空，检查是否所有任务都完成了
                    continue
                print(f"[Worker {worker_id}] Error: {e}")
                continue
    
    finally:
        worker.close()
        stats = worker.get_stats()
        print(f"[Worker {worker_id}] Finished. Stats per dataset:")
        for ds, (cnt_4k, cnt_obfd) in stats.items():
            print(f"  {ds}: 4k={cnt_4k}, obfd={cnt_obfd}")


def run_tokenize_mixed(datasets: List[str], num_processes: int):
    """
    运行 Tokenize 步骤 - 混合处理多个数据集
    使用任务队列实现动态负载均衡，避免大文件导致的等待问题
    
    改进说明:
    - 使用 multiprocessing.Manager().Queue() 作为共享任务队列
    - 每个 worker 从队列中动态获取任务
    - 处理完一个文件后立即获取下一个，实现负载均衡
    - 同一数据集的多个文件可被多个进程并行处理
    """
    print(f"\n{'='*20} Step 1: Tokenize (Mixed Datasets with Queue) {'='*20}")
    print(f"Datasets to process: {datasets}")
    print(f"Supported file formats: {Config.SUPPORTED_SUFFIXES}")
    
    client = storage.Client()
    bucket = client.bucket(Config.BUCKET_NAME)
    
    # 收集所有文件任务: (file_path, dataset_name)
    all_tasks = []
    
    for ds_name in datasets:
        prefix = Config.INPUT_PREFIX_MAP.get(ds_name)
        if not prefix:
            print(f"  {ds_name}: Unknown dataset, skipping")
            continue
        
        # 列出所有文件，根据后缀过滤
        count = 0
        for blob in client.list_blobs(bucket, prefix=prefix):
            if is_supported_file(blob.name):
                file_path = f"gs://{Config.BUCKET_NAME}/{blob.name}"
                all_tasks.append((file_path, ds_name))
                count += 1
            else:
                print(f"[ERROR] {ds_name}: {blob.name} is not supported, skipping")
        
        print(f"  {ds_name}: {count} files")
    
    print(f"\nTotal files to process: {len(all_tasks)}")
    
    if not all_tasks:
        print("No files found. Exiting.")
        return
    
    # 打乱任务顺序，使不同数据集的文件混合分布
    random.seed(42)
    random.shuffle(all_tasks)
    print("Tasks shuffled for better distribution.")
    
    # 确定实际进程数
    actual_processes = min(num_processes, len(all_tasks))
    total_files = len(all_tasks)
    print(f"Starting {actual_processes} workers (requested: {num_processes})...")
    
    # 创建共享任务队列和计数器
    manager = multiprocessing.Manager()
    task_queue = manager.Queue()
    
    # 创建共享计数器用于跟踪全局进度 (使用 Manager 的 Value，可以被 pickle)
    global_counter = manager.Value('i', 0)
    counter_lock = manager.Lock()
    
    # 将所有任务放入队列
    for task in all_tasks:
        task_queue.put(task)
    
    # 放入结束信号 (每个 worker 一个)
    for _ in range(actual_processes):
        task_queue.put(None)
    
    print(f"Task queue created with {total_files} tasks.")
    
    # 启动 workers
    process_args = [(task_queue, i, global_counter, counter_lock, total_files) for i in range(actual_processes)]
    
    with multiprocessing.Pool(processes=actual_processes) as pool:
        pool.starmap(tokenize_queue_worker, process_args)
    
    print("Tokenize (Mixed with Queue) completed.")


# ==============================================================================
# Step 2: OBFD Packing
# ==============================================================================

class _SegmentTree:
    """线段树，用于高效查找剩余空间"""
    def __init__(self, maxval: int):
        self.maxval = maxval
        self.tree_size = 1 << (maxval - 1).bit_length()
        self.tree = [0] * (2 * self.tree_size)

    def add(self, val):
        if val <= 0 or val > self.maxval:
            return
        i = self.tree_size + val - 1
        self.tree[i] = val
        while i > 1:
            i >>= 1
            left, right = self.tree[i << 1], self.tree[(i << 1) + 1]
            self.tree[i] = left if left >= right else right

    def remove(self, val):
        if val <= 0 or val > self.maxval:
            return
        i = self.tree_size + val - 1
        self.tree[i] = 0
        while i > 1:
            i >>= 1
            left, right = self.tree[i << 1], self.tree[(i << 1) + 1]
            self.tree[i] = left if left >= right else right

    def search(self, val):
        if val > self.maxval:
            return 0
        if self.tree[1] < val:
            return 0
        i = 1
        while i < self.tree_size:
            if self.tree[i << 1] >= val:
                i = i << 1
            else:
                i = (i << 1) + 1
        return self.tree[i]


class StreamingOBFD:
    """流式 OBFD 打包器"""
    
    def __init__(self, max_seq_length: int = 4096, flush_threshold: int = 50):
        self.max_seq_length = max_seq_length
        self.flush_threshold = flush_threshold
        self.segment_tree = _SegmentTree(max_seq_length)
        self.segment_tree.add(max_seq_length)
        self.space_to_bin = defaultdict(deque)

    def add_sequence(self, input_ids: List[int]) -> Iterator[List[int]]:
        """处理输入序列"""
        total_len = len(input_ids)
        start_idx = 0
        
        # 阶段 1: 处理满块
        while total_len - start_idx >= self.max_seq_length:
            yield input_ids[start_idx:start_idx + self.max_seq_length]
            start_idx += self.max_seq_length

        if start_idx == total_len:
            return

        # 阶段 2: 提取尾部
        current_ids = input_ids[start_idx:]
        seq_len = len(current_ids)

        # 阶段 3: OBFD 算法
        space = self.segment_tree.search(seq_len)
        
        if space < self.max_seq_length:
            target_bin = self.space_to_bin[space].popleft()
            if not self.space_to_bin[space]:
                self.segment_tree.remove(space)
        else:
            target_bin = []

        target_bin.extend(current_ids)
        remaining_space = self.max_seq_length - len(target_bin)
        
        if remaining_space < self.flush_threshold:
            yield target_bin
        else:
            self.space_to_bin[remaining_space].append(target_bin)
            self.segment_tree.add(remaining_space)
            
    def flush(self) -> Iterator[List[int]]:
        """输出所有剩余的 bin"""
        for _, bins in self.space_to_bin.items():
            while bins:
                yield bins.popleft()
        self.space_to_bin.clear()


def obfd_worker(output_dir: str, file_list: List[str], worker_id: int):
    """OBFD 打包 worker"""
    print(f"[Worker {worker_id}] Started OBFD packing. Files: {len(file_list)}")
    
    tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_NAME, use_fast=True)
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
    
    # 分组处理
    chunks = [file_list[i:i + Config.FILES_PER_GROUP] 
              for i in range(0, len(file_list), Config.FILES_PER_GROUP)]
    
    start_time = time.time()
    
    for chunk_idx, chunk in enumerate(chunks):
        packer = StreamingOBFD(
            max_seq_length=Config.MAX_SEQ_LENGTH, 
            flush_threshold=Config.OBFD_FLUSH_THRESHOLD
        )
        
        # 读取数据
        ds = tf.data.TFRecordDataset(chunk, num_parallel_reads=tf.data.AUTOTUNE)
        ds = ds.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        
        for tensor_ids in ds.as_numpy_iterator():
            input_ids = tensor_ids.tolist()
            
            for packed_seq in packer.add_sequence(input_ids):
                if writer is None:
                    writer = get_writer()
                
                # 添加 BOS
                final_ids = [bos_id] + packed_seq
                if len(final_ids) > Config.MAX_SEQ_LENGTH + 1:
                    final_ids = final_ids[:Config.MAX_SEQ_LENGTH + 1]
                
                write_to_tfrecord(writer, final_ids)
                total_saved += 1
                
                if total_saved % Config.PACKED_SAMPLES_PER_FILE == 0:
                    writer.close()
                    writer = get_writer()
        
        # Flush
        for packed_seq in packer.flush():
            if writer is None:
                writer = get_writer()
            final_ids = [bos_id] + packed_seq
            if len(final_ids) > Config.MAX_SEQ_LENGTH + 1:
                final_ids = final_ids[:Config.MAX_SEQ_LENGTH + 1]
            write_to_tfrecord(writer, final_ids)
            total_saved += 1
            
            if total_saved % Config.PACKED_SAMPLES_PER_FILE == 0:
                writer.close()
                writer = get_writer()
        
        del packer
        del ds
        gc.collect()
        
        elapsed = (time.time() - start_time) / 60
        print(f"[Worker {worker_id}] Group {chunk_idx+1}/{len(chunks)} done. Saved: {total_saved}. Time: {elapsed:.1f}m")
    
    if writer:
        writer.close()
    print(f"[Worker {worker_id}] OBFD packing done. Total: {total_saved}")


def run_obfd_mixed(num_processes: int):
    """
    运行 OBFD Packing 步骤 - 混合所有数据集
    收集所有数据集的 obfd 目录下的文件，混合在一起进行 OBFD 打包
    """
    print(f"\n{'='*20} Step 2: OBFD Packing (Mixed All Datasets) {'='*20}")
    
    output_dir = Config.MIXED_OBFD_PACKED_DIR
    print(f"Output: {output_dir}")
    
    # 收集所有数据集的 obfd 文件
    all_files = []
    for ds_name in Config.ALL_DATASETS:
        paths = get_output_paths(ds_name)
        input_dir = paths['obfd']
        
        print(f"Scanning {ds_name}: {input_dir}")
        bucket_name, prefix = parse_gcs_path(input_dir)
        files = list_gcs_files(bucket_name, prefix, suffix=".tfrecord")
        print(f"  Found {len(files)} files from {ds_name}")
        all_files.extend(files)
    
    print(f"\nTotal obfd files from all datasets: {len(all_files)}")
    
    if not all_files:
        print("No files found. Exiting.")
        return
    
    # 打乱文件顺序，使不同数据集的数据混合
    random.seed(42)
    random.shuffle(all_files)
    print("Files shuffled for better mixing.")
    
    # 分片
    chunk_size = len(all_files) // num_processes + 1
    chunks = [all_files[i:i + chunk_size] for i in range(0, len(all_files), chunk_size)]
    
    # 过滤空分片
    chunks = [c for c in chunks if c]
    
    process_args = [(output_dir, chunk, i) for i, chunk in enumerate(chunks)]
    
    print(f"Starting {len(process_args)} workers...")
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.starmap(obfd_worker, process_args)
    
    print(f"OBFD Packing (Mixed) completed.")


# ==============================================================================
# Step 3: Shuffle
# ==============================================================================

def shuffle_worker(output_dir: str, file_list: List[str], worker_id: int, seed: int):
    """Shuffle worker - 使用内存缓冲区进行 shuffle"""
    print(f"[Worker {worker_id}] Started shuffling. Files: {len(file_list)}")
    
    random.seed(seed + worker_id)  # 每个 worker 使用不同的种子
    
    # 使用缓冲区进行 shuffle
    buffer = []
    writer_idx = 0
    total_written = 0
    writer = None
    
    def get_writer():
        nonlocal writer_idx
        fname = f"Rank{worker_id:03d}.{writer_idx:04d}.tfrecord"
        fpath = os.path.join(output_dir, fname)
        writer_idx += 1
        return tf.io.TFRecordWriter(fpath)
    
    def flush_buffer():
        nonlocal writer, total_written
        if not buffer:
            return
        
        # Shuffle 缓冲区
        random.shuffle(buffer)
        
        for ids in buffer:
            if writer is None:
                writer = get_writer()
            
            write_to_tfrecord(writer, ids)
            total_written += 1
            
            if total_written % Config.SHUFFLE_OUTPUT_SAMPLES_PER_FILE == 0:
                writer.close()
                writer = get_writer()
        
        buffer.clear()
    
    start_time = time.time()
    
    # 首先 shuffle 文件顺序
    file_list = list(file_list)
    random.shuffle(file_list)
    
    # 读取数据到缓冲区
    ds = tf.data.TFRecordDataset(file_list, num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    for tensor_ids in ds.as_numpy_iterator():
        buffer.append(tensor_ids.tolist())
        
        # 缓冲区满时 flush
        if len(buffer) >= Config.SHUFFLE_BUFFER_SIZE:
            flush_buffer()
            elapsed = (time.time() - start_time) / 60
            print(f"[Worker {worker_id}] Written: {total_written}, Time: {elapsed:.1f}m")
    
    # Flush 剩余数据
    flush_buffer()
    
    if writer:
        writer.close()
    
    elapsed = (time.time() - start_time) / 60
    print(f"[Worker {worker_id}] Shuffle done. Total: {total_written}, Time: {elapsed:.1f}m")


def run_shuffle_mixed(num_processes: int):
    """
    运行 Shuffle 步骤 - 将 4k 数据和 packed 数据混合在一起 shuffle
    
    Shuffle 原理:
    1. 收集所有数据: 各数据集的 4k 文件 + mixed_obfd_packed 的文件
    2. 文件级 Shuffle: 打乱文件顺序，使 4k 和 packed 数据交错
    3. 样本级 Shuffle: 使用内存缓冲区进行样本级别的 shuffle
    4. 统一输出: 所有数据输出到同一个目录
    """
    print(f"\n{'='*20} Step 3: Shuffle (4k + Packed Mixed) {'='*20}")
    print(f"Shuffle buffer size: {Config.SHUFFLE_BUFFER_SIZE}")
    print(f"Shuffle seed: {Config.SHUFFLE_SEED}")
    print(f"Output: {Config.SHUFFLED_OUTPUT_DIR}")
    
    all_files = []
    
    # 1. 收集所有 4k 数据 (来自各个数据集)
    # 注意：除 dclm 外的数据集只取 1/10
    print("\nCollecting 4k files:")
    print("  (Note: non-dclm datasets are sampled at 1/10)")
    total_4k = 0
    total_4k_before_sample = 0
    
    for ds_name in Config.ALL_DATASETS:
        paths = get_output_paths(ds_name)
        input_dir = paths['4k']
        bucket_name, prefix = parse_gcs_path(input_dir)
        files = list_gcs_files(bucket_name, prefix, suffix=".tfrecord")
        total_4k_before_sample += len(files)
        
        if ds_name == 'dclm':
            # dclm: 保留所有文件
            print(f"  {ds_name}: {len(files)} files (100%)")
            all_files.extend(files)
            total_4k += len(files)
        else:
            # 其他数据集: 只取 1/10
            # 使用固定种子保证可复现性
            random.seed(Config.SHUFFLE_SEED)
            sample_size = max(1, len(files) // 10)  # 至少保留 1 个文件
            sampled_files = random.sample(files, min(sample_size, len(files)))
            print(f"  {ds_name}: {len(sampled_files)}/{len(files)} files (10%)")
            all_files.extend(sampled_files)
            total_4k += len(sampled_files)
    
    print(f"  Total 4k files: {total_4k} (from {total_4k_before_sample} before sampling)")
    
    # 2. 收集 packed 数据 (来自 mixed_obfd_packed)
    print("\nCollecting packed files:")
    bucket_name, prefix = parse_gcs_path(Config.MIXED_OBFD_PACKED_DIR)
    packed_files = list_gcs_files(bucket_name, prefix, suffix=".tfrecord")
    print(f"  mixed_obfd_packed: {len(packed_files)} files")
    all_files.extend(packed_files)
    
    print(f"\n{'='*40}")
    print(f"Total files to shuffle: {len(all_files)}")
    print(f"  - 4k files: {total_4k}")
    print(f"  - Packed files: {len(packed_files)}")
    print(f"{'='*40}")
    
    if not all_files:
        print("No files found. Exiting.")
        return
    
    # 3. 全局 shuffle 文件列表 (关键：让 4k 和 packed 数据混合)
    random.seed(Config.SHUFFLE_SEED)
    random.shuffle(all_files)
    print("\nFiles shuffled (4k and packed mixed together).")
    
    # 4. 分片分配给各个进程
    actual_processes = min(num_processes, len(all_files))
    chunk_size = len(all_files) // actual_processes + 1
    chunks = [all_files[i:i + chunk_size] for i in range(0, len(all_files), chunk_size)]
    chunks = [c for c in chunks if c]
    
    output_dir = Config.SHUFFLED_OUTPUT_DIR
    process_args = [
        (output_dir, chunk, i, Config.SHUFFLE_SEED) 
        for i, chunk in enumerate(chunks)
    ]
    
    print(f"Starting {len(process_args)} workers...")
    
    with multiprocessing.Pool(processes=len(process_args)) as pool:
        pool.starmap(shuffle_worker, process_args)
    
    print(f"\nShuffle completed. Output: {Config.SHUFFLED_OUTPUT_DIR}")


# ==============================================================================
# 主程序
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Data Processing Pipeline')
    parser.add_argument('--step', type=str, required=True,
                        choices=['tokenize', 'obfd', 'shuffle', 'all'],
                        help='Processing step: tokenize, obfd, shuffle, or all')
    parser.add_argument('--dataset', type=str, default='all',
                        help='Dataset(s) for tokenize: single name, comma-separated list, or "all"')
    parser.add_argument('--num-processes', type=int, default=Config.NUM_PROCESSES,
                        help='Number of parallel processes')
    parser.add_argument('--files-per-group', type=int, default=Config.FILES_PER_GROUP,
                        help='OBFD: files per group. Larger = better packing but more memory. (default: 120)')
    parser.add_argument('--shuffle-buffer-size', type=int, default=Config.SHUFFLE_BUFFER_SIZE,
                        help='Shuffle: buffer size. Larger = better shuffle but more memory. (default: 100000)')
    parser.add_argument('--shuffle-seed', type=int, default=Config.SHUFFLE_SEED,
                        help='Shuffle: random seed for reproducibility. (default: 42)')
    
    args = parser.parse_args()
    
    # 更新配置
    Config.FILES_PER_GROUP = args.files_per_group
    Config.SHUFFLE_BUFFER_SIZE = args.shuffle_buffer_size
    Config.SHUFFLE_SEED = args.shuffle_seed
    
    print(f"Configuration:")
    print(f"  - Processes: {args.num_processes}")
    print(f"  - Files per OBFD group: {Config.FILES_PER_GROUP}")
    print(f"  - Shuffle buffer size: {Config.SHUFFLE_BUFFER_SIZE}")
    print(f"  - Shuffle seed: {Config.SHUFFLE_SEED}")
    
    # Step 1: Tokenize (混合处理多个数据集)
    if args.step in ['tokenize', 'all']:
        # 解析数据集参数
        if args.dataset == 'all':
            datasets = Config.ALL_DATASETS
        else:
            # 支持逗号分隔的多个数据集
            datasets = [ds.strip() for ds in args.dataset.split(',')]
        
        run_tokenize_mixed(datasets, args.num_processes)
    
    # Step 2: OBFD Packing (混合所有数据集)
    if args.step in ['obfd', 'all']:
        run_obfd_mixed(min(args.num_processes // 3, 100))
    
    # Step 3: Shuffle (混合所有数据集)
    if args.step in ['shuffle', 'all']:
        run_shuffle_mixed(min(args.num_processes // 3, 100))
    
    print("\n" + "="*50)
    print("All processing completed!")
    print("="*50)


if __name__ == "__main__":
    main()

# python obdf.py --step all --dataset all --num-processes 500