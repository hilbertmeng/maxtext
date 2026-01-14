"""
统一数据处理流水线：Tokenize -> OBFD Packing -> Sample

流程说明：
  Step 1 (Tokenize): 将原始文本 tokenize，长文本切分为 4k 块，短文本存入 obfd
  Step 2 (OBFD):     每个数据集的短序列打包成 4k 块
  Step 3 (Sample):   从每个数据集的 4k 和 obfd_packed 中随机抽取 N 条数据保存到 train 文件夹

用法:
    # 1. Tokenize 单个数据集
    python processed.py --step tokenize --dataset dclm
    
    # 2. Tokenize 多个数据集
    python processed.py --step tokenize --dataset dclm,wiki,math
    
    # 3. Tokenize 所有数据集
    python processed.py --step tokenize --dataset all
    
    # 4. OBFD Packing (每个数据集单独打包)
    python processed.py --step obfd --dataset all
    
    # 5. Sample (从4k和obfd_packed中抽取数据到train，使用预配置的采样数量)
    python processed.py --step sample --dataset all
    
    # 6. 完整流水线 (tokenize -> obfd -> sample)
    python processed.py --step all --dataset all
    
    # 7. 控制 OBFD 每次处理的文件数 (内存不足时减小此值)
    python processed.py --step obfd --files-per-group 50
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
import re

# ==============================================================================
# 配置区域
# ==============================================================================
DATASET_NAME = "olmo-mix-1124/"
DATASET_NAME = "dolmino-mix-1124"

class Config:
    """全局配置"""
    # GCS 配置
    BUCKET_NAME = "newproject-1-data-xm4d5"
    PROJECT_ROOT = f"gs://newproject-1-llm_base_models_us-east5/data/v4.5-1.5B/{DATASET_NAME}-processed0113"
    
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
    
    # Sample 步骤配置
    # 每个数据集的采样数量
    DATASET_SAMPLE_SIZES = {
        'dclm': 590000,
        'flan': 207500,
        'math': 260000,
        'pes2o': 73125,
        'stackexchange': 30625,
        'wiki': 88875,
    }
    SAMPLE_OUTPUT_SAMPLES_PER_FILE = 10000  # 每个输出文件的样本数
    SAMPLE_SEED = 1234  # 随机种子，保证可复现性
    
    # 进程配置
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
        'obfd': f"{base}/{dataset_name}/obfd/",
        'obfd_packed': f"{base}/{dataset_name}/obfd_packed/",
        'train': f"{base}/{dataset_name}/train/",
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
# Step 1: Tokenize (每个数据集单独处理，支持多进程处理单文件)
# ==============================================================================

class TokenizeWorker:
    """Tokenize 工作器 - 处理单个数据集，所有数据存入 obfd"""
    
    def __init__(self, dataset_name: str, output_obfd: str, worker_id: int):
        self.dataset_name = dataset_name
        self.output_obfd = output_obfd
        self.worker_id = worker_id
        
        # 初始化 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_NAME, use_fast=True)
        self.eos_id = self.tokenizer.eos_token_id
        
        # 计数器和 writer
        self.count_obfd = 0
        self.writer_obfd = None
    
    def _get_writer(self, output_dir: str, count: int, samples_per_file: int) -> tf.io.TFRecordWriter:
        """获取或创建 writer"""
        file_idx = count // samples_per_file
        fname = f'Rank{self.worker_id:03}.{file_idx:04}.tfrecord'
        fpath = os.path.join(output_dir, fname)
        return tf.io.TFRecordWriter(fpath)
    
    def _save_to_obfd(self, input_ids: List[int]):
        """保存数据到 obfd"""
        if self.count_obfd % Config.DATA_OBFD_NUM_PER_FILE == 0:
            if self.writer_obfd:
                self.writer_obfd.close()
            self.writer_obfd = self._get_writer(
                self.output_obfd, self.count_obfd, Config.DATA_OBFD_NUM_PER_FILE
            )
        
        write_to_tfrecord(self.writer_obfd, input_ids)
        self.count_obfd += 1
    
    def process_text(self, text: str):
        """
        处理单条文本，所有数据存入 obfd
        - 长文本：切分为 4k 块，每块存入 obfd（不加 BOS/EOS，OBFD 阶段会加）
        - 短文本：直接存入 obfd（加 EOS）
        """
        if not text:
            return
        
        input_ids = self.tokenizer.encode(text)
        
        # 处理长文本：切分为 4k 块存入 obfd
        while len(input_ids) >= Config.LENGTH_THRESHOLD:
            # 切出 4k 块（不加 BOS，OBFD 阶段会判断长度后添加）
            save_ids = input_ids[:Config.MAX_SEQ_LENGTH]
            self._save_to_obfd(save_ids)
            input_ids = input_ids[Config.MAX_SEQ_LENGTH:]
        
        # 处理剩余/短文本：加 EOS 后存入 obfd
        if input_ids:
            input_ids = input_ids + [self.eos_id]
            self._save_to_obfd(input_ids)
    
    def close(self):
        """关闭 writer"""
        if self.writer_obfd:
            self.writer_obfd.close()
    
    def get_stats(self) -> int:
        """获取统计信息"""
        return self.count_obfd


_GLOBAL_TOKENIZE_WORKER = None


def _pool_worker_id() -> int:
    """为 multiprocessing.Pool 子进程生成稳定的 worker_id（同一个子进程生命周期内恒定）。"""
    ident = multiprocessing.current_process()._identity
    if not ident:
        return 0
    # Pool worker identity is 1..N
    return int(ident[0]) - 1


def _init_tokenize_pool_worker(dataset_name: str):
    """Pool initializer：在每个子进程内创建并持有一个 TokenizeWorker（writer+count 跨任务复用）。"""
    global _GLOBAL_TOKENIZE_WORKER
    worker_id = _pool_worker_id()
    paths = get_output_paths(dataset_name)
    _GLOBAL_TOKENIZE_WORKER = TokenizeWorker(dataset_name, paths['obfd'], worker_id)
    # 兜底：子进程退出时关闭 writer，避免最后一个文件句柄未 flush
    import atexit  # pylint: disable=import-outside-toplevel
    atexit.register(_GLOBAL_TOKENIZE_WORKER.close)


def tokenize_batch_worker_persistent(texts: List[str], chunk_id: int, file_idx: int) -> int:
    """
    Tokenize 批处理 worker（用于“单文件多进程模式”的持久 Pool）。
    - 每个子进程只创建一次 TokenizeWorker，并复用其 writer/count
    - 返回本次 task 新增写入的 obfd 样本数（delta），便于主进程统计
    """
    global _GLOBAL_TOKENIZE_WORKER
    if _GLOBAL_TOKENIZE_WORKER is None:
        # 理论上不会发生（除非未用 initializer），这里做个兜底
        worker_id = _pool_worker_id()
        paths = get_output_paths("unknown")
        _GLOBAL_TOKENIZE_WORKER = TokenizeWorker("unknown", paths['obfd'], worker_id)

    worker = _GLOBAL_TOKENIZE_WORKER
    worker_id = worker.worker_id
    before = worker.get_stats()

    print(f"[File{file_idx}-Chunk{chunk_id}-Rank{worker_id}] Started. Lines to process: {len(texts)}")
    start_time = time.time()
    total_lines = len(texts)

    for i, text in enumerate(texts):
        worker.process_text(text)
        if (i + 1) % 10000 == 0:
            elapsed = (time.time() - start_time) / 60
            progress = 100 * (i + 1) / total_lines if total_lines else 100.0
            print(
                f"[File{file_idx}-Chunk{chunk_id}-Rank{worker_id}] "
                f"Progress: {i+1}/{total_lines} ({progress:.1f}%) | "
                f"obfd_total: {worker.get_stats()} | Time: {elapsed:.1f}m"
            )

    after = worker.get_stats()
    delta = after - before
    elapsed = (time.time() - start_time) / 60
    print(
        f"[File{file_idx}-Chunk{chunk_id}-Rank{worker_id}] Finished. "
        f"Lines: {total_lines}, obfd_delta={delta}, obfd_total={after}, Time: {elapsed:.1f}m"
    )
    return delta


def tokenize_batch_worker(texts: List[str], worker_id: int, dataset_name: str):
    """
    Tokenize 批处理 worker - 处理分配的文本列表
    无锁竞争，每个进程独立处理自己的数据块
    
    Args:
        texts: 分配给该 worker 的文本列表
        worker_id: Worker ID
        dataset_name: 数据集名称
    
    Returns:
        处理的样本数
    """
    print(f"[Worker {worker_id}] Started. Lines to process: {len(texts)}")
    
    paths = get_output_paths(dataset_name)
    worker = TokenizeWorker(dataset_name, paths['obfd'], worker_id)
    
    start_time = time.time()
    total_lines = len(texts)
    
    for i, text in enumerate(texts):
        worker.process_text(text)
        
        # 每 10000 行报告一次进度
        if (i + 1) % 10000 == 0:
            elapsed = (time.time() - start_time) / 60
            cnt_obfd = worker.get_stats()
            progress = 100 * (i + 1) / total_lines
            print(f"[Worker {worker_id}] Progress: {i+1}/{total_lines} ({progress:.1f}%) | "
                  f"obfd: {cnt_obfd} | Time: {elapsed:.1f}m")
    
    worker.close()
    cnt_obfd = worker.get_stats()
    elapsed = (time.time() - start_time) / 60
    print(f"[Worker {worker_id}] Finished. Lines: {total_lines}, obfd={cnt_obfd}, Time: {elapsed:.1f}m")
    return cnt_obfd


def tokenize_files_worker(file_list: List[str], worker_id: int, dataset_name: str):
    """
    多文件 tokenize worker - 处理分配的文件列表
    无锁竞争，每个进程独立处理自己的文件，计数持续累加
    
    Args:
        file_list: 分配给该 worker 的文件列表
        worker_id: Worker ID
        dataset_name: 数据集名称
    
    Returns:
        处理的 obfd 样本数
    """
    print(f"[Worker {worker_id}] Started. Files to process: {len(file_list)}")
    
    paths = get_output_paths(dataset_name)
    worker = TokenizeWorker(dataset_name, paths['obfd'], worker_id)
    
    # 缓存 bucket 对象
    bucket_cache = {}
    
    start_time = time.time()
    total_files = len(file_list)
    total_lines = 0
    
    for file_idx, file_path in enumerate(file_list):
        # 处理单个文件
        file_lines = 0
        for data in read_jsonl_file(file_path, bucket_cache):
            text = data.get('text', '')
            worker.process_text(text)
            total_lines += 1
            file_lines += 1
        
        # 每处理完一个文件报告进度
        elapsed = (time.time() - start_time) / 60
        cnt_obfd = worker.get_stats()
        print(f"[Worker {worker_id}] File {file_idx+1}/{total_files} done. "
              f"Lines: {total_lines} (+{file_lines}) | obfd: {cnt_obfd} | Time: {elapsed:.1f}m")
    
    worker.close()
    cnt_obfd = worker.get_stats()
    elapsed = (time.time() - start_time) / 60
    print(f"[Worker {worker_id}] Finished. Files: {total_files}, Lines: {total_lines}, obfd: {cnt_obfd}, Time: {elapsed:.1f}m")
    return cnt_obfd


def run_tokenize_single_file_multiprocess(dataset_name: str, file_path: str, num_processes: int):
    """
    运行 Tokenize 步骤 - 多进程处理单个文件
    先读取所有文本到内存，然后按比例分配给各个进程独立处理
    无锁竞争，性能更好
    """
    print(f"\n  Processing file: {file_path}")
    print(f"  Using {num_processes} processes")
    
    start_time = time.time()
    
    # Step 1: 读取所有文本到内存
    print(f"  Reading file into memory...")
    all_texts = []
    bucket_cache = {}
    
    try:
        for data in read_jsonl_file(file_path, bucket_cache):
            text = data.get('text', '')
            if text:
                all_texts.append(text)
                
                if len(all_texts) % 100000 == 0:
                    elapsed = (time.time() - start_time) / 60
                    print(f"  Read {len(all_texts)} lines, Time: {elapsed:.1f}m")
    except Exception as e:
        print(f"  Error reading file: {e}")
        return
    
    total_lines = len(all_texts)
    read_time = (time.time() - start_time) / 60
    print(f"  Read completed. Total lines: {total_lines}, Time: {read_time:.1f}m")
    
    if total_lines == 0:
        print(f"  No data to process. Skipping.")
        return
    
    # Step 2: 将数据分配给各个进程
    actual_processes = min(num_processes, total_lines)
    chunk_size = (total_lines + actual_processes - 1) // actual_processes

    process_args = []
    for chunk_id in range(actual_processes):
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, total_lines)
        if start_idx < end_idx:
            texts_chunk = all_texts[start_idx:end_idx]
            # file_idx 在这个 wrapper 场景下未知，传 -1 仅用于日志
            process_args.append((texts_chunk, chunk_id, -1))

    print(f"  Data split into {len(process_args)} chunks, ~{chunk_size} lines each")
    print(f"  Starting {num_processes} persistent workers (pool initializer creates per-process writers)...")

    # Step 3: 多进程并行处理（单文件模式也用持久 worker，避免每个 task 都重新建 TokenizeWorker）
    process_start = time.time()
    with multiprocessing.Pool(
        processes=num_processes,
        initializer=_init_tokenize_pool_worker,
        initargs=(dataset_name,),
    ) as pool:
        results = pool.starmap(tokenize_batch_worker_persistent, process_args)

    total_obfd = sum(results)
    process_time = (time.time() - process_start) / 60
    total_time = (time.time() - start_time) / 60

    print(f"  File processing completed:")
    print(f"    - Total lines: {total_lines}")
    print(f"    - Total obfd samples: {total_obfd}")
    print(f"    - Read time: {read_time:.1f}m")
    print(f"    - Process time: {process_time:.1f}m")
    print(f"    - Total time: {total_time:.1f}m")
    return total_obfd


def run_tokenize_single_dataset(dataset_name: str, num_processes: int):
    """
    运行 Tokenize 步骤 - 单个数据集
    自动选择处理模式：
    - 文件数 >= num_processes: 多文件模式 (每个进程处理多个文件)
    - 文件数 < num_processes: 单文件模式 (多进程处理单个文件)
    """
    print(f"\n{'-'*40}")
    print(f"Tokenizing dataset: {dataset_name}")
    print(f"{'-'*40}")
    
    client = storage.Client()
    bucket = client.bucket(Config.BUCKET_NAME)
    
    prefix = Config.INPUT_PREFIX_MAP.get(dataset_name)
    if not prefix:
        print(f"  Unknown dataset: {dataset_name}, skipping")
        return
    
    # 收集文件
    files = []
    for blob in client.list_blobs(bucket, prefix=prefix):
        if is_supported_file(blob.name):
            file_path = f"gs://{Config.BUCKET_NAME}/{blob.name}"
            files.append(file_path)

    # if re.search(r'flan|wiki|stackexchange|pes2o|math', dataset_name):
    # if 'dclm' not in dataset_name:
    if 'dclm' in dataset_name:
        files = random.sample(files, max(1, len(files) // 8))
    elif 'pes2o' in dataset_name or 'wiki' in dataset_name or 'flan' in dataset_name:
        files = random.sample(files, max(1, len(files) // 10))
    elif 'math' in dataset_name or 'stackexchange' in dataset_name: 
        files = random.sample(files, max(1, len(files) // 6))
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    print(f"Dataset {dataset_name} Found {len(files)} files")
    
    if not files:
        print(f"  No files found. Skipping.")
        return
    
    # 打乱文件顺序
    random.seed(42)
    random.shuffle(files)
    
    total_files = len(files)
    
    # 根据文件数选择处理模式
    if total_files >= num_processes:
        # 多文件模式：将文件平均分配给各个进程
        print(f"  Mode: Multi-file (files >= processes)")
        actual_processes = num_processes
        
        # 将文件平均分配给各个进程
        chunk_size = (total_files + actual_processes - 1) // actual_processes
        process_args = []
        for i in range(actual_processes):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, total_files)
            if start_idx < end_idx:
                files_chunk = files[start_idx:end_idx]
                process_args.append((files_chunk, i, dataset_name))
        
        print(f"  Files split into {len(process_args)} chunks, ~{chunk_size} files each")
        print(f"  Starting {len(process_args)} workers...")
        
        start_time = time.time()
        with multiprocessing.Pool(processes=len(process_args)) as pool:
            results = pool.starmap(tokenize_files_worker, process_args)
        
        total_obfd = sum(results)
        elapsed = (time.time() - start_time) / 60
        print(f"  Multi-file mode completed. Total obfd: {total_obfd}, Time: {elapsed:.1f}m")
    else:
        # 单文件模式：多进程处理每个文件
        print(f"  Mode: Single-file multi-process (files < processes)")
        print(f"  Will use {num_processes} processes per file")

        # 关键改动：复用同一个 Pool，让每个子进程维护自己的 TokenizeWorker(writer+count)，
        # file 命名中的 n = count // DATA_OBFD_NUM_PER_FILE 可跨文件连续增长。
        total_obfd = 0
        with multiprocessing.Pool(
            processes=num_processes,
            initializer=_init_tokenize_pool_worker,
            initargs=(dataset_name,),
        ) as pool:
            for i, file_path in enumerate(files):
                print(f"\n  [{i+1}/{total_files}] Processing file...")
                # 读取文件、切 chunk 仍在主进程；写入在子进程持续累加
                obfd_count = _run_tokenize_single_file_multiprocess_with_pool(
                    dataset_name=dataset_name,
                    file_path=file_path,
                    pool=pool,
                    num_processes=num_processes,
                    file_idx=i,
                )
                total_obfd += obfd_count

        print(f"  Single-file mode completed. Total obfd: {total_obfd}")
    
    print(f"  Tokenize for {dataset_name} completed.")


def _run_tokenize_single_file_multiprocess_with_pool(
    dataset_name: str,
    file_path: str,
    pool: multiprocessing.pool.Pool,
    num_processes: int,
    file_idx: int,
) -> int:
    """
    单文件多进程（复用外部 Pool）：
    - pool 里的每个子进程都有一个持久 TokenizeWorker（initializer 创建）
    - 本函数只负责：主进程读取文件、分 chunk、派发给 pool
    """
    print(f"\n  Processing file [{file_idx}]: {file_path}")
    print(f"  Using {num_processes} processes (persistent pool)")

    start_time = time.time()
    print(f"  Reading file into memory...")
    all_texts = []
    bucket_cache = {}

    try:
        for data in read_jsonl_file(file_path, bucket_cache):
            text = data.get('text', '')
            if text:
                all_texts.append(text)
                if len(all_texts) % 100000 == 0:
                    elapsed = (time.time() - start_time) / 60
                    print(f"  Read {len(all_texts)} lines, Time: {elapsed:.1f}m")
    except Exception as e:
        print(f"  Error reading file: {e}")
        return 0

    total_lines = len(all_texts)
    read_time = (time.time() - start_time) / 60
    print(f"  Read completed. Total lines: {total_lines}, Time: {read_time:.1f}m")

    if total_lines == 0:
        print(f"  No data to process. Skipping.")
        return 0

    actual_processes = min(num_processes, total_lines)
    chunk_size = (total_lines + actual_processes - 1) // actual_processes
    process_args = []
    for chunk_id in range(actual_processes):
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, total_lines)
        if start_idx < end_idx:
            texts_chunk = all_texts[start_idx:end_idx]
            process_args.append((texts_chunk, chunk_id, file_idx))

    print(f"  Data split into {len(process_args)} chunks, ~{chunk_size} lines each")
    print(f"  Dispatching chunks to persistent pool...")

    process_start = time.time()
    results = pool.starmap(tokenize_batch_worker_persistent, process_args)

    total_obfd = sum(results)
    process_time = (time.time() - process_start) / 60
    total_time = (time.time() - start_time) / 60

    print(f"  File [{file_idx}] processing completed:")
    print(f"    - Total lines: {total_lines}")
    print(f"    - Total obfd samples: {total_obfd}")
    print(f"    - Read time: {read_time:.1f}m")
    print(f"    - Process time: {process_time:.1f}m")
    print(f"    - Total time: {total_time:.1f}m")
    return total_obfd


def run_tokenize(datasets: List[str], num_processes: int):
    """
    运行 Tokenize 步骤 - 每个数据集单独处理
    """
    print(f"\n{'='*20} Step 1: Tokenize (Per Dataset) {'='*20}")
    print(f"Datasets to process: {datasets}")
    print(f"Supported file formats: {Config.SUPPORTED_SUFFIXES}")
    
    for ds_name in datasets:
        run_tokenize_single_dataset(ds_name, num_processes)
    
    print("\nTokenize completed for all datasets.")


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
    """
    OBFD 打包 worker
    - 长度 >= LENGTH_THRESHOLD (4050): 直接添加 BOS 保存（长序列）
    - 长度 < LENGTH_THRESHOLD: 使用 OBFD 算法拼接（短序列）
    """
    print(f"[Worker {worker_id}] Started OBFD packing. Files: {len(file_list)}")
    
    tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_NAME, use_fast=True)
    bos_id = tokenizer.bos_token_id
    
    writer_idx = 0
    total_saved = 0
    long_count = 0  # 长序列计数（直接保存）
    packed_count = 0  # 短序列计数（OBFD 拼接）
    writer = None
    
    def get_writer():
        nonlocal writer_idx
        fname = f"Rank{worker_id:03d}.{writer_idx:04d}.tfrecord"
        fpath = os.path.join(output_dir, fname)
        writer_idx += 1
        return tf.io.TFRecordWriter(fpath)
    
    def save_sequence(seq):
        """保存序列到文件"""
        nonlocal writer, total_saved
        if writer is None:
            writer = get_writer()
        
        # 添加 BOS
        final_ids = [bos_id] + seq
        if len(final_ids) > Config.MAX_SEQ_LENGTH + 1:
            final_ids = final_ids[:Config.MAX_SEQ_LENGTH + 1]
        
        write_to_tfrecord(writer, final_ids)
        total_saved += 1
        
        if total_saved % Config.PACKED_SAMPLES_PER_FILE == 0:
            writer.close()
            writer = get_writer()
    
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
            seq_len = len(input_ids)
            
            # 判断长度：>= 4050 直接保存，< 4050 使用 OBFD 拼接
            if seq_len >= Config.LENGTH_THRESHOLD:
                # 长序列：直接保存（可能需要截断）
                if seq_len > Config.MAX_SEQ_LENGTH:
                    input_ids = input_ids[:Config.MAX_SEQ_LENGTH]
                save_sequence(input_ids)
                long_count += 1
            else:
                # 短序列：使用 OBFD 算法拼接
                for packed_seq in packer.add_sequence(input_ids):
                    save_sequence(packed_seq)
                    packed_count += 1
        
        # Flush 剩余的短序列
        for packed_seq in packer.flush():
            save_sequence(packed_seq)
            packed_count += 1
        
        del packer
        del ds
        gc.collect()
        
        elapsed = (time.time() - start_time) / 60
        print(f"[Worker {worker_id}] Group {chunk_idx+1}/{len(chunks)} done. "
              f"Total: {total_saved} (long: {long_count}, packed: {packed_count}). Time: {elapsed:.1f}m")
    
    if writer:
        writer.close()
    print(f"[Worker {worker_id}] OBFD packing done. Total: {total_saved} (long: {long_count}, packed: {packed_count})")


def run_obfd_per_dataset(num_processes: int, datasets: List[str] = None):
    """
    运行 OBFD Packing 步骤 - 每个数据集单独打包
    
    Args:
        num_processes: 每个数据集使用的进程数
        datasets: 要处理的数据集列表，None 表示所有数据集
    """
    print(f"\n{'='*20} Step 2: OBFD Packing (Per Dataset) {'='*20}")
    
    if datasets is None:
        datasets = Config.ALL_DATASETS
    
    print(f"Datasets to process: {datasets}")
    
    for ds_name in datasets:
        print(f"\n{'-'*40}")
        print(f"Processing dataset: {ds_name}")
        print(f"{'-'*40}")
        
        paths = get_output_paths(ds_name)
        input_dir = paths['obfd']
        output_dir = paths['obfd_packed']
        
        print(f"  Input:  {input_dir}")
        print(f"  Output: {output_dir}")
        
        # 收集该数据集的 obfd 文件
        bucket_name, prefix = parse_gcs_path(input_dir)
        files = list_gcs_files(bucket_name, prefix, suffix=".tfrecord")
        
        print(f"  Found {len(files)} obfd files")
        
        if not files:
            print(f"  No files found for {ds_name}. Skipping.")
            continue
        
        # 打乱文件顺序
        random.seed(42)
        random.shuffle(files)
        
        # 分片
        actual_processes = min(num_processes, len(files))
        chunk_size = len(files) // actual_processes + 1
        chunks = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]
        chunks = [c for c in chunks if c]
        
        process_args = [(output_dir, chunk, i) for i, chunk in enumerate(chunks)]
        
        print(f"  Starting {len(process_args)} workers for {ds_name}...")
        
        with multiprocessing.Pool(processes=len(process_args)) as pool:
            pool.starmap(obfd_worker, process_args)
        
        print(f"  {ds_name} OBFD packing completed.")
    
    print(f"\nOBFD Packing (Per Dataset) completed for all datasets.")


# ==============================================================================
# Step 3: Sample (从 4k 和 obfd_packed 中抽取数据到 train)
# ==============================================================================

def sample_worker(output_dir: str, files_with_indices: List[Tuple[str, List[int]]], worker_id: int):
    """
    Sample worker - 根据给定的文件和局部索引抽取样本
    
    Args:
        output_dir: 输出目录
        files_with_indices: [(文件路径, [要抽取的局部索引列表]), ...]
        worker_id: Worker ID
    """
    total_target = sum(len(indices) for _, indices in files_with_indices)
    print(f"[Worker {worker_id}] Started sampling. Files: {len(files_with_indices)}, Target samples: {total_target}")
    
    if total_target == 0:
        print(f"[Worker {worker_id}] No samples to extract. Done.")
        return 0
    
    writer_idx = 0
    total_written = 0
    writer = None
    
    def get_writer():
        nonlocal writer_idx
        fname = f"Rank{worker_id:03d}.{writer_idx:04d}.tfrecord"
        fpath = os.path.join(output_dir, fname)
        writer_idx += 1
        return tf.io.TFRecordWriter(fpath)
    
    start_time = time.time()
    
    for file_path, local_indices in files_with_indices:
        if not local_indices:
            continue
        
        local_indices_set = set(local_indices)
        
        # 读取单个文件
        ds = tf.data.TFRecordDataset([file_path])
        ds = ds.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
        
        current_idx = 0
        for tensor_ids in ds.as_numpy_iterator():
            if current_idx in local_indices_set:
                if writer is None:
                    writer = get_writer()
                
                write_to_tfrecord(writer, tensor_ids.tolist())
                total_written += 1
                
                if total_written % Config.SAMPLE_OUTPUT_SAMPLES_PER_FILE == 0:
                    writer.close()
                    writer = get_writer()
            
            current_idx += 1
    
    if writer:
        writer.close()
    
    elapsed = (time.time() - start_time) / 60
    print(f"[Worker {worker_id}] Sampling done. Total: {total_written}, Time: {elapsed:.1f}m")
    return total_written


def run_sample_per_dataset(num_processes: int, datasets: List[str] = None):
    """
    运行 Sample 步骤 - 从每个数据集的 obfd_packed 中随机抽取数据
    支持多进程并行采样
    
    Args:
        num_processes: 进程数
        datasets: 要处理的数据集列表，None 表示所有数据集
    """
    print(f"\n{'='*20} Step 3: Sample (Per Dataset) {'='*20}")
    
    if datasets is None:
        datasets = Config.ALL_DATASETS
    
    print(f"Datasets to process: {datasets}")
    print(f"Sample sizes per dataset: {Config.DATASET_SAMPLE_SIZES}")
    print(f"Random seed: {Config.SAMPLE_SEED}")
    
    for ds_name in datasets:
        print(f"\n{'-'*40}")
        print(f"Sampling dataset: {ds_name}")
        print(f"{'-'*40}")
        
        # 获取该数据集的采样数量
        sample_size = Config.DATASET_SAMPLE_SIZES.get(ds_name)
        if sample_size is None:
            print(f"  No sample size configured for {ds_name}. Skipping.")
            continue
        print(f"  Target sample size: {sample_size}")
        
        paths = get_output_paths(ds_name)
        output_dir = paths['train']
        
        # 收集该数据集的 obfd_packed 文件
        bucket_name, prefix = parse_gcs_path(paths['obfd_packed'])
        all_files = list_gcs_files(bucket_name, prefix, suffix=".tfrecord")
        print(f"  obfd_packed files: {len(all_files)}")
        print(f"  Output: {output_dir}")
        
        if not all_files:
            print(f"  No files found for {ds_name}. Skipping.")
            continue
        
        # Step 1: 统计每个文件的样本数
        print(f"  Counting samples per file...")
        file_sample_counts = []  # [(file_path, sample_count), ...]
        total_samples = 0
        for f in all_files:
            ds = tf.data.TFRecordDataset([f])
            count = sum(1 for _ in ds)
            file_sample_counts.append((f, count))
            total_samples += count
        
        print(f"  Total samples available: {total_samples}")
        
        # 确定实际抽取数量
        actual_sample_size = min(sample_size, total_samples)
        print(f"  Samples to extract: {actual_sample_size}")
        
        if actual_sample_size == 0:
            print(f"  No samples to extract. Skipping.")
            continue
        
        # Step 2: 随机选择全局样本索引
        random.seed(Config.SAMPLE_SEED)
        global_indices = sorted(random.sample(range(total_samples), actual_sample_size))
        
        # Step 3: 将全局索引映射到 (文件, 局部索引)
        # 构建文件索引范围: [(start, end, file_path), ...]
        file_ranges = []
        cumulative = 0
        for file_path, count in file_sample_counts:
            file_ranges.append((cumulative, cumulative + count, file_path))
            cumulative += count
        
        # 将全局索引分配到各个文件
        # file_to_local_indices: {file_path: [local_indices]}
        file_to_local_indices = defaultdict(list)
        file_idx = 0
        for global_idx in global_indices:
            # 找到包含该全局索引的文件
            while file_idx < len(file_ranges) and global_idx >= file_ranges[file_idx][1]:
                file_idx += 1
            
            if file_idx < len(file_ranges):
                start, end, file_path = file_ranges[file_idx]
                local_idx = global_idx - start
                file_to_local_indices[file_path].append(local_idx)
        
        # Step 4: 准备多进程任务
        # 将文件分配给各个 worker
        files_with_indices = [(f, file_to_local_indices.get(f, [])) for f, _ in file_sample_counts]
        # 只保留有样本要抽取的文件
        files_with_indices = [(f, indices) for f, indices in files_with_indices if indices]
        
        print(f"  Files with samples to extract: {len(files_with_indices)}")
        
        # 分配给各个进程
        actual_processes = min(num_processes, len(files_with_indices))
        if actual_processes == 0:
            print(f"  No files to process. Skipping.")
            continue
        
        # 按文件数量均匀分配
        chunk_size = len(files_with_indices) // actual_processes + 1
        worker_tasks = []
        for i in range(actual_processes):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, len(files_with_indices))
            if start < end:
                worker_tasks.append((output_dir, files_with_indices[start:end], i))
        
        print(f"  Starting {len(worker_tasks)} workers...")
        
        start_time = time.time()
        
        # 多进程执行
        with multiprocessing.Pool(processes=len(worker_tasks)) as pool:
            results = pool.starmap(sample_worker, worker_tasks)
        
        total_written = sum(results)
        elapsed = (time.time() - start_time) / 60
        print(f"  {ds_name} sampling completed. Total: {total_written}/{actual_sample_size}, Time: {elapsed:.1f}m")
        
        if total_written != actual_sample_size:
            print(f"  WARNING: Expected {actual_sample_size} samples, but got {total_written}")
    
    print(f"\nSample (Per Dataset) completed for all datasets.")


# ==============================================================================
# 主程序
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Data Processing Pipeline')
    parser.add_argument('--step', type=str, required=True,
                        choices=['tokenize', 'obfd', 'sample', 'all'],
                        help='Processing step: tokenize, obfd, sample, or all')
    parser.add_argument('--dataset', type=str, default='all',
                        help='Dataset(s): single name, comma-separated list, or "all"')
    parser.add_argument('--num-processes', type=int, default=Config.NUM_PROCESSES,
                        help='Number of parallel processes')
    parser.add_argument('--files-per-group', type=int, default=Config.FILES_PER_GROUP,
                        help='OBFD: files per group. Larger = better packing but more memory. (default: 120)')
    parser.add_argument('--sample-seed', type=int, default=Config.SAMPLE_SEED,
                        help='Sample: random seed for reproducibility. (default: 1234)')
    
    args = parser.parse_args()
    
    # 更新配置
    Config.FILES_PER_GROUP = args.files_per_group
    Config.SAMPLE_SEED = args.sample_seed
    
    # 解析数据集参数
    if args.dataset == 'all':
        datasets = Config.ALL_DATASETS
    else:
        datasets = [ds.strip() for ds in args.dataset.split(',')]
    
    print(f"Configuration:")
    print(f"  - Step: {args.step}")
    print(f"  - Datasets: {datasets}")
    print(f"  - Processes: {args.num_processes}")
    print(f"  - Files per OBFD group: {Config.FILES_PER_GROUP}")
    print(f"  - Sample sizes: {Config.DATASET_SAMPLE_SIZES}")
    print(f"  - Sample seed: {Config.SAMPLE_SEED}")
    
    # Step 1: Tokenize (每个数据集单独处理)
    if args.step in ['tokenize', 'all']:
        run_tokenize(datasets, args.num_processes)
    
    # Step 2: OBFD Packing (每个数据集单独打包)
    if args.step in ['obfd', 'all']:
        run_obfd_per_dataset(args.num_processes, datasets)
    
    # Step 3: Sample (从 4k 和 obfd_packed 中抽取数据到 train)
    if args.step in ['sample', 'all']:
        run_sample_per_dataset(args.num_processes, datasets)
    
    print("\n" + "="*50)
    print("All processing completed!")
    print("="*50)


if __name__ == "__main__":
    main()

# pip install orjson smart_open
# 使用示例：
# Tokenize：
# python processed.py --step tokenize --dataset stackexchange --num-processes 2
# python processed.py --step tokenize --dataset pes2o --num-processes 10
# python processed.py --step tokenize --dataset wiki --num-processes 50
# python processed.py --step tokenize --dataset math --num-processes 50
# python processed.py --step tokenize --dataset dclm --num-processes 100
# python processed.py --step tokenize --dataset flan --num-processes 100

# OBFD：
# python processed.py --step obfd --dataset stackexchange --files-per-group 120 --num-processes 1

# python processed.py --step obfd --dataset dclm --files-per-group 60 --num-processes 2
# python processed.py --step obfd --dataset wiki --files-per-group 120 --num-processes 1
# python processed.py --step obfd --dataset flan --files-per-group 200 --num-processes 1
# python processed.py --step obfd --dataset math --files-per-group 200 --num-processes 1
# python processed.py --step obfd --dataset pes2o --files-per-group 200 --num-processes 1
# python processed.py --step obfd --dataset stackexchange --files-per-group 200 --num-processes 1


# Sample # 获取手动配置的采样数量
# python processed.py --step sample --dataset stackexchange
# 完整流程：
# python processed.py --step all --dataset all
