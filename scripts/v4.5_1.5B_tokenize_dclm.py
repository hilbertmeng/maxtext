import os
import io
import time
import orjson
import zstandard as zstd
import tensorflow as tf
from google.cloud import storage
from transformers import AutoTokenizer
import multiprocessing
from functools import partial

# pip install orjson smart_open
## only dclm dataset data


# 配置路径
BASE_4k_DIR = 'gs://newproject-1-llm_base_models_us-east5/data/v4.5-1.5B/olmo-mix-1124-r0.1/dclm/4k/'
BASE_obfd_DIR = 'gs://newproject-1-llm_base_models_us-east5/data/v4.5-1.5B/olmo-mix-1124-r0.1/dclm/obfd/'
BUCKET_NAME = "newproject-1-data-xm4d5"
PREFIX = "datasets/olmo-mix-1124/data/dclm"
DATA_4k_NUM_PER_FILE = 10000
DATA_OBFD_NUM_PER_FILE = 50000

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def write_to_tfrecord(writer, input_ids):
    feature = {
        "input_ids": _int64_feature(input_ids),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())

def close_file(writer):
    if writer:
        try:
            writer.close()
        except Exception as e:
            print(f"Error closing writer: {e}")

def writer_factory(writer, output_dir, file_idx, worker_id):
    """
    创建 Writer，文件名包含 worker_id 以避免多进程冲突
    文件名格式: worker_{worker_id}_{idx}.tfrecord
    """
    close_file(writer)
    # 注意：这里增加了 worker_id 前缀
    save_name = f'Rank{worker_id:03}.{file_idx:04}.tfrecord'
    save_file = os.path.join(output_dir, save_name)
    # print(f'[Worker {worker_id}] New file: {save_file}')
    writer = tf.io.TFRecordWriter(save_file)
    return writer

def process_shard(file_list, worker_id):
    """
    单个进程的工作函数
    """
    print(f"[Worker {worker_id}] Started. Processing {len(file_list)} files.")
    
    # 在进程内部初始化资源
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    dctx = zstd.ZstdDecompressor()
    # 建议加上 use_fast=True 提高速度
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B", use_fast=True)
    
    max_length = 4096
    length_threshold = 4050
    
    count_4k = 0
    count_obfd = 0
    writer_4k = None
    writer_obfd = None
    total_processed = 0
    start_time = time.time()

    try:
        for blob_name in file_list:
            print(f"[Worker {worker_id}] Processing file: {blob_name}")
            blob = bucket.blob(blob_name)
            
            # 使用流式读取，减少内存占用
            try:
                with blob.open("rb") as compressed_stream:
                    with dctx.stream_reader(compressed_stream) as reader:
                        text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                        for line in text_stream:
                            try:
                                data = orjson.loads(line)
                                text = data.get('text', '')
                                if not text: continue
                                
                                input_ids = tokenizer.encode(text)
                                
                                # 处理长文本 (4k)
                                while len(input_ids) >= length_threshold:
                                    # 添加 BOS token
                                    save_ids = [tokenizer.bos_token_id] + input_ids[:max_length] 
                                    
                                    if count_4k % DATA_4k_NUM_PER_FILE == 0:
                                        file_idx = count_4k // DATA_4k_NUM_PER_FILE
                                        writer_4k = writer_factory(writer_4k, BASE_4k_DIR, file_idx, worker_id)
                                    
                                    # 修正：原代码这里写的是 input_ids，应该是 save_ids (包含bos)
                                    write_to_tfrecord(writer_4k, save_ids)
                                    
                                    input_ids = input_ids[max_length:]
                                    count_4k += 1
                                
                                # 处理剩余文本或短文本 (obfd)
                                if input_ids:
                                    input_ids += [tokenizer.eos_token_id] # add eos id
                                    if count_obfd % DATA_OBFD_NUM_PER_FILE == 0:
                                        file_idx = count_obfd // DATA_OBFD_NUM_PER_FILE
                                        writer_obfd = writer_factory(writer_obfd, BASE_obfd_DIR, file_idx, worker_id)
                                    write_to_tfrecord(writer_obfd, input_ids)
                                    count_obfd += 1

                                total_processed += 1
                                if total_processed % 1000 == 0:
                                    elapsed = (time.time() - start_time) / 60
                                    print(f'[Worker {worker_id}] Proc: {total_processed} | 4k: {count_4k} | obfd: {count_obfd} | Time: {elapsed:.1f}m')
                                    
                            except Exception as e:
                                print(f"[Worker {worker_id}] Error parsing line: {e}")
                                continue
            except Exception as e:
                print(f"[Worker {worker_id}] Error reading file {blob_name}: {e}")
                continue
                
    finally:
        # 确保进程结束时关闭文件
        close_file(writer_4k)
        close_file(writer_obfd)
        print(f"[Worker {worker_id}] Finished.")

def main():
    print("Listing blobs from GCS...")
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    # 只获取文件名列表，不要传递 Blob 对象（Blob对象难以序列化）
    blobs = [b.name for b in client.list_blobs(bucket, prefix=PREFIX) if b.name.endswith('.jsonl.zstd')]
    print(f"Total files to process: {len(blobs)}")

    num_processes = multiprocessing.cpu_count()
    # num_processes = 20
    print(f"Starting processing with {num_processes} processes...")

    chunk_size = len(blobs) // num_processes + 1
    chunks = [blobs[i:i + chunk_size] for i in range(0, len(blobs), chunk_size)]
    
    # 准备参数: [(file_list_1, 0), (file_list_2, 1), ...]
    process_args = []
    for i, chunk in enumerate(chunks):
        if chunk:
            process_args.append((chunk, i))

    # 4. 启动进程池
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.starmap(process_shard, process_args)

if __name__ == "__main__":
    main()