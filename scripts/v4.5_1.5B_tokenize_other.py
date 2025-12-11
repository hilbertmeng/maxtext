import os
import time
import orjson
import smart_open  # pip install smart_open[gcs]
import tensorflow as tf
from google.cloud import storage
from transformers import AutoTokenizer
import multiprocessing

# =================配置区域=================
BUCKET_NAME = "newproject-1-data-xm4d5"
PROJECT_ROOT = "gs://newproject-1-llm_base_models_us-east5/data/v4.5-1.5B/olmo-mix-1124-r0.1"

DATASETS_TO_PROCESS = ['algebraic-stack', 'arxiv', 'open-web-math', 'pes2o', 'starcoder', 'wiki']

# 输入数据的 GCS 前缀映射 (假设都在 datasets/olmo-mix-1124/data/ 下)
INPUT_PREFIX_MAP = {
    'algebraic-stack': "datasets/olmo-mix-1124/data/algebraic-stack",
    'arxiv': "datasets/olmo-mix-1124/data/arxiv",
    'open-web-math': "datasets/olmo-mix-1124/data/open-web-math",
    'pes2o': "datasets/olmo-mix-1124/data/pes2o",
    'starcoder': "datasets/olmo-mix-1124/data/starcoder",
    'wiki': "datasets/olmo-mix-1124/data/wiki"
}

DATA_4k_NUM_PER_FILE = 10000
DATA_OBFD_NUM_PER_FILE = 50000

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def write_to_tfrecord(writer, input_ids):
    feature = {"input_ids": _int64_feature(input_ids)}
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())

def writer_factory(output_dir, file_idx, worker_id):
    filename = f'Rank{worker_id:03}.{file_idx:04}.tfrecord'
    path = os.path.join(output_dir, filename)
    return tf.io.TFRecordWriter(path)

def process_shard(dataset_name, output_base_4k, output_base_obfd, file_paths, worker_id):
    """
    Worker 处理函数
    file_paths: 完整的 gs:// 路径列表
    """
    print(f"[Worker {worker_id}-{dataset_name}] Started. Files: {len(file_paths)}")
    
    # 初始化 Tokenizer
    # use_fast=True 非常重要，否则处理大量数据会很慢
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B", use_fast=True)
    
    max_length = 4096
    length_threshold = 4050 
    
    count_4k = 0
    count_obfd = 0
    writer_4k = None
    writer_obfd = None
    
    processed_files = 0
    
    try:
        for full_path in file_paths:
            try:
                count = 0
                with smart_open.open(full_path, "rb") as f:
                    for line in f:
                        try:
                            count += 1
                            if count % 10000 == 0:
                                print(f"[Worker {worker_id}] Processing file {full_path} count: {count}")
                            # 解析 JSON
                            data = orjson.loads(line)
                            text = data.get('text', '')
                            if not text: continue
                            
                            input_ids = tokenizer.encode(text)
                            
                            # --- 逻辑 A: 处理长文本 (存入 4k) ---
                            # DCLM 逻辑：只要够长，就切一段下来，前面加 BOS，存入 4k 数据集
                            while len(input_ids) >= length_threshold:
                                # [BOS] + text
                                save_ids = [tokenizer.bos_token_id] + input_ids[:max_length]
                                
                                if count_4k % DATA_4k_NUM_PER_FILE == 0:
                                    if writer_4k: writer_4k.close()
                                    idx = count_4k // DATA_4k_NUM_PER_FILE
                                    writer_4k = writer_factory(output_base_4k, idx, worker_id)
                                
                                write_to_tfrecord(writer_4k, save_ids)
                                
                                input_ids = input_ids[max_length:]
                                count_4k += 1
                            
                            # --- 逻辑 B: 处理剩余/短文本 (存入 obfd) ---
                            if input_ids:
                                # DCLM 逻辑：剩下的部分，后面加 EOS，存入 obfd 待拼接
                                input_ids = input_ids + [tokenizer.eos_token_id]
                                
                                if count_obfd % DATA_OBFD_NUM_PER_FILE == 0:
                                    if writer_obfd: writer_obfd.close()
                                    idx = count_obfd // DATA_OBFD_NUM_PER_FILE
                                    writer_obfd = writer_factory(output_base_obfd, idx, worker_id)
                                    
                                write_to_tfrecord(writer_obfd, input_ids)
                                count_obfd += 1
                                
                        except Exception as e:
                            # 忽略单行解析错误
                            continue
                            
            except Exception as e:
                print(f"[Worker {worker_id}] Error opening file {full_path}: {e}")
                continue
            
            processed_files += 1
            if processed_files % 10 == 0:
                 print(f"[Worker {worker_id}] Processed {processed_files} files...")

    finally:
        if writer_4k: writer_4k.close()
        if writer_obfd: writer_obfd.close()
        print(f"[Worker {worker_id}-{dataset_name}] Finished. 4k_count: {count_4k}, obfd_count: {count_obfd}")


def main():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    
    num_processes = multiprocessing.cpu_count() 
    num_processes = 50
    s, e = 0, 1
    for ds_name in DATASETS_TO_PROCESS[s:e]:
        print(f"\n{'='*20} Processing Dataset: {ds_name} {'='*20}")
        
        prefix = INPUT_PREFIX_MAP.get(ds_name)
        if not prefix:
            print(f"Skipping {ds_name}, no prefix found.")
            continue
            
        out_4k = f"{PROJECT_ROOT}/{ds_name}/4k/"
        out_obfd = f"{PROJECT_ROOT}/{ds_name}/obfd/"
        
        print(f"Listing blobs in {prefix}...")
        blobs = [os.path.join(f"gs://{BUCKET_NAME}", b.name) for b in client.list_blobs(bucket, prefix=prefix) 
                 if b.name.endswith('jsonl') or b.name.endswith('json.gz')]
        
        print(f"Total files: {len(blobs)}")
        if len(blobs) == 0: continue

        chunk_size = len(blobs) // num_processes + 1
        chunks = [blobs[i:i + chunk_size] for i in range(0, len(blobs), chunk_size)]
        
        process_args = []
        for i, chunk in enumerate(chunks):
            if chunk:
                process_args.append((ds_name, out_4k, out_obfd, chunk, i))

        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.starmap(process_shard, process_args)
            
    print("\nAll Datasets Processed (Step 1).")


if __name__ == "__main__":
    main()