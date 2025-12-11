import time
from multiprocessing import Pool
import smart_open
from google.cloud import storage
import orjson
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np

# ==============================================================================
# 1. Worker 逻辑
# ==============================================================================

global_tokenizer = None

def init_worker():
    """每个进程启动时加载一次 Tokenizer"""
    global global_tokenizer
    try:
        global_tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B")
    except Exception as e:
        print(f"⚠️ Tokenizer load failed: {e}")

def process_lines(lines_chunk, worker_id):
    """
    接收一个 list 的 lines (bytes 或 str)，统计 token 数
    """
    tokenizer = global_tokenizer
    local_count = 0
    
    for i, line in enumerate(lines_chunk):
        try:
            if i % 10000 == 0:
                print(f"[Worker]-{worker_id} Processing line count: {i}/{len(lines_chunk)}")
            line_json = orjson.loads(line)
            text = line_json.get('text', '')
            if text:
                # 统计: encode长度 + 1个EOS
                local_count += len(tokenizer.encode(text)) + 1
        except Exception:
            continue
            
    return local_count

# ==============================================================================
# 2. 主流程：加载 -> 切分 -> 并行
# ==============================================================================

def load_all_lines_to_memory(bucket_name, prefix):
    """
    将所有文件的内容按行读取到内存列表中
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    
    all_data = []
    print(f"📥 Loading {len(blobs)} files into memory... (This may take a while)")
    
    t0 = time.time()
    for blob in blobs:
        path = f"gs://{bucket_name}/{blob.name}"
        print(f"   -> Reading {path} ...")
        
        # 直接 readlines()，利用你的大内存
        with smart_open.open(path, "rb") as f:
            lines = f.readlines()
            all_data.extend(lines)
            
    print(f"✅ Loaded {len(all_data)} lines in {time.time()-t0:.2f}s.")
    return all_data

if __name__ == "__main__":
    # --- 配置 ---
    dataset = 'wiki'
    bucket_name = "newproject-1-data-xm4d5"
    prefix = f"datasets/olmo-mix-1124/data/{dataset}"
    num_processes = 64
    # -----------

    # 1. 读取所有数据到内存 (List of bytes)
    all_lines = load_all_lines_to_memory(bucket_name, prefix)
    
    if not all_lines:
        print("❌ No data found.")
        exit()

    # 2. 按行数平均切分 (List Splitting)
    # np.array_split 虽然是针对numpy的，但处理list也很好用，能自动处理除不尽的情况
    print(f"🔪 Splitting {len(all_lines)} lines into {num_processes} chunks...")
    
    # 为了避免 numpy 将内容转为 array (太慢且占内存)，我们需要手动切分 list
    chunk_size = len(all_lines) // num_processes + 1
    chunks = [all_lines[i:i + chunk_size] for i in range(0, len(all_lines), chunk_size)]
    
    # 释放原始大 List 引用，协助 GC (虽然在 fork 模式下可能 Copy-on-Write)
    del all_lines 
    
    print(f"📦 Created {len(chunks)} tasks. Launching pool...")

    # 3. 多进程执行
    total_tokens = 0
    start_time = time.time()
    
    with Pool(processes=num_processes, initializer=init_worker) as p:
        # 使用 tqdm 显示进度
        iterator = p.imap_unordered(process_lines, chunks, [i for i in range(len(chunks))])
        
        with tqdm(total=len(chunks), unit="chunk") as pbar:
            for count in iterator:
                total_tokens += count
                
                # 更新进度条
                pbar.update(1)
                
                # 动态显示 Token 数
                if total_tokens > 1e9:
                    pbar.set_postfix(tokens=f"{total_tokens/1e9:.3f}B")
                else:
                    pbar.set_postfix(tokens=f"{total_tokens/1e6:.1f}M")

    # 4. 结果
    print("\n" + "="*50)
    print(f"✅ Processing Complete")
    print(f"🔢 Total Tokens: {total_tokens:,}")
    print(f"📊 Billions: {total_tokens / 1e9:.6f} B")
    print(f"⏱️ Time: {time.time() - start_time:.2f} s")
    print("="*50)