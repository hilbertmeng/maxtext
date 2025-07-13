import os
import random
import sys
import socket
import re

os.environ["JAX_PLATFORMS"] = "cpu"

from google.cloud import storage
import io
import multiprocessing
from multiprocessing import Pool
from functools import partial
import tensorflow as tf
from transformers import AutoTokenizer
import gcsfs
from tqdm import tqdm
import json
from multiprocessing import set_start_method
import math
import time
from etils import epath
from collections import defaultdict
import smart_open
import orjson
import argparse
import sentencepiece as spm

TOKENIZER_NAME = 'unigram'
MAX_LEN = 4097

if TOKENIZER_NAME == 'qwen':
    TOKENIZER_PATH = "/home/lishengping/my_qwen2_tokenizer_trained_on_60files_70000"
    TOKENIZER_SIZE = 70000
    EOS_ID = [0] # <|endoftext|>
    BOS_ID = [1] #  <|im_start|>

elif TOKENIZER_NAME == 'unigram':
    TOKENIZER_PATH = "/home/lishengping/spm_model_70000vocab_55G_bpe_character_coverage0.99999.model" # ：“ 等符号合一起
    TOKENIZER_PATH = "/home/lishengping/spm_model_70000vocab_55G_bpe_character_coverage0.99999.extended_special.model" # ：“ 等符号分开
    EOS_ID = [17] # <|endoftext|>
    BOS_ID = [18] #  <|im_start|>
    TOKENIZER_SIZE = 70000


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def write_to_tfrecord(writer, input_ids):
    feature = {
        "input_ids": _int64_feature(input_ids),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())


# SPLIT_IDS = {302, 4, 307, 304, 357, 376, 359, 507} # 空格 \n。.？?！!
def split_after_match(lst):
    for i in range(len(lst) - 1):
        if lst[i] == 4 and lst[i+1] != 4:
            return lst[i+1: ]
    return []


chapter_pat = re.compile('第(\d|[零一二三四五六七八九十百千]){1,}(章|节|卷|回)|^【\d+】|^\d+\.|^0\d+')
chapter_en_pat = re.compile('Chapter ?\d+|^【\d+】|^\d+\.|^0\d+')
chapter_digit = re.compile('(^-?\d{1,6}$)')

def match_chapter(line):
    if chapter_digit.search(line) or chapter_pat.search(line) or chapter_en_pat.search(line):
        return True
    return False
ahthor_pat = re.compile(
    "Qidian|Novel (status|words)|书友群|广大书友|求推荐票|-分[頁页]-|感谢.*(打赏|支持)|手机用户请到阅读|抱歉，更的晚|（群号|三更.{,2}第.更|推荐票|&amp;&amp;&amp;&amp|分割线|&[lg]t\;"
)
poison_content = re.compile(r'未 ?完待续|本章完|【已屏蔽|最新章节|/div>')

def contains_chinese(text):
    return re.search(r'[\u4e00-\u9fff]', text) is not None


def match_unused_content(line):
    line = line.strip()
    if poison_content.search(line) or ahthor_pat.search(line) or match_chapter(line):
    # if poison_content.search(line) or ahthor_pat.search(line):
        return True
    else:
        return False


def filter_unused_line(lines):
    lines = [l for l in lines if not match_unused_content(l)]
    return lines
    
class QwenTokenizer():
    def __init__(self, tokenizer_path, save_dir, rank):
        if TOKENIZER_NAME == 'qwen':
            self.tokenizer = AutoTokenizer.from_pretrained(
                TOKENIZER_PATH, use_fast=True, trust_remote_code=True
            )
        else:
            self.tokenizer = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
        
        self.partial_tokenize = partial(self.tokenize, max_len=MAX_LEN, bos_id=BOS_ID)
        assert len(self.tokenizer) == TOKENIZER_SIZE, print(len(self.tokenizer))
        self.next_ids = []
        self.count = 0
        self.perfile_nums = 10000
        self.save_dir = save_dir
        self.writer = None
        self.rank = rank + RANK_START_INDEX
        self.writer_factory()

    def writer_factory(self):
        if self.writer is not None:
            self.writer.close()
        save_path = os.path.join(self.save_dir, f'R{self.rank:03}.{self.count // self.perfile_nums:06}')
        print(f'Newest save path: {save_path} count: {self.count}')
        self.writer = tf.io.TFRecordWriter(save_path)
    
    def tokenize(self, text, max_len=4097, bos_id:list=[]):
        try:
            input_ids = self.tokenizer.encode(text)
            if input_ids[0] == 302: # 302为空格
                input_ids = input_ids[1: ]
        except:
            import pickle
            pickle.dump(text, open(f'error_{data_type}/{self.count}.pkl', 'wb'))
            print(f'error======')
            return []
        if bos_id:
            max_len -= 1
        self.next_ids += input_ids #  加上上个step保留的id
        # total_ids = []
        while len(self.next_ids) >= max_len:
            save_ids = self.next_ids[: max_len]
            if len(save_ids) == max_len:
                save_ids = bos_id + save_ids
                write_to_tfrecord(self.writer, save_ids)
                self.count += 1
                if self.count % self.perfile_nums == 0:
                    self.writer_factory()
                # total_ids.append(save_ids)
                self.next_ids = split_after_match(self.next_ids[max_len: ])
            else:
                self.next_ids = save_ids
                save_ids = []
        return []
 
def check_text_length(line):
    # 0524 add filter， 有些乱码数据很长一段
    text = line['text']
    words = text.split()
    char_count = line['meta']['char_count']
    if char_count < 2 or len(words) < 2:
        return False
    # 计算单词的平常长度
    word_mean_len = char_count / len(words)
    if word_mean_len > 5000:
        print(f'\n\nError line name: {line["meta"]}\n\n')
        return False
    else:
        return True

def process_data(args):
    cur_rank_pathes, rank, workers = args
    print(f'rank: {rank} cur_rank_pathes: {len(cur_rank_pathes)}')
    qwen_tokenizer = QwenTokenizer(TOKENIZER_PATH, SAVE_DIR, rank)
    for path in cur_rank_pathes:
        with smart_open.open(path, MODE) as f:
            cur_rank_lines = f.readlines()
            print(f'path:{path}, {len(cur_rank_lines)}')

        for i in tqdm(range(len(cur_rank_lines)), desc=f'Rank-{rank}'):
            line = cur_rank_lines[i]
            line = orjson.loads(line)
            text = line['text'] # 一本书
            wen_or_ju = '。' if random.randint(0, 1) else '？'
            text = re.subn('？。|\?。', wen_or_ju, text)[0]
            text_split = text.split('\n')
            text_split = filter_unused_line(text_split)
            if not check_text_length(line):
                continue
            per = 800
            if len(text_split) > per:
                # 一次Tokenize很长的数据会很慢，需要split。
                for lnx in tqdm(range(0, len(text_split), per), desc=f'Rank-{rank}-sub-{i}'):
                    inp = text_split[lnx: lnx + per]
                    inp = '\n'.join(inp)
                    qwen_tokenizer.partial_tokenize(inp)
            else:
                text = '\n'.join(text_split)
                qwen_tokenizer.partial_tokenize(text)
            if len(qwen_tokenizer.next_ids) < 300 and data_type == 'train': # 小于300token的书，扔掉
                    qwen_tokenizer.next_ids = []
            else:
                qwen_tokenizer.next_ids += EOS_ID

    qwen_tokenizer.writer.close()
    return qwen_tokenizer.count


def encode_file(pathes, workers=6):
    print(f'pathes {len(pathes)}')
    pool = Pool(processes=workers)
    perrank_line_num = math.ceil(len(pathes) / workers)
    args = ([pathes[rank * perrank_line_num: (rank + 1) * perrank_line_num], rank, workers] for rank in range(workers))
    counts = pool.map(process_data, args)  # 包含每个进程的返回值
    pool.close()
    pool.join()
    return counts


if __name__ == "__main__":
    # load args
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help="")
    parser.add_argument('--bucketes', type=str, default='0-40')
    parser.add_argument('--file_start', type=int, default=0)
    parser.add_argument('--file_end', type=int, default=None)
    parser.add_argument('--data_type', type=str, default="valid", help="valid, train")
    parser.add_argument('--workers', type=int, default=1, help="")   # GPU索引
    parser.add_argument('--work_index', type=int, default=0, help="")   # GPU索引
    parser.add_argument('--rank_start_index', type=int, default=0, help="")   # GPU索引

    args = parser.parse_args()

    random.seed(args.seed)

    file_start = args.file_start
    file_end = args.file_end
    data_type = args.data_type
    MODE = 'r' if data_type == 'valid' else 'rb'
    RANK_START_INDEX = args.rank_start_index

    os.makedirs(f'error_{data_type}', exist_ok=True)

    # set_start_method("spawn")  # tpu-vm
    num_processes = multiprocessing.cpu_count()
    print(f"num_processes: {num_processes}")
    meta_path = f'gs://newproject-1-jax_llm_data_europe-west4/xiaomeng/v3.5mini/meta.json'
    meta_path = epath.Path(meta_path)
   
    if data_type == 'valid':
        pathes = ['gs://newproject-1-jax_llm_data_europe-west4/xiaomeng/v3.5mini/jsonl/valid_concat.jsonl']
        SAVE_DIR = f'gs://newproject-1-jax_llm_data_europe-west4/xiaomeng/v3.5mini/unigram_tfids0714/validation'
        print(f'SAVE_DIR: {SAVE_DIR}')
    else:
        buckets = args.bucketes.split('-')
        bucket_start = int(buckets[0])
        bucket_end = int(buckets[1]) if len(buckets) == 2 else bucket_start  + 1
        # SAVE_DIR = f'gs://newproject-1-jax_llm_data_europe-west4/xiaomeng/v3.5mini/unigram_tfids0713/B{bucket_start}-{bucket_end}'
        SAVE_DIR = f'gs://newproject-1-jax_llm_data_europe-west4/xiaomeng/v3.5mini/unigram_tfids0714/B0-40'
        pathes = [f'gs://newproject-1-jax_llm_data_europe-west4/xiaomeng/v3.5mini/jsonl/2nd-shuffled-data_bucket-{bucket}-{index:03}-of-025.jsonl.zst' for bucket in range(bucket_start, bucket_end) for index in range(25)]
    print(f'total file nums: {len(pathes)}')
    run_pathes = pathes[file_start: file_end]
    counts = encode_file(run_pathes, workers=int(args.workers))
    print(f'counts: {counts}')
    meta_dict = {SAVE_DIR: counts}
    print(meta_dict)
    with meta_path.open('a') as f:
        meta_dict = json.dumps(meta_dict, ensure_ascii=False)
        f.write(f'{meta_dict}\n')

"""
# 文件不会被拆分为多个进程处理

多进程处理多个文件:
# Usage:
TPU_NAME=llm-jax-v5p-8-10; ZONE=europe-west4-b
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="/home/lishengping/miniconda3/bin/pip install -U tiktoken smart_open[gcs] gcsfs orjson transformers" --project=newproject-1-451205

gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="sudo rm -r /home/lishengping/tokenizer;gsutil cp -r gs://newproject-1-jax_llm_data_us-east5/xiaomeng/v3.5mini/tokenizer/my_qwen2_tokenizer_trained_on_60files_70000 /home/lishengping/" --project=newproject-1-451205
or: unigram
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="sudo rm -r /home/lishengping/tokenizer;gsutil cp -r gs://newproject-1-jax_llm_data_us-east5/xiaomeng/v3.5mini/tokenizer/spm_model_70000vocab_55G_bpe_character_coverage0.99999.extended_special.model /home/lishengping/" --project=newproject-1-451205
# scp
TPU_NAME=llm-jax-v5p-8-10; ZONE=europe-west4-b
SCRIPT=/Users/lishengping/codes/jax_projects/maxtext/scripts/process_files.py
gcloud compute tpus tpu-vm scp $SCRIPT $TPU_NAME:/home/lishengping/processed.py  --zone=$ZONE  --worker=all  --project=newproject-1-451205

# 
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=0 --command="killall processed.py;/home/lishengping/miniconda3/bin/python processed.py --bucket 0-40 --workers 1 --rank_start_index 0 --data_type valid 2>&1 | tee val.log" --project=newproject-1-451205

gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=0 --command="killall processed.py;/home/lishengping/miniconda3/bin/python processed.py --bucket 0-20 --workers 50 --rank_start_index 0 --data_type train 2>&1 | tee train.log " --project=newproject-1-451205
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=0 --command="killall processed.py;/home/lishengping/miniconda3/bin/python processed.py --bucket 20-40 --workers 50 --rank_start_index 50 --data_type train" --project=newproject-1-451205
"""