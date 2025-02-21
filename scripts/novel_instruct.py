import time
import os
import sys
import random
import multiprocessing
from multiprocessing import set_start_method
import socket

os.environ["JAX_PLATFORMS"] = "cpu"

import tensorflow as tf
from transformers import AutoTokenizer
from etils import epath
import math
from tqdm import tqdm

import re
from datetime import datetime
from collections import deque, defaultdict


prompt_dict = {'下': ['Go on with the paragraph in a similar style of language.',
              'Continue the paragraph while maintaining a consistent style in the language.',
              '针对下文，以一种风格上连贯的方式进行创作性扩写。',
              '请继续扩展下文，确保风格连贯一致。',
              'Write a paragraph ensuring that the style is consistent from start to finish.',
              'Compose a paragraph, making sure the writing style remains consistent throughout.',
              'Keep extending the paragraph, following the same style of language.',
              'Continue expanding the paragraph while adhering to the same language style.',
              '继续这一段落的故事，注意保持原有风格的一致性。',
              '续写这一段，保持风格与前文一致。',
              'Create a paragraph where the style remains constant throughout.',
              'Develop a paragraph in which the style remains steady and unchanged from beginning to end.',
              '延续以下段落的风格，进行相应的续写工作。',
              '请保持段落风格的一致性，并对其进行续写。',
              'Your assignment is to craft a paragraph with a consistent stylistic approach.',
              'Your task is to write a paragraph while maintaining a uniform stylistic tone.',
              '保持原段落的写作风格，对其进行扩展和深化。',
              '请继续扩展原段落，并保持其写作风格的统一。',
              'Proceed with the paragraph, maintaining the same style of language.',
              'Carry on with the paragraph while keeping the language style consistent.',
              'Compose a paragraph, ensuring stylistic consistency throughout.',
              'Write a paragraph and ensure the style is consistent from start to finish.',
              'Produce a paragraph in which the style does not vary from beginning to end.',
              'Create a paragraph where the writing style remains unchanged throughout.',
              'Construct a paragraph, keeping the writing style uniform in the entire passage.',
              'Build a paragraph, ensuring the writing style stays uniform throughout.',
              '对于下面的Paragraph，使用一致性的风格进行续写。',
              '请对以下段落进行续写，确保风格统一。',
              'Continue crafting the paragraph, maintaining the established language style.',
              'Keep developing the paragraph while preserving the established style.',
              '请以与下面段落相同的风格，进行创作性的续写。',
              '请保持与下方段落相同的风格，进行创作性续写。',
              'Extend the paragraph, keeping the language tone and style similar.',
              'Expand the paragraph while keeping the tone and style consistent.',
              '针对以下段落，以相同的风格进行补充写作。',
              '请使用相同的风格，对以下段落进行补充写作。',
              '续写下面的段落，并确保风格与原文保持一致。',
              '请继续下面的段落，并保持风格一致性。',
              'Focus on writing a paragraph with a steady and unvarying style.',
              'Concentrate on creating a paragraph that maintains a steady and uniform style.',
              'Your challenge is to author a paragraph that maintains a consistent style.',
              'Your task is to write a paragraph that preserves consistency in style.',
              'Develop a paragraph that demonstrates consistency in style.',
              'Craft a paragraph that clearly maintains stylistic consistency.',
              'Proceed with extending the paragraph, preserving the linguistic style.',
              'Continue expanding the paragraph, ensuring the linguistic style remains consistent.',
              'Your task is to write paragraph in a consistent style.',
              'Your assignment is to draft a paragraph while keeping the writing style consistent.',
              'Carry on writing the paragraph, sticking to the current language style.',
              'Keep writing the paragraph, ensuring that the current language style remains unchanged.',
              'Resume the paragraph, mirroring the existing language style.',
              'Continue the paragraph by closely following the current language style.',
              '请在下面的段落后续写，确保风格的连贯性。',
              '请继续下面的段落，并确保风格连贯一致。',
              'Please compose a paragraph maintaining a uniform style throughout.',
              'Please write a paragraph, ensuring uniformity in style from beginning to end.',
              '请继续下面的段落，并保持原有的写作风格一致。',
              '请续写下方段落，保持原有风格的一致性。',
              '在接下来的写作中，延续这一段落的风格，并进行扩展。',
              '在接下来的写作中，请延续该段落的风格并进行扩展。',
              'Keep writing the paragraph, using a similar linguistic style.',
              'Continue writing the paragraph while maintaining a similar linguistic approach.',
              'Continue the paragraph, ensuring the language style remains consistent.',
              'Proceed with the paragraph, making sure the language style stays consistent.',
              '请续写下文',
              '续写下文',
              '续写'],
             '上': ['针对上文，以一种风格上连贯的方式进行创作性扩写。',
              '请继续扩展上文，确保风格连贯一致。',
              '继续这一段落的故事，注意保持原有风格的一致性。',
              '续写这一段，保持风格与前文一致。',
              '延续以上段落的风格，进行相应的续写工作。',
              '请保持段落风格的一致性，并对其进行续写。',
              '保持原段落的写作风格，对其进行扩展和深化。',
              '请继续扩展原段落，并保持其写作风格的统一。',
              '对于上面的Paragraph，使用一致性的风格进行续写。',
              '请对以上段落进行续写，确保风格统一。',
              '请以与上面段落相同的风格，进行创作性的续写。',
              '请保持与上方段落相同的风格，进行创作性续写。',
              '针对以上段落，以相同的风格进行补充写作。',
              '请使用相同的风格，对以上段落进行补充写作。',
              '续写上面的段落，并确保风格与原文保持一致。',
              '请继续上面的段落，并保持风格一致性。',
              '请在上面的段落后续写，确保风格的连贯性。',
              '请继续上面的段落，并确保风格连贯一致。',
              '请继续上面的段落，并保持原有的写作风格一致。',
              '请续写上方段落，保持原有风格的一致性。',
              '延续这一段落的风格，并进行扩展。',
              '请延续该段落的风格并进行扩展。',
              '请续写上文',
              '续写上文',
              '续写']}

# null_content = re.compile(
#     "Qidian|Novel (name|status|words|category)|书友群|广大书友|求推荐票|---分頁---|感谢.*(打赏|支持)|手机用户请到阅读|抱歉，更的晚|（群号|三更.{,2}第.更|推荐票|&amp;&amp;&amp;&amp"
# )
# poison_content = re.compile(r'本章完|第(\d|[零一二三四五六七八九十百千]){1,}(章|节|卷)|(^\d{1,5}$)|未 ?完待续|(^\d{1,5}\.)')
# chapter_pat = re.compile('第(\d|[零一二三四五六七八九十百千]){1,}(章|节|卷)')
chapter_pat = re.compile('第(\d|[零一二三四五六七八九十百千]){1,}(章|节|卷|回)|^【\d+】|^\d+\.|^0\d+')

chapter_en_pat = re.compile('Chapter ?\d+|^【\d+】|^\d+\.|^0\d+')

novel_name_pat = re.compile('Novel name')
novel_category_pat = re.compile('Novel category')
ahthor_pat = re.compile(
    "Qidian|Novel (status|words)|书友群|广大书友|求推荐票|-分[頁页]-|感谢.*(打赏|支持)|手机用户请到阅读|抱歉，更的晚|（群号|三更.{,2}第.更|推荐票|&amp;&amp;&amp;&amp|分割线|&[lg]t\;"
)
poison_content = re.compile(r'未 ?完待续|本章完')
chapter_digit = re.compile('(^-?\d{1,6}$)')


def match_name_category(line):
    if not novel_name_pat.match(line):
        line = 'Novel name: \n'
    if not novel_category_pat.match(line):
        line = 'Novel category: \n'
    return line


def match_chapter(line):
    if chapter_pat.search(line):
        return True
    return False


def match_en_chapter(line):
    if chapter_en_pat.search(line):
        return True
    return False


def match_digit_chapter(line):
    if chapter_digit.match(line):
         return True
    return False


def match_unused_content(line):
    if poison_content.search(line) or ahthor_pat.search(line):
        return True
    else:
        return False


TOKENIZER_PATH = "/home/lishengping/tokenizer"
EOS_ID = 151643 # <|endoftext|>
BOS_ID = 151646 #  <|extra_0|>

class DataProcessor:
    def __init__(
        self,
        data_pathfile,
        save_dir,
        data_type="zh",
        max_seq_len=2048,
        ratio=1.0,
        shuffle=True,
        epoches=100,
    ):
        bucket = True
        if bucket:
            self.path_map = {
                "zh": ["/mnt/nvme1/kf/data/69shuba", "gs://jax_llm_data_us-east5/xiaomeng/zh_data"],
                "en": ["/mnt/nvme1/kf/data/formal_data", "gs://jax_llm_data_us-east5/xiaomeng/en_data"],
            }
        else:
            self.path_map = {
                "zh": ["/mnt/nvme1/kf/data/69shuba", "/nas2/xiaomeng/zh_data"],
                "en": ["/mnt/nvme1/kf/data/formal_data", "/nas2/xiaomeng/en_data"],
            }
        self.data_pathfile = data_pathfile
        self.data_type = data_type
        self.shuffle = shuffle
        self.epoches = epoches
        self.ratio = min(ratio, 1)
        assert self.ratio > 0
        print("Init tokenizer.....")
        self.tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_PATH, use_fast=False, trust_remote_code=True
        )
        print("Init tokenizer finished.....")
        self.max_seq_len = max_seq_len
        self.save_dir = save_dir
        self.books_pathlist = []
        self.write_line = 0
        self.writer = None
        self.extract_filepath()
        self.file_count = 0
        self.book_end_id = self.tokenizer.encode('<|endoftext|>\n')
        self.user = self.tokenizer.encode('user:\n')
        self.assistant = self.tokenizer.encode('\n\nassistant:\n')

        self.pad_id = -100

        self.book_input_ids = [BOS_ID] + self.user
        self.book_masks = [self.pad_id] * len(self.book_input_ids)

        self.segment_num = {'zh': 2, 'en': 0} # 每本书取前多少条数据作为一次迭代
        self.last_n_lines = deque(maxlen=3)

        self.prompt_ids = defaultdict(list)
        for k, vs in prompt_dict.items():
            for v in vs:
                if k == '上':
                    v =  '\n' + v.strip()
                else:
                    v =  v.strip() + '\n'
                v_id = self.tokenizer.encode(v)
                self.prompt_ids[k].append(v_id)

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def extract_filepath(self):
        random.seed(42)
        self.books_pathlist = []
        with epath.Path(self.data_pathfile).open('r') as f:
            for line in f:
                line = line.strip().replace(*self.path_map[self.data_type])
                self.books_pathlist.append(line)
        if self.shuffle:
            self.books_pathlist = random.sample(
                self.books_pathlist, k=int(len(self.books_pathlist) * self.ratio)
            )

    def convert_line_to_ids(self, line):
        return self.tokenizer.encode(line)

    def writer_factory(self):
        if self.write_line % self.per_file_line_num == 0:
            if self.write_line > 0: self.writer.close()
            name = f"{self.data_type}_R{self.rank}_E{self.epoch}_b{self.write_line}"
            save_path = os.path.join(self.save_dir, name)
            self.writer = tf.io.TFRecordWriter(save_path)
            self.file_count += 1

    def write_file(self):
        insert_index = random.randint(50, 1000)
        fir_input_ids = self.book_input_ids[: insert_index]
        sec_input_ids = self.book_input_ids[insert_index:]

        fir_mask_ids = self.book_masks[: insert_index]
        sec_mask_ids = self.book_masks[insert_index:]

        if random.randint(0, 1):
            insert_prompt = '上'
        else:
            insert_prompt = '下'
        prompt_ids = self.prompt_ids[insert_prompt]
        prompt_id = random.sample(prompt_ids, k=1)[0]

        if insert_prompt == '上':
            input_ids = fir_input_ids + prompt_id + self.assistant + sec_input_ids
            # masks = len(fir_input_ids) * [0] + len(prompt_id) * [0] + len(self.assistant) * [0]  + sec_mask_ids
        else:
            input_ids = [BOS_ID] + self.user + prompt_id + fir_input_ids[1+len(self.user): ] + self.assistant + sec_input_ids
        
        masks = (len(prompt_id) + len(fir_input_ids) + len(self.assistant)) * [self.pad_id]  + sec_mask_ids  # Assistant之后才计算loss

        input_ids = input_ids[: self.max_seq_len]
        masks = masks[: self.max_seq_len]

        assert len(input_ids) == len(masks)

        masks[1] = self.pad_id  # lsp: bos id 后的第一个token不计算loss
        feature = {
            "input_ids": self._int64_feature(input_ids),
            "labels": self._int64_feature(masks),
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        self.writer.write(example.SerializeToString())
        self.write_line += 1
        # 每次都从bos开始
        self.book_input_ids = [BOS_ID] + self.user
        self.book_masks = [self.pad_id] * len(self.book_input_ids)

    def process_book(self, path, start_index=0):
        with epath.Path(path).open('r') as fr:
            lines = fr.readlines()
            # 去掉很短的书
            if len(lines) < 200:
                print(f'Short book: {path}')
                self.run_book_index.pop(path)
                return 
            count = 0
            _n = 1
            index = start_index
            while index < len(lines):
            # for index in range(start_index, len(lines), 1):
                line = lines[index].strip()
                index += 1
                if match_digit_chapter(line): continue
                if not line:
                    _n += 1  # 统计换行
                    continue
                clear_space_line = line.replace(' ', '')  # 统计重复句子
                if self.last_n_lines.count(clear_space_line) >= 1:
                    print(f'重复: {clear_space_line} path: {path}')
                    continue
                if self.data_type == 'zh' and match_unused_content(line):
                    continue
                if len(clear_space_line) > 15 and self.data_type == 'zh':
                    self.last_n_lines.append(clear_space_line)
                _n = min(2, _n) # 最多2个换行
                line = '\n' * _n + line # n 个 换行
                _n = 1
                ids = self.convert_line_to_ids(line)
                if (self.data_type == 'zh' and match_chapter(line)) or (self.data_type == 'en' and match_en_chapter(line)) :
                    masks = len(ids) * [self.pad_id]
                else:
                    masks = len(ids) * [1]
                    
                if index == len(lines) - 1:
                    ids.extend(self.book_end_id)
                    masks.extend([1] * len(self.book_end_id))
                self.book_input_ids.extend(ids)
                self.book_masks.extend(masks)
                # print(f'book_masks: {len(self.book_masks)} book_input_ids: {len(self.book_input_ids)}')

                if len(self.book_input_ids) >= self.max_seq_len:
                    self.writer_factory()
                    self.write_file()
                    e = max(len(lines) - index - 3500, 1)
                    index = index + random.randint(1, e)
                    if count >= self.segment_num[self.data_type]:  # 每个epoch从每本书中读取 几条 数据
                        self.run_book_index[path] = index # 读取到书的第几行
                        index = len(lines) - 1
                        break
                    count += 1
            if index == len(lines) - 1:
                # 换书之后清空
                self.book_input_ids = [BOS_ID] + self.user
                self.book_masks = [self.pad_id] * len(self.book_input_ids)
                self.run_book_index.pop(path) # 结束后，删除该书
            

    def __len__(self):
        return self.size

    def size(self):
        return len(self.books_pathlist)

    def run(self, start, end, rank):
        random.seed(42)
        time_start = time.time()
        print(f"Rank: {rank}. book: {len(self.books_pathlist)} self.epoches: {self.epoches}")
        self.processor_bookes = self.books_pathlist[start: end]
        # s = random.randint(0, 1000) if self.data_type == 'zh' else random.randint(0, 100)
        self.run_book_index = {book: random.randint(0, 100) for book in self.processor_bookes} # 随机开头索引
        for epoch in tqdm(range(self.epoches), desc=f'Rank-{rank}'):
            N = len(self.run_book_index)
            if N == 0: break  # 所有书处理完成中断
            self.epoch = epoch
            # 前1轮的时候，可以允许数据不写满per_file_line_num
            if self.epoch == 1:
                self.write_line = 0
                self.writer.close()
            items = list(self.run_book_index.items())
            #每轮都shuffle一遍
            random.shuffle(items)
            self.run_book_index = dict(items)
            for path, start_index in tqdm(self.run_book_index.copy().items(), desc=f'Processing-epoch{epoch}-{rank}'):
                self.process_book(path, start_index)
        try:
            self.writer.close()
        except Exception as e:
            pass

def process_book_wrapper(args):
    rank, LANG, WORKERS, max_seq_len, ratio = args
    bucket = True
    if bucket:
        data_pathfiles = {
            "zh": "gs://jax_llm_data_us-east5/xiaomeng/zh_data/69shuba.filelist.shuffled",
            "en": "gs://jax_llm_data_us-east5/xiaomeng/en_data/allfile.filelist.shuffed",
        }
    else:
        data_pathfiles = {
            "zh": "/nas2/xiaomeng/zh_data/69shuba.filelist.shuffled",
            "en": "/nas2/xiaomeng/en_data/allfile.filelist.shuffed",
        }
    # tokenizer_path = "baichuan-inc/Baichuan2-13B-Base"
    tokenizer_path = "Qwen/Qwen-14B"
    model_name = os.path.basename(tokenizer_path)
    today = datetime.today()
    formatted_date = today.strftime("%m%d")
    if bucket:
        save_dir = f"gs://jax_llm_data_us-east5/xiaomeng/{LANG}_data_{model_name}_{formatted_date}"
    else:
        raise ValueError(f'Now version only support bucket is True')
    if rank == 0:
        print(f'save_dir: {save_dir}')
    processor = DataProcessor(
        data_pathfiles[LANG],
        save_dir,
        data_type=LANG,
        max_seq_len=max_seq_len,
        ratio=ratio,
    )
    processor.rank = rank
    processor.per_file_line_num = 10000
    processor.epoches = 1
    every_rank_nums = math.ceil(len(processor.books_pathlist) / WORKERS)
    start = int(rank * every_rank_nums)
    end = int((rank + 1) * every_rank_nums)
    print(f"Rank: {rank} start: {start} end: {end}")
    processor.run(start, end, rank)
    return rank


if __name__ == "__main__":
    random.seed(42)
    set_start_method("spawn")  # tpu-vm
    num_processes = multiprocessing.cpu_count()
    print(f"num_processes: {num_processes}")

    WORKERS = int(sys.argv[1])
    host_num = int(sys.argv[2])
    data_dtype = sys.argv[3]


    # hostname = socket.gethostname()
    # host_id = hostname
    # if isinstance(host_id, str) and len(host_id) > 5:
    #     host_id = host_id.rsplit("-", maxsplit=1)[-1]
    # host_id = int(host_id)
    host_id = 0
    host_num = int(host_num)


    print(f'data_dtype: {data_dtype}')

    mid_host = 4

    # if data_dtype == 'en':
    #     if host_id < mid_host:
    #         exit(0)
    #     else:
    #         host_id -= mid_host
    # else:
    #     if host_id >= mid_host:
    #         exit(0)
    #     else:
    #         pass
        
    workers_perhost = WORKERS // host_num

    worker_start = host_id * workers_perhost
    worker_end = (host_id + 1) * workers_perhost

    print(f'WORKERS: {WORKERS} worker_start: {worker_start} worker_end: {worker_end}')
    if worker_start >= WORKERS:
        exit(0)

    ratio = 1.0
    max_seq_len = 4097
    pool = multiprocessing.Pool(processes=WORKERS)
    args = (
        [rank, LANG, WORKERS, max_seq_len, ratio]
        for rank in range(worker_start, worker_end, 1)
        for LANG in [data_dtype]
    )
    results = pool.map(process_book_wrapper, args)  # 包含每个进程的返回值
    pool.close()
    pool.join()

# Usage:
'''
TPU_NAME=llm-jax-mqy-v5p-16-71-paxml; ZONE=us-east5-a; WORKERS=20; HOST_NUM=1; DATA_TYPE='zh'
TPU_NAME=llm-jax-v4-256-0; ZONE=us-central2-b; WORKERS=200; HOST_NUM=5; DATA_TYPE='zh'
TPU_NAME=llm-jax-v3-128-10; ZONE=us-east1-d; WORKERS=180; HOST_NUM=4; DATA_TYPE='en'
ZONE=us-east5-a; WORKERS=10; HOST_NUM=1; DATA_TYPE='en'

TPU_NAME=llm-jax-v5p-64-10	; ZONE=europe-west4-b; WORKERS=100; HOST_NUM=1; DATA_TYPE='en'
TPU_NAME=llm-jax-v5p-64-10; ZONE=us-east5-a; WORKERS=100; HOST_NUM=1; DATA_TYPE='en'

gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="/home/lishengping/miniconda3/bin/pip install tiktoken" --project=ntpu-413714
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="sudo rm -r /home/lishengping/tokenizer; gsutil cp -r gs://llm_base_models_us-east5/qwen/tokenizer /home/lishengping/" --project=ntpu-413714
SCRIPT=/Users/lishengping/codes/jax_projects/maxtext/scripts/novel_instruct.py
gcloud compute tpus tpu-vm scp $SCRIPT $TPU_NAME:/home/lishengping/processed.py  --zone=$ZONE  --worker=all  --project=ntpu-413714

TPU_NAME=llm-jax-v3-128-10; ZONE=us-east1-d; WORKERS=30; HOST_NUM=1; DATA_TYPE='en'
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=2 --command="killall processed.py;/home/lishengping/miniconda3/bin/python processed.py $WORKERS $HOST_NUM $DATA_TYPE| tee $DATA_TYPE_processed.log"

gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="sudo rm -r /home/lishengping/tokenizer; gsutil cp -r gs://llm_base_models_us-east5/qwen/tokenizer /home/lishengping/" --project=ntpu-413714
'''