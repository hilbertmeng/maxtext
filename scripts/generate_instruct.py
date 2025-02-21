from datasets import load_from_disk
# from datasets import load_dataset
import random
import tensorflow as tf
import re
from transformers import AutoTokenizer
import time


tokenizer_path = '/nas2/lishengping/qwen_xm3p5_tokenizer'
tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, use_fast=True, trust_remote_code=True
        )

path = '/nas2/libianbian/sft_shuffled_dataset_v3/'

dataset = load_from_disk(path)
dataset = dataset.shuffle(seed=42)

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def write_to_tfrecord(writer, input_ids, labels):
    feature = {
        "input_ids": _int64_feature(input_ids),
        "labels": _int64_feature(labels),

    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())


start = time.time()

mask_role = False
mask_user = True
max_seq_length = 4096
splice = True
start_id = 151646
end_id = 151643
block_size = 20000
save_path = 'instruct_role_play.train.tfrecord.v3'
train_writer = tf.io.TFRecordWriter(save_path)
save_path = 'instruct_role_play.valid.tfrecord.v3'
eval_writer = tf.io.TFRecordWriter(save_path)
pad_id = -100  # 不计算mask的部分

# 151643: endoftext
merged_input_ids, merged_masks = [], []
role_set = set()
for n, example in enumerate(dataset):
   
    print(f'Processing: {n}/{dataset.shape[0]} take: {time.time() - start:.3f}s')
    input_ids, masks = [], []
    conversations = example['conversations']
    if len(conversations) == 1: continue
    for idx, conversation in enumerate(conversations):
        role = conversation['role'].strip()
        if not role: break
        role_set.add(role)
        if role in ['human', 'user']:
            role = 'user'
        elif role in ['gpt', 'bot']:
            role = 'assistant'
        elif role == '':
            print(f'error role: {role}')

        role_ = role + ':\n'
        if idx < len(conversations) - 1:
            split = '\n\n'
        else:
            split = ''
        content = re.subn('<\|eot\|>', '\n\n', conversation['content'])[0]
        content = content.strip() + split
        role_ids = tokenizer.encode(role_)
        content_ids = tokenizer.encode(content)
        role_content_ids = role_ids + content_ids
        if mask_role:
            mask = len(role_ids) * [pad_id] + content_ids
        elif mask_user:
            if role == 'user':
                mask = len(role_content_ids) * [pad_id]
            else:
                mask = len(role_ids) * [pad_id] + content_ids
        else:
            ValueError('Unknow mask type.')
        masks.extend(mask)
        input_ids.extend(role_content_ids)
    masks.append(end_id)
    input_ids.append(end_id)
    assert len(input_ids) == len(masks), print(len(input_ids), len(masks))

    input_ids = input_ids[: max_seq_length]
    masks = masks[: max_seq_length]
  
    merged_input_ids.append([input_ids, masks])
 #   if n > 10000: break
    if (n + 1) % block_size == 0 or n == dataset.shape[0] - 1:
        fn = n // block_size
        save_path = f'role_instruct_data/instruct_role_play.tfrecord.v3.F{fn}'
        writer = tf.io.TFRecordWriter(save_path)
        print(f'N: {n} Start write to tfrecord. length: {len(merged_input_ids)}')
        # 排序会导致一条数据中很多条非常短的拼在一起，虽然有利于训练效率，但是对训练效果不是很友好
        # sorted_merged_input_ids = sorted(merged_input_ids, key=lambda x: len(x[0])) 
        sorted_merged_input_ids = merged_input_ids
        random.shuffle(sorted_merged_input_ids)

        current_input_ids, current_masks = [], []
        for (input_ids, masks) in sorted_merged_input_ids:
            assert len(input_ids) == len(masks), print(len(input_ids), len(masks))
            if splice:
                if len(current_input_ids) + len(input_ids) > max_seq_length:
                    write_to_tfrecord(writer, [start_id] + current_input_ids, [pad_id] + current_masks) # 这里不进行pad，在tfrecord读取的时候会自动pad
                    current_input_ids = input_ids
                    current_masks = masks
                else:
                    current_input_ids += input_ids
                    current_masks += masks
            else:
                write_to_tfrecord(writer, input_ids, masks) # 这里不进行pad，在tfrecord读取的时候会自动pad
        merged_input_ids = []

        writer.close()
