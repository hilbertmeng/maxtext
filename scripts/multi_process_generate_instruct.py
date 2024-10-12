from datasets import load_from_disk
# from datasets import load_dataset
import random
import tensorflow as tf
import re
from transformers import AutoTokenizer
import time
import multiprocessing as mp
import numpy as np



def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def write_to_tfrecord(writer, input_ids, labels):
    feature = {
        "input_ids": _int64_feature(input_ids),
        "labels": _int64_feature(labels),

    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())



def process_chunk(chunk, chunk_id, block_size, mask_role, mask_user, mask_value, max_seq_length, splice, start_id, end_id, save_dir):
    merged_input_ids = []
    role_set = set()

    start = time.time()
    for n, example in enumerate(chunk):
        if n % 100 == 0:
            print(f'processed-{chunk_id}: {n}/{len(chunk)} take: {time.time() - start:.3f}')
        input_ids, masks = [], []
        conversations = example['conversations']
        if len(conversations) == 1:
            continue
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
            split = '\n\n' if idx < len(conversations) - 1 else ''
            content = re.subn('<\|eot\|>', '\n\n', conversation['content'])[0]
            content = content.strip() + split
            role_ids = tokenizer.encode(role_)
            content_ids = tokenizer.encode(content)
            role_content_ids = role_ids + content_ids

            if mask_role:
                mask = len(role_ids) * [mask_value] + content_ids
            elif mask_user:
                if role == 'user':
                    mask = len(role_content_ids) * [mask_value]
                else:
                    mask = len(role_ids) * [mask_value] + content_ids
            else:
                raise ValueError('Unknown mask type.')

            masks.extend(mask)
            input_ids.extend(role_content_ids)

        masks.append(end_id)
        input_ids.append(end_id)
        assert len(input_ids) == len(masks), print(len(input_ids), len(masks))

        input_ids = input_ids[:max_seq_length]
        masks = masks[:max_seq_length]

        merged_input_ids.append([input_ids, masks])

        if (n + 1) % block_size == 0 or n == len(chunk) - 1:
            e = (n + 1) // block_size
            save_path = f'{save_dir}/instruct_role_play.tfrecord.{version}.R{chunk_id}.F{e}'
            writer = tf.io.TFRecordWriter(save_path)
            # 排序会导致一条数据中很多条非常短的拼在一起，虽然有利于训练效率，但是对训练效果不是很友好
 #           sorted_merged_input_ids = sorted(merged_input_ids, key=lambda x: len(x[0]))
            sorted_merged_input_ids = merged_input_ids
            random.shuffle(sorted_merged_input_ids)
            current_input_ids, current_masks = [], []
            for input_ids, masks in sorted_merged_input_ids:
                assert len(input_ids) == len(masks), print(len(input_ids), len(masks))
                if splice:
                    if len(current_input_ids) + len(input_ids) > max_seq_length:
                        write_to_tfrecord(writer, [start_id] + current_input_ids, [mask_value] + current_masks)
                        current_input_ids = input_ids
                        current_masks = masks
                    else:
                        current_input_ids += input_ids
                        current_masks += masks
                else:
                    write_to_tfrecord(writer, input_ids, masks)

            merged_input_ids = []
            writer.close()


def parallel_process_dataset(dataset, num_processes, block_size, mask_role, mask_user, mask_value, max_seq_length, splice, start_id, end_id, save_dir):
    dataset_chunks = np.array_split(dataset, num_processes)

    processes = []
    for i, chunk in enumerate(dataset_chunks):
        p = mp.Process(target=process_chunk, args=(chunk, i, block_size, mask_role, mask_user, mask_value, max_seq_length, splice, start_id, end_id, save_dir))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

if __name__ == '__main__':
    start = time.time()

    tokenizer_path = '/nas2/lishengping/qwen_xm3p5_tokenizer'
    tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, use_fast=True, trust_remote_code=True
            )
    version = 'new1'
 #   version = 'v3'

    path = f'/nas2/libianbian/sft_shuffled_dataset_{version}/'

    dataset = load_from_disk(path)
    dataset = dataset.shuffle(seed=42)

    mask_role = False
    mask_user = True
    max_seq_length = 4096
    splice = True
    start_id = 151646
    end_id = 151643
    block_size = 80000  # 每隔8万条数据存储一次文件
    save_dir = 'role_instruct_data_multi_process2'
    mask_value = -100  # 不计算mask的部分

    # Number of parallel processes
    num_processes = mp.cpu_count()  # You can limit this to fewer processes if needed
    print(f'num_processes: {num_processes}')
    num_processes = 5
    # Assuming `dataset` is already loaded
    parallel_process_dataset(
        dataset=dataset,
        num_processes=num_processes,
        block_size=block_size,
        mask_role=mask_role,
        mask_user=mask_user,
        mask_value=mask_value,
        max_seq_length=max_seq_length,
        splice=splice,
        start_id=start_id,
        end_id=end_id,
        save_dir=save_dir
    )

    print(f'Total processing time: {time.time() - start:.3f}s')