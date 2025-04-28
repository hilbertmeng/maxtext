import os
import re
from datetime import datetime
import time
import sys
from google.cloud import storage
import tensorflow as tf
from tensorboardX import writer, SummaryWriter
import numpy as np
import random

np.random.seed(42)
random.seed(42)


def extract_loss(pathes, name, scale=4):
    start_time = time.time()
    pathes.sort()
    losses = []
    steps = []
    total = []
    tags = set()
    global_step = 0
    for path in pathes:
        summaries = tf.compat.v1.train.summary_iterator(path)
        for step, e in enumerate(summaries):
            # if step == 1000000: break
            global_step += 1
            for v in e.summary.value:
                if v.tensor.dtype == 7:
                    y = v.tensor.string_val[0].decode('utf-8')
                    if v.tag.count('/') > 1:
                        continue
                    y = y.split('/')[0]
                elif v.tag == name:
                    # print(v)
                    y = v.simple_value * scale
                else:
                    y = v.simple_value
                total.append([v.tag, y, e.step])
                if v.tag == name:
                    loss = v.simple_value
                    losses.append(loss * scale)
                    steps.append(e.step)
                tags.add(v.tag)
            if global_step % 50000 == 0:
                print(f'Reading: {global_step} take: {time.time()-start_time:.3f}s')
    return steps, losses, tags, total


def extract_pathes(bucket_name, directory_path):
    client = storage.Client()
    pathes = []
    for blob in client.list_blobs(bucket_name, prefix=directory_path):
        abs_path = os.path.join(f'gs://{bucket_name}', blob.name)
        pathes.append(abs_path)
    pathes.sort()
    return pathes

# path = 'llm_base_models_us-east5/v5p_256/7B/MuddLlama2Medium_wdMask_OptPax0304/tensorboard/MuddLlama2Medium_wdMask_OptPax0304/events.out.tfevents.1741054359.t1v-n-2d5f58bd-w-0'

bucket_name = 'llm_base_models_us-east5'
directory_path = 'v5p_256/7B/MuddLlama2Medium_wdMask_OptPax0304/tensorboard/MuddLlama2Medium_wdMask_OptPax0304/'
tanh_pathes = extract_pathes(bucket_name, directory_path)

# path = 'gs://jax_llm_data_europe-west4/dcformer_compare_experiments/muddformer_logs/vit/tensorboards/vit_S16_mudd_dense1.0Init_tanh_muddDrop0.1_0107_2/events.out.tfevents.1736236580.t1v-n-24cbcd57-w-0'
name = 'learning/loss'
tanh_steps, tanh_losses, tanh_tags, tanh_total = extract_loss(tanh_pathes, name, scale=1)


tensorboard_dir = './'
print(f'tensorboard_dir: {tensorboard_dir}')
tb_writer = writer.SummaryWriter(tensorboard_dir)


start_time = time.time()
tags = set()
for step, t in enumerate(tanh_total):
    # time.sleep(sleep_time)
    if step % 10000 == 0:
        print(f'step: {step} take: {time.time() - start_time:.3f}s')
    step = t[-1]
    if isinstance(t[1], str):
        tb_writer.add_text(*t[:2])
    else:
        tags.add(t[0])
        if step % 5 != 0 and 'eval' not in t[0]: 
            continue
        if 'layer' in t[0]:
            if step % 25 == 0:
                tb_writer.add_scalar(*t)
        else:
            tb_writer.add_scalar(*t)
tb_writer.close()