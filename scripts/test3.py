import os
import sys
import json
import random
import time
import argparse
from datetime import datetime

import torch
import numpy as np
from transformers import AutoTokenizer
from flask import Flask, request, jsonify

from configuration_dcformer import DCFormerConfig
from modeling_dcformer import DCFormer
from modeling_dcformer import DCFormer, match_weight
import math

app = Flask(__name__)

def set_random_seed(seed):
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)


# 加载你的模型
def load_model(config: dict, checkpoint_path: str, max_batch_size: int, max_seq_length: int):
    config = DCFormerConfig(**config)
    dcformer = DCFormer(config)
    _ = dcformer.to(device=device,dtype=torch.float16)
    print('match weight')
    w = torch.load(checkpoint_path,map_location='cpu')
    dcformer = match_weight(dcformer, w)
    print('setup cache')
    _ = dcformer.to(device=device,dtype=torch.float16)
    with torch.device(device):
        dcformer.setup_caches(max_batch_size=max_batch_size, max_seq_length=max_seq_length, set_kv_cache=True)
    return dcformer


def _temperature(scores, temperature):
    return scores / temperature

user_penalty_words = [' \n\n\n', '\n\n\n', '<|endoftext|>']
user_penalty_word_ids = [14731, 1406, 151643]
user_penalty_scale = 1.5
def _penalty(input_ids:torch.LongTensor, scores: torch.FloatTensor, penalty: float) -> torch.FloatTensor:
    input_ids = input_ids.long()[..., -50:]
    score = torch.gather(scores, 1, input_ids)
    score = torch.where(score < 0, score * penalty, score / penalty)
    scores_processed = scores.scatter(1, input_ids, score)
    scores_processed[..., user_penalty_word_ids] = scores_processed[..., user_penalty_word_ids] / user_penalty_scale
    return scores_processed


def _top_p(input_ids: torch.LongTensor, scores: torch.FloatTensor, top_p: float) -> torch.FloatTensor:
    min_tokens_to_keep = 1
    filter_value = -float("Inf")
    sorted_logits, sorted_indices = torch.sort(scores, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

    # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
    # Keep at least min_tokens_to_keep
    sorted_indices_to_remove[..., -min_tokens_to_keep :] = 0

    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    scores_processed = scores.masked_fill(indices_to_remove, filter_value)
    return scores_processed


def _top_k(input_ids: torch.LongTensor, scores: torch.FloatTensor, top_k: int) -> torch.FloatTensor:
    filter_value = -float("Inf")
    top_k = min(top_k, scores.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
    scores_processed = scores.masked_fill(indices_to_remove, filter_value)
    return scores_processed


def _sample(generated_ids, last_logit, **kwargs):
    temperature = kwargs.get('temperature', 1.0)
    penalty = kwargs.get('penalty', 1.05)
    top_p = kwargs.get('top_p', 0.9)
    top_k = kwargs.get('top_k', 30)
    last_logit = _temperature(last_logit, temperature=temperature)
    last_logit = _penalty(generated_ids, last_logit, penalty=penalty)
    last_logit = _top_p(generated_ids, last_logit, top_p=top_p)
    last_logit = _top_k(generated_ids, last_logit, top_k=top_k)
    prob = torch.nn.functional.softmax(last_logit, dim=-1)
    next_token = torch.multinomial(prob, num_samples=1)
    token_p = prob[..., next_token.view(-1)]
    return next_token, token_p


def decode_one_token(model, cur_token, input_pos, generated_ids, **kwargs):
    logits = model(cur_token, input_pos=input_pos, return_tensor=True)
    return logits


def _generate(decode_one_token, model, input_token, input_pos, generated_ids, **kwargs):
    sample = kwargs.get('sample', False)
    logits = decode_one_token(model, input_token, input_pos, generated_ids, **kwargs)
    last_logit = logits[:, -1]
    if sample:
        next_token, token_p = _sample(generated_ids, last_logit, **kwargs)
    else:
        scores = torch.softmax(last_logit, dim=-1)
        next_token = torch.argmax(last_logit, dim=-1)
        token_p = scores[..., next_token]
        next_token = next_token[..., None]
    return next_token, token_p

def prefill_decode(self, input_ids, input_pos):
    logits = self.forward(input_ids, input_pos=input_pos, return_tensor=True)
    return logits

def generate(self, input_ids, decode_one_token, **kwargs):
    debug_ids = []
    num_tokens_to_generate = kwargs.get('num_tokens_to_generate', 50)
    stop_id = kwargs.get('stop_id', 151643)
    batch_size, seq_length = input_ids.shape
    input_pos = torch.arange(seq_length, device=self.device)
    # 151646:<|extra_0|>作为惩罚id，不能用0，0表示!
    generated_ids = torch.tensor([151646], device=self.device, dtype=torch.int).repeat(
                                                    self.max_batch_size, seq_length + num_tokens_to_generate)
    generated_ids[:, :seq_length] = input_ids.to(self.device).to(torch.int)

    logits = self.forward(input_ids, input_pos=input_pos, return_tensor=True)
    last_logit = logits[:, -1]

    sample = kwargs.get('sample', False)
    if sample:
        _next_token, token_p = _sample(generated_ids, last_logit, **kwargs)
    else:
        scores = torch.softmax(last_logit, dim=-1)
        _next_token = torch.argmax(last_logit, dim=-1)
        token_p = scores[..., _next_token]
        _next_token = _next_token[..., None]
    debug_ids.append([_next_token.view(-1).item(), token_p.view(-1).item()])

    next_token = torch.zeros(self.max_batch_size, 1, device=self.device, dtype=torch.int)
    next_token[ :batch_size] = _next_token
    generated_ids[:, seq_length] = _next_token
    input_pos = torch.tensor([seq_length], device=self.device)
    for _ in range(1, num_tokens_to_generate):
        next_token, token_p = _generate(decode_one_token, self, next_token.clone(), input_pos, generated_ids, **kwargs)
        debug_ids.append([next_token.view(-1).item(), token_p.view(-1).item()])
        if next_token.item() == stop_id : break
        generated_ids[:, input_pos+1] = next_token.int()[:batch_size]
        input_pos += 1
    generated_ids = generated_ids[:, :input_pos + 1]
    print(f'generated ids: {debug_ids}')
    return generated_ids


def write_to_file(response):
    writer = open('xiaomeng_server_output.log', 'a+')
    response['time'] = str(datetime.now())
    write_str = json.dumps(response, ensure_ascii=False)
    writer.write(f'{write_str}\n')
    writer.close()
    return write_str


@app.route('/infer', methods=['POST'])
def predict():
    start = time.time()
    data = request.json
    seed = data.get('seed', 1234)
    set_random_seed(seed)
    print(f'data:\n{data}')
    prompt = data['prompt']
    # temperature = data.get('temperature', 1.0)
    # penalty = data.get('penalty', 1.0)
    # top_p = data.get('top_p', 0.95)
    # top_k = data.get('top_k', 30)
    # num_tokens_to_generate = data.get('num_tokens_to_generate', 10)
    complie = data.get('complie', False)
    start_id = data.get('start_id', 151646) # extra_0
    forward_func = compiled_decode_one_token if complie else decode_one_token

    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    input_ids = torch.cat([torch.tensor([[start_id]]), input_ids], dim=-1)
    print(f'input_ids: {input_ids.shape}')
    with torch.no_grad():
        generated_ids = generate(dcformer, input_ids.to(device), forward_func, **data)
        text = tokenizer.decode(generated_ids[0])
    take = f'{time.time() - start:.3f}'
    print(f'take: {take}s')
    response = {'response': {"output": text, "input": data, "interval": take}}
    print(f'response:\n{response}')
    response = write_to_file(response)
    return response


if __name__ == '__main__':
    # python server.py --checkpoint_path /home/lishengping/mengqy/data/PileDCSlimLlama7B4Kx4x256x1v5p_checkpoint_00440000.torch.bin
    # curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"seed\": 42, \"prompt\": \"怎么做西红柿鸡蛋?\n\", \"temperature\": 0.6, \"penalty\": 1.1, \"sample\": true, \"num_tokens_to_generate\": 100, \"top_p\": 0.85, \"top_k\": 20, \"complie\": true}"
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Directory to restore model')
    parser.add_argument('--device_id', type=int, default=0, help='cuda device index or id')
    parser.add_argument('--port', type=int, default=5000, help='server api port')

    args = parser.parse_args()

    device_id = args.device_id
    checkpoint_path = args.checkpoint_path

    device = torch.device(f'cuda:{device_id}')
    max_batch_size = 1
    max_seq_length = 4096 * 1
    model_config = {
                    'vocab_size': 152064,
                    "norm_eps": 1e-06,
                    "block_size": 4096,
                    'dim':4096,
                    'n_head':32,
                    'n_layer': 48,
                    'window_type':'LGLL',
                    'mgate': True,
                    'mgate_dim': 44,
                    'intermediate_size': 5632,
                    'use_qk_norm': True,
                    'rope_base': 500000.0 # 4k model 100000, but 32k model is 500000
                }
    # model init
    dcformer = load_model(model_config, checkpoint_path, max_batch_size, max_seq_length)
    checkpoint_dir = os.path.dirname(checkpoint_path)
    tokenizer_path = os.path.join(checkpoint_dir, 'tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, trust_remote_code=True)
    compiled_decode_one_token = torch.compile(decode_one_token, mode="reduce-overhead", fullgraph=True)
    # compiled_prefill_func = torch.compile(prefill_decode, mode="reduce-overhead", fullgraph=True)

    app.run(host='0.0.0.0', port=args.port)