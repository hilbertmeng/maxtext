# Requirement:
#   pip install "openai<1.0"
# Usage:
#   python openai_api.py
# Visit http://localhost:8000/docs for documents.
import os
import base64
import math
import copy
import json
import time
from argparse import ArgumentParser
from contextlib import asynccontextmanager
from pprint import pprint
from typing import Dict, List, Literal, Optional, Union
import random

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

from modeling_dcformer import DCFormer


DEFAULT_STOP = ['<|extra_0|>', '<|endoftext|>']

def set_random_seed(seed):
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)


class BasicAuthMiddleware(BaseHTTPMiddleware):

    def __init__(self, app, username: str, password: str):
        super().__init__(app)
        self.required_credentials = base64.b64encode(
            f'{username}:{password}'.encode()).decode()

    async def dispatch(self, request: Request, call_next):
        authorization: str = request.headers.get('Authorization')
        if authorization:
            try:
                schema, credentials = authorization.split()
                if credentials == self.required_credentials:
                    return await call_next(request)
            except ValueError:
                pass

        headers = {'WWW-Authenticate': 'Basic'}
        return Response(status_code=401, headers=headers)


def _gc(forced: bool = False):
    global args
    if args.disable_gc and not forced:
        return

    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@asynccontextmanager
async def lifespan(app: FastAPI):  # collects GPU memory
    yield
    _gc(forced=True)


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


class ModelCard(BaseModel):
    id: str
    object: str = 'model'
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = 'owner'
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


class ModelList(BaseModel):
    object: str = 'list'
    data: List[ModelCard] = []


class ChatMessage(BaseModel):
    role: Literal['user', 'assistant', 'system', 'function']
    content: Optional[str]
    function_call: Optional[Dict] = None


class DeltaMessage(BaseModel):
    role: Optional[Literal['user', 'assistant', 'system']] = None
    content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    functions: Optional[List[Dict]] = None
    temperature: Optional[float] = 0.9
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 30
    max_tokens: Optional[int] = 50
    max_completion_tokens: Optional[int] = 50
    stream: Optional[bool] = False
    stop: Optional[List[str]] = ['<|extra_0|>', '<|endoftext|>']
    presence_penalty: Optional[float] = 1.1
    seed: Optional[int] = 42
    extra_configs: Optional[Dict] = None
    request_format: Optional[str] = 'other'  # chat or other


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Union[ChatMessage]
    finish_reason: Literal['stop', 'length', 'function_call']


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal['stop', 'length']]


class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal['chat.completion', 'chat.completion.chunk']
    choices: List[Union[ChatCompletionResponseChoice,
                        ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))


@app.get('/v1/models', response_model=ModelList)
async def list_models():
    global model_args
    model_card = ModelCard(id='xm-7b-32k-chat')
    return ModelList(data=[model_card])


# To work around that unpleasant leading-\n tokenization issue!
def add_extra_stop_words(stop_words):
    if stop_words:
        _stop_words = []
        _stop_words.extend(stop_words)
        for x in stop_words:
            s = x.lstrip('\n')
            if s and (s not in _stop_words):
                _stop_words.append(s)
        return _stop_words
    else:
        stop_words = []
    return stop_words


def trim_stop_words(response, stop_words):
    if stop_words:
        for stop in stop_words:
            idx = response.find(stop)
            if idx != -1:
                response = response[:idx]
    return response


TOOL_DESC = (
    '{name_for_model}: Call this tool to interact with the {name_for_human} API.'
    ' What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters}'
)

REACT_INSTRUCTION = """Answer the following questions as best you can. You have access to the following APIs:

{tools_text}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tools_name_text}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!"""

_TEXT_COMPLETION_CMD = object()


def parse_messages(messages, functions):
    if all(m.role != 'user' for m in messages):
        raise HTTPException(
            status_code=400,
            detail='Invalid request: Expecting at least one user message.',
        )

    messages = copy.deepcopy(messages)
    if messages[0].role == 'system':
        system = messages.pop(0).content.lstrip('\n').rstrip()
    else:
        system = 'You are a helpful assistant.'

    if functions:
        tools_text = []
        tools_name_text = []
        for func_info in functions:
            name = func_info.get('name', '')
            name_m = func_info.get('name_for_model', name)
            name_h = func_info.get('name_for_human', name)
            desc = func_info.get('description', '')
            desc_m = func_info.get('description_for_model', desc)
            tool = TOOL_DESC.format(
                name_for_model=name_m,
                name_for_human=name_h,
                # Hint: You can add the following format requirements in description:
                #   "Format the arguments as a JSON object."
                #   "Enclose the code within triple backticks (`) at the beginning and end of the code."
                description_for_model=desc_m,
                parameters=json.dumps(func_info['parameters'],
                                      ensure_ascii=False),
            )
            tools_text.append(tool)
            tools_name_text.append(name_m)
        tools_text = '\n\n'.join(tools_text)
        tools_name_text = ', '.join(tools_name_text)
        instruction = (REACT_INSTRUCTION.format(
            tools_text=tools_text,
            tools_name_text=tools_name_text,
        ).lstrip('\n').rstrip())
    else:
        instruction = ''

    messages_with_fncall = messages
    messages = []
    for m_idx, m in enumerate(messages_with_fncall):
        role, content, func_call = m.role, m.content, m.function_call
        content = content or ''
        # content = content.lstrip('\n').rstrip()
        if role == 'function':
            if (len(messages) == 0) or (messages[-1].role != 'assistant'):
                raise HTTPException(
                    status_code=400,
                    detail=
                    'Invalid request: Expecting role assistant before role function.',
                )
            messages[-1].content += f'\nObservation: {content}'
            if m_idx == len(messages_with_fncall) - 1:
                # add a prefix for text completion
                messages[-1].content += '\nThought:'
        elif role == 'assistant':
            if len(messages) == 0:
                raise HTTPException(
                    status_code=400,
                    detail=
                    'Invalid request: Expecting role user before role assistant.',
                )
            if func_call is None:
                if functions:
                    content = f'Thought: I now know the final answer.\nFinal Answer: {content}'
            else:
                f_name, f_args = func_call['name'], func_call['arguments']
                if not content.startswith('Thought:'):
                    content = f'Thought: {content}'
                content = f'{content}\nAction: {f_name}\nAction Input: {f_args}'
            if messages[-1].role == 'user':
                messages.append(
                    ChatMessage(role='assistant',
                                content=content.lstrip('\n').rstrip()))
            else:
                messages[-1].content += '\n' + content
        elif role == 'user':
            messages.append(
                ChatMessage(role='user',
                            content=content.lstrip('\n').rstrip()))
        else:
            raise HTTPException(
                status_code=400,
                detail=f'Invalid request: Incorrect role {role}.')

    query = _TEXT_COMPLETION_CMD
    if messages[-1].role == 'user':
        query = messages[-1].content
        messages = messages[:-1]

    if len(messages) % 2 != 0:
        raise HTTPException(status_code=400, detail='Invalid request')

    history = []  # [(Q1, A1), (Q2, A2), ..., (Q_last_turn, A_last_turn)]
    for i in range(0, len(messages), 2):
        if messages[i].role == 'user' and messages[i + 1].role == 'assistant':
            usr_msg = messages[i].content.lstrip('\n').rstrip()
            bot_msg = messages[i + 1].content.lstrip('\n').rstrip()
            if instruction and (i == len(messages) - 2):
                usr_msg = f'{instruction}\n\nQuestion: {usr_msg}'
                instruction = ''
            history.append([usr_msg, bot_msg])
        else:
            raise HTTPException(
                status_code=400,
                detail=
                'Invalid request: Expecting exactly one user (or function) role before every assistant role.',
            )
    if instruction:
        assert query is not _TEXT_COMPLETION_CMD
        query = f'{instruction}\n\nQuestion: {query}'
    return query, history, system


def parse_response(response):
    func_name, func_args = '', ''
    i = response.find('\nAction:')
    j = response.find('\nAction Input:')
    k = response.find('\nObservation:')
    if 0 <= i < j:  # If the text has `Action` and `Action input`,
        if k < j:  # but does not contain `Observation`,
            # then it is likely that `Observation` is omitted by the LLM,
            # because the output text may have discarded the stop word.
            response = response.rstrip() + '\nObservation:'  # Add it back.
        k = response.find('\nObservation:')
        func_name = response[i + len('\nAction:'):j].strip()
        func_args = response[j + len('\nAction Input:'):k].strip()

    if func_name:
        response = response[:i]
        t = response.find('Thought: ')
        if t >= 0:
            response = response[t + len('Thought: '):]
        response = response.strip()
        choice_data = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(
                role='assistant',
                content=response,
                function_call={
                    'name': func_name,
                    'arguments': func_args
                },
            ),
            finish_reason='function_call',
        )
        return choice_data

    z = response.rfind('\nFinal Answer: ')
    if z >= 0:
        response = response[z + len('\nFinal Answer: '):]
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role='assistant', content=response),
        finish_reason='stop',
    )
    return choice_data


def _temperature(scores, temperature):
    return scores / temperature

user_penalty_words = [' \n\n\n', '\n\n\n', '<|endoftext|>']
user_penalty_word_ids = [14731, 1406, 151643]
user_penalty_scale = 1.5
def _penalty(input_ids:torch.LongTensor, scores: torch.FloatTensor, penalty: float) -> torch.FloatTensor:
    input_ids = input_ids.long()
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
    penalty = kwargs.get('repetition_penalty', 1.0)
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
    with torch.no_grad():
        logits = model(cur_token, input_pos=input_pos, return_tensor=True)
    return logits


def _generate(decode_one_token, model, input_token, input_pos, generated_ids, **kwargs):
    sample = kwargs.get('sample', True)
    logits = decode_one_token(model, input_token, input_pos, generated_ids, **kwargs)
    last_logit = logits[:, -1]
    if sample:
        next_token, token_p = _sample(generated_ids[:, :input_pos], last_logit, **kwargs)
    else:
        scores = torch.softmax(last_logit, dim=-1)
        next_token = torch.argmax(last_logit, dim=-1)
        token_p = scores[..., next_token]
        next_token = next_token[..., None]
    return next_token, token_p


def prefill_decode(self, input_ids, input_pos, return_tensor):
    with torch.no_grad():
        logits = self.forward(input_ids, input_pos=input_pos, return_tensor=return_tensor)
    return logits


placeholder_ids = torch.zeros([1, 256], device='cuda:0').to(torch.int)
placeholder_pos = torch.arange(256, device='cuda:0').to(torch.int)

def common_generate(self, input_ids, **kwargs):
    print(f'common_generate kwargs:\n{kwargs}')
    debug_ids = []
    num_tokens_to_generate = kwargs['max_completion_tokens']
    stop_ids = kwargs.get('stop_ids', [151643, 151646])
    batch_size, seq_length = input_ids.shape
    # 151646:<|extra_0|>作为惩罚id，不能用0，0表示"!"
    generated_ids = torch.tensor([151646], device=self.device, dtype=torch.int).repeat(
                                                    self.max_batch_size, seq_length + num_tokens_to_generate)
    generated_ids[:, :seq_length] = input_ids.to(self.device).to(torch.int)
    stream = kwargs['stream']
    print(f'stream: {stream}')
    prefill_compile = kwargs.get('prefill_compile', False)
    decode_one_token_compile = kwargs.get('decode_one_token_compile', False)
    if kwargs.get('prefill_pad'):
        prefill_length = input_ids.shape[1]
        prefill_chunk_size = 256
        chunks = math.ceil(prefill_length / prefill_chunk_size)
        prefill_pad_length = prefill_chunk_size * chunks - prefill_length
        input_ids = torch.nn.functional.pad(input_ids, pad=[0, prefill_pad_length])
        input_pos = torch.arange(seq_length + prefill_pad_length, device=self.device)
        print(f'prefill_length: {prefill_length} prefill_chunk_size: {prefill_chunk_size} chunks: {chunks} prefill_pad_length: {prefill_pad_length}')
        for c in range(chunks):
            placeholder_ids[:] = input_ids[:, c * prefill_chunk_size: (c + 1) * prefill_chunk_size]
            print(f'placeholder_ids: {placeholder_ids.shape}')
            placeholder_pos[:] = input_pos[c * prefill_chunk_size: (c + 1) * prefill_chunk_size]
            with torch.no_grad():  # 这个必须加上，否则爆显存。在predict函数加不好使
                if not prefill_compile:
                    logits = self.forward(placeholder_ids, input_pos=placeholder_pos, return_tensor=True)
                else:
                    logits = compiled_prefill_func(self, placeholder_ids, input_pos=placeholder_pos, return_tensor=True)
        last_logit = logits[:, -prefill_pad_length - 1]
    else:
        input_pos = torch.arange(seq_length, device=self.device)
        with torch.no_grad():
            logits = self.forward(input_ids, input_pos=input_pos, return_tensor=True)
        last_logit = logits[:, -1]
    sample = kwargs.get('sample', True)
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
    if stream:
        yield generated_ids[:, seq_length], generated_ids[:, seq_length]
    input_pos = torch.tensor([seq_length], device=self.device)
    for inx in range(1, num_tokens_to_generate):
        if decode_one_token_compile:
            next_token, token_p = _generate(compiled_decode_one_token, self, next_token.clone(), input_pos, generated_ids, **kwargs)
        else:
            next_token, token_p = _generate(decode_one_token, self, next_token.clone(), input_pos, generated_ids, **kwargs)
        debug_ids.append([next_token.view(-1).item(), token_p.view(-1).item()])
        if next_token.item() in stop_ids : break
        generated_ids[:, input_pos+1] = next_token.int()[:batch_size]
        input_pos += 1
        if stream:
            yield generated_ids[:, seq_length+1], generated_ids[:, input_pos]
    if not stream:
        generated_ids = generated_ids[:, :input_pos + 1]
        yield generated_ids  # return generated_ids 不行


# completion mode, not chat mode
def text_complete_last_message(history, stop_words_ids, system, **gen_kwargs):
    user = ''
    assistant = ''
    prompt = ''
    im_start = '<|extra_0|>'
    im_end = ''
    for i, (query, response) in enumerate(history):
        query = query.lstrip('\n').rstrip()
        response = response.lstrip('\n').rstrip()
        prompt += f'{im_start}{user}{query}{im_end}'
        prompt += f'{assistant}{response}{im_end}'
    input_ids = torch.tensor([tokenizer.encode(prompt)]).to(model.device)
    output = common_generate(model, input_ids, **gen_kwargs)
    output = tokenizer.decode(output, errors='ignore')
    assert output.startswith(prompt)
    output = output[len(prompt):]
    # output = trim_stop_words(output, ['<|endoftext|>']) # lsp: remove lm_end
    print(f'<completion>\n{prompt}\n<!-- *** -->\n{output}\n</completion>')
    return output


@app.post('/v1/chat/completions', response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    global model, tokenizer
    print(f'extra_configs: {request.extra_configs}')
    gen_kwargs = {}
    # extra_configs优先
    if request.extra_configs is not None:
        gen_kwargs['top_k'] = request.extra_configs.get('top_k', getattr(request, 'top_k', 30))
        gen_kwargs['top_p'] = request.extra_configs.get('top_p', getattr(request, 'top_p', 0.95))
        gen_kwargs['temperature'] = request.extra_configs.get('temperature', getattr(request, 'temperature', 1.0))
        gen_kwargs['seed'] = request.extra_configs.get('seed', getattr(request, 'seed', 0))
        gen_kwargs['repetition_penalty'] = request.extra_configs.get('repetition_penalty', getattr(request, 'presence_penalty', 1.0))
        gen_kwargs['prefill_pad'] = request.extra_configs.get('prefill_pad', getattr(request, 'prefill_pad', False))
        gen_kwargs['prefill_compile'] = request.extra_configs.get('prefill_compile', getattr(request, 'prefill_compile', False))
        gen_kwargs['decode_one_token_compile'] = request.extra_configs.get('decode_one_token_compile', getattr(request, 'decode_one_token_compile', False))
        
    else:
        gen_kwargs['top_k'] = getattr(request, 'top_k', 30)
        gen_kwargs['top_p'] = getattr(request, 'top_p', 0.95)
        gen_kwargs['temperature'] = getattr(request, 'temperature', 0.9)
        gen_kwargs['seed'] = getattr(request, 'seed', 42)
        gen_kwargs['repetition_penalty'] = getattr(request, 'presence_penalty', 1.1)
        gen_kwargs['prefill_pad'] = getattr(request, 'prefill_pad', False)
        gen_kwargs['prefill_compile'] = getattr(request, 'prefill_compile', False)
        gen_kwargs['decode_one_token_compile'] = getattr(request, 'decode_one_token_compile', False)

    if gen_kwargs['prefill_compile']:
        assert gen_kwargs['prefill_pad'] > 0, print(f'`prefill_pad` must be > 0 when `prefill_compile` is True...')

    if gen_kwargs['temperature'] < 0.01:
        gen_kwargs['top_k'] = 1  # greedy decoding

    if request.max_completion_tokens is not None:
        gen_kwargs['max_completion_tokens'] = request.max_completion_tokens
    else:
        if request.max_tokens is not None:
            gen_kwargs['max_completion_tokens'] = request.max_tokens
        else:
            gen_kwargs['max_completion_tokens'] = 50

    gen_kwargs['max_completion_tokens'] = min(gen_kwargs['max_completion_tokens'], 4096) # 最多生成4096个token
    gen_kwargs['stream'] = getattr(request, 'stream', False)
    gen_kwargs['request_format'] = getattr(request, 'request_format', 'chat')
    set_random_seed(gen_kwargs['seed'])

    stop_words = add_extra_stop_words(request.stop) # return a list
    stop_words.extend(DEFAULT_STOP) # 添加默认停止词
    stop_words_ids = [tokenizer.encode(s)[0] for s in stop_words] if stop_words else []
    gen_kwargs['stop_ids'] = stop_words_ids


    query, history, system = parse_messages(request.messages, request.functions)
    print(f'gen_kwargs00: {gen_kwargs}')
    if request.stream:
        if request.functions:
            raise HTTPException(
                status_code=400,
                detail=
                'Invalid request: Function calling is not yet implemented for stream mode.',
            )
        generate = stream_predict(query, history, request.model, stop_words_ids, system=system, **gen_kwargs)
        return EventSourceResponse(generate, media_type='text/event-stream')

    print(f'_TEXT_COMPLETION_CMD: {_TEXT_COMPLETION_CMD}')
    print(f'query: {query}')
    print(f'history: {history}')
    print(f'system: {system}')
    print(f'stop_words0: {stop_words}')
    print(f'stop_words_ids0: {stop_words_ids}')
    if query is _TEXT_COMPLETION_CMD: # 如果query不是user的时候走这
        response = text_complete_last_message(query,
                                            history,
                                            stop_words_ids=stop_words_ids,
                                            system=system,
                                            **gen_kwargs
                                            )
    else:
        response, _ = predict(query,
                        history,
                        stop_words_ids=stop_words_ids,
                        system=system,
                        **gen_kwargs
                        )
        print('<predict>')
        pprint(history, indent=2)
        print(f'{query}\n<!-- *** -->\n{response}\n</predict>')
    _gc()
    # response = trim_stop_words(response, stop_words)
    if request.functions:
        choice_data = parse_response(response)
    else:
        choice_data = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(role='assistant', content=response),
            finish_reason='stop',
        )
    return ChatCompletionResponse(model=request.model,
                                  choices=[choice_data],
                                  object='chat.completion')


def _dump_json(data: BaseModel, *args, **kwargs) -> str:
    try:
        return data.model_dump_json(*args, **kwargs)
    except AttributeError:  # pydantic<2.0.0
        return data.json(*args, **kwargs)  # noqa


def predict(
        query: str,
        history: Optional[List],
        stop_words_ids: Optional[List[List[int]]] = None,
        system: str = "You are a helpful assistant.",
        **gen_kwargs,
    ):
    """stream chat output"""
    if history is None:
        history = []
    prompt = ''
    im_start = '<|extra_0|>'
    if gen_kwargs.get('request_format') == 'chat':
        user = 'user:\n'
        assistant = 'assistant:\n'
        im_end = '\n\n'
        query = f'{user}{query}{im_end}{assistant}'
        for i, (q, response) in enumerate(history):
            q = q.lstrip('\n').rstrip()
            response = response.lstrip('\n').rstrip()
            prompt += f'{user}{q}{im_end}'
            prompt += f'{assistant}{response}{im_end}'
        prompt = prompt[: -len(im_end)]
    else:
        for i, (q, response) in enumerate(history):
            prompt = f'{q}{response}'
    prompt = im_start + prompt + query
    print(f'function predict prompt:\n{prompt}\n')
    input_ids = torch.tensor([tokenizer.encode(prompt)]).to(model.device)

    def stream_generator():
        outputs = []
        for prompt_ids, tokens in common_generate(model, input_ids, **gen_kwargs):
            prompt_id = prompt_ids[0]
            token = tokens[0]
            outputs.append(token.item())
            yield prompt_id, tokenizer.decode(outputs, skip_special_tokens=True, errors='ignore')

    def unstream_generator():
        outputs = next(common_generate(model, input_ids, **gen_kwargs))
        output = outputs[0] # batch size = 1
        output = tokenizer.decode(output, errors='ignore')
        assert output.startswith(prompt)
        output = output[len(prompt):]
        print(f'<completion>\n{prompt}\n<!-- *** -->\n{output}\n</completion>')
        return output, None

    if gen_kwargs['stream']:
        return stream_generator()
    else:
        return unstream_generator()


async def stream_predict(
    query: str,
    history: List[List[str]],
    model_id: str,
    stop_words_ids: List[str],
    system: str,
    **gen_kwargs: Dict,
):
    global model, tokenizer
    # 开头输出None
    # choice_data = ChatCompletionResponseStreamChoice(
    #     index=0, delta=DeltaMessage(role='assistant'), finish_reason=None)
    # chunk = ChatCompletionResponse(model=model_id,
    #                                choices=[choice_data],
    #                                object='chat.completion.chunk')
    # yield '{}'.format(_dump_json(chunk, exclude_unset=True))
    # delay_token_num = max([len(x) for x in stop_words]) if stop_words_ids else
    delay_token_num, current_length = 0, 0
    response_generator = predict(query,
                                    history=history,
                                    stop_words_ids=stop_words_ids,
                                    system=system,
                                    **gen_kwargs)
    for prompt_id, _new_response in response_generator:
        if len(_new_response) <= delay_token_num:
            continue
        new_response = _new_response[:-delay_token_num] if delay_token_num else _new_response

        if len(new_response) == current_length:
            continue

        new_text = new_response[current_length:]
        current_length = len(new_response)

        choice_data = ChatCompletionResponseStreamChoice(
            index=0, delta=DeltaMessage(content=new_text), finish_reason=None)
        chunk = ChatCompletionResponse(model=model_id,
                                       choices=[choice_data],
                                       object='chat.completion.chunk')
        yield '{}'.format(_dump_json(chunk, exclude_unset=True))

    if current_length != len(_new_response):  # 一直不走这？
        # Determine whether to print the delay tokens
        # delayed_text = _new_response[current_length:]
        # new_text = trim_stop_words(delayed_text, stop_words)
        if len(new_text) > 0:
            choice_data = ChatCompletionResponseStreamChoice(
                index=0, delta=DeltaMessage(content=new_text), finish_reason=None)
            chunk = ChatCompletionResponse(model=model_id,
                                        choices=[choice_data],
                                        object='chat.completion.chunk')
            yield '{}'.format(_dump_json(chunk, exclude_unset=True))

    # 结尾输出None
    # choice_data = ChatCompletionResponseStreamChoice(index=0,
    #                                                  delta=DeltaMessage(),
    #                                                  finish_reason='stop')
    # chunk = ChatCompletionResponse(model=model_id,
    #                                choices=[choice_data],
    #                                object='chat.completion.chunk')
    # yield '{}'.format(_dump_json(chunk, exclude_unset=True))
    yield '[DONE]'
    _gc()


def load_model(config: dict, checkpoint_path: str, max_batch_size: int, max_seq_length: int):
    '''加载dcformer模型'''
    assert os.path.isdir(checkpoint_path), print(f'checkpoint_path: {checkpoint_path}')
    dcformer = DCFormer.from_pretrained(checkpoint_path, trust_remote_code=False, device_map=device_map, torch_dtype=torch.float16)
    print('setup cache')
    with torch.device(dcformer.device):
        dcformer.setup_caches(max_batch_size=max_batch_size, max_seq_length=max_seq_length, set_kv_cache=True)
    return dcformer


def check_shutdown():
    global server_shutdown
    if server_shutdown:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="this server has been shutdown"
        )


@app.get("/health")
@app.post("/health")
@app.get("/GraphService/cm2_status")
@app.post("/GraphService/cm2_status")
@app.get("/SearchService/cm2_status")
@app.post("/SearchService/cm2_status")
@app.get("/status")
@app.post("/status")
@app.post("/health_check")
async def health():
    check_shutdown()
    return "ok"


@app.get("/")
async def health():
    check_shutdown()
    return {"status": "home"}


@app.get("/worker_status")
def worker_status():
    check_shutdown()
    return {"available_concurrency": 1, "alive": True,}
    

def _get_args():
    parser = ArgumentParser()
    parser.add_argument(
        '-c',
        '--checkpoint-path',
        type=str,
        default='Qwen/Qwen-7B-Chat',
        help='Checkpoint name or path, default to %(default)r',
    )
    parser.add_argument('--api-auth', help='API authentication credentials')
    parser.add_argument('--cpu-only',
                        action='store_true',
                        help='Run demo with CPU only')
    parser.add_argument('--server-port',
                        type=int,
                        default=9010,
                        help='Demo server port.')
    parser.add_argument(
        '--server-name',
        type=str,
        default='127.0.0.1',
        help=
        'Demo server name. Default: 127.0.0.1, which is only visible from the local computer.'
        ' If you want other computers to access your server, use 0.0.0.0 instead.',
    )
    parser.add_argument(
        '--disable-gc',
        action='store_true',
        help='Disable GC after each response generated.',
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    server_shutdown = False

    args = _get_args()

    if args.api_auth:
        app.add_middleware(BasicAuthMiddleware,
                           username=args.api_auth.split(':')[0],
                           password=args.api_auth.split(':')[1])

    if args.cpu_only:
        device_map = 'cpu'
    else:
        device_map = 'cuda'

    checkpoint_path = args.checkpoint_path
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
    model = load_model(model_config, checkpoint_path, max_batch_size, max_seq_length).eval()
    checkpoint_dir = os.path.dirname(checkpoint_path)
    tokenizer_path = os.path.join(checkpoint_dir, 'tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, trust_remote_code=True)

    compiled_decode_one_token = torch.compile(decode_one_token, mode="reduce-overhead", fullgraph=True)
    compiled_prefill_func = torch.compile(prefill_decode, mode="reduce-overhead", fullgraph=True, dynamic=True)
    
    uvicorn.run(app, host=args.server_name, port=args.server_port, workers=1)