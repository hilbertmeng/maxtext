import requests
import json
import pickle
import time

headers = {
    'Content-Type': 'application/json',
}
json_data = {
    'seed': 0,
    'prompt': '',
    'temperature': 0.7, #0.5
    'penalty': 1.1, #1.1
    'sample': True,
    'num_tokens_to_generate': 20, # 生成的最大长度
    'top_p': 0.9, #0.9
    'top_k': 30, #30
    'complie': False, # 如果没有编译过，第一次编译时间大约需要3min
    'start_id': 151646,
    'stop_id': 151643,
}
prefix = 'Novel name: \nNovel category: \nx\n第xx章 \n\n'

import pandas as pd
data = pd.read_excel('/home/lishengping/lsp/data/小梦和天启大模型测评.xlsx', sheet_name='online_vs_qwen_1227')

outputs = []
for i, text in enumerate(data['input'].unique()):
    # // if i < 9: continue
    prompt = prefix + text
    json_data['prompt'] = prompt
    json_data['temperature'] = 0.7
    json_data['num_tokens_to_generate'] = 200
    json_data['complie'] = False
    print(f'prompt:\n{text}')
    for seed in [0, 42, 1234]:
        json_data['seed'] = seed
        response = requests.post('http://0.0.0.0:9000/infer', headers=headers, json=json_data)
        result = json.loads(response.text)
        output = result['response']['output']
        real_output = output.strip('<|extra_0|>').replace(prompt, '')
        outputs.append([prompt, real_output])
        print(f'output{i}-seed{seed}:\n{real_output}\n\n')
        print('============================================================')
    time.sleep(5)

pickle.dump(outputs, open('xm3.5.l32k.result.pkl', 'wb'))