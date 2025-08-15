import subprocess
import os
import re
from datetime import datetime
import time
import sys
from google.cloud import storage
import tensorflow as tf
import numpy as np
import random

import plotly.io as pio
import plotly.graph_objects as go

# !pip install plotly==6.1.0


np.random.seed(42)
random.seed(42)

pio.renderers.default = "notebook"  # 或者 "notebook_connected" / "iframe"

def plotly_curve(model_data, key, window_size=1):
    # model_data = {
    #     "Model A": {"steps": [1, 2, 3, 4], "losses": [0.9, 0.7, 0.5, 0.4]},
    #     "Model B": {"steps": [1, 2, 3, 4], "losses": [0.95, 0.8, 0.6, 0.45]},
    #     "Model C": {"steps": [1, 2, 3, 4], "losses": [1.0, 0.85, 0.65, 0.5]},
    # }
    fig = go.Figure()
    for model_name, data in model_data.items():
       
        y = data[key]
        # print(f'orig y: {len(y)} window_size: {window_size}')
        steps = data['steps']
        if window_size > 1:
            y = [
            np.mean(data[key][i:i+window_size])
            for i in range(0, len(data[key]), window_size)
            ]
            steps = [i*5 for i in range(0, len(data['steps']), window_size)]
            
        fig.add_trace(go.Scatter(
            x=steps,
            y=y,
            mode='lines+markers',
            name=model_name,
            hovertemplate='Step: %{x}<br>Loss: %{y}<extra>' + model_name + '</extra>',
            line=dict(width=2, ),
            marker=dict(size=2, symbol='circle'),
        ))
    
    fig.update_layout(
        title="Loss Curve for Multiple Models",
        xaxis_title="Steps",
        yaxis_title="Loss",
        hovermode="x unified",
        template="plotly",  # 默认模板（白底）
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray'
        )
    )
    fig.show()


def extract_tag_values(pathes, name='learning/loss'):
    pathes.sort()
    values = {'steps': [], name: []}
    tags = set()
    for path in pathes:
        summaries = tf.compat.v1.train.summary_iterator(path)
        for step, e in enumerate(summaries):
            for v in e.summary.value:
                if v.tag == name:
                    y = v.simple_value
                    values['steps'].append(e.step)
                    values[name].append(y)
                tags.add(v.tag)
    
    return values


def extract_pathes(dir_):
    command = f'gsutil ls {dir_}'
    pathes = subprocess.run(command, stdout=subprocess.PIPE, shell=True, text=True)
    mtp_pathes = [p.strip() for p in pathes.stdout.split('\n') if p.strip()]
    return mtp_pathes

# llama_dir = 'gs://newproject-1-llm_base_models_us-east5/experiments/lma2-0.4B-qknormF-wdMask-PaxOpt-0304/tensorboard/'
llama_dir = 'gs://newproject-1-llm_base_models_europe-west4/experiments/Llama2Medium0813/tensorboard/'
llama_pathes = extract_pathes(llama_dir)
llama_data = extract_tag_values(llama_pathes)
print(f'Finished......')
mtp_dir = 'gs://newproject-1-llm_base_models_europe-west4/experiments/MTPLlama2Medium0813/tensorboard/'
mtp_pathes = extract_pathes(mtp_dir)
mtp_data = extract_tag_values(mtp_pathes)
print(f'Finished......')
de_dir = 'gs://newproject-1-llm_base_models_europe-west4/experiments/Llama2MediumDeepEmbed0813/tensorboard/'
de_pathes = extract_pathes(de_dir)
de_data = extract_tag_values(de_pathes)
print(f'Finished......')


model_data = {}
model_data['llama-0.4B'] = llama_data
model_data['llama-mtp-0.4B'] = mtp_data
model_data['llama-deepembed-0.4B'] = de_data
# model_data['mudd-llama-0.4B'] = mudd_data
# model_data['mudd-llama-mtp-0.4B'] = mudd_mtp_data

key = 'learning/loss'
plotly_curve(model_data, key, window_size=10)