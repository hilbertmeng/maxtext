import os
import sys
new_path_entry = "/home/lishengping/projects/maxtext"
new_path_entry2 = "/home/lishengping/projects/maxtext/MaxText/"
sys.path.append(new_path_entry)
sys.path.append(new_path_entry2)
# os.environ["PATH"] = f"{new_path_entry}:{os.environ['PATH']}"

from etils import epath
import orbax.checkpoint as ocp
import json
from collections import defaultdict
# os.environ["JAX_PLATFORMS"] = "cpu"
# from smart_open import open
from flax.traverse_util import flatten_dict, unflatten_dict
import orbax.checkpoint
import orbax
import jax
import functools
import time
from copy import copy, deepcopy
from MaxText import checkpointing, pyconfig, max_utils
from MaxText.train import setup_mesh_and_model, create_data_iterator
from MaxText.max_utils import init_initial_state, unbox_logicallypartioned
import optax
from optax._src.transform import ScaleByAdamState

def load_ckpt(exp, step=1000):
    argv = [
        'MaxText/train.py',
        'MaxText/configs/base.yml',
        'base_output_directory=gs://newproject-1-llm_projects_us-east5/log/',
        f'run_name={exp}',
        'dataset_path=gs://newproject-1-common_datasets_us-east5/pythia_pile_idxmaps_tfrecord',
        f'exp_class={exp}',
        'save_config=False',
        'skip_jax_distributed_system=True',
    ]
    config = pyconfig.initialize(argv)
    init_rng, writer, checkpoint_manager, mesh, model, learning_rate_schedule, tx = setup_mesh_and_model(config)
    data_iterator, eval_data_iterator = create_data_iterator(config, mesh)
    abstract_state, state_mesh_annotations, state_mesh_shardings = max_utils.get_abstract_state(model, tx, config, init_rng, mesh, is_training=True)
    if step == 0:
        is_training = True
        init_state_partial = functools.partial(init_initial_state, model, tx, config, is_training)
        init_state_partial.__name__ = "initialize_state"
        state = jax.jit(init_state_partial, in_shardings=None, out_shardings=state_mesh_shardings)(init_rng)
        state = unbox_logicallypartioned(state)
    else:
        data_iterator.meta_dict['checkpoint_step'] = step
        restored, raw_params = checkpointing.load_state_if_possible(
            checkpoint_manager,
            data_iterator,
            config.load_parameters_path,
            config.load_full_state_path,
            abstract_state,
            config.enable_single_replica_ckpt_restoring,
            config.dataset_type,
            config=config, # lsp
        )
        state = restored['items']
    return config, state, abstract_state, checkpoint_manager

def save_ckpt(config, checkpoint_manager, step, state):
    chunk_byte_size = config.checkpoint_storage_target_data_file_size_bytes
    save_args = jax.tree.map(lambda _: orbax.checkpoint.SaveArgs(chunk_byte_size=chunk_byte_size), state)
    checkpoint_manager.save(
        step,
        args=orbax.checkpoint.args.Composite(
            items=orbax.checkpoint.args.PyTreeSave(
                item=state, save_args=save_args, ocdbt_target_data_file_size=chunk_byte_size
            )
        ),
    )

def _convert_state(params, params2, src_num_layers=6): # params/mu/nu
    print('src_num_layers: ', src_num_layers)
    for k in params2.keys():
        str_k = '.'.join(k)
        if 'decoder.layers' in str_k:
            lidx = int(k[2].split('_')[-1])
            if lidx >= src_num_layers-1 and 'mudd_mlp' in str_k:
                print(f'skip {k} because it is mudd layer after src_num_layers')
                continue
        if k in params:
            if params2[k].shape != params[k].shape:
                print(f'shape mismatch: {k} {params2[k].shape} {params[k].shape}')
                continue
            params2[k] = deepcopy(params[k])
            print('same key: ', k)
        else:
            k1 = list(k)
            k1[2] = f'layers_{lidx%src_num_layers}'
            k1 = tuple(k1)
            if k1 not in params:
                print(f'key not found: {k1}')
                continue
            if params2[k].shape != params[k1].shape:
                print(f'shape mismatch: {k} {params2[k].shape} {params[k1].shape}')
                continue
            params2[k] = deepcopy(params[k1])
            print(f'target key {k}')
            print(f'src    key {k1}')
        print('--------------------------------')
    return params2

def convert_state(state, state2, convert_opt_state=False, convert_step=False, src_num_layers=6):
    # convert params
    params = flatten_dict(state.params)
    params2 = flatten_dict(state2.params)
    params2 = _convert_state(params, params2, src_num_layers=src_num_layers)
    
    # convert opt_state
    if convert_opt_state:
        mu = flatten_dict(state.opt_state.mu)
        nu = flatten_dict(state.opt_state.nu)
        mu2 = flatten_dict(state2.opt_state.mu)
        nu2 = flatten_dict(state2.opt_state.nu)

        mu2 = _convert_state(mu, mu2, src_num_layers=src_num_layers)
        nu2 = _convert_state(nu, nu2, src_num_layers=src_num_layers)
        opt_state = ScaleByAdamState(count=state.opt_state.count, mu=unflatten_dict(mu2), nu=unflatten_dict(nu2))    
    else:
        opt_state = state2.opt_state
    if convert_step:
        step = state.step
    else:
        step = state2.step
    state2 = state2.replace(params=unflatten_dict(params2), step=step, opt_state=opt_state)
    return state2

if __name__ == "__main__":
    # convert_opt_state=False; convert_step=False
    # exp = 'Llama2MediumBaseL6'; # exp2 = 'Llama2MediumBaseGrowth5xTokens'; step = 13500; step2 = 0
    # exp2 = 'Llama2MediumBaseGrowth5xTokensDisOpt'
    # exp = 'Llama2MediumBaseL6WSDLr1e3'; exp2 = 'Llama2MediumBaseGrowth5xTokensDisOptBaseLr1e3Step7k'; step = 7000; step2 = 0
    # exp = 'Llama2MediumBaseL6WSDLr1e3'; exp2 = 'Llama2MediumBaseGrowth5xTokensDisOptBaseLr1e3Step4k'; step = 4000; step2 = 0
    # exp = 'Llama2MediumBaseL6WSDLr1e3'; exp2 = 'Llama2MediumBaseGrowth5xTokensDisOptBaseLr1e3Step10k'; step = 10000; step2 = 0
    # exp = 'Llama2MediumBaseL6WSDLr1e3'; exp2 = 'Llama2MediumBaseGrowth5xTokensDisOptBaseLr1e3Step1k'; step = 1000; step2 = 0
    # exp = 'Llama2MediumBase2M6LNeox'; exp2 = 'Llama2MediumBase2MGrowth5kNeox'; step = 5000; step2 = 0 

    # exp = 'Llama2MediumBaseL6WSDLr1e3Interval100'; exp2 = 'Llama2MediumBaseGrowth5xTokensDisOptBaseLr1e3Step0p5k'; step = 500; step2 = 0
    # exp = 'Llama2MediumBaseL6WSDLr1e3Interval100'; exp2 = 'Llama2MediumBaseGrowth5xTokensDisOptBaseLr1e3Step0p8k'; step = 800; step2 = 0
    # exp = 'Llama2MediumBaseL6WSDLr1e3Interval100'; exp2 = 'Llama2MediumBaseGrowth5xTokensDisOptBaseLr1e3Step0p2k'; step = 200; step2 = 0

    convert_opt_state=True; convert_step=True
    # exp = 'Llama2MediumBaseL12Tokens5XInterval100'; exp2 = 'Llama2MediumBaseTokens5XGrowthSteps0p2kF2'; step = 200; step2 = 200
    # exp = 'Llama2MediumBaseL12Tokens5XInterval100'; exp2 = 'Llama2MediumBaseTokens5XGrowthSteps2kF2'; step = 2000; step2 = 2000
    # exp = 'Llama2MediumBaseL12Tokens5XInterval100'; exp2 = 'Llama2MediumBaseTokens5XGrowthSteps0p5kF2'; step = 500; step2 = 500
    # exp = 'Llama2MediumBaseL12Tokens5XInterval100'; exp2 = 'Llama2MediumBaseTokens5XGrowthSteps2kF2'; step = 2000; step2 = 2000
    exp ='DC3MuddLlama2MediumBase5xTokensL12'; exp2 ='DC3MuddLlama2MediumBase5xTokensGrowth1k'; step = 1000; step2 = 1000;


    
    config, state, abstract_state, checkpoint_manager = load_ckpt(exp, step=step)
    config2, state2, abstract_state2, checkpoint_manager2 = load_ckpt(exp2, step=0)
    state2 = convert_state(state, state2, convert_opt_state=convert_opt_state, convert_step=convert_step, src_num_layers=config.num_decoder_layers)
    print('state2.opt_state.count, state2.step: ', state2.opt_state.count, state2.step)

    save_ckpt(config2, checkpoint_manager2, step2, state2)
    time.sleep(15)

    # match step file for continue training
    if step2 == 0:
        cmd1 = f"gsutil cp gs://newproject-1-llm_projects_us-east5/log/{exp}/checkpoints/{step}/skip_file_and_step.json ./"
        _ = os.system(cmd1)
        time.sleep(5)
        with open('skip_file_and_step.json', 'r') as f:
            data = json.load(f)
        with open('skip_file_and_step.json', 'w') as f:
            data['checkpoint_step'] = step2
            json.dump(data, f)
        print('data: ', data)

        cmd1 = f"gsutil cp skip_file_and_step.json gs://newproject-1-llm_projects_us-east5/log/{exp2}/checkpoints/{step2}/skip_file_and_step.json"
        cmd2 = f"gsutil cp skip_file_and_step.json gs://newproject-1-llm_projects_us-east5/log/{exp2}/checkpoints/"
    else:
        cmd1 = f"gsutil cp gs://newproject-1-llm_projects_us-east5/log/{exp}/checkpoints/{step}/skip_file_and_step.json gs://newproject-1-llm_projects_us-east5/log/{exp2}/checkpoints/{step2}/skip_file_and_step.json"
        cmd2 = f"gsutil cp gs://newproject-1-llm_projects_us-east5/log/{exp}/checkpoints/{step}/skip_file_and_step.json gs://newproject-1-llm_projects_us-east5/log/{exp2}/checkpoints/"
    out1 = os.system(cmd1)
    out2 = os.system(cmd2)
    time.sleep(5)
    print('cmd1, cmd2, out1, out2: ', cmd1, cmd2, out1, out2)