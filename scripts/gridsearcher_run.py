import os
import sys
sys.path.append(os.getcwd())

from string import Template
import time
import os
from gridsearcher import GridSearcher, SchedulingConfig, TorchRunConfig
from tqdm import tqdm
import argparse
import psutil
import math
import socket
import gpustat

def get_arg_parse():
    def wait_for_pids(pids):
        while any([psutil.pid_exists(int(pid)) for pid in pids]):
            print(f'waiting for processes {pids} to end...')
            time.sleep(60)

    def get_pids_per_gpu():
        # num_gpus = torch.cuda.device_count()
        num_gpus = 8
        pids = [0] * num_gpus
        gpus = gpustat.new_query().gpus
        for gid in range(num_gpus):
            for p in gpus[gid]['processes']:
                pids[gid] = p['pid']
        return pids

    parser = argparse.ArgumentParser()

    # wait for the pids in "wait_pids" to finish before starting current runs
    parser.add_argument('--wait_pids', nargs='+', default=None, required=False)

    # wait "wait_secs" seconds before starting current runs
    parser.add_argument('--wait_secs', type=int, default=None, required=False)

    # wait for all current jobs on all GPUs to finish before starting current runs
    parser.add_argument("--wait_current", action="store_true", default=False)

    # path to script to be run
    parser.add_argument('--script_path', type=str, required=True)

    # root directory to store the artefacts
    parser.add_argument('--root_dir', type=str, required=True)

    # wandb entity used for your runs
    parser.add_argument('--wandb_entity', type=str, required=True)
    args = parser.parse_args()

    if args.wait_current:
        wait_for_pids(pids=get_pids_per_gpu())
    else:
        if args.wait_pids is not None:
            wait_for_pids(pids=args.wait_pids)
        if args.wait_secs is not None:
            print(f'Waiting {args.wait_secs} seconds')
            for _ in tqdm(range(args.wait_secs)):
                time.sleep(1)
    return args

def main(script_path, root_dir, wandb_project, wandb_group_prefix, wandb_entity, adapter_type, params):
    gs = GridSearcher(script=script_path, defaults=dict())
    gs.add_param('adapter_type', adapter_type)

    if adapter_type in ['lora', 'dylora']:
        wandb_group = '${model_name}_${dataset_name}_${adapter_type}_r=${rank}_E=${epochs}_s=${lora_scaling}_tr=${train_ranks}'
    elif adapter_type == 'matryoshka':
        wandb_group = '${model_name}_${dataset_name}_${adapter_type}_mask=${matryoshka_mask_type}_r=${rank}_E=${epochs}_s=${lora_scaling}_tr=${train_ranks}'

    wandb_job_type = 'lr=${lr}'

    if len(wandb_group_prefix) > 0:
        wandb_group = f'{wandb_group_prefix}_{wandb_group}'

    wandb_name = f'{wandb_group}_{wandb_job_type}' + '_seed=${seed}'

    gs.add_param('wandb_entity', wandb_entity)
    gs.add_param('wandb_project', wandb_project)
    gs.add_param('wandb_group', Template(wandb_group))
    gs.add_param('wandb_job_type', Template(wandb_job_type))
    gs.add_param('wandb_name', Template(wandb_name))

    output_dir = os.path.join(root_dir, wandb_project, wandb_name)
    gs.run(
        param_name_for_exp_root_folder='output_dir',
        exp_folder=Template(output_dir),
        cfg_sched=SchedulingConfig(
            distributed_training=False,
            max_jobs_per_gpu=1,
            gpus=gpus,
            params_values=params,
        ),
        cfg_torchrun=TorchRunConfig(
            launch_blocking=0,
            torchrun=False,
            master_addr='127.0.0.1',
            master_port=29500 + gpus[0],
            rdzv_backend='c10d',
        ),
        # debug=True,
    )

if __name__ == '__main__':
    args = get_arg_parse()

    gpus = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
    ]
    for ADAPTER_TYPE in [
            'lora',
            'dylora',
            'matryoshka',
        ]:

        grid = {
            'model_name': [
                'llama3.2-1B-i',
                'llama3.1-8B-i',
            ],
            'target_layers': ['q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj'],
            'device_batch_size': [4],
            'grad_acc_steps': [8],

            'dataset_name': ['gsm8k'], 'epochs': [3], 'eval_shots':['3,8'],

            'rank': [
                32,
                256,
            ],
            'lora_scaling': [
                'none', # s_k = 1
                'rank', # s_k = 1 / rank
                'sqrt', # sk = 1 / sqrt(rank)
            ],
            'train_ranks': [
                '1,2,4,8,16,32,64,128,256', # will be truncated to contain train ranks <= rank
            ],
            'eval_ranks': ['1,2,4,8,16,32,64,128,256'],
            'seed': [
                42,
                666,
                2408,
            ],
            'lr': [
                '5e-6',
                '6e-6',
                '7e-6',
                '8e-6',
                '9e-6',

                '1e-5',
                '2e-5',
                '3e-5',
                '4e-5',
                '5e-5',
                '6e-5',
                '7e-5',
                '8e-5',
                '9e-5',

                '1e-4',
                '2e-4',
                '3e-4',
                '4e-4',
                '5e-4',
                '6e-4',
                '8e-4',
                '9e-4',

                '1e-3',
            ]
        }

        if ADAPTER_TYPE == 'matryoshka':
            grid['matryoshka_mask_type'] = [
                'diag', # the efficient version that uses matrix P
                # 'binary', # the version with binary masks (use "diag")
            ]

        main(
            script_path=args.script_path,
            root_dir=args.root_dir,
            wandb_project='matryoshka_lora',
            wandb_group_prefix='', # add some prefix for wandb group if needed
            wandb_entity=args.wandb_entity,
            adapter_type=ADAPTER_TYPE,
            params=grid,
        )