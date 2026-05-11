import os
import sys
sys.path.append(os.getcwd())
import argparse
from string import Template
from gridsearcher import GridSearcher, SchedulingConfig, TorchRunConfig
import socket
import wandb

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_entity', type=str, required=True)
    parser.add_argument('--wandb_project', type=str, required=True)
    parser.add_argument('--script_path', type=str, required=True)
    parser.add_argument('--eval_dataset', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    return parser.parse_args()

def main(args, gpus, params):
    gs = GridSearcher(script=args.script_path, defaults=dict())
    output_dir = args.output_dir
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
    args = get_args()

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

    wandb_entity = args.wandb_entity
    wandb_project = args.wandb_project

    eval_dataset = args.eval_dataset

    api = wandb.Api()
    runs = api.runs(f'{wandb_entity}/{wandb_project}')

    main(
        args=args,
        gpus=gpus,
        params={
            'wandb_entity': [wandb_entity],
            'wandb_project': [wandb_project],
            'eval_dataset': [eval_dataset],
            'wandb_run_id': [r.id for r in runs], # loop through wandb runs
        }
    )
