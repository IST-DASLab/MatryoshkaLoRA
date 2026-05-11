import argparse
import time

import numpy as np
import torch

from utils.eval_utils import evaluate_model_on_dataset
from metrics import AURAC_v1, AURAC_v2
from utils.model_utils import get_model_tokenizer, replace_linear_with_lora, set_inference_rank
import wandb
import os

parser = argparse.ArgumentParser(description="Extra evals for existing runs")
parser.add_argument("--wandb_entity", type=str, required=True)
parser.add_argument("--wandb_project", type=str, required=True)
parser.add_argument("--wandb_run_id", type=str, required=True)
parser.add_argument("--eval_dataset", type=str, required=True)
parser.add_argument("--output_dir", type=str, help="placeholder, not used")
args = parser.parse_args()

num_shots = {
    'arc_challenge': 25,
    'hellaswag': 10,
}.get(args.eval_dataset, 5)

print(f'Using {num_shots}-eval for {args.eval_dataset}')

run = wandb.Api().run(f'{args.wandb_entity}/{args.wandb_project}/{args.wandb_run_id}')
cfg = run.config

adapter_file_names = sorted([
    f
    for f in os.listdir(cfg['output_dir'])
    if f.startswith("lora_adapters_step=") and f.endswith(".pt")
])

print(adapter_file_names)
file = adapter_file_names[-1] # evaluate only at the last step
step = int(file.replace('lora_adapters_step=', '').replace('.pt', ''))
lora_path = os.path.join(cfg['output_dir'], file)


with torch.no_grad():
    model, tokenizer = get_model_tokenizer(cfg['model_name'])
    model = replace_linear_with_lora(
        model,
        adapter_type=cfg['adapter_type'],
        r=cfg['rank'],
        target_layers=cfg['target_layers'].split(','),
        # kwargs below
        train_ranks=cfg['train_ranks'],
        mask_type=cfg['matryoshka_mask_type'],
        scaling=cfg['lora_scaling'])

    print(f'model_name: {cfg["model_name"]}')
    print(f'adapter_type: {cfg["adapter_type"]}')
    print(f'rank: {cfg["rank"]}')
    print(f'target_layers: {cfg["target_layers"]}')
    print(f'train_ranks: {cfg["train_ranks"]}')
    print(f'matryoshka_mask_type: {cfg["matryoshka_mask_type"]}')
    print(f'lora_scaling: {cfg["lora_scaling"]}')

    model.to('cuda').eval()
    for name, p in model.named_parameters():
        if 'lora_' in name:
            p.zero_()

    lora_params = torch.load(lora_path, map_location='cpu')
    load_status = model.load_state_dict(lora_params, strict=False)

    for name, p in model.named_parameters():
        if 'lora_' in name:
            if p.sum() == 0:
                print(f'Sum of weights in module {name} is zero! This is likely an issue with loading LoRA adapters.')

    accs = []
    for inf_rank in cfg['eval_ranks']:
        set_inference_rank(model, inf_rank)

        print('#' * 20)
        print(f'##### {num_shots}-shots evaluation for rank {inf_rank} at step {step}')
        print('#' * 20)

        start = time.time()
        accuracy = evaluate_model_on_dataset(
            model,
            tokenizer,
            eval_batch_size=8,
            dataset_name=args.eval_dataset,
            few_shots=num_shots,
            metric='acc,none')
        end = time.time()

        run.summary[f'eval/{args.eval_dataset}_time_{num_shots}-shots_rank={inf_rank}'] = end - start
        run.summary[f'eval/{args.eval_dataset}_acc_{num_shots}-shots_rank={inf_rank}'] = accuracy
        run.summary[f'eval/{args.eval_dataset}_step'] = step
        run.update()
        accs.append(accuracy)

        print(f'{num_shots}-shots accuracy for rank {inf_rank} at step {step}: {accuracy:.2f}')
    # end for inf_rank

run.summary[f'{args.eval_dataset}_AURAC_{num_shots}-shots_v1'] = AURAC_v1(accs, cfg['eval_ranks'])
run.summary[f'{args.eval_dataset}_AURAC_{num_shots}-shots_v2'] = AURAC_v2(accs, cfg['eval_ranks'])
run.summary[f'{args.eval_dataset}_AVG_{num_shots}-shots'] = np.mean(accs)
run.update()