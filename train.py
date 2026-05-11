import gc
import os
import time

import torch
import wandb
import random
import numpy as np
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)

from adapters import LoraLayer
from adapters.dylora import DyLoraLayer
from config import parse_args
from data import (
    # get_datasets,
    gsm8k_load_train,
)
from data.platypus import platypus_load_train
from utils.eval_utils import evaluate_model_on_dataset
from metrics import AURAC_v2, AURAC_v1
from utils.model_utils import (
    get_model_tokenizer,
    replace_linear_with_lora,
    save_lora_weights,
    print_model,
    print_trainable_params,
    set_inference_rank,
)

class CustomTrainer(Trainer):
    def __init__(self, *args, rank, **kwargs):
        self.train_ranks = kwargs.pop('train_ranks')

        super().__init__(*args, **kwargs)

        self.rank = rank

        print(f'[CustomTrainer] train_ranks = {self.train_ranks}')

        # [2 ** i for i in range(1 + int(math.log2(self.rank)))]
        self.lora_modules_list = [
            module
            for module in self._get_model().modules()
            if isinstance(module, LoraLayer)
        ]
        self.forward_step_count = 0 # used to log the randomly sampled rank

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Overridden to inject the dynamic rank sampling before the forward pass for DyLoRA
        """
        index = random.randint(0, len(self.train_ranks) - 1) # both ends are included
        rand_inf_rank = self.train_ranks[index]

        is_dylora = False
        for module in self.lora_modules_list:
            if isinstance(module, DyLoraLayer):
                module.set_inference_rank(rand_inf_rank)
                is_dylora = True

        if is_dylora:
            self.forward_step_count += 1
            wandb.log({
                'dylora/forward_step': self.forward_step_count,
                'dylora/sampled_rank': rand_inf_rank,
            })

        return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)

    def _save_checkpoint(self, model, trial, metrics=None):
        """
        Force the trainer to ONLY call save_model and skip the
        massive optimizer/scheduler saves.
        """
        # Determine the checkpoint folder name
        # checkpoint_folder = f"checkpoint-{self.state.global_step}"
        # output_dir = os.path.join(self.args.output_dir, checkpoint_folder)

        save_lora_weights(
            model=self._get_model(),
            output_dir=self.args.output_dir,
            step=self.state.global_step
        )

    def _get_model(self):
        unwrapped_model = self.model
        if hasattr(unwrapped_model, "_orig_mod"):
            unwrapped_model = unwrapped_model._orig_mod
        if hasattr(unwrapped_model, "module"):
            unwrapped_model = unwrapped_model.module
        # return self.model.module if hasattr(self.model, 'module') else self.model
        return unwrapped_model

def purge_gpu_memory():
    # 1. Clear out any references
    gc.collect()
    # 2. Flush the CUDA cache
    torch.cuda.empty_cache()
    # 3. Reset peak memory stats (optional, for debugging)
    torch.cuda.reset_peak_memory_stats()

def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)

def train(args):
    set_seed(args.seed)

    model, tokenizer = get_model_tokenizer(args.model_name)
    print_model(model)
    model = replace_linear_with_lora(
        model,
        adapter_type=args.adapter_type,
        r=args.rank,
        target_layers=args.target_layers.split(','),
        # kwargs below
        train_ranks=args.train_ranks,
        mask_type=args.matryoshka_mask_type,
        scaling=args.lora_scaling)

    print_trainable_params(model)

    set_inference_rank(model, inf_rank=None)

    if args.dataset_name == 'gsm8k':
        train_data = gsm8k_load_train(tokenizer)
    elif args.dataset_name == 'platypus':
        train_data = platypus_load_train(tokenizer)
    else:
        raise ValueError(f'Unknown dataset {args.dataset_name}')

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        job_type=args.wandb_job_type,
        name=args.wandb_name,
        config=args
    )
    training_args = TrainingArguments(
        output_dir=args.output_dir,

        per_device_train_batch_size=args.device_batch_size,
        gradient_accumulation_steps=args.grad_acc_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,

        seed=args.seed,
        data_seed=args.seed,

        report_to="wandb",
        logging_steps=args.log_steps,

        tf32=True,
        bf16=True,

        save_strategy="epoch",

        # Evaluation
        do_eval=False,
        eval_strategy="no",
        remove_unused_columns=False,
    )
    # model = torch.compile(model)
    trainer = CustomTrainer(
        rank=args.rank, # new!
        model=model,
        args=training_args,
        train_dataset=train_data,
        # eval_dataset=eval_data,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        # compute_metrics=lambda pred: gsm8k_compute_accuracy(pred, tokenizer)
        train_ranks=args.train_ranks,
    )
    # trainer.add_callback(WandbCallback())

    model.config.use_cache = True
    model.generation_config.use_cache = True

    trainer.train()

    del trainer, model, train_data

@torch.no_grad()
def multi_rank_eval(args):
    eval_batch_size = 32
    # eval_batch_size = {
    #     'llama3.2-1B-i': 8,
    #     'llama3.1-8B-i': 8,
    # }.get(args.model_name, 8)

    if args.dataset_name == 'gsm8k':
        metric = 'exact_match,strict-match'
    elif args.dataset_name in ['arc-challenge', 'hellaswag', 'mmlu', 'winogrande']:
        metric = 'acc,none'
    else:
        raise RuntimeError(f'Not sure which metric to use for dataset {args.dataset_name}')

    wandb.define_metric("eval/*", step_metric="eval/step")

    adapter_file_names = sorted([
        f
        for f in os.listdir(args.output_dir)
        if f.startswith("lora_adapters_step=") and f.endswith(".pt")
    ])

    model, tokenizer = get_model_tokenizer(args.model_name)
    model.to('cuda').eval()

    # Evaluate the pre-trained model before loading LoRA adapters
    for shots in args.eval_shots:
        pretrained_acc_file = os.path.join(args.output_dir, f'pretrained_accuracy_{shots}-shots.txt')

        start = time.time()
        accuracy = evaluate_model_on_dataset(
            model,
            tokenizer,
            eval_batch_size=eval_batch_size,
            few_shots=shots,
            dataset_name=args.dataset_name,
            metric=metric)
        end = time.time()

        with open(pretrained_acc_file, 'w') as w:
            w.write(f'{accuracy}\n')

        print('#' * 20)
        print(f'### [{args.model_name}][{args.adapter_type}] {shots}-shots accuracy of pre-trained model {args.model_name}: {accuracy}')
        print('#' * 20)
        wandb.log({
            f'eval/time_{shots}-shots_pretrained': end - start,
            f'eval/{args.dataset_name}_acc_{shots}-shots_pretrained': accuracy,
        })

    file = adapter_file_names[-1] # evaluate only at the last step
    step = int(file.replace('lora_adapters_step=', '').replace('.pt', ''))
    lora_path = os.path.join(args.output_dir, file)

    model, tokenizer = get_model_tokenizer(args.model_name)
    model = replace_linear_with_lora(
        model,
        adapter_type=args.adapter_type,
        r=args.rank,
        target_layers=args.target_layers.split(','),
        # kwargs below
        train_ranks=args.train_ranks,
        mask_type=args.matryoshka_mask_type,
        scaling=args.lora_scaling)

    model.to('cuda').eval()
    for name, p in model.named_parameters():
        if 'lora_' in name:
            p.zero_()

    lora_params = torch.load(lora_path, map_location='cpu')
    load_status = model.load_state_dict(lora_params, strict=False)

    # print(f'Load status: {load_status}')

    with torch.no_grad():
        for name, p in model.named_parameters():
            if 'lora_' in name:
                if p.sum() == 0:
                    print(f'Sum of weights in module {name} is zero! This is likely an issue with loading LoRA adapters.')

    for shots in args.eval_shots:
        accs = []
        for inf_rank in args.eval_ranks:
            set_inference_rank(model, inf_rank)

            print('#' * 20)
            print(f'##### [{args.model_name}][{args.adapter_type}] {shots}-shots evaluation for rank {inf_rank} at step {step}')
            print('#' * 20)

            start = time.time()
            accuracy = evaluate_model_on_dataset(
                model,
                tokenizer,
                eval_batch_size=eval_batch_size,
                few_shots=shots,
                dataset_name=args.dataset_name,
                metric=metric)
            end = time.time()

            wandb.log({
                f'eval/{args.dataset_name}_time_{shots}-shots_rank={inf_rank}': end - start,
                f'eval/{args.dataset_name}_acc_{shots}-shots_rank={inf_rank}': accuracy,
                f'eval/{args.dataset_name}_step': step,
            })
            accs.append(accuracy)

            print(f'{shots}-shots accuracy for rank {inf_rank} at step {step}: {accuracy:.2f}')
        # end for inf_rank

        wandb.log({
            f'{args.dataset_name}_AURAC_{shots}-shots_v1': AURAC_v1(accs, args.eval_ranks),
            f'{args.dataset_name}_AURAC_{shots}-shots_v2': AURAC_v2(accs, args.eval_ranks),
            f'{args.dataset_name}_AVG_{shots}-shots': np.mean(accs),
        })
    # end for shots

    del model, tokenizer, lora_params
    purge_gpu_memory()

if __name__ == "__main__":
    args = parse_args()

    ##### START CHECKS
    # kill spawned jobs based on certain criteria
    # if 'gemma' in args.model_name:
    #     exit(666)
    ##### END CHECKS

    start = time.time()
    train(args)
    end = time.time()
    wandb.log({f'eval/elapsed_train': end-start})

    purge_gpu_memory()

    multi_rank_eval(args)