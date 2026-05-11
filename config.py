import argparse

def preprocess(args):
    """
    Example:
        rank = 16
        train_ranks = '1,2,4,8,16,32'
        final: train_ranks = [1, 2, 4, 8, 16]: 32 is removed because it is greater than rank
    """
    args.eval_ranks = sorted([r for r in map(int, args.eval_ranks.split(',')) if r <= args.rank])
    print(f'[config][preprocess] train_ranks = {args.train_ranks}, eval_ranks = {args.eval_ranks}')

    args.train_ranks = sorted([r for r in map(int, args.train_ranks.split(',')) if r <= args.rank])

    args.eval_shots = sorted([
        shots
        for shots in map(int, args.eval_shots.split(','))
    ])

    return args

def parse_args():
    parser = argparse.ArgumentParser(description="LoRA Variant Experimentation Script")

    parser.add_argument("--model_name", type=str, default="llama3.2-1B", help="Model ID on HuggingFace")

    parser.add_argument("--dataset_name", type=str, default="alpaca", help="Dataset ID on HuggingFace")
    parser.add_argument("--adapter_type", type=str, default="lora", choices=["lora", "dylora", "matryoshka"])
    parser.add_argument("--matryoshka_mask_type", default="larger", type=str, choices=['binary', 'diag'])
    parser.add_argument("--target_layers", type=str, help="Comma separated keywords for layers")

    # LoRA Hyperparameters
    parser.add_argument("--rank", type=int, default=16, help="The rank (r) for the LoRA adapters")
    parser.add_argument("--lora_scaling", default="none", type=str, choices=['none', 'rank', 'sqrt'])
    parser.add_argument("--train_ranks", type=str, default='1', help="The train ranks separated by a space")
    parser.add_argument("--eval_ranks", type=str, default='1', help="The eval ranks separated by a space")
    parser.add_argument("--eval_shots", type=str, default='0,8', help="Comma-separated num shots for evaluation")

    # Training Hyperparameters
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--epochs", type=float, default=3, help="Number of training epochs")
    parser.add_argument("--device_batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--grad_acc_steps", type=int, default=4, help="Grad acc steps")
    parser.add_argument("--log_steps", type=int, default=1, help="Log steps frequency")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for `seed` and `data_seed`")

    # WandB args
    parser.add_argument("--wandb_entity", default=None, type=str)
    parser.add_argument("--wandb_project", default="my-project", type=str)
    parser.add_argument("--wandb_group", default=None, type=str)
    parser.add_argument("--wandb_job_type", default=None, type=str)
    parser.add_argument("--wandb_name", default=None, type=str)

    return preprocess(parser.parse_args())
