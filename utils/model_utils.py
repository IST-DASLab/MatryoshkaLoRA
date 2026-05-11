import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from adapters import *
from adapters.dylora import DyLoraLayer
from adapters.matryoshka import MatryoshkaLayer


def get_model_tokenizer(model_name):
    model_id = {
        'llama3.2-1B-i': 'meta-llama/Llama-3.2-1B-Instruct',
        'llama3.1-8B-i': 'meta-llama/Llama-3.1-8B-Instruct',
    }[model_name]

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype="auto",  # Full precision (No 4-bit)
        trust_remote_code=True,
    )
    return model, tokenizer

def replace_linear_with_lora(model, adapter_type, r, target_layers, **kwargs):
    # disable gradient for all layers
    for p in model.parameters():
        p.requires_grad = False

    for name, module in model.named_modules():
        if any(proj in name for proj in target_layers):  # Target specific projections
            parent_name, child_name = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
            target = getattr(parent, child_name)

            # Create new adapter layer
            if adapter_type == 'lora':
                new_layer = LoraLayer(target, rank=r, scaling=kwargs['scaling'])
                # print(f'Created LoraLayer for layer {name}')
            elif adapter_type == 'matryoshka':
                new_layer = MatryoshkaLayer(target, rank=r, scaling=kwargs['scaling'], train_ranks=kwargs['train_ranks'], mask_type=kwargs['mask_type'])
                # print(f'Created MatryoshkaLayer for layer {name}')
            elif adapter_type == 'dylora':
                new_layer = DyLoraLayer(target, rank=r, scaling=kwargs['scaling'])
                # print(f'Created DyLoraLayer for layer {name}')
            else:
                raise RuntimeError(f'Unknown adapter type: {adapter_type}')
            setattr(parent, child_name, new_layer)

            # Freeze base weights
            # new_layer.base_layer.weight.requires_grad = False

    return model

def save_lora_weights(model, output_dir, step):
    # Filter for only trainable parameters
    lora_state_dict = {
        k: v
        for k, v in model.state_dict().items()
        if "lora_" in k
    }
    full_path = os.path.join(output_dir, f"lora_adapters_step={step:05d}.pt")
    torch.save(lora_state_dict, full_path)
    print(f"LoRA adapters saved to {full_path}")

def manual_inspect(model, tokenizer, prompt):
    model.eval()
    inputs = tokenizer(f"Instruction: {prompt}\nOutput:", return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


def load_inference_model(model_id, adapter_path, lora_type="custom", r=16):
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Base Model (Full Precision / BF16)
    print(f"Loading base model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        trust_remote_code=True
    )

    # 3. Re-apply the Patching logic
    # This creates the lora_A and lora_B parameters in the model structure
    print(f"Patching model with {lora_type} structure...")
    model = replace_linear_with_lora(model, adapter_type=lora_type, r=r)

    # 4. Load the saved Adapter weights
    print(f"Loading adapter weights from {adapter_path}")
    adapter_weights = torch.load(f"{adapter_path}/adapter_model.bin")

    # Use strict=False because adapter_weights only contains LoRA params,
    # but the model now contains both base and LoRA params.
    missing_keys, unexpected_keys = model.load_state_dict(adapter_weights, strict=False)

    # Verification: check if all lora keys were loaded
    lora_keys = [k for k in adapter_weights.keys()]
    print(f"Loaded {len(lora_keys)} adapter tensors successfully.")

    model.eval()
    return model, tokenizer

    # Usage
    # model, tokenizer = load_inference_model(
    #     model_id="meta-llama/Llama-3.2-1B",
    #     adapter_path="./lora_experiment",
    #     lora_type="custom",
    #     r=16
    # )

def print_model(model):
    param_count = 0
    for i, (name, p) in enumerate(model.named_parameters()):
        param_count += p.numel()
        print(f'{i=}\n\t{name=}\n\t{p.numel()=:_}\n\t{p.shape=}\n\t{p.requires_grad=}\n')
    print(f'\nTotal params: {param_count:_}')

def print_trainable_params(model):
    trainable = 0
    total = 0

    frozen_layers = []
    trainable_layers = []

    for i, (name, p) in enumerate(model.named_parameters()):
        total += p.numel()

        tpl = (i, name, str(tuple(p.shape)))

        if p.requires_grad:
            trainable += p.numel()
            trainable_layers.append(tpl)
        else:
            frozen_layers.append(tpl)

    print(f'----- Frozen Params -----')
    for idx, name, shape in frozen_layers:
        print(f'#{idx:3d} grads:no {shape:16s} {name}')

    print(f'----- Trainable Params -----')
    for idx, name, shape in trainable_layers:
        print(f'#{idx:3d} grads:yes {shape:16s} {name}')

    print(f'----- Param Counts -----')
    print(f'    Total: {total:_} (100%)')
    print(f'   Frozen: {total-trainable:_} ({(total-trainable)/total * 100:.2f}%)')
    print(f'Trainable: {trainable:_} ({trainable / total * 100.:.2f}%)')
    print(f'----- End Stats -----')

def set_inference_rank(model, inf_rank):
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            # print(f'Setting inf_rank to {inf_rank} for layer {name}')
            module.set_inference_rank(inf_rank)