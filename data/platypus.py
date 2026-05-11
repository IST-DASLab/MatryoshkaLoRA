from datasets import load_dataset

def platypus_load_train(tokenizer, train_size=None, max_length=512):
    dataset = load_dataset("garage-bAInd/Open-Platypus", split="train")

    if train_size:
        dataset = dataset.shuffle(seed=42).select(range(train_size))

    def tokenize_function(examples):
        tokenized_full = []
        tokenized_labels = []

        for instruction, input_text, output in zip(
            examples["instruction"],
            examples["input"],
            examples["output"],
        ):
            # Instruction-style formatting (LLaMA-style)
            if input_text:
                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

            answer = output + tokenizer.eos_token

            p_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
            a_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]

            full_ids = p_ids + a_ids
            l_ids = [-100] * len(p_ids) + a_ids

            tokenized_full.append(full_ids[:max_length])
            tokenized_labels.append(l_ids[:max_length])

        return {
            "input_ids": tokenized_full,
            "labels": tokenized_labels,
        }

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
    )

    return tokenized_dataset