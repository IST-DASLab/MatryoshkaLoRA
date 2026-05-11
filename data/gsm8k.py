from datasets import load_dataset

def gsm8k_load_train(tokenizer, train_size=None, max_length=512):
    dataset = load_dataset("gsm8k", "main", split="train")

    if train_size:
        dataset = dataset.shuffle(seed=42).select(range(train_size))

    def tokenize_function(examples):
        tokenized_full = []
        tokenized_labels = []

        for q, a in zip(examples["question"], examples["answer"]):
            # Llama-style formatting
            prompt = f"Question: {q}\nAnswer: "
            answer = f"{a}{tokenizer.eos_token}"

            # Tokenize separately to calculate lengths
            p_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
            a_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]

            # Combine: [Prompt] + [Answer]
            full_ids = p_ids + a_ids
            # Labels: [-100 for Prompt] + [Actual IDs for Answer]
            # -100 tells the Trainer to ignore these tokens in loss calculation
            l_ids = [-100] * len(p_ids) + a_ids

            # Truncate to max_length
            tokenized_full.append(full_ids[:max_length])
            tokenized_labels.append(l_ids[:max_length])

        return {"input_ids": tokenized_full, "labels": tokenized_labels}

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
    )

    return tokenized_dataset

