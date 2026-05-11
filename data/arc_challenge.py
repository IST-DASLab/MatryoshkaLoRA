from datasets import load_dataset

def arc_challenge_load_train(tokenizer, train_size=None, max_length=512):
    dataset = load_dataset("ai2_arc", "ARC-Challenge", split="train")

    if train_size:
        dataset = dataset.shuffle(seed=42).select(range(train_size))

    def tokenize_function(examples):
        tokenized_full = []
        tokenized_labels = []

        for q, choices, answer_key in zip(
            examples["question"],
            examples["choices"],
            examples["answerKey"]
        ):
            labels = choices["label"]   # ['A','B','C','D']
            texts = choices["text"]     # answer strings

            # Build prompt
            prompt = f"Question: {q}\n"
            for l, t in zip(labels, texts):
                prompt += f"{l}. {t}\n"
            prompt += "Answer: "

            # Get correct answer text
            correct_idx = labels.index(answer_key)
            answer = texts[correct_idx] + tokenizer.eos_token

            # Tokenize separately
            p_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
            a_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]

            # Combine
            full_ids = p_ids + a_ids
            l_ids = [-100] * len(p_ids) + a_ids

            # Truncate
            tokenized_full.append(full_ids[:max_length])
            tokenized_labels.append(l_ids[:max_length])

        return {
            "input_ids": tokenized_full,
            "labels": tokenized_labels
        }

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
    )

    return tokenized_dataset