import torch
import lm_eval
from lm_eval.models.huggingface import HFLM

@torch.no_grad()
def evaluate_model_on_dataset(model, tokenizer, eval_batch_size, dataset_name, few_shots=25, metric='acc,none'):
    model.eval()

    lm_eval_model = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=eval_batch_size,
    )

    results = lm_eval.simple_evaluate(
        model=lm_eval_model,
        tasks=[dataset_name],
        num_fewshot=few_shots,
    )

    score = results["results"][dataset_name][metric]
    return score * 100

