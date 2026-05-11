from utils.eval_utils import evaluate_model_on_dataset
from utils.model_utils import get_model_tokenizer

def main(model_name, dataset_name, shots, metric):
    info = (f'model_name={model_name}\n'
            f'dataset_name={dataset_name}\n'
            f'shots={shots}\n'
            f'metric={metric}')
    print(info)
    model, tokenizer = get_model_tokenizer(model_name)
    model.to('cuda').eval()

    accuracy = evaluate_model_on_dataset(
        model,
        tokenizer,
        eval_batch_size=8,
        few_shots=shots,
        dataset_name=dataset_name,
        metric=metric)

    file_name = f'pretrained_{model_name}_{dataset_name}_{shots}-shots_{metric}.txt'
    with open(file_name, 'w') as w:
        w.write(f'{info}\n')
        w.write(f'accuracy={accuracy}%\n')

if __name__ == '__main__':
    model_name = 'llama3.1-8B-i'

    dataset_name = 'arc_challenge'
    # dataset_name = 'hellaswag'
    # dataset_name = 'mmlu'
    # dataset_name = 'winogrande'

    num_shots = {
        'arc_challenge': 25,
        'hellaswag': 10,
        'mmlu': 5,
        'winogrande': 5,
    }[dataset_name]

    main(model_name=model_name, dataset_name=dataset_name, shots=num_shots, metric='acc,none')
