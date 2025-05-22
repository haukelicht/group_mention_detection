
import os
import shutil
import json
from timeit import default_timer as timer

from utils import read_jsonl
from schema import LABEL2ID as label2id
from finetuning_utils import (
    create_token_classification_dataset, 
    tokenize_and_align_sequence_labels,
    DEFAULT_TRAINING_ARGS as default_training_args
)

import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    set_seed,
)

from finetuning_utils import compute_seqeval_metrics

import optuna

from typing import Dict, List, Callable, Any, Optional

def prepare_datasets(
        tokenizer: AutoTokenizer,
        data_train: List[Dict],
        data_dev: List[Dict],
    ):
    # creat the training split
    train_dataset = create_token_classification_dataset(data_train)
    train_dataset = train_dataset.map(lambda example: tokenize_and_align_sequence_labels(example, tokenizer=tokenizer), batched=True)
    train_dataset = train_dataset.select_columns(['labels', 'input_ids', 'attention_mask'])
    
    # creat the validation split
    dev_dataset = create_token_classification_dataset(data_dev)
    dev_dataset = dev_dataset.map(lambda example: tokenize_and_align_sequence_labels(example, tokenizer=tokenizer), batched=True)
    dev_dataset = dev_dataset.select_columns(['labels', 'input_ids', 'attention_mask'])

    return train_dataset, dev_dataset
    

def run_hyperparameter_search(
    model_name: str,
    label2id: Dict[str, int],
    metric: str,
    data_train: List[Dict],
    data_dev: List[Dict],
    hp_fun: Callable[[optuna.trial.Trial], Dict[str, Any]] = None,
    n_trials: int=10,
    seed: int=42,
    out_dir: str='hpsearch',
    additional_train_args: Optional[Dict[str, Any]] = None,
):
    print('Starting HP search for model', model_name)
    
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True, add_prefix_space=True)
    
    train_dataset, dev_dataset = prepare_datasets(tokenizer, data_train, data_dev)
    
    train_args = default_training_args.copy()
    if additional_train_args is not None:
        train_args.update(additional_train_args)

    print(additional_train_args)
    print(train_args)

    # define train args
    train_args = dict(
        **train_args,
        # how to select "best" model
        metric_for_best_model=metric,
        greater_is_better=True,
        load_best_model_at_end=True,
        save_total_limit=2,
        # when to evaluate
        eval_strategy='epoch',
        # when to save
        save_strategy='epoch',
        # logging
        logging_strategy='no',
        # where to store results
        output_dir=out_dir,
        report_to='none',
        # misc
        fp16=torch.cuda.is_available() and 'deberta' not in model_name,
        # reproducibility
        seed=seed,
        data_seed=seed,
        full_determinism=True
    )
    train_args = TrainingArguments(**train_args)

    def model_init():
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.num_labels = len(label2id)
        config.label2id = label2id
        config.id2label = {v: k for k, v in label2id.items()}
        return AutoModelForTokenClassification.from_pretrained(model_name, config=config, device_map='auto', torch_dtype='auto', trust_remote_code=True)
    
    # create Trainer
    trainer = Trainer( 
        model_init=model_init,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.03)]
    )
    print('using devive:', str(trainer.model.device))
    
    # define sampler
    optuna_sampler = optuna.samplers.TPESampler(
        seed=seed, 
        consider_prior=True, 
        prior_weight=1.0, 
        consider_magic_clip=True, 
        consider_endpoints=False,
        n_startup_trials=int(n_trials/2), 
        multivariate=False,
        group=False, 
        warn_independent_sampling=True, 
        constant_liar=False
    )
    
    print(n_trials)
    s = timer()
    best_run = trainer.hyperparameter_search(
        n_trials=n_trials,
        direction='maximize', 
        hp_space=hp_fun,
        backend='optuna',
        **{"sampler": optuna_sampler}
    )
    e = timer()
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
        
    out = best_run.hyperparameters
    out[metric] = best_run.objective
    out['elapsed'] = e-s
    
    return out


# #### Load libraries
    
def compute_metrics(p): 
    results = compute_seqeval_metrics(p, label_list=list(label2id.keys()))
    return {'seqeval-SG_f1': results['social group']['f1-score']}

def main(args):

    assert os.path.exists(args.train_data_file), f"Train data file {args.train_data_file} does not exist"
    assert os.path.exists(args.dev_data_file), f"Dev data file {args.dev_data_file} does not exist"
    
    # argument parsing and configuration
    if isinstance(args.model_names, str):
        args.model_names = [mn.strip() for mn in args.model_names.split(',')]
    
    # load the data
    data_train = read_jsonl(args.train_data_file, replace_newlines=True)
    data_dev = read_jsonl(args.dev_data_file, replace_newlines=True)

    dest = os.path.join(args.experiment_results_path, args.experiment_name)
    if os.path.exists(dest):
        if args.overwrite:
            print(f"Experiment folder {dest} already exists. Overwriting...")
            shutil.rmtree(dest)
        else:
            raise ValueError(f"Experiment folder {dest} already exists. Please remove it or choose a different name.")
    else:
        os.makedirs(dest)

    # #### hyperparameter search function
    
    def hp_space(trial):
        return {
            'learning_rate': trial.suggest_categorical('learning_rate', args.learning_rates),
            'per_device_train_batch_size': trial.suggest_categorical('per_device_train_batch_size', args.train_batch_sizes),
            'weight_decay': trial.suggest_categorical('weight_decay', args.weight_decays),
        }
    
    set_seed(args.seed)
    results = {}
    for i, model_name in enumerate(args.model_names):
        results[model_name] = run_hyperparameter_search(
            # model config
            model_name=model_name,
            label2id=label2id,
            # data
            data_train=data_train,
            data_dev=data_dev,
            # search params
            metric='seqeval-SG_f1',
            hp_fun=hp_space,
            n_trials=args.n_trials,
            # kwargs
            additional_train_args={
                'gradient_accumulation_steps': args.gradient_accumulation_steps
            },
            # reproducibility
            seed=args.seed,
            out_dir=os.path.join(dest, 'hpsearch')
        )
        fp = os.path.join(dest, 'results.jsonl')
        with open(fp, 'w' if i==0 else 'a') as file:
            json.dump(results, file)
            file.write('\n')

    # finally: write config and results to experiment folder
    config = args.__dict__
    config['grid'] = {
        'learning_rate': args.learning_rates,
        'train_batch_size': args.train_batch_sizes,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'weight_decay': args.weight_decays,
    }

    fp = os.path.join(dest, 'config.json')
    with open(fp, 'w') as file:
        json.dump(config, file, indent=2)

    fp = os.path.join(dest, 'results.json')
    with open(fp, 'w') as file:
        json.dump(results, file, indent=2)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Hyperparameter search for token classification')
    
    parser.add_argument('--model_names', nargs='+', type=str, required=True, help='List of model names to use for training')
    parser.add_argument('--experiment_name', type=str, default='hyperparameter_search', help='Name of the experiment')
    parser.add_argument('--experiment_results_path', type=str, default='results', help='Path to store the experiment results')
    
    parser.add_argument('--n_trials', type=int, default=10, help='Number of trials for hyperparameter search')
    parser.add_argument('--learning_rates', nargs='+', type=float, default=[1e-6, 5e-6, 1e-5, 3e-5, 5e-5])
    parser.add_argument('--train_batch_sizes', nargs='+', type=int, default=[8, 16, 32])
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of gradient accumulation steps')
    parser.add_argument('--weight_decays', nargs='+', type=str, default = [0.01, 0.1, 0.3])

    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing experiment results')
    
    parser.add_argument('--train_data_file', type=str, required=True, help='Path to the data file')
    parser.add_argument('--dev_data_file', type=str, required=True, help='Path to the data file')

    parser.add_argument('--seed', type=int, default=1234)

    args = parser.parse_args()

    main(args)
else:
    # CLI arguments
    from types import SimpleNamespace

    args = SimpleNamespace()

    args.model_names = 'roberta-base'
    args.experiment_name = 'base-model-comparison'
    args.experiment_results_path = 'results'
    
    args.n_trials = 10
    args.learning_rates = [1e-6, 5e-6, 1e-5, 3e-5, 5e-5]
    args.train_batch_sizes = [8, 16, 32]
    args.weight_decays = [0.01, 0.1, 0.3]
    args.gradient_accumulation_steps = 1
    
    args.overwrite = True

    args.train_data_file = 'splits/train.jsonl'
    args.dev_data_file = 'splits/dev.jsonl'

    
    args.seed = 1234
