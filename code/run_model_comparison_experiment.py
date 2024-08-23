# CLI arguments
from types import SimpleNamespace

args = SimpleNamespace()

args.model_names = 'microsoft/deberta-v3-base,roberta-base,bert-base-cased,distilbert-base-cased'
args.experiment_name = 'uk-manifestos_model-comparison'
args.experiment_results_path = './../results/experiments'

args.data_file = '../data/annotation/labeled/uk-manifestos_all_labeled.jsonl'
args.types = 'SG,PG,PI,ORG,ISG'
args.discard_types = 'unsure'

args.test_size = 0.1
args.dev_size = 0.25
args.seed = 1234
args.nrepeats = 1

args.metric = 'seqeval-SG_f1'


# argument parsing and configuration
args.model_names = [mn.strip() for mn in args.model_names.split(',')]
args.types = [t.strip() for t in args.types.split(',')]

args.discard_types = [t.strip() for t in args.discard_types.split(',')]

scheme = ['O'] + ['I-'+t for t in args.types] + ['B-'+t for t in args.types]
label2id = {l: i for i, l in enumerate(scheme)}
id2label = {i: l for i, l in enumerate(scheme)}

# #### Load libraries


import os
import shutil
import json
from timeit import default_timer as timer
import jsonlines
import random
import numpy as np
import pandas as pd

from utils.classification import (
    prepare_token_labels, 
    create_token_classification_dataset, 
    tokenize_and_align_sequence_labels
)

import torch
#import accelerate
import transformers
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)

from sklearn.model_selection import train_test_split
from utils.evaluation import correct_iob2
from seqeval.metrics import classification_report

import optuna


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# transformers.logging.set_verbosity_error()
# os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
# os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'


device = 'cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
if device == 'cpu':
    raise ValueError('CUDA not available')
device = torch.device(device)
print(f'Using device "{str(device)}"')


set_seed(args.seed)


# slim evaluation metrics function
def compute_token_classification_metrics(p, types):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    label_list = ['O']
    label_list += ['I-'+t for t in types]
    label_list += ['B-'+t for t in types]

    true_predictions = [
        correct_iob2([label_list[p] for (p, l) in zip(prediction, label) if l != -100])
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        correct_iob2([label_list[l] for (p, l) in zip(prediction, label) if l != -100])
        for prediction, label in zip(predictions, labels)
    ]
    
    # Span level (Seqeval)
    result = classification_report(true_labels, true_predictions, output_dict=True)
    
    return {'seqeval-SG_f1': result['SG']['f1-score']}
    
compute_metrics = lambda x: compute_token_classification_metrics(x, types=args.types)


# #### hyperparameter search function


trial_learning_rates = [9e-6, 2e-5, 4e-5]
trial_train_batch_sizes = [8, 16, 32]
trial_weight_decays = [0.01, 0.1, 0.3]

def hp_space(trial):
    return {
        'learning_rate': trial.suggest_categorical('learning_rate', trial_learning_rates),
        'per_device_train_batch_size': trial.suggest_categorical('per_device_train_batch_size', trial_train_batch_sizes),
        'weight_decay': trial.suggest_categorical('weight_decay', trial_weight_decays),
    }

def run_hyperparameter_search(
    model_name,
    label2id,
    metric,
    data,
    train_idxs,
    dev_idxs,
    seed,
    device=device,
    hp_fun=hp_space,
    n_trials=3,
    out_dir='hpsearch'
):
    print('Starting HP search for model', model_name)
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, add_prefix_space=True)
    
    # creat the training split
    train_dataset = create_token_classification_dataset([data[idx] for idx in train_idxs])
    train_dataset = train_dataset.map(lambda example: tokenize_and_align_sequence_labels(example, tokenizer=tokenizer), batched=True)
    train_dataset = train_dataset.remove_columns('tokens')
    
    # creat the validation split
    dev_dataset = create_token_classification_dataset([data[idx] for idx in dev_idxs])
    dev_dataset = dev_dataset.map(lambda example: tokenize_and_align_sequence_labels(example, tokenizer=tokenizer), batched=True)
    dev_dataset = dev_dataset.remove_columns('tokens')
    
    # define train args
    train_args = TrainingArguments(
        # hyperparameters
        num_train_epochs=10,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=32,
        weight_decay=0.01,
        # how to select "best" model
        metric_for_best_model=metric,
        greater_is_better=True,
        load_best_model_at_end=True,
        save_total_limit=2,
        # when to evaluate
        evaluation_strategy='epoch',
        # when to save
        save_strategy='epoch',
        # logging
        logging_strategy='no',
        # where to store results
        output_dir=out_dir,
        report_to='none',
        # misc
        fp16=device == 'cuda:0',
        # reproducibility
        seed=seed,
        data_seed=seed,
        full_determinism=True
    )

    # model init function
    def model_init():
        model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(label2id), label2id=label2id)
        model.to(device);
        return model
    
    # create Trainer
    trainer = Trainer( 
        model_init=model_init,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=3)]
    )
    
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


# #### load and prepare data


# load the data
with jsonlines.open(args.data_file) as reader:
    data = [d for d in reader]

# extract the relevant info
def parse_record(d):
    return {'id': d['id'], 'tokens': d['tokens'], 'labels': d['labels']['BSCModel']}

data = [parse_record(d) for d in data]

# reshuffle
random.Random(args.seed).shuffle(data)


tokens, labels = prepare_token_labels(data, discard_unsure='unsure' in args.discard_types)
for d, t, l in zip(data, tokens, labels):
    d['tokens'], d['labels'] = t, l


data = [{'tokens': toks, 'labels': labs} for toks, labs in zip(tokens, labels)]


# split train data (discard test set to prevent leakage)
tmp, _ = train_test_split(range(len(data)), test_size=args.test_size, random_state=args.seed)

# split into train and val datasets by documents
train_indexes, val_indexes = train_test_split(tmp, test_size=args.dev_size, random_state=args.seed)
print(len(train_indexes), len(val_indexes))


dest = os.path.join(args.experiment_results_path, args.experiment_name)
os.makedirs(dest, exist_ok=True)


results = dict()
for model_name in args.model_names:
    results[model_name] = run_hyperparameter_search(
        # model config
        model_name=model_name,
        label2id=label2id,
        device=device,
        # data
        data=data,
        train_idxs=train_indexes,
        dev_idxs=val_indexes,
        # search parames
        metric=args.metric,
        hp_fun=hp_space,
        n_trials=10,
        # reproducibility
        seed=args.seed,
        out_dir=os.path.join(dest, 'hpsearch')
    )



# finally: write config and results to experiment folder
config = args.__dict__
config['grid'] = {
    'learning_rate': trial_learning_rates,
    'train_batch_size': trial_train_batch_sizes,
    'weight_decay': trial_weight_decays
}

fp = os.path.join(dest, 'config.json')
with open(fp, 'w') as file:
    json.dump(config, file, indent=2)

fp = os.path.join(dest, 'results.json')
with open(fp, 'w') as file:
    json.dump(results, file, indent=2)


