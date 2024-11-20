from types import SimpleNamespace

args = SimpleNamespace()


args.experiment_name = 'uk-manifestos_roberta'
args.experiment_results_path = '../../results/classifiers'

args.data_file = '../../data/annotation/labeled/uk-manifestos_all_labeled.jsonl'
args.types = 'SG,PG,PI,ORG,ISG'
args.discard_types = 'unsure'

args.test_size = 0.1
args.dev_size = 0.1
args.seed = 1234

args.model_name = 'roberta-base'
args.epochs=10
args.learning_rate=2e-5
args.train_batch_size=8
args.eval_batch_size=64
args.weight_decay=0.01

# args.model_name = 'microsoft/deberta-v3-base'
# args.epochs=10
# args.learning_rate = 9e-06
# args.train_batch_size = 16
# args.eval_batch_size=64
# args.weight_decay = 0.01

args.metric = 'seqeval-SG_f1'


# argument parsing and configuration
args.types = [t.strip() for t in args.types.split(',')]
args.discard_types = [t.strip() for t in args.discard_types.split(',')]

scheme = ['O'] + ['I-'+t for t in args.types] + ['B-'+t for t in args.types]
label2id = {l: i for i, l in enumerate(scheme)}
id2label = {i: l for i, l in enumerate(scheme)}

# #### Load libraries


import os
import json
import jsonlines

import random
import numpy as np
import pandas as pd
from datasets import DatasetDict

import torch
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    set_seed
)

from utils.classification import (
    prepare_token_labels, 
    split_data,
    create_token_classification_dataset, 
    tokenize_and_align_sequence_labels,
    train_and_test
)

from utils.evaluation import parse_token_classifier_prediction_output, compute_token_classification_metrics, parse_eval_result


device = 'cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print('using device:', device)


set_seed(args.seed)


# #### tokenizer and model


tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, add_prefix_space=True)
assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)


def model_init():
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name, 
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label
    )
    model.to(device);
    return model


# custom helper functions
def compute_metrics(p):
    labels, predictions = parse_token_classifier_prediction_output(p)
    return compute_token_classification_metrics(y_true=labels, y_pred=predictions, label2id=label2id)


# #### load and prepare data


with jsonlines.open(args.data_file) as reader:
    data = [d for d in reader]

def parse_record(d):
    return {'id': d['id'], 'tokens': d['tokens'], 'labels': d['labels']['BSCModel']}

data = [parse_record(d) for d in data]


# reshuffle
random.Random(args.seed).shuffle(data)


tokens, labels = prepare_token_labels(data, discard_unsure='unsure' in args.discard_types)
for d, t, l in zip(data, tokens, labels):
    d['tokens'], d['labels'] = t, l


data_splits = split_data(data, test_size=args.test_size, dev_size=args.dev_size, seed=args.seed, return_dict=True)


dataset = DatasetDict({split: create_token_classification_dataset(data) for split, data in data_splits.items()})
dataset = dataset.map(lambda example: tokenize_and_align_sequence_labels(example, tokenizer=tokenizer), batched=True)
dataset = dataset.remove_columns('tokens')


# #### prepare training


print('Data split sizes:', dataset.num_rows)


# #### Train


# train & test
model, model_path, test_res = train_and_test(
    experiment_name=args.experiment_name,
    experiment_results_path=args.experiment_results_path,
    run_id=None,
    model_init=model_init,
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    train_dat=dataset['train'],
    dev_dat=dataset['dev'],
    test_dat=dataset['test'] if 'test' in dataset.keys() else None,
    compute_metrics=compute_metrics,
    metric=args.metric,
    epochs=args.epochs,
    learning_rate=args.learning_rate,
    train_batch_size=args.train_batch_size,
    eval_batch_size=args.eval_batch_size,
    weight_decay=args.weight_decay,
    early_stopping=True,
    seed=args.seed,
)

if test_res is not None:
    # {m: v for m, v in test_res.items() if 'f1' in m and '-SG' in m}
    print('Test set results')
    print(parse_eval_result(test_res, types=['SG'], remove_prefix='test_'))
else:
    dev_res = model.evaluate(dataset['dev'], metric_key_prefix='dev')
    print('Dev set results')
    print(parse_eval_result(dev_res, types=['SG'], remove_prefix='dev_'))

# finally: write config and split_sizes to experiment folder
dest = os.path.join(args.experiment_results_path, args.experiment_name)
os.makedirs(dest, exist_ok=True)

fp = os.path.join(dest, 'config.json')
with open(fp, 'w') as file:
    json.dump(args.__dict__, file, indent=2)
