# ### Setup
from types import SimpleNamespace

args = SimpleNamespace()

args.experiment_name = 'uk-manifestos_5x5-crossval_deberta-finetuning'
args.experiment_results_path = './../results/experiments'

args.data_file = '../data/annotation/labeled/uk-manifestos_all_labeled.jsonl'
args.types = 'SG,PG,PI,ORG,ISG'
args.discard_types = 'unsure'

args.nrepeats = 5
args.nfolds = 5
args.seeds = '1234,2345,3456,4567,5678'
args.test_size = 0.1

#args.model_name = 'roberta-base'
#args.epochs=10
#args.learning_rate=2e-05
#args.train_batch_size=8
#args.weight_decay=0.01
#args.eval_batch_size=64

args.model_name = 'microsoft/deberta-v3-base'
args.epochs=10
args.learning_rate = 4e-05
args.train_batch_size = 32
args.eval_batch_size=64
args.weight_decay = 0.3

args.metric = 'seqeval-SG_f1'

# argument parsing and configuration
args.seeds = [int(seed.strip()) for seed in args.seeds.split(',')]
assert len(args.seeds) == args.nrepeats

args.types = [t.strip() for t in args.types.split(',')]
args.discard_types = [t.strip() for t in args.discard_types.split(',')]

scheme = ['O'] + ['I-'+t for t in args.types] + ['B-'+t for t in args.types]
label2id = {l: i for i, l in enumerate(scheme)}
id2label = {i: l for i, l in enumerate(scheme)}
NUM_LABELS = len(label2id)


# #### Load libraries


import os
import json
import jsonlines
from collections import Counter
import re
import random
import numpy as np
import pandas as pd

from utils.classification import (
    prepare_token_labels, 
    create_token_classification_dataset, 
    tokenize_and_align_sequence_labels,
    train_and_test
)

import torch
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    set_seed,
)

from sklearn.model_selection import GroupKFold
from utils.evaluation import parse_token_classifier_prediction_output, compute_token_classification_metrics


device = 'cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'Using device "{str(device)}"')


set_seed(args.seeds[0])


# #### Model and tokenizer


tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, add_prefix_space=True)
assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)


def model_init():
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name, 
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label
    )
    model = model.to(device)
    return model


# custom helper functions
def compute_metrics(p):
    labels, predictions = parse_token_classifier_prediction_output(p)
    return compute_token_classification_metrics(y_true=labels, y_pred=predictions, label2id=label2id)


# ### Prepare data


def parse_record(d):
    return {
        'id': d['id'], 
        'tokens': d['tokens'], 
        'labels': d['labels']['BSCModel'], 
        'doc': re.sub(r'-\d+-\d+$', '', d['metadata']['sentence_id'])
    }

with jsonlines.open(args.data_file) as reader:
    data = [parse_record(d) for d in reader]

# reshuffle
random.Random(args.seeds[0]).shuffle(data)


# get document (group) indicators
sentence_docs = [d['doc'] for d in data]
print('# groups:', len(Counter(sentence_docs)))
sentence_docs = np.array(sentence_docs)


tokens, labels = prepare_token_labels(data, discard_unsure='unsure' in args.discard_types)
for d, t, l in zip(data, tokens, labels):
    d['tokens'], d['labels'] = t, l


data = [{'tokens': toks, 'labels': labs} for toks, labs in zip(tokens, labels)]


# ### Run Experiment


# to keep track of train/val/test sizes
split_sizes = list()

repeats = GroupKFold(n_splits=round(1/args.test_size))
for i, (train_idxs, test_idxs) in enumerate(repeats.split(data, groups=sentence_docs)):
    if i == args.nrepeats:
        break
    
    # get seed
    seed = args.seeds[i]
    # set seed
    set_seed(seed)
    rng = np.random.RandomState(seed)
    
    # creat the test split
    test_dataset = create_token_classification_dataset([data[idx] for idx in test_idxs])
    test_dataset = test_dataset.map(lambda example: tokenize_and_align_sequence_labels(example, tokenizer=tokenizer), batched=True)
    test_dataset = test_dataset.remove_columns('tokens')
    
    folds = GroupKFold(n_splits=args.nfolds)
    for j, (trn, dev) in enumerate(folds.split(train_idxs, groups=sentence_docs[train_idxs])):
        run_id = str(f'rep{i+1:02d}-fold{j+1:02d}')
        
        split_sizes.append((run_id, len(trn), len(dev), len(test_idxs)))
        
        print('-'*53)
        print(f'{run_id}: # train: {len(trn)}; # dev: {len(dev)}; # test: {len(test_idxs)}\n')
        
        # shuffle train and val set indexes
        rng.shuffle(trn)
        rng.shuffle(dev)
        
        # creat the training split
        train_dataset = create_token_classification_dataset([data[idx] for idx in trn])
        train_dataset = train_dataset.map(lambda example: tokenize_and_align_sequence_labels(example, tokenizer=tokenizer), batched=True)
        train_dataset = train_dataset.remove_columns('tokens')
        
        # creat the validation split
        dev_dataset = create_token_classification_dataset([data[idx] for idx in dev])
        dev_dataset = dev_dataset.map(lambda example: tokenize_and_align_sequence_labels(example, tokenizer=tokenizer), batched=True)
        dev_dataset = dev_dataset.remove_columns('tokens')

        # train & evaluate
        model, model_path, test_res = train_and_test(
            experiment_name=args.experiment_name,
            experiment_results_path=args.experiment_results_path,
            run_id=run_id,
            model_init=model_init,
            tokenizer=tokenizer,
            data_collator=DataCollatorForTokenClassification(tokenizer),
            train_dat=train_dataset,
            dev_dat=dev_dataset,
            test_dat=test_dataset,
            compute_metrics=compute_metrics,
            metric=args.metric,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            weight_decay=args.weight_decay,
            early_stopping=True,
            seed=seed,
            save_best_model=False,
            save_tokenizer=False
        )


# finally: write config and split_sizes to experiment folder
dest = os.path.join(args.experiment_results_path, args.experiment_name)
os.makedirs(dest, exist_ok=True)

fp = os.path.join(dest, 'config.json')
with open(fp, 'w') as file:
    json.dump(args.__dict__, file, indent=2)

split_sizes = pd.DataFrame(split_sizes, columns=['run_id', 'n_train', 'n_val', 'n_test'])
fp = os.path.join(dest, 'split_sizes.tsv')
split_sizes.to_csv(fp, sep='\t', index=False)


