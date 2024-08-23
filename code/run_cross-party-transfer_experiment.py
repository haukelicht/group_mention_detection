from types import SimpleNamespace

args = SimpleNamespace()

args.experiment_name = 'uk-manifestos_cross-party-transfer_roberta-finetuning'
args.experiment_results_path = './../results/experiments'

args.data_file = '../data/annotation/labeled/uk-manifestos_all_labeled.jsonl'
args.types = 'SG,PG,PI,ORG,ISG'
args.discard_types = 'unsure'

args.test_size = 0.1
args.seeds = '1234,2345,3456,4567,5678'
args.nrepeats = 5
args.nchunks = 5
args.source_domain_test_size = 0.1
args.target_domain_test_size = 0.2

args.source_domain_key = 'party'
args.source_domain_values = 'conservatives,labour,51620,51320'

args.model_name = 'roberta-base'
args.metric = 'seqeval-SG_f1'
args.epochs=10
args.learning_rate=2e-5
args.train_batch_size=8
args.eval_batch_size=64
args.weight_decay=0.01


# argument parsing and configuration
args.seeds = [int(seed.strip()) for seed in args.seeds.split(',')]
assert len(args.seeds) == args.nrepeats

args.types = [t.strip() for t in args.types.split(',')]
args.discard_types = [t.strip() for t in args.discard_types.split(',')]

scheme = ['O'] + ['I-'+t for t in args.types] + ['B-'+t for t in args.types]
label2id = {l: i for i, l in enumerate(scheme)}
id2label = {i: l for i, l in enumerate(scheme)}

args.source_domain_values = [v.strip() for v in args.source_domain_values.split(',')]


# #### Load libraries

import os
import shutil
import json
import jsonlines
from collections import Counter
import re
import random
import numpy as np
import pandas as pd
import math

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
device = torch.device(device)
print(f'Using device "{str(device)}"')


set_seed(args.seeds[0])


# #### Model and tokenizer


tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, add_prefix_space=True)
assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)


def model_init(model_name_or_path: str=args.model_name):
    model = AutoModelForTokenClassification.from_pretrained(
        model_name_or_path, 
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


# #### load and preprocess data


def parse_record(d):
    dat = {
        'id': d['id'], 
        'tokens': d['tokens'], 
        'labels': d['labels']['BSCModel'],
        'doc': re.sub(r'-\d+-\d+$', '', d['metadata']['sentence_id'])
    }
    dat[args.source_domain_key] = d['metadata'][args.source_domain_key]
    return dat

with jsonlines.open(args.data_file) as reader:
    data = [parse_record(d) for d in reader]


source_data = [d for d in data if d[args.source_domain_key] in args.source_domain_values]
target_data = [d for d in data if d[args.source_domain_key] not in args.source_domain_values]
print(len(data), len(source_data), len(target_data))
del data


# shuffle data (reproducably)
random.Random(args.seeds[0]).shuffle(source_data)
random.Random(args.seeds[0]).shuffle(target_data)


# get document (group) indicators
source_sentence_docs = [d['doc'] for d in source_data]
target_sentence_docs = [d['doc'] for d in target_data]
print('# groups (source):', len(Counter(source_sentence_docs)))
print('# groups (target):', len(Counter(target_sentence_docs)))

source_sentence_docs = np.array(source_sentence_docs)
target_sentence_docs = np.array(target_sentence_docs)


tokens, labels = prepare_token_labels(source_data, discard_unsure='unsure' in args.discard_types)
for d, t, l in zip(source_data, tokens, labels):
    d['tokens'], d['labels'] = t, l

source_data = [{'tokens': toks, 'labels': labs} for toks, labs in zip(tokens, labels)]


tokens, labels = prepare_token_labels(target_data, discard_unsure='unsure' in args.discard_types)
for d, t, l in zip(target_data, tokens, labels):
    d['tokens'], d['labels'] = t, l

target_data = [{'tokens': toks, 'labels': labs} for toks, labs in zip(tokens, labels)]


# ## Run Experiment


split_sizes = list()

repeats = GroupKFold(n_splits=math.ceil(1/args.source_domain_test_size))
for i, (train_idxs, test_idxs) in enumerate(repeats.split(source_data, groups=source_sentence_docs)):
    if i == args.nrepeats: 
        break
    
    # get and set seed
    seed = args.seeds[i]
    set_seed(seed)
    
    # create source domain test split
    src_test_dataset = create_token_classification_dataset([source_data[idx] for idx in test_idxs])
    src_test_dataset = src_test_dataset.map(lambda example: tokenize_and_align_sequence_labels(example, tokenizer=tokenizer), batched=True)
    src_test_dataset = src_test_dataset.remove_columns('tokens')
    
    # create source domain train/dev split
    gkf = GroupKFold(n_splits=5)
    src_trn, src_dev = next(gkf.split(train_idxs, groups=source_sentence_docs[train_idxs]))
    
    src_train_dataset = create_token_classification_dataset([source_data[idx] for idx in src_trn])
    src_train_dataset = src_train_dataset.map(lambda example: tokenize_and_align_sequence_labels(example, tokenizer=tokenizer), batched=True)
    src_train_dataset = src_train_dataset.remove_columns('tokens')

    src_dev_dataset = create_token_classification_dataset([source_data[idx] for idx in src_dev])
    src_dev_dataset = src_dev_dataset.map(lambda example: tokenize_and_align_sequence_labels(example, tokenizer=tokenizer), batched=True)
    src_dev_dataset = src_dev_dataset.remove_columns('tokens')

    # create target domain train/test split
    gkf = GroupKFold(n_splits=math.ceil(1/args.target_domain_test_size))
    tgt_trn_idxs, tgt_test_idxs = next(gkf.split(target_data, groups=target_sentence_docs))
    
    tgt_test_dataset = create_token_classification_dataset([target_data[idx] for idx in tgt_test_idxs])
    tgt_test_dataset = tgt_test_dataset.map(lambda example: tokenize_and_align_sequence_labels(example, tokenizer=tokenizer), batched=True)
    tgt_test_dataset = tgt_test_dataset.remove_columns('tokens')
    
    print('-'*53)
    run_id = str(f'rep{i+1:02d}-baseline')
    print(f'{run_id}: # train: {len(src_trn)}; # dev: {len(src_dev)}; # test: {len(test_idxs)}')
    split_sizes.append( [run_id, len(src_trn), len(src_dev), len(test_idxs)])
    
    # train & test baseline
    _, model_path, _ = train_and_test(
        experiment_name=args.experiment_name,
        experiment_results_path=args.experiment_results_path,
        run_id=run_id,
        model_init=lambda: model_init(args.model_name),
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        train_dat=src_train_dataset,
        dev_dat=src_dev_dataset,
        test_dat=src_test_dataset,
        extra_test_dat=tgt_test_dataset,
        append_test_results=True,
        compute_metrics=compute_metrics,
        metric=args.metric,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        weight_decay=args.weight_decay,
        early_stopping=True,
        seed=seed
    )
    last_checkpoints = [model_path]
    
    for j, chunk in enumerate(np.array_split(tgt_trn_idxs, args.nchunks)):
        
        # create train/val splits
        gkf = GroupKFold(n_splits=math.ceil(1/args.target_domain_test_size))
        trn, dev = next(gkf.split(chunk, groups=target_sentence_docs[chunk]))
        
        run_id = str(f'rep{i+1:02d}-adapt{j+1:02d}')
        print(f'{run_id}: # train: {len(trn)}; # dev: {len(dev)}; # test: {len(tgt_test_dataset)}')
        split_sizes.append( [run_id, len(trn), len(dev), len(tgt_test_idxs)])
        
        tgt_train_dataset = create_token_classification_dataset([target_data[idx] for idx in trn])
        tgt_train_dataset = tgt_train_dataset.map(lambda example: tokenize_and_align_sequence_labels(example, tokenizer=tokenizer), batched=True)
        tgt_train_dataset = tgt_train_dataset.remove_columns('tokens')

        tgt_dev_dataset = create_token_classification_dataset([target_data[idx] for idx in dev])
        tgt_dev_dataset = tgt_dev_dataset.map(lambda example: tokenize_and_align_sequence_labels(example, tokenizer=tokenizer), batched=True)
        tgt_dev_dataset = tgt_dev_dataset.remove_columns('tokens')

        # continue training from best checkpoint of last run
        last_checkpoint = last_checkpoints[-1]
        _, model_path, _ = train_and_test(
            experiment_name=args.experiment_name,
            experiment_results_path=args.experiment_results_path,
            run_id=run_id,
            model_init=lambda: model_init(last_checkpoint),
            tokenizer=tokenizer,
            data_collator=DataCollatorForTokenClassification(tokenizer),
            train_dat=tgt_train_dataset,
            dev_dat=tgt_dev_dataset,
            test_dat=src_test_dataset,
            extra_test_dat=tgt_test_dataset,
            append_test_results=True,
            compute_metrics=compute_metrics,
            metric=args.metric,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            weight_decay=args.weight_decay,
            early_stopping=True,
            seed=seed
        )
        last_checkpoints.append(model_path)
        # free up disc space
        remove = last_checkpoints.pop(0)
        shutil.rmtree(remove)


# finally: write config and split_sizes to experiment folder
fp = os.path.join(args.experiment_results_path, args.experiment_name, 'config.json')
with open(fp, 'w') as file:
    json.dump(args.__dict__, file, indent=2)

split_sizes = pd.DataFrame(split_sizes, columns=['run_id', 'n_train', 'n_dev', 'n_test'])
fp = os.path.join(args.experiment_results_path, args.experiment_name, 'split_sizes.tsv')
split_sizes.to_csv(fp, sep='\t', index=False)


