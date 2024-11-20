
# ## Evaluate Dolinksy-Huber-Horne dictionary against test set examples in our human-annotated UK manifesto sentences
from types import SimpleNamespace

args = SimpleNamespace()

args.experiment_name = 'dhh_dictionary_eval_uk-manifestos'
args.experiment_results_path = '../../../results/validation'
args.overwrite_results = False

args.human_labeled_data_file = '../../../data/annotation/labeled/uk-manifestos_all_labeled.jsonl'
args.dictionary_labeled_data_file = '../../../data/validation/dhh_dictionary/uk-manifestos_dhh-dictionary_token-annotations.jsonl'

args.test_size = 0.1
args.nfolds = 5

args.tokenizer_name = 'roberta-base'

args.seed = 1234


# #### Load libraries


import os

import re
import random

import numpy as np
import pandas as pd

import json
import jsonlines

from transformers import AutoTokenizer, PreTrainedTokenizerFast

from utils.classification import (
    prepare_token_labels,
    create_token_classification_dataset, 
    tokenize_and_align_sequence_labels,
)

from sklearn.model_selection import GroupKFold
from utils.evaluation import compute_token_classification_metrics, parse_eval_result


types = ['SG', 'PG', 'PI', 'ORG', 'ISG']
scheme = ['O'] + ['I-'+t for t in types] + ['B-'+t for t in types]
label2id = {l: i for i, l in enumerate(scheme)}
id2label = {i: l for i, l in enumerate(scheme)}


results_path = os.path.join(args.experiment_results_path, args.experiment_name)
os.makedirs(results_path, exist_ok=True)


# #### load the data

with jsonlines.open(args.human_labeled_data_file) as reader:
    data = [d for d in reader]

def parse_record(d):
    return {'id': d['id'], 'tokens': d['tokens'], 'labels': d['labels']['BSCModel'], 'doc': re.sub(r'-\d+-\d+$', '', d['metadata']['sentence_id'])}

data = [parse_record(d) for d in data]

random.Random(args.seed).shuffle(data)

# get document (group) indicators
sentence_docs = [d['doc'] for d in data]
sentence_docs = np.array(sentence_docs)

data = {d['id']: d for d in data}


# #### load dictionary-labeled data

def parse_record(d):
    return {'id': d['id'], 'tokens': d['tokens'], 'labels': d['annotations']['dhh_dictionary']}

with jsonlines.open(args.dictionary_labeled_data_file) as reader:
    dictionary_annotations = {d['id']: parse_record(d) for d in reader}


# ensure compatibility of data
for id in data.keys():
    if id not in dictionary_annotations:
        print(f'No annotations for {id}')
    if len(data[id]['tokens']) != len(dictionary_annotations[id]['tokens']):
        print(f'Tokens length mismatch for {id}')
    if len(data[id]['labels']) != len(dictionary_annotations[id]['labels']):
        print(f'Labels length mismatch for {id}')


# #### prepare evaluation


# to allow for head-to-head comparison with the token classifier, we need to ensure that the tokens are represented in the same way
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True, add_prefix_space=True)
assert isinstance(tokenizer, PreTrainedTokenizerFast)

# discard unsure labels
tokens, labels = prepare_token_labels(list(data.values()), discard_unsure=True)
dataset_humans = create_token_classification_dataset([{'tokens': toks, 'labels': labs} for toks, labs in zip(tokens, labels)])
dataset_humans = dataset_humans.map(lambda example: tokenize_and_align_sequence_labels(example, tokenizer=tokenizer), batched=True)

# note: take the tokens from the human-labeled data to ensure that word indexes are aligned across datasets
dataset_dictionary = create_token_classification_dataset([{'tokens': d['tokens'], 'labels': dictionary_annotations[id]['labels']} for id, d in data.items()])
dataset_dictionary = dataset_dictionary.map(lambda example: tokenize_and_align_sequence_labels(example, tokenizer=tokenizer), batched=True)


# note: to compare the dictionary classifications with labels induced from human annotations, 
#       we need to align the label categories
def recode_labels(labels, map):
    if len(labels) == 0:
        return labels
    if isinstance(labels[0], int):
        return [map[l] for l in labels]
    return [[map[l] for l in labs] for labs in labels]


recode_map = dict(zip(range(len(scheme)), np.zeros(len(scheme), dtype=int)))
recode_map[-100] = -100
recode_map[label2id['I-SG']] = 1
recode_map[label2id['B-SG']] = 2


labels_human = recode_labels(dataset_humans['labels'], recode_map)
labels_dictionary = dataset_dictionary['labels']


# note: this evaluation scheme mirrors our 5x5 cross-validation setup, making results directly comparable
eval_res = list()

folds = GroupKFold(n_splits=round(1/args.test_size))
for i, (_, test_index) in enumerate(folds.split(data, groups=sentence_docs)):
    if i == args.nfolds:
        break
    
    res = compute_token_classification_metrics(
        y_true=[labels_human[idx] for idx in test_index], 
        y_pred=[labels_dictionary[idx] for idx in test_index], 
        label2id=label2id,
    )

    fn = f'fold{i+1:02d}-test_results.json'
    fp = os.path.join(results_path, fn)
    if not os.path.exists(fp) or args.overwrite_results:
        with open(fp, 'w') as file:
            json.dump(res, file)
    eval_res.append(res)


means = pd.DataFrame(eval_res).apply(lambda x: x.mean(), axis=0).to_dict()
print(parse_eval_result(means, types=['SG']))


