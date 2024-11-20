from types import SimpleNamespace

args = SimpleNamespace()

args.experiment_name = 'dhh_dictionary_expansion'
args.experiment_results_path = '../../../results/validation'
args.overwrite_results = False

args.input_file = '../../../data/exdata/thau2019/thau2019_spans_matched_to_manifesto_texts.jsonl'
args.dictionaries_path = '../../../data/validation/dhh_dictionary/dictionary_expansion'

args.human_labeled_data_file = '../../../data/annotation/exdata/uk-manifestos_thau2019_annotations.jsonl'

args.test_size = 0.1
args.nfolds = 5

args.tokenizer_name = 'roberta-base'

args.seed = 1234


import os
import json
import jsonlines

import numpy as np
import pandas as pd

import re

from copy import deepcopy
from tempfile import NamedTemporaryFile

from utils.corpus import DoccanoAnnotationsCorpus
from utils.dictionary import apply_keywords

import random
from transformers import AutoTokenizer

from utils.classification import (
    prepare_token_labels,
    create_token_classification_dataset, 
    tokenize_and_align_sequence_labels,
)

from sklearn.model_selection import GroupKFold
from utils.evaluation import compute_token_classification_metrics, parse_eval_result

from tqdm.auto import tqdm


results_path = os.path.join(args.experiment_results_path, args.experiment_name)
os.makedirs(results_path, exist_ok=True)


label2id = {
    'O': 0,
    'I-SG': 1,
    'B-SG': 2,
}


# ### load annotations parsed from Thau (2019) data


def parse_record(d):
    return {'id': d['id'], 'tokens': d['tokens'], 'labels': d['annotations']['thau2019'], 'doc': re.sub(r'-\d+-\d+$', '', d['id'])}

def recode_labels(labels):
    obs = np.array(labels, dtype=int)
    zeros = obs == 0
    # inside span
    obs[obs <= 10] = 1
    # begin of span
    obs[obs > 10] = 2
    # outside span
    obs[zeros] = 0
    return obs.tolist()

data = []
with jsonlines.open(args.human_labeled_data_file, 'r') as reader:
    for line in reader:
        doc = parse_record(line)
        doc['labels'] = recode_labels(doc['labels'])
        data.append(doc)

random.Random(args.seed).shuffle(data)

# get document (group) indicators
sentence_docs = [d['doc'] for d in data]
sentence_docs = np.array(sentence_docs)

data = {d['id']: d for d in data}


# to allow for head-to-head comparison with the token classifier, we need to ensure that the tokens are represented in the same way
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True, add_prefix_space=True)

# discard unsure labels
tokens, labels = prepare_token_labels(list(data.values()), discard_unsure=False)
dataset_humans = create_token_classification_dataset([{'tokens': toks, 'labels': labs} for toks, labs in zip(tokens, labels)])
dataset_humans = dataset_humans.map(lambda example: tokenize_and_align_sequence_labels(example, tokenizer=tokenizer), batched=True)


# #### load raw text data


fields = ['id', 'text']
with jsonlines.open(args.input_file) as reader:
    texts = [{k: doc[k] for k in fields} for doc in reader]


# ### apply and evaluate the dictionaries


keywords_files = os.listdir(args.dictionaries_path)


for keywords_file in keywords_files:
    print(f'Processing {keywords_file}')
    k = re.search(r'.+_k(\d+)\.json', keywords_file).group(1)

    # read and prepare keywords
    with open(os.path.join(args.dictionaries_path, keywords_file), 'r') as f:
        keywords = json.load(f)
    keywords = {c: [re.escape(kw) for kw in kws] for c, kws in keywords.items()}

    # apply the keywords
    docs = deepcopy(texts)
    for doc in tqdm(docs, desc=f'  applying the keywords ...'):
        res = apply_keywords(doc['text'], keywords)
        doc['label'] = [list(r[2])+['SG'] for r in res ]

    # convert to token-level annotations
    print('  conveting to token-level annotations ...')
    def parse_record(d):
        return {'id': d.id, 'tokens': d.tokens, 'labels': d.annotations['dictionary']}

    with NamedTemporaryFile(mode='w') as f:
        # write to temporary JSONL file
        with jsonlines.open(f.name, mode='w') as writer:
            for i, doc in enumerate(docs): 
                writer.write(doc)
        
        # convert from character-level to token-level annotations
        acorp = DoccanoAnnotationsCorpus(label2id)
        acorp.load_from_jsonlines(fp=f.name, annotator_id='dictionary', verbose=True)

        # keep only relevant data
        dictionary_annotations = {d.id: parse_record(d) for d in acorp.docs}
        del acorp
    
    # convert to tokenized dataset
    print('  converting to tokenized dataset ...')
    # note: taking the same tokenizer across experiments ensures comparability
    dataset_dictionary = create_token_classification_dataset([{'tokens': d['tokens'], 'labels': dictionary_annotations[id]['labels']} for id, d in data.items()])
    dataset_dictionary = dataset_dictionary.map(lambda example: tokenize_and_align_sequence_labels(example, tokenizer=tokenizer), batched=True)

    eval_res = list()

    # note: here we adopt the same experimental setup as when evaluating the token classifier to ensure comparability
    folds = GroupKFold(n_splits=round(1/args.test_size))
    for i, (_, test_index) in tqdm(enumerate(folds.split(data, groups=sentence_docs)), desc='cross-validating dictionary against human annotations', total=args.nfolds):
        if i == args.nfolds:
            break
        
        res = compute_token_classification_metrics(
            y_true=[dataset_humans['labels'][idx] for idx in test_index], 
            y_pred=[dataset_dictionary['labels'][idx] for idx in test_index], 
            label2id=label2id
        )

        fn = f'expanded_dictionary_k{k}-fold{i+1:02d}-test_results.json'
        fp = os.path.join(results_path, fn)
        if not os.path.exists(fp) or args.overwrite_results:
            with open(fp, 'w') as file:
                json.dump(res, file)
        eval_res.append(res)

    means = pd.DataFrame(eval_res).apply(lambda x: x.mean(), axis=0).to_dict()
    print('evaluation results averaged across folds:')
    print(parse_eval_result(means, types=['SG']))
    print()


# IMPORTANT: doc-level scores not very meaningful because the annotations we have reconstructed from Thau's data only inlcude sentences with min. one social group appeal annotation