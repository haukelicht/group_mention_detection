
from types import SimpleNamespace

args = SimpleNamespace()

args.experiment_name = 'uk-manifestos_roberta'
args.experiment_results_path = '../../results/classifiers'
args.overwrite_results = False

args.batch_size = 32

args.data_file = '../../data/annotation/labeled/uk-manifestos_all_labeled.jsonl'
args.types = 'SG,PG,PI,ORG,ISG'
args.discard_types = 'unsure'

args.test_size = 0.1
args.dev_size = 0.1
args.seed = 1234


args.types = [t.strip() for t in args.types.split(',')]
args.discard_types = [t.strip() for t in args.discard_types.split(',')]


import os
import sys
sys.path.append(os.path.abspath('../../code'))

import jsonlines
import random

import numpy as np
import pandas as pd

import torch
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
)

from utils.span_metrics import extract_spans

from tqdm.auto import tqdm


results_path = os.path.join(args.experiment_results_path, args.experiment_name)
model_path = os.path.join(args.experiment_results_path , args.experiment_name, 'best_model')


device = 'cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


set_seed(args.seed)


# ### Load the model


model = AutoModelForTokenClassification.from_pretrained(model_path)
model.to(device);
model.eval();

id2label = model.config.id2label
label2id = model.config.label2id


# ### Get the test set


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


tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, add_prefix_space=True)

dataset = create_token_classification_dataset(data_splits['test'])
dataset = dataset.map(lambda example: tokenize_and_align_sequence_labels(example, tokenizer=tokenizer), batched=True)
dataset = dataset.remove_columns('tokens')


# ### Error analysis

data_collator = DataCollatorForTokenClassification(tokenizer, padding=True)

res = list()

n = len(dataset)
for i in tqdm(range(0, n, args.batch_size)):
    ds = dataset.select(range(i, min(i+args.batch_size, n)))
    batch = data_collator(ds)
    attention_masks = batch.attention_mask
    with torch.no_grad():
        predictions = model(**batch.to(device)).logits.cpu().argmax(dim=-1).numpy()

    for j, (d, p, m) in enumerate(zip(ds, predictions, attention_masks)):
        # get attention mask
        amask = m==1
        # remove trailing </s> token
        amask[amask.sum()-1] = 0
        # remove heading <s> token
        amask[0] = 0

        l = len(d['input_ids'])-1
        toks = list()
        labs = list()
        preds = list()
        for lab, tok, pred in zip(d['labels'][1:l], d['input_ids'][1:l], p[amask]):
            if lab == -100:
                toks[-1].append(tok)
            else:
                labs.append(lab)
                toks.append([tok])
                preds.append(pred)
        labs = [id2label[l] for l in labs]
        preds = [id2label[l] for l in preds]

        spans = extract_spans(labs, tokens=toks)
        # compute span-wise recall
        for s, span in enumerate(spans):
            recl = np.mean([l[2:] == span[1] for l in preds[span[2]:span[3]]])
            tids = [t for tok in span[0] for t in (tok if isinstance(tok, list) else [tok])]
            res.append((i+j, s, tokenizer.decode(tids).strip(), span[1], recl))

res_df = pd.DataFrame(res, columns=['sentence_nr', 'span_nr', 'text', 'type', 'recall'])

fp = os.path.join(results_path, 'evaluation_in_testset.tsv')
if not os.path.exists(fp) or args.overwrite_results:
    res_df.to_csv(fp, sep='\t', index = False)


# share completely wrong by group type
res_df['completely_wrong'] = res_df.recall == 0.0
tmp = res_df.groupby(['type'])[['recall', 'completely_wrong']]
tmp.mean(numeric_only=True).loc[args.types].join(pd.DataFrame(tmp.size(), columns=['n']))


res_df['span_length'] = res_df.text.apply(lambda t: t.count(' ') + 1)
# print(res_df.span_length.max())
# print(np.quantile(res_df.span_length.values, q =[.1, .25, .5, .75, .9, .95], ))


res_df['span_length_bins'] = pd.cut(res_df.span_length.values, bins=[0, 1, 2, 4, 16])
tmp = res_df.groupby(['type', 'span_length_bins'], observed=False).agg({'recall': 'mean', 'completely_wrong': 'mean', 'span_nr': 'count'})
tmp = tmp.loc['SG']
tmp.rename(columns={'recall': 'mean_recall', 'completely_wrong': 'share_completely_wrong', 'span_nr': 'n'}, inplace=True)
fp = os.path.join(results_path, 'testset_recall_by_span_length_bins.tsv')
if not os.path.exists(fp) or args.overwrite_results:
    tmp.to_csv(fp, sep='\t')


train_spans = {t: [] for t in args.types}
for doc in data_splits['train']:
    spans = extract_spans(labels=[id2label[l] for l in doc['labels']], tokens=doc['tokens'])
    for span in spans:
        mention = tokenizer.decode(tokenizer(span[0], is_split_into_words=True, add_special_tokens=False)['input_ids']).strip()
        train_spans[span[1]].append(mention)

sg_train_spans = train_spans['SG']
unique_train_spans = set(sg_train_spans)
# print('# spans:', len(sg_train_spans))
# print('# unique spans:', len(unique_train_spans))
# print('# spans / # unique spans', len(sg_train_spans)/len(unique_train_spans) )

res_df['seen'] = res_df.text.apply(lambda s: s in unique_train_spans)
tmp = res_df.groupby(['type', 'seen'])[['recall']]
tmp = tmp.mean().loc[['SG']].join(pd.DataFrame(tmp.size(), columns=['n']))

fp = os.path.join(results_path, 'testset_recall_generalization.tsv')
if not os.path.exists(fp) or args.overwrite_results:
    tmp.to_csv(fp, sep='\t')

