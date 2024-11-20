from types import SimpleNamespace

args = SimpleNamespace()

args.data_file = '../../data/annotation/exdata/uk-manifestos_thau2019_annotations.jsonl'

args.model_path = '../../results/classifiers/uk-manifestos_roberta/best_model'
args.batch_size = 32

args.output_file = '../../results/classifiers/uk-manifestos_roberta/evaluation_in_thau2019-manifesto-annotations.tsv'
args.overwrite_output = False

args.types = 'SG,PG,PI,ORG,ISG'


args.types = [t.strip() for t in args.types.split(',')]

import os
import sys
sys.path.append(os.path.abspath('../../code'))

import jsonlines
from collections import Counter
import numpy as np
import pandas as pd

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
)

from utils.classification import (
    create_token_classification_dataset, 
    tokenize_and_align_sequence_labels,
)

# from utils.dataset import create_dataset_with_tokenizer, UnlabeledDataset
from utils.span_metrics import extract_spans


device = 'cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


model = AutoModelForTokenClassification.from_pretrained(args.model_path)
model.to(device);
model.eval();

id2label = model.config.id2label
label2id = model.config.label2id


# ### Prepare data


def parse_record(d):
    return {'id': d['id'], 'tokens': d['tokens'], 'labels': d['annotations']['thau2019']}

def recode_labels(labels):
    obs = np.array(labels, dtype=int)
    zeros = obs == 0
    # inside span
    obs[obs <= 10] = label2id['I-SG']
    # begin of span
    obs[obs > 10] = label2id['B-SG']
    # outside span
    obs[zeros] = 0
    return obs.tolist()

def get_group_category(labels):
    obs = np.array(labels, dtype=int)
    zeros = obs == 0
    # begin of span
    obs[obs > 10] -= 10
    # outside span
    obs[zeros] = 0
    return obs.tolist()

data = []
with jsonlines.open(args.data_file, 'r') as reader:
    for line in reader:
        doc = parse_record(line)
        doc['categories'] = get_group_category(doc['labels'])
        doc['labels'] = recode_labels(doc['labels'])
        data.append(doc)


tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, add_prefix_space=True)

dataset = create_token_classification_dataset(data)
dataset = dataset.map(lambda example: tokenize_and_align_sequence_labels(example, tokenizer=tokenizer), batched=True)
dataset = dataset.remove_columns('tokens')


# ### Compare predictions for spans extracted in Thau (2019) data


from tqdm.auto import tqdm

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
        # remove leading <s> token
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
            recl_micro = np.mean([l != 'O' for l in preds[span[2]:span[3]]])
            cat = data[i+j]['categories'][span[2]]
            tids = [t for tok in span[0] for t in (tok if isinstance(tok, list) else [tok])]
            res.append((i+j, s, tokenizer.decode(tids).strip(), cat, preds[span[2]][2:], recl, recl_micro))


cols = ['sentence_nr', 'span_nr', 'text', 'cat', 'pred_type', 'recall', 'recall_micro']
res_df = pd.DataFrame(res, columns=cols)


thau_cats = {
    1: 'Age/generation',
    2: 'Economic class',
    3: 'Ethnicity/race',
    4: 'Gender',
    5: 'Geography',
    6: 'Health',
    7: 'Nationality',
    8: 'Religion',
    9: 'Other',
   10: 'none',
}
res_df['thau_category'] = res_df['cat'].apply(lambda c: thau_cats[c])


# average recall by type
tmp = res_df.groupby(['thau_category'])[['recall', 'recall_micro']]
tmp.mean(numeric_only=True).join(pd.DataFrame(tmp.size(), columns=['n']))


# get mapping of sentence ID to number
tmp = pd.DataFrame([{'sentence_id': d['id'], 'sentence_nr': i} for i, d in enumerate(data)])
# join
out = pd.merge(tmp, res_df, on='sentence_nr')
del out['sentence_nr']

# save
if not os.path.exists(args.output_file) or args.overwrite_output:
    out.to_csv(args.output_file, sep='\t', index=False)


