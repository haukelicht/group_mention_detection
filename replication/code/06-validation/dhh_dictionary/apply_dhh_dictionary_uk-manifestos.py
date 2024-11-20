#encoding: UTF-8
from types import SimpleNamespace

args = SimpleNamespace()
args.data_path = '../../../data/annotation/annotations'
args.data_folder_pattern = 'uk-manifestos'

args.keywords_file = '../../../data/exdata/dhh_dictionary/keywords.csv'

args.output_path = '../../../data/validation/dhh_dictionary'
args.overwrite_output = False

import os
import re

import numpy as np
import pandas as pd

import jsonlines

from collections import Counter

from utils.corpus import DoccanoAnnotationsCorpus

from utils.dictionary import glob_to_regex, apply_keywords

# read sentence IDs in samples distributed for annotations
subdirs = [os.path.join(args.data_path, d, 'input-manifests') for d in os.listdir(args.data_path) if d.startswith(args.data_folder_pattern)]
fps = [os.path.join(d, f) for d in subdirs for f in os.listdir(d) if f.endswith('.manifest')]
fps.sort()

docs = {}
fields = {'id': 'id', 'source': 'text'}
for fp in fps:
    with jsonlines.open(fp) as reader:
        for line in reader:
            if line['id'] in docs:
                continue
            docs[line['id']] = {n: line[o] for o, n in fields.items()}



# fix minor glitch in the data
docs['c60c93194a1f4b2d3b8a725c1ea05734']['text'] = docs['c60c93194a1f4b2d3b8a725c1ea05734']['text'].removesuffix('</p>')


docs = list(docs.values())


# ### apply dictionary

keywords_wider = pd.read_csv(args.keywords_file)

keywords = {c: [glob_to_regex(v) for v in vals if not pd.isna(v)] for c, vals in keywords_wider.T.iterrows()}

for doc in docs:
    res = apply_keywords(doc['text'], keywords)
    doc['label'] = [list(r[2])+['SG'] for r in res ]

# write to jsonl
fp = os.path.join(args.output_path, 'uk-manifestos_dhh-dictionary_sequence-annotations.jsonl')
if not os.path.exists(fp) or args.overwrite_output:
    with jsonlines.open(fp, mode='w') as writer:
        for doc in docs:
            writer.write(doc)


# read first (we merge the rest to this one)
cat2code = {'O': 0, 'I-SG': 1, 'B-SG': 2}
acorp = DoccanoAnnotationsCorpus(cat2code)
acorp.load_from_jsonlines(fp=fp, annotator_id='dhh_dictionary', verbose=False)

# ### clean tokens

replace_chars = {    
    # Cc
    '\x91': '"',
    '\x92': '"',
    # Co
    u'\uf02f': '',
    # No
    '½': '1/2',
    # Po
    '·': '',
    # Sk
    '\^': ' ',
    # Sm
    '¬': '-'
}

for p, r in replace_chars.items():
    p = re.compile(p, re.UNICODE)
    for i in range(acorp.ndocs):
        acorp.docs[i].text = re.sub(p, r, acorp.docs[i].text)
        acorp.docs[i].tokens = [re.sub(p, r, tok) for tok in acorp.docs[i].tokens]

# print(acorp.docs[5])

# ### write to disk
fp = os.path.join(args.output_path, 'uk-manifestos_dhh-dictionary_token-annotations.jsonl')
if not os.path.exists(fp) or args.overwrite_output:
    acorp.save_as_jsonlines(fp, encoding='utf-8')


