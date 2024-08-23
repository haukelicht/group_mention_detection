#encoding: UTF-8
from types import SimpleNamespace

args = SimpleNamespace()
args.data_path = '../data/annotation/coded'
args.data_folder_pattern = 'uk-manifestos'
args.data_annotations_folder = 'input-manifests'
args.data_file_format = 'jsonl'

args.output_path = '../data/validation/dhh_dictionary'


import os
import re

import numpy as np
import pandas as pd

import jsonlines

from collections import Counter

from utils.corpus import DoccanoAnnotationsCorpus

from utils.dictionary import glob_to_regex, apply_keywords


subdirs = [os.path.join(args.data_path, d, args.data_annotations_folder) for d in os.listdir(args.data_path) if args.data_folder_pattern in d and 'emotionality' not in d]
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


fp = os.path.join(args.output_path, 'keywords.csv')
keywords_wider = pd.read_csv(fp)


keywords = {c: [glob_to_regex(v) for v in vals if not pd.isna(v)] for c, vals in keywords_wider.T.iterrows()}


for doc in docs:
    res = apply_keywords(doc['text'], keywords)
    doc['label'] = [list(r[2])+['SG'] for r in res ]


# print(docs[9])
# print(docs[9]['label'])
# docs[9]['text'][19:26]


# write to jsonl
fp = os.path.join(args.output_path, 'uk-manifestos_dhh-dictionary_sequence-annotations.jsonl')
#fp = '../data/annotation/dictionary/uk-manifestos/group-mentions-annotation_uk-manifestos_dictionary.jsonl'
with jsonlines.open(fp, mode='w') as writer:
    for doc in docs:
        writer.write(doc)


# read first (we merge the rest to this one)
cat2code = {'O': 0, 'I-SG': 1, 'B-SG': 2}
acorp = DoccanoAnnotationsCorpus(cat2code)
acorp.load_from_jsonlines(fp=fp, annotator_id='dhh_dictionary', verbose=args.verbose)


# ### clean tokens


# toks = set()
# all_chars = Counter()
# for doc in acorp.docs:
#     for tok in doc.tokens:
#         toks.add(tok)
#         all_chars.update([c for c in tok])
# from utils.unicode import CATEGORIES as char_cats
# 
# del char_cats['Ll']
# del char_cats['Lu']
# del char_cats['Nd']
# 
# for k, v in char_cats.items():
#     regx = r'\p{'+k+'}'
#     m = [c for c in all_chars.keys() if regex.match(regx, c)]
#     if len(m) > 0:
#         print(k, end='\t')
#         print(v, end='\t')
#         print(m)


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


# ### write to disk

fp = os.path.join(args.output_path, 'uk-manifestos_dhh-dictionary_token-annotations.jsonl')
acorp.save_as_jsonlines(fp, encoding='utf-8')


