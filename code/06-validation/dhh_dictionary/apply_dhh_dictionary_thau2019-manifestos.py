from types import SimpleNamespace

args = SimpleNamespace()
args.input_file = '../../../data/exdata/thau2019/thau2019_spans_matched_to_manifesto_texts.jsonl'
args.keywords_file = '../../../data/exdata/dhh_dictionary/keywords.csv'
args.output_path = '../../../data/validation/dhh_dictionary'
args.overwrite_output = False


import os
import jsonlines

import pandas as pd

from utils.corpus import DoccanoAnnotationsCorpus

from utils.dictionary import glob_to_regex, apply_keywords


# #### load data


fields = ['id', 'text']
with jsonlines.open(args.input_file) as reader:
    docs = [{k: doc[k] for k in fields} for doc in reader]


# ### apply dictionary


keywords_wider = pd.read_csv(args.keywords_file)

keywords = {c: [glob_to_regex(v) for v in vals if not pd.isna(v)] for c, vals in keywords_wider.T.iterrows()}


for doc in docs:
    res = apply_keywords(doc['text'], keywords)
    doc['label'] = [list(r[2])+['SG'] for r in res ]


# write sequence annotations to jsonl
fp = os.path.join(args.output_path, 'thau2019-manifestos-spans_dhh-dictionary_sequence-annotations.jsonl')
if not os.path.exists(fp) or args.overwrite_output:
    with jsonlines.open(fp, mode='w') as writer:
        for doc in docs: writer.write(doc)


# #### convert sequence to token-level annotations


# read first (we merge the rest to this one)
cat2code = {'O': 0, 'I-SG': 1, 'B-SG': 2}
acorp = DoccanoAnnotationsCorpus(cat2code)
acorp.load_from_jsonlines(fp=fp, annotator_id='dhh_dictionary', verbose=False)


# write to disk
fp = os.path.join(args.output_path, 'thau2019-manifestos-spans_dhh-dictionary_token-annotations.jsonl')
if not os.path.exists(fp) or args.overwrite_output:
    acorp.save_as_jsonlines(fp, encoding='utf-8')


