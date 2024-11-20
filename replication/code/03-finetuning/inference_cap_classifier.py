from types import SimpleNamespace

args = SimpleNamespace()

args.input_file = '../../data/corpora/uk-manifesto_sentences.tsv'
args.id_col = 'sentence_id'
args.text_col = 'text'
args.metadata_cols = ['manifesto_id', 'party', 'date']

args.model_path = '../../results/classifiers/cap_classifier/best_model/'
args.batch_size = 64

args.output_file = '../../data/labeled/uk-manifesto_sentences_lab+con_cap_labeled.tsv'
args.label_col_name = 'majortopic_recoded_label'
args.overwrite = True

args.test = False
args.verbose = True


import os
import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

device = 'cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
device = torch.device(device)
if args.verbose: print('using device:', str(device))


tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(args.model_path)

# construct the pipeline
classifier = pipeline(
    task='text-classification', 
    model=model, 
    tokenizer=tokenizer,
    device=device,
    batch_size=args.batch_size
)


# read the input file
sep = None
if args.input_file.endswith('.tsv') or args.input_file.endswith('.tab'):
    sep = '\t'
elif args.input_file.endswith('.csv'):
    sep = ','
else:
    raise ValueError('input file must be a tab-separated (.tsv, .tab) or comma-separated (.csv) file')

df = pd.read_csv(args.input_file, sep=sep)

# hard-code years covered by human-labeled CAP training data 
years = [1983, 1987, 1992, 1997, 2001, 2005, 2010, 2015]

# subset to relevant parties
df = df[df.partyname.isin(['Labour Party', 'Conservative Party'])]
# subset to relevant years
df = df[df.date.isin(years)]

if args.test:
    n_ = args.batch_size*10
    if n_ < len(df):
        df = df.sample(n=n_, random_state=42)


cols = [args.id_col, args.text_col] + args.metadata_cols
df = df[cols]


if os.path.exists(args.output_file) and not args.overwrite:
    raise ValueError('output file already exists. Set --overwrite to overwrite it.')
elif not os.path.exists(os.path.dirname(args.output_file)):
    os.makedirs(os.path.dirname(args.output_file))

sep = None
if args.output_file.endswith('.tsv') or args.input_file.endswith('.tab'):
    sep = '\t'
elif args.output_file.endswith('.csv'):
    sep = ','
else:
    raise ValueError('output_file file must be a tab-separated (.tsv, .tab) or comma-separated (.csv) file')


if args.verbose: print(f'Predicting labels for {len(df)} inputs')
preds = classifier(df[args.text_col].to_list())


preds = pd.DataFrame(preds)
preds = preds.rename(columns={'label': args.label_col_name, 'score': args.label_col_name.replace('_label', '_score')})


df = pd.concat([df.reset_index(drop=True), preds], axis=1)


df.to_csv(args.output_file, sep=sep, index=False)


