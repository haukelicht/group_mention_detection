
from types import SimpleNamespace

args = SimpleNamespace()

args.input_file = './../data/manifestos/all_uk_manifesto_sentences.tsv'
args.id_col = 'sentence_id'
args.text_col = 'text'
args.metadata_cols = 'manifesto_id,party,date'
args.model_path = '../results/classifiers/uk-manifestos_roberta/best_model/'
args.batch_size = 64
args.output_file = '../data/labeled/all_uk_manifesto_sentences_predicted_labels.jsonl'
args.return_spanlevel = True
args.test = False
args.verbose = True

args.metadata_cols = [c.strip() for c in args.metadata_cols.split(',')]

# ## Setup


import pandas as pd
import jsonlines

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

device = 'cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
device = torch.device(device)
if args.verbose: print('using device:', str(device))


tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, add_prefix_space=True)
model = AutoModelForTokenClassification.from_pretrained(args.model_path)

# construct the pipeline
classifier = pipeline(
    task='ner', 
    model=model, 
    tokenizer=tokenizer,
    aggregation_strategy='simple',
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
if args.test:
    n_ = args.batch_size*10
    if n_ < len(df):
        df = df.sample(n=n_, random_state=42)


if args.verbose: print(f'Predicting labels for {len(df)} inputs')
preds = classifier(df[args.text_col].to_list())


# add predicted spans to the dataframe
df['spans'] = [
    [
        [span['start'], span['end'], span['entity_group']]
        for span in spans
    ]
    for spans in preds 
]


# write as JSONL to the output file
if args.verbose: print(f'Writing text and predicted labels in JSONL format to {args.output_file}')
with jsonlines.open(args.output_file, 'w') as file:
    for _, d in df.iterrows():
        out = {
            'id': d[args.id_col],
            'text': d[args.text_col],
            'labels': d['spans'],
            'metadata': {c: d[c] for c in args.metadata_cols},
        }
        file.write(out)


# unnesting data frame to span level
if args.return_spanlevel:
    # get relevant columns
    df = df[[args.id_col, args.text_col] + args.metadata_cols + ['spans']]
    # get span index (within text unit)
    df.loc[:, 'span_nr'] = df.spans.apply(lambda x: list(range(len(x))))
    # explode nested list of spans to span level (like tidyr::unnest_longer in R)
    df = df.explode(['spans', 'span_nr'])

    # drop inputs with no predicted spans
    df = df[~df.spans.isna()]

    df['span_nr'] = df.span_nr+1
    df.rename(columns={'text': 'sentence_text'}, inplace=True)
    # get the span label and text (a.k.a 'mention')
    df['label'] = df.apply(lambda r: r.spans[2], axis=1)
    df['text'] = df.apply(lambda r: r.sentence_text[r.spans[0]:r.spans[1]], axis=1)


    # bring the colums in the right order
    df = df[args.metadata_cols + [args.id_col, 'sentence_text', 'span_nr', 'label', 'text']]

    args.output_file = args.output_file.replace('_labels.jsonl', '_spans.tsv')
    if args.verbose: print(f'Writing span-level predictions in TSV format to {args.output_file}')
    df.to_csv(args.output_file, sep='\t', index=False, encoding='utf-8')


