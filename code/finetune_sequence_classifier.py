# ## Setup

from types import SimpleNamespace

args = SimpleNamespace()


args.experiment_name = 'cap_classifier'
args.experiment_results_path = './../results/classifiers'

args.data_file = '../data/exdata/cap/con+lab_manifesto_cap_codes.csv'
args.text_col = 'text'
args.label_col = 'majortopic_recoded_label'

args.seed = 1234
args.test_size = 2000
args.dev_size = 2000
args.split_stratify_cols = 'party,year'

args.model_name = 'roberta-base'
args.epochs = 10
args.learning_rate = 5e-05
args.weight_decay = 0.2
args.train_batch_size = 32
args.eval_batch_size = 64

args.metric = 'f1_macro'

args.split_stratify_cols = [] if args.split_stratify_cols is None else [c.strip() for c in args.split_stratify_cols.split(',')]

# ### load libraries


from collections import Counter
import numpy as np
import pandas as pd

import torch
import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, set_seed

from utils.classification import create_sequence_classification_dataset, split_data, train_and_test
from utils.evaluation import parse_sequence_classifier_prediction_output, compute_sequence_classification_metrics


def compute_metrics(p):
    labels, predictions = parse_sequence_classifier_prediction_output(p)
    return compute_sequence_classification_metrics(labels, predictions)


def get_class_weights(data: datasets.Dataset):
    cnts = dict(sorted(Counter(data['label']).items()))
    weights = len(data)/np.array(list(cnts.values()))
    weights = weights/sum(weights)
    class_weights = {c: float(w) for c, w in zip(cnts.keys(), weights)}
    return class_weights


device = 'cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
device = torch.device(device)
print('Using device:', str(device))

set_seed(args.seed)

# ### train/dev/test split


df = pd.read_csv(args.data_file)

cols = [args.text_col, args.label_col] + args.split_stratify_cols
df = df[cols]

data_splits = split_data(df, test_size=args.test_size, dev_size=args.dev_size, stratify_by=args.split_stratify_cols, seed=args.seed, return_dict=True)

dataset = datasets.DatasetDict({s: create_sequence_classification_dataset(df, text_field=args.text_col, label_field=args.label_col) for s, df in data_splits.items()})

label2id = {l: i for i, l in enumerate(sorted(dataset['train'].unique('label')))}
id2label = {v: k for k, v in label2id.items()}

# ### Tokenize texts


## load a model and its tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

def preprocess(examples):
    output = tokenizer(examples['text'], truncation=True)
    output['label'] = [label2id[l] for l in examples['label']]
    return output 

dataset = dataset.map(preprocess, batched=True)

# ### Train classifier


class_weights = get_class_weights(dataset['train'])

def model_init():
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=len(label2id), label2id=label2id, id2label=id2label)
    model.to(device);
    return model


model, model_path, test_res = train_and_test(
    experiment_name=args.experiment_name,
    experiment_results_path=args.experiment_results_path,
    run_id=None,
    model_init=model_init,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    train_dat=dataset['train'],
    dev_dat=dataset['dev'],
    test_dat=dataset['test'],
    compute_metrics=compute_metrics,
    metric=args.metric,
    class_weights=class_weights,
    epochs=args.epochs,
    learning_rate=args.learning_rate,
    train_batch_size=args.train_batch_size,
    eval_batch_size=args.eval_batch_size,
    weight_decay=args.weight_decay,
    early_stopping=True,
    seed=args.seed,
)


print('Test set results of the best model:', test_res)


