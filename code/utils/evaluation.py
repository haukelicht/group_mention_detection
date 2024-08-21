import numpy as np
import pandas as pd
import jsonlines


import warnings

from seqeval.metrics import classification_report as seqeval_classification_report
from .span_metrics import compute_span_metrics
from .relaxed_span_metrics import compute_relaxed_span_metrics
from sklearn.metrics import precision_recall_fscore_support, classification_report, balanced_accuracy_score, accuracy_score

from transformers import Trainer
from transformers.trainer_utils import PredictionOutput

from typing import List, Dict, Optional, Tuple, Union, Literal

def correct_iob2(labels: List[str]) -> List[str]:
    prev = None
    edit = list()
    for i, l in enumerate(labels):
        if (i == 0 or prev == 'O') and l[0] == 'I':
            edit.append(i)
        prev = l
    if len(edit) > 0:
        labels = [l.replace('I-', 'B-') if i in edit else l for i, l in enumerate(labels)]
    return labels

def parse_token_classifier_prediction_output(p: PredictionOutput):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    return labels, predictions


def compute_token_classification_metrics(
        y_true: List[List[int]], 
        y_pred: List[List[int]], 
        label2id: Dict[str, int], 
    ) -> Dict[str, float]:
    
    label_list = list(label2id.keys())
    types = list(set([l[2:] for l in label_list if l != 'O']))
    
    # encode label IDs to labels
    predictions = [
        correct_iob2([label_list[p] for (p, l) in zip(preds, labs) if l != -100])
        for preds, labs in zip(y_pred, y_true)
    ]
    labels = [
        correct_iob2([label_list[l] for (_, l) in zip(preds, labs) if l != -100])
        for preds, labs in zip(y_pred, y_true)
    ]

    metrics = ['precision', 'recall', 'f1-score']
    keys = ['macro avg', 'micro avg'] + types
    results = {}
    
    # Span level (Seqeval)
    result = seqeval_classification_report(labels, predictions, output_dict=True, zero_division=0.0)
    # flatten
    result = {k: result[k] for k in keys if k in result}
    # format
    result = {
        # format: metric name <=> metric value
        str(f"{k.replace(' avg', '')}_{m.replace('f1-score', 'f1')}"): scores[m] 
        # iterate over class-wise results
        for k, scores in result.items()
        # iterate over metrics
        for m in metrics
    }
    results['seqeval'] = result
    
    # Span level (own metrics)
    
    result = compute_span_metrics(y_true=labels, y_pred=predictions, types=types)
    # flatten
    result = {k: result[k] for k in keys if k in result}
    # format
    result = {
        # format: metric name <=> metric value
        str(f"{k.replace(' avg', '')}_{m.replace('f1-score', 'f1')}"): s
        # iterate over class-wise results
        for k, scores in result.items()
        # iterate over metrics
        for m, s in scores.items()
    }
    results['spanlevel'] = result

    # relaxed span-level metrics:
    result = compute_relaxed_span_metrics(y_true=labels, y_pred=predictions)
    # flatten
    result = {k: result[k] for k in keys if k in result}
    # format
    result = {
        # format: metric name <=> metric value
        str(f"{k.replace(' avg', '')}_{m.replace('f1-score', 'f1')}_relaxed"): s
        # iterate over class-wise results
        for k, scores in result.items()
        # iterate over metrics
        for m, s in scores.items()
    }
    results['spanlevel'].update(result)
    
    # Document level
    overall = [[], []]
    by_type = {t: [[], []] for t in types}
    for o, p in zip(labels, predictions):
        overall[0].append(int(any(l != 'O' for l in o)))
        overall[1].append(int(any(l != 'O' for l in p)))
        for t in types:
            by_type[t][0].append(int(any(t == l[2:] for l in o)))
            by_type[t][1].append(int(any(t == l[2:] for l in p)))
    result = {}
    p, r, f1, _ = precision_recall_fscore_support(overall[0], overall[1], average='binary', zero_division=0.0)
    result['micro_precision'] = p
    result['micro_recall'] = r
    result['micro_f1'] = f1
    for t in types:
        p, r, f1, _ = precision_recall_fscore_support(by_type[t][0], by_type[t][1], average='binary', zero_division=0.0)
        result[t+'_precision'] = p
        result[t+'_recall'] = r
        result[t+'_f1'] = f1
    results['doclevel'] = result
    
    # Word level
    # flatten the list of lists and discard the B/I distinction
    predictions = [l if l=='O' else l[2:] for labs in predictions for l in labs]
    labels = [l if l=='O' else l[2:] for labs in labels for l in labs]

    result = classification_report(labels, predictions, output_dict=True, zero_division=0.0)
    out = {}
    keys = ['macro avg'] + ['O'] + types
    result = {k: result[k] for k in keys if k in result}
    tmp = {
        # format: metric name <=> metric value
        str(f"{c.replace(' avg', '')}_{m.replace('-score', '')}"): res[m] 
        # iterate over class-wise results
        for c, res in result.items()
        # iterate over metrics
        for m in metrics
    }
    out.update(tmp)
    results['wordlevel'] = out
    
    results = {k+'-'+m: v for k, res in results.items() for m, v in res.items()}
    return results
# # example
# obs   = [[0, 0, 3, 1, 0, 3, 1, 0, 0, 4 ]]
# preds = [[0, 0, 3, 1, 0, 3, 0, 0, 0, 4 ]]
# label2id={'O': 0, 'I-PER': 1, 'I-LOC': 2, 'B-PER': 3, 'B-LOC': 4}
# compute_token_classification_metrics(obs, preds, label2id)


def parse_eval_result(x, types: Optional[List[str]]=None, remove_prefix: Optional[str]=None):
    keys = list(x.keys()) if remove_prefix is None else [k.removeprefix(remove_prefix) for k in x.keys()]
    x = pd.DataFrame(x.values(), index=keys, columns=['value'])
    x = x[x.index.str.contains('-')]
    x = x.reset_index().rename(columns={'index': 'tmp'})
    x[['scheme', 'metric']] = x['tmp'].str.split('-', expand=True)
    x[['type', 'metric', 'kind']] = x['metric'].str.split('_', expand=True)
    x.drop(columns=['tmp'], inplace=True)

    if types is not None:
        x = x[x.type.isin(types)]

    # pivot values from column 'metric' to columns using values from 'value'
    x = x.pivot(index=['scheme', 'type', 'kind'], columns='metric', values='value').reset_index(drop=False)
    x.loc[x.kind.isna(), 'kind'] = ''

    # remove the name 'metric' from the index
    x.columns.name = None
    x.set_index('scheme', inplace=True)
    
    return x.loc[['seqeval', 'spanlevel', 'wordlevel', 'doclevel'], :]


def save_eval_log(trainer: Trainer, fp: str, append: bool=False):
    eval_log = [l for l in trainer.state.log_history if 'eval_loss' in l]
    with jsonlines.open(fp, 'a' if append else 'w') as writer:
        writer.write(eval_log)

# Sentence classification


def parse_sequence_classifier_prediction_output(p: PredictionOutput):
    logits, labels = p.predictions, p.label_ids
    predictions = np.argmax(logits, axis=1)
    return labels, predictions

def compute_sequence_classification_metrics(
        y_true: List[List[int]], 
        y_pred: List[List[int]]
    ) -> Dict[str, float]:
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        # metrics
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0.0)
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(y_true, y_pred, average='micro', zero_division=0.0)
        acc_balanced = balanced_accuracy_score(y_true, y_pred)
        acc_not_balanced = accuracy_score(y_true, y_pred)

    result = {
        'accuracy': acc_not_balanced,
        'f1_macro': f1_macro,
        'accuracy_balanced': acc_balanced,
        'f1_micro': f1_micro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
    }

    return result
