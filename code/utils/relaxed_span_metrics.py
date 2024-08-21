from .span_utils import extract_spans
import numpy as np

from typing import List, Dict

def _compute_span_scores(spans, ref, average=None):
    """
    Compute score for spans.

    Given a list of spans and a list of reference labels, this function computes the 
    accuracy of spans given the reference labels.

    Args:
        spans (list): A list of spans, recording for each span (as a 4-tuple)
            the list of its tokens (or None), the span's type (str), 
            and the span's start and end index (ints).
        ref (list): A list of predicted or observed labels for each token in the input
            sequence.
        average (None, 'micro', or 'macro'): how to aggregate the span-wise accuracy scores:
           - `None`: by span type
           - `'micro'`: ignoring span types
           - `'macro'`: first by span types, then across span types
    """
    if len(spans) == 0:
        return {} if average is None else np.nan
    if average=='micro':
        scores = list()
        for _, typ, s, e in spans:
            score = sum(l[2:] == typ for l in ref[s:e])/(e-s)
            scores.append(score)
        return np.array(scores).mean()
    else:
        types = set(s[1] for s in spans)
        scores = {t: [] for t in types}
        for _, typ, s, e in spans:
            score = sum(l[2:] == typ for l in ref[s:e])/(e-s)
            scores[typ].append(score)
        scores = {t: np.array(s).mean() for t, s in scores.items()}
        if average=='macro':
            return np.array(list(scores.values())).mean()
        return scores

# # test
# obs    = ['O',     'O', 'B-PER',  'I-PER', 'O',   'B-PER',  'I-PER',   'O',    'O',  'B-LOC'     ]
# pred   = ['O',     'O', 'B-PER',  'I-PER', 'O',   'B-PER',  'O',       'O',    'O',  'B-LOC'     ]
# print(compute_span_score(extract_spans(obs), ref=pred))
# print(compute_span_score(extract_spans(obs), ref=pred, average='micro'))
# print(compute_span_score(extract_spans(obs), ref=pred, average='macro'))

def _compute_spanwise_scores(y_true, y_pred, average=None, zero_division=np.nan):
    """
    Compute average spanwise precision, recall, and F1 score.

    Given a lists of samples' y_trueerved labels and a list of samples' corresponding predicted labels, 
    this function computes the cross-sample average precision, recall, and F1 score.

    Args:
        y_true (list): A list of samples' observed labels (one list of labels per sample).
        y_pred (list): A list of samples' predicted labels (one list of labels per sample).
        average (None, 'micro', or 'macro'): how to aggregate the span-wise scores:
           - `None`: by span type
           - `'micro'`: ignoring span types
           - `'macro'`: first by span types, then across span types
        zero_division: how to handle zero devision. Pass None (default) or a float value to return 
           in the case of zero division at the batch level
    """
    f1score = lambda p, r: zero_division if p+r == 0 else 2*(p*r)/(p+r)
    
    if average is not None:
        # create list to store span-wise metrics
        scores = list()
        for o, p in zip(y_true, y_pred):
            # compute span's average precision
            prec = _compute_span_scores(extract_spans(p), ref=o, average=average)
            # compute span's average recall
            recl = _compute_span_scores(extract_spans(o), ref=p, average=average)
            # compute span's average F1
            f1 = f1score(prec, recl)
            # add to collector
            scores.append((prec, recl, f1))
        
        # average across spans
        scores = np.array([*scores])
        if np.isnan(scores).all():
            return np.nan, np.nan, np.nan
        scores = np.nanmean(scores, axis=0)
        return tuple(scores)
    
    else:
        # get all types in y_true and pred labels
        types = set(l[2:] for o in y_true for l in o if l != 'O')
        types.update(set(l[2:] for o in y_pred for l in o if l != 'O'))
        # create dicts to store spans' type-wise scores
        precs, recls, f1s = {t: [] for t in types}, {t: [] for t in types}, {t: [] for t in types}
        for o, p in zip(y_true, y_pred):
            # compute span's type-wise precisions 
            prec = _compute_span_scores(extract_spans(p), ref=o, average=None)
            for k, v in prec.items(): precs[k].append(v)
            # compute span's type-wise recalls
            recl = _compute_span_scores(extract_spans(o), ref=p, average=None)
            for k, v in recl.items(): recls[k].append(v)
            # compute span's type-wise F1 scores
            f1 = {t: f1score(prec[t], recl[t]) for t in types if t in prec and t in recl}
            for k, v in f1.items(): f1s[k].append(v)
        # compute cross-span type-wise averages
        precs = {k: np.nanmean(np.array(v)) if len(v) > 0 else zero_division for k,v in precs.items()}
        recls = {k: np.nanmean(np.array(v)) if len(v) > 0 else zero_division for k,v in recls.items()}
        f1s   = {k: np.nanmean(np.array(v)) if len(v) > 0 else zero_division for k,v in f1s.items()}
        # transpose to dict
        scores = {t: (precs[t] if t in precs else None, recls[t] if t in recls else None, f1s[t] if t in f1s else None) for t in types}
        return scores


def compute_relaxed_span_metrics(
        y_true: List[List[str]], 
        y_pred: List[List[str]], 
    ) -> Dict[str, Dict[str, float]]:
    """
    Compute precision, recall, and f1-score for each entity type and for the micro and macro average.

    Args:

        y_true (List[List[str]]): A list of lists of of observed labels for each token in each input sequence.
        y_pred (List[List[str]]): A list of lists of predicted labels for each token in each input sequence.
        data are considered.

    Returns:
        socres: Dict[str, Dict[str, float]]
            The precision, recall, and f1-score computed for each 
            entity type, as micro, and as macro average.
    """
    m = ['precision', 'recall', 'f1']
    result = {}
    for typ, scores in _compute_spanwise_scores(y_true, y_pred).items():
        result[typ] = {m[i]: score for i, score in enumerate(list(scores))}
    for avg in ['macro', 'micro']:
        result[avg+' avg'] = {m[i]: s for i, s in enumerate(_compute_spanwise_scores(y_true, y_pred, average=avg))}
    return result