import numpy as np
import pandas as pd


from copy import deepcopy
import warnings

from typing import List, Dict, Union, Optional, Literal, Tuple

# NOTE: there should be a function in seqeval.utils with a similar functionality
def extract_spans(
        labels: List[str], 
        tokens: Optional[List[str]]=None
    ) -> List[Tuple[Union[List[str], None], str, int, int]]:
    """
    Extracts all spans from a sequence of predicted token labels.

    This function assumes that the labels are in IOB2 scheme, where each span
    starts with a 'B' tag (for beginning) and subsequent tokens in the same span
    are tagged with an 'I' tag (for inside). Tokens outside of any span are
    tagged with an 'O' tag (for outside).

    Args:
        labels (list): A list of predicted labels for each token in the input
            sequence.
        tokens (list, optional): A list of tokens in the input sequence. If
            provided, the function will return the tokens for each span in the
            output. If not provided, the function will only return the start and
            end indices of each span in the input sequence.

    Returns:
        list: A list of tuples, where each tuple contains the following
            elements:
            - A list of tokens in the span (None if `tokens=None`).
            - The type of the span.
            - The start index of the span in the input sequence.
            - The (exclusive) end index of the span in the input sequence.
    """
    spans = []
    current_span = []
    current_type = None
    current_start = None
    
    for i, label in enumerate(labels):
        if label == 'O':
            if current_span:
                # End the current span and add it to the list of spans.
                spans.append([current_span, current_type, current_start, i])
                current_span = []
                current_type = None
                current_start = None
        elif label.startswith('B'):
            if current_span:
                # End the current span and add it to the list of spans.
                spans.append([current_span, current_type, current_start, i])
            # Start a new span.
            current_span = [tokens[i]] if tokens is not None else [None]
            current_type = label[2:]  # Remove the 'B-' prefix.
            current_start = i
        elif label.startswith('I'):
            if current_span and current_type == label[2:]:
                # Add the current token to the current span.
                if tokens is not None:
                    current_span.append(tokens[i])
                else:
                    current_span.append(None)
            else:
                if current_span:
                    # End the current span and add it to the list of spans.
                    spans.append([current_span, current_type, current_start, i])
                # Start a new span.
                current_span = [tokens[i]] if tokens is not None else [None]
                current_type = label[2:]  # Remove the 'I-' prefix.
                # current_start = max(0, i - 1)
                current_start = i
    
    if current_span:
        # End the final span and add it to the list of spans.
        spans.append([current_span, current_type, current_start, len(labels)])
    
    if tokens is None:
        for span in spans:
            span[0] = None
    return [tuple(span) for span in spans]
# # Example
# tokens = ['Today', ',', 'Barack', 'Obama', 'and', 'Justin', 'Trudeau', 'meet', 'in', 'Osnabrügge']
# obs    = ['O',     'O', 'B-PER',  'I-PER', 'O',   'B-PER',  'I-PER',   'O',    'O',  'B-LOC'     ]
# pred   = ['O',     'O', 'B-PER',  'I-PER', 'O',   'B-PER',  'O',       'O',    'O',  'B-LOC'     ]
# print(extract_spans(obs, tokens))
# print(extract_spans(obs))

def _keep_type_annotations(x: List[str], type: Optional[str]=None):
    return [l if l == 'O' else l[0] if type is None else l[0] if l[2:] == type else 'O' for l in x]
# # Examples
# labels = ['B-a', 'I-a', 'O'  , 'B-b', 'I-b', 'I-b', 'I-b', 'I-b', 'I-b', 'I-b', 'O'  , 'O', 'O']
# print(_keep_type_annotations(labels, type='a'))
# print(_keep_type_annotations(labels, type='b'))
# print(_keep_type_annotations(labels, type=None))


def _align_b_tags(obs: List[str], pred: List[str]) -> Tuple[List[str], List[str]]:
    obs = deepcopy(obs)
    pred = deepcopy(pred)
    assert len(obs) == len(pred)
    for i in range(1, len(obs)):
        o_tag, o_type =  obs[i][0],  obs[i][2:]
        p_tag, p_type = pred[i][0], pred[i][2:]
        if p_tag == 'I' and o_tag == 'B' and p_type == o_type and pred[i-1][0] == 'I' and pred[i-1][2:] == p_type:
            pred[i] = 'B-' + p_type if len(p_type) > 0 else 'B'
    return obs, pred


def _compute_span_scores(
        spans: List[Tuple[Union[List[str], None], str, int, int]], 
        ref: List[str]
    ):
    if len(spans) == 0:
        return []
    
    scores = []
    for _, typ, s, e in spans:
        score = sum(l != 'O' for l in ref[s:e])/(e-s)
        scores.append((range(s, e), score))
    return scores


def _check_range_overlap(x: range, y: range) -> bool:
    """Check if two ranges overlap."""
    return bool(set(x).intersection(set(y)))


def _f1(p, r, nan_to_zero: bool=False): 
    """Compute the F1 score from precision and recall"""
    if nan_to_zero:
        p = 0.0 if np.isnan(p) else p
        r = 0.0 if np.isnan(r) else r
    if np.isnan(p) or np.isnan(r):
        return np.nan
    elif p + r == 0:
        return 0.0
    else:
        return 2 * p * r / (p + r)


def _align_spanwise_scores(precisions, recalls, nan_to_zero: bool=False):
    aligned = []

    if len(precisions) == 0 and len(recalls) == 0:
        return [], [], []
    
    # match precision to recall  scores
    remove_prec = set()
    remove_recl = set()
    # try to match precision scores to recall scores
    for i, (pl, ps) in enumerate(precisions):
        for j, (rl, rs) in enumerate(recalls):
            if _check_range_overlap(pl, rl):
                s = min(min(pl), min(rl))
                aligned.append((j, s, ps, rs, len(pl)))
                remove_prec.add(i)
                remove_recl.add(j)
    precisions = [(i, p) for i, p in enumerate(precisions) if i not in remove_prec]
    recalls    = [(i, r) for i, r in enumerate(recalls)    if i not in remove_recl]

    # add any remaining
    for i, (pl, ps) in precisions:
        aligned.append((-1*i, min(pl), ps, 0.0 if nan_to_zero else np.nan, len(pl)))
    for j, (rl, rs) in recalls:
        aligned.append((j, min(rl), 0.0 if nan_to_zero else np.nan, rs, 1))

    # sort by token location
    aligned = sorted(aligned, key=lambda x: x[2])
        
    # put in data frame
    aligned = pd.DataFrame(aligned, columns=['i', 's', 'p', 'r', 'l'])

    # compute F1 score
    aligned['f'] = aligned.apply(lambda x: _f1(x['p'], x['r'], nan_to_zero=nan_to_zero), axis=1)

    # for each observed span, compute weigthed average of precision, recall, and F1-score by predicted span(s) lengths
    def weighted_avg(x):
        nas = np.isnan(x)
        if np.all(nas):
            return 0.0 if nan_to_zero else np.nan
        return np.average(x[~nas], weights=aligned.loc[x.index, 'l'].values[~nas])
    aligned = aligned.groupby('i').agg({'p': weighted_avg, 'r': weighted_avg, 'f': weighted_avg}).reset_index(drop=True)

    return tuple(aligned.to_dict(orient='list').values())


def _compute_spanwise_scores(
        y_true: List[str], 
        y_pred: List[str], 
        types: Optional[List[str]]=None,
        average: Optional[Literal['micro', 'macro']]=None, 
        nan_to_zero: bool=True
    ):

    if types is None:
        types = list(set(y_true) | set(y_pred))
        types = list(set([t[2:] for t in types if t != 'O']))
        if len(types) == 0:
            types = [None]
    
    # if micro acerage, replace entity types with None and report this type's scores at the end
    if average == 'micro':
        types = ['micro']
    
    scores = {}
    for typ in types:
        # keep only annotations of type `typ`
        obs = _keep_type_annotations(y_true, type=None if average=='micro' else typ) 
        pred = _keep_type_annotations(y_pred, type=None if average=='micro' else typ)
        # algin B- tags of predicted spans with observed spans
        obs, pred = _align_b_tags(obs, pred)
        # compute span-wise precision and recall
        precs = _compute_span_scores(extract_spans(pred), ref=obs)
        recls = _compute_span_scores(extract_spans(obs), ref=pred)
        scores[typ] = _align_spanwise_scores(precs, recls, nan_to_zero=nan_to_zero)
    
    if average == 'micro':
        return scores['micro']
    
    if average == 'macro':
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            prec = np.nanmean([scores[typ][0] for typ in types])
            recl = np.nanmean([scores[typ][1] for typ in types])
            f1   = np.nanmean([scores[typ][2] for typ in types])
        return prec, recl, f1

    return scores


def compute_span_metrics(
        y_true: List[List[str]], 
        y_pred: List[List[str]], 
        types: Optional[List[str]]=None
    ) -> Dict[str, Dict[str, float]]:
    """
    Compute precision, recall, and f1-score for each entity type and for the micro and macro average.

    Args:

        y_true (List[List[str]]): A list of lists of of observed labels for each token in each input sequence.
        y_pred (List[List[str]]): A list of lists of predicted labels for each token in each input sequence.
        types (List[str], optional): A list of entity types to consider. If None, all entity types in the observed 
        data are considered.

    Returns:
        socres: Dict[str, Dict[str, float]]
            The precision, recall, and f1-score computed in two ways for each 
            entity type, as micro, and as macro average.
    
            1. Span-wise (``spanwise``): computed in two steps
                1. Computes precision, recall, and the F1 score for each
                   span in each sequences (e.g., sentences) in ``y_true`` 
                   and ``y_pred``.
                2. Averages these scores over all spans.
            2. Sequence average (``seqavg``): computed in three steps
                1. Computes precision, recall, and the F1 score for each
                   span in each sequences (e.g., sentences) in ``y_true`` 
                   and ``y_pred``.
                2. Averages these scores within sequences (e.g. sentences).
                3. Averages these averages across sequences (e.g. sentences).
            
            The micro average ignores entity types when computing these metrics.
            For example, when a predicted Location (LOC) entity is matched with an
            observed Person (PER) entity, the micro average will consider this a 
            success.

            The macro average first computes the spanwise and sequence average precision, 
            recall, and f1-score scores by type and then averages them across types.

    Interpretation:
        
        Spanwise scores estimate precision and recall for individual spans.
        
        - ``precision_spanwise``: The share of tokens in a predicted span
          (of a certain entity type) that are also part of observed ("true") span.
        - ``recall_spanwise``: The share of tokens in an observed ("true") span
          that have been predicted to belong to a span (of the given entity type).
        - ``f1-score_spanwise``: The arithmetic mean of ``precision_spanwise`` and 
          ``recall_spanwise``
        
        Sequence average scores estimate what precision and recall one can 
        expect, on average, for all spans of a given type in a typical sequence
        (e.g., a sentence).

        - ``precision_seqavg``: The share of predicted spans (of a certain entity type)
          that match the observed ("true") span in a sequence.
        - ``recall_seqavg``: The share of observed ("true") spans in a sequence that 
          have been predicted correctly.
        - ``f1-score_spanwise``: The arithmetic mean of ``precision_seqavg`` and 
          ``recall_seqavg``
    """
    # TODO: consider adding argument that indicates whether to compute span-wise or sequence average scores
 
    nan_to_zero = True # this makes behavior in edge cases compatible with `seqeval`

    if types is None:
        types = list(set(y_true) | set(y_pred))
        types = list(set([t[2:] for t in types if t != 'O']))
        if len(types) == 0:
            types = [None]
    
    fields = types + ['micro avg']
    seq_avgs  = ({typ: [] for typ in fields}, {typ: [] for typ in fields}, {typ: [] for typ in fields})
    spanwise = ({typ: [] for typ in fields}, {typ: [] for typ in fields}, {typ: [] for typ in fields})
    
    for i, (o, p) in enumerate(zip(y_true, y_pred)):
        typewise = _compute_spanwise_scores(o, p, types=types, nan_to_zero=nan_to_zero)
        micro = _compute_spanwise_scores(o, p, types=types, average='micro', nan_to_zero=nan_to_zero)
        with warnings.catch_warnings():
          warnings.simplefilter("ignore", category=RuntimeWarning)
          for m in range(3):
              for typ in types:
                  spanwise[m][typ].extend(typewise[typ][m])
                  if len(spanwise[m][typ]) > 0:
                      seq_avgs[m][typ].append(np.nanmean(spanwise[m][typ]))
              spanwise[m]['micro avg'].extend(micro[m])
              if len(micro[m]) > 0:
                seq_avgs[m]['micro avg'].append(np.nanmean(micro[m]))
    
    results = {}
    mets = ['precision', 'recall', 'f1-score']
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for field in fields:
            results[field] = {}
            for m, i in zip(mets, range(3)):
                results[field][f'{m}_spanwise'] = np.nanmean(spanwise[i][field])
            for m, i in zip(mets, range(3)):
                results[field][f'{m}_seqavg']   = np.nanmean( seq_avgs[i][field])
        results['macro avg'] = {}
        for m in results['micro avg'].keys():
            results['macro avg'][m] = np.nanmean([results[typ][m] for typ in types])

    return results
