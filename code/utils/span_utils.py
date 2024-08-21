import numpy as np

from typing import List, Dict, Tuple, Union, Optional

# note: should be a function in seqeval.utils with a similar functionality
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
