import numpy as np
from datasets import Dataset
from seqeval.metrics import classification_report
from transformers.trainer_utils import PredictionOutput
from typing import Dict, List, Union

# see also: https://github.com/haukelicht/group_mention_detection/blob/main/replication/code/utils/classification.py
def tokenize_and_align_sequence_labels(examples, tokenizer, **kwargs) -> Dict:
    # source: simplied from  https://github.com/huggingface/transformers/blob/730a440734e1fb47c903c17e3231dac18e3e5fd6/examples/pytorch/token-classification/run_ner.py#L442
    tokenized_inputs = tokenizer(examples['tokens'], is_split_into_words=True, **kwargs)

    labels = []
    for i, label in enumerate(examples['labels']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs['labels'] = labels
    return tokenized_inputs

# see also: https://github.com/haukelicht/group_mention_detection/blob/main/replication/code/utils/classification.py
def create_token_classification_dataset(
    data: List[Dict], 
    tokens_field: str='tokens',
    labels_field: Union[None, str]='labels'
):
    dataset = Dataset.from_list(data)
    if tokens_field != 'tokens':
        dataset = dataset.rename_column(tokens_field, 'tokens')
    if labels_field is not None and labels_field != 'labels':
        dataset = dataset.rename_column(labels_field, 'labels')
    required = ['tokens'] if labels_field is None else ['tokens', 'labels']
    rm = [c for c in dataset.column_names if c not in required]
    if len(rm) > 0:
        dataset = dataset.remove_columns(rm)
    return dataset

DEFAULT_TRAINING_ARGS = dict(
    num_train_epochs=10,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=32,
    weight_decay=0.01,
    warmup_ratio=0.1,
)


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

def compute_seqeval_metrics(p: PredictionOutput, label_list: List[str]):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    true_predictions = [
        correct_iob2([label_list[p] for (p, l) in zip(prediction, label) if l != -100])
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        correct_iob2([label_list[l] for (_, l) in zip(prediction, label) if l != -100])
        for prediction, label in zip(predictions, labels)
    ]
    
    results = classification_report(true_labels, true_predictions, output_dict=True)

    return results