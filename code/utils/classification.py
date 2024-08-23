import os
import json
import shutil
import pandas as pd

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    PreTrainedTokenizer, 
    DefaultDataCollator, 
    TrainingArguments,
    TrainerCallback,
    EarlyStoppingCallback,
    Trainer,
)

from sklearn.model_selection import train_test_split
import gc

from typing import List, Dict, Union, Optional, Callable, Tuple


# ------------------------------------------------
#  Dataset splitting
# ------------------------------------------------

def _check_split_sizes(dev_size, test_size, n):
        if dev_size and isinstance(dev_size, float): assert 0 < dev_size < 1, "dev_size must be in (0, 1)"
        if test_size and isinstance(test_size, float): assert 0 < test_size < 1, "test_size must be in (0, 1)"
        if (
            dev_size and isinstance(dev_size, float)
            and 
            test_size and isinstance(test_size, float)
        ): 
            assert (dev_size + test_size) < 1, "dev_size + test_size must be less than 1"

        if dev_size and isinstance(dev_size, int): assert dev_size < n, "dev_size must be in less than "+str(n)
        if test_size and isinstance(test_size, int): assert test_size < n, "test_size must be less than "+str(n)

        # compute sizes
        n_test = 0 if test_size is None else test_size if isinstance(test_size, int) else int(test_size * n) 
        n_dev =  0 if dev_size  is None else dev_size  if isinstance(dev_size, int)  else int(dev_size  * n)
        assert n_test + n_dev < n, "test_size + dev_size must be less than the number of examples"

        return n_dev, n_test

def _split_data_frame(
        df: pd.DataFrame,
        dev_size: float=0.15,
        test_size: float=0.15,
        stratify_by: Optional[Union[str, List[str]]]=None,
        seed: int=42,
        return_dict: bool=False
    ):
    """Split a DataFrame into training, development, and test sets.

    Args:

    df: pd.DataFrame
        The DataFrame to split.
    dev_size: float
        The proportion of the data to include in the development set.
    test_size: float
        The proportion of the data to include in the test set.
    stratify_by: str or list of str, optional
        Column(s) to use for stratified splitting. If a single column is 
        provided, the data will be stratified by the values in that column. 
        If multiple columns are provided, the data will be stratified by 
        the unique combinations of values in those columns.
    seed: int
        Random seed for reproducibility.
    return_dict: bool
        Whether to return the splits as a dictionary.
    """
    dev_size, test_size = _check_split_sizes(dev_size, test_size, len(df))
    
    if stratify_by:
        if isinstance(stratify_by, str):
            stratify_by = [stratify_by]
        for col in stratify_by:
            assert col in df.columns, f"Column '{col}' not found in ``df``. cannot use for stratified splitting."
        # create a grouping indicator based on the stratification columns
        df['__stratum__'] = df.groupby(stratify_by).ngroup()
    
    tmp, df_test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df['__stratum__'] if stratify_by else None)
    df_train, df_dev = train_test_split(tmp, test_size=dev_size, random_state=seed, stratify=tmp['__stratum__'] if stratify_by else None)
    del df, tmp
    gc.collect()
    
    if return_dict:
        return {'train': df_train, 'dev': df_dev, 'test': df_test}
    else:
        return df_train, df_dev, df_test

def _split_corpus(
        corpus: List[Dict],
        test_size: Union[None, float, int]=0.2, 
        dev_size: Union[None, float, int]=0.2, 
        stratify_by: Optional[Union[str, List[str]]]=None,
        seed: int=42,
        return_dict: bool=False
    ):
    """Split a cropus into training, development, and test sets.

    Args:

    df: List[Dict]
        The corpus to split. Must be a list of dictionaries.
    dev_size: float
        The proportion of the data to include in the development set.
    test_size: float
        The proportion of the data to include in the test set.
    stratify_by: str or list of str, optional
        Metadata field(s) to use for stratified splitting. If a single field is 
        provided, the data will be stratified by the values in that field in the metadata. 
        If multiple columns are provided, the data will be stratified by 
        the unique combinations of values of these fields in the metadata.
    seed: int
        Random seed for reproducibility.
    return_dict: bool
        Whether to return the splits as a dictionary.
    """
    n = len(corpus)
    dev_size, test_size = _check_split_sizes(dev_size, test_size, n)

    if stratify_by:
        assert all('metadata' in doc for doc in corpus), "Stratification requires 'metadata' field in each document's dictionary"
        if isinstance(stratify_by, str):
            stratify_by = [stratify_by]
        for field in stratify_by:
            assert all(field in doc['metadata'] for doc in corpus), f"Field '{field}' not found in 'metadata' of all documents"
        # create a grouping indicator based on the stratification columns
        strata = ['__'.join([str(doc['metadata'][field]) for field in stratify_by]) for doc in corpus]
    else:
        strata = None
        
    idxs = list(range(n))
    tmp, test_idxs = train_test_split(idxs, test_size=test_size, random_state=seed, stratify=strata)# if test_size > 0 else (idxs, [])
    strata = [strata[i] for i in test_idxs] if stratify_by else None
    train_idxs, dev_idxs = train_test_split(tmp, test_size=dev_size, random_state=seed, stratify=strata)# if dev_size > 0 else (idxs, [])

    train, dev, test = [corpus[i] for i in train_idxs], [corpus[i] for i in dev_idxs], [corpus[i] for i in test_idxs]
    
    if return_dict:
        return {'train': train, 'dev': dev, 'test': test}
    else:
        return train, dev, test

def split_data(
        data: Union[pd.DataFrame, List[Dict]],
        test_size: Union[None, float, int]=0.2,
        dev_size: Union[None, float, int]=0.2,
        stratify_by: Optional[Union[str, List[str]]]=None,
        seed: int=42,
        return_dict: bool=False
    ):
    """Split a dataset into training, development, and test sets.

    df: List[Dict]
        The data to split. Must be a data frame or a list of dictionaries.
    dev_size: float
        The proportion of the data to include in the development set.
    test_size: float
        The proportion of the data to include in the test set.
    stratify_by: str or list of str, optional
        Metadata field(s)/column(s) to use for stratified splitting. If a single field is 
        provided, the data will be stratified by the values in that field/column. 
        If multiple columns are provided, the data will be stratified by 
        the unique combinations of values in these fields/columns.
    seed: int
        Random seed for reproducibility.
    return_dict: bool
        Whether to return the splits as a dictionary.
    """
    if isinstance(data, pd.DataFrame):
        return _split_data_frame(data, dev_size, test_size, stratify_by, seed, return_dict)
    elif isinstance(data, list) and all(isinstance(doc, dict) for doc in data):
        return _split_corpus(data, test_size, dev_size, stratify_by, seed, return_dict)
    else:
        raise ValueError('`data` must be a pandas DataFrame or a list of dictionaries')  

# ------------------------------------------------
#  Token Classification
# ------------------------------------------------

# TODO: can we generalize this
def prepare_token_labels(data: List[Dict], discard_unsure: bool=False) -> (List[List[str]], List[List[int]]):
    """
    Note this function is psecific to our coding scheme that has an additional "unsure" category, 
    which is mapped to 6 (I-unsure) and 12 (B-unsure) in the labels.
    Here we just remove it and re-index the other label categories
    """
    tokens = list()
    labels = list()
    for doc in data:
        tokens.append(list())
        labels.append(list())
        for i, tok in enumerate(doc["tokens"]):
            if len(tok) > 0:
                tokens[-1].append(tok)
                lab = doc["labels"][i]
                if not discard_unsure:
                    # recode "unsure" to outside
                    if lab in [6, 12]: lab = 0 
                    # recode all B-* ("beginning of span") label indicators
                    if lab > 5: lab -= 1 
                labels[-1].append(lab)
    return tokens, labels

def create_token_classification_dataset(
    corpus: List[Dict], 
    tokens_field: str='tokens',
    labels_field: Union[None, str]='labels'
):
    dataset = Dataset.from_list(corpus)
    if tokens_field != 'tokens':
        dataset = dataset.rename_column(tokens_field, 'tokens')
    if labels_field is not None and labels_field != 'labels':
        dataset = dataset.rename_column(labels_field, 'labels')
    required = ['tokens'] if labels_field is None else ['tokens', 'labels']
    rm = [c for c in dataset.column_names if c not in required]
    if len(rm) > 0:
        dataset = dataset.remove_columns(rm)
    return dataset


def tokenize_and_align_sequence_labels(examples, tokenizer, **kwargs):
    # source: simplied from  https://github.com/huggingface/transformers/blob/730a440734e1fb47c903c17e3231dac18e3e5fd6/examples/pytorch/token-classification/run_ner.py#L442
    tokenized_inputs = tokenizer(examples['tokens'], is_split_into_words=True, truncation=True, **kwargs)

    labels = []
    for i, label in enumerate(examples['labels']):
        # map tokens to their respective word
        word_ids = tokenized_inputs.word_ids(batch_index=i)  
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  
        	# set the special tokens to -100
            if word_idx is None:
                label_ids.append(-100)
            # only label the first token of a given word
            elif word_idx != previous_word_idx:  
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs['labels'] = labels
    return tokenized_inputs

# ------------------------------------------------
#  Sequence classification
# ------------------------------------------------

def create_sequence_classification_dataset(
        corpus: Union[pd.DataFrame, List[Dict]], 
        text_field: str='text', 
        label_field: str='label'
    ) -> Dataset:
    dataset = Dataset.from_list(corpus) if isinstance(corpus, list) else Dataset.from_pandas(corpus)
    if text_field != 'text':
        dataset = dataset.rename_column(text_field, 'text')
    if label_field != 'label':
        dataset = dataset.rename_column(label_field, 'label')
    required = ['text', 'label']
    rm = [c for c in dataset.column_names if c not in required]
    if len(rm) > 0:
        dataset = dataset.remove_columns(rm)
    return dataset


# ------------------------------------------------
#  Trainer
# ------------------------------------------------
from torch.nn import CrossEntropyLoss

class ClassWeightsTrainer(Trainer):

    def __init__(self, class_weights: Union[List, Dict[Union[int, str], float]], **kwargs):
        """
        argument ``class_weights`` should be a dictionary mapping class labels to weights or a list of only the weights
        """
        super().__init__(**kwargs)
        # self.model = self.model.to(self.model.device)
        if len(class_weights) != self.model.config.num_labels:
            raise ValueError(f'length of `class_weights` must be {self.model.config.num_labels}')
        if isinstance(class_weights, dict):
            if set(class_weights.keys()) != set(self.model.config.id2label.keys()):
                raise ValueError(f'keys of `class_weights` mismatch label classes {list(self.model.config.id2label.keys())}')
            class_weights = [v for k, v in sorted(class_weights.items(), key=lambda item: item[1])]
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.model.device)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get('labels')
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get('logits')
        # compute custom loss
        loss_fct = CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


class WriteValidationResultsCallback(TrainerCallback):
    """Trainer callback to write validation set results to disk while training"""
    def __init__(self, path='validation_results.jsonl', overwrite=True):
        super().__init__()
        self.path = path
        if overwrite:
            with open(self.path, 'w') as f:
                f.write('')
    
    def on_evaluate(self, args, state, control, **kwargs):
        validation_results = state.log_history[-1]
        with open(self.path, "a") as f:
            f.write(json.dumps(validation_results) + "\n")


def train_and_test(
    experiment_name: str,
    experiment_results_path: str,
    run_id: Union[None, str],
    model_init: Callable,
    tokenizer: PreTrainedTokenizer,
    data_collator: DefaultDataCollator,
    train_dat: Dataset,
    dev_dat: Union[None, Dataset],
    test_dat: Union[None, Dataset],
    compute_metrics: Callable,
    metric: str,
    class_weights: Optional[Union[List, Dict[Union[int, str], float]]]=None,
    epochs: int = TrainingArguments.num_train_epochs,
    learning_rate: float = TrainingArguments.learning_rate,
    train_batch_size: int = TrainingArguments.per_device_train_batch_size,
    gradient_accumulation_steps: int = TrainingArguments.gradient_accumulation_steps,
    fp16_training: bool = True,
    eval_batch_size: int = TrainingArguments.per_device_eval_batch_size,
    weight_decay: float = TrainingArguments.weight_decay,
    early_stopping: bool = True,
    early_stopping_patience: int = 3,
    early_stopping_threshold: float = 0.03,
    seed: int = 42,
    save_best_model: bool = True,
    save_tokenizer: bool = True,
) -> Tuple[Trainer, str, Dict[str, float]]:
    """
    Fine-tune and evaluate a Transformer model.

    Args:
        experiment_name (str): 
            Name of the experiment. Used for creating directories for saving results.
        experiment_results_path (str): 
            Base path where experiment results will be saved.
        run_id (Union[None, str]): 
            Optional unique identifier for the run. If None, an identifier will be generated.
        model_init (Callable): 
            A function that initializes the model to be trained.
        tokenizer (PreTrainedTokenizer): 
            Tokenizer for preprocessing text data.
        data_collator (DefaultDataCollator): 
            Data collator that batches data samples.
        train_dat (Dataset): 
            The dataset used for training.
        dev_dat (Union[None, Dataset]): 
            The dataset used for validation. If None, validation is skipped.
        test_dat (Union[None, Dataset]): 
            The dataset used for testing. If None, testing is skipped.
        compute_metrics (Callable): 
            Function to compute metrics based on predictions and true labels.
        metric (str): 
            Name of the metric to be used for evaluation.
        epochs (int): 
            Number of training epochs. Defaults to TrainingArguments.num_train_epochs.
        learning_rate (float): 
            Learning rate for the optimizer. Defaults to TrainingArguments.learning_rate.
        train_batch_size (int): 
            Batch size for training. Defaults to TrainingArguments.per_device_train_batch_size.
        gradient_accumulation_steps (int): 
            Number of steps to accumulate gradients before updating model parameters. Defaults to TrainingArguments.gradient_accumulation_steps.
        fp16_training (bool): 
            Whether to use mixed precision training. Defaults to True.
        eval_batch_size (int): 
            Batch size for evaluation. Defaults to TrainingArguments.per_device_eval_batch_size.
        weight_decay (float): 
            Weight decay for the optimizer. Defaults to TrainingArguments.weight_decay.
        early_stopping (bool): 
            Whether to use early stopping. Defaults to True.
        early_stopping_patience (int): 
            Number of evaluations with no improvement after which training will be stopped. Defaults to 3.
        early_stopping_threshold (float): 
            Minimum change in the monitored metric to qualify as an improvement. Defaults to 0.03.
        seed (int): 
            Random seed for reproducibility. Defaults to 42.
        
    Returns:
        Trainer: 
            Trainer object used for training.
        str: 
            Path to the best model checkpoint.
        dict: 
            Evaluation results on the test set.
    """
    results_path = os.path.join(experiment_results_path, experiment_name)
    os.makedirs(results_path, exist_ok=True)

    output_path = os.path.join(results_path, 'checkpoints')
    logs_path = os.path.join(results_path, 'logs')

    # note: the following training options depend on the availability of a dev set and will be disabled if none is provided
    #  - evaluating after each epoch
    #  - early stopping
    #  - saving at most 2 models during training
    #  - saving the best model at the end
    #  - saving the dev results

    training_args = TrainingArguments(
        # hyperparameters
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_eval_batch_size=eval_batch_size,
        weight_decay=weight_decay,
        optim='adamw_torch',
        # how to select "best" model
        do_eval=dev_dat is not None,
        metric_for_best_model=metric,
        load_best_model_at_end=True,
        # when to evaluate
        evaluation_strategy='epoch',
        # when to save
        save_strategy='epoch',
        save_total_limit=2 if dev_dat is not None else None, # don't save all model checkpoints
        # where to store results
        output_dir=output_path,
        overwrite_output_dir=True,
        # logging
        logging_dir=logs_path,
        logging_strategy='epoch',
        report_to='none',
        # efficiency
        use_mps_device=torch.backends.mps.is_available(),
        fp16=fp16_training if torch.cuda.is_available() else False,
        fp16_full_eval=False,
        # reproducibility
        seed=seed,
        data_seed=seed,
        full_determinism=True
    )

    # build callbacks
    callbacks = []
    if early_stopping:
        if dev_dat is None:
            raise ValueError('Early stopping requires a dev data set')
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience, early_stopping_threshold=early_stopping_threshold))
    if dev_dat:
        fn = run_id+'-dev_results.jsonl' if run_id else 'dev_results.jsonl'
        fp = os.path.join(results_path, fn)
        callbacks.append(WriteValidationResultsCallback(path=fp))

    # train
    trainer_args = dict(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dat,
        eval_dataset=dev_dat if dev_dat is not None else None,
        tokenizer=tokenizer,
        data_collator=data_collator if data_collator is not None else None,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    if class_weights:
        trainer_args['class_weights'] = class_weights
        trainer = ClassWeightsTrainer(**trainer_args)
    else:
        trainer = Trainer(**trainer_args)
    
    print('Training ...')
    _ = trainer.train()

    # save best model to results folder
    # CAVEAT: this is not the "best" model if no dev_dat is provided
    dest = run_id+'-best_model' if run_id else 'best_model'
    dest = os.path.join(results_path, dest)
    if os.path.exists(dest):
        shutil.rmtree(dest)
    if save_best_model:
        trainer.save_model(dest)
        # save tokenizer to best_model folder
        if save_tokenizer:
            tokenizer.save_pretrained(dest)

    # evaluate
    if test_dat:
        print('Evaluating ...')
        res = trainer.evaluate(test_dat, metric_key_prefix='test')
        print(res)
        fn = run_id+'-test_results.json' if run_id else 'test_results.json'
        fp = os.path.join(results_path, fn)
        with open(fp, 'w') as file:
            json.dump(res, file)
    else:
      res = None

    # finally: clean up
    if os.path.exists(output_path):
        # TODO: reconsider this when dev_dat is None (in this case, no best model will be copied and deliting the output path would delete any model checkpoints)
        shutil.rmtree(output_path)
    if os.path.exists(logs_path):
        shutil.rmtree(logs_path)

    return trainer, dest, res
