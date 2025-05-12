
import os
import json
import shutil

from utils import read_jsonl
from schema import LABEL2ID as label2id, ID2LABEL as id2label
from finetuning_utils import (
    create_token_classification_dataset, 
    tokenize_and_align_sequence_labels,
    DEFAULT_TRAINING_ARGS as default_training_args
)

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    set_seed,
)

from finetuning_utils import compute_metrics

from typing import Dict, Optional


def finetune_and_evaluate(
    model_name: str,
    out_dir: str,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    data_train: Dataset,
    data_dev: Dataset,
    data_test: Dataset,
    # hyperparameters
    n_epochs: int=10,
    learning_rate: float=2e-5,
    train_batch_size: int=8,
    weight_decay: float=0.01,
    # for early stopping (if dev set is provided)
    metric: Optional[str]=None,
    greater_is_better: Optional[bool]=None,
    # reproducibility
    seed: int=42,
):
    print('Finetuning', model_name)
    
    checkpoints_dir = os.path.join(out_dir, 'checkpoints')
    
    # define train args
    train_args = dict(
        output_dir=checkpoints_dir,
        # hyperparameters
        num_train_epochs=n_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=train_batch_size*2,
        weight_decay=weight_decay,
        warmup_ratio=0.1,
        optim='adamw_torch',
        # how to select "best" model
        metric_for_best_model=metric if data_dev is not None else None,
        greater_is_better=greater_is_better,
        load_best_model_at_end=True if data_dev is not None else False,
        save_total_limit=2,
        # when to evaluate
        eval_strategy='epoch' if data_dev is not None else 'no',
        # when to save
        save_strategy='epoch' if data_dev is not None else 'no',
        # logging
        logging_strategy='no',
        # where to store results
        report_to='none',
        # misc
        fp16=torch.cuda.is_available() and 'deberta' not in model_name,
        # reproducibility
        seed=seed,
        data_seed=seed,
        full_determinism=True
    )
    train_args = TrainingArguments(**train_args)

    def model_init():
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.num_labels = len(label2id)
        config.label2id = label2id
        config.id2label = {v: k for k, v in label2id.items()}
        return AutoModelForTokenClassification.from_pretrained(model_name, config=config, device_map='auto', torch_dtype='auto', trust_remote_code=True)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True, add_prefix_space=True)
    
    callbacks = []
    if data_dev is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.03))

    # create Trainer
    trainer = Trainer( 
        model_init=model_init,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        args=train_args,
        train_dataset=data_train,
        eval_dataset=data_dev if data_dev is not None else None,
        compute_metrics=lambda p: compute_metrics(p, id2label),
        callbacks=callbacks
    )
    print('using devive:', str(trainer.model.device))
    
    trainer.train()
    
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    # trainer.log_metrics('train', trainer.state.log_history[-1])

    if data_test is not None:
        test_res = trainer.evaluate(data_test, metric_key_prefix='test')
        with open(os.path.join(out_dir, 'test_results.json'), 'w') as f:
            json.dump(test_res, f, indent=2)
    
    # finally: clean up
    if os.path.exists(checkpoints_dir):
        shutil.rmtree(checkpoints_dir)

# #### Load libraries

def main(args):

    if isinstance(args.train_data_files, str):
        args.train_data_files = [f.strip() for f in args.train_data_files.split(',')]
    if isinstance(args.dev_data_files, str):
        args.dev_data_files = [f.strip() for f in args.dev_data_files.split(',')]
    if isinstance(args.test_data_files, str):
        args.test_data_files = [f.strip() for f in args.test_data_files.split(',')]

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True, use_fast=True, add_prefix_space=True)
    
    # load the data
    if args.train_data_files is not None:
        data_train = read_jsonl(args.train_data_files[0], replace_newlines=True)
        if len(args.train_data_files) > 1:
            for f in args.train_data_files[1:]:
                data_train += read_jsonl(f, replace_newlines=True)
        # creat the training split
        train_dataset = create_token_classification_dataset(data_train)
        train_dataset = train_dataset.map(lambda example: tokenize_and_align_sequence_labels(example, tokenizer=tokenizer), batched=True)
        train_dataset = train_dataset.select_columns(['labels', 'input_ids', 'attention_mask'])
    
    if args.dev_data_files is not None:
        data_dev = read_jsonl(args.dev_data_files[0], replace_newlines=True)
        if len(args.dev_data_files) > 1:
            for f in args.dev_data_files[1:]:
                data_dev += read_jsonl(f, replace_newlines=True)
        # creat the validation split
        dev_dataset = create_token_classification_dataset(data_dev)
        dev_dataset = dev_dataset.map(lambda example: tokenize_and_align_sequence_labels(example, tokenizer=tokenizer), batched=True)
        dev_dataset = dev_dataset.select_columns(['labels', 'input_ids', 'attention_mask'])
    else:
        dev_dataset = None
    
    if args.test_data_files is not None:
        data_test = read_jsonl(args.test_data_files[0], replace_newlines=True)
        if len(args.test_data_files) > 1:
            for f in args.test_data_files[1:]:
                data_test += read_jsonl(f, replace_newlines=True)
        # creat the test split
        test_dataset = create_token_classification_dataset(data_test)
        test_dataset = test_dataset.map(lambda example: tokenize_and_align_sequence_labels(example, tokenizer=tokenizer), batched=True)
        test_dataset = test_dataset.select_columns(['labels', 'input_ids', 'attention_mask'])
    else:
        test_dataset = None
    
    if os.path.exists(args.model_dir):
        if args.overwrite:
            print(f"Experiment folder {args.model_dir} already exists. Overwriting...")
            shutil.rmtree(args.model_dir)
        else:
            raise ValueError(f"Experiment folder {args.model_dir} already exists. Please remove it or choose a different name.")
    else:
        os.makedirs(args.model_dir)

    set_seed(args.seed)
    finetune_and_evaluate(
        # model config
        model_name=args.checkpoint,
        label2id=label2id,
        id2label=id2label,
        # output dir
        out_dir=args.model_dir,
        # data
        data_train=train_dataset,
        data_dev=dev_dataset,
        data_test=test_dataset,
        # hyperparameters
        n_epochs=args.epochs,
        learning_rate=args.learning_rate,
        train_batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        # for early stopping (if dev set is provided)
        metric='seqeval-social group_f1' if dev_dataset is not None else None,
        greater_is_better=True if dev_dataset is not None else None,
        # reproducibility
        seed=args.seed,
    )

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Hyperparameter search for token classification')
    
    parser.add_argument('checkpoint', type=str, help='Model to finetune')
    parser.add_argument('-d', '--model_dir', type=str, default='hyperparameter_search', help='Name of the experiment')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing experiment results')
    
    parser.add_argument('--train_data_files',  nargs='+', type=str, required=True)
    parser.add_argument('--dev_data_files',  nargs='+', type=str, default=None)
    parser.add_argument('--test_data_files',  nargs='+', type=str, default=None)

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--weight_decay', type=float, default=0.01)

    parser.add_argument('--seed', type=int, default=1234)

    args = parser.parse_args()

    main(args)
else:
    # CLI arguments
    from types import SimpleNamespace

    args = SimpleNamespace()

    args.checkpoinz = 'roberta-base'
    args.model_dir = 'roberta-base-group-mention-detector-uk-manifestos'
    args.overwrite = True

    args.train_data_files = 'splits/train.jsonl splits/dev.jsonl'
    args.dev_data_files = None
    args.dev_data_files = 'splits/test.jsonl'
    
    args.epochs = 10
    args.learning_rate = 2e-5
    args.train_batch_size = 8
    args.weight_decay = 0.01
    
    args.seed = 1234
