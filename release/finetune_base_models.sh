#!/bin/bash

MODEL="roberta-base"
python finetune_token_classifier.py $MODEL \
    -d "models/$MODEL-group-mention-detector-uk-manifestos" --overwrite \
    --train_data_files splits/train.jsonl splits/dev.jsonl \
    --test_data_files splits/test.jsonl \
    --epochs 6 --learning_rate 5e-05 --batch_size 16 --weight_decay 0.3

MODEL="answerdotai/ModernBERT-base"
python finetune_token_classifier.py $MODEL \
    -d "models/$MODEL-group-mention-detector-uk-manifestos" --overwrite \
    --train_data_files splits/train.jsonl splits/dev.jsonl \
    --test_data_files splits/test.jsonl \
    --epochs 2 --learning_rate 5e-05 --batch_size 8 --weight_decay 0.3

MODEL="EuroBERT/EuroBERT-610m"
python finetune_token_classifier.py $MODEL \
    -d "models/$MODEL-group-mention-detector-uk-manifestos" --overwrite \
    --train_data_files splits/train.jsonl splits/dev.jsonl \
    --test_data_files splits/test.jsonl \
    --epochs 5 --learning_rate 1e-05 --batch_size 32 --weight_decay 0.01
