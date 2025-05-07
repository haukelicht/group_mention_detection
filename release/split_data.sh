#!/bin/bash

DATAPATH="../replication/data/annotation/labeled"

python3 data_splitting.py \
    --input_file "$DATAPATH/uk-manifestos_all_labeled.jsonl" \
    --splitter 'minoverlap' \
    --test_size 0.15 --dev_size 0.10 \
    --output_dir "splits" \
    --seed 1234
