
MODELS="roberta-large answerdotai/ModernBERT-large"
DATAPATH="splits"
N_TRIALS=20
EXPERIMENT_NAME="large-model-comparison"
EXPERIMENT_RESULTS_PATH="results"

python hyperparameter_search.py \
    --model_names $MODELS \
    --n_trials $N_TRIALS \
    --train_data_file "$DATAPATH/train.jsonl" --dev_data_file "$DATAPATH/dev.jsonl" \
    --experiment_name $EXPERIMENT_NAME \
    --experiment_results_path $EXPERIMENT_RESULTS_PATH \
    --overwrite
