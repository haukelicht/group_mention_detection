# Release of finetuned token classification models

This folder contains all scripts used to finetune models on the group mention annotations collected for the paper 

> Licht, Hauke and Ronja Sczepanski. 2025. "Detecting Group Mentions in Political Rhetoric: A Supervised Learning Approach" *British Journal of Political Science*, online first.

These models will be released in the Hugging Face model hub.

## Data

We only provide models finetuned on the model-aggregated human group mention annotations for our UK manifesto sentences corpus (see [data file](../replication/data/annotation/labeled/uk-manifestos_all_labeled.jsonl)).

The other two annotated datasets we contribute (UK *House of Commons* parliamentary speech transcripts sentences and German party manifesto sentences) are too small for finetuning reliable, domain-specific models.

## Scripts overview

1. **data splitting** 
    - `data_splitting.py`: python script implementing an embedding-based data splitting strategy that minimizes data leakage between the train, dev, and test sets at the mention level (because any unique verbatim mention might occur in multiple sentences in our data)
    - `split_data.sh`: shell script for generating train, dev, and test data splits of roughly 60%, 20% and 20% of sentences of the data, respecitvely. The resulting splits area written to [splits/](./splits/)
2. **hyper-parameter search**
    - `hyperparameter_search.py`: python script implementing hyperparameter search for token classifier finetuning using `optuna`. The script hard-codes the set of candidate values to examine: 
    ```python
    trial_learning_rates = [1e-6, 5e-6, 1e-5, 3e-5, 5e-5]
    trial_train_batch_sizes = [8, 16, 32]
    trial_weight_decays = [0.01, 0.1, 0.3]
    ```
    - `search_hyperparameters_base.sh`: Performs 20 hyperparameter search trials on the train and dev splits for three "base" models each: `roberta-base`, `answerdotai/ModernBERT-base`, and `EuroBERT/EuroBERT-610m`. Allows identifying the "optimal" learning rate, training batch size, and weight decay values for each model based on dev set performance.
    - `search_hyperparameters_large.sh`: Performs 20 hyperparameter search trials on the train and dev splits for three "large" models each: `roberta-large`, `answerdotai/ModernBERT-large`, `EuroBERT/EuroBERT-2.1B`. Allows identifying the "optimal" learning rate, training batch size, and weight decay values for each model based on dev set performance. 
3. **model finetuning**
    - `finetune_token_classifier.py`: python script implementing token classifier finetuning.
    - `finetune_base_models.sh`: shell script finetunes base models (`roberta-base`, `answerdotai/ModernBERT-base`, and `EuroBERT/EuroBERT-610m`) each with the "optimal" hyperparameters identified through hyperparameters search
    - `finetune_large_models.sh`: shell script finetunes base models (`roberta-large`, `answerdotai/ModernBERT-large`, and `EuroBERT/EuroBERT-2.1B`) each with the "optimal" hyperparameters identified through hyperparameters search


## Reproducibility

### System information 

CUDA (`nvcc --version`):

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Tue_Oct_29_23:50:19_PDT_2024
Cuda compilation tools, release 12.6, V12.6.85
Build cuda_12.6.r12.6/compiler.35059454_0
```

### Python setup

```bash
conda create -y -n group_mention_detection python=3.10 pip
conda activate group_mention_detection
pip install -r requirements.txt
```

