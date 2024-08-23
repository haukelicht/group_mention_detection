#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24G
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hauke.licht@uibk.ac.at
#SBATCH --job-name=finetune_uk-man_token_classifier
#SBATCH --output=logs/%x.log
#SBATCH --error=logs/%x.err


# ~+~+~ Setup +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #

module load stack/2024-06  gcc/12.2.0  python_cuda/3.11.6 eth_proxy

# helpers for logging
ts() { date '+%Y-%m-%d %H:%M:%S'; }
message() { echo -e "[$(ts)] $1"; }

source ./../venv/bin/activate

export HF_HOME=$(pwd)/../.hf_models


# ~+~+~ Experiments ~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #

# message "Running model comparison and hypter-parameter search"
# python run_model_comparison_experiment.py # takes ~3h

# message "Running 5x5 cross-validation experiment"
# python run_5x5_crossval_experiment.py # takes ~4h

# message "Running training size experiment"
# python run_training-size_experiment.py # takes ~3h

# message "Running cross-party transfer experiment"
# python run_cross-party-transfer_experiment.py # takes ~1h

# message "Running cross-domain transfer experiment"
# python run_cross-domain-transfer_experiment.py # takes ~1h

# message "Running cross-lingual transfer experiment"
# python run_cross-lingual-transfer_experiment.py # takes ~1h

# ~+~+~ Fine-tuning and inference ~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #

message "Training token classifier on labeled sentences from UK manifestos"
python finetune_token_classifier.py # takes ~0:01h

message "Applying token classifier to unlabeled UK manifesto sentences"
python inference_token_classifier.py # takes ~0:01h


# ~+~+~ Measurement validation and benchmarking ~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #

# message "Training CAP classifier on labeled sentences from UK Con and Lab manifestos"
# python finetune_sequence_classifier.py

# message "Applying CAP classifier to unlabeled UK manifesto sentences"
# python inference_cap_classifier.py


message "Done!"

