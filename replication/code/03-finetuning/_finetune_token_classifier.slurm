#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hauke.licht@uibk.ac.at
#SBATCH --job-name=token_classifier_finetuning
#SBATCH --output=logs/%x.log
#SBATCH --error=logs/%x.err


# ~+~+~ Setup +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #

module load stack/2024-06  gcc/12.2.0  python_cuda/3.11.6 eth_proxy

# helpers for logging
ts() { date '+%Y-%m-%d %H:%M:%S'; }
message() { echo -e "[$(ts)] $1"; }

source ./../../venv/bin/activate

export HF_HOME=$(pwd)../../.hf_models


# ~+~+~ Token classifier fine-tuning and inference +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #

message "Training token classifier on labeled sentences from UK manifestos"
python finetune_token_classifier.py # takes ~0:10h

message "Applying token classifier to unlabeled UK manifesto sentences"
python inference_token_classifier.py # takes ~0:01h


message "Done!"

