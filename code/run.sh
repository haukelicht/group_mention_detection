#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24G
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hauke.licht@uibk.ac.at
#SBATCH --job-name=run
#SBATCH --output=logs/%x.log
#SBATCH --error=logs/%x.err

module load stack/2024-06  gcc/12.2.0  python_cuda/3.11.6 eth_proxy

# helpers for logging
ts() { date '+%Y-%m-%d %H:%M:%S'; }
message() { echo -e "[$(ts)] $1"; }

source ./../venv/bin/activate

export HF_HOME=$(pwd)/../.hf_models


# python -c -u "import torch; print(torch.cuda.memory_summary())"

message "Running model comparison and hypter-parameter search"
python run_model_comparison_experiment.py

# message "Training token classifier on labeled sentences from UK manifestos"
# python finetune_token_classifier.py

# message "Applying token classifier to unlabeled UK manifesto sentences"
# python inference_token_classifier.py


# message "Training CAP classifier on labeled sentences from UK Con and Lab manifestos"
# python finetune_sequence_classifier.py

# message "Applying CAP classifier to unlabeled UK manifesto sentences"
# python inference_cap_classifier.py

message "Done!"
