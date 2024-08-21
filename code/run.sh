#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24G
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hauke.licht@uibk.ac.at
#SBATCH --job-name=sentence_split_esp
#SBATCH --output=%x.log
#SBATCH --error=%x.err

module load stack/2024-06  gcc/12.2.0  python_cuda/3.11.6 eth_proxy

source ./../venv/bin/activate

HF_HOME=./../.hf_models
TRANSFORMERS_CACHE=./../.hf_models

python run_model_comparison_experiment.py



