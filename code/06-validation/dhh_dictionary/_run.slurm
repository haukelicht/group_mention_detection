#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hauke.licht@uibk.ac.at
#SBATCH --job-name=validation_dhh_dictionary
#SBATCH --output=logs/%x.log
#SBATCH --error=logs/%x.err


module load stack/2024-06  gcc/12.2.0  python/3.11.6 eth_proxy

# helpers for logging
ts() { date '+%Y-%m-%d %H:%M:%S'; }
message() { echo -e "[$(ts)] $1"; }

source ./../../../venv/bin/activate

export HF_HOME=$(pwd)/../../../.hf_models

message "Evaluating DDH dictionary in test set of our human-labeled annotations"
python apply_dhh_dictionary_uk-manifestos.py
python evaluate_dhh_dictionary_uk-manifestos.py

message "Evaluating DDH dictionary in annotations we could reconstruct from Thau's (2019) data"
python apply_dhh_dictionary_thau2019-manifestos.py
python evaluate_dhh_dictionary_thau2019-manifestos.py

message "Evaluating (naively) expanded DDH dictionaries in annotations we could reconstruct from Thau's (2019) data"
python evaluate_expanded_dhh_dictionaries_thau2019-manifestos.py

message "Done!"

