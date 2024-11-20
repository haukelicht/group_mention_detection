# Replication materials

The [`code/`](./code), [`data/`](./code), and [`results/`](./result) folders in [`replication/`](./replication) contain the replication code, data, and results.
These materials can also be obtained from Harvard Dataverse: https://doi.org/10.7910/DVN/QCOQ0T

# Requirements

## Software 

The code was written in R (4.2.2) and Python (3.10). 

The **python environment** is managed with `conda`.
The required python libraries and versions are documented in the file [`python_requirements.txt`](./python_requirements.txt).
To replicate the python setup, run the following command in the terminal:

```bash
conda create -y --name group_mention_detection python=3.10 pip
conda activate group_mention_detection
pip install -r python_requirements.txt
```

The **R environment** is managed by `renv`.
The required R packages and versions are documented in the file [`r_requirements.txt`](./r_requirements.txt):
To replicate the python setup, run the following command in the terminal in R:

```R
devtools::install_version("renv", version = "0.15.5")
library(renv)
init(bare = TRUE)
pkgs <- readLines("r_requirements.txt")
install(pkgs)
```

## Hardware

We ran all analyses requiring GPU computing (model fine-tuning and inference) on a HPC cluster using SLURM jobs with a single NVIDIA RTX 4090 (24GB GPU memory).

All other code was run on a MacBook Pro 2021 with an Apple M1 Chip and 32 GB RAM.
