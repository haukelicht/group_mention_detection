# Social group mention detection with supervised learning

This repository contains the code and data for the paper "Detecting Group Mentions in Political Rhetoric: A Supervised Learning Approach" forthcoming at the *British Journal of Political Science* by Hauke Licht (hauke.licht@uibk.ac.at) and Ronja Sczepanski (ronja.sczepanski@sciencespo.fr).

**Abstract**<br>
> Politicians appeal to social groups to court their electoral support. 
> However, quantifying which groups politicians refer to, claim to represent, or address in their public communication presents researchers with challenges. 
> We propose a supervised learning approach for extracting group mentions from political texts.
> We first collect human annotations to determine the passages of a text that refer to social groups.
> We then fine-tune a transformer language model for contextualized supervised classification at the word level.
> Applied to unlabeled texts, our approach enables researchers to automatically detect and extract word spans that contain group mentions.
> We illustrate our approach in two applications, generating new empirical insights into how British parties use social groups in their rhetoric.
> Our method allows detecting and extracting mentions of social groups from various sources of texts, creating new possibilities for empirical research in political science.

The paper **preprint** is avaialble at OSF: https://osf.io/ufb96/ <br>
The replication materials are also available at **Harvard Dataverse**: https://doi.org/10.7910/DVN/QCOQ0T

Please **cite the paper** when using code or data from this repository:

```bibtex
@article{licht_detecting_2024,
	author = {Licht, Hauke and Sczepanski, Ronja},
	year = {forthcoming},
	title = {Detecting Group Mentions in Political Rhetoric: A Supervised Learning Approach},
	shorttitle = {Detecting Group Mentions in Political Rhetoric},
	url = {https://osf.io/ufb96/},
	journal = {British Journal of Political Science},
	volume = {},
	number = {},
	keywords = {social groups, political rhetoric, computational text analysis, supervised classification}
}
```

## Notebooks

The [`notebooks/`](./notebooks) folder contains Jupyter notebooks that demonstrate how to implement and validate the group mention detection method we propose.

## Replication materials

The [`code/`](./code), [`data/`](./code), and [`results/`](./result) folders contain the replication code, data, and results.
These materials can also be obtained from Harvard Dataverse: https://doi.org/10.7910/DVN/QCOQ0T


## Requirements

### Software 

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

### Hardware

We ran all analyses requiring GPU computing (model fine-tuning and inference) on a HPC cluster using SLURM jobs with a single NVIDIA RTX 4090 (24GB GPU memory).

All other code was run on a MacBook Pro 2021 with an Apple M1 Chip and 32 GB RAM.
