# Group mention detection with supervised learning


This repository contains the replication code and data for the paper "Detecting Group Mentions in Political Rhetoric: A Supervised Learning Approach" ([preprint]([ufb96](https://osf.io/ufb96/))) by Hauke Licht (hauke.licht@uibk.ac.at) and Ronja Sczepanski (ronja.sczepanski@sciencespo.fr) forthcoming in the *British Journal of Political Science*.

> **Abstract**\
> Politicians appeal to social groups to court their electoral support. However, quantifying which groups politicians refer to, claim to represent, or address in their public communication presents researchers with challenges. We propose a supervised learning approach for extracting group mentions from political texts. We first collect human annotations to determine the passages of a text that refer to social groups. We then fine-tune a transformer language model for contextualized supervised classification at the word level. Applied to unlabeled texts, our approach enables researchers to automatically detect and extract word spans that contain group mentions. We illustrate our approach in two applications, generating new empirical insights into how British parties use social groups in their rhetoric. Our method allows for detecting and extracting mentions of social groups from various sources of texts, creating new possibilities for empirical research in political science.

Please cite the paper when using code or data from this repository:

```latex
@misc{licht_detecting_2025,
	author = {Licht, Hauke and Sczepanski, Ronja},
	year = {2025},
	title = {Detecting Group Mentions in Political Rhetoric: A Supervised Learning Approach},
	shorttitle = {Detecting Group Mentions in Political Rhetoric},
	journal = {{British} {Journal} of {Political} {Science}},
	keywords = {social groups, political rhetoric, computational text analysis, supervised classification}
}
```

## Notebooks

The [notebooks](/notebooks) folder contains Jupyter notebooks that demonstrate how to implement and validate the group mention detection method we propose.
