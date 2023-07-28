# Group mention detection with supervised learning


This repository contains the code for the paper "Who are they talking about? Detecting mentions of social groups in political texts with supervised learning" ([preprint]([ufb96](https://osf.io/ufb96/)))by Hauke Licht (hauke.licht@wiso.uni-koeln.de) and Ronja Sczepanski (ronja.sczepanski@eup.gess.ethz.ch).

> **Abstract**\
> Politicians appeal to social groups to court their electoral support and secure their political survival. But quantifying which groups politicians refer to, claim to represent, or address in their public communication presents researchers with challenges. We propose a novel supervised learning approach for identifying group mentions in political texts. We first collect human annotations to determine the exact text passages that refer to social groups. We then fine-tune a Transformer language model for contextualized supervised classification at the word level. Applied to unlabeled texts, our approach enables researchers to detect and extract word spans that contain group mentions automatically. We illustrate the reliability, validity, and flexibility of our approach in a study of British parties’ election manifestos, parliamentary questions, as well as German party manifestos. Our application demonstrates that our method en- ables highly reliable retrieval of group mentions at scale and new quantitative insights into group-based political rhetoric.


Please cite the paper when using code or data from this repository:

```latex
@misc{licht_who_2023,
	author = {Licht, Hauke and Sczepanski, Ronja},
	year = {2023},
	date = {2023-06-20},
	title = {Who are they talking about? Detecting mentions of social groups in political texts with supervised learning},
	shorttitle = {Who are they talking about?},
	url = {https://osf.io/ufb96/},
	doi = {10.31219/osf.io/ufb96},
	publisher = {{OSF} Preprints},
	keywords = {social groups, political rhetoric, computational text analysis, supervised classification}
}
```

## Notebooks

The [notebooks](/notebooks) folder contains Jupyter notebooks that demonstrate how to implement and validate the group mention detection method we propose.
