# Changelog

## Upcoming

### TODO

## 0.1.3 2022-11-12

### Added
- Fully implemented Chen et. al's (2021) UMAP and HDBSCAN algorithm.
- Exported UMAP-HDBSCAN to script

### Literature
- Read [Luo, Jia-Wei et. al. 2022, Machine learning classification of CHIME fast radio bursts: I. Supervised Methods](https://doi.org/10.1093/mnras/stac3206)
- Read [Zhu-Ge, Jia-Ming et. al. 2022, Machine Learning Classification of Fast Radio Bursts: II. Unsupervised Methods](http://arxiv.org/abs/2210.02471)
- Read [CHIME/FRB 2020, Periodic activity from a fast radio burst source](doi:10.1038/s41586-020-2398-2)
- Read [Jake VanderPlas 2022, Python Data Science Handbook](https://www.oreilly.com/library/view/python-data-science/9781491912126/)

## 0.1.2 2022-10-25

### Added
- Reproduction of Chen et. al.'s (2021) UMAP dimension reduction in [#3](https://github.com/Thaza-Kun/sarjana/issues/3). Have not yet implemented HDBSCAN.

### Literature
- Read [Uncloaking hidden repeating fast radio bursts with unsupervised machine learning](https://ui.adsabs.harvard.edu/abs/2022MNRAS.509.1227C/abstract) by Bo Han Chen et. al. (2021). Refer to https://github.com/Thaza-Kun/sarjana/issues/3#issuecomment-1280387079
  - Skimmed through [UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction](https://ui.adsabs.harvard.edu/abs/2018arXiv180203426M/abstract) by McInnes et. al. (2018) to understand the algorithm used in Chen2021. The algortihm is implemented in https://github.com/lmcinnes/umap


## 0.1.1 2022-10-14

### Added
- Used [Quarto](https://quarto.org) for authoring
  * Converted proposal written in docs to quarto markdown. [#2](https://github.com/Thaza-Kun/sarjana/issues/2)
- Data fetching from [FRBSTATS](https://www.herta-experiment.org/frbstats/) in [`notebooks/01-get-data.ipynb`](notebooks/01-get-data.ipynb)