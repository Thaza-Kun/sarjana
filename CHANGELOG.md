# Changelog

## 0.2.0 2023-01-10
Report number `#19` - `#41`

### Added
- Added algorithm scoring and logging for debugging and reproducibility.
- Added journal club presentation file
- Added workflow files to replicate different results with arbitrary algorithm order.
  - UMAP--HDBSCAN
  - HDBSCAN on important features
- Documented code functions.
- Extracted the list of repeater candidate from 3 sources (Chen 2021, Luo Jia-Wei 2022, and Zhu-Ge 2022)
- Implemented function to calculate: 
  - energy 
  - brightness temperature
  - luminosity distance
  - radio luminosity
- Tried to replicate Cui, Xiang-Han et al (2020) but failed.

## Bugs
- Fixed major bug in UMAP--HDBSCAN workflow where HDBSCAN is not clustering properly.

### Literature
- Read [Bailes, Matthew 2022, The discovery and scientific potential of fast radio bursts](https://www.science.org/doi/10.1126/science.abj3043).This is just an overview of current FRB research with nothing particularly of note. The author works closely with Lorimer so they wrote in detail about how the Lorimer burst came about and how FRB research expanded after that.
- Read [Cui, Xiang-Han et. al. 2021, Fast radio bursts: do repeaters and non-repeaters originate in statistically similar ensembles?](https://doi.org/10.1093/mnras/staa3351)
- Read [Cui, Xiang-Han et. al. 2021, Statistical properties of fast radio bursts elucidate their origins: magnetars are favored over gamma-ray bursts](https://doi.org/10.1088/1674-4527/21/8/211)
- Read [Hashimoto, Tetsuya et. al. 2019, Luminosity–duration relation of fast radio bursts](https://doi.org/10.1093/mnras/stz1715)
- Read [Luo, Rui et. al. 2020, On the FRB luminosity function – – II. Event rate density](https://doi.org/10.1093/mnras/staa704)
- Read [Ivezić et. al. 2020, Statistics, data mining, and machine learning in astronomy: a practical Python guide for the analysis of survey data](https://press.princeton.edu/books/hardcover/9780691198309/statistics-data-mining-and-machine-learning-in-astronomy) Chapter 4.7
- Read [Eric Feigelson & G. Jogesh Babu 2012, Modern Statistics for Astronomy: With R Applications](https://astrostatistics.psu.edu/MSMA/) Chapter 3-3.4.5

## 0.1.3 2022-11-12
Report number `#7` - `#18`

### Added
- Fully implemented Chen et. al's (2021) UMAP and HDBSCAN algorithm.
- Exported UMAP-HDBSCAN to script

### Literature
- Read [Luo, Jia-Wei et. al. 2022, Machine learning classification of CHIME fast radio bursts: I. Supervised Methods](https://doi.org/10.1093/mnras/stac3206)
- Read [Zhu-Ge, Jia-Ming et. al. 2022, Machine Learning Classification of Fast Radio Bursts: II. Unsupervised Methods](http://arxiv.org/abs/2210.02471)
- Read [CHIME/FRB 2020, Periodic activity from a fast radio burst source](doi:10.1038/s41586-020-2398-2)
- Read [Jake VanderPlas 2022, Python Data Science Handbook](https://www.oreilly.com/library/view/python-data-science/9781491912126/)

## 0.1.2 2022-10-25
Report number `#7` - `#18`

### Added
- Reproduction of Chen et. al.'s (2021) UMAP dimension reduction in [#3](https://github.com/Thaza-Kun/sarjana/issues/3). Have not yet implemented HDBSCAN.

### Literature
- Read [Uncloaking hidden repeating fast radio bursts with unsupervised machine learning](https://ui.adsabs.harvard.edu/abs/2022MNRAS.509.1227C/abstract) by Bo Han Chen et. al. (2021). Refer to https://github.com/Thaza-Kun/sarjana/issues/3#issuecomment-1280387079
  - Skimmed through [UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction](https://ui.adsabs.harvard.edu/abs/2018arXiv180203426M/abstract) by McInnes et. al. (2018) to understand the algorithm used in Chen2021. The algortihm is implemented in https://github.com/lmcinnes/umap


## 0.1.1 2022-10-14
Report number `#1` - `#6`

### Added
- Used [Quarto](https://quarto.org) for authoring
  * Converted proposal written in docs to quarto markdown. [#2](https://github.com/Thaza-Kun/sarjana/issues/2)
- Data fetching from [FRBSTATS](https://www.herta-experiment.org/frbstats/) in [`notebooks/01-get-data.ipynb`](notebooks/01-get-data.ipynb)