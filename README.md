# TTI Explorer - Sensitivity Analysis and Optimisation (COVID-19)

This is a repository for `tti-explorer` simulation analysis. This repository is built to analyse and explore the impact of various test-trace-isolate strategies and social distancing measures on the spread of COVID-19 in the UK. It also aims to employ unconstrained and constrained optimisation techniques to discover optimal strategy for reducing the disease' effective reproduction number.

**Note**:
- the `tti-explorer` library that contains the simulation code can be found on [tti-explorer](https://github.com/rs-delve/tti-explorer).
- Accompanying papers include [Kucharski et al. (2020)](https://www.medrxiv.org/content/10.1101/2020.04.23.20077024v1), [Klepac et al. (2018)](https://researchonline.lshtm.ac.uk/id/eprint/4647173/), [He et al. (2020)](https://rs-delve.github.io/pdfs/2020-05-27-effectiveness-and-resource-requirements-of-tti-strategies.pdf), [The Delve Initiative (2020)](https://rs-delve.github.io/reports/2020/05/27/test-trace-isolate.html).

## Requirements:
### tti_explorer
- Python 3.6+
- numpy
- scipy
- pandas
- matplotlib
- dataclasses (for Python 3.6)
### scripts, tests and notebooks
- jupyter
- tqdm
- pytest

## Folder Structure:
- *data*: contains datasets used for the simulation.
- *tti_explorer*: contains related simulations codes.
- *notebooks*: contains analysis codes (sensitivity analysis, causal analysis, policy optimisation).
- *results*: contains experiments results (in csv and pickle).
- *paper.pdf*: our paper documenting methods and results.


## Setup:
```bash
git clone https://github.com/rs-delve/tti-explorer
cd tti-explorer
pip install -r requirements.txt
pip install .
```

## Authors:
- Maleakhi Wijaya: maw219@cam.ac.uk
- Chuan Tan: ct538@cam.ac.uk
- Jakub Mach: jakub.t.mach@gmail.com
