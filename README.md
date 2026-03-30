# SDSS-V Solar Neighborhood Census Selection Function and Subpopulation Probabilities

This repository provides the code to calculate the selection function for the SDSS-V Solar Neighborhood Census (SNC) relative to the Gaia Catalog of Nearby Stars ([GCNS](https://ui.adsabs.harvard.edu/abs/2021A%26A...649A...6G/abstract)). With this selection function, the code allows for the forward modeling of subpopulation probabilities across the HR diagram. A use case would be selecting all stars in the SNC with [Fe/H] < -1 and the forward model would evaluate the likely probability of selecting [Fe/H] < -1 stars from the GCNS in discrete bins across the HR diagram.

## Installation

The code can be installed with `pip`:
```
git clone https://github.com/imedan/snc_sf
cd snc_sf
pip install .
```
or in a fresh virtual environment with `poetry` to fully replicate the development environment
```
git clone https://github.com/imedan/snc_sf
cd snc_sf
conda create -n "snc_sf" python=3.11
conda activate snc_sf
pip install poetry
poetry install
```

To fully utilize the code, SDSS-V and GCNS data is need. The SDSS-V DR19 `astra` summary data can be accessed [here](https://data.sdss.org/sas/dr19/spectro/astra/0.6.0/summary/). The required GCNS data will automatically be downloaded when first initializing a `snc_sf.selection_function.SNCSelectionFunction()` object.

## Examples

A number of examples using the code are located in [`notebooks/`](https://github.com/imedan/snc_sf/tree/main/notebooks). These examples are fully explained in the paper accompanying this work. In summary, the examples included related to the following:
- [`dr19_obs.ipynb`](https://github.com/imedan/snc_sf/blob/main/notebooks/dr19_obs.ipynb): Constructs the base SNC 100 pc dataset and creates the plots in Figure 1 of the paper.
- [`SF_example_plots.ipynb`](https://github.com/imedan/snc_sf/blob/main/notebooks/SF_example_plots.ipynb): Example of the selecrtion function for different regions on the sky, as shown in Figure 2 of the paper.
- [`forward_model_GCNS_mock_proof.py`](https://github.com/imedan/snc_sf/blob/main/notebooks/forward_model_GCNS_mock_proof.py): Script that replicates the example with mock data in Section 4.1 and Figure 3 in the paper.
- [`forward_model_halpha_ex.py`](https://github.com/imedan/snc_sf/blob/main/notebooks/forward_model_halpha_ex.py): Script that replicates the example examing the distribution of H-alpha emitters across the HR diagram in Section 4.2 and Figure 4 in the paper.
- [`forward_model_MF.py`](https://github.com/imedan/snc_sf/blob/main/notebooks/forward_model_MF.py): Script that replicates the example that calculates the mass function in bins of metallicity in Section 4.3 in the paper.
- [`plot_mf_results.py`](https://github.com/imedan/snc_sf/blob/main/notebooks/plot_mf_results.py): Script that creates the mass function plots in Figure 5 in the paper.
- [`appendix_example.ipynb`](https://github.com/imedan/snc_sf/blob/main/notebooks/appendix_example.ipynb): Example to demonstrate basic code usage as illustrated in the Appendix of the paper.