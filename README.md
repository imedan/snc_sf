# SDSS-V Solar Neighborhood Census Selection Function and Subpopulation Probabilities

This repository provides the code to calculate the selection function for the SDSS-V Solar Neighborhood Census (SNC) relative to the Gaia Catalog of Nearby Stars ([GCNS](https://ui.adsabs.harvard.edu/abs/2021A%26A...649A...6G/abstract)). With this selection function, the code allows for the forward modeling of subpopulation probabilities across the HR diagram. A use case would be selecting all stars in the SNC with [Fe/H] < -1 and the forward model would evaluate the likely probability of selecting [Fe/H] < -1 stars from the GCNS across the HR diagram.

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

A number of examples using the code are located in [`notebooks/`](https://github.com/imedan/snc_sf/tree/main/notebooks). These examples are fully explained in the paper accompanying this work.