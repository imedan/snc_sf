from astropy.coordinates import SkyCoord
import astropy.units as u
import healpy as hp
import numpy as np
import polars as pl
from importlib.resources import open_binary, files
import ast
from scipy.sparse import coo_matrix
from typing import Tuple
from astroquery.vizier import Vizier
from astroquery.gaia import Gaia


def download_gcns_data(file_type: str):
    """
    Download any missing GCNS data from Vizier

    Parameters
    ---------
    file_type: str
        Which table from GCNS to download. Options are
        'selected' (the selected objects in the catalog),
        'maglim' (the magnitude limit healpix map), or
        'distpdf' (the distance PDF of each sample)
    """
    if file_type == 'selected':
        viziertab = 'J/A+A/649/A6/table1c'
        savefile = files('snc_sf.sf_files') / 'GCNS-result.csv'
    elif file_type == 'maglim':
        viziertab = 'J/A+A/649/A6/maglim'
        savefile = files('snc_sf.sf_files') / 'GCNS_healpix_maglim.fit'
    elif file_type == 'distpdf':
        viziertab = 'J/A+A/649/A6/distpdf'
        savefile = files('snc_sf.sf_files') / 'GNSC_distpdf.csv'
    else:
        raise ValueError('Not a valid entry for file_type!')

    if file_type == 'selected':
        Gaia.ROW_LIMIT = -1
        gcns_tab = Gaia.load_table('external.gaiaedr3_gcns_main_1')
        job = Gaia.launch_job_async('select * from external.gaiaedr3_gcns_main_1')
        gcns = [job.get_results()]
    else:
        vizier = Vizier(columns=["**"], row_limit=-1)
        res = vizier.get_catalogs_async(viziertab)
        gcns = Vizier._parse_result(res)
    if file_type == 'maglim':
        gcns[0].write(savefile, format='fits')
    else:
        gcns[0].write(savefile, format='csv')


def coord2healpix(coord: SkyCoord, nside: int, nest: bool = True) -> np.ndarray:
    """
    Calculate the HealPix index for a set of coordinates

    Parameters
    ----------
    coord: astropy.coordinates.SkyCoord
        Astropy coordinates of the data
    
    nside: int
        HealPix nside for the transformation

    nest: bool
        If to do nested or not

    Returns
    -------
    hpind: np.ndarray
        HealPix indencies of the coordinates.
    """
    if hasattr(coord, "ra"):
        phi = coord.ra.rad
        theta = 0.5 * np.pi - coord.dec.rad
        hpind = hp.pixelfunc.ang2pix(nside, theta, phi, nest=nest)
    elif hasattr(coord, "l"):
        phi = coord.l.rad
        theta = 0.5 * np.pi - coord.b.rad
        hpind = hp.pixelfunc.ang2pix(nside, theta, phi, nest=nest)
    else:
        raise ValueError('Coordinate must be ra,dec or l,b')
    return hpind


def calculateSF(data: pl.DataFrame, sf_bins: dict, gcsn: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate the counts of the observed subsample and return
    that dataframe needed for the selection function. Is based off
    of Gaia Catalog of Nearby Stars

    Parameters
    ----------
    data: pl.DataFrame
        DataFrame of the data. Must have columns for
        healpix, phot_g_mean_mag, g_rp

    sf_bins: dict
        Binning for the selection function. Needs to have the keys:
        healpix, phot_g_mean_mag, g_rp. Here, healpix is the order to use,
        phot_g_mean_mag is a list of [lower bound, upper bound, bin_width], and
        g_rp is a list of [lower bound, upper bound, bin_width].
    
    sf_file: str
        Path to the data for Gaia Catalog of Nearby Stars. If None, will default
        to default path.
    
    Returns
    -------
    subsamp: pl.DataFrame
        The k, n and km, nm values use to calculate the posterior
        of the probability of selecting a source in a bin
    """
    # columns to select
    select_str = ''
    for key in sf_bins.keys():
        select_str += f'{key}_, '
    select_str = select_str[:-2]

    # the where statement
    where_str = ''
    for key in sf_bins.keys():
        if key != 'healpix':
            bins = np.arange(sf_bins[key][0], sf_bins[key][1], sf_bins[key][2])
            where_str += f'{key} >= {bins.min()} AND {key} < {bins.max()} AND '
    where_str = where_str[:-4]

    if len(where_str) > 0:
        sample = gcsn.sql(query=f'''       
            WITH subsamp AS (
                    SELECT {select_str}
                FROM self
                WHERE {where_str}
            )
            SELECT {select_str}, COUNT(*) AS n
            FROM subsamp
            GROUP BY {select_str}
            '''
                        )

        subsamp = data.sql(query=f'''       
            WITH subsamp AS (
                    SELECT {select_str}
                FROM self
                WHERE {where_str}
            )
            SELECT {select_str}, COUNT(*) AS k
            FROM subsamp
            GROUP BY {select_str}
            '''
                        )
    else:
        sample = gcsn.sql(query=f'''       
            WITH subsamp AS (
                    SELECT {select_str}
                FROM self
            )
            SELECT {select_str}, COUNT(*) AS n
            FROM subsamp
            GROUP BY {select_str}
            '''
                        )

        subsamp = data.sql(query=f'''       
            WITH subsamp AS (
                    SELECT {select_str}
                FROM self
            )
            SELECT {select_str}, COUNT(*) AS k
            FROM subsamp
            GROUP BY {select_str}
            '''
                        )

    subsamp = sample.join(subsamp, on=[f'{key}_' for key in sf_bins.keys()],
                          how='left')
    subsamp = subsamp.with_columns(pl.col("k").fill_null(strategy="zero"))
    return subsamp


def cal_veff(phot_g_mean_mag: np.ndarray | pl.Series,
             parallax: np.ndarray | pl.Series,
             galb: np.ndarray | pl.Series,
             order: int,
             G_lim: np.ndarray | pl.Series | float,
             healpix: np.ndarray | pl.Series,
             full_sky: bool = False) -> np.ndarray | pl.Series:
    """
    Calculate the effective volume of the data

    Parameters
    ---------
    phot_g_mean_mag: np.ndarray | pl.Series
        Gaia G mag of the data.
    
    parallax: np.ndarray | pl.Series
        Parallax of the data in mas.
    
    galb: np.ndarray | pl.Series
        The Galactic latitude of the data in radians.
    
    order: int
        Healpix order for estimating the sky coverage
    
    G_lim: np.ndarray | pl.Series | float
        The limiting magnitude assumed.
    
    Returns
    -------
    Veff: np.ndarray | pl.Series
        The effective volume of the data in pc^3
    """
    # get solid angle approximation in bins of magntiude
    if full_sky:
        hpbins = len(np.unique(healpix))
        solid_ang = hp.nside2pixarea(2 ** order) * hpbins
    else:
        solid_ang = hp.nside2pixarea(2 ** order)
    MG = phot_g_mean_mag + 5 * np.log10(1e-3 * parallax) + 5

    dmax = 10 ** ((G_lim - MG) / 5 + 1)
    dmax[dmax > 100] = 100

    H = 365  # scale height of thin disc in pc

    zeta = dmax * np.sin(abs(galb)) / H

    Veff = solid_ang * (H / abs(np.sin(galb))) ** 3 * (2 - (zeta ** 2 + 2 * zeta + 2) * np.exp(-zeta))
    # do not contribute where parallax < 10
    Veff[parallax < 10] = 0.
    if isinstance(Veff, pl.Series):
       Veff = Veff.rename('Veff')
    return Veff


def calc_subsample_p(k: np.ndarray | pl.Series,
                     n: np.ndarray | pl.Series,
                     RNG: np.random._generator.Generator = np.random.default_rng(666)) -> np.ndarray:
    """
    Calculate the probability of target being in a subsample

    Parameters
    ----------    
    k: np.ndarray | pl.Series
        The number of stars in a bin for the 100 pc subsample.
    
    n: np.ndarray | pl.Series
        The number of stars in a bin in the Gaia Catalog of Nearby Stars.
    
    RNG: np.random._generator.Generator
            Random generator with some seed.
    
    Returns
    -------
    pselect: np.ndarray
        The probability of selecting that star in the subsample.
    """
    alpha = k + 1
    beta = n - k + 1
    beta[k > n] = 1

    pselect = np.zeros(len(beta)) + np.nan
    pselect[beta > 0] = RNG.beta(alpha[beta > 0], beta[beta > 0])
    return pselect


def calc_1d_index(bin_idx: list,
                  bin_edges: list) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    For a ND binning, calculate the flattened 1D indexes

    Parameters
    ----------
    bin_idx: list
        List of of the digitized indexes of the data. Each index of list
        should be np.array.

    bin_edges: list
        List of the bin edges or simply the number of bins in the
        array for each dimension minus 1.

    Returns
    -------
    idx_1d: np.ndarray
        The flattened 1D index for the data
    
    valid: np.ndarray
        If the index is valid within the gridding

    max_idx: float
        The maximum size index of the flattened 1D index
    """
    ns = np.array([be if isinstance(be, int) else len(be) - 1 for be in bin_edges])

    idx_1d = np.ravel_multi_index(bin_idx, bin_edges, mode='clip')
    
    max_idx = np.prod(ns)

    valid = np.ones(len(bin_idx[0]), dtype=bool)
    for i in range(len(bin_idx)):
        valid &= bin_idx[i] >= 0
        valid &= bin_idx[i] < ns[i]
    
    return idx_1d, valid, max_idx


def build_effective_sel_factor(model_idx: np.ndarray,
                               sf_idx: np.ndarray,
                               weights: np.ndarray,
                               max_model_idx: int,
                               max_sf_idx: int) -> np.ndarray:
    """
    Calculate the sparse matrix of weights used to calculate the
    normalizing factor for the forward model

    Parameters
    -----------
    model_idx: np.ndarray
        1D flattened indexes for the GCNS data. These indexes are
        for the grid you are forward modeling the number
        densities onto
    
    sf_idx: np.ndarray
        1D flattened indexes for the GCNS data. These indexes are
        for the grid the selection function is calculated onto.

    weights: np.ndarray
        The weights to apply to the sparse matrix.
        Could be, e.g. the selection function probabilities of
        the observed data for the GCNS data, or some volume
        weighting.
    
    max_model_idx: int
        Maximum index possible in model_idx.
    
    max_sf_idx: int
        Maximum index possible in sf_idx.

    Returns
    -------
    A_j: np.ndarray
        The effective selection factor used to normalize the
        log probability.
    """
    # Sparse matrix
    A_jk_sparse = coo_matrix((weights, (sf_idx, model_idx)),
                             shape=(max_sf_idx, max_model_idx))

    # Convert to CSR for efficient row operations
    A_jk_csr = A_jk_sparse.tocsr()
    return A_jk_csr
    