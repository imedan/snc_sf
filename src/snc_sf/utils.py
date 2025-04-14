from astropy.coordinates import SkyCoord
import astropy.units as u
import healpy as hp
import numpy as np
import polars as pl
from importlib.resources import open_binary
import ast


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


def calculateSF(data: pl.DataFrame, sf_file:str = None) -> pl.DataFrame:
    """
    Calculate the counts of the observed subsample and return
    that dataframe needed for the selection function

    Parameters
    ----------
    data: pl.DataFrame
        DataFrame of the data. Must have columns for
        healpix, phot_g_mean_mag, g_rp
    
    sf_file: str
        Path to the data for the selection function. If None, will default
        to precomputed one.
    
    Returns
    -------
    subsamp: pl.DataFrame
        The k, n and km, nm values use to calculate the posterior
        of the probability of selecting a source in a bin
    """
    if sf_file is None:
        sf_file = open_binary('snc_sf.sf_files', '100pc_SF.csv').name
    
    with open(sf_file, 'r') as f:
        subSF_mock_dict = ast.literal_eval(f.readline().strip("#").strip("\n"))
    subSF_mock = pl.read_csv(sf_file, skip_rows=1)

    subsamp = data.sql(query=f'''       
                        WITH subsamp AS (
                              SELECT
                                healpix_,
                                CAST(floor((phot_g_mean_mag - {subSF_mock_dict['phot_g_mean_mag'][0]}) / {subSF_mock_dict['phot_g_mean_mag'][2]}) AS int) AS phot_g_mean_mag_,
                                CAST(floor(((g_rp) - {subSF_mock_dict['g_rp'][0]}) / {subSF_mock_dict['g_rp'][2]}) AS int) AS g_rp_
                            FROM self
                            WHERE g_rp > {subSF_mock_dict['g_rp'][0]}
                                  AND g_rp < {subSF_mock_dict['g_rp'][1]}
                                  AND phot_g_mean_mag > {subSF_mock_dict['phot_g_mean_mag'][0]}
                                  AND phot_g_mean_mag < {subSF_mock_dict['phot_g_mean_mag'][1]} AND parallax > 10
                        )
                        SELECT 
                            healpix_,
                            phot_g_mean_mag_,
                            g_rp_,
                            COUNT(*) AS k
                        FROM subsamp
                        GROUP BY healpix_, phot_g_mean_mag_, g_rp_
                        '''
                       )

    subsamp = subsamp.join(subSF_mock, on=['healpix_', 'phot_g_mean_mag_', 'g_rp_'])
    return subsamp


def cal_veff(healpix: np.ndarray | pl.Series,
             phot_g_mean_mag: np.ndarray | pl.Series,
             parallax: np.ndarray | pl.Series,
             galb: np.ndarray | pl.Series,
             order: int,
             G_lim: float) -> np.ndarray | pl.Series:
    """
    Calculate the effective volume of the data

    Parameters
    ---------
    healpix: np.ndarray | pl.Series
        The healpix indecies of the data.
    
    phot_g_mean_mag: np.ndarray | pl.Series
        Gaia G mag of the data.
    
    parallax: np.ndarray | pl.Series
        Parallax of the data in mas.
    
    galb: np.ndarray | pl.Series
        The Galactic latitude of the data in radians.
    
    order: int
        Healpix order used.
    
    G_lim: float
        The limiting magnitude assumed.
    
    Returns
    -------
    Veff: np.ndarray | pl.Series
        The effective volume of the data in pc^3
    """
    solid_ang = 4 * np.pi * len(np.unique(healpix)) / hp.order2npix(order)

    MG = phot_g_mean_mag + 5 * np.log10(1e-3 * parallax) + 5

    dmax = 10 ** ((G_lim - MG) / 5 + 1)
    dmax[dmax > 100] = 100

    H = 365  # scale height of thin disc in pc

    zeta = dmax * np.sin(abs(galb)) / H

    Veff = solid_ang * (H / abs(np.sin(galb))) ** 3 * (2 - (zeta ** 2 + 2 * zeta + 2) * np.exp(-zeta))
    if isinstance(Veff, pl.Series):
       Veff = Veff.rename('Veff')
    return Veff


def calc_subsample_p(km: np.ndarray | pl.Series,
                     nm: np.ndarray | pl.Series,
                     k: np.ndarray | pl.Series,
                     n: np.ndarray | pl.Series,
                     RNG: np.random._generator.Generator = np.random.default_rng(666)) -> np.ndarray:
    """
    Calculate the probability of target being in a subsample

    Parameters
    ----------
    km: np.ndarray | pl.Series
        The number of stars within 100 pc in a bin according to Gaia Mock catalog.
    
    nm: np.ndarray | pl.Series
        The number of stars in a bin according to Gaia Mock catalog.
    
    k: np.ndarray | pl.Series
        The number of stars in a bin for the 100 pc subsample.
    
    n: np.ndarray | pl.Series
        The number of stars in a bin in the Gaia catalog.
    
    RNG: np.random._generator.Generator
            Random generator with some seed.
    
    Returns
    -------
    pselect: np.ndarray
        The probability of selecting that star in the subsample.
    """
    alpham = km + 1
    betam = nm - km + 1
    frac = RNG.beta(alpham, betam)

    nf = np.round(n * frac)
    nf[nf < k] = k[nf < k]

    alpha = k + 1
    beta = nf - k + 1

    pselect = np.zeros(len(beta)) + np.nan
    pselect[beta > 0] = RNG.beta(alpha[beta > 0], beta[beta > 0])
    return pselect
