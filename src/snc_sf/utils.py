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
        Path to the precalculated data for the selection function
    
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
                                healpix AS healpix_,
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
