from astropy.coordinates import SkyCoord
import astropy.units as u
import healpy as hp
import numpy as np
import polars as pl
from importlib.resources import open_binary
import ast
from collections.abc import Callable

from .utils import coord2healpix, calculateSF, cal_veff, calc_subsample_p


class SNCSelectionFunction(object):
    """
    Sample thhe SNC selection function based on some input data
    consisting of stars within 100 pc

    Parameters
    ----------
    data_file: str
        Path to the csv file that has the data. Must include following columns (from Gaia):
        source_id, ra, ra_error, dec, dec_error, parallax, parallax_error,
        g_rp,  phot_g_mean_mag, phot_g_mean_flux_over_error
    
    sf_file: str
        Path to the data for the selection function. If None, will default
        to precomputed one.

    G_lim: float
        Gaia G mag limit assumed.

    Attributes
    ----------
    data: pl.DataFrame
        DataFrame of data_file.

    coord: astropy.coordinates.SkyCoord
        Coordinates of the data.
    
    subSF_mock_dict: dict
        dict of the bins used for the subsample selection function
    
    subsamp: pl.DataFrame
        The k, n and km, nm values use to calculate the posterior
        of the probability of selecting a source in a bin
    """
    def __init__(self, data_file:str, sf_file:str = None, G_lim: float = 20):
        # load the data
        self.data = pl.read_csv(data_file)

        # calculate error in G mag
        sigmaG_0 = 0.0027553202
        e_Gmag   = np.sqrt((-2.5 / np.log(10) * self.data['phot_g_mean_flux_over_error'] ** -1) ** 2 + sigmaG_0**2)
        self.data = self.data.with_columns(phot_g_mean_mag_error=e_Gmag)

        # get SkyCoord of data
        self.coord = SkyCoord(ra=np.array(self.data['ra']) * u.deg,
                              dec=np.array(self.data['dec']) * u.deg,
                              distance=np.array(1000 / self.data['parallax']) * u.pc,
                              frame='icrs')
        
        self.G_lim = G_lim

        # grab binning of SF
        self.sf_file = sf_file
        if self.sf_file is None:
            self.sf_file = open_binary('snc_sf.sf_files', '100pc_SF.csv').name
    
        with open(self.sf_file, 'r') as f:
            self.subSF_mock_dict = ast.literal_eval(f.readline().strip("#").strip("\n"))

        # add healpix index colum
        healpix = coord2healpix(self.coord,
                                 nside=2 ** self.subSF_mock_dict['healpix'])
        self.data = self.data.with_columns(healpix_=pl.Series(healpix))

        # add the indecies for the data
        phot_g_mean_mag_ = np.digitize(self.data['phot_g_mean_mag'],
                                       np.arange(self.subSF_mock_dict['phot_g_mean_mag'][0],
                                                 self.subSF_mock_dict['phot_g_mean_mag'][1],
                                                 self.subSF_mock_dict['phot_g_mean_mag'][2])) - 1

        g_rp_ = np.digitize(self.data['g_rp'],
                            np.arange(self.subSF_mock_dict['g_rp'][0],
                                      self.subSF_mock_dict['g_rp'][1],
                                      self.subSF_mock_dict['g_rp'][2])) - 1
        self.data = self.data.with_columns(phot_g_mean_mag_=phot_g_mean_mag_, g_rp_=g_rp_)

        # calculate the subselection
        self.subsamp = calculateSF(self.data, sf_file=self.sf_file)

        # join to the subselection
        self.data = self.data.join(self.subsamp, on=['healpix_', 'phot_g_mean_mag_', 'g_rp_'], how='left')

    def sample_posterior(self, nsamps: int,
                         RNG: np.random._generator.Generator = np.random.default_rng(666)):
        """
        Sample the posterior of the subsample section function

        Parameters
        ---------
        nsamps: int
            Number of samples to return.
        
        RNG: np.random._generator.Generator
            Random generator with some seed.
        """
        # get the effective volume samples
        Veff_samps = np.zeros((len(self.data), nsamps))
        for i in range(nsamps):
            idx = RNG.choice(len(self.coord), len(self.coord))
            Veff_samps[:, i] = cal_veff(
                self.coord[idx],
                RNG.normal(self.data['phot_g_mean_mag'],
                           self.data['phot_g_mean_mag_error']),
                RNG.normal(self.data['parallax'],
                           self.data['parallax_error']),
                self.coord.galactic.b.rad,
                3,  # use larger order to estimate sky coverage
                self.G_lim)
        Veff_samps[Veff_samps <= 0] = np.nan
        self.data = self.data.with_columns(Veff_samps=Veff_samps)
        del Veff_samps

        # get the posterior samples for the subsample selection
        pselect_samps = np.zeros((len(self.data), nsamps))
        for i in range(nsamps):
            pselect_samps[:, i] = calc_subsample_p(
                np.array(self.data['km']),
                np.array(self.data['nm']),
                np.array(self.data['k']),
                np.array(self.data['n']),
                RNG)
        self.data = self.data.with_columns(pselect_samps=pselect_samps)
        del pselect_samps

        # save number of samples for posterior
        self.nsamps = nsamps

    def bootstrap_values(self,
                         func: Callable,
                         filt: pl.Expr = None,
                         data_expr: list = [],
                         args: tuple | list = ()) -> np.ndarray:
        """
        Bootstrap some value from a function based on posterior samples

        Parameters
        ----------
        func: function
            Function to calculate some parameter for the bootstrap
            based on the posterior samples from the selection function.
            The first input must be 'weights', which is the weight applied to a star
            defined as 1 / (Veff * pselect), i.e. func(weights, *data_args, *args). Output of func
            must be a float or np.ndarray.

        filt: pl.Expr
            Polars expression for the filter to be placed on the data
            for the calculation
        
        data_expr: list
            Additional data from the dataset to be used within func for the calculation.
            Each index of the list should be a Polars expression. The code below will
            turn this into a numpy array based on the filter, which is what will then
            be passed to func as *data_args.
        
        args: list
            Additional arguments to be passed to func, that are not data from the
            dataset

        Returns
        -------
        nboot: np.ndarray
            The bootstrapped values of size (self.nsamps, N). Here N
            depends on output from func. If output of func is array, nboot will be ND
            with N being sahpe of output. Otherwise, nboot will be 1D array.
        """
        if filt is None:
            filtered = self.data.filter()
        else:
            filtered = self.data.filter(filt)
       
        # test output to get size
        pselect = filtered.select(pl.col("pselect_samps").arr.get(0)).to_numpy().reshape((-1, ))
        Veff = filtered.select(pl.col("Veff_samps").arr.get(0)).to_numpy().reshape((-1, ))
        ev = np.isfinite(pselect) & np.isfinite(Veff)
        idx = np.random.choice(np.sum(ev), np.sum(ev))
        if len(data_expr) > 0:
            data_args = tuple([filtered.select(de).to_numpy().reshape((-1, ))[ev][idx] for de in data_expr])
        else:
            data_args = ()
        test_out = func(1 / (pselect[ev][idx] * Veff[ev][idx]), *data_args, *args)

        # create nboot with right shape
        nboot_shape = [self.nsamps]
        if isinstance(test_out, np.ndarray):
            nboot_shape += list(test_out.shape)
        nboot = np.zeros(nboot_shape)

        # do the boostrap
        for i in range(self.nsamps):
            pselect = filtered.select(pl.col("pselect_samps").arr.get(i)).to_numpy().reshape((-1, ))
            Veff = filtered.select(pl.col("Veff_samps").arr.get(i)).to_numpy().reshape((-1, ))
            ev = np.isfinite(pselect) & np.isfinite(Veff)
            idx = np.random.choice(np.sum(ev), np.sum(ev))
            if len(data_expr) > 0:
                data_args = tuple([filtered.select(de).to_numpy().reshape((-1, ))[ev][idx] for de in data_expr])
            else:
                data_args = ()
            nboot[i] = func(1 / (pselect[ev][idx] * Veff[ev][idx]), *data_args, *args)
        return nboot
