from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits
import healpy as hp
import numpy as np
import polars as pl
from importlib.resources import open_binary
import ast
from collections.abc import Callable
from gaiaunlimited.selectionfunctions import DR3SelectionFunctionTCG

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
    
    sf_bins: dict
        Binning for the selection function. Needs to have the keys:
        healpix, phot_g_mean_mag, g_rp. Here, healpix is the order to use,
        phot_g_mean_mag is a list of [lower bound, upper bound, bin_width], and
        g_rp is a list of [lower bound, upper bound, bin_width].

    G_lim: float
        Gaia G mag limit assumed.

    RNG: np.random._generator.Generator
            Random generator with some seed.

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
    def __init__(self, data_file:str, sf_bins: dict,
                 G_lim: float = 20,
                 RNG: np.random._generator.Generator = np.random.default_rng(666)):
        self.RNG = RNG
        # grab GCSN for SF
        self.sf_bins = sf_bins
        self.sf_file = open_binary('snc_sf.sf_files', 'GCNS-result.csv').name
        
        self.gcsn = pl.read_csv(self.sf_file)
        gcns = gcns.join(pl.read_csv(open_binary('snc_sf.sf_files', 'GNSC_distpdf.csv').name),
                         left_on='source_id', right_on='GaiaEDR3')

        self.coord_gcns = SkyCoord(ra=np.array(self.gcsn['ra']) * u.deg,
                                   dec=np.array(self.gcsn['dec']) * u.deg,
                                   frame='icrs')
        healpix = coord2healpix(self.coord_gcns,
                                nside=2 ** sf_bins['healpix'])
        self.gcsn = self.gcsn.with_columns(
            healpix_=pl.Series(healpix),
            g_rp=pl.col('phot_g_mean_mag') - pl.col('phot_rp_mean_mag'))
        
        # get the Glim for GCNS
        healpix_5 = coord2healpix(self.coord_gcns,
                                  nside=2 ** 5)
        self.gcns = self.gcns.with_columns(healpix_5=pl.Series(healpix_5))
        maglim = fits.open(open_binary('snc_sf.sf_files', 'GCNS_healpix_maglim.fit').name)[1].data
        self.gcns = self.gcns.join(pl.DataFrame({'healpix_5': np.arange(hp.order2npix(5)), 'maglim': maglim['mag80']}),
                                   on='healpix_5', how='left')

        # load the data
        self.data = pl.read_csv(data_file)
        self.data = self.data.filter(np.isin(self.data['source_id'], self.gcsn['source_id']))

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

        # add healpix index colum
        healpix = coord2healpix(self.coord,
                                 nside=2 ** self.sf_bins['healpix'])
        self.data = self.data.with_columns(healpix_=pl.Series(healpix))

        # add the indecies for the data
        phot_g_mean_mag_ = np.digitize(self.data['phot_g_mean_mag'],
                                       np.arange(self.sf_bins['phot_g_mean_mag'][0],
                                                 self.sf_bins['phot_g_mean_mag'][1],
                                                 self.sf_bins['phot_g_mean_mag'][2])) - 1

        g_rp_ = np.digitize(self.data['g_rp'],
                            np.arange(self.sf_bins['g_rp'][0],
                                      self.sf_bins['g_rp'][1],
                                      self.sf_bins['g_rp'][2])) - 1
        self.data = self.data.with_columns(phot_g_mean_mag_=phot_g_mean_mag_, g_rp_=g_rp_)

        # calculate the subselection
        self.subsamp = calculateSF(self.data, self.sf_bins, self.gcsn)
        self.subsamp = self.subsamp.with_columns(pl.col("k").fill_null(strategy="zero"))

        # get posterior samples
        self.sample_posterior()

        # join to the subselection
        self.data = self.data.join(self.subsamp,
                                   on=['healpix_', 'phot_g_mean_mag_', 'g_rp_'],
                                   how='left')
        self.gcns = self.gcns.join(self.subsamp, 
                                   on=['healpix_', 'phot_g_mean_mag_', 'g_rp_'],
                                   how='left')

        # get the emperical Gaia DR3 selection function
        mapHpx7 = DR3SelectionFunctionTCG()
        completeness = mapHpx7.query(self.coord,
                                     np.array(self.data['phot_g_mean_mag']))
        self.data = self.data.with_columns(completeness=completeness)

    def sample_posterior(self):
        """
        Sample the posterior of the subsample section function

        Parameters
        ---------
        nsamps: int
            Number of samples to return.
        """
        # nsamps from GCNS
        nsamps = 99

        # get the effective volume samples
        Veff_samps = np.zeros((len(self.gcsn), nsamps))
        for i in range(nsamps):
            Veff_samps[:, i] = cal_veff(
                self.gcns['phot_g_mean_mag'].to_numpy(),
                1 / self.gcns[f'Dist{nsamps + 1}'].to_numpy(),
                self.coord_gcns.galactic.b.rad,
                self.sf_bins['healpix'],
                self.gcns['maglim'].to_numpy())
        Veff_samps[Veff_samps <= 0] = 0.
        self.gcns = self.gcns.with_columns(Veff_samps=Veff_samps)
        del Veff_samps

        # get the posterior samples for the subsample selection
        pselect_samps = np.zeros((len(self.subsamp), nsamps))
        for i in range(nsamps):
            pselect_samps[:, i] = calc_subsample_p(
                self.subsamp['k'].to_numpy(),
                self.subsamp['n'].to_numpy(),
                self.RNG)
        self.subsamp = self.subsamp.with_columns(pselect_samps=pselect_samps)
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
            defined as 1 / (Veff * pselect * completeness),
            i.e. func(weights, *data_args, *args). Output of func
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
        completeness = filtered.select(pl.col("completeness")).to_numpy().reshape((-1,))
        ev = np.isfinite(pselect) & np.isfinite(Veff) & np.isfinite(completeness)
        idx = np.random.choice(np.sum(ev), np.sum(ev))
        if len(data_expr) > 0:
            data_args = tuple([filtered.select(de).to_numpy().reshape((-1, ))[ev][idx] for de in data_expr])
        else:
            data_args = ()
        test_out = func(1 / (pselect[ev][idx] * Veff[ev][idx] * completeness[ev][idx]),
                        *data_args, *args)

        # create nboot with right shape
        nboot_shape = [self.nsamps]
        if isinstance(test_out, np.ndarray):
            nboot_shape += list(test_out.shape)
        nboot = np.zeros(nboot_shape)

        # do the boostrap
        for i in range(self.nsamps):
            pselect = filtered.select(pl.col("pselect_samps").arr.get(i)).to_numpy().reshape((-1, ))
            Veff = filtered.select(pl.col("Veff_samps").arr.get(i)).to_numpy().reshape((-1, ))
            ev = np.isfinite(pselect) & np.isfinite(Veff) & np.isfinite(completeness)
            idx = np.random.choice(np.sum(ev), np.sum(ev))
            if len(data_expr) > 0:
                data_args = tuple([filtered.select(de).to_numpy().reshape((-1, ))[ev][idx] for de in data_expr])
            else:
                data_args = ()
            nboot[i] = func(1 / (pselect[ev][idx] * Veff[ev][idx] * completeness[ev][idx]),
                            *data_args, *args)
        return nboot
