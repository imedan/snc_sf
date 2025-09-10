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

from .utils import (coord2healpix, calculateSF, cal_veff,
                    calc_subsample_p, calc_1d_index, build_effective_sel_factor)


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

    MG_bins: list
        MG binning of HR diagram to use for the forward model. Is 
        list of [lower bound, upper bound, bin_width]. For g_rp bins, will use
        the same as sf_bins.

    RNG: np.random._generator.Generator
            Random generator with some seed.

    mean: bool
        If true, then only do one sample at the mean
        of the distriubtion

    calc_SF: bool
        Calculate selection function when initializing

    Attributes
    ----------
    data: pl.DataFrame
        DataFrame of data_file.

    gcns: pl.DataFrame
        DataFrame of GCNS data.

    coord: astropy.coordinates.SkyCoord
        Coordinates of the data.

    coord_gcns: astropy.coordinates.SkyCoord
        Coordinates of GCNS data.
    
    subsamp: pl.DataFrame
        The k, n and km, nm values use to calculate the posterior
        of the probability of selecting a source in a bin
    """
    def __init__(self, data_file:str, sf_bins: dict,
                 MG_bins: list,
                 RNG: np.random._generator.Generator = np.random.default_rng(666),
                 mean: bool = False,
                 calc_SF: bool = True):
        self.RNG = RNG
        self.mean = mean
        # grab GCNS for SF
        self.sf_bins = sf_bins
        self.MG_bins = MG_bins
        self.sf_file = open_binary('snc_sf.sf_files', 'GCNS-result.csv').name
        
        self.gcns = pl.read_csv(self.sf_file)
        self.gcns = self.gcns.join(
            pl.read_csv(open_binary('snc_sf.sf_files', 'GNSC_distpdf.csv').name),
            left_on='source_id', right_on='GaiaEDR3')

        self.coord_gcns = SkyCoord(ra=np.array(self.gcns['ra']) * u.deg,
                                   dec=np.array(self.gcns['dec']) * u.deg,
                                   frame='icrs')
        healpix = coord2healpix(self.coord_gcns,
                                nside=2 ** sf_bins['healpix'])
        self.gcns = self.gcns.with_columns(
            healpix_=pl.Series(healpix),
            g_rp=pl.col('phot_g_mean_mag') - pl.col('phot_rp_mean_mag'),
            MG=pl.col('phot_g_mean_mag') + 5 * np.log10(1e-3 * pl.col('parallax')) + 5,
            log_parallax=np.log10(pl.col('parallax')))  # add this for binning
        
        # get the Glim for GCNS
        healpix_5 = coord2healpix(self.coord_gcns,
                                  nside=2 ** 5)
        self.gcns = self.gcns.with_columns(healpix_5=pl.Series(healpix_5))
        maglim = fits.open(open_binary('snc_sf.sf_files', 'GCNS_healpix_maglim.fit').name)[1].data
        self.gcns = self.gcns.join(pl.DataFrame({'healpix_5': np.arange(hp.order2npix(5)), 'maglim': maglim['mag80']}),
                                   on='healpix_5', how='left')

        # load the data
        self.data = pl.read_csv(data_file)
        self.data = self.data.filter(np.isin(self.data['source_id'], self.gcns['source_id']))

        # calculate error in G mag
        sigmaG_0 = 0.0027553202
        e_Gmag   = np.sqrt((-2.5 / np.log(10) * self.data['phot_g_mean_flux_over_error'] ** -1) ** 2 + sigmaG_0**2)
        self.data = self.data.with_columns(phot_g_mean_mag_error=e_Gmag)

        # get SkyCoord of data
        self.coord = SkyCoord(ra=np.array(self.data['ra']) * u.deg,
                              dec=np.array(self.data['dec']) * u.deg,
                              distance=np.array(1000 / self.data['parallax']) * u.pc,
                              frame='icrs')

        # add healpix index colum
        healpix = coord2healpix(self.coord,
                                 nside=2 ** self.sf_bins['healpix'])
        self.data = self.data.with_columns(
            healpix_=pl.Series(healpix),
            MG=pl.col('phot_g_mean_mag') + 5 * np.log10(1e-3 * pl.col('parallax')) + 5,
            log_parallax=np.log10(pl.col('parallax')))  # add this for binning)
        
        # grab the distance posterior samples
        cols = [f'Dist{i}' for i in range(1, 100)]
        self.data = self.data.join(self.gcns[['source_id', 'maglim'] + cols],
                                   on='source_id')

        # add the indecies for the data
        for key in self.sf_bins.keys():
            if key != 'healpix':
                key_index = np.digitize(self.data[key],
                                        np.arange(*self.sf_bins[key])) - 1
                self.data = self.data.with_columns(pl.Series(f'{key}_', key_index))

        MG_ = np.digitize(self.data['MG'], np.arange(*self.MG_bins)) - 1
        self.data = self.data.with_columns(MG_=MG_)

        # add the indecies for the GCNS
        for key in self.sf_bins.keys():
            if key != 'healpix':
                key_index = np.digitize(self.gcns[key],
                                        np.arange(*self.sf_bins[key])) - 1
                self.gcns = self.gcns.with_columns(pl.Series(f'{key}_', key_index))

        MG_ = np.digitize(self.gcns['MG'], np.arange(*self.MG_bins)) - 1
        self.gcns = self.gcns.with_columns(MG_=MG_)

        # calc the SF
        if calc_SF:
            self.calculate_selection_func()


    def calculate_selection_func(self):
        """
        Now calculate the selection function for the data
        """
        # calculate the subselection
        self.subsamp = calculateSF(self.data, self.sf_bins, self.gcns)

        # get posterior samples
        self.sample_posterior()

        # join to the subselection
        self.data = self.data.join(self.subsamp,
                                   on=[f'{key}_' for key in self.sf_bins.keys()],
                                   how='left')
        self.gcns = self.gcns.join(self.subsamp, 
                                   on=[f'{key}_' for key in self.sf_bins.keys()],
                                   how='left')
        
        # zero out k = 0 things
        self.gcns = self.gcns.with_columns(
            pl.when(pl.col("k") == 0)
            .then(pl.lit([0.] * self.nsamps).cast(pl.Array(pl.Float64, self.nsamps)))
            .otherwise(pl.col("pselect_samps"))
            .alias("pselect_samps")
        )

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
        if self.mean:
            nsamps = 1
            # get the effective volume samples for gcns
            Veff_samps = np.zeros((len(self.gcns), nsamps))
            for i in range(nsamps):
                Veff_samps[:, i] = cal_veff(
                    self.gcns['phot_g_mean_mag'].to_numpy(),
                    self.gcns['parallax'].to_numpy(),
                    self.coord_gcns.galactic.b.rad,
                    self.sf_bins['healpix'],
                    self.gcns['maglim'].to_numpy(),
                    self.gcns['healpix_'].to_numpy())
            Veff_samps[Veff_samps <= 0] = 0.
            self.gcns = self.gcns.with_columns(Veff_samps=Veff_samps)
            del Veff_samps

            # get the effective volume samples for data
            Veff_samps = np.zeros((len(self.data), nsamps))
            for i in range(nsamps):
                Veff_samps[:, i] = cal_veff(
                    self.data['phot_g_mean_mag'].to_numpy(),
                    self.data['parallax'].to_numpy(),
                    self.coord.galactic.b.rad,
                    self.sf_bins['healpix'],
                    self.data['maglim'].to_numpy(),
                    self.data['healpix_'].to_numpy(),
                    full_sky=False)
            Veff_samps[Veff_samps <= 0] = 0.
            self.data = self.data.with_columns(Veff_samps=Veff_samps)
            del Veff_samps

            # get the posterior samples for the subsample selection
            pselect_samps = np.zeros((len(self.subsamp), nsamps))
            for i in range(nsamps):
                pselect_samps[:, i] = (self.subsamp['k'].to_numpy() + 1) / \
                                      (self.subsamp['n'].to_numpy() + 2)
            self.subsamp = self.subsamp.with_columns(pselect_samps=pselect_samps)
            del pselect_samps
        else:
            nsamps = 99
            # get the effective volume samples for gcns
            Veff_samps = np.zeros((len(self.gcns), nsamps))
            for i in range(nsamps):
                Veff_samps[:, i] = cal_veff(
                    self.gcns['phot_g_mean_mag'].to_numpy(),
                    1 / self.gcns[f'Dist{i + 1}'].to_numpy(),
                    self.coord_gcns.galactic.b.rad,
                    self.sf_bins['healpix'],
                    self.gcns['maglim'].to_numpy(),
                    self.gcns['healpix_'].to_numpy())
            Veff_samps[Veff_samps <= 0] = 0.
            self.gcns = self.gcns.with_columns(Veff_samps=Veff_samps)
            del Veff_samps

            # get the effective volume samples for data
            Veff_samps = np.zeros((len(self.data), nsamps))
            for i in range(nsamps):
                Veff_samps[:, i] = cal_veff(
                    self.data['phot_g_mean_mag'].to_numpy(),
                    1 / self.data[f'Dist{i + 1}'].to_numpy(),
                    self.coord.galactic.b.rad,
                    self.sf_bins['healpix'],
                    self.data['maglim'].to_numpy(),
                    self.data['healpix_'].to_numpy(),
                    full_sky=False)
            Veff_samps[Veff_samps <= 0] = 0.
            self.data = self.data.with_columns(Veff_samps=Veff_samps)
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

    
    def evalutate_Ajk(self, weight_volume: bool = False):
        """
        Evalutate sparse matrix of the effective selection
        factor.

        Parameters
        ----------
        weight_volume: bool
            If to include Veff in the weighting for, e.g.
            number density evaluation
        """
        # do all the 1D flattened indexing
        bin_edges_sel = []
        for key in self.sf_bins.keys():
            if key == 'healpix':
                bin_edges_sel.append(hp.order2npix(self.sf_bins['healpix']))
            else:
                bin_edges_sel.append(len(np.arange(*self.sf_bins[key])) - 1)

        bin_idx = [self.gcns[f'{key}_'].to_numpy() for key in self.sf_bins.keys()]

        self.idx_sel_gcns, self.valid_sel_gcns, self.max_idx_sel_gcns = calc_1d_index(bin_idx, bin_edges_sel)

        bin_edges_mod = [len(np.arange(*self.sf_bins['g_rp'])) - 1,
                         len(np.arange(*self.MG_bins)) - 1]

        bin_idx = [self.gcns['g_rp_'].to_numpy(),
                   self.gcns['MG_'].to_numpy()]

        self.idx_mod_gcns, self.valid_mod_gcns, self.max_idx_mod_gcns = calc_1d_index(bin_idx, bin_edges_mod)

        # get all of the A_jk
        self.gcns_valid = self.valid_mod_gcns & self.valid_sel_gcns

        self.A_jks = []
        for i in range(self.nsamps):
            Sf = self.gcns['pselect_samps'].to_numpy()[:, i][self.gcns_valid]
            Vmax = self.gcns['Veff_samps'].to_numpy()[:, i][self.gcns_valid]
            ev_weight = (np.isfinite(Vmax)) & (Vmax > 0)  # Always only include things within 100 pc?
            if weight_volume:
                weight = Sf * Vmax
            else:
                weight = Sf
            Ajk = build_effective_sel_factor(self.idx_mod_gcns[self.gcns_valid][ev_weight],
                                             self.idx_sel_gcns[self.gcns_valid][ev_weight],
                                             weight[ev_weight],
                                             self.max_idx_mod_gcns,
                                             self.max_idx_sel_gcns)
            self.A_jks.append(Ajk)


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
