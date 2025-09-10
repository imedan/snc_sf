from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits
import healpy as hp
import numpy as np
import polars as pl
from importlib.resources import open_binary
import ast
from collections.abc import Callable
from typing import Tuple
from gaiaunlimited.selectionfunctions import DR3SelectionFunctionTCG

import jax
from jax.experimental.sparse import BCOO
import jax.numpy as jnp
import optax
from jax.nn import sigmoid
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from jax.scipy.special import logsumexp
try:
    from jaxlib._jax import ArrayImpl
except ModuleNotFoundError:
    from jaxlib.xla_extension import ArrayImpl

from .utils import (coord2healpix, calculateSF, cal_veff,
                    calc_subsample_p, calc_1d_index, build_effective_sel_factor)
from .optimize import sigmoid_inv, objective_jax_multi, compute_single_loglike


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

    idx_sel_gcns: np.array
        flattened, 1D index of the GCNS for the selection function bins

    valid_sel_gcns: np.array
        If the 1D index of idx_sel_gcns is valid

    max_idx_sel_gcns: int
        maximum size of idx_sel_gcns
    
    idx_mod_gcns: np.array
        flattened, 1D index of the GCNS for the forward model, HR diagram bins

    valid_mod_gcns: np.array
        If the 1D index of idx_mod_gcns is valid

    max_idx_mod_gcns: int
        maximum size of idx_mod_gcns
    
    gcns_valid: np.array
        Where both idx_sel_gcns and idx_mod_gcns are valid
    
    A_jks: list
        List of sparse arrays for effective selection factor matrix

    weight_volume: bool
        If A_jks are weighted by volume or not.
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
        self.weight_volume = weight_volume
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


    def forward_model(self,
                      filter_data: pl.DataFrame) -> Tuple[ArrayImpl, ArrayImpl, np.ndarray]:
        """
        Perform the forward model to calculate the subpopulation probability
        across the HR diagram for the GCNS

        Parameters
        ----------
        filter_data: pl.DataFrame
            The filtered dataset of self.data for the subpopulation

        Returns
        --------
        p_warm: jaxlib._jax.ArrayImpl
            The resulting HR diagram probability of the subpopulation from
            the adam warmup. This is a 1D raveled array.

        p_samples: jaxlib._jax.ArrayImpl
            The resulting posterior samples of HR diagram probability of the
            subpopulation from the MCMC. This is of shape (samples, 1D raveled index).

        ev_valid: np.ndarray
            Where in the filter data both selection function and model indexes
            are valid.
        """
       # do the indexing for the filtered data
        bin_edges = []
        for key in self.sf_bins.keys():
            if key == 'healpix':
                bin_edges.append(hp.order2npix(self.sf_bins['healpix']))
            else:
                bin_edges.append(len(np.arange(*self.sf_bins[key])) - 1)

        bin_idx = [filter_data[f'{key}_'].to_numpy() for key in self.sf_bins.keys()]

        idx_sel_data, valid_sel_data, max_idx_sel_data = calc_1d_index(bin_idx, bin_edges)

        bin_edges = [len(np.arange(*self.sf_bins['g_rp'])) - 1,
                     len(np.arange(*self.MG_bins)) - 1]

        bin_idx = [filter_data['g_rp_'].to_numpy(),
                   filter_data['MG_'].to_numpy()]

        idx_mod_data, valid_mod_data, max_idx_mod_data = calc_1d_index(bin_idx, bin_edges)

        # convert for jax and transpose for objective
        A_jks_bcoo = tuple(BCOO.from_scipy_sparse(Ajk) for Ajk in self.A_jks)

        ### OPTIMIZE ADAM ###
        ev_valid = valid_sel_data & valid_mod_data
        S_data = filter_data['pselect_samps'].to_numpy()[ev_valid]
        Vmax_data = filter_data['Veff_samps'].to_numpy()[ev_valid]

        # Prepare JAX arrays once
        if self.weight_volume:
            weights_data_j = jnp.asarray(S_data * Vmax_data)
        else:
            weights_data_j = jnp.asarray(S_data)
        idx_mod_data_j = jnp.asarray(idx_mod_data[ev_valid], dtype=jnp.int32)
        idx_sel_data_j = jnp.asarray(idx_sel_data[ev_valid], dtype=jnp.int32)


        # starting point is the observed number density
        p0 = np.zeros(max_idx_mod_data) + 0.01
        p0 = jnp.array(p0)
        theta_init = sigmoid_inv(p0)

        # start with adam as a warmup to get closer to the right spot
        learning_rate = 1e-2
        num_adam_steps = 1000

        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(theta_init)

        @jax.jit
        def step(theta, opt_state, weights_data_j, A_jks_bcoo, idx_mod_data_j):
            loss, grads = jax.value_and_grad(objective_jax_multi)(theta, weights_data_j,
                                                                  A_jks_bcoo, idx_mod_data_j)
            updates, opt_state = optimizer.update(grads, opt_state)
            theta = optax.apply_updates(theta, updates)
            return theta, opt_state, loss

        theta = theta_init
        for i in range(num_adam_steps):
            theta, opt_state, loss = step(theta, opt_state, weights_data_j,
                                          A_jks_bcoo, idx_mod_data_j)
            if i % 50 == 0:
                print(f"Adam step {i}, loss={loss}")

        # Use the warmed-up theta as LBFGS start
        theta_init_warm = theta
        p_warm = jnp.zeros(A_jks_bcoo[0].shape[1])
        p_warm = p_warm.at[:].set(sigmoid(theta_init_warm))
        
        ### OPTIMIZE MCMC ###
        def model(weights_data, A_jks_bcoo, idx_mod_data):
            # Prior on transformed n
            theta = numpyro.sample('theta', dist.Normal(theta_init_warm, 5.0))
            p = jnp.zeros(A_jks_bcoo[0].shape[1])
            p = p.at[:].set(sigmoid(theta))
            
            log_like_samples = []
            for i in range(len(A_jks_bcoo)):
                log_like_samples.append(compute_single_loglike(p, A_jks_bcoo[i],
                                                               weights_data[:, i], idx_mod_data))
            
            log_like_samples = jnp.stack(log_like_samples)

            log_like_samples = jnp.where(jnp.isfinite(log_like_samples), log_like_samples, -1e10)

            log_like = logsumexp(log_like_samples) - jnp.log(len(A_jks_bcoo))
            numpyro.factor("marginal_loglike", log_like)
        
        nuts_kernel = NUTS(model)
        mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=2000)
        mcmc.run(jax.random.PRNGKey(0), weights_data_j, A_jks_bcoo, idx_mod_data_j)
        samples = mcmc.get_samples()

        p_samples = jnp.zeros((samples['theta'].shape[0], A_jks_bcoo[0].shape[1]))
        for i in range(samples['theta'].shape[0]):
            p_samples = p_samples.at[:, i].set(sigmoid(samples['theta'][:, i]))
        
        return p_warm, p_samples, ev_valid
