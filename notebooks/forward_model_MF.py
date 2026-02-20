from snc_sf.selection_function import *
from snc_sf.utils import *
from snc_sf.optimize import *

import numpy as np
import polars as pl
import matplotlib.pylab as plt
import os
from matplotlib.colors import LogNorm
plt.style.use('%s/mystyle.mplstyle' % os.environ['MPL_STYLES'])
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxopt
from astropy.table import Table, vstack
from typing import Tuple
from scipy.interpolate import LinearNDInterpolator
from tqdm import trange
from scipy.optimize import minimize


def mass_est(MK, *args, feh=None):
    if feh is None:
        mass = 10 ** np.polyval(args[::-1], MK - 7.5)
    else:
        massi = 10 ** np.polyval(args[:-1][::-1], MK - 7.5)
        mass = (1 + args[-1] * feh) * massi
    mass[MK < 4.5] = np.nan
    mass[MK > 10.5] = np.nan
    return mass


def _parse_params(params: list) -> Tuple[float, float, float, float, bool]:
    """
    Accepts params as length 2 or 4:
      [logC, alpha]            -> single power law
      [logC, alpha1, alpha2, log_mb] -> broken power law
    Returns parsed params
    """
    p = np.asarray(params, dtype=float)
    if p.size == 2:
        logC, alpha = p
        C = 10.0**logC
        alpha1 = float(alpha)
        alpha2 = float(alpha)   # unused for single but set for convenience
        mb = np.inf             # break at +inf => no "above" bins
        is_broken = False
    elif p.size == 4:
        logC, alpha1, alpha2, log_mb = p
        C = 10.0**logC
        mb = 10.0**log_mb
        is_broken = True
    else:
        raise ValueError("params must have length 2 (single PL) or 4 (broken PL)")
    return C, float(alpha1), float(alpha2), float(mb), is_broken


def xi_from_params(params: list | np.ndarray, m: np.ndarray):
    """
    Return xi(m) = dN/dm value(s) for array-like m using params.
    Accepts 2 or 4 parameter forms.
    """
    C, alpha1, alpha2, mb, is_broken = _parse_params(params)
    m = np.asarray(m, dtype=float)
    out = np.empty_like(m, dtype=float)
    low = m < mb
    high = ~low
    out[low] = C * m[low]**(-alpha1)
    if np.any(high):
        cont = mb**(alpha2 - alpha1)
        out[high] = C * cont * m[high]**(-alpha2)
    return out


@jax.jit
def _integral_power_law_jax(a, b, C, alpha):
    """
    Integral of C * m^{-alpha} dm from a to b, JAX-compatible.
    Uses jnp.where to avoid branching on traced values.
    """
    # alpha == 1 case: C * log(b/a)
    # alpha != 1 case: C / (1 - alpha) * (b^(1-alpha) - a^(1-alpha))
    safe_exp = 1.0 - alpha
    integral_general = C / safe_exp * (b**safe_exp - a**safe_exp)
    integral_log = C * jnp.log(b / a)
    return jnp.where(jnp.abs(alpha - 1.0) < 1e-10, integral_log, integral_general)


@jax.jit
def model_bin_density_jax(edges: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    """
    JAX model bin density for a single params vector [logC, alpha1, alpha2, log_mb].
    All branching is done with jnp.where so this is fully JIT/vmap-compatible.

    edges : shape (N+1,)  — bin edges
    params: shape (4,)    — [logC, alpha1, alpha2, log_mb]
    returns: shape (N,)   — average differential density per bin
    """
    logC, alpha1, alpha2, log_mb = params[0], params[1], params[2], params[3]
    C  = 10.0**logC
    mb = 10.0**log_mb

    lo = edges[:-1]   # (N,)
    hi = edges[1:]    # (N,)
    width = hi - lo

    # Continuity factor so the two power laws join at mb
    C2 = C * (mb**(alpha2 - alpha1))

    # ---- bins fully below break (hi <= mb) ----
    integral_below = _integral_power_law_jax(lo, hi, C, alpha1)
    y_below = integral_below / width

    # ---- bins fully above break (lo >= mb) ----
    integral_above = _integral_power_law_jax(lo, hi, C2, alpha2)
    y_above = integral_above / width

    # ---- bins that straddle the break ----
    # clamp integration limits so they're valid even when this bin doesn't straddle
    lo_clamp = jnp.minimum(lo, mb)
    hi_clamp = jnp.maximum(hi, mb)
    integral_cross = (
        _integral_power_law_jax(lo_clamp, mb,      C,  alpha1) +
        _integral_power_law_jax(mb,       hi_clamp, C2, alpha2)
    )
    y_cross = integral_cross / width

    # Select the right case per bin using jnp.where (no Python branching)
    below_mask = hi <= mb
    above_mask = lo >= mb
    # straddle = not below and not above

    y_model = jnp.where(below_mask, y_below,
               jnp.where(above_mask, y_above,
                         y_cross))
    return y_model


@jax.jit
def obj_logsq_jax(params: jnp.ndarray, edges: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    JAX objective: sum of squared log residuals.
    Returns a large value (1e9) where y or ymod are non-positive via jnp.where.

    params: (4,)
    edges:  (N+1,)
    y:      (N,)
    """
    ymod = model_bin_density_jax(edges, params)

    # Valid mask: both y and ymod positive
    valid = (y > 0) & (ymod > 0)

    log_resid_sq = jnp.where(valid, (jnp.log(y) - jnp.log(ymod))**2, 0.0)
    loss = jnp.sum(log_resid_sq)

    # If no valid points at all, return a large penalty
    # (jnp.where keeps this differentiable)
    return jnp.where(jnp.any(valid), loss, 1e9)


def fit_broken_powerlaw_batch(
    Nmass_boot: np.ndarray,
    mass_bins: np.ndarray,
    p0: np.ndarray,
    bounds: list,
) -> np.ndarray:
    """
    Fit a broken power law to every row of Nmass_boot in parallel using
    jaxopt.LBFGSB + jax.vmap.

    Parameters
    ----------
    Nmass_boot : (size, n_bins)  — one histogram per row
    mass_bins  : (n_bins+1,)    — bin edges
    p0         : (4,)           — initial guess (numpy)
    bounds     : list of (lo, hi) tuples, length 4

    Returns
    -------
    params_boot : (size, 4) numpy array
    """
    edges = jnp.array(mass_bins, dtype=jnp.float64)
    Y     = jnp.array(Nmass_boot, dtype=jnp.float64)   # (size, n_bins)
    size  = Y.shape[0]

    # Broadcast p0 across all rows
    P0 = jnp.broadcast_to(jnp.array(p0, dtype=jnp.float64), (size, 4))

    # Convert bounds to JAX arrays
    lower = jnp.array([b[0] for b in bounds], dtype=jnp.float64)
    upper = jnp.array([b[1] for b in bounds], dtype=jnp.float64)

    # Build the jaxopt solver.
    # fun receives (params, y_row) — y_row is the per-sample "hyperparameter"
    solver = jaxopt.LBFGSB(
        fun=obj_logsq_jax,
        maxiter=500,
        tol=1e-6,
    )

    # vmap run_one over the batch dimension
    def run_one(p0_row, y_row):
        result = solver.run(
            p0_row,
            bounds=(lower, upper),
            edges=edges,
            y=y_row,
        )
        return result.params

    # jax.vmap maps over the first axis of both P0 and Y
    batched_run = jax.vmap(run_one)

    params_jax = batched_run(P0, Y)
    return np.array(params_jax)


if __name__ == '__main__':
    # set up bins and filtering
    MG_bin_list = [0, 20, 0.25]

    sf_bins={'healpix': 3,
            'phot_g_mean_mag': [0, 22, 1],
            'g_rp': [-0.4, 2.2, 0.05]}

    n_g_rp = len(np.arange(*sf_bins['g_rp'])) - 1
    n_mg = len(np.arange(*MG_bin_list)) - 1

    # get Zach's overluminouse stars from: https://zenodo.org/records/18500082?preview=1&token=eyJhbGciOiJIUzUxMiIsImlhdCI6MTc3MDMyOTI0MywiZXhwIjoxODMwMjk3NTk5fQ.eyJpZCI6ImRjMzRkMTNlLTIxNzAtNGE5Ni05OGMyLTI3MDY5NGU1MjljZiIsImRhdGEiOnt9LCJyYW5kb20iOiI5NzY5ZmY5NjA4NDhlZDBlMDFkNGIxNjFiN2E1NWU0YSJ9.nQqeKD-sk-RbMyhJLbbGbHk-NzpKaJtkzDApknGXn5lOv2xWx-GV-xcTiZpxwGiIpkV7nXsg5Cb23OThf16q0A
    overluminous_cat = pl.read_csv('overluminous_catalog.csv')

    pre_filt = ( (pl.col('ruwe') < 1.4) & (pl.col('ipd_frac_multi_peak') == 0) &
                # (4.74 * (pl.col("pmra") ** 2 + pl.col("pmdec") ** 2).sqrt() / pl.col("parallax") < 40)  &
                 (~pl.col("source_id").is_in(overluminous_cat.filter(pl.col('ol'))['source_id'])) &
                 (pl.col('phot_g_mean_mag') + 5 * (pl.col('parallax').log10() + np.log10(1e-3)) + 5 < 10 * (pl.col('phot_g_mean_mag') - pl.col('phot_rp_mean_mag')) + 5) )

    if not os.path.isfile('aspcap_slam_fe_h_corr_snc_forward_data.csv'):
        # initilize
        mean = True
        data_file = '../../obs_100pc_edr3.csv'
        sf = SNCSelectionFunction(data_file,
                                sf_bins,
                                MG_bin_list,
                                mean=mean,
                                pre_filt=pre_filt)
        
        # load the metallicity data
        tbl = Table.read('../../../DR19/astraFrankenstein-0.6.0.fits', hdu=1)
        names = [name for name in tbl.colnames if len(tbl[name].shape) <= 1]
        slam = pl.from_pandas(tbl[names].to_pandas())


        data_slam = sf.data.join(slam, left_on='source_id', right_on='gaia_dr3_source_id')
        data_slam = data_slam.unique('source_id', keep='last')

        data_slam = data_slam.with_columns(
            pl.col("pipeline").cast(pl.Utf8)
        )

        data_slam = data_slam.filter((pl.col('fe_h').is_not_null()) &
                                    ((pl.col("pipeline").str.contains("slam")) |
                                    (pl.col("pipeline").str.contains("aspcap"))) &
                                    (pl.col('teff') > 3200))
        
        # correction to aspcap low teff metallicities based on wide binaries
        x = data_slam['teff'].to_numpy()
        y = data_slam['fe_h'].to_numpy()
        fe_h_corr = y + (-6.095e-11 * x ** 3 + 9.164e-07 * x ** 2 - 0.004564 * x + 7.538)
        data_slam = data_slam.with_columns(fe_h_corr=fe_h_corr)
        when_corr = (pl.col("pipeline").str.contains("aspcap")) & (pl.col('teff') <= 4600)
        data_slam = data_slam.with_columns(fe_h_corr=pl.when(when_corr).then(pl.col('fe_h_corr')).otherwise(pl.col('fe_h')))

        # save to file for testing
        cols_save = ['source_id', 'ra', 'ra_error', 'dec', 'dec_error', 'parallax',
                    'parallax_error', 'g_rp', 'phot_g_mean_mag', 'phot_g_mean_flux_over_error',
                    'fe_h_corr']
        data_slam[cols_save].write_csv('aspcap_slam_fe_h_corr_snc_forward_data.csv')

    # setup the forward model
    mean = False
    sf = SNCSelectionFunction('aspcap_slam_fe_h_corr_snc_forward_data.csv',
                              sf_bins, MG_bin_list,
                              mean=mean, pre_filt=pre_filt)
    # get A_jks
    sf.evalutate_Ajk(weight_volume=False)
    # bins for plotting
    g_rp_bins = np.arange(*sf.sf_bins['g_rp'])
    MG_bins = np.arange(*MG_bin_list)

    # run the forward model
    fe_h_bins = np.array([-0.6, -0.3, -0.15, -0.075, 0., 0.075, 0.2])
    feh_data = []
    p_samples_feh = []
    ev_valid_feh = []

    for i in range(len(fe_h_bins) - 1):
        filter_data = sf.data.filter((pl.col('fe_h_corr') > fe_h_bins[i]) & (pl.col('fe_h_corr') <= fe_h_bins[i + 1]))
        p_samples, ev_valid = sf.forward_model(filter_data,
                                               num_warmup=500,
                                               num_samples=1000,
                                               num_chains=1)

        feh_data.append(filter_data)
        p_samples_feh.append(p_samples)
        ev_valid_feh.append(ev_valid)

    RNG = np.random.default_rng(666)

    # get mass from Mann relation
    args = np.array([-0.642, -0.208, -8.43e-4, 7.78e-3, 1.42e-3, -2.13e-4])
    args_mh = np.array([-0.647, -0.207, -6.53e-4, 7.13e-3, 1.84e-5, -2.13e-4, -0.0035])

    MK = sf.gcns['k_m_2mass'].to_numpy() + 5 * np.log10(1e-3 * sf.gcns['parallax'].to_numpy()) + 5

    mass = mass_est(MK, *args)

    # interpolate for things without MK
    sf.gcns = sf.gcns.with_columns(bp_rp=pl.col('phot_bp_mean_mag')-pl.col('phot_rp_mean_mag'))
    sf.gcns = sf.gcns.with_columns(
                MBP=pl.col('phot_bp_mean_mag') + 5 * np.log10(1e-3 * pl.col('parallax')) + 5,
                MRP=pl.col('phot_rp_mean_mag') + 5 * np.log10(1e-3 * pl.col('parallax')) + 5)
    
    points = sf.gcns[['MG', 'MBP', 'MRP']].to_numpy()
    ev = np.isfinite(mass) & np.isfinite(points[:, 0]) & np.isfinite(points[:, 1]) & np.isfinite(points[:, 2])
    interp = LinearNDInterpolator(points[ev], mass[ev], fill_value=np.nan)
    mass_gcns = interp(sf.gcns['MG'].to_numpy(), sf.gcns['MBP'].to_numpy(), sf.gcns['MRP'].to_numpy())
    sf.gcns = sf.gcns.with_columns(mass=mass_gcns)

    # now get the mass function and fit broken power law to it
    gcns_filter = sf.gcns.filter(pl.Series(sf.gcns_valid))
    Ntot, _, _ = np.histogram2d(gcns_filter['g_rp'].to_numpy(), gcns_filter['MG'].to_numpy(),
                                bins=[g_rp_bins, MG_bins])

    MG = sf.gcns['MG'].to_numpy()
    g_rp = sf.gcns['g_rp'].to_numpy()

    ev_filter = (MG < 10 * g_rp + 5) & sf.gcns_valid

    source_ids = sf.gcns['source_id'].to_numpy()[ev_filter]


    evs = []
    for k in range(p_samples_feh[0].shape[1]):
        evs.append(sf.idx_mod_gcns[ev_filter] == k)

    Nmasses = []
    params = []

    mass_bins = np.linspace(0.2, 0.7, 13)

    for i in trange(len(feh_data)):
        # get the number in HR diagram space based on draws
        Nboot = RNG.binomial(Ntot.astype(int), p_samples_feh[i].reshape((-1, n_g_rp, n_mg)))

        Nmass_boot = np.zeros((Nboot.shape[0], sf.data['Veff_samps'].to_numpy().shape[1], len(mass_bins) - 1))
        
        # boot strap the mass function
        for j in range(len(Nboot)):
            ndesire = Nboot[j].ravel()
            sids_j = []
            for k in range(len(ndesire)):
                if ndesire[k] > 0:
                    sids_j += list(RNG.choice(source_ids[evs[k]], ndesire[k], replace=False))
            filt_data = sf.gcns.filter(pl.col('source_id').is_in(sids_j))
            massesj = filt_data['mass'].to_numpy()

            # digitze bins
            ev_mass = (massesj >= mass_bins[0]) & \
                    (massesj < mass_bins[-1])
            bin_indices = np.digitize(massesj[ev_mass], mass_bins) - 1

            # bright the weights back to all sky
            # by default is weighted by size of healpix bin
            weights = filt_data['Veff_samps'].to_numpy()[ev_mass, :]
            weights /= hp.nside2pixarea(2 ** sf.sf_bins['healpix'])
            weights *= 4 * np.pi
            weights = 1 / weights
            
            # mask out 1 / 0
            finite_mask = np.isfinite(weights)

            n_bins = len(mass_bins) - 1
            histograms = np.zeros((weights.shape[1], n_bins))

            weights_masked = np.where(finite_mask, weights, 0)

            # do the boostrap
            for k in range(histograms.shape[0]):
                np.add.at(histograms[k], bin_indices, weights_masked[:, k])
            Nmass_boot[j] = histograms
            Nmass_boot[j] /= np.diff(mass_bins)
        # save results
        Nmass_boot = Nmass_boot.reshape(-1, Nmass_boot.shape[-1])
        Nmass_boot = np.nan_to_num(Nmass_boot,
                                   nan=0.0, posinf=0.0, neginf=0.0)
        Nmasses.append(Nmass_boot)

        # fit broken power law
        size = len(Nmass_boot)
        params_boot = np.zeros((size, 4))

        # do fit with median to get good initial guess
        x = mass_bins
        y = np.nanpercentile(Nmass_boot, 50, axis=0)
        widths = x[1:] - x[:-1]
        y_valid = y[np.isfinite(y) & (y > 0)]
        approx_C = np.max(y_valid) * np.mean(np.diff(x)) if len(y_valid) > 0 else 1e-3
        p0 = [np.log10(max(approx_C, 1e-6)), 1.0, 2.5, np.log10(0.5)]
        bounds = [(-20, 20), (-10, 10.0), (-10, 10),
                  (np.log10(0.3), np.log10(0.7))]
        
        res = minimize(
            lambda p, edges, y: float(obj_logsq_jax(jnp.array(p), jnp.array(edges), jnp.array(y))),
            p0, args=(x, y), method='L-BFGS-B', bounds=bounds,
        )
        p0 = res.x

        # do dummy run to force compile
        _dummy_y = np.ones((4, len(mass_bins) - 1))
        _dummy_p0 = np.array([0.0, 1.0, 2.5, np.log10(0.5)])
        fit_broken_powerlaw_batch(_dummy_y, mass_bins, _dummy_p0, bounds)

        params_boot = fit_broken_powerlaw_batch(Nmass_boot, mass_bins, p0, bounds)
        params.append(params_boot)

        # plot the results
        plt.figure(figsize=(15, 7))
        plt.hist(mass_bins[:-1], bins=mass_bins, weights=np.nanpercentile(Nmass_boot, 50, axis=0),
                histtype='step', edgecolor='k', lw=2, label=f'{fe_h_bins[i]:.3f} < [Fe/H] < {fe_h_bins[i + 1]:.3f}')
        plt.bar(x=mass_bins[:-1], height=np.nanpercentile(Nmass_boot, 97.5, axis=0) -
                                    np.nanpercentile(Nmass_boot, 2.5, axis=0),
                bottom=np.nanpercentile(Nmass_boot, 2.5, axis=0), width=np.diff(mass_bins),
                align='edge', linewidth=0, color='k', alpha=0.25)

        
        x_cont = np.linspace(x[0], x[-1], 100)
        xi_boot = np.array([xi_from_params(params_boot[j], x_cont) for j in range(len(params_boot))])
        plt.plot(x_cont,
                 np.nanpercentile(xi_boot, 50, axis=0),
                                '--', c='r', lw=2)
        lower = np.nanpercentile(xi_boot, 2.5, axis=0)
        upper = np.nanpercentile(xi_boot, 97.5, axis=0)
        plt.fill_between(x_cont, lower, upper, color='r', alpha=0.3)
        
        plt.legend(prop={'size': 16})
        plt.grid()
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel(r'Mass (M$_\odot$)')
        plt.ylabel(r'Mass Function (#/pc$^3$/M$_\odot$)')
        plt.xlim(mass_bins.min(), mass_bins.max())
        plt.savefig(f'paper_plots/mass_function/mf_feh_bin_{i + 1}.png',
                    bbox_inches='tight')
        plt.close()

    # make plots of the fitted parameters
    label = [r'$\alpha_1$', r'$\alpha_2$', r'$m_b$']
    savename = ['alpha_1', 'alpha_2', 'mb']
    for pi in range(1, 4):
        plt.figure(figsize=(15, 7))
        for i in range(len(feh_data)):
            midp = (fe_h_bins[i] + fe_h_bins[i + 1]) / 2
            if savename[pi - 1] != 'mb':
                plt.scatter(midp, np.nanpercentile(params[i], 50, axis=0)[pi], c='k')
                plt.errorbar([midp], [np.nanpercentile(params[i], 50, axis=0)[pi]],
                            yerr=np.diff(np.nanpercentile(params[i], [16, 50, 84], axis=0)[:, pi]).reshape((2, -1)),
                            color='k',
                            fmt='None',
                            xerr=[(fe_h_bins[i + 1] - fe_h_bins[i]) / 2])
            else:
                plt.scatter(midp, 10 ** np.nanpercentile(params[i], 50, axis=0)[pi], c='k')
                plt.errorbar([midp], [10 ** np.nanpercentile(params[i], 50, axis=0)[pi]],
                            yerr=np.diff(10 ** np.nanpercentile(params[i], [16, 50, 84], axis=0)[:, pi]).reshape((2, -1)),
                            color='k',
                            fmt='None',
                            xerr=[(fe_h_bins[i + 1] - fe_h_bins[i]) / 2])
        plt.grid()
        plt.ylabel(label[pi - 1])
        plt.xlabel('[Fe/H]')
        plt.savefig(f'paper_plots/mass_function/{savename[pi - 1]}_vs_feh.png',
                    bbox_inches='tight')
        plt.close()

    # save the results if need to look at later
    save_dict = {}
    for i, (ps, nm, pr) in enumerate(zip(p_samples_feh, Nmasses, params)):
        save_dict[f'p_samples_feh_{i}'] = ps
        save_dict[f'Nmasses_{i}'] = nm
        save_dict[f'params_{i}'] = pr
    save_dict['fe_h_bins'] = fe_h_bins

    np.savez_compressed('mf_results.npz', **save_dict)

    # loading example
    # data = np.load('mf_results.npz')
    # n_bins = len(data['fe_h_bins']) - 1

    # p_samples_feh = [data[f'p_samples_feh_{i}'] for i in range(n_bins)]
    # Nmasses       = [data[f'Nmasses_{i}']       for i in range(n_bins)]
    # params        = [data[f'params_{i}']        for i in range(n_bins)]
