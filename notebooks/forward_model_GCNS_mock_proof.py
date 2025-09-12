from snc_sf.selection_function import *
from snc_sf.utils import *
from snc_sf.optimize import *

import numpy as np
from scipy.stats import multivariate_normal
import polars as pl
import matplotlib.pylab as plt
import os
from matplotlib.colors import LogNorm
plt.style.use('%s/mystyle.mplstyle' % os.environ['MPL_STYLES'])
from scipy.optimize import minimize
from tqdm import tqdm
import jax
jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax.nn import softplus, sigmoid
from jax.experimental.sparse import BCOO   # JAX sparse
from jaxopt import LBFGS
import optax
from jax import lax


if __name__ == '__main__':
    MG_bin_list = [0, 20, 0.5]

    sf_bins={'healpix': 3,
            'phot_g_mean_mag': [0, 22, 1],
            'g_rp': [-0.4, 2.2, 0.1]}

    n_g_rp = len(np.arange(*sf_bins['g_rp'])) - 1
    n_mg = len(np.arange(*MG_bin_list)) - 1

    data_file = '../../obs_100pc_edr3.csv'

    # initilize
    mean = True
    sf = SNCSelectionFunction(data_file, sf_bins, MG_bin_list, mean=mean)

    # get A_jks
    sf.evalutate_Ajk(weight_volume=False)

    # now filter the data based on some made up subsmaple selection function

    g_rp_bins = np.arange(*sf.sf_bins['g_rp'])
    MG_bins = np.arange(*MG_bin_list)

    X, Y = np.meshgrid(0.5 * (g_rp_bins[:-1] + g_rp_bins[1:]),
                    0.5 * (MG_bins[:-1] + MG_bins[1:]), indexing='ij')

    # p_true = multivariate_normal([0.5, 6], [[0.02, 2], [0.2, 5.]])

    # # get the test 'true' number density
    # ptrue = p_true.pdf(np.column_stack((X.ravel(),
    #                                     Y.ravel())))
    # ptrue /= np.nanmax(ptrue)
    # ptrue *= 0.9
    # ptrue = ptrue.reshape(X.shape)

    ptrue = (np.sin(Y) + 1) * 0.4


    # select based on above
    RNG = np.random.default_rng(666)

    # get number and downselect
    N, _, _ = np.histogram2d(sf.gcns['g_rp'].to_numpy()[sf.gcns_valid], sf.gcns['MG'].to_numpy()[sf.gcns_valid],
                            bins=[g_rp_bins, MG_bins])
    Ntarg = int(np.sum(N * ptrue))
    source_ids = sf.gcns['source_id'].to_numpy()[sf.gcns_valid]
    source_id_test = RNG.choice(source_ids, Ntarg,
                                p=ptrue.ravel()[sf.idx_mod_gcns[sf.gcns_valid]] / np.sum(ptrue.ravel()[sf.idx_mod_gcns[sf.gcns_valid]]),
                                replace=False)

    filter_data = sf.data.filter(pl.col('source_id').is_in(source_id_test))

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    dens = ax1.imshow(ptrue.T, origin='lower', aspect='auto',
            extent=(g_rp_bins.min(), g_rp_bins.max(), MG_bins.min(), MG_bins.max()))
    plt.colorbar(dens, ax=ax1)
    ax1.invert_yaxis()
    ax1.set_xlabel(r'$G - RP$')
    ax1.set_ylabel(r'$M_G$')
    ax1.set_title('Subpopulation Probability')
    ax1.grid()

    _, _, _, dens = ax2.hist2d(filter_data['g_rp'].to_numpy(), filter_data['MG'].to_numpy(),
                            bins=[g_rp_bins, MG_bins], norm=LogNorm())
    plt.colorbar(dens, ax=ax2)
    ax2.invert_yaxis()
    ax2.grid()
    ax2.set_xlabel(r'$G - RP$')
    ax2.set_ylabel(r'$M_G$')
    ax2.set_title('Mock Dataset')
    plt.savefig('paper_plots/mock_example/mock_data.png', bbox_inches='tight')
    plt.close()

    # run optimization
    p_samples, ev_valid = sf.forward_model(filter_data)

    # plot results
    p_mean = jnp.mean(p_samples, axis=0)
    p_cred_low = jnp.percentile(p_samples, 2.5, axis=0)
    p_cred_high = jnp.percentile(p_samples, 97.5, axis=0)

    Ntot, _, _ = np.histogram2d(sf.gcns['g_rp'].to_numpy()[sf.gcns_valid], sf.gcns['MG'].to_numpy()[sf.gcns_valid],
                            bins=[g_rp_bins, MG_bins])

    Nboot = RNG.binomial(Ntot.astype(int), p_samples.reshape((-1, n_g_rp, n_mg)))

    # source_ids = sf.gcns['source_id'].to_numpy()[sf.gcns_valid]

    # for i in range(len(p_samples)):
    #     # Ntarg = int(np.sum(Ntot.ravel() * n_samples[i]))
    #     # norm = 1. / np.sum(n_samples[i][idx_mod_gcns[gcns_valid]])
    #     # source_id_testi = RNG.choice(source_ids, Ntarg,
    #     #                              p=n_samples[i][idx_mod_gcns[gcns_valid]] * norm, replace=False)
    #     # filter_datai = sf.data.filter(pl.col('source_id').is_in(source_id_testi))
    #     # Nboot[i], _, _ = np.histogram2d(filter_datai['g_rp'].to_numpy(), filter_datai['MG'].to_numpy(),
    #     #                         bins=[g_rp_bins, MG_bins])
    #     Nboot[i] = RNG.binomial(Ntot.astype(int), p_samples[i].reshape((n_g_rp, n_mg)))

    Nobs, _, _ = np.histogram2d(filter_data['g_rp'].to_numpy()[ev_valid], filter_data['MG'].to_numpy()[ev_valid],
                                bins=[g_rp_bins, MG_bins])


    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(36, 10))
    dens = ax1.imshow(np.nanpercentile(Nboot, 50, axis=0).T, origin='lower', aspect='auto',
            extent=(g_rp_bins.min(), g_rp_bins.max(), MG_bins.min(), MG_bins.max()),
                    norm=LogNorm(vmin=1e-1, vmax=1e4))
    plt.colorbar(dens, ax=ax1)
    ax1.invert_yaxis()
    ax1.grid()
    ax1.set_xlabel(r'$G - RP$')
    ax1.set_ylabel(r'$M_G$')
    ax1.set_title('Forward Modeling GCNS\nFrom Selected SNC stars')

    dens = ax2.imshow(Ntot.T * ptrue.T, origin='lower', aspect='auto',
            extent=(g_rp_bins.min(), g_rp_bins.max(), MG_bins.min(), MG_bins.max()), 
                    norm=LogNorm(vmin=1e-1, vmax=1e4))
    plt.colorbar(dens, ax=ax2)
    ax2.invert_yaxis()
    ax2.grid()
    ax2.set_xlabel(r'$G - RP$')
    ax2.set_ylabel(r'$M_G$')
    ax2.set_title('Directly Selecting GCNS (True)')

    dens = ax3.imshow(Nobs.T, origin='lower', aspect='auto',
            extent=(g_rp_bins.min(), g_rp_bins.max(), MG_bins.min(), MG_bins.max()),
                    norm=LogNorm(vmin=1e-1, vmax=1e4))
    plt.colorbar(dens, ax=ax3)
    ax3.invert_yaxis()
    ax3.grid()
    ax3.set_title('Selected SNC stars')
    ax3.set_xlabel(r'$G - RP$')
    ax3.set_ylabel(r'$M_G$')
    plt.savefig('paper_plots/mock_example/mcmc_results.png', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(15, 7))
    plt.hist(MG_bins[:-1], bins=MG_bins, weights=np.nansum(Ntot * ptrue, axis=0),
            histtype='step', edgecolor='k', lw=2, label='Directly Selecting GCNS (True)')
    plt.hist(MG_bins[:-1], bins=MG_bins, weights=np.nansum(Nobs, axis=0),
            histtype='step', edgecolor='b', lw=2, label='Selected SNC stars')

    plt.hist(MG_bins[:-1], bins=MG_bins, weights=np.nansum(np.nanpercentile(Nboot, 50, axis=0), axis=0),
            histtype='step', edgecolor='r', lw=2, label='Forward Modeling GCNS\nFrom Selected SNC stars')
    plt.bar(x=MG_bins[:-1], height=np.nansum(np.nanpercentile(Nboot, 97.5, axis=0), axis=0) -
                                np.nansum(np.nanpercentile(Nboot, 2.5, axis=0), axis=0),
            bottom=np.nansum(np.nanpercentile(Nboot, 2.5, axis=0), axis=0), width=np.diff(MG_bins),
            align='edge', linewidth=0, color='red', alpha=0.25, zorder=-1)

    plt.legend(prop={'size':16})
    plt.grid()
    plt.yscale('log')
    plt.xlabel(r'$M_G$')
    plt.ylabel('N')
    plt.savefig('paper_plots/mock_example/mcmc_results_1D.png', bbox_inches='tight')
    plt.close()













