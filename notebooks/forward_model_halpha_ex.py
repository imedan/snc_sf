from snc_sf.selection_function import *
from snc_sf.utils import *
from snc_sf.optimize import *

import numpy as np
import polars as pl
import matplotlib.pylab as plt
import os
import os.path
from matplotlib.colors import LogNorm
plt.style.use('%s/mystyle.mplstyle' % os.environ['MPL_STYLES'])
import jax
jax.config.update("jax_enable_x64", False)
from astropy.table import Table


if __name__ == '__main__':
    # get the full dataset
    MG_bin_list = [0, 20, 0.25]

    use_g_rp = False
    if use_g_rp:
        sf_bins={'healpix': 3,
                'phot_g_mean_mag': [0, 22, 1],
                'g_rp': [-0.4, 2.2, 0.05]}
        hr_col = 'g_rp'
    else:
        sf_bins={'healpix': 3,
                'phot_g_mean_mag': [0, 22, 1],
                'bp_rp': [-0.4, 5.25, 0.1]}
        hr_col = 'bp_rp'

    n_col = len(np.arange(*sf_bins[hr_col])) - 1
    n_mg = len(np.arange(*MG_bin_list)) - 1



    if not os.path.isfile('LF_snc_forward_data.csv'):
        data_file = 'obs_100pc_edr3.csv'

        # initilize
        mean = False
        sf = SNCSelectionFunction(data_file, sf_bins, MG_bin_list, mean=mean)

        # cross match with lineforest
        tbl = Table.read('../../../DR19/astraAllStarLineForest-0.6.0.fits', hdu=1)  # can download here: https://data.sdss.org/sas/dr19/spectro/astra/0.6.0/summary/astraAllStarLineForest-0.6.0.fits.gz
        names = [name for name in tbl.colnames if len(tbl[name].shape) <= 1]
        LF = pl.from_pandas(tbl[names].to_pandas())

        allstar = Table.read('../../../DR19/mwmAllStar-0.6.0.fits', hdu=1)  # can download here: https://data.sdss.org/sas/dr19/spectro/astra/0.6.0/summary/mwmAllStar-0.6.0.fits.gz

        data_LF = sf.data.filter(pl.col('source_id').is_in(allstar['gaia_dr3_source_id'])).join(LF, left_on='source_id', right_on='gaia_dr3_source_id', how='left')
        data_LF = data_LF.unique('source_id', keep='last')

        # save to file for testing
        cols_save = ['source_id', 'ra', 'ra_error', 'dec', 'dec_error', 'parallax',
                        'parallax_error', 'g_rp', 'phot_g_mean_mag', 'phot_g_mean_flux_over_error',
                        'eqw_h_alpha', 'abs_h_alpha', 'detection_stat_h_alpha', 'detection_raw_h_alpha']
        data_LF[cols_save].write_csv('LF_snc_forward_data.csv')

    # initialize with LF dataset
    mean = False
    sf = SNCSelectionFunction('LF_snc_forward_data.csv', sf_bins, MG_bin_list, mean=mean)

    # get A_jks
    sf.evalutate_Ajk(weight_volume=False)

    # have the bins for plotting
    col_bins = np.arange(*sf.sf_bins[hr_col])
    MG_bins = np.arange(*MG_bin_list)

    # run for emmission and absoprtion data
    filter_data_ab = sf.data.filter((pl.col('eqw_h_alpha') > -1) | (pl.col('eqw_h_alpha').is_null()))
    p_samples_ab, ev_valid_ab = sf.forward_model(filter_data_ab, num_warmup=5000, num_samples=2000, num_chains=2)

    filter_data_em = sf.data.filter((pl.col('eqw_h_alpha') < -1.))
    p_samples_em, ev_valid_em = sf.forward_model(filter_data_em, num_warmup=5000, num_samples=2000, num_chains=2)

    # plot the results
    RNG = np.random.default_rng(666)

    gcns_filter = sf.gcns.filter(pl.Series(sf.gcns_valid))

    # subpopulation probability
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(32, 10))
    dens = ax1.imshow(np.nanpercentile(p_samples_em.reshape((-1, n_col, n_mg)), 2.5, axis=0).T,
                    origin='lower', aspect='auto',
            extent=(col_bins.min(), col_bins.max(), MG_bins.min(), MG_bins.max()),
                    vmin=0, vmax=1, cmap='inferno')
    ax1.invert_yaxis()
    ax1.grid()
    ax1.set_title('Subpopulation Probability (2.5%)')

    dens = ax2.imshow(np.nanpercentile(p_samples_em.reshape((-1, n_col, n_mg)), 50, axis=0).T,
                    origin='lower', aspect='auto',
            extent=(col_bins.min(), col_bins.max(), MG_bins.min(), MG_bins.max()),
                    vmin=0, vmax=1, cmap='inferno')
    ax2.invert_yaxis()
    ax2.grid()
    ax2.set_title('Subpopulation Probability (50%)')


    dens = ax3.imshow(np.nanpercentile(p_samples_em.reshape((-1, n_col, n_mg)), 97.5, axis=0).T,
                    origin='lower', aspect='auto',
            extent=(col_bins.min(), col_bins.max(), MG_bins.min(), MG_bins.max()),
                    vmin=0, vmax=1, cmap='inferno')
    ax3.invert_yaxis()
    ax3.grid()
    ax3.set_title('Subpopulation Probability (97.5%)')

    f.subplots_adjust(right=0.84)
    cbar_ax = f.add_axes([0.85, 0.15, 0.01, 0.7])
    f.colorbar(dens, cax=cbar_ax, label=r'$p_{\mathsf{sub}, k}$')

    for ax in [ax1, ax2, ax3]:
        if use_g_rp:
            ax.set_xlabel(r'$G-RP$')
        else:
            ax.set_xlabel(r'$BP-RP$')
        ax.set_ylabel(r'$M_G$')

    plt.savefig('paper_plots/halpha_ex/forward_mod_prob_emission.png', bbox_inches='tight')
    plt.close()


    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(32, 10))
    dens = ax1.imshow(np.nanpercentile(p_samples_ab.reshape((-1, n_col, n_mg)), 2.5, axis=0).T,
                    origin='lower', aspect='auto',
            extent=(col_bins.min(), col_bins.max(), MG_bins.min(), MG_bins.max()),
                    vmin=0, vmax=1, cmap='inferno')
    ax1.invert_yaxis()
    ax1.grid()
    ax1.set_title('Subpopulation Probability (2.5%)')

    dens = ax2.imshow(np.nanpercentile(p_samples_ab.reshape((-1, n_col, n_mg)), 50, axis=0).T,
                    origin='lower', aspect='auto',
            extent=(col_bins.min(), col_bins.max(), MG_bins.min(), MG_bins.max()),
                    vmin=0, vmax=1, cmap='inferno')
    ax2.invert_yaxis()
    ax2.grid()
    ax2.set_title('Subpopulation Probability (50%)')


    dens = ax3.imshow(np.nanpercentile(p_samples_ab.reshape((-1, n_col, n_mg)), 97.5, axis=0).T,
                    origin='lower', aspect='auto',
            extent=(col_bins.min(), col_bins.max(), MG_bins.min(), MG_bins.max()),
                    vmin=0, vmax=1, cmap='inferno')
    ax3.invert_yaxis()
    ax3.grid()
    ax3.set_title('Subpopulation Probability (97.5%)')

    f.subplots_adjust(right=0.84)
    cbar_ax = f.add_axes([0.85, 0.15, 0.01, 0.7])
    f.colorbar(dens, cax=cbar_ax, label=r'$p_{\mathsf{sub}, k}$')

    for ax in [ax1, ax2, ax3]:
        if use_g_rp:
            ax.set_xlabel(r'$G-RP$')
        else:
            ax.set_xlabel(r'$BP-RP$')
        ax.set_ylabel(r'$M_G$')


    plt.savefig('paper_plots/halpha_ex/forward_mod_prob_absorption.png', bbox_inches='tight')
    plt.close()

    # M dwarf selection
    Ntot, _, _ = np.histogram2d(gcns_filter[hr_col].to_numpy(), gcns_filter['MG'].to_numpy(),
                                bins=[col_bins, MG_bins])
    Nboot_em = RNG.binomial(Ntot.astype(int), p_samples_em.reshape((-1, n_col, n_mg)))
    Nboot_ab = RNG.binomial(Ntot.astype(int), p_samples_ab.reshape((-1, n_col, n_mg)))

    f, (ax2) = plt.subplots(1, 1, figsize=(12, 10))

    dens = ax2.imshow(np.nanpercentile(Nboot_em, 50, axis=0).T, origin='lower', aspect='auto',
            extent=(col_bins.min(), col_bins.max(), MG_bins.min(), MG_bins.max()),
                    norm=LogNorm(), cmap='inferno')
    plt.colorbar(dens, ax=ax2, label='N')
    ax2.set_ylim(8, 17.5)
    ax2.invert_yaxis()
    ax2.grid()
    ax2.set_title('Forward Model Selection (50%)')

    for ax in [ax2]:
        if use_g_rp:
            ax2.set_xlim(1, 2)
            ax.set_xlabel(r'$G-RP$')
        else:
            ax2.set_xlim(1.5, 5)
            ax.set_xlabel(r'$BP-RP$')
        ax.set_ylabel(r'$M_G$')


    plt.savefig('paper_plots/halpha_ex/forward_mod_select_emission.png', bbox_inches='tight')
    plt.close()


    f, (ax2) = plt.subplots(1, 1, figsize=(12, 10))

    dens = ax2.imshow(np.nanpercentile(Nboot_ab, 50, axis=0).T, origin='lower', aspect='auto',
            extent=(col_bins.min(), col_bins.max(), MG_bins.min(), MG_bins.max()),
                    norm=LogNorm(), cmap='inferno')
    plt.colorbar(dens, ax=ax2, label='N')
    ax2.set_ylim(8, 17.5)
    ax2.invert_yaxis()
    ax2.grid()
    ax2.set_title('Forward Model Selection (50%)')

    for ax in [ax2]:
        if use_g_rp:
            ax2.set_xlim(1, 2)
            ax.set_xlabel(r'$G-RP$')
        else:
            ax2.set_xlim(1.5, 5)
            ax.set_xlabel(r'$BP-RP$')
        ax.set_ylabel(r'$M_G$')


    plt.savefig('paper_plots/halpha_ex/forward_mod_select_absorption.png', bbox_inches='tight')
    plt.close()


    # plot M dwarf section ratio
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(36, 10))

    dens = ax1.imshow(np.nanpercentile(Nboot_em, 50, axis=0).T, origin='lower', aspect='auto',
            extent=(col_bins.min(), col_bins.max(), MG_bins.min(), MG_bins.max()),
                    norm=LogNorm(), cmap='inferno')
    plt.colorbar(dens, ax=ax1, label='N')
    ax1.set_title('Emission\n' + r'(E.W. $< -1 \ \AA$)')

    dens = ax2.imshow(np.nanpercentile(Nboot_ab, 50, axis=0).T, origin='lower', aspect='auto',
            extent=(col_bins.min(), col_bins.max(), MG_bins.min(), MG_bins.max()),
                    norm=LogNorm(), cmap='inferno')
    plt.colorbar(dens, ax=ax2, label='N')
    ax2.set_title('Comparison Sample\n' + r'(E.W. $> -1 \ \AA$ or E.W. is Null)')

    Nem = np.nanpercentile(Nboot_em, 50, axis=0).T
    Nab = np.nanpercentile(Nboot_ab, 50, axis=0).T

    dens = ax3.imshow(Nem / (Nem + Nab),
                    origin='lower', aspect='auto',
            extent=(col_bins.min(), col_bins.max(), MG_bins.min(), MG_bins.max()),
                    cmap='seismic', vmin=0, vmax=1)
    plt.colorbar(dens, ax=ax3, label=r'N / N$_{tot}$')
    ax3.set_title('Emission /  (Emission + Comparison Sample)\n')

    for ax in [ax1, ax2, ax3]:
        if use_g_rp:
            ax.set_xlabel(r'$G-RP$')
            ax.set_xlim(1, 2)
        else:
            ax2.set_xlim(1.5, 5)
            ax.set_xlabel(r'$BP-RP$')
        ax.set_ylabel(r'$M_G$')
        ax.set_ylim(8, 17.5)
        ax.invert_yaxis()
        ax.grid()

    plt.savefig('paper_plots/halpha_ex/forward_mod_compare_Ms.png', bbox_inches='tight')
    plt.close()
