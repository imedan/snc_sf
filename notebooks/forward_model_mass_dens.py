from snc_sf.selection_function import *
from snc_sf.utils import *
from snc_sf.optimize import *

import numpy as np
import polars as pl
import matplotlib.pylab as plt
import matplotlib.cm as cm
import matplotlib
from matplotlib.ticker import ScalarFormatter, LogFormatter, FuncFormatter, MultipleLocator, AutoMinorLocator, FixedLocator
import os
from matplotlib.colors import LogNorm
plt.style.use('%s/mystyle.mplstyle' % os.environ['MPL_STYLES'])
from astropy.table import Table, vstack
from scipy.interpolate import LinearNDInterpolator
from tqdm import trange


def mass_est(MK, *args, feh=None):
    """
    Get mass based on Mann 2019 relation
    """
    if feh is None:
        mass = 10 ** np.polyval(args[::-1], MK - 7.5)
    else:
        massi = 10 ** np.polyval(args[:-1][::-1], MK - 7.5)
        mass = (1 + args[-1] * feh) * massi
    mass[MK < 4.5] = np.nan
    mass[MK > 10.5] = np.nan
    return mass


if __name__ == '__main__':
    # =============================
    # SET UP DATA FOR FORWARD MODEL
    # =============================
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
                'bp_rp': [-0.4, 5.25, 0.15]}
        hr_col = 'bp_rp'

    n_col = len(np.arange(*sf_bins[hr_col])) - 1
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
        data_file = 'obs_100pc_edr3.csv'
        sf = SNCSelectionFunction(data_file,
                                sf_bins,
                                MG_bin_list,
                                mean=mean,
                                pre_filt=pre_filt)
        
        # load the metallicity data
        tbl = Table.read('../../../DR19/astraFrankenstein-0.6.0.fits', hdu=1)  # can download here: https://data.sdss.org/sas/dr19/spectro/astra/0.6.0/summary/astraMWMLite-0.6.0.fits.gz
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

    # =======================================
    # RUN THE FORWARD MODEL FOR EACH FE/H BIN
    # =======================================
    mean = False
    sf = SNCSelectionFunction('aspcap_slam_fe_h_corr_snc_forward_data.csv',
                              sf_bins, MG_bin_list,
                              mean=mean, pre_filt=pre_filt)
    # get A_jks
    sf.evalutate_Ajk(weight_volume=False)
    # bins for plotting
    col_bins = np.arange(*sf.sf_bins[hr_col])
    MG_bins = np.arange(*MG_bin_list)

    # run the forward model
    fe_h_bins = np.array([-0.6, -0.3, -0.15, -0.05, 0.05, 0.2])
    feh_data = []
    p_samples_feh = []
    ev_valid_feh = []

    for i in range(len(fe_h_bins) - 1):
        filter_data = sf.data.filter((pl.col('fe_h_corr') > fe_h_bins[i]) & (pl.col('fe_h_corr') <= fe_h_bins[i + 1]))
        p_samples, ev_valid = sf.forward_model(filter_data,
                                               num_warmup=5000,
                                               num_samples=2000,
                                               num_chains=2)

        feh_data.append(filter_data)
        p_samples_feh.append(p_samples)
        ev_valid_feh.append(ev_valid)

    RNG = np.random.default_rng(666)

    # =======================================
    # ESTIMATE MASS FOR GCNS USING MANN 2019
    # =======================================
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

    # ======================================================
    # FIT MASS DENSITY FOR EACH METALLICITY BIN
    # ======================================================
    gcns_filter = sf.gcns.filter(pl.Series(sf.gcns_valid))
    Ntot, _, _ = np.histogram2d(gcns_filter[hr_col].to_numpy(), gcns_filter['MG'].to_numpy(),
                                bins=[col_bins, MG_bins])

    MG = sf.gcns['MG'].to_numpy()
    g_rp = sf.gcns['g_rp'].to_numpy()

    ev_filter = (MG < 10 * g_rp + 5) & sf.gcns_valid

    source_ids = sf.gcns['source_id'].to_numpy()[ev_filter]


    evs = []
    for k in range(p_samples_feh[0].shape[1]):
        evs.append(sf.idx_mod_gcns[ev_filter] == k)

    Nmasses = []
    mass_bins = np.linspace(0.2, 0.7, 6)

    for i in trange(len(feh_data)):
        # get the number in HR diagram space based on draws
        Nboot = RNG.binomial(Ntot.astype(int), p_samples_feh[i].reshape((-1, n_col, n_mg)))

        Nmass_boot = np.zeros((Nboot.shape[0], sf.data['Veff_samps'].to_numpy().shape[1], len(mass_bins) - 1))
        
        # boot strap the mass densities using each sample of chain
        # and each Vmax value
        for j in range(len(Nboot)):
            ndesire = Nboot[j].ravel()
            sids_j = []
            # randomlly select GCNS based on p_sub
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
        # save results
        Nmass_boot = Nmass_boot.reshape(-1, Nmass_boot.shape[-1])
        Nmass_boot = np.nan_to_num(Nmass_boot,
                                   nan=0.0, posinf=0.0, neginf=0.0)
        Nmasses.append(Nmass_boot)

        # plot the results
        plt.figure(figsize=(15, 7))
        plt.hist(mass_bins[:-1], bins=mass_bins, weights=np.nanpercentile(Nmass_boot, 50, axis=0),
                histtype='step', edgecolor='k', lw=2, label=f'{fe_h_bins[i]:.3f} < [Fe/H] < {fe_h_bins[i + 1]:.3f}')
        plt.bar(x=mass_bins[:-1], height=np.nanpercentile(Nmass_boot, 97.5, axis=0) -
                                    np.nanpercentile(Nmass_boot, 2.5, axis=0),
                bottom=np.nanpercentile(Nmass_boot, 2.5, axis=0), width=np.diff(mass_bins),
                align='edge', linewidth=0, color='k', alpha=0.25)
        
        plt.legend(prop={'size': 16})
        plt.grid()
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel(r'Mass (M$_\odot$)')
        plt.ylabel(r'Number Density (#/pc$^3$)')
        plt.xlim(mass_bins.min(), mass_bins.max())
        plt.savefig(f'paper_plots/mass_dens/mf_feh_bin_{i + 1}.png',
                    bbox_inches='tight')
        plt.close()


    # =====================
    # PLOT AND SAVE RESULTS
    # =====================
    n_bins = len(fe_h_bins) - 1
    cmap = matplotlib.colormaps["inferno"]
    normalized_values = np.linspace(0, 1, n_bins + 1) 
    colors_rgba = cmap(normalized_values)

    plt.figure(figsize=(15, 7))

    for i in range(n_bins):
        # plot the obs MF
        Nmass_boot = Nmasses[i]
        mids = (mass_bins[:-1] + mass_bins[1:]) / 2
        plt.scatter(mids, np.log10(np.nanpercentile(Nmass_boot, 50, axis=0)),
                    c=colors_rgba[i],
                    label=f'{fe_h_bins[i]:.2f} < [Fe/H] < {fe_h_bins[i + 1]:.2f}')
        plt.errorbar(mids, np.log10(np.nanpercentile(Nmass_boot, 50, axis=0)),
                     # xerr=np.diff(mass_bins) / 2,
                     yerr=np.diff(np.log10(np.nanpercentile(Nmass_boot, [2.5, 50, 97.5], axis=0)), axis=0),
                     ecolor=colors_rgba[i], fmt='none')
    plt.legend(prop={'size': 16})
    plt.grid()
    plt.xscale('log')
    ax = plt.gca()
    x_ticks = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    ax.xaxis.set_major_locator(FixedLocator(x_ticks))
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    formatter.set_useOffset(False)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    plt.xlabel(r'Mass (M$_\odot$)')
    plt.ylabel(r'$\log_{10}$ Numer Density (#/pc$^3$)')
    plt.xlim(mass_bins.min(), mass_bins.max())
    plt.savefig(f'paper_plots/mass_dens/mf_all.png',
                bbox_inches='tight')
    plt.close()

    # save the results if need to look at later
    save_dict = {}
    for i, (ps, nm) in enumerate(zip(p_samples_feh, Nmasses)):
        save_dict[f'p_samples_feh_{i}'] = ps
        save_dict[f'Nmasses_{i}'] = nm
    save_dict['mass_bins'] = mass_bins
    save_dict['fe_h_bins'] = fe_h_bins

    np.savez_compressed('mass_dens_results.npz', **save_dict)

    # loading example
    # data = np.load('mf_results.npz')
    # n_bins = len(data['fe_h_bins']) - 1

    # p_samples_feh = [data[f'p_samples_feh_{i}'] for i in range(n_bins)]
    # Nmasses       = [data[f'Nmasses_{i}']       for i in range(n_bins)]
