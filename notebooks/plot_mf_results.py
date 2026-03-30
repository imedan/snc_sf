import numpy as np
import matplotlib.pylab as plt
import os
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
import matplotlib
from matplotlib.ticker import ScalarFormatter, LogFormatter, FuncFormatter, MultipleLocator, AutoMinorLocator, FixedLocator
from forward_model_MF import xi_from_params
plt.style.use('%s/mystyle.mplstyle' % os.environ['MPL_STYLES'])


if __name__ == '__main__':
    # loading example
    data = np.load('mf_results.npz')
    n_bins = len(data['fe_h_bins']) - 1
    fe_h_bins = data['fe_h_bins']

    p_samples_feh = [data[f'p_samples_feh_{i}'] for i in range(n_bins)]
    Nmasses       = [data[f'Nmasses_{i}']       for i in range(n_bins)]
    params        = [data[f'params_{i}']        for i in range(n_bins)]

    mass_bins = np.linspace(0.2, 0.7, Nmasses[0].shape[-1] + 1)

    cmap = matplotlib.colormaps["inferno"]
    normalized_values = np.linspace(0, 1, n_bins + 1) 
    colors_rgba = cmap(normalized_values)

    plt.figure(figsize=(15, 7))

    for i in range(n_bins):
        # plot the obs MF
        Nmass_boot = Nmasses[i]
        # plt.hist(mass_bins[:-1], bins=mass_bins, weights=np.nanpercentile(Nmass_boot, 50, axis=0),
        #         histtype='step', edgecolor=colors_rgba[i], lw=2, label=f'{fe_h_bins[i]:.3f} < [Fe/H] < {fe_h_bins[i + 1]:.3f}')
        # plt.bar(x=mass_bins[:-1], height=np.nanpercentile(Nmass_boot, 97.5, axis=0) -
        #                             np.nanpercentile(Nmass_boot, 2.5, axis=0),
        #         bottom=np.nanpercentile(Nmass_boot, 2.5, axis=0), width=np.diff(mass_bins),
        #         align='edge', linewidth=0, color=colors_rgba[i], alpha=0.3)
        mids = (mass_bins[:-1] + mass_bins[1:]) / 2
        plt.scatter(mids, np.log10(np.nanpercentile(Nmass_boot, 50, axis=0)),
                    c=colors_rgba[i],
                    label=f'{fe_h_bins[i]:.2f} < [Fe/H] < {fe_h_bins[i + 1]:.2f}')
        plt.errorbar(mids, np.log10(np.nanpercentile(Nmass_boot, 50, axis=0)),
                     yerr=np.diff(np.log10(np.nanpercentile(Nmass_boot, [2.5, 50, 97.5], axis=0)), axis=0),
                     ecolor=colors_rgba[i], fmt='none')
        
        
        # plot the fit
        params_boot = params[i]
        x_cont = np.linspace(mass_bins[0], mass_bins[-1], 100)
        xi_boot = np.array([xi_from_params(params_boot[j], x_cont) for j in range(len(params_boot))])
        plt.plot(x_cont,
                 np.log10(np.nanpercentile(xi_boot, 50, axis=0)),
                                '--', c=colors_rgba[i], lw=2)
        lower = np.log10(np.nanpercentile(xi_boot, 2.5, axis=0))
        upper = np.log10(np.nanpercentile(xi_boot, 97.5, axis=0))
        plt.fill_between(x_cont, lower, upper, color=colors_rgba[i], alpha=0.3)

    plt.legend(prop={'size': 16})
    plt.grid()
    plt.xscale('log')
    ax = plt.gca()
    x_ticks = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]  # adjust to match your mass_bins range
    ax.xaxis.set_major_locator(FixedLocator(x_ticks))
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    formatter.set_useOffset(False)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    plt.xlabel(r'Mass (M$_\odot$)')
    plt.ylabel(r'$\log_{10}$ Mass Function (#/pc$^3$/M$_\odot$)')
    plt.xlim(mass_bins.min(), mass_bins.max())
    plt.savefig(f'paper_plots/mass_function/mf_all.png',
                bbox_inches='tight')
    plt.close()

    label = [r'$\alpha_1$', r'$\alpha_2$', r'$m_b$']
    savename = ['alpha_1', 'alpha_2', 'mb']
    for pi in range(1, 4):
        plt.figure(figsize=(10, 7))
        for i in range(len(params)):
            midp = (fe_h_bins[i] + fe_h_bins[i + 1]) / 2
            if savename[pi - 1] != 'mb':
                plt.scatter(midp, np.nanpercentile(params[i], 50, axis=0)[pi],
                            c=colors_rgba[i])
                plt.errorbar([midp], [np.nanpercentile(params[i], 50, axis=0)[pi]],
                            yerr=np.diff(np.nanpercentile(params[i], [16, 50, 84], axis=0)[:, pi]).reshape((2, -1)),
                            color=colors_rgba[i],
                            fmt='None',
                            xerr=[(fe_h_bins[i + 1] - fe_h_bins[i]) / 2])
            else:
                plt.scatter(midp, 10 ** np.nanpercentile(params[i], 50, axis=0)[pi],
                            c=colors_rgba[i])
                plt.errorbar([midp], [10 ** np.nanpercentile(params[i], 50, axis=0)[pi]],
                            yerr=np.diff(10 ** np.nanpercentile(params[i], [16, 50, 84], axis=0)[:, pi]).reshape((2, -1)),
                            color=colors_rgba[i],
                            fmt='None',
                            xerr=[(fe_h_bins[i + 1] - fe_h_bins[i]) / 2])
        plt.grid()
        plt.ylabel(label[pi - 1])
        plt.xlabel('[Fe/H]')
        plt.savefig(f'paper_plots/mass_function/{savename[pi - 1]}_vs_feh_color.png',
                    bbox_inches='tight')
        plt.close()
