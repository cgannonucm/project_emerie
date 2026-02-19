#!/usr/bin/env python3
"""
Tidal mass history plotting for Galacticus subhalos and NSphere T1DES runs.

Provides:
  - plot_masshistory_galacticus(): bound mass evolution from Galacticus semi-analytic model.
  - plot_masshistory_nsphere(): bound mass evolution from NSphere CDM simulations.
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from analysis_scripts.mpl_standardize import mpl_params_update
from subscript.defaults import ParamKeys
from colossus.cosmology import cosmology

from analysis_scripts.subhalo_timeseries import subhalo_timeseries_summary
from sidmcommon.nsphere_suite import summarize_nsphere_suite


def plot_masshistory_galacticus(ax, galacticus_hdf5, halo_id, comso):
    """
    Plot the bound mass history of a subhalo as a function of time since infall.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on.
    galacticus_hdf5 : h5py.File
        Open Galacticus HDF5 file.
    halo_id : int
        Node index of the subhalo to plot.
    comso : colossus.cosmology.Cosmology
        Cosmology object for converting redshift to cosmic time [Gyr].
    """
    ts_data = subhalo_timeseries_summary(galacticus_hdf5, tree_index=0)

    if halo_id not in ts_data:
        ax.text(0.5, 0.5, f'Node {halo_id}\nnot found',
                transform=ax.transAxes, ha='center', va='center', fontsize=9)
        ax.set_title(f'Node {halo_id}')
        return

    ts = ts_data[halo_id]
    zsnaps = ts['zsnaps']
    mass_bound = ts['data'][ParamKeys.mass_bound]
    mass_infall = ts['data'][ParamKeys.mass_basic]

    # Convert redshift to time since infall (Gyr)
    t_infall = comso.age(zsnaps[-1])
    t_since_infall = comso.age(zsnaps) - t_infall

    # Normalize by infall mass (first snapshot as satellite)
    mass_ratio = mass_bound / mass_infall[0]

    ax.plot(t_since_infall, mass_ratio, label='Galacticus')



def plot_masshistory_nsphere(ax, sim_dir, halo_id, cachedir=None, refresh=False):
    """
    Plot the bound mass history of a T1DES NSphere subhalo.

    Uses summarize_nsphere_suite to compute and cache bound mass time series,
    then overlays on ax as mass ratio vs. time since infall.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on.
    sim_dir : str
        Path to the NSphere simulation suite directory.
    halo_id : int
        Halo ID matching the nsphere_params_<halo_id> subdirectory.
    cachedir : str
        Directory where the cached HDF5 summary is written.
        Cache file written to: {cachedir}/masshistory-{basename(sim_dir)}.hdf5
    refresh : bool
        If True, recompute the summary even if the cache exists.
    """
    cachedir = cachedir if cachedir is not None else 'out/cache'

    sim_name = os.path.basename(os.path.normpath(sim_dir))
    output_file = os.path.join(cachedir, f'masshistory-{sim_name}.hdf5')
    os.makedirs(cachedir, exist_ok=True)

    summary = summarize_nsphere_suite(sim_dir, output_file=output_file, refresh=refresh)

    if halo_id not in summary:
        return

    # NSphere t=0 is infall; convert Myr -> Gyr for shared x-axis with Galacticus
    time_gyr = summary[halo_id]['time'] * 1e-3 # Convert Myr to Gyr

    mass_bound = summary[halo_id]['massbound']
    mass_infall = mass_bound[0]
    mass_ratio = mass_bound / mass_infall

    ax.plot(time_gyr, mass_ratio, label='NSphere (CDM)', linestyle='--')


if __name__ == '__main__':
    mpl_params_update()

    cosmo = cosmology.setCosmology('planck18')

    gal_file = 'data/galacticus/galacticus-v2/mh1e13_z05_rmax_vmax.hdf5'
    nsphere_dir = './data/NSphere/NSphere-galacticus-hr-cdm-test'
    nsphere_cachedir = 'out/nsphere_summary'
    halo_ids = [3375, 3660, 3731, 8876, 2328, 6267, 10101, 3960, 3488]

    fig, axes = plt.subplots(3, 3, figsize=(12, 10))

    with h5py.File(gal_file, 'r') as gout:
        for ax, halo_id in zip(axes.flatten(), halo_ids):
            plot_masshistory_galacticus(ax, gout, halo_id, cosmo)
            plot_masshistory_nsphere(ax, nsphere_dir, halo_id, nsphere_cachedir)

            ax.set_yscale('log')
            ax.set_ylim(5e-4, 1.1)
            ax.set_xlim(0, 8)
            ax.set_xlabel('Time since infall [Gyr]')
            ax.set_ylabel(r'$M_\mathrm{bound} / M_\mathrm{infall}$')
            ax.set_title(f'Halo {halo_id}')

    axes[0,0].legend(loc='upper right', fontsize=9)

    fig.suptitle('Subhalo Bound Mass Histories', y=1.01)
    fig.tight_layout()

    os.makedirs('out/plots', exist_ok=True)
    fig.savefig('out/plots/masshistory_sample.png', bbox_inches='tight')
    plt.show()
