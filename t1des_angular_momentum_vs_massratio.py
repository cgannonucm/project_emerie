#!/usr/bin/env python3
"""
Scatter plot of initial angular momentum magnitude vs final bound mass ratio
(NSphere / Galacticus) for the sample subhalos.

Initial angular momentum: |L| = |r x v| at the first satellite snapshot
from the Galacticus time-series.

Final bound mass ratio: NSphere final bound mass / Galacticus final bound mass,
following the pattern in t1des_massratio.py.
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from colossus.cosmology import cosmology

from analysis_scripts.mpl_standardize import mpl_params_update
from analysis_scripts.subhalo_timeseries import subhalo_timeseries_summary
from sidmcommon.nsphere_suite import summarize_nsphere_suite
from sidmcommon.galacticus import get_nodedata_byids
from subscript.tabulatehdf5 import tabulate_trees
from subscript.defaults import ParamKeys


def get_angular_momentum(galacticus_hdf5, halo_ids, snapshot='initial'):
    """
    Compute the angular momentum magnitude for each subhalo.

    |L| = |r x v| at the specified snapshot from the Galacticus time-series.

    Parameters
    ----------
    galacticus_hdf5 : h5py.File
        Open Galacticus HDF5 file.
    halo_ids : list of int
        Node indices of the subhalos.
    snapshot : str
        Which snapshot to use: 'initial' (infall, last element) or
        'final' (present day, first element).

    Returns
    -------
    dict
        Mapping {halo_id: |L|} in units of Mpc km/s.
    """
    ts_data = subhalo_timeseries_summary(galacticus_hdf5, tree_index=0)

    idx = -1 if snapshot == 'initial' else 0

    result = {}
    for halo_id in halo_ids:
        if halo_id not in ts_data:
            continue

        data = ts_data[halo_id]['data']

        pos = np.array([
            data['satellitePositionX'][idx],
            data['satellitePositionY'][idx],
            data['satellitePositionZ'][idx],
        ])
        vel = np.array([
            data['satelliteVelocityX'][idx],
            data['satelliteVelocityY'][idx],
            data['satelliteVelocityZ'][idx],
        ])

        L_mag = np.linalg.norm(np.cross(pos, vel))
        result[halo_id] = L_mag

    return result


def get_final_massratio(sim_dir, galacticus_hdf5, output_file=None, refresh=False):
    """
    Compute the final bound mass ratio (NSphere / Galacticus) for each subhalo.

    Parameters
    ----------
    sim_dir : str
        Path to the NSphere simulation suite directory.
    galacticus_hdf5 : h5py.File
        Open Galacticus HDF5 file.
    output_file : str, optional
        Path for the cached NSphere summary HDF5 file.
    refresh : bool
        If True, recompute the NSphere summary.

    Returns
    -------
    dict
        Mapping {halo_id: final_mass_nsphere / final_mass_galacticus}.
    """
    if output_file is None:
        os.makedirs('out/nsphere_summary', exist_ok=True)
        output_file = 'out/nsphere_summary/summary_temp'

    summary = summarize_nsphere_suite(sim_dir, output_file, refresh=refresh)

    tree = tabulate_trees(galacticus_hdf5)[0]
    mass_bound_gal = get_nodedata_byids(list(summary.keys()), tree, ParamKeys.mass_bound)

    result = {}
    for halo_id, val in summary.items():
        result[halo_id] = val['massbound'][-1] / mass_bound_gal[halo_id]

    return result


def plot_angular_momentum_vs_massratio(ax, galacticus_hdf5, sim_dir,
                                       snapshot='initial',
                                       nsphere_cachefile=None, refresh=False):
    """
    Scatter plot of |L| vs final bound mass ratio.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on.
    galacticus_hdf5 : h5py.File
        Open Galacticus HDF5 file.
    sim_dir : str
        Path to the NSphere simulation suite directory.
    snapshot : str
        Which snapshot to use for angular momentum:
        'initial' (infall) or 'final' (present day).
    nsphere_cachefile : str, optional
        Path for the cached NSphere summary HDF5 file.
    refresh : bool
        If True, recompute the NSphere summary.
    """
    massratio = get_final_massratio(sim_dir, galacticus_hdf5,
                                    output_file=nsphere_cachefile, refresh=refresh)
    halo_ids = list(massratio.keys())
    L_dict = get_angular_momentum(galacticus_hdf5, halo_ids, snapshot=snapshot)

    # Keep only halos present in both
    common_ids = [hid for hid in halo_ids if hid in L_dict]
    L_arr = np.array([L_dict[hid] for hid in common_ids])
    mr_arr = np.array([massratio[hid] for hid in common_ids])

    label = snapshot.capitalize()
    ax.scatter(L_arr, mr_arr, edgecolors='k', linewidths=0.5, zorder=2, label=label)

    ax.set_xlabel(r'$|\mathbf{L}|$ [Mpc km/s]')
    ax.set_ylabel(r'Final $M_\mathrm{bound}^\mathrm{NSphere} / M_\mathrm{bound}^\mathrm{Galacticus}$')
    ax.axhline(1.0, color='grey', linestyle='--', linewidth=0.8, zorder=1)


if __name__ == '__main__':
    mpl_params_update()

    gal_file = 'data/galacticus/galacticus-v2/mh1e13_z05_rmax_vmax.hdf5'
    nsphere_dir = './data/NSphere/NSphere-galacticus-hr-cdm-test'
    nsphere_cachefile = 'out/nsphere_summary/summary_massratio.hdf5'

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    with h5py.File(gal_file, 'r') as gout:
        plot_angular_momentum_vs_massratio(axes[0], gout, nsphere_dir,
                                           snapshot='initial',
                                           nsphere_cachefile=nsphere_cachefile)
        plot_angular_momentum_vs_massratio(axes[1], gout, nsphere_dir,
                                           snapshot='final',
                                           nsphere_cachefile=nsphere_cachefile)

    axes[0].set_title('Initial (Infall)')
    axes[1].set_title('Final (Present Day)')

    for ax in axes:
        ax.set_xscale('log')
        ax.set_yscale('log')

    fig.suptitle('Angular Momentum vs Final Bound Mass Ratio', y=1.01)
    fig.tight_layout()

    os.makedirs('out/plots', exist_ok=True)
    fig.savefig('out/plots/angular_momentum_vs_massratio.png', bbox_inches='tight')
    plt.show()
