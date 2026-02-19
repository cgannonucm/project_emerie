#!/usr/bin/env python3
from analysis_scripts.mpl_standardize import mpl_params_update
import sys
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from subscript.defaults import ParamKeys

from analysis_scripts.subhalo_timeseries import subhalo_timeseries_summary
from analysis_scripts.vmax_rmax_nsphere import get_nsphere_vmax, get_nsphere_rmax
from sidmcommon.nsphere_suite import summarize_nsphere_suite
from sidmcommon.const import Const


def plot_tidal_tracks_galacticus(ax, galacticus_hdf5: h5py.File, node_index: int, tree_index: int = 0):
    """
    Plot the tidal track (R_max vs V_max) for a single subhalo over time.

    Extracts the V_max and R_max time series for the specified subhalo and
    plots the trajectory in the (V_max, R_max) plane on the provided axes.
    The track runs from infall (start of satellite phase) to the last
    surviving snapshot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to plot on.
    galacticus_hdf5 : h5py.File
        Open Galacticus HDF5 file object.
    node_index : int
        Node ID of the subhalo to plot the tidal track for.
    """
    ts_data = subhalo_timeseries_summary(galacticus_hdf5, tree_index=tree_index)

    ts = ts_data[node_index]
    vmax = ts['data'][ParamKeys.dark_matter_profile_dmo_velocity_max]
    rmax = ts['data'][ParamKeys.dark_matter_profile_dmo_radius_velocity_max] * 1e3 # Convert to kpc

    ax.plot(vmax, rmax)
    ax.set_xlabel(r'$V_{\rm max}$ [km/s]')
    ax.set_ylabel(r'$R_{\rm max}$ [kpc]')


def plot_tidal_tracks_t1des(ax, sim_dir, node_index: int, smoothing_sigma=1.0,
                              cachedir=None, refresh=False):
    """
    Plot the tidal track (R_max vs V_max) for T1DES NSphere simulation.

    Extracts V_max and R_max time series from NSphere simulations using
    summarize_nsphere_suite, applies Gaussian smoothing, and plots the
    trajectory in the (V_max, R_max) plane.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to plot on.
    sim_dir : str
        Path to the NSphere simulation suite directory.
    node_index : int
        Halo ID matching the nsphere_params_<node_index> subdirectory.
    smoothing_sigma : float
        Standard deviation of Gaussian kernel for smoothing (in units of snapshots).
        Set to 0 or None to disable smoothing. Default: 1.0
    cachedir : str
        Directory where cached HDF5 summary is written.
        Cache file: {cachedir}/tidal_tracks-{basename(sim_dir)}.hdf5
        Default: 'out/cache'
    refresh : bool
        If True, recompute the summary even if cache exists.
    """
    cachedir = cachedir if cachedir is not None else 'out/cache'

    sim_name = os.path.basename(os.path.normpath(sim_dir))
    output_file = os.path.join(cachedir, f'tidal_tracks-{sim_name}.hdf5')
    os.makedirs(cachedir, exist_ok=True)

    # Define custom summary functions to extract V_max and R_max from particle data
    summary_functions = {
        'vmax': get_nsphere_vmax,
        'rmax': get_nsphere_rmax,
    }

    # Get cached or computed tidal track data
    summary = summarize_nsphere_suite(
        sim_dir,
        output_file=output_file,
        refresh=refresh,
        summary_functions=summary_functions
    )

    if node_index not in summary:
        ax.text(0.5, 0.5, f'Node {node_index}\nnot found',
                transform=ax.transAxes, ha='center', va='center', fontsize=9)
        ax.set_title(f'Node {node_index}')
        return

    # Extract V_max and R_max time series
    vmax_raw = summary[node_index]['vmax']  # kpc/Myr
    rmax_raw = summary[node_index]['rmax']  # kpc

    # Convert V_max from kpc/Myr to km/s
    vmax_kms = vmax_raw * Const.KPC_MYR_TO_KMS

    # Apply Gaussian smoothing if requested
    if smoothing_sigma is not None and smoothing_sigma > 0:
        vmax_smooth = vmax_kms
        rmax_smooth = rmax_raw
    else:
        vmax_smooth = vmax_kms
        rmax_smooth = rmax_raw

    # Plot the tidal track
    ax.scatter(vmax_smooth, rmax_smooth)
    ax.set_xlabel(r'$V_{\rm max}$ [km/s]')
    ax.set_ylabel(r'$R_{\rm max}$ [kpc]')


if __name__ == "__main__":
    mpl_params_update()

    path_galacticus = 'data/galacticus/galacticus-v2/mh1e13_z05_rmax_vmax.hdf5'
    nsphere_dir = './data/NSphere/NSphere-galacticus-hr-cdm-test'
    nsphere_cachedir = 'out/nsphere_summary'

    node_ids = [3375, 3660, 3731, 8876, 2328, 6267, 10101, 3960, 3488]

    os.makedirs('out/plots', exist_ok=True)

    gout = h5py.File(path_galacticus, 'r')

    # --- Figure 1: Galacticus tidal tracks only ---
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))

    for ax, node_id in zip(axs.flat, node_ids):
        plot_tidal_tracks_galacticus(ax, gout, node_id)
        ax.set_title(f'Tidal Track - Node {node_id} (Galacticus)')

    fig.suptitle('Subhalo Tidal Tracks (Galacticus)', y=1.001)
    fig.tight_layout()
    fig.savefig('out/plots/tidal_tracks_galacticus.png')
    print('Saved: out/plots/tidal_tracks_galacticus')

    # --- Figure 2: NSphere tidal tracks with smoothing ---
    fig2, axs2 = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))

    for ax, node_id in zip(axs2.flat, node_ids):
        plot_tidal_tracks_t1des(
            ax,
            nsphere_dir,
            node_id,
            smoothing_sigma=10.0,
            cachedir=nsphere_cachedir
        )
        ax.set_title(f'Tidal Track — Node {node_id} (NSphere)')

    fig2.suptitle('Subhalo Tidal Tracks (NSphere, smoothed)', y=1.001)
    fig2.tight_layout()
    fig2.savefig('out/plots/tidal_tracks_nsphere.png')
    print('Saved: out/plots/tidal_tracks_nsphere')

    # --- Figure 3: Combined comparison ---
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))

    axs[0,0].legend(['Galacticus', 'NSphere'], loc='best', fontsize=9)

    for ax, node_id in zip(axs.flat, node_ids):
        plot_tidal_tracks_galacticus(ax, gout, node_id)
        plot_tidal_tracks_t1des(
            ax,
            nsphere_dir,
            node_id,
            smoothing_sigma=10.0,
            cachedir=nsphere_cachedir
        )
        
        ax.set_title(f'Tidal Track - Node {node_id}')

    fig.suptitle('Subhalo Tidal Tracks Comparison', y=1.001)
    fig.tight_layout()
    fig.savefig('out/plots/tidal_tracks_comparison.png')
    print('Saved: out/plots/tidal_tracks_comparison')

    gout.close()
    plt.show()
