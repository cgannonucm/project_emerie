#!/usr/bin/env python3
from analysis_scripts.mpl_standardize import mpl_params_update
import sys
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from colossus.cosmology import cosmology
from subscript.defaults import ParamKeys

from analysis_scripts.subhalo_timeseries import subhalo_timeseries_summary
from analysis_scripts.apocenter import get_apocenter
from analysis_scripts.vmax_rmax_nsphere import get_nsphere_vmax, get_nsphere_rmax
from sidmcommon.nsphere_suite import summarize_nsphere_suite
from sidmcommon.const import Const


def plot_tidal_tracks_galacticus(ax, galacticus_hdf5: h5py.File, node_index: int, tree_index: int = 0):
    """
    Plot the tidal track (R_max vs V_max) for a single subhalo over time.

    Extracts the V_max and R_max time series for the specified subhalo and
    plots the trajectory in the (V_max, R_max) plane on the provided axes,
    colored by time since infall.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to plot on.
    galacticus_hdf5 : h5py.File
        Open Galacticus HDF5 file object.
    node_index : int
        Node ID of the subhalo to plot the tidal track for.
    tree_index : int
        Index of the merger tree to process. Default: 0.
    """
    cosmo = cosmology.setCosmology('planck18')
    ts_data = subhalo_timeseries_summary(galacticus_hdf5, tree_index=tree_index)

    ts = ts_data[node_index]
    vmax = ts['data'][ParamKeys.dark_matter_profile_dmo_velocity_max]
    rmax = ts['data'][ParamKeys.dark_matter_profile_dmo_radius_velocity_max] * 1e3  # Convert to kpc
    zsnaps = ts['zsnaps']

    # Time since infall [Gyr]
    time_gyr = cosmo.age(zsnaps) - cosmo.age(np.max(zsnaps))

    # Build line segments colored by time since infall
    points = np.column_stack([rmax, vmax])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    norm = Normalize(vmin=time_gyr.min(), vmax=time_gyr.max())
    lc = LineCollection(segments, cmap='copper', norm=norm, linewidths=4, zorder=2)
    lc.set_array(time_gyr[:-1])
    ax.add_collection(lc)
    ax.autoscale_view()
    ax.set_ylabel(r'$V_{\rm max}$ [km/s]')
    ax.set_xlabel(r'$R_{\rm max}$ [kpc]')


def plot_tidal_tracks_t1des(ax, sim_dir, node_index: int, galacticus_hdf5=None,
                              cachedir=None, refresh=False):
    """
    Plot the tidal track (R_max vs V_max) at apocenter passages for T1DES.

    Extracts V_max and R_max time series from NSphere simulations, then
    selects only the snapshots closest to each apocenter time (from
    Galacticus orbital data) and plots those points.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to plot on.
    sim_dir : str
        Path to the NSphere simulation suite directory.
    node_index : int
        Halo ID matching the nsphere_params_<node_index> subdirectory.
    galacticus_hdf5 : h5py.File
        Open Galacticus HDF5 file, required for apocenter detection.
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

    # Extract V_max, R_max, and time since infall
    vmax_raw = summary[node_index]['vmax']  # kpc/Myr
    rmax = summary[node_index]['rmax']  # kpc
    time_myr = summary[node_index]['time']  # Myr

    # Convert V_max from kpc/Myr to km/s, time from Myr to Gyr
    vmax_kms = vmax_raw * Const.KPC_MYR_TO_KMS
    time_gyr = time_myr / 1e3

    # Get apocenter times and select closest NSphere snapshots
    apo_data = get_apocenter(galacticus_hdf5, [node_index], sigma_frac=1)
    t_apo, _ = apo_data[node_index]

    if len(t_apo) > 0:
        # t_apo is in Gyr (time since infall), match to time_gyr
        apo_idx = np.array([np.argmin(np.abs(time_gyr - ta)) for ta in t_apo])

        norm = Normalize(vmin=time_gyr.min(), vmax=time_gyr.max())
        sc = ax.scatter(rmax[apo_idx], vmax_kms[apo_idx],
                        c=time_gyr[apo_idx], cmap='copper', norm=norm, s=80,
                        edgecolors='k', linewidths=0.5, zorder=3)
        ax.plot(rmax[apo_idx], vmax_kms[apo_idx], 'k--', linewidth=0.8,
                alpha=0.5, zorder=2)
        ax.figure.colorbar(sc, ax=ax, label='Time since infall [Gyr]')

    ax.set_xlabel(r'$R_{\rm max}$ [kpc]')
    ax.set_ylabel(r'$V_{\rm max}$ [km/s]')


def plot_tidal_tracks_t1des_full(ax, sim_dir, node_index: int,
                                cachedir=None, refresh=False):
    """
    Plot the full tidal track (R_max vs V_max) for T1DES NSphere simulation.

    Unlike plot_tidal_tracks_t1des, this plots all snapshots (not just
    apocenters) as a gradient line colored by time since infall.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to plot on.
    sim_dir : str
        Path to the NSphere simulation suite directory.
    node_index : int
        Halo ID matching the nsphere_params_<node_index> subdirectory.
    cachedir : str
        Directory where cached HDF5 summary is written.
        Default: 'out/cache'
    refresh : bool
        If True, recompute the summary even if cache exists.
    """
    cachedir = cachedir if cachedir is not None else 'out/cache'

    sim_name = os.path.basename(os.path.normpath(sim_dir))
    output_file = os.path.join(cachedir, f'tidal_tracks-{sim_name}.hdf5')
    os.makedirs(cachedir, exist_ok=True)

    summary_functions = {
        'vmax': get_nsphere_vmax,
        'rmax': get_nsphere_rmax,
    }

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

    vmax_raw = summary[node_index]['vmax']  # kpc/Myr
    rmax = summary[node_index]['rmax']  # kpc
    time_myr = summary[node_index]['time']  # Myr

    vmax_kms = vmax_raw * Const.KPC_MYR_TO_KMS
    time_gyr = time_myr / 1e3

    norm = Normalize(vmin=time_gyr.min(), vmax=time_gyr.max())
    sc = ax.scatter(rmax, vmax_kms, c=time_gyr, cmap='copper', norm=norm,
                    s=10, edgecolors='k', linewidths=0.3, zorder=2)
    ax.figure.colorbar(sc, ax=ax, label='Time since infall [Gyr]')
    ax.set_xlabel(r'$R_{\rm max}$ [kpc]')
    ax.set_ylabel(r'$V_{\rm max}$ [km/s]')


if __name__ == "__main__":
    mpl_params_update()

    path_galacticus = 'data/galacticus/galacticus-v2/mh1e13_z05_rmax_vmax.hdf5'
    nsphere_dir = './data/NSphere/NSphere-galacticus-lr-cdm'
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
            galacticus_hdf5=gout,
            cachedir=nsphere_cachedir,
            refresh=False
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
            galacticus_hdf5=gout,
            cachedir=nsphere_cachedir
        )
        
        ax.set_title(f'Tidal Track - Node {node_id}')

    fig.suptitle('Subhalo Tidal Tracks Comparison', y=1.001)
    fig.tight_layout()
    fig.savefig('out/plots/tidal_tracks_comparison.png')
    print('Saved: out/plots/tidal_tracks_comparison')

    # --- Figure 4: Full tidal tracks comparison (all snapshots) ---
    fig4, axs4 = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))

    for ax, node_id in zip(axs4.flat, node_ids):
        plot_tidal_tracks_galacticus(ax, gout, node_id)
        plot_tidal_tracks_t1des_full(
            ax,
            nsphere_dir,
            node_id,
            cachedir=nsphere_cachedir
        )
        ax.set_title(f'Tidal Track - Node {node_id}')

    fig4.suptitle('Subhalo Tidal Tracks Comparison (Full)', y=1.001)
    fig4.tight_layout()
    fig4.savefig('out/plots/tidal_tracks_comparison2.png')
    print('Saved: out/plots/tidal_tracks_comparison2')

    gout.close()
    plt.show()
