#!/usr/bin/env python3
from analysis_scripts.mpl_standardize import mpl_params_update
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt

from subscript.defaults import ParamKeys

from analysis_scripts.subhalo_timeseries import subhalo_timeseries_summary


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


if __name__ == "__main__":
    mpl_params_update()

    path_galacticus = 'data/galacticus/galacticus-v2/mh1e13_z05_rmax_vmax.hdf5'

    node_ids = [3375, 3660, 3731, 8876, 2328, 6267, 10101, 3960, 3488]

    gout = h5py.File(path_galacticus, 'r')

    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))

    for ax, node_id in zip(axs.flat, node_ids):
        plot_tidal_tracks_galacticus(ax, gout, node_id)
        ax.set_title(f'Tidal Track — Node {node_id}')

    fig.tight_layout()

    fig.savefig(f'out/plots/tidal_tracks_galacticus.png')

    gout.close()
