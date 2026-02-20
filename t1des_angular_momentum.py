#!/usr/bin/env python3
"""
Angular momentum magnitude over time for Galacticus subhalos.

Computes L = |r x v| where r = (x, y, z) and v = (vx, vy, vz)
are the satellite position and velocity vectors from Galacticus.
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from colossus.cosmology import cosmology
from subscript.defaults import ParamKeys

from analysis_scripts.mpl_standardize import mpl_params_update
from analysis_scripts.subhalo_timeseries import subhalo_timeseries_summary


def plot_angular_momentum_galacticus(ax, galacticus_hdf5, halo_id, cosmo):
    """
    Plot the magnitude of the angular momentum vector over time for a subhalo.

    Angular momentum is computed as L = |r x v| where r and v are the
    satellite position and velocity vectors from Galacticus.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on.
    galacticus_hdf5 : h5py.File
        Open Galacticus HDF5 file.
    halo_id : int
        Node index of the subhalo to plot.
    cosmo : colossus.cosmology.Cosmology
        Cosmology object for converting redshift to cosmic time [Gyr].
    """
    ts_data = subhalo_timeseries_summary(galacticus_hdf5, tree_index=0)

    if halo_id not in ts_data:
        ax.text(0.5, 0.5, f'Node {halo_id}\nnot found',
                transform=ax.transAxes, ha='center', va='center', fontsize=9)
        ax.set_title(f'Halo {halo_id}')
        return

    ts = ts_data[halo_id]
    zsnaps = ts['zsnaps']
    data = ts['data']

    # Position and velocity vectors [Mpc, km/s]
    pos = np.column_stack([
        data['satellitePositionX'],
        data['satellitePositionY'],
        data['satellitePositionZ'],
    ])
    vel = np.column_stack([
        data['satelliteVelocityX'],
        data['satelliteVelocityY'],
        data['satelliteVelocityZ'],
    ])
    
    mass = data[ParamKeys.mass_bound]

    # L = r x v, then take magnitude
    L_vec = np.cross(pos, vel)
    L_mag = np.linalg.norm(L_vec, axis=1)

    # Time since infall [Gyr]
    t_infall = cosmo.age(zsnaps[-1])
    t_since_infall = cosmo.age(zsnaps) - t_infall

    ax.plot(t_since_infall, L_mag)


if __name__ == '__main__':
    mpl_params_update()

    cosmo = cosmology.setCosmology('planck18')

    gal_file = 'data/galacticus/galacticus-v2/mh1e13_z05_rmax_vmax.hdf5'
    halo_ids = [3375, 3660, 3731, 8876, 2328, 6267, 10101, 3960, 3488]

    fig, axes = plt.subplots(3, 3, figsize=(12, 10))

    with h5py.File(gal_file, 'r') as gout:
        for ax, halo_id in zip(axes.flatten(), halo_ids):
            plot_angular_momentum_galacticus(ax, gout, halo_id, cosmo)

            ax.set_xlabel('Time since infall [Gyr]')
            ax.set_ylabel(r'$|\mathbf{L}|$ [M_\odot Mpc km/s]')
            ax.set_title(f'Halo {halo_id}')
            ax.set_yscale('log')

    fig.suptitle('Subhalo Angular Momentum Histories', y=1.01)
    fig.tight_layout()

    os.makedirs('out/plots', exist_ok=True)
    fig.savefig('out/plots/angular_momentum_sample.png', bbox_inches='tight')
    plt.show()
