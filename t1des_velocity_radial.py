#!/usr/bin/env python3
"""
Radial orbital velocity over time for Galacticus subhalos.

Computes v_r = (v . r_hat) where r_hat = (x, y, z) / r is the
unit radial vector from the host center.
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from colossus.cosmology import cosmology

from analysis_scripts.mpl_standardize import mpl_params_update
from analysis_scripts.subhalo_timeseries import subhalo_timeseries_summary


def plot_radial_velocity(ax, galacticus_hdf5, halo_id, cosmo, tree_index=0):
    """
    Plot the radial orbital velocity v_r(t) for a subhalo over time.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on.
    galacticus_hdf5 : h5py.File
        Open Galacticus HDF5 file.
    halo_id : int
        Node index of the subhalo.
    cosmo : colossus.cosmology.Cosmology
        Cosmology object for converting redshift to cosmic time.
    tree_index : int
        Merger tree index. Default: 0.
    """
    ts_data = subhalo_timeseries_summary(galacticus_hdf5, tree_index=tree_index)

    if halo_id not in ts_data:
        ax.text(0.5, 0.5, f'Node {halo_id}\nnot found',
                transform=ax.transAxes, ha='center', va='center', fontsize=9)
        ax.set_title(f'Halo {halo_id}')
        return

    ts = ts_data[halo_id]
    zsnaps = ts['zsnaps']
    data = ts['data']

    # Position [Mpc] and velocity [km/s]
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

    # Radial unit vector
    r = np.linalg.norm(pos, axis=1, keepdims=True)
    r_hat = pos / r

    # Radial velocity: v_r = v . r_hat
    v_r = np.sum(vel * r_hat, axis=1)

    # Time since infall [Gyr]
    t_infall = cosmo.age(zsnaps[-1])
    t_since_infall = cosmo.age(zsnaps) - t_infall

    ax.plot(t_since_infall, v_r)
    ax.axhline(0, color='k', ls='--', lw=0.5)
    ax.set_xlabel('Time since infall [Gyr]')
    ax.set_ylabel(r'$v_r$ [km/s]')

if __name__ == '__main__':
    mpl_params_update()

    cosmo = cosmology.setCosmology('planck18')

    gal_file = 'data/galacticus/galacticus-v2/mh1e13_z05_rmax_vmax.hdf5'
    halo_ids = [3375, 3660, 3731, 8876, 2328, 6267, 10101, 3960, 3488]

    fig, axes = plt.subplots(3, 3, figsize=(12, 10))

    with h5py.File(gal_file, 'r') as gout:
        for ax, halo_id in zip(axes.flatten(), halo_ids):
            plot_radial_velocity(ax, gout, halo_id, cosmo)
            ax.set_title(f'Halo {halo_id}')

    fig.suptitle('Subhalo Radial Velocity Histories', y=1.01)
    fig.tight_layout()

    os.makedirs('out/plots', exist_ok=True)
    fig.savefig('out/plots/t1des_radial.png', bbox_inches='tight')
    plt.show()
