#!/usr/bin/env python3
"""
Apocenter detection from Galacticus subhalo time-series data.

Interpolates the orbital radius vs time for each subhalo and identifies
local maxima (apocenters).
"""

import os
import h5py
import numpy as np
import astropy.units as u
import astropy.constants as const
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema
from colossus.cosmology import cosmology
from colossus.halo import profile_nfw

from subscript.tabulatehdf5 import tabulate_trees
from subscript.scripts.nodes import nodedata
from subscript.scripts import nfilters as nf
from subscript.defaults import ParamKeys

try:
    from analysis_scripts.subhalo_timeseries import subhalo_timeseries_summary
except ModuleNotFoundError:
    from subhalo_timeseries import subhalo_timeseries_summary


def _get_smoothing_sigma(galacticus_hdf5, halo_id, ts_data, sigma_frac=0.1):
    """
    Compute a physically motivated Gaussian smoothing sigma for a subhalo.

    Uses the dynamical time at the subhalo's final orbital radius within
    the host NFW profile:
        t_dyn = sqrt(a^3 / (G * M_enc))
    where a is the final radius and M_enc is the host mass enclosed.

    The smoothing sigma (in units of the interpolated time-grid spacing)
    is set to sigma_frac * t_dyn / dt_fine.

    Parameters
    ----------
    galacticus_hdf5 : h5py.File
        Open Galacticus HDF5 file.
    halo_id : int
        Node index of the subhalo.
    ts_data : dict
        Time-series data from subhalo_timeseries_summary.
    sigma_frac : float
        Fraction of the dynamical time to use as the smoothing width.
        Default: 0.1.

    Returns
    -------
    float
        Smoothing sigma in units of the interpolated time-grid spacing.
    """
    cosmo = cosmology.setCosmology('planck18')

    # Host halo properties
    tree = tabulate_trees(galacticus_hdf5)[0]
    host_mass = nodedata(tree, key=ParamKeys.mass_basic, nfilter=nf.hosthalos)  # M_sun
    host_conc = nodedata(tree, key='concentration', nfilter=nf.hosthalos)

    # Final redshift from subhalo time-series (index 0 = present day)
    z_final = ts_data[halo_id]['zsnaps'][0]

    # Host NFW profile
    host_profile = profile_nfw.NFWProfile(
        M=host_mass, c=host_conc, z=z_final, mdef='vir'
    )

    # Final orbital radius [Mpc -> kpc/h for colossus]
    data = ts_data[halo_id]['data']
    r_final_mpc = np.sqrt(
        data['satellitePositionX'][0]**2
        + data['satellitePositionY'][0]**2
        + data['satellitePositionZ'][0]**2
    )
    r_final_kpch = r_final_mpc * 1e3 * cosmo.Hz(0) / 100  # Mpc -> kpc/h

    # Enclosed mass [M_sun/h]
    M_enc_msun_h = host_profile.enclosedMass(r_final_kpch)
    M_enc_msun = M_enc_msun_h / (cosmo.Hz(0) / 100)

    # Dynamical time: t_dyn = sqrt(a^3 / (G * M_enc))
    r_final = (r_final_mpc * u.Mpc).to(u.m)
    M_enc = (M_enc_msun * u.Msun).to(u.kg)
    t_dyn = np.sqrt(r_final**3 / (const.G * M_enc)).to(u.Gyr).value  # Gyr

    # Convert to interpolated grid units
    zsnaps = ts_data[halo_id]['zsnaps']
    tinf = cosmo.age(zsnaps) - cosmo.age(np.max(zsnaps))
    n_fine = len(tinf) * 10
    dt_fine = (tinf.max() - tinf.min()) / (n_fine - 1)

    sigma = sigma_frac * t_dyn / dt_fine
    return sigma


def get_apocenter(galacticus_hdf5, halo_ids, sigma_frac=0.1):
    """
    Find apocenter passages for each subhalo from Galacticus time-series.

    Computes the 3D orbital radius from satellite position components,
    interpolates to 10x resolution, applies a Gaussian filter with a
    physically motivated width (fraction of the dynamical time at the
    final orbital radius), and identifies local maxima (apocenters).

    Parameters
    ----------
    galacticus_hdf5 : h5py.File
        Open Galacticus HDF5 file.
    halo_ids : list of int
        Node indices of the subhalos.
    sigma_frac : float
        Fraction of the dynamical time used as the Gaussian smoothing
        width. Default: 0.1.

    Returns
    -------
    dict
        Mapping {halo_id: (times, radii)} where times [Gyr] and radii [Mpc]
        are arrays of apocenter values. Empty arrays if no maxima found.
    """
    cosmo = cosmology.setCosmology('planck18')
    ts_data = subhalo_timeseries_summary(galacticus_hdf5, tree_index=0)

    result = {}
    for halo_id in halo_ids:
        if halo_id not in ts_data:
            result[halo_id] = (np.array([]), np.array([]))
            continue

        data = ts_data[halo_id]['data']
        zsnaps = ts_data[halo_id]['zsnaps']

        # Cosmic time from redshift [Gyr]
        tinf = cosmo.age(zsnaps) - cosmo.age(np.max(zsnaps))  # Time since infall

        # 3D orbital radius [Mpc]
        r = np.sqrt(
            data['satellitePositionX']**2
            + data['satellitePositionY']**2
            + data['satellitePositionZ']**2
        )

        # Interpolate to 10x resolution before smoothing
        t_fine = np.linspace(tinf.min(), tinf.max(), len(tinf) * 10)
        r_fine = interp1d(tinf, r, kind='cubic')(t_fine)

        # Compute physically motivated smoothing sigma
        sigma = _get_smoothing_sigma(
            galacticus_hdf5, halo_id, ts_data, sigma_frac=sigma_frac
        )

        # Smooth interpolated radius before peak detection
        r_smooth = gaussian_filter1d(r_fine, sigma=sigma)

        # Find local maxima on smoothed data
        max_idx = argrelextrema(r_smooth, np.greater)[0]

        result[halo_id] = (t_fine[max_idx], r_smooth[max_idx])

    return result


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_standardize import mpl_params_update

    mpl_params_update()
    cosmo = cosmology.setCosmology('planck18')

    gal_file = 'data/galacticus/galacticus-v2/mh1e13_z05_rmax_vmax.hdf5'
    test_ids = [3375, 3660, 3731, 8876, 2328, 6267, 10101, 3960, 3488]

    sigma_frac = 0.7

    with h5py.File(gal_file, 'r') as gout:
        apocenters = get_apocenter(gout, test_ids, sigma_frac=sigma_frac)
        ts_data = subhalo_timeseries_summary(gout, tree_index=0)

        # Pre-compute per-halo sigma for plotting
        halo_sigmas = {}
        for hid in test_ids:
            halo_sigmas[hid] = _get_smoothing_sigma(
                gout, hid, ts_data, sigma_frac=sigma_frac
            )

    nrows, ncols = 3, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             sharey=True)

    for ax, halo_id in zip(axes.flat, test_ids):
        data = ts_data[halo_id]['data']
        zsnaps = ts_data[halo_id]['zsnaps']
        t = cosmo.age(zsnaps) - cosmo.age(np.max(zsnaps))  # Time since infall [Gyr]
        r = np.sqrt(
            data['satellitePositionX']**2
            + data['satellitePositionY']**2
            + data['satellitePositionZ']**2
        )
        t_fine = np.linspace(t.min(), t.max(), len(t) * 10)
        r_fine = interp1d(t, r, kind='cubic')(t_fine)
        r_smooth = gaussian_filter1d(r_fine, sigma=halo_sigmas[halo_id])

        ax.plot(t, r * 1e3, '-', alpha=0.4, label='Raw')
        ax.plot(t_fine, r_smooth * 1e3, '-', label='Smoothed')

        t_apo, r_apo = apocenters[halo_id]
        if len(t_apo) > 0:
            ax.scatter(t_apo, r_apo * 1e3, color='red', zorder=3,
                       label='Apocenters')

        ax.set_xlabel('Time Since Infall [Gyr]')
        ax.set_title(f'Node {halo_id}')
        ax.legend(fontsize='small')

    for ax in axes[:, 0]:
        ax.set_ylabel('Orbital radius [kpc]')

    fig.suptitle('Apocenter Detection')
    fig.tight_layout()

    print('Saving plot to out/plots/debug/apocenter_test.pdf')
    fig.savefig('out/plots/debug/apocenter_test.pdf', bbox_inches='tight')
