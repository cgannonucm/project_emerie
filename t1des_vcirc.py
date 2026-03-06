#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

from analysis_scripts.vmax_rmax_nsphere import get_nsphere_vcirc_profile
from sidmcommon.nsphere import read_particles_all_sim, read_auxillary_sim
from sidmcommon.const import Const


def plot_vcirc_profile(ax, sim_path, label=None, snapshot_indices=None,
                       plot_kwargs=None, cmap='viridis'):
    '''Plot circular velocity profile (R vs V_circ) for a single NSphere halo.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on.
    sim_path : str
        Path to NSphere simulation directory.
    label : str, optional
        Label for the plot title.
    snapshot_indices : list of int, optional
        Snapshot indices to plot. Default: [0].
    plot_kwargs : dict, optional
        Additional keyword arguments passed to ax.plot().
    cmap : str
        Colormap name for distinguishing snapshots.
    '''
    plot_kwargs = plot_kwargs if plot_kwargs is not None else {}
    if snapshot_indices is None:
        snapshot_indices = [0]

    particles = read_particles_all_sim(sim_path)
    aux = read_auxillary_sim(sim_path)
    time_myr = aux['timestep']

    colors = plt.colormaps.get_cmap(cmap).resampled(len(snapshot_indices))

    for i, idx in enumerate(snapshot_indices):
        snap = particles[idx]
        vcirc_kms    , r     = get_nsphere_vcirc_profile(snap) 
        vcirc_kms_all, r_all = get_nsphere_vcirc_profile(snap, bound_only=False) 

        vcirc_kms *= Const.KPC_MYR_TO_KMS
        vcirc_kms_all *= Const.KPC_MYR_TO_KMS

        #sort_idx = np.argsort(radii)
        t_label = f't = {time_myr[idx]:.0f} Myr'
        ax.plot(r, vcirc_kms,
                color=colors(i), rasterized=True, label=t_label,
                **plot_kwargs)

        #ax.plot(r_all, vcirc_kms_all,
        #        color='black', rasterized=True, label=t_label,
        #        linestyle='--',
        #        **plot_kwargs)

    if label is not None:
        ax.set_title(label)


if __name__ == '__main__':
    from analysis_scripts.mpl_standardize import mpl_params_update

    mpl_params_update()

    simdir = './data/NSphere/NSphere-galacticus-hr-cdm-test'
    halo_ids = [3375, 3660, 3731, 8876, 2328, 6267, 10101, 3960, 3488]

    fig, axes = plt.subplots(3, 3, figsize=(14, 12))

    snapshot_indices = np.arange(0, 10)  # Plot first 4 snapshots for each halo

    for ax, halo_id in zip(axes.flat, halo_ids):
        sim_path = f'{simdir}/nsphere_params_{halo_id}'
        plot_vcirc_profile(ax, sim_path, label=f'Halo {halo_id}',
                           snapshot_indices=snapshot_indices)
        ax.legend(fontsize=7, loc='lower right')
        ax.loglog()

    for ax in axes[-1, :]:
        ax.set_xlabel('R [kpc]')
    for ax in axes[:, 0]:
        ax.set_ylabel(r'$V_{\rm circ}$ [km/s]')

    fig.suptitle('Circular Velocity Profiles (NSphere CDM)', fontsize=18)
    fig.tight_layout()
    fig.savefig('out/plots/t1des_vcirc.png', dpi=150)
    print('Saved: out/plots/t1des_vcirc.png')
