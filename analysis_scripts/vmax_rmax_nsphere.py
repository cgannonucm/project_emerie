import numpy as np
from sidmcommon.const import Const

def _vc2_profile(snap):
    """Squared circular velocity profile for a single snapshot."""
    radii = snap['R']         # kpc
    m_enc = snap['mass']      # M☉, cumulative (rank-ordered by R)
    return Const.G_KPC_MYR * m_enc / radii  # (kpc/Myr)²


def get_nsphere_vmax(data):
    """
    Maximum circular velocity at each output snapshot.

    Parameters
    ----------
    data : ndarray, shape (nsteps, nparticles)
        Particle data returned by read_particles_all_sim().
        Fields: rank (i32), R (f32, kpc), Vrad (f32, kpc/Myr),
                L (f32, kpc²/Myr), mass (f64, M☉ cumulative).

    Returns
    -------
    vmax : ndarray, shape (nsteps,)
        V_max in kpc/Myr at each snapshot.
    """
    vmax = np.zeros(len(data))
    for isnap, snap in enumerate(data):
        vmax[isnap] = np.sqrt(np.max(_vc2_profile(snap)))
    return vmax


def get_nsphere_rmax(data):
    """
    Radius of maximum circular velocity at each output snapshot.

    Parameters
    ----------
    data : ndarray, shape (nsteps, nparticles)
        Particle data returned by read_particles_all_sim().

    Returns
    -------
    rmax : ndarray, shape (nsteps,)
        R_max in kpc at each snapshot.
    """
    rmax = np.zeros(len(data))
    for isnap, snap in enumerate(data):
        vc2 = _vc2_profile(snap)
        rmax[isnap] = snap['R'][np.argmax(vc2)]
    return rmax


if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt
    from .mpl_standardize import mpl_params_update
    from sidmcommon.nsphere import read_particles_all_sim, read_auxillary_sim

    mpl_params_update()

    halo_id = 10101
    sim_path = f'./data/NSphere/NSphere-galacticus-hr-cdm-test/nsphere_params_{halo_id}'

    print(f'Loading particle data for halo {halo_id}...')
    particles = read_particles_all_sim(sim_path)
    aux = read_auxillary_sim(sim_path)
    time = aux['timestep']  # Myr

    print('Computing V_max and R_max...')
    vmax = get_nsphere_vmax(particles)
    rmax = get_nsphere_rmax(particles)
    del particles

    # kpc/Myr → km/s conversion
    vmax_kms = vmax * Const.KPC_MYR_TO_KMS

    os.makedirs('out/plots', exist_ok=True)

    # --- Figure 1: V_max and R_max vs time ---
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    axes[0].plot(time, vmax_kms, color='C0')
    axes[0].set_ylabel(r'$V_{\rm max}$ [km/s]')
    axes[0].set_title(f'Tidal track — Halo {halo_id}')

    axes[1].plot(time, rmax, color='C1')
    axes[1].set_ylabel(r'$R_{\rm max}$ [kpc]')
    axes[1].set_xlabel('Time [Myr]')

    fig.tight_layout()
    fig.savefig('out/plots/vmax_rmax_vs_time.pdf')
    print('Saved: out/plots/vmax_rmax_vs_time.pdf')

    # --- Figure 2: Tidal track (R_max vs V_max) ---
    fig2, ax2 = plt.subplots(figsize=(6, 5))

    sc = ax2.scatter(vmax_kms, rmax, c=time, cmap='viridis', s=10)
    cbar = fig2.colorbar(sc, ax=ax2)
    cbar.set_label('Time [Myr]')

    ax2.set_xlabel(r'$V_{\rm max}$ [km/s]')
    ax2.set_ylabel(r'$R_{\rm max}$ [kpc]')
    ax2.set_title(f'Tidal track — Halo {halo_id}')

    fig2.tight_layout()
    fig2.savefig('out/plots/debug/vmax_rmax_tidal_track.pdf')
    print('Saved: out/plots/debug/vmax_rmax_tidal_track.pdf')

    plt.show()
