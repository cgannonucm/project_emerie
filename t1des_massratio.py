#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import h5py
import subscript.scripts.nfilters as nf
from subscript.scripts.nodes import nodedata
import h5py
from subscript.tabulatehdf5 import  tabulate_trees
from subscript.defaults import ParamKeys
from mpl_standardize import mpl_params_update

from sidmcommon.nsphere_suite import summarize_nsphere_suite
from sidmcommon.galacticus import get_nodedata_byids

def plot_massratio_histogram(ax, simdir, path_galacticus_out, 
                             output_file=None, label=None, 
                             xlim=(1e-2, 1e1), plot_kwargs=None):
    plot_kwargs = plot_kwargs if plot_kwargs is not None else {}

    if output_file is None:
        output_file = 'out/nsphere_summary/summary_temp'

    # Summarize nsphere output, disable caching to ensure we get the latest data
    summary = summarize_nsphere_suite(simdir, output_file, refresh=False)

    file_galacticus = h5py.File(path_galacticus_out, 'r')

    tree_galacticus = tabulate_trees(file_galacticus)[0]

    nfilter = nf.r2d(None, 2e-2, 5e-2, [1,0, 0])
    nfilter = nf.logical_and(nfilter, 
                                nf.subhalos_valid(None, 1e9, 1e10, ParamKeys.mass_basic))                            


    mass_bound_gal = nodedata(tree_galacticus, [ParamKeys.mass_bound], nfilter=nfilter)

    mass_bound_gal = get_nodedata_byids(list(summary.keys()), tree_galacticus, ParamKeys.mass_bound)
    
    massratio = []
    massratio_lr = []
    
    halo_ids = []

    for halo_id, val in summary.items():
        massratio.append(val['massbound'][-1] / mass_bound_gal[halo_id])
        massratio_lr.append(summary[halo_id]['massbound'][-1] / mass_bound_gal[halo_id])        
        halo_ids.append(halo_id)

    massratio = np.array(massratio)
    massratio_lr = np.array(massratio_lr)


    ax.hist(massratio, bins=np.geomspace(xlim[0], xlim[1], 20), alpha=0.5, label=label, **plot_kwargs)
    ax.set_xlabel('Final Bound Mass Ratio (NSphere / Galacticus)')
    ax.set_ylabel('Count')


if __name__ == "__main__":
    mpl_params_update()

    path_galacticus = 'data/galacticus/mh1e13_z05_test.hdf5'
    simdir = 'data/nsphere/mh1e13_z05_test'
    path_out = 'out/plots/massratio_histogram.png'

    fig, ax = plt.subplots(figsize=(9,6))
    plot_massratio_histogram(ax, simdir, path_galacticus, label='NSphere (Not Finalized)')

    plt.xscale('log')
    plt.legend()
    plt.tight_layout()

    fig.savefig(path_out)
