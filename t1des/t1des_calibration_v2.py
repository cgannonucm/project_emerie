#!/usr/bin/env python3
import numpy as np
from sidmcommon.t1des_pipeline import run_pipeline
from t1des_prepare import prepare_output_directories

import subscript.scripts.nfilters as nf
from subscript.defaults import ParamKeys


if __name__ == "__main__":
    nfilter = nf.r2d(None, 2e-2, 5e-2, [1,0, 0])
    nfilter = nf.logical_and(nfilter, 
                             nf.subhalos_valid(None, 1e9, 1e10, ParamKeys.mass_basic))
    
    alpha_array = np.linspace(0.3, 0.7, 9)

    name = "nsphere_galacticus_lr_tuning_"
    description = f"Tuning learning rate for NSphere runs matching Galacticus subhalo mass history. Alpha values: {alpha_array}"
    galacticus_hdf5 = '../data/galacticus/galacticus-v2/mh1e13_z05_rmax_vmax.hdf5'

    dir_out         = prepare_output_directories(name=name, dir_out='parameters/tuning', description=description, galacticus_hdf5=galacticus_hdf5)    

    for a in alpha_array:
        _name = name + f'_alpha_{a:.3f}'
        run_pipeline(galacticus_hdf5=galacticus_hdf5, 
                                dir_out=dir_out, 
                                nparticles_final=int(1e3),
                                tree_index=0,
                                nfilter=nfilter,
                                name=_name,
                                timesteps_per_dt=2000,
                                alpha=a)
    print(f'Parameter files for tuning written to {dir_out}.')

        





