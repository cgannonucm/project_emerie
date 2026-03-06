#!/usr/bin/env python3
import numpy as np
import os
from sidmcommon.t1des_pipeline import run_pipeline
from datetime import datetime

import subscript.scripts.nfilters as nf
from subscript.defaults import ParamKeys
import shutil

def prepare_output_directories(name, dir_out, description, galacticus_hdf5):
    time = datetime.now().strftime('%Y%m%d_%H%M%S')
    dir_name = os.path.join(dir_out, f'{name}_{time}')  
    os.makedirs(dir_name, exist_ok=False)

    with open(os.path.join(dir_name, 'info.txt'), 'w') as f:
        f.write(description)

    shutil.copy(galacticus_hdf5, os.path.join(dir_name, 'galacticus.hdf5'))
    
    return dir_name