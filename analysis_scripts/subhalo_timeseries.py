#!/usr/bin/env python3
import os
import pickle
import hashlib
from pathlib import Path

import h5py
import numpy as np

from subscript.tabulatehdf5 import tabulate_trees
from subscript.scripts.nodes import nodedata
from subscript.scripts import nfilters as nf
from subscript.tracking import track_subhalos, track_subhalo


def subhalo_timeseries_summary(galacticus_hdf5: h5py.File, tree_index: int, refresh=False) -> dict:
    """
    Extract per-subhalo time-series data for all subhalos in a Galacticus tree.

    Retrieves all subhalo node IDs at the last snapshot of the given tree, runs
    track_subhalos across all snapshots, then filters each subhalo's time-series
    via track_subhalo (removing isolated/unbound snapshots). Results are cached
    to disk using pickle.

    Parameters
    ----------
    galacticus_hdf5 : h5py.File
        Open HDF5 file object for the Galacticus simulation output.
    tree_index : int
        Index of the merger tree to process.
    refresh : bool, optional
        If True, bypass cache and recompute results. Default is False.

    Returns
    -------
    dict
        Dictionary mapping node IDs (int) to dicts with keys:
        - 'data': dict of {param_key: time_series_array}
        - 'zsnaps': corresponding redshift array
    """
    file_path = galacticus_hdf5.filename

    # Build cache filename: {stem}-{hash[:16]}-tree{tree_index}.pkl
    file_stem = Path(file_path).stem
    file_hash = hashlib.sha256(open(file_path, 'rb').read()).hexdigest()[:16]
    cache_path_name = f"{file_stem}-{file_hash}-tree{tree_index}.pkl"
    cache_path = os.path.join(Path(file_path).parent, cache_path_name)

    if os.path.exists(cache_path) and not refresh:
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    # Get all subhalo node IDs at the last output of this tree
    trees = tabulate_trees(galacticus_hdf5)
    node_ids = nodedata(trees[tree_index], key='nodeIndex', nfilter=nf.subhalos)

    # Track all subhalos across all snapshots
    dat_subhalos, zsnap_subhalos = track_subhalos(
        galacticus_hdf5,
        nodeIndices=node_ids,
        treeIndex=tree_index
    )

    # Determine which param_keys were tracked (exclude the internal 'zsnap' key)
    first_id = node_ids[0]
    param_keys = [k for k in dat_subhalos[first_id].keys() if k != 'zsnap']

    # Build result dict: filter each subhalo's time-series to satellite-only snapshots
    result = {}
    for node_id in node_ids:
        filtered_data, filtered_zsnaps = track_subhalo(
            dat_subhalos, zsnap_subhalos, node_id, param_keys
        )
        result[node_id] = {'data': filtered_data, 'zsnaps': filtered_zsnaps}

    with open(cache_path, 'wb') as f:
        pickle.dump(result, f)

    return result

if __name__ == "__main__":
    path_galacticus_test = '../data/galacticus/mh1e13_z05_test.hdf5'
    with h5py.File(path_galacticus_test, 'r') as f:
        subhalo_timeseries_summary(f, tree_index=0)
