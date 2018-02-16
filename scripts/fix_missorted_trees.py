import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpi4py import MPI
import h5py
from neuroh5.io import append_cell_attributes, bcast_cell_attributes
from neuroh5.h5py_io_utils import *
import time


comm = MPI.COMM_WORLD


soma_coords_path = '../datasets/Full_Scale_Control/dentate_Full_Scale_Control_coords_distances_20171109.h5'
trees_path = '../datasets/Full_Scale_Control/DGC_forest_syns_weights_20171121_compressed.h5'
orig_reindex_map_path = '../datasets/Full_Scale_Control/tree_reindex_20170615.dat'
interp_trees_coords_path = '../morphologies/Somata_interpolated_tree_coords_051117.h5'
output_coords_path = '../datasets/Full_Scale_Control/dentate_Full_Scale_Control_coords_20180112.h5'

trees_f = h5py.File(trees_path, 'r')
interp_trees_coords_f = h5py.File(interp_trees_coords_path, 'r')

df = pd.read_csv(orig_reindex_map_path, sep='\t', header=None)

orig_map_interp_coords_to_trees = df.values[:,0]

# a subset of 1e6 trees that passed interpolation
# same subset as in orig_map_interp_coords_to_trees
# valid_gids = interp_trees_coords_f['Interpolated Coordinates']['New Tree GIDs'][:]


def get_attr_dict(index, group, keys):
    current = {}
    for attr in keys:
        if index != group[attr]['Cell Index'][index]:
            result = np.where(group[attr]['Cell Index'][:] == index)[0]
            if not any(result):
                raise ValueError('get_attr_dict: Cell Index not found: %i' % index)
            else:
                index = result[0]
        pointer = group[attr]['Attribute Pointer'][index]
        current[attr] = group[attr]['Attribute Value'][pointer]
    return current


def get_adhoc_attr_dict(index, group, old_keys, new_keys):
    current = {}
    for i, attr in enumerate(old_keys):
        current[new_keys[i]] = group[attr][index]
    return current


def match_attrs(test, target, keys):
    if all([test[attr] == target[attr] for attr in keys]):
        return True
    else:
        return False


tree_coords = {}
interp_tree_coords = {}
get_tree_coords = lambda index: get_attr_dict(index, trees_f['Populations']['GC']['Trees'],
                               ['X Coordinate', 'Y Coordinate', 'Z Coordinate'])
get_interp_tree_coords = lambda index: get_adhoc_attr_dict(index, interp_trees_coords_f['Interpolated Coordinates'],
                                            ['U', 'X', 'Y', 'Z'],
                                            ['U Coordinate', 'X Coordinate', 'Y Coordinate', 'Z Coordinate'])

interp_trees_u_coords = \
    np.array(interp_trees_coords_f['Interpolated Coordinates']['U'])[orig_map_interp_coords_to_trees]
new_map_trees_to_sorted_trees = np.argsort(interp_trees_u_coords)

new_map_interp_trees_to_sorted_soma_coords = orig_map_interp_coords_to_trees[new_map_trees_to_sorted_trees]

trees_x_pointers = np.array(
    trees_f['Populations']['GC']['Trees']['X Coordinate']['Attribute Pointer'])[new_map_trees_to_sorted_trees]
trees_x = np.array(
    trees_f['Populations']['GC']['Trees']['X Coordinate']['Attribute Value'])[trees_x_pointers]
somas_x = np.array(interp_trees_coords_f['Interpolated Coordinates']['X'])[new_map_interp_trees_to_sorted_soma_coords]
print 'Two new maps are consistent with each other: %s' % str(all(trees_x == somas_x))

"""
start_time = time.time()

target_population = 'GC'

reindex_map_dict = {}
target_namespace = 'Tree Reindex Map'
for new_gid, old_gid in enumerate(new_map_trees_to_sorted_trees):
    reindex_map_dict[old_gid] = {'New Cell Index': np.array([new_gid], dtype='uint32')}
print 'Built Tree Reindex Map; elapsed time: %.1f s' % (time.time() - start_time)
trees_f.close()
start_time = time.time()
append_cell_attributes(comm, output_coords_path, target_population, reindex_map_dict, namespace=target_namespace,
                       io_size=1)
print 'Appended Tree Reindex Map; elapsed time: %.1f s' % (time.time() - start_time)

start_time = time.time()
soma_coords_dict = {}
target_namespace = 'Coordinates'
source = interp_trees_coords_f['Interpolated Coordinates']
for relative_gid, old_index in enumerate(new_map_interp_trees_to_sorted_soma_coords):
    soma_coords_dict[relative_gid] = {'L Coordinate': np.array([source['L'][old_index]], dtype='float32'),
                                      'U Coordinate': np.array([source['U'][old_index]], dtype='float32'),
                                      'V Coordinate': np.array([source['V'][old_index]], dtype='float32'),
                                      'X Coordinate': np.array([source['X'][old_index]], dtype='float32'),
                                      'Y Coordinate': np.array([source['Y'][old_index]], dtype='float32'),
                                      'Z Coordinate': np.array([source['Z'][old_index]], dtype='float32')
                                      }
print 'Built Soma Coordinates; elapsed time: %.1f s' % (time.time() - start_time)
interp_trees_coords_f.close()
start_time = time.time()
append_cell_attributes(comm, output_coords_path, target_population, soma_coords_dict, namespace=target_namespace,
                       io_size=1)
print 'Appended Soma Coordinates; elapsed time: %.1f s' % (time.time() - start_time)
"""