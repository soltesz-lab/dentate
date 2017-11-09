import sys, os
from mpi4py import MPI # Must come before importing NEURON
import h5py
import numpy as np


def get_cell_attributes_gid_index_map(comm, file_path, population, namespace, index_name='Cell Index'):
    """

    :param comm: MPI communicator
    :param file_path: str (path to neuroh5 file)
    :param population: str
    :param namespace: str
    :param index_name: str
    :return: dict
    """
    with h5py.File(file_path, 'r', driver='mpio', comm=comm) as f:
        group = f['Populations'][population][namespace].itervalues().next()
        dataset = group[index_name]
        with dataset.collective:
            index_map = dict(zip(dataset[:], xrange(dataset.shape[0])))
    return index_map


def get_cell_attributes_by_gid(gid, comm, file_path, index_map, population, namespace, start_gid=0,
                               pointer_name='Attribute Pointer', value_name='Attribute Value'):
    """

    :param gid: int
    :param comm: MPI communicator
    :param file_path: str (path to neuroh5 file)
    :param gid_map: dict of int: int; maps gid to attribute index
    :param population: str
    :param namespace: str
    :param start_gid: int
    :param pointer_name: str
    :param value_name: str
    :return: dict
    """
    try:
        index = index_map[gid - start_gid]
    except KeyError:
        index = 0
    in_dataset = index is not None
    attr_dict = {}
    with h5py.File(file_path, 'r', driver='mpio', comm=comm) as f:
        group = f['Populations'][population][namespace]
        for attribute in group:
            pointer_dataset = group[attribute][pointer_name]
            with pointer_dataset.collective:
                start = pointer_dataset[index]
                end = pointer_dataset[index + 1]
            value_dataset = group[attribute][value_name]
            with value_dataset.collective:
                attr_dict[attribute] = value_dataset[start:end]
    if in_dataset:
        return attr_dict
    else:
        return None


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.rank
    file_path = '../datasets/Full_Scale_Control/DGC_forest_test_syns_weights_20171107.h5'
    population = 'GC'
    namespace = 'Weights'
    index_map = get_cell_attributes_gid_index_map(comm, file_path, population, namespace)
    gid = index_map.keys()[rank]
    attr_dict = get_cell_attributes_by_gid(gid, comm, file_path, index_map, population, namespace)
    print 'Rank: %i, gid: %i, num_syns: %i' % (rank, gid, len(attr_dict['syn_id']))
    sys.stdout.flush()