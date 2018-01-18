
import sys, time, gc
from mpi4py import MPI
import h5py
from neuroh5.io import NeuroH5CellAttrGen, append_cell_attributes, read_population_ranges, bcast_cell_attributes, \
    NeuroH5ProjectionGen
import numpy as np
from collections import defaultdict
import click
from utils import *
import stimulus
from itertools import izip_longest
from neuroh5_io_utils import *

"""
stimulus_path: contains namespace with 1D spatial rate map attribute ('rate')
weights_path: contains namespace with initial weights ('Weights'), applied plasticity rule and writes new weights to
 'Structured Weights' namespace
connections_path: contains existing mapping of syn_id to source_gid

10% of GCs will have a subset of weights modified according to a slow time-scale plasticity rule, the rest inherit the
    unaltered initial log-normal weights
    
TODO: Rather than choosing peak_locs randomly, have the peak_locs depend on the previous weight distribution.
"""

try:
    import mkl
    mkl.set_num_threads(1)
except:
    pass

script_name = 'test_h5py_neuroh5_compatibility.py'


@click.command()
@click.option("--file-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--namespace", type=str, default='Synapse Attributes')
@click.option("--attribute", type=str, default='syn_locs')
@click.option("--population", type=str, default='GC')
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--cache-size", type=int, default=50)
@click.option("--trajectory-id", type=int, default=0)
def main(file_path, namespace, attribute, population, io_size, chunk_size, value_chunk_size, cache_size, trajectory_id):
    """

    :param file_path: str
    :param namespace: str
    :param attribute: str
    :param population: str
    :param io_size: int
    :param chunk_size: int
    :param value_chunk_size: int
    :param cache_size: int
    :param trajectory_id: int
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank

    if io_size == -1:
        io_size = comm.size
    if rank == 0:
        print '%s: %i ranks have been allocated' % (script_name, comm.size)
    sys.stdout.flush()

    trajectory_namespace = 'Trajectory %s' % str(trajectory_id)

    arena_dimension = 100.  # minimum distance from origin to boundary (cm)
    default_run_vel = 30.  # cm/s
    spatial_resolution = 1.  # cm

    with h5py.File(file_path, 'a', driver='mpio', comm=comm) as f:
        if trajectory_namespace not in f:
            print 'Rank: %i; Creating %s datasets' % (rank, trajectory_namespace)
            group = f.create_group(trajectory_namespace)
            t, x, y, d = stimulus.generate_trajectory(arena_dimension=arena_dimension, velocity=default_run_vel,
                                                      spatial_resolution=spatial_resolution)
            for key, value in zip(['x', 'y', 'd', 't'], [x, y, d, t]):
                dataset = group.create_dataset(key, (value.shape[0],), dtype='float32')
                with dataset.collective:
                    dataset[:] = value.astype('float32', copy=False)
        else:
            print 'Rank: %i; Reading %s datasets' % (rank, trajectory_namespace)
            group = f[trajectory_namespace]
            dataset = group['x']
            with dataset.collective:
                x = dataset[:]
            dataset = group['y']
            with dataset.collective:
                y = dataset[:]
            dataset = group['d']
            with dataset.collective:
                d = dataset[:]
            dataset = group['t']
            with dataset.collective:
                t = dataset[:]

    target = population

    pop_ranges, pop_size = read_population_ranges(comm, file_path)
    target_gid_offset = pop_ranges[target][0]

    start_time = time.time()

    attr_gen = NeuroH5CellAttrGen(comm, file_path, target, io_size=io_size, cache_size=cache_size,
                                             namespace=namespace)

    index_map = get_cell_attributes_gid_index_map(comm, file_path, target, namespace)

    maxiter = 10
    for itercount, (target_gid, attr_dict) in enumerate(attr_gen):
        print 'Rank: %i receieved target_gid: %s from the attribute generator.' % (rank, str(target_gid))
        attr_dict2 = get_cell_attributes_by_gid(target_gid, comm, file_path, index_map, target, namespace)
        if np.all(attr_dict[attribute][:] == attr_dict2[attribute][:]):
            print 'Rank: %i; attributes match!' % rank
        else:
            print 'Rank: %i; attributes do not match.' % rank
        comm.barrier()
        if itercount > maxiter:
            break


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])
