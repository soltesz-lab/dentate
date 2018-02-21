"""
This script will use an h5py collective write operation to write a sample Trajectory dataset, while also reading
cell_attributes from a neuroH5 file.
file-path: contains cell_attributes in the provided namespace
"""
from dentate.utils import *
from mpi4py import MPI
import h5py
from neuroh5.io import read_population_ranges, NeuroH5CellAttrGen
import numpy as np
import click
import dentate.stimulus as stimulus
from neuroh5.h5py_io_utils import *


@click.command()
@click.option("--file-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--namespace", type=str, default='Synapse Attributes')
@click.option("--attribute", type=str, default='syn_locs')
@click.option("--population", type=str, default='GC')
@click.option("--io-size", type=int, default=-1)
@click.option("--cache-size", type=int, default=50)
@click.option("--trajectory-id", type=int, default=0)
def main(file_path, namespace, attribute, population, io_size, cache_size, trajectory_id):
    """

    :param file_path: str (path)
    :param namespace: str
    :param attribute: str
    :param population: str
    :param io_size: int
    :param cache_size: int
    :param trajectory_id: int
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank

    if io_size == -1:
        io_size = comm.size
    if rank == 0:
        print '%s: %i ranks have been allocated' % (os.path.basename(__file__).split('.py')[0], comm.size)
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

    pop_ranges, pop_size = read_population_ranges(file_path, comm=comm)
    target_gid_offset = pop_ranges[target][0]

    attr_gen = NeuroH5CellAttrGen(file_path, target, comm=comm, io_size=io_size, cache_size=cache_size,
                                             namespace=namespace)
    index_map = get_cell_attributes_index_map(comm, file_path, target, namespace)

    maxiter = 10
    matched = 0
    processed = 0
    for itercount, (target_gid, attr_dict) in enumerate(attr_gen):
        print 'Rank: %i receieved target_gid: %s from the attribute generator.' % (rank, str(target_gid))
        attr_dict2 = select_cell_attributes(target_gid, comm, file_path, index_map, target, namespace,
                                            population_offset=target_gid_offset)
        if np.all(attr_dict[attribute][:] == attr_dict2[attribute][:]):
            print 'Rank: %i; cell attributes match!' % rank
            matched += 1
        else:
            print 'Rank: %i; cell attributes do not match.' % rank
        comm.barrier()
        processed += 1
        if itercount > maxiter:
            break
    matched = comm.gather(matched, root=0)
    processed = comm.gather(processed, root=0)
    if comm.rank == 0:
        print '%i / %i processed gids had matching cell attributes returned by both read methods' % \
              (np.sum(matched), np.sum(processed))


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1,sys.argv)+1):])
