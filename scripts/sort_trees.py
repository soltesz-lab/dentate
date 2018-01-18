import sys
from mpi4py import MPI
import numpy as np
from neuroh5.io import append_cell_trees, bcast_cell_attributes, NeuroH5TreeGen
from utils import *
import pprint
import click


@click.command()
@click.option("--population", required=True, type=str)
@click.option("--forest-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--output-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--index-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--debug", is_flag=True)
def main(population, forest_path, output_path, index_path, io_size, chunk_size, value_chunk_size, debug):
    """

    :param population: str
    :param forest_path: str (path)
    :param output_path: str (path)
    :param index_path: str (path)
    :param io_size: int
    :param chunk_size: int
    :param value_chunk_size: int
    :param debug: bool
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank

    if io_size == -1:
        io_size = comm.size
    if rank == 0:
        print '%i ranks have been allocated' % comm.size
    sys.stdout.flush()

    reindex_map = {}
    reindex_map_gen = bcast_cell_attributes(comm, 0, index_path, population, namespace='Tree Reindex Map')
    for gid, attr_dict in reindex_map_gen:
        reindex_map[gid] = attr_dict['New Cell Index'][0]
    new_trees_dict = {}
    count = 0
    for gid, old_trees_dict in NeuroH5TreeGen(comm, forest_path, population, io_size=io_size):
        if gid is not None and gid in reindex_map:
            new_gid = reindex_map[gid]
            new_trees_dict[new_gid] = old_trees_dict
            print 'Rank: %i mapping old_gid: %i to new_gid: %i' % (comm.rank, gid, new_gid)
            sys.stdout.flush()
        comm.barrier()
        if count > 2:
            break
        count += 1
    append_cell_trees(comm, output_path, population, new_trees_dict, io_size=io_size)
    """
    print 'old_trees: old gid: %i; new_gid: %i' % (gid, new_gid)
    pprint.pprint(old_trees_dict.keys())
    sys.stdout.flush()
    break
    """

    if comm.rank == 0:
        print 'Appended sorted trees to %s' % output_path
        sys.stdout.flush()


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1,sys.argv)+1):],
         standalone_mode=False)
