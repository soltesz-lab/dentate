import sys, random, click, logging
from mpi4py import MPI
import numpy as np
from neuroh5.io import read_population_ranges, append_cell_trees, append_cell_attributes, bcast_cell_attributes, NeuroH5TreeGen
from dentate.utils import *

script_name = 'reindex_trees.py'


@click.command()
@click.option("--population", required=True, type=str)
@click.option("--forest-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--output-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--index-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--index-namespace", type=str, default='Tree Reindex')
@click.option("--coords-namespace", type=str, default='Interpolated Coordinates')
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--verbose", '-v', is_flag=True)
def main(population, forest_path, output_path, index_path, index_namespace, coords_namespace, io_size, chunk_size, value_chunk_size, verbose):
    """

    :param population: str
    :param forest_path: str (path)
    :param output_path: str (path)
    :param index_path: str (path)
    :param io_size: int
    :param chunk_size: int
    :param value_chunk_size: int
    :param verbose: bool
    """
    
    config_logging(verbose)
    logger = get_script_logger(script_name)

    comm = MPI.COMM_WORLD
    rank = comm.rank

    if io_size == -1:
        io_size = comm.size
    if rank == 0:
        logger.info('%i ranks have been allocated' % comm.size)

    (pop_ranges, _)  = read_population_ranges(output_path)

    (population_start, population_count) = pop_ranges[population]
     
    reindex_map1 = {}
    reindex_map_gen = bcast_cell_attributes(index_path, population, 0, namespace=index_namespace)
    for gid, attr_dict in reindex_map_gen:
        reindex_map1[gid] = attr_dict['New Cell Index'][0]

    old_coords_dict = {}
    coords_map_gen = bcast_cell_attributes(index_path, population, 0, namespace=coords_namespace)
    for gid, attr_dict in coords_map_gen:
        old_coords_dict[gid] = attr_dict

    reindex_keys = None
    if rank == 0:
        reindex_keys = random.sample(list(reindex_map1), population_count)
    reindex_keys = comm.bcast(reindex_keys, root=0)
        
    reindex_map = { k : reindex_map1[k] for k in reindex_keys }

    gid_map = { k: i+population_start for i,k in enumerate(reindex_keys) }
    
    new_coords_dict = {}
    new_trees_dict = {}
    count = 0
    for gid, old_trees_dict in NeuroH5TreeGen(forest_path, population, io_size=io_size, comm=comm, topology=False):
        if gid is not None and gid in reindex_map:
            reindex_gid = reindex_map[gid]
            new_gid = gid_map[gid]
            new_trees_dict[new_gid] = old_trees_dict
            new_coords_dict[new_gid] = old_coords_dict[gid]
            logger.info('Rank: %i mapping old gid: %i to new gid: %i' % (comm.rank, gid, new_gid))
        comm.barrier()
        count += 1
    append_cell_trees(output_path, population, new_trees_dict, io_size=io_size, comm=comm)
    append_cell_attributes(output_path, population, new_coords_dict, \
                           namespace=coords_namespace, io_size=io_size, comm=comm)

    if comm.rank == 0:
        logger.info('Appended reindexed trees to %s' % output_path)


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])
