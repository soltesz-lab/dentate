import gc
import logging
import os
import random
import sys

import numpy as np

import click
import dentate
import dentate.utils as utils
from dentate.connection_generator import ConnectionProb
from dentate.connection_generator import generate_uv_distance_connections
from dentate.env import Env
from dentate.geometry import measure_distances
from dentate.neuron_utils import configure_hoc_env
from mpi4py import MPI
from neuroh5.io import NeuroH5TreeGen
from neuroh5.io import append_cell_attributes
from neuroh5.io import append_cell_trees
from neuroh5.io import bcast_cell_attributes
from neuroh5.io import read_population_ranges

sys_excepthook = sys.excepthook
def mpi_excepthook(type, value, traceback):
    sys_excepthook(type, value, traceback)
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)
sys.excepthook = mpi_excepthook


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
    
    utils.config_logging(verbose)
    logger = utils.get_script_logger(os.path.basename(__file__))

    comm = MPI.COMM_WORLD
    rank = comm.rank

    if io_size == -1:
        io_size = comm.size
    if rank == 0:
        logger.info('%i ranks have been allocated' % comm.size)

    random.seed(13)

    (forest_pop_ranges, _)  = read_population_ranges(forest_path)
    (forest_population_start, forest_population_count) = forest_pop_ranges[population]


    (pop_ranges, _)  = read_population_ranges(output_path)

    (population_start, population_count) = pop_ranges[population]

    if rank == 0:
        logger.info('reading new cell index map...')
    reindex_map1 = {}
    reindex_map_gen = bcast_cell_attributes(index_path, population, namespace=index_namespace, 
                                            root=0, comm=comm)
    for gid, attr_dict in reindex_map_gen:
        reindex_map1[gid] = attr_dict['New Cell Index'][0]


    if rank == 0:
        logger.info('reading cell coordinates...')
    old_coords_dict = {}
    coords_map_gen = bcast_cell_attributes(index_path, population, namespace=coords_namespace,
                                           root=0, comm=comm)
    for gid, attr_dict in coords_map_gen:
        old_coords_dict[gid] = attr_dict

    gc.collect()
    if rank == 0:
        logger.info('sampling cell population reindex...')
        from guppy import hpy
        h = hpy()
        logger.info(h.heap())

    reindex_map = None
    if rank == 0:
        reindex_map = {}
        N = len(reindex_map1)
        K = 0
        while K < population_count:
            i = random.randint(forest_population_start, forest_population_start+N-1)
            if i in reindex_map1:
                reindex_map[i] = reindex_map1[i]
                del(reindex_map1[i])
                K = K+1
    reindex_map = comm.bcast(reindex_map, root=0)

    if rank == 0:
        logger.info('computing new population index...')

    gid_map = { k: i+population_start for i,k in enumerate(sorted(reindex_map.keys())) }
    
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

    MPI.Finalize()

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv)+1):])
