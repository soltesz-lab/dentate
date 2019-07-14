import os, sys, time
from collections import defaultdict
import logging
import numpy as np
import click
from mpi4py import MPI
import dentate
from dentate import utils
from dentate.utils import viewitems, zip, RunningStats
from neuroh5.io import NeuroH5ProjectionGen, append_cell_attributes, read_population_ranges

sys_excepthook = sys.excepthook
def mpi_excepthook(type, value, traceback):
    sys_excepthook(type, value, traceback)
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)
sys.excepthook = mpi_excepthook

def combine_rstats(a, b, dtype):
    combined = RunningStats.combine(a, b)
    return combined
    
mpi_op_combine_rstats = MPI.Op.Create(combine_rstats, commute=True)

@click.command()
@click.option("--connections-path", '-p', required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--destination", '-d', type=str)
@click.option("--sources", '-s', type=str, multiple=True)
@click.option("--io-size", type=int, default=-1)
@click.option("--verbose", "-v", is_flag=True)
def main(connections_path, destination, sources, io_size, verbose):
    """

    :param connections_path: str
    :param destination: string
    :param sources: string list
    :param verbose:  bool
    """

    utils.config_logging(verbose)
    logger = utils.get_script_logger(os.path.basename(__file__))

    comm = MPI.COMM_WORLD
    rank = comm.rank

    if io_size == -1:
        io_size = comm.size
    if rank == 0:
        logger.info('%i ranks have been allocated' % comm.size)

    pop_ranges, pop_size = read_population_ranges(connections_path, comm=comm)

    count = 0
    gid_count = 0
    start_time = time.time()

    connection_gen_list = [NeuroH5ProjectionGen(connections_path, source, destination, \
                                                    namespaces=['Connections'], \
                                                    comm=comm) for source in sources]

    distance_stats_dict = { source: RunningStats() for source in sources }
    for attr_gen_package in zip(*connection_gen_list):
        local_time = time.time()
        conn_attr_dict = None
        destination_gid = attr_gen_package[0][0]
        if not all([attr_gen_items[0] == destination_gid for attr_gen_items in attr_gen_package]):
            raise Exception('Rank: %i; destination: %s; destination_gid %i not matched across multiple attribute generators: %s' %
                            (rank, destination, destination_gid,
                             str([attr_gen_items[0] for attr_gen_items in attr_gen_package])))
        if destination_gid is not None:
            for (this_destination_gid, (source_gid_array, conn_attr_dict)), source in zip(attr_gen_package, sources):
                for j in range(len(source_gid_array)):
                    this_source_gid = source_gid_array[j]
                    this_distance = conn_attr_dict['Connections']['distance'][j]
                    distance_stats_dict[source].update(this_distance)
            count += 1
        else:
            logger.info('Rank: %i received destination_gid as None' % rank)
        gid_count += 1

    for source in sorted(distance_stats_dict):
        distance_stats = distance_stats_dict[source]
        all_stats = comm.reduce(distance_stats, root=0, op=mpi_op_combine_rstats)
        if rank == 0:
            logger.info('Projection %s -> %s: mean distance: n=%d min=%.2f max=%.f mean=%.2f variance=%.3f' % \
                        (source, destination, all_stats.n, all_stats.min, all_stats.max, \
                        all_stats.mean(), all_stats.variance()))
    
        
    global_count = comm.gather(count, root=0)
    if rank == 0:
        logger.info('destination: %s; %i ranks obtained distances for %i cells in %.2f s' % \
                    (destination, comm.size, np.sum(global_count), time.time() - start_time))
    MPI.Finalize()


if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv)+1):])
