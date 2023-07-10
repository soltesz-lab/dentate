import os, sys, time, gc
from itertools import islice
import logging, click
from collections import defaultdict
from mpi4py import MPI
import h5py
import numpy as np
import dentate.synapses as synapses
from dentate import utils
from dentate.env import Env
from neuroh5.io import NeuroH5ProjectionGen, append_cell_attributes, read_population_ranges

sys_excepthook = sys.excepthook
def mpi_excepthook(type, value, traceback):
    sys_excepthook(type, value, traceback)
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)
sys.excepthook = mpi_excepthook

def split_every(n, iterable):
    i = iter(iterable)
    piece = list(islice(i, n))
    while piece:
        yield piece
        piece = list(islice(i, n))

mu = 0.
sigma = 3.0


@click.command()
@click.option("--config", required=True, type=str)
@click.option("--config-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--weights-path", required=True, type=click.Path(file_okay=True, dir_okay=False))
@click.option("--weights-namespace", type=str, default='Log-Normal Weights')
@click.option("--weights-name", type=str, default='AMPA')
@click.option("--min-weight", type=float, default=0.0)
@click.option("--max-weight", type=float, default=1.0)
@click.option("--connections-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--destination", '-d', type=str)
@click.option("--sources", '-s', type=str, multiple=True)
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--cache-size", type=int, default=50)
@click.option("--write-size", type=int, default=100)
@click.option("--verbose", "-v", is_flag=True)
@click.option("--dry-run", is_flag=True)
def main(config, config_prefix, weights_path, weights_namespace, weights_name, min_weight, max_weight, connections_path, destination, sources, io_size, chunk_size, value_chunk_size, write_size, cache_size, verbose, dry_run):
    """

    :param weights_path: str
    :param weights_namespace: str
    :param connections_path: str
    :param io_size: int
    :param chunk_size: int
    :param value_chunk_size: int
    :param cache_size: int
    :param verbose:  bool
    :param dry_run:  bool
    """

    utils.config_logging(verbose)
    logger = utils.get_script_logger(os.path.basename(__file__))

    comm = MPI.COMM_WORLD
    rank = comm.rank

    env = Env(comm=comm, config=config, config_prefix=config_prefix)

    if max_weight < min_weight:
        x = max_weight
        max_weight = min_weight
        min_weight = x
    
    if io_size == -1:
        io_size = comm.size
    if rank == 0:
        logger.info(f"{comm.size} ranks have been allocated")

    if (not dry_run) and (rank==0):
        if not os.path.isfile(weights_path):
            input_file  = h5py.File(connections_path,'r')
            output_file = h5py.File(weights_path,'w')
            input_file.copy('/H5Types',output_file)
            input_file.close()
            output_file.close()
    comm.barrier()

    seed_offset = int(env.model_config['Random Seeds']['GC Log-Normal Weights 1'])

    pop_ranges, pop_size = read_population_ranges(connections_path, comm=comm)

    count = 0
    gid_count = 0
    start_time = time.time()

    connection_gen_list = [NeuroH5ProjectionGen(connections_path, source, destination, \
                                                    namespaces=['Synapses'], cache_size=cache_size, \
                                                    comm=comm, io_size=io_size) for source in sources]

    weights_dict = {}
    for attr_gen_package in utils.zip_longest(*connection_gen_list):
        local_time = time.time()
        source_syn_dict = defaultdict(list)
        source_gid_array = None
        conn_attr_dict = None
        destination_gid = attr_gen_package[0][0]
        if not all([attr_gen_items[0] == destination_gid for attr_gen_items in attr_gen_package]):
            raise Exception(f"Rank: {rank}; destination: {destination}; destination_gid {destination_gid} not matched "
                            f"across multiple attribute generators: {list([attr_gen_items[0] for attr_gen_items in attr_gen_package])}")
        if destination_gid is not None:
            seed = int(destination_gid + seed_offset)
            for this_destination_gid, (source_gid_array, conn_attr_dict) in attr_gen_package:
                for j in range(len(source_gid_array)):
                    this_source_gid = source_gid_array[j]
                    this_syn_id = conn_attr_dict['Synapses']['syn_id'][j]
                    source_syn_dict[this_source_gid].append(this_syn_id)
            weights_dict[destination_gid] = \
              synapses.generate_log_normal_weights(weights_name, mu, sigma, seed, source_syn_dict, clip=(min_weight, max_weight))
            logger.info(f"Rank {rank}; destination: {destination}; destination gid {destination_gid}; sources: {sources}; "
                        f"generated log-normal weights for {len(weights_dict[destination_gid]['syn_id'])} inputs in "
                        f"{(time.time() - local_time):.02f} s")
            count += 1
        else:
            logger.info(f"Rank: {rank} received destination_gid as None")
        gid_count += 1

    if not dry_run:
        if write_size > 0:
            items = split_every(write_size, weights_dict.items())
            for chunk in items:
                weights_chunk = dict(chunk)
                append_cell_attributes( weights_path, destination, weights_chunk, namespace=weights_namespace,
                                    comm=comm, io_size=io_size, chunk_size=chunk_size, value_chunk_size=value_chunk_size)
                comm.barrier()
        else:
            append_cell_attributes( weights_path, destination, weights_dict, namespace=weights_namespace,
                                    comm=comm, io_size=io_size, chunk_size=chunk_size, value_chunk_size=value_chunk_size)
            comm.barrier()

        
    global_count = comm.gather(count, root=0)
    if rank == 0:
        logger.info(f"destination: {destination}; {comm.size} ranks generated log-normal weights "
                    f"for {np.sum(global_count)} cells in {(time.time() - start_time):.02f} s")
    MPI.Finalize()


if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv)+1):])
