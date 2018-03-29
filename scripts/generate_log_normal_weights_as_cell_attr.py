
import sys, time, gc
from mpi4py import MPI
from neuroh5.io import NeuroH5ProjectionGen, append_cell_attributes, read_population_ranges
import dentate
from dentate.env import Env
import numpy as np
from collections import defaultdict
import click
from itertools import izip, izip_longest
import logging
logging.basicConfig()


"""
stimulus_path: contains namespace with 1D spatial rate map attribute ('rate')
weights_path: contains namespace with initial weights ('Weights'), applied plasticity rule and writes new weights to
 'Structured Weights' namespace
connections_path: contains existing mapping of syn_id to source_gid

10% of GCs will have a subset of weights modified according to a slow time-scale plasticity rule, the rest inherit the
    unaltered initial log-normal weights
    
TODO: Rather than choosing peak_locs randomly, have the peak_locs depend on the previous weight distribution.
"""

script_name = 'generate_log_normal_weights_as_cell_attr.py'
logger = logging.getLogger(script_name)

local_random = np.random.RandomState()

# yields a distribution of synaptic weights with mean  ~>1., and tail ~2.-4.
mu = 0.
sigma = 0.35


@click.command()
@click.option("--config", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--weights-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--weights-namespace", type=str, default='Weights')
@click.option("--connections-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--destination", '-d', type=str)
@click.option("--sources", '-s', type=str, multiple=True)
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--cache-size", type=int, default=50)
@click.option("--seed-offset", type=int, default=4)
@click.option("--verbose", "-v", is_flag=True)
@click.option("--dry-run", is_flag=True)
def main(weights_path, weights_namespace, connections_path, destination, sources, io_size, chunk_size, value_chunk_size, cache_size,
         verbose, dry_run):
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

    if verbose:
        logger.setLevel(logging.INFO)

    comm = MPI.COMM_WORLD
    rank = comm.rank

    env = Env(comm=comm, configFile=config)

    if io_size == -1:
        io_size = comm.size
    if rank == 0:
        logger.info('%s: %i ranks have been allocated' % (script_name, comm.size))

    if (not dry_run) and (rank==0):
        if not os.path.isfile(weights_path):
            input_file  = h5py.File(connections_path,'r')
            output_file = h5py.File(weights_path,'w')
            input_file.copy('/H5Types',output_file)
            input_file.close()
            output_file.close()
    comm.barrier()

    seed_offset = int(env.modelConfig['Random Seeds']['PP Log-Normal Weights 1'])

    pop_ranges, pop_size = read_population_ranges(connections_path, comm=comm)
    destination_gid_offset = pop_ranges[destination][0]

    count = 0
    start_time = time.time()

    connection_gen_list = []
    for source in sources:
        connection_gen_list.append(NeuroH5ProjectionGen(connections_path, source, destination, namespaces=['Synapses'], \
                                                        comm=comm, io_size=io_size, cache_size=cache_size))

    for itercount, attr_gen_package in enumerate(izip_longest(*connection_gen_list)):
        local_time = time.time()
        source_syn_map = defaultdict(list)
        source_weights = None
        source_gid_array = None
        conn_attr_dict = None
        syn_weight_map = {}
        weights_dict = {}
        destination_gid = attr_gen_package[0][0]
        if not all([attr_gen_items[0] == destination_gid for attr_gen_items in attr_gen_package]):
            raise Exception('Rank: %i; destination: %s; destination_gid not matched across multiple attribute generators: %s' %
                            (rank, destination, destination_gid,
                             str([attr_gen_items[0] for attr_gen_items in attr_gen_package])))
        if destination_gid is not None:
            local_random.seed(int(destination_gid + seed_offset))
            for this_destination_gid, (source_gid_array, conn_attr_dict) in attr_gen_package:
                for j in xrange(len(source_gid_array)):
                    this_source_gid = source_gid_array[j]
                    this_syn_id = conn_attr_dict['Synapses'][0][j]
                    source_syn_map[this_source_gid].append(this_syn_id)
            source_weights = local_random.lognormal(mu, sigma, len(source_syn_map))
            # weights are synchronized across all inputs from the same source_gid
            for this_source_gid, this_weight in zip(source_syn_map, source_weights):
                for this_syn_id in source_syn_map[this_source_gid]:
                    syn_weight_map[this_syn_id] = this_weight
            weights_dict[destination_gid - destination_gid_offset] = \
                {'syn_id': np.array(syn_weight_map.keys()).astype('uint32', copy=False),
                 'weight': np.array(syn_weight_map.values()).astype('float32', copy=False)}
            logger.info('Rank %i; destination: %s; destination_gid %i; generated log-normal weights for %i inputs from %i sources in ' \
                        '%.2f s' % (rank, destination, destination_gid, len(syn_weight_map), len(source_weights),
                                    time.time() - local_time))
            count += 1
        else:
            logger.info('Rank: %i received destination_gid as None' % rank)
        if not dry_run:
            append_cell_attributes( weights_path, destination, weights_dict, namespace=weights_namespace,
                                    comm=comm, io_size=io_size, chunk_size=chunk_size, value_chunk_size=value_chunk_size)
            # print 'Rank: %i, just after append' % rank
        del source_syn_map
        del source_weights
        del syn_weight_map
        del source_gid_array
        del conn_attr_dict
        del weights_dict
        gc.collect()

    global_count = comm.gather(count, root=0)
    if rank == 0:
        logger.info('destination: %s; %i ranks generated log-normal weights for %i cells in %.2f s' % \
                    (destination, comm.size, np.sum(global_count), time.time() - start_time))


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])
