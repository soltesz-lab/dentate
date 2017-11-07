
import sys, time, gc
from mpi4py import MPI
from neuroh5.io import NeuroH5CellAttrGen, append_cell_attributes, read_population_ranges, bcast_cell_attributes, \
    NeuroH5ProjectionGen
import numpy as np
from collections import defaultdict
import click
from utils import *
from itertools import izip, izip_longest

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

script_name = 'generate_DG_GC_log_normal_weights_as_cell_attr.py'

local_random = np.random.RandomState()

# yields a distribution of synaptic weights with mean  ~>1., and tail ~2.-4.
mu = 0.
sigma = 0.35


@click.command()
@click.option("--weights-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--weights-namespace", type=str, default='Weights')
@click.option("--connections-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--cache-size", type=int, default=50)
@click.option("--seed-offset", type=int, default=4)
@click.option("--debug", is_flag=True)
def main(weights_path, weights_namespace, connections_path, io_size, chunk_size, value_chunk_size, cache_size,
         seed_offset, debug):
    """

    :param weights_path: str
    :param weights_namespace: str
    :param connections_path: str
    :param io_size: int
    :param chunk_size: int
    :param value_chunk_size: int
    :param cache_size: int
    :param seed_offset: int
    :param debug:  bool
    """
    # make sure random seeds are not being reused for various types of stochastic sampling
    seed_offset *= 2e6

    comm = MPI.COMM_WORLD
    rank = comm.rank

    if io_size == -1:
        io_size = comm.size
    if rank == 0:
        print '%i ranks have been allocated' % comm.size
    sys.stdout.flush()

    source_population_list = ['MC', 'MPP', 'LPP']
    target = 'GC'

    pop_ranges, pop_size = read_population_ranges(comm, connections_path)
    target_gid_offset = pop_ranges[target][0]

    count = 0
    start_time = time.time()

    connection_gen_list = []
    for source in source_population_list:
        connection_gen_list.append(NeuroH5ProjectionGen(comm, connections_path, source, target, io_size=io_size,
                                                           cache_size=cache_size, namespaces=['Synapses']))

    maxiter = 100 if debug else None
    for itercount, attr_gen_package in enumerate(izip_longest(*connection_gen_list)):
        local_time = time.time()
        source_syn_map = defaultdict(list)
        source_weights = None
        source_gid_array = None
        conn_attr_dict = None
        syn_weight_map = {}
        weights_dict = {}
        target_gid = attr_gen_package[0][0]
        if not all(attr_gen_items[0] == target_gid for attr_gen_items in attr_gen_package):
            message = 'target: %s; target_gid: %i not matched across multiple connection_gens' % (target, target_gid)
            print 'Rank: %i; %s' % (comm.rank, message)
            sys.stdout.flush()
            raise Exception(message)
        if target_gid is not None:
            local_random.seed(int(target_gid + seed_offset))
            for this_target_gid, (source_gid_array, conn_attr_dict) in attr_gen_package:
                for j in xrange(len(source_gid_array)):
                    this_source_gid = source_gid_array[j]
                    this_syn_id = conn_attr_dict['Synapses'][0][j]
                    source_syn_map[this_source_gid].append(this_syn_id)
            source_weights = local_random.lognormal(mu, sigma, len(source_syn_map))
            # weights are synchronized across all inputs from the same source_gid
            for this_source_gid, this_weight in zip(source_syn_map, source_weights):
                for this_syn_id in source_syn_map[this_source_gid]:
                    syn_weight_map[this_syn_id] = this_weight
            weights_dict[target_gid - target_gid_offset] = \
                {'syn_id': np.array(syn_weight_map.keys()).astype('uint32', copy=False),
                 'weight': np.array(syn_weight_map.values()).astype('float32', copy=False)}
            print 'Rank %i; target: %s; gid %i; generated log-normal weights for %i inputs from %i sources in ' \
                  '%.2f s' % (rank, target, target_gid, len(syn_weight_map), len(source_weights),
                              time.time() - local_time)
            count += 1
        else:
            print 'Rank: %i received target_gid as None' % comm.rank
        if not debug:

            append_cell_attributes(comm, weights_path, target, weights_dict, namespace=weights_namespace,
                                   io_size=io_size, chunk_size=chunk_size, value_chunk_size=value_chunk_size)
            print 'Rank: %i, just after append' % comm.rank
        sys.stdout.flush()
        del source_syn_map
        del source_weights
        del syn_weight_map
        del source_gid_array
        del conn_attr_dict
        del weights_dict
        gc.collect()
        if debug and maxiter is not None and itercount > maxiter:
            break
    print 'Rank: %i exited the loop' % comm.rank
    global_count = comm.gather(count, root=0)
    if rank == 0:
        print 'target: %s; %i ranks generated log-normal weights for %i cells in %.2f s' % \
              (target, comm.size, np.sum(global_count), time.time() - start_time)


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])