from function_lib import *
from collections import Counter
from mpi4py import MPI
from neurotrees.io import NeurotreeAttrGen
from neurotrees.io import append_cell_attributes
from neurotrees.io import bcast_cell_attributes
from neurotrees.io import population_ranges
import click


try:
    import mkl
    mkl.set_num_threads(1)
except:
    pass


# log_normal weights: 1./(sigma * x * np.sqrt(2. * np.pi)) * np.exp(-((np.log(x)-mu)**2.)/(2. * sigma**2.))

script_name = 'compute_DG_GC_log_normal_weights.py'

# make sure random seeds are not being reused for various types of stochastic sampling
weights_seed_offset = int(4 * 2e6)

local_random = np.random.RandomState()
# yields a distribution of synaptic weights with mean  ~>1., and tail ~2.-4.
mu = 0.
sigma = 0.35


@click.command()
@click.option("--weights-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--weights-namespace", type=str, default='Weights')
@click.option("--connectivity-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--connectivity-namespace", type=str, default='Connectivity')
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--cache-size", type=int, default=50)
@click.option("--debug", is_flag=True)
def main(weights_path, weights_namespace, connectivity_path, connectivity_namespace, io_size, chunk_size,
         value_chunk_size, cache_size, debug):
    """

    :param weights_path:
    :param weights_namespace:
    :param connectivity_path:
    :param connectivity_namespace:
    :param io_size:
    :param chunk_size:
    :param value_chunk_size:
    :param cache_size:
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank

    if io_size == -1:
        io_size = comm.size
    if rank == 0:
        print '%i ranks have been allocated' % comm.size
    sys.stdout.flush()

    population_range_dict = population_ranges(MPI._addressof(comm), connectivity_path)

    target_population = 'GC'
    count = 0
    start_time = time.time()
    attr_gen = NeurotreeAttrGen(MPI._addressof(comm), connectivity_path, target_population, io_size=io_size,
                                cache_size=cache_size, namespace=connectivity_namespace)
    if debug:
        attr_gen = [attr_gen.next() for i in xrange(2)]
    for gid, connectivity_dict in attr_gen:
        local_time = time.time()
        weights_dict = {}
        syn_ids = np.array([], dtype='uint32')
        weights = np.array([], dtype='float32')
        if gid is not None:
            for population in ['MPP', 'LPP']:
                indexes = np.where((connectivity_dict[connectivity_namespace]['source_gid'][:] >=
                                    population_range_dict[population][0]) &
                                   (connectivity_dict[connectivity_namespace]['source_gid'][:] <
                                    population_range_dict[population][0] + population_range_dict[population][1]))[0]
                syn_ids = np.append(syn_ids,
                                    connectivity_dict[connectivity_namespace]['syn_id'][indexes]).astype('uint32',
                                                                                                         copy=False)
            local_random.seed(gid + weights_seed_offset)
            weights = np.append(weights, local_random.lognormal(mu, sigma, len(syn_ids))).astype('float32', copy=False)
            weights_dict[gid] = {'syn_id': syn_ids, 'weight': weights}
            print 'Rank %i: took %.2f s to compute synaptic weights for %s gid %i' % \
                  (rank, time.time() - local_time, target_population, gid)
            count += 1
        if not debug:
            append_cell_attributes(MPI._addressof(comm), weights_path, target_population, weights_dict,
                                   namespace=weights_namespace, io_size=io_size, chunk_size=chunk_size,
                                   value_chunk_size=value_chunk_size)
        sys.stdout.flush()
        del syn_ids
        del weights
        del weights_dict
        gc.collect()

    global_count = comm.gather(count, root=0)
    if rank == 0:
        print '%i ranks took %.2f s to compute synaptic weights for %i %s cells' % \
              (comm.size, time.time() - start_time, np.sum(global_count), target_population)


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])