from function_lib import *
from mpi4py import MPI
from neuroh5.io import NeuroH5CellAttrGen, append_cell_attributes, bcast_cell_attributes, read_population_ranges
import click
from env import Env


try:
    import mkl
    mkl.set_num_threads(1)
except:
    pass


# log_normal weights: 1./(sigma * x * np.sqrt(2. * np.pi)) * np.exp(-((np.log(x)-mu)**2.)/(2. * sigma**2.))

script_name = 'compute_DG_GC_log_normal_weights.py'

local_random = np.random.RandomState()
# yields a distribution of synaptic weights with mean  ~>1., and tail ~2.-4.
mu = 0.
sigma = 0.35


@click.command()
@click.option("--config", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--weights-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--weights-namespace", type=str, default='Weights')
@click.option("--connectivity-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--connectivity-namespace", type=str, default='Connectivity')
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--cache-size", type=int, default=50)
@click.option("--seed", type=int, default=4)
@click.option("--debug", is_flag=True)
def main(config, weights_path, weights_namespace, connectivity_path, connectivity_namespace, io_size, chunk_size,
         value_chunk_size, cache_size, seed, debug):
    """

    :param weights_path:
    :param weights_namespace:
    :param connectivity_path:
    :param connectivity_namespace:
    :param io_size:
    :param chunk_size:
    :param value_chunk_size:
    :param cache_size:
    :param seed:
    :param debug:
    """
    # make sure random seeds are not being reused for various types of stochastic sampling
    weights_seed_offset = int(seed * 2e6)

    comm = MPI.COMM_WORLD
    rank = comm.rank

    env = Env(comm=comm, configFile=config)

    if io_size == -1:
        io_size = comm.size
    if rank == 0:
        print '%i ranks have been allocated' % comm.size
    sys.stdout.flush()

    population_range_dict = population_ranges(comm, connectivity_path)

    target_population = 'GC'
    count = 0
    start_time = time.time()

    if env.nodeRanks is None:
          (graph, a) = scatter_read_graph(env.comm,connectivityFilePath,io_size=env.IOsize,
                                          projections=[(presyn_name, postsyn_name)],
                                          namespaces=['Synapses','Connections'])
        else:
          (graph, a) = scatter_read_graph(env.comm,connectivityFilePath,io_size=env.IOsize,
                                          node_rank_map=env.nodeRanks,
                                          projections=[(presyn_name, postsyn_name)],
                                          namespaces=['Synapses','Connections'])
        
    edge_iter = graph[postsyn_name][presyn_name]

    connection_dict = env.connection_generator[postsyn_name][presyn_name].connection_properties
    kinetics_dict = env.connection_generator[postsyn_name][presyn_name].synapse_kinetics

    syn_id_attr_index = a[postsyn_name][presyn_name]['Synapses']['syn_id']
    distance_attr_index = a[postsyn_name][presyn_name]['Connections']['distance']

    for (postsyn_gid, edges) in edge_iter:
          
      postsyn_cell   = env.pc.gid2cell(postsyn_gid)
      cell_syn_dict  = cell_synapses_dict[postsyn_gid]
      
      presyn_gids    = edges[0]
      edge_syn_ids   = edges[1]['Synapses'][syn_id_attr_index]

      weights_dict = {}
      syn_ids = np.array([], dtype='uint32')
      weights = np.array([], dtype='float32')
                                                                                                         copy=False)
      local_random.seed(gid + weights_seed_offset)
      weights = np.append(weights, local_random.lognormal(mu, sigma, len(syn_ids))).astype('float32', copy=False)
      weights_dict[gid] = {'syn_id': syn_ids, 'weight': weights}

    global_count = comm.gather(count, root=0)
    if rank == 0:
        print '%i ranks took %.2f s to compute synaptic weights for %i %s cells' % \
              (comm.size, time.time() - start_time, np.sum(global_count), target_population)


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])
