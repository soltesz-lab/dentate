import sys, os, time, gc, click, logging
from collections import defaultdict
import numpy as np
from mpi4py import MPI
import neuroh5
from neuroh5.io import append_cell_attributes, read_population_ranges, bcast_cell_attributes, \
    scatter_read_cell_attributes, NeuroH5ProjectionGen
import dentate
from dentate.env import Env
from dentate import utils, stimulus
from dentate.utils import *
import h5py


"""
stimulus_path: contains namespace with 1D spatial rate map attribute ('rate')
weights_path: contains namespace with initial weights ('Weights'), applied plasticity rule and writes new weights to
 'Structured Weights' namespace
connections_path: contains existing mapping of syn_id to source_gid

10% of GCs will have a subset of weights modified according to a slow time-scale plasticity rule, the rest inherit the
    unaltered initial log-normal weights
    
"""
def weights_dict_alltoall(comm, syn_name, initial_weights_dict, query, clear=False):
    rank = comm.rank
    send_syn_ids = []
    send_weights = []
    for gid in query:
        if gid in initial_weights_dict:
            this_initial_weights_dict = initial_weights_dict[gid]
            syn_ids = this_initial_weights_dict['syn_id']
            weights = this_initial_weights_dict[syn_name]
            send_syn_ids.append(syn_ids)
            send_weights.append(weights)
            if clear:
                del(initial_weights_dict[gid])
        else:
            send_syn_ids.append(None)
            send_weights.append(None)
    recv_syn_ids = comm.alltoall(send_syn_ids)
    recv_weights = comm.alltoall(send_weights)
    syn_weight_dict = {}
    for syn_id_array, recv_weight_array in zip(recv_syn_ids, recv_weights):
        if syn_id_array is not None:
            for (syn_id, weight) in zip(syn_id_array, recv_weight_array):
                syn_weight_dict[int(syn_id)] = float(weight) 
    return syn_weight_dict


def source_rate_map_dict_alltoall(comm, stimulus_attrs, query, clear=False):
    rank = comm.rank
    send_source_gids = []
    send_source_rate_maps = []
    for source_dict in query:
        this_send_source_gids = {}
        this_send_source_rate_maps = {}
        for source in source_dict:
            this_send_source_gids[source] = []
            this_send_source_rate_maps[source] = []
            source_gid_dict = source_dict[source]
            for gid in source_gid_dict:
                if gid in stimulus_attrs[source]:
                    rate_map = stimulus_attrs[source][gid]['Trajectory Rate Map']
                    this_send_source_gids[source].append(gid)
                    this_send_source_rate_maps[source].append(rate_map)
                    if clear:
                        del(stimulus_attrs[source][gid])
        send_source_gids.append(this_send_source_gids)
        send_source_rate_maps.append(this_send_source_rate_maps)
    recv_source_gids = comm.alltoall(send_source_gids)
    recv_source_rate_maps = comm.alltoall(send_source_rate_maps)
    source_rate_map_dict = defaultdict(lambda: defaultdict(list))
    for source_gids, source_rate_maps in zip(recv_source_gids, recv_source_rate_maps):
        for source in source_gids:
            this_source_rate_map_dict = source_rate_map_dict[source]
            for (source_gid, rate_map) in zip(source_gids[source], source_rate_maps[source]):
                this_source_rate_map_dict[int(source_gid)] = rate_map
    return source_rate_map_dict
    

sys_excepthook = sys.excepthook
def mpi_excepthook(type, value, traceback):
    sys_excepthook(type, value, traceback)
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)
#sys.excepthook = mpi_excepthook

peak_rate_dict = {'MPP': 20., 'LPP': 20., 'CA3c': 20.}  # Hz


@click.command()
@click.option("--config", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--stimulus-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--stimulus-namespace", type=str, default='Vector Stimulus')
@click.option("--output-weights-path", required=False, type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option("--weights-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--synapse-name", type=str, default='AMPA')
@click.option("--initial-weights-namespace", type=str, default='Weights')
@click.option("--structured-weights-namespace", type=str, default='Structured Weights')
@click.option("--connections-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--destination", '-d', type=str)
@click.option("--sources", '-s', type=str, multiple=True)
@click.option("--arena-id", type=str, default='A')
@click.option("--trajectory-id", type=str, default='Diag')
@click.option("--target-sparsity", type=float, default=0.1)
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--cache-size", type=int, default=50)
@click.option("--write-size", type=int, default=1)
@click.option("--verbose", "-v", is_flag=True)
@click.option("--dry-run", is_flag=True)
def main(config, stimulus_path, stimulus_namespace, output_weights_path, weights_path, synapse_name, initial_weights_namespace,
         structured_weights_namespace, connections_path, destination, sources, arena_id, trajectory_id, target_sparsity,
         io_size, chunk_size, value_chunk_size, cache_size, write_size, verbose, dry_run):
    """

    :param config: str (path to .yaml file)
    :param stimulus_path: str (path to .h5 file)
    :param stimulus_namespace: str
    :param weights_path: str (path to .h5 file)
    :param initial_weights_namespace: str
    :param structured_weights_namespace: str
    :param connections_path: str (path to .h5 file)
    :param destination: str
    :param sources: list of str
    :param stimulus_id: int
    :param target_sparsity: float
    :param io_size:
    :param chunk_size:
    :param value_chunk_size:
    :param cache_size:
    :param write_size:
    :param verbose:
    :param dry_run:
    :return:
    """
    """

    :param stimulus_path: str
    :param stimulus_namespace: str
    :param weights_path: str
    :param initial_weights_namespace: str
    :param structured_weights_namespace: str
    :param connections_path: str
    :param io_size: int
    :param chunk_size: int
    :param value_chunk_size: int
    :param cache_size: int
    :param stimulus_id: int
    :param target_sparsity: float
    :param dry_run:  bool
    """

    utils.config_logging(verbose)
    logger = utils.get_script_logger(__file__)

    comm = MPI.COMM_WORLD
    rank = comm.rank
    nranks = comm.size

    if io_size == -1:
        io_size = comm.size
    if rank == 0:
        logger.info('%s: %i ranks have been allocated' % (__file__, comm.size))

    env = Env(comm=comm, config_file=config, io_size=io_size)

    if output_weights_path is None:
        output_weights_path = weights_path

    
    if (not dry_run) and (rank==0):
        if not os.path.isfile(output_weights_path):
            input_file  = h5py.File(weights_path,'r')
            output_file = h5py.File(output_weights_path,'w')
            input_file.copy('/H5Types',output_file)
            input_file.close()
            output_file.close()
    env.comm.barrier()


    this_structured_weights_namespace = '%s %s %s' % (structured_weights_namespace, arena_id, trajectory_id)
    this_stimulus_namespace = '%s %s %s' % (stimulus_namespace, arena_id, trajectory_id)


    if rank == 0:
        logger.info('Reading initial weights data from %s...' % weights_path)
    env.comm.barrier()
    cell_attributes_dict = scatter_read_cell_attributes(weights_path, destination, 
                                                        namespaces=[initial_weights_namespace], 
                                                        comm=env.comm, io_size=env.io_size)
    initial_weights_dict = None
    if initial_weights_namespace in cell_attributes_dict:
        initial_weights_iter = cell_attributes_dict[initial_weights_namespace]
        initial_weights_dict = { gid: attr_dict for gid, attr_dict in initial_weights_iter }
    else:
        raise RuntimeError('Initial weights namespace %s was not found in file %s' % (initial_weights_namespace, weights_path))
    
    logger.info('Rank %i; destination: %s; read synaptic weights for %i cells' %
                (rank, destination, len(initial_weights_dict)))

    if rank == 0:
        logger.info('Reading stimulus data from %s...' % stimulus_path)
    env.comm.barrier()
    stimulus_attrs = {}
    for source in sources:
        stimulus_attr_dict = scatter_read_cell_attributes(stimulus_path, source, namespaces=[this_stimulus_namespace], \
                                                         mask=set(['Trajectory Rate Map']), io_size=env.io_size, comm=env.comm)
        if this_stimulus_namespace in stimulus_attr_dict:
            stimulus_attrs[source] = { gid: attr_dict for gid, attr_dict in stimulus_attr_dict[this_stimulus_namespace] }
            if rank == 0:
                logger.info('Read stimulus data for %i cells in population %s' % (len(stimulus_attrs[source]), source))
        else:
            raise RuntimeError('Stimulus namespace %s was not found in file %s' % (this_stimulus_namespace, stimulus_path))

    local_random = np.random.RandomState()

    seed_offset = int(env.modelConfig['Random Seeds']['GC Structured Weights'])
    spatial_resolution = env.stimulus_config['Spatial Resolution'] # cm

    arena = env.stimulus_config['Arena'][arena_id]
    
    if trajectory_id not in arena.trajectories:
        raise RuntimeError('Trajectory with ID: %s not specified by configuration at file path: %s' %
                           (trajectory_id, config_prefix + '/' + config))
    trajectory = arena.trajectories[trajectory_id]

    default_run_vel = trajectory.velocity  # cm/s

    trajectory_namespace = 'Trajectory %s %s' % (str(arena_id), str(trajectory_id))
    if rank == 0:
        with h5py.File(stimulus_path) as f:
          group = f[trajectory_namespace]
          dataset = group['x']
          x = dataset[:]
          dataset = group['y']
          y = dataset[:]
          dataset = group['d']
          d = dataset[:]
          dataset = group['t']
          t = dataset[:]
    else:
        x = None
        y = None
        d = None
        t = None

    x = env.comm.bcast(x, root=0)
    y = env.comm.bcast(y, root=0)
    d = env.comm.bcast(d, root=0)
    t = env.comm.bcast(t, root=0)
    
    plasticity_window_dur = 4.  # s
    plasticity_kernel_sigma = plasticity_window_dur * default_run_vel / 3. / np.sqrt(2.)  # cm
    plasticity_kernel = lambda d, d_offset: np.exp(-((d - d_offset) / plasticity_kernel_sigma) ** 2.)
    plasticity_kernel = np.vectorize(plasticity_kernel, excluded=[1])
    max_plasticity_kernel_area = np.sum(plasticity_kernel(d, np.max(d) / 2.)) * spatial_resolution

    count = 0
    gid_count = 0
    structured_count = 0
    start_time = time.time()

    connection_gen_list = []

    connection_gen_list = [ NeuroH5ProjectionGen(connections_path, source, destination, namespaces=['Synapses'], comm=comm) \
                               for source in sources ]

    structured_weights_dict = {}
    for itercount, attr_gen_package in enumerate(zip_longest(*connection_gen_list)):
        local_time = time.time()
        source_syn_map = defaultdict(lambda: defaultdict(list))
        syn_weight_dict = {}
        syn_peak_index_map = {}
        modulated_inputs = 0
        source_gid_array = None
        conn_attr_dict = None
        destination_gid = attr_gen_package[0][0]
        if not all([attr_gen_items[0] == destination_gid for attr_gen_items in attr_gen_package]):
            raise Exception('Rank: %i; destination: %s; destination_gid not matched across multiple attribute '
                            'generators: %s' % (rank, destination,
                                                [attr_gen_items[0] for attr_gen_items in attr_gen_package]))

        query_initial_weights_dict = comm.alltoall([destination_gid]*nranks)
        syn_weight_dict = weights_dict_alltoall(comm, synapse_name, initial_weights_dict, query_initial_weights_dict, clear=True)
        
        if destination_gid is not None:

            logger.info('Rank %i; destination: %s; gid %i; received synaptic weights for %i synapses' %
                        (rank, destination, destination_gid, len(syn_weight_dict)))
            local_random.seed(int(destination_gid + seed_offset))
            for source, (this_destination_gid, (source_gid_array, conn_attr_dict)) in zip_longest(sources, attr_gen_package):
                syn_ids = conn_attr_dict['Synapses']['syn_id']
                this_source_syn_map = source_syn_map[source]
                for i in range(len(source_gid_array)):
                    this_source_gid = source_gid_array[i]
                    this_syn_id = syn_ids[i]
                    this_syn_wgt = syn_weight_dict[this_syn_id]
                    this_source_syn_map[this_source_gid].append((this_syn_id, this_syn_wgt))

        query_source_rate_map_dict = comm.alltoall([{source : list(source_syn_map[source].keys()) for source in sources}]*nranks)
        source_rate_map_dict = source_rate_map_dict_alltoall(comm, stimulus_attrs, query_source_rate_map_dict)

        if destination_gid is not None:
        
            if local_random.uniform() <= target_sparsity:
                modify_weights = True
            else:
                modify_weights = False
            for source in sources:
                peak_rate = peak_rate_dict[source]
                this_source_syn_map = source_syn_map[source]
                this_source_rate_map = source_rate_map_dict[source]
                if modify_weights:
                    source_syn_wgt_sum = {} 
                    for source_gid, source_syns in viewitems(this_source_syn_map):
                        wgt_sum = np.sum(np.asarray([ item[1] for item in source_syns ]))
                        source_syn_wgt_sum[source_gid] = wgt_sum
                    ordered_source_gids = sorted(viewitems(source_syn_wgt_sum), key=lambda item: item[1], reverse=True)
                    candidate_source_gids = [ item[0] for item in ordered_source_gids[:int(len(ordered_source_gids)*.2)] ]
                    candidate_locs = set([])
                    for this_source_gid in candidate_source_gids:
                        rate_map = this_source_rate_map[this_source_gid]
                        for index in np.nditer(np.where(rate_map >= np.median(rate_map))):
                            candidate_locs.add(d[index])
                    peak_loc = local_random.choice(list(candidate_locs))
                    this_plasticity_kernel = plasticity_kernel(d, peak_loc)
                else:
                    peak_loc = None
                    this_plasticity_kernel = None
                    
                for this_source_gid in this_source_rate_map:
                    rate_map = this_source_rate_map[this_source_gid]
                    peak_index = np.where(rate_map == np.max(rate_map))[0][0]
                    if modify_weights:
                        norm_rate_map = rate_map / peak_rate
                        this_plasticity_signal = (np.sum(np.multiply(norm_rate_map, this_plasticity_kernel)) * \
                                                 spatial_resolution) / max_plasticity_kernel_area
                        delta_weight = 2. * this_plasticity_signal
                    else:
                        delta_weight = 0.
                    for this_syn_id, this_syn_wgt in this_source_syn_map[this_source_gid]:
                        syn_peak_index_map[this_syn_id] = peak_index
                        if delta_weight >= 0.1:
                            modulated_inputs += 1
                        syn_weight_dict[this_syn_id] += delta_weight
            structured_weights_dict[destination_gid] = \
                {'syn_id': np.array(list(syn_peak_index_map.keys())).astype('uint32', copy=False),
                 synapse_name: np.array([syn_weight_dict[syn_id] for syn_id in syn_peak_index_map]).astype('float32',
                                                                                                      copy=False),
                 'peak index': np.array(list(syn_peak_index_map.values())).astype('uint32', copy=False),
                 'structured': np.array([(1 if modify_weights else 0)], dtype='uint8')}
            if modify_weights:
                logger.info('Rank %i; destination: %s; gid %i; generated structured weights for %i/%i inputs in %.2f '
                            's' % (rank, destination, destination_gid, modulated_inputs, len(syn_weight_dict),
                                   time.time() - local_time))
                structured_count += 1
            else:
                logger.info('Rank %i; destination: %s; gid %i; calculated input peak location for %i inputs in %.2f s (not'
                            ' selected for structured weights)' %
                            (rank, destination, destination_gid, len(syn_weight_dict), time.time() - local_time))
            count += 1
        else:
            logger.info('Rank: %i received None' % rank)
        gid_count += 1
        if gid_count % write_size == 0:
            if not dry_run:
                append_cell_attributes(output_weights_path, destination, structured_weights_dict,
                                       namespace=this_structured_weights_namespace, comm=env.comm, io_size=env.io_size,
                                       chunk_size=chunk_size, value_chunk_size=value_chunk_size)
            structured_weights_dict.clear()
        del syn_weight_dict
        del source_syn_map
        del syn_peak_index_map
        del source_gid_array
        del conn_attr_dict
        # gc.collect()
    if not dry_run:
        append_cell_attributes(output_weights_path, destination, structured_weights_dict,
                               namespace=this_structured_weights_namespace, comm=env.comm, io_size=env.io_size,
                               chunk_size=chunk_size, value_chunk_size=value_chunk_size)
    global_count = comm.gather(count, root=0)
    global_structured_count = comm.gather(structured_count, root=0)
    if rank == 0:
        logger.info('destination: %s; %i ranks processed %i cells (%i assigned structured weights) in %.2f s' %
                    (destination, comm.size, np.sum(global_count), np.sum(global_structured_count),
                     time.time() - start_time))
    MPI.Finalize()


if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv)+1):])
