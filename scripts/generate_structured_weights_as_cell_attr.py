import sys, os, time, gc, click, logging
from collections import defaultdict
from functools import reduce
import numpy as np
import dlib
import scipy.optimize as opt
from mpi4py import MPI
import neuroh5
from neuroh5.io import append_cell_attributes, read_population_ranges, bcast_cell_attributes, \
    scatter_read_cell_attributes, read_cell_attribute_selection, NeuroH5ProjectionGen
import dentate
from dentate.env import Env
from dentate import utils, stimulus
from dentate.utils import *
import h5py



sys_excepthook = sys.excepthook
def mpi_excepthook(type, value, traceback):
    sys_excepthook(type, value, traceback)
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)
sys.excepthook = mpi_excepthook

def interactive_callback_plasticity_fit(**kwargs):
    import matplotlib.pyplot as plt
    fig = plt.figure(constrained_layout=3)
    gs = fig.add_gridspec(4, 4)
    initial_weights = kwargs['initial_weights'] 
    modified_weights = kwargs['modified_weights']
    initial_ratemap = kwargs['initial_ratemap']
    modified_ratemap = kwargs['modified_ratemap']
    target_ratemap =  kwargs['target_ratemap']
    plasticity_kernel =  kwargs['plasticity_kernel']
    delta_weights =  kwargs['delta_weights']
    syn_ids = np.asarray(sorted(initial_weights.keys()))
    initial_weights_array = np.asarray([initial_weights[k] for k in syn_ids])
    hist_bins = np.linspace(np.min(initial_weights_array), np.max(initial_weights_array))
    initial_weights_hist, initial_weights_bin_edges = np.histogram(initial_weights_array, bins=hist_bins)
    modified_weights_array = np.asarray([modified_weights[k] for k in syn_ids])
    hist_bins = np.linspace(np.min(modified_weights_array), np.max(modified_weights_array))
    modified_weights_hist, modified_weights_bin_edges = np.histogram(modified_weights_array, bins=hist_bins)
    hist_bins = np.linspace(np.min(delta_weights), np.max(delta_weights))
    delta_weights_hist, delta_weights_bin_edges = np.histogram(delta_weights, bins=hist_bins)
    xlimits = (min(np.min(initial_weights_array), np.min(modified_weights_array)),
               max(np.max(initial_weights_array), np.max(modified_weights_array)))
    ax = fig.add_subplot(gs[0, :])
    ax.plot(initial_weights_bin_edges[:-1], initial_weights_hist, marker='.')
    ax.set(ylabel='number of synapses')
    ax.set_xlim(xlimits)
    ax.set_yscale('log')
    ax = fig.add_subplot(gs[1, :])
    ax.plot(modified_weights_bin_edges[:-1], modified_weights_hist, marker='.', color='r')
    ax.set(xlabel='synaptic weight', ylabel='number of synapses')
    ax.set_xlim(xlimits)
    ax.set_yscale('log')
    ax = fig.add_subplot(gs[2, :])
    ax.plot(delta_weights_bin_edges[:-1], delta_weights_hist, marker='.')
    ax.set_yscale('log')
    ax.set(xlabel='synaptic weight delta', ylabel='number of synapses')
    ax = fig.add_subplot(gs[3, 0])
    ax.set_title('Initial rate map')
    p = ax.imshow(initial_ratemap, origin='lower')
    fig.colorbar(p, ax=ax)
    ax = fig.add_subplot(gs[3, 1])
    ax.set_title('Modified rate map')
    p = ax.imshow(modified_ratemap, origin='lower')
    fig.colorbar(p, ax=ax)
    ax = fig.add_subplot(gs[3, 2])
    ax.set_title('Target')
    p = ax.imshow(target_ratemap, origin='lower')
    fig.colorbar(p, ax=ax)
    ax = fig.add_subplot(gs[3, 3])
    ax.set_title('Plasticity kernel')
    ax.imshow(plasticity_kernel, origin='lower')
    plt.show()

def syn_weights_dict_alltoall(comm, syn_name, initial_weights_dict, query, clear=False):
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


def input_features_dict_alltoall(comm, features_attrs, query, clear=False):
    rank = comm.rank
    send_source_gids = []
    send_source_features_attrs = []
    for source_dict in query:
        this_send_source_gids = {}
        this_send_source_features_attrs = {}
        for source in source_dict:
            this_send_source_gids[source] = []
            this_send_source_features_attrs[source] = []
            source_gid_dict = source_dict[source]
            for gid in source_gid_dict:
                if gid in features_attrs[source]:
                    this_features_attrs = features_attrs[source][gid]
                    this_send_source_gids[source].append(gid)
                    this_send_source_features_attrs[source].append(this_features_attrs)
                    if clear:
                        del(features_attrs[source][gid])
        send_source_gids.append(this_send_source_gids)
        send_source_features_attrs.append(this_send_source_features_attrs)
    recv_source_gids = comm.alltoall(send_source_gids)
    recv_source_features_attrs = comm.alltoall(send_source_features_attrs)
    source_features_attrs_dict = defaultdict(lambda: defaultdict(list))
    for source_gids, source_features_attrs in zip(recv_source_gids, recv_source_features_attrs):
        for source in source_gids:
            this_source_features_attrs_dict = source_features_attrs_dict[source]
            for (source_gid, features_attrs) in zip(source_gids[source], source_features_attrs[source]):
                this_source_features_attrs_dict[int(source_gid)] = features_attrs
    return source_features_attrs_dict


def plasticity_fit(phi, plasticity_kernel, plasticity_inputs, source_syn_map, logger, scaled_weight=False, max_iter=100, interactive=False):
    source_gids = sorted(plasticity_inputs.keys())
    initial_weights = []
    for i, source_gid in enumerate(source_gids):
        _, initial_weight = source_syn_map[source_gid][0]
        initial_weights.append(initial_weight)
    w = np.asarray(initial_weights, dtype=np.float64)
    A = np.column_stack([plasticity_inputs[gid].reshape((-1,)).astype(np.float64) for gid in source_gids])
    initial_ratemap = phi(np.dot(A, w))
    b = plasticity_kernel.reshape((-1,)).astype(np.float64)

    lb = -np.inf
    ub = 3.

    #res = opt.lsq_linear(A, b, bounds=(lb, ub), lsmr_tol='auto')
    def residual(x, w, A, b, phi):
        res = np.subtract(phi(np.dot(A, np.add(w, x))), b)
        return res

    x0 = np.random.rand(len(w),)
    optres = opt.least_squares(residual, x0, args = (w, A, b, phi), bounds=(lb,ub), xtol=1e-2, ftol=1e-2,
                                   method='dogbox', verbose=2 if interactive else 0)
    
    delta_weights = optres.x
    logger.info('Least squares fit status: %d (%s)' % (optres.status, optres.message))
    logger.info('Delta weights: min: %f max: %f' % (np.min(delta_weights), np.max(delta_weights)))
        
    syn_weights = {}
    for source_gid, delta_weight in zip(source_gids, delta_weights):
        syn_count = len(source_syn_map[source_gid])
        if syn_count > 0:
            if scaled_weight:
                delta_weight = delta_weight / float(syn_count)
            for syn_id, initial_weight in source_syn_map[source_gid]:
                syn_weights[syn_id] = max(delta_weight + initial_weight, 0.)

    if interactive:
        modified_weights = np.clip(np.add(w, delta_weights), 0., None)
        modified_ratemap = phi(np.dot(A, modified_weights))
        logger.info('Initial rate map: min: %f max: %f' % (np.min(initial_ratemap), np.max(initial_ratemap)))
        logger.info('Target: min: %f max: %f' % (np.min(b), np.max(b)))
        logger.info('Initial weights: min: %f max: %f' % (np.min(w), np.max(w)))
        logger.info('Modified weights: min: %f max: %f' % (np.min(modified_weights), np.max(modified_weights)))
        logger.info('Modified rate map: min: %f max: %f' % (np.min(modified_ratemap), np.max(modified_ratemap)))
        initial_weights_dict = {}
        for i, source_gid in enumerate(source_gids):
            syn_id, initial_weight = source_syn_map[source_gid][0]
            initial_weights_dict[syn_id] = initial_weight
        interactive_callback_plasticity_fit(initial_weights=initial_weights_dict,
                                            modified_weights=syn_weights,
                                            delta_weights=delta_weights,
                                            initial_ratemap=initial_ratemap.reshape(plasticity_kernel.shape),
                                            modified_ratemap=modified_ratemap.reshape(plasticity_kernel.shape),
                                            plasticity_kernel=plasticity_kernel,
                                            target_ratemap=b.reshape(plasticity_kernel.shape))
        

    return syn_weights

def gauss2d(x=0, y=0, mx=0, my=0, sx=1, sy=1, A=1.):
    return A * np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))


def sigmoid_phi(x, a = 0.05, peak_rate = 1.):
    res = peak_rate / (1. + np.exp(-a * x))
    return res

def exp_phi(x, a = 0.025):
    res = 1. / np.exp(-a * x)
    return res
    

@click.command()
@click.option("--config", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--input-features-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--input-features-namespaces", type=str, multiple=True, default=['Place Selectivity', 'Grid Selectivity'])
@click.option("--output-weights-path", required=False, type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option("--weights-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--synapse-name", type=str, default='AMPA')
@click.option("--initial-weights-namespace", type=str, default='Weights')
@click.option("--structured-weights-namespace", type=str, default='Structured Weights')
@click.option("--connections-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--destination", '-d', type=str)
@click.option("--sources", '-s', type=str, multiple=True)
@click.option("--arena-id", '-a', type=str, default='A')
@click.option("--max-iter", type=int, default=100)
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--cache-size", type=int, default=50)
@click.option("--write-size", type=int, default=1)
@click.option("--scatter-io", is_flag=True)
@click.option("--verbose", "-v", is_flag=True)
@click.option("--dry-run", is_flag=True)
@click.option("--interactive", is_flag=True)
def main(config, input_features_path, input_features_namespaces, output_weights_path, weights_path, synapse_name, initial_weights_namespace,
         structured_weights_namespace, connections_path, destination, sources, arena_id, max_iter, 
         io_size, chunk_size, value_chunk_size, cache_size, write_size, scatter_io, verbose, dry_run, interactive):
    """

    :param config: str (path to .yaml file)
    :param input_features_path: str (path to .h5 file)
    :param weights_path: str (path to .h5 file)
    :param initial_weights_namespace: str
    :param structured_weights_namespace: str
    :param connections_path: str (path to .h5 file)
    :param destination: str
    :param sources: list of str
    :param io_size:
    :param chunk_size:
    :param value_chunk_size:
    :param cache_size:
    :param write_size:
    :param verbose:
    :param dry_run:
    :return:
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


    this_structured_weights_namespace = '%s %s' % (structured_weights_namespace, arena_id)
    this_input_features_namespaces = ['%s %s' % (input_features_namespace, arena_id) for input_features_namespace in input_features_namespaces]


    initial_weights_dict = None
    if scatter_io:
        if rank == 0:
            logger.info('Reading initial weights data from %s...' % weights_path)
        env.comm.barrier()
        cell_attributes_dict = scatter_read_cell_attributes(weights_path, destination, 
                                                            namespaces=[initial_weights_namespace], 
                                                            comm=env.comm, io_size=env.io_size)
        if initial_weights_namespace in cell_attributes_dict:
            initial_weights_iter = cell_attributes_dict[initial_weights_namespace]
            initial_weights_dict = { gid: attr_dict for gid, attr_dict in initial_weights_iter }
        else:
            raise RuntimeError('Initial weights namespace %s was not found in file %s' % (initial_weights_namespace, weights_path))
    
        logger.info('Rank %i; destination: %s; read synaptic weights for %i cells' %
                    (rank, destination, len(initial_weights_dict)))


    features_attrs = defaultdict(dict)
    features_attr_names = ['Num Fields', 'Field Width', 'Peak Rate', 'X Offset', 'Y Offset', 'Arena Rate Map']
        
    if scatter_io:
        if rank == 0:
            logger.info('Reading selectivity features data from %s...' % input_features_path)
        env.comm.barrier()
        
        features_attr_dict = scatter_read_cell_attributes(input_features_path, destination, namespaces=this_input_features_namespaces, \
                                                          mask=set(features_attr_names), io_size=env.io_size, comm=env.comm)
        for this_features_namespace in this_input_features_namespaces:
            count = 0
            if this_features_namespace in features_attr_dict:
                destination_features_attrs = features_attrs[destination]
                for gid, attr_dict in features_attr_dict[this_features_namespace]:
                    destination_features_attrs[gid] = attr_dict
                    count += 1
                if rank == 0:
                    logger.info('Read %s feature data for %i cells in population %s' % (this_features_namespace, count, destination))

        for source in sources:
            features_attr_dict = scatter_read_cell_attributes(input_features_path, source, namespaces=this_input_features_namespaces, \
                                                              mask=set(features_attr_names), io_size=env.io_size, comm=env.comm)
            for this_features_namespace in this_input_features_namespaces:
                if this_features_namespace in features_attr_dict:
                    count = 0
                    source_features_attrs = features_attrs[source]
                    for gid, attr_dict in features_attr_dict[this_features_namespace]:
                        source_features_attrs[gid] = attr_dict
                        count += 1
                    if rank == 0:
                        logger.info('Read %s feature data for %i cells in population %s' % (this_features_namespace, count, source))


    local_random = np.random.RandomState()

    seed_offset = int(env.modelConfig['Random Seeds']['GC Structured Weights'])
    spatial_resolution = env.stimulus_config['Spatial Resolution'] # cm

    arena = env.stimulus_config['Arena'][arena_id]
    default_run_vel = arena.properties['default run velocity']  # cm/s

    x, y = stimulus.get_2D_arena_spatial_mesh(arena, spatial_resolution)
    
    plasticity_kernel = lambda x, y, x_loc, y_loc, sx, sy: gauss2d(x-x_loc, y-y_loc, sx=sx, sy=sy)
    plasticity_kernel = np.vectorize(plasticity_kernel, excluded=[2,3,4,5])
    gid_count = 0
    start_time = time.time()

    connection_gen_list = [ NeuroH5ProjectionGen(connections_path, source, destination, namespaces=['Synapses'], comm=comm) \
                               for source in sources ]

    structured_weights_dict = {}
    for iter_count, attr_gen_package in enumerate(zip_longest(*connection_gen_list)):
        
        local_time = time.time()
        source_syn_map = defaultdict(lambda: defaultdict(list))
        syn_weight_dict = {}
        destination_gid = attr_gen_package[0][0]
        if not all([attr_gen_items[0] == destination_gid for attr_gen_items in attr_gen_package]):
            raise Exception('Rank: %i; destination: %s; destination_gid not matched across multiple attribute '
                            'generators: %s' % (rank, destination,
                                                [attr_gen_items[0] for attr_gen_items in attr_gen_package]))

        if scatter_io:
            query_initial_weights_dict = comm.alltoall([destination_gid]*nranks)
            syn_weight_dict = syn_weights_dict_alltoall(comm, synapse_name, initial_weights_dict, query_initial_weights_dict, clear=True)
        else:
            if destination_gid is None:
                selection=[]
            else:
                selection=[destination_gid]
            initial_weights_iter = read_cell_attribute_selection(weights_path, destination, 
                                                                 namespace=initial_weights_namespace, 
                                                                 comm=env.comm, selection=selection)
            syn_weight_attr_dict = dict(initial_weights_iter)

            if destination_gid is not None:
                syn_ids = syn_weight_attr_dict[destination_gid]['syn_id']
                weights = syn_weight_attr_dict[destination_gid][synapse_name]

                for (syn_id, weight) in zip(syn_ids, weights):
                    syn_weight_dict[int(syn_id)] = float(weight) 

        if destination_gid is not None:

            logger.info('Rank %i; destination: %s; gid %i; received synaptic weights for %i synapses' %
                        (rank, destination, destination_gid, len(syn_weight_dict)))
            local_random.seed(int(destination_gid + seed_offset))
            for source, (this_destination_gid, (source_gid_array, conn_attr_dict)) in zip_longest(sources, attr_gen_package):
                syn_ids = conn_attr_dict['Synapses']['syn_id']
                this_source_syn_map = source_syn_map[source]
                count = 0
                for i in range(len(source_gid_array)):
                    this_source_gid = source_gid_array[i]
                    this_syn_id = syn_ids[i]
                    this_syn_wgt = syn_weight_dict[this_syn_id]
                    this_source_syn_map[this_source_gid].append((this_syn_id, this_syn_wgt))
                    count += 1
                logger.info('Rank %i; destination: %s; gid %i; %d synaptic weights from source population %s' %
                                (rank, destination, destination_gid, count, source))
                    
        if scatter_io:
            query_source_input_features = comm.alltoall([{source : list(source_syn_map[source].keys()) for source in sources}]*nranks)
            source_input_features = input_features_dict_alltoall(comm, features_attrs, query_source_input_features)
            query_dest_input_features = comm.alltoall([{destination : [destination_gid]}]*nranks)
            dest_input_features = input_features_dict_alltoall(comm, features_attrs, query_dest_input_features)
        else:
            if destination_gid is None:
                selection=[]
            else:
                selection=[destination_gid]
            dest_input_features = defaultdict(dict)
            source_input_features = defaultdict(dict)
            for input_features_namespace in this_input_features_namespaces:
                input_features_iter = read_cell_attribute_selection(input_features_path, destination, 
                                                                    namespace=input_features_namespace,
                                                                    mask=set(features_attr_names), 
                                                                    comm=env.comm, selection=selection)
                count = 0
                for gid, attr_dict in input_features_iter:
                    dest_input_features[destination][gid] = attr_dict
                    count += 1
                if rank == 0:
                    logger.info('Read %s feature data for %i cells in population %s' % (input_features_namespace, count, destination))
            for source in sources:
                if len(dest_input_features[destination]) > 0:
                    source_gids = list(source_syn_map[source].keys())
                else:
                    source_gids = []
                for input_features_namespace in this_input_features_namespaces:
                    input_features_iter = read_cell_attribute_selection(input_features_path, source, 
                                                                        namespace=input_features_namespace,
                                                                        mask=set(features_attr_names), 
                                                                        comm=env.comm, selection=source_gids)
                    this_source_input_features = source_input_features[source]
                    count = 0
                    for gid, attr_dict in input_features_iter:
                        this_source_input_features[gid] = attr_dict
                        count += 1
                    if rank == 0:
                        logger.info('Read %s feature data for %i cells in population %s' % (input_features_namespace, count, source))

        if destination_gid is not None:

            if destination_gid in dest_input_features[destination]:
                this_input_features = dest_input_features[destination][destination_gid]
                this_num_fields = this_input_features['Num Fields'][0]
                this_field_width = this_input_features['Field Width']
                this_x_offset = this_input_features['X Offset']
                this_y_offset = this_input_features['Y Offset']
                this_peak_rate = this_input_features['Peak Rate']
            else:
                this_input_features = None
                this_num_fields = None
                this_field_width = None
                this_x_offset = None
                this_y_offset = None
                this_peak_rate = None

            structured = False
            if (this_input_features is not None) and this_num_fields > 0:
                structured = True
                    
            if structured:
                this_peak_locs = zip(np.nditer(this_x_offset), np.nditer(this_y_offset))
                this_sigmas    = [width / 3. / np.sqrt(2.) for width in np.nditer(this_field_width)] # cm
                this_plasticity_kernel = reduce(np.add, [plasticity_kernel(x, y, peak_loc[0], peak_loc[1], sigma, sigma)
                                                             for peak_loc, sigma in zip(this_peak_locs, this_sigmas)]) * this_peak_rate

                this_plasticity_inputs = {}
                source_gid_syn_map = {}
                for source in sources:
                    for source_gid in source_input_features[source]:
                        source_gid_syn_idwgts = source_syn_map[source][source_gid]
                        source_gid_syn_map[source_gid] = source_gid_syn_idwgts
                        source_gid_input_features = source_input_features[source][source_gid]
                        source_peak_rate = source_gid_input_features['Peak Rate']
                        source_rate_map = source_gid_input_features['Arena Rate Map']
                        norm_rate_map = source_rate_map / source_peak_rate
                        this_plasticity_inputs[source_gid] = norm_rate_map

                plasticity_source_syn_map = {}
                for source in sources:
                    for source_gid in source_syn_map[source]:
                        plasticity_source_syn_map[source_gid] = source_syn_map[source][source_gid]
                        
                this_peak_locs = zip(np.nditer(this_x_offset), np.nditer(this_y_offset))
                logger.info('computing plasticity fit for gid %d: peak locs: %s field widths: %s' %
                                (destination_gid, str([x for x in this_peak_locs]), str(this_field_width)))
                this_syn_weights = plasticity_fit(exp_phi, this_plasticity_kernel, this_plasticity_inputs,
                                                  plasticity_source_syn_map, logger, max_iter=max_iter, interactive=interactive)

                this_syn_ids = sorted(this_syn_weights.keys())
            
                structured_weights_dict[destination_gid] = \
                    {'syn_id': np.array(this_syn_ids).astype('uint32', copy=False),
                     synapse_name: np.array([this_syn_weights[syn_id]
                                         for syn_id in this_syn_ids]).astype('float32', copy=False) }
                logger.info('Rank %i; destination: %s; gid %i; generated structured weights for %i inputs in %.2f '
                        's' % (rank, destination, destination_gid, len(this_syn_weights),
                               time.time() - local_time))
                gid_count += 1
        else:
            logger.info('Rank: %i received None' % rank)

        if iter_count % write_size == 0:
            if not dry_run:
                append_cell_attributes(output_weights_path, destination, structured_weights_dict,
                                       namespace=this_structured_weights_namespace, comm=env.comm, io_size=env.io_size,
                                       chunk_size=chunk_size, value_chunk_size=value_chunk_size)
                count = comm.reduce(len(structured_weights_dict), op=MPI.SUM, root=0)
                if rank == 0:
                    logger.info('Destination: %s; appended structured weights for %i cells' % (destination, count))
            structured_weights_dict.clear()
            gc.collect()
        del(syn_weight_dict)
        del(source_input_features)
        del(dest_input_features)

    if not dry_run:
        append_cell_attributes(output_weights_path, destination, structured_weights_dict,
                               namespace=this_structured_weights_namespace, comm=env.comm, io_size=env.io_size,
                               chunk_size=chunk_size, value_chunk_size=value_chunk_size)
        if rank == 0:
            count = comm.reduce(len(structured_weights_dict), op=MPI.SUM, root=0)
            logger.info('Destination: %s; appended structured weights for %i cells' % (destination, count))

    global_count = comm.gather(gid_count, root=0)
    if rank == 0:
        logger.info('destination: %s; %i ranks assigned structured weights to %i cells in %.2f s' %
                    (destination, comm.size, np.sum(global_count), time.time() - start_time))

    MPI.Finalize()


if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv)+1):])
