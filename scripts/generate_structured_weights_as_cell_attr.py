
import sys, os, time, gc, click, logging, pprint
from collections import defaultdict
import numpy as np
from mpi4py import MPI
import neuroh5
from neuroh5.io import append_cell_attributes, read_population_ranges, \
    scatter_read_cell_attribute_selection, scatter_read_cell_attributes, \
    scatter_read_graph_selection, read_graph_info
import dentate
from dentate.env import Env
from dentate import utils, stimulus, synapses
from dentate.utils import Context, is_interactive, viewitems, zip_longest
import h5py

context = Context()

sys_excepthook = sys.excepthook
def mpi_excepthook(type, value, traceback):
    sys_excepthook(type, value, traceback)
    sys.stdout.flush()
    sys.stderr.flush()
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)
sys.excepthook = mpi_excepthook


def read_weights(weights_path, weights_namespace, synapse_name, destination, selection, comm, io_size, 
                 weights_by_syn_id_dict, logger=None):

    if logger is not None:
        logger.info("reading weights from namespace %s..." % weights_namespace)
    if weights_path is not None:
        weights_iter = \
          scatter_read_cell_attribute_selection(weights_path, destination,
                                                namespace=weights_namespace,
                                                selection=selection, 
                                                comm=comm, io_size=io_size)

        weights_gid_count = 0
        weights_syn_count = 0
        for this_gid, syn_weight_attr_dict in weights_iter:

            syn_ids = syn_weight_attr_dict['syn_id']
            weights = syn_weight_attr_dict[synapse_name]

            for (syn_id, weight) in zip(syn_ids, weights):
                weights_by_syn_id_dict[this_gid][int(syn_id)] = float(weight)
            weights_gid_count += 1
            weights_syn_count += len(syn_ids)

        if logger is not None:
            logger.info('destination: %s; read initial synaptic weights for %i gids and %i syns' %
                        (destination, weights_gid_count, weights_syn_count))

    return weights_by_syn_id_dict


def init_selectivity_config(destination_gid, spatial_resolution, 
                            arena, arena_margin, arena_margin_size, 
                            coordinates, field_width, field_width_scale, peak_rate, 
                            target_selectivity_type, selectivity_type_index,
                            dst_input_features_attr_dict, target_selectivity_features_dict,
                            target_selectivity_config_dict, target_field_width_dict):


    this_target_selectivity_features_dict = dst_input_features_attr_dict.get(destination_gid, {})
    this_target_selectivity_features_dict['Selectivity Type'] = np.asarray([target_selectivity_type], dtype=np.uint8)

    if len(coordinates) > 0:
        num_fields = len(coordinates)
        this_target_selectivity_features_dict['X Offset'] =  np.asarray([x[0] for x in coordinates],
                                                                        dtype=np.float32)
        this_target_selectivity_features_dict['Y Offset'] =  np.asarray([x[1] for x in coordinates],
                                                                        dtype=np.float32)
        this_target_selectivity_features_dict['Num Fields'] = np.asarray([num_fields], dtype=np.uint8)
    elif 'Num Fields' in this_target_selectivity_features_dict:
        num_fields = this_target_selectivity_features_dict['Num Fields'][0]
    else:
        num_fields = 0

    if field_width is not None:
        this_target_selectivity_features_dict['Field Width'] = np.asarray([field_width]*num_fields, dtype=np.float32)
    elif 'Field Width' in this_target_selectivity_features_dict:
        this_field_width = this_target_selectivity_features_dict['Field Width']
        this_target_selectivity_features_dict['Field Width'] = this_field_width[:num_fields]
    else:
        this_field_width = np.asarray([], dtype=np.float32)

    if peak_rate is not None:
        this_target_selectivity_features_dict['Peak Rate'] = np.asarray([peak_rate]*num_fields, dtype=np.float32)

    if num_fields > 0:
        input_cell_config = stimulus.get_input_cell_config(target_selectivity_type,
                                                           selectivity_type_index,
                                                           selectivity_attr_dict=this_target_selectivity_features_dict)
        arena_margin_size = max(arena_margin_size, np.max(input_cell_config.field_width) * arena_margin)
        
        arena_x, arena_y = stimulus.get_2D_arena_spatial_mesh(arena, spatial_resolution,
                                                              margin=arena_margin_size)

        target_map = np.asarray(input_cell_config.get_rate_map(arena_x, arena_y,
                                                               scale=field_width_scale),
                                dtype=np.float32)
        this_target_selectivity_features_dict['Arena Rate Map'] = target_map
        target_selectivity_features_dict[destination_gid] = this_target_selectivity_features_dict
        target_field_width_dict[destination_gid] = input_cell_config.field_width
        target_selectivity_config_dict[destination_gid] = input_cell_config

    return arena_margin_size


def init_syn_weight_dicts(destination, non_structured_sources,
                          edge_iter_dict, edge_attr_info,
                          initial_weights_by_syn_id_dict,
                          initial_weights_by_source_gid_dict,
                          non_structured_weights_by_syn_id_dict,
                          non_structured_weights_by_source_gid_dict,
                          reference_weights_by_syn_id_dict,
                          reference_weights_by_source_gid_dict,
                          source_gid_set_dict,
                          syn_count_by_source_gid_dict,
                          syn_ids_by_source_gid_dict,
                          structured_syn_id_count):

    syn_counts_by_source = {}

    if destination not in edge_iter_dict:
        return syn_counts_by_source

    for source, edge_iter in viewitems(edge_iter_dict[destination]):

        syn_counts_by_source[source] = {}
        syn_id_attr_index = edge_attr_info[destination][source]['Synapses']['syn_id']

        for this_gid, edges in edge_iter:
            (source_gid_array, edge_attr_dict) = edges
            syn_ids = edge_attr_dict['Synapses'][syn_id_attr_index]
            this_initial_weights_by_syn_id_dict = initial_weights_by_syn_id_dict[this_gid]
            this_initial_weights_by_source_gid_dict = initial_weights_by_source_gid_dict[this_gid]
            this_non_structured_weights_by_syn_id_dict = non_structured_weights_by_syn_id_dict[this_gid]
            this_non_structured_weights_by_source_gid_dict = non_structured_weights_by_source_gid_dict[this_gid]
            this_syn_ids_by_source_gid_dict = syn_ids_by_source_gid_dict[this_gid]
            this_syn_count_by_source_gid_dict = syn_count_by_source_gid_dict[this_gid]
            this_reference_weights_by_syn_id_dict = None
            this_reference_weights_by_source_gid_dict = None
            if reference_weights_by_syn_id_dict is not None:
                this_reference_weights_by_syn_id_dict = reference_weights_by_syn_id_dict[this_gid]
                this_reference_weights_by_source_gid_dict = reference_weights_by_source_gid_dict[this_gid]

            count = 0
            for i in range(len(source_gid_array)):
                this_source_gid = source_gid_array[i]
                this_syn_id = syn_ids[i]
                if this_syn_id in this_initial_weights_by_syn_id_dict:
                    this_syn_wgt = this_initial_weights_by_syn_id_dict[this_syn_id]
                    if this_source_gid not in this_initial_weights_by_source_gid_dict:
                        this_initial_weights_by_source_gid_dict[this_source_gid] = this_syn_wgt
                    if this_reference_weights_by_syn_id_dict is not None:
                        this_reference_weights_by_source_gid_dict[this_source_gid] = \
                          this_reference_weights_by_syn_id_dict[this_syn_id]
                elif this_syn_id in this_non_structured_weights_by_syn_id_dict:
                    this_syn_wgt = this_non_structured_weights_by_syn_id_dict[this_syn_id]
                    if this_source_gid not in this_non_structured_weights_by_source_gid_dict:
                        this_non_structured_weights_by_source_gid_dict[this_source_gid] = this_syn_wgt
                this_syn_ids_by_source_gid_dict[this_source_gid].append(this_syn_id)
                this_syn_count_by_source_gid_dict[this_source_gid] += 1

                source_gid_set_dict[source].add(this_source_gid)

                count += 1

            syn_counts_by_source[source][this_gid] = count

            if source not in non_structured_sources:
                structured_syn_id_count[this_gid] += len(syn_ids)

    return syn_counts_by_source
        
@click.command()
@click.option("--config", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--coordinates", '-c', type=(float, float), multiple=True)
@click.option("--field-width", type=float)
@click.option("--gid", '-g', type=int, multiple=True)
@click.option("--input-features-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--input-features-namespaces", type=str, multiple=True, default=['Place Selectivity', 'Grid Selectivity'])
@click.option("--initial-weights-path", required=False, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--output-features-namespace", required=False, type=str)
@click.option("--output-features-path", required=False, type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option("--output-weights-path", required=True, type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option("--reference-weights-path", required=False, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--h5types-path", required=False, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--synapse-name", type=str, default='AMPA')
@click.option("--initial-weights-namespace", type=str, default='Weights')
@click.option("--output-weights-namespace", type=str, default='Structured Weights')
@click.option("--reference-weights-namespace", type=str, default='Weights')
@click.option("--reference-weights-are-delta", type=bool, default=False)
@click.option("--connections-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--destination", '-d', type=str)
@click.option("--sources", '-s', type=str, multiple=True)
@click.option("--non-structured-sources", '-n', type=str, multiple=True)
@click.option("--non-structured-weights-namespace", type=str, default='Weights')
@click.option("--non-structured-weights-path", required=False, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--arena-id", '-a', type=str, default='A')
@click.option("--field-width-scale", type=float, default=1.25)
@click.option("--max-delta-weight", type=float, default=4.)
@click.option("--max-opt-iter", type=int, default=1000)
@click.option("--max-weight-decay-fraction", type=float, default=1.)
@click.option("--optimize-method", type=str, default='L-BFGS-B')
@click.option("--optimize-tol", type=float, default=1e-4)
@click.option("--optimize-grad", is_flag=True)
@click.option("--peak-rate", type=float)
@click.option("--reference-weights-are-delta", type=bool, default=False)
@click.option("--target-amplitude", type=float, default=3.)
@click.option("--arena-margin", type=float, default=0.)
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--cache-size", type=int, default=1)
@click.option("--write-size", type=int, default=1)
@click.option("--verbose", "-v", is_flag=True)
@click.option("--dry-run", is_flag=True)
@click.option("--plot", is_flag=True)
@click.option("--show-fig", is_flag=True)
@click.option("--save-fig", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--debug", is_flag=True)
def main(config, coordinates, field_width, gid, input_features_path, input_features_namespaces, initial_weights_path, output_features_namespace, output_features_path, output_weights_path, reference_weights_path, h5types_path, synapse_name, initial_weights_namespace, output_weights_namespace, reference_weights_namespace, connections_path, destination, sources, non_structured_sources, non_structured_weights_namespace, non_structured_weights_path, arena_id, field_width_scale, max_delta_weight, max_opt_iter, max_weight_decay_fraction, optimize_method, optimize_tol, optimize_grad, peak_rate, reference_weights_are_delta, arena_margin, target_amplitude, io_size, chunk_size, value_chunk_size, cache_size, write_size, verbose, dry_run, plot, show_fig, save_fig, debug):
    """

    :param config: str (path to .yaml file)
    :param input_features_path: str (path to .h5 file)
    :param initial_weights_path: str (path to .h5 file)
    :param initial_weights_namespace: str
    :param output_weights_namespace: str
    :param connections_path: str (path to .h5 file)
    :param destination: str
    :param sources: list of str
    :param io_size:
    :param chunk_size:
    :param value_chunk_size:
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

    if plot and (not save_fig) and (not show_fig):
        show_fig = True

    if (not dry_run) and (rank==0):
        if not os.path.isfile(output_weights_path):
            if initial_weights_path is not None:
                input_file  = h5py.File(initial_weights_path,'r')
            elif h5types_path is not None:
                input_file  = h5py.File(h5types_path,'r')
            else:
                raise RuntimeError('h5types input path must be specified when weights path is not specified.')
            output_file = h5py.File(output_weights_path,'w')
            input_file.copy('/H5Types',output_file)
            input_file.close()
            output_file.close()
    env.comm.barrier()

    LTD_output_weights_namespace = 'LTD %s %s' % (output_weights_namespace, arena_id)
    LTP_output_weights_namespace = 'LTP %s %s' % (output_weights_namespace, arena_id)
    this_input_features_namespaces = ['%s %s' % (input_features_namespace, arena_id) for input_features_namespace in input_features_namespaces]

    selectivity_type_index = { i: n for n, i in viewitems(env.selectivity_types) }
    target_selectivity_type_name = 'place'
    target_selectivity_type = env.selectivity_types[target_selectivity_type_name]
    features_attrs = defaultdict(dict)
    source_features_attr_names = ['Selectivity Type', 'Num Fields', 'Field Width', 'Peak Rate',
                                  'Module ID', 'Grid Spacing', 'Grid Orientation', 'Field Width Concentration Factor', 
                                  'X Offset', 'Y Offset']
    target_features_attr_names = ['Selectivity Type', 'Num Fields', 'Field Width', 'Peak Rate', 
                                  'X Offset', 'Y Offset']

    seed_offset = int(env.model_config['Random Seeds']['GC Structured Weights'])
    spatial_resolution = env.stimulus_config['Spatial Resolution'] # cm

    arena = env.stimulus_config['Arena'][arena_id]
    default_run_vel = arena.properties['default run velocity']  # cm/s

    gid_count = 0
    start_time = time.time()

    target_gid_set = None
    if len(gid) > 0:
        target_gid_set = set(gid)
    projections = [ (source, destination) for source in sources ]
    graph_info = read_graph_info(connections_path, namespaces=['Connections', 'Synapses'], read_node_index=True)
    for projection in projections:
        if projection not in graph_info:
            raise RuntimeError('Projection %s -> %s is not present in connections file.' % projection)
        if target_gid_set is None:
            target_gid_set = set(graph_info[projection][1])
        
    all_sources = sources + non_structured_sources
    dst_gids = []
    if target_gid_set is not None:
        for i, gid in enumerate(target_gid_set):
            if i%nranks == rank:
                dst_gids.append(gid)

    dst_input_features_attr_dict = {}
    for this_input_features_namespace in this_input_features_namespaces:
        feature_count = 0
        gid_count = 0
        logger.info('Rank %d: reading %s feature data for population %s' % (rank, this_input_features_namespace, destination))
        input_features_iter = scatter_read_cell_attribute_selection(input_features_path, destination, 
                                                                    namespace=this_input_features_namespace,
                                                                    mask=set(target_features_attr_names),
                                                                    selection=dst_gids,
                                                                    io_size=env.io_size, comm=env.comm)
        for gid, attr_dict in input_features_iter:
            gid_count += 1
            if (len(coordinates) > 0) or (attr_dict['Num Fields'][0] > 0):
                dst_input_features_attr_dict[gid] = attr_dict
                feature_count += 1

        logger.info('Rank %d: read %s feature data for %i / %i cells in population %s' % (rank, this_input_features_namespace, gid_count, feature_count, destination))
        feature_count = env.comm.reduce(feature_count, op=MPI.SUM, root=0)
        env.comm.barrier()
        if rank == 0:
            logger.info('Read %s feature data for %i cells in population %s' % (this_input_features_namespace, feature_count, destination))

    dst_gids = list(dst_input_features_attr_dict.keys())
    all_dst_gids = env.comm.allgather(dst_gids)
    all_dst_gids = sorted([item for sublist in all_dst_gids for item in sublist])
    dst_gid_rank_map = {}
    for i, gid in enumerate(all_dst_gids):
        this_rank = i%nranks
        dst_gid_rank_map[gid] = this_rank

    dst_input_features_sendbuf = [list() for i in range(nranks)]
    for gid, features_dict in viewitems(dst_input_features_attr_dict):
        dst_input_features_sendbuf[dst_gid_rank_map[gid]].append((gid, features_dict))

    dst_input_features_recvbuf = env.comm.alltoall(dst_input_features_sendbuf)
    env.comm.barrier()
    dst_input_features_attr_dict.clear()
    for l in dst_input_features_recvbuf:
        for gid, features_dict in l:
            dst_input_features_attr_dict[gid] = features_dict
    dst_gids = list(dst_input_features_attr_dict.keys())

    dst_count = env.comm.reduce(len(dst_gids), op=MPI.SUM, root=0)
    env.comm.barrier()

    logger.info("Rank %d has %d destination gids" % (rank, len(dst_gids)))
    if rank == 0:
        logger.info('Total %d destination gids' % dst_count)

    max_dst_count = env.comm.allreduce(len(dst_gids), op=MPI.MAX)
    env.comm.barrier()

    max_iter_count = max_dst_count
    output_features_dict = {}
    LTP_output_weights_dict = {}
    LTD_output_weights_dict = {}
    for iter_count in range(max_iter_count):

        gc.collect()

        local_time = time.time()
        selection = []
        if  len(dst_gids) > 0:
            dst_gid = dst_gids.pop()
            selection.append(dst_gid)
            logger.info('Rank %i received %d' % (rank, dst_gid))

        arena_margin_size = 0.
        arena_margin = max(arena_margin, 0.)

        target_selectivity_features_dict = {}
        target_selectivity_config_dict = {}
        target_field_width_dict = {}

        for destination_gid in selection:
            arena_margin_size = init_selectivity_config(destination_gid, 
                                                        spatial_resolution,
                                                        arena, arena_margin, arena_margin_size,
                                                        coordinates, field_width, field_width_scale, peak_rate,
                                                        target_selectivity_type,
                                                        selectivity_type_index,
                                                        dst_input_features_attr_dict, 
                                                        target_selectivity_features_dict,
                                                        target_selectivity_config_dict,
                                                        target_field_width_dict)
 
        arena_x, arena_y = stimulus.get_2D_arena_spatial_mesh(arena, spatial_resolution,
                                                              margin=arena_margin_size)

        selection = list(target_selectivity_features_dict.keys())

        initial_weights_by_source_gid_dict = defaultdict(lambda: dict())
        initial_weights_by_syn_id_dict = \
          read_weights(initial_weights_path, initial_weights_namespace, synapse_name,
                       destination, selection, env.comm, env.io_size, defaultdict(lambda: dict()), 
                       logger=logger if rank == 0 else None)

        non_structured_weights_by_source_gid_dict = defaultdict(lambda: dict())
        non_structured_weights_by_syn_id_dict = None
        if len(non_structured_sources) > 0:
            non_structured_weights_by_syn_id_dict = \
             read_weights(non_structured_weights_path, non_structured_weights_namespace, synapse_name,
                          destination, selection, env.comm, env.io_size, defaultdict(lambda: dict()),
                          logger=logger if rank == 0 else None)

            
        reference_weights_by_syn_id_dict = None
        reference_weights_by_source_gid_dict = defaultdict(lambda: dict())
        if reference_weights_path is not None:
            reference_weights_by_syn_id_dict = \
             read_weights(reference_weights_path, reference_weights_namespace, synapse_name,
                          destination, selection, env.comm, env.io_size, defaultdict(lambda: dict()),
                          logger=logger if rank == 0 else None)

        source_gid_set_dict = defaultdict(set)
        syn_count_by_source_gid_dict = defaultdict(lambda: defaultdict(int))
        syn_ids_by_source_gid_dict = defaultdict(lambda: defaultdict(list))
        structured_syn_id_count = defaultdict(int)


        projections = [ (source, destination) for source in all_sources ]
        edge_iter_dict, edge_attr_info = scatter_read_graph_selection(connections_path, selection=selection,
                                                                      namespaces=['Synapses'], 
                                                                      projections=projections,
                                                                      comm=env.comm, io_size=env.io_size)

        syn_counts_by_source = init_syn_weight_dicts(destination, non_structured_sources,
                                                     edge_iter_dict, edge_attr_info,
                                                     initial_weights_by_syn_id_dict,
                                                     initial_weights_by_source_gid_dict,
                                                     non_structured_weights_by_syn_id_dict,
                                                     non_structured_weights_by_source_gid_dict,
                                                     reference_weights_by_syn_id_dict,
                                                     reference_weights_by_source_gid_dict,
                                                     source_gid_set_dict,
                                                     syn_count_by_source_gid_dict,
                                                     syn_ids_by_source_gid_dict,
                                                     structured_syn_id_count)
    
        for source in syn_counts_by_source:
            for this_gid in syn_counts_by_source[source]:
                count = syn_counts_by_source[source][this_gid]
                logger.info('Rank %i; destination: %s; gid %i; %d edges from source population %s' %
                            (rank, destination, this_gid, count, source))

        input_rate_maps_by_source_gid_dict = {}
        if len(non_structured_sources) > 0:
            non_structured_input_rate_maps_by_source_gid_dict = {}
        else:
            non_structured_input_rate_maps_by_source_gid_dict = None

        for source in all_sources:
            source_gids = list(source_gid_set_dict[source])
            for input_features_namespace in this_input_features_namespaces:
                input_features_iter = scatter_read_cell_attribute_selection(input_features_path, source, 
                                                                            namespace=input_features_namespace,
                                                                            mask=set(source_features_attr_names), 
                                                                            selection=source_gids,
                                                                            comm=env.comm, io_size=env.io_size)
                count = 0
                for this_gid, attr_dict in input_features_iter:
                    this_selectivity_type = attr_dict['Selectivity Type'][0]
                    this_selectivity_type_name = selectivity_type_index[this_selectivity_type]
                    input_cell_config = stimulus.get_input_cell_config(this_selectivity_type,
                                                                       selectivity_type_index,
                                                                       selectivity_attr_dict=attr_dict)
                    this_arena_rate_map = np.asarray(input_cell_config.get_rate_map(arena_x, arena_y),
                                                     dtype=np.float32)
                    if source in non_structured_sources:
                        non_structured_input_rate_maps_by_source_gid_dict[this_gid] = this_arena_rate_map
                    else:
                        input_rate_maps_by_source_gid_dict[this_gid] = this_arena_rate_map
                    count += 1

        for destination_gid in selection:

            if is_interactive:
                context.update(locals())

            save_fig_path = None
            if save_fig is not None:
                save_fig_path = '%s/Structured Weights %s %d.png' % (save_fig, destination, destination_gid)
                
            reference_weight_dict = None
            if reference_weights_path is not None:
                reference_weight_dict = reference_weights_by_source_gid_dict[destination_gid]
                
            normalized_LTP_delta_weights_dict, LTD_delta_weights_dict, arena_LS_map = \
               synapses.generate_structured_weights(destination_gid,
                                                 target_map=target_selectivity_features_dict[destination_gid]['Arena Rate Map'],
                                                 initial_weight_dict=initial_weights_by_source_gid_dict[destination_gid],
                                                 reference_weight_dict=reference_weight_dict,
                                                 reference_weights_are_delta=reference_weights_are_delta,
                                                 reference_weights_namespace=reference_weights_namespace,
                                                 input_rate_map_dict=input_rate_maps_by_source_gid_dict,
                                                 non_structured_input_rate_map_dict=non_structured_input_rate_maps_by_source_gid_dict,
                                                 non_structured_weights_dict=non_structured_weights_by_source_gid_dict[destination_gid],
                                                 syn_count_dict=syn_count_by_source_gid_dict[destination_gid],
                                                 max_delta_weight=max_delta_weight,
                                                 max_opt_iter=max_opt_iter,
                                                 max_weight_decay_fraction=max_weight_decay_fraction,
                                                 target_amplitude=target_amplitude,
                                                 arena_x=arena_x, arena_y=arena_y,
                                                 optimize_method=optimize_method,
                                                 optimize_tol=optimize_tol,
                                                 optimize_grad=optimize_grad,
                                                 verbose=verbose if rank == 0 else False, 
                                                 plot=plot, show_fig=show_fig,
                                                 save_fig=save_fig_path,
                                                 fig_kwargs={'gid': destination_gid,
                                                             'field_width': target_field_width_dict[destination_gid]})
            input_rate_maps_by_source_gid_dict.clear()
            output_features_dict[destination_gid] = \
               { fld: target_selectivity_features_dict[destination_gid][fld]
                 for fld in ['Selectivity Type',
                             'Num Fields',
                             'Field Width',
                             'Peak Rate',
                             'X Offset',
                             'Y Offset']}

            this_structured_syn_id_count = structured_syn_id_count[destination_gid]
            output_syn_ids = np.empty(this_structured_syn_id_count, dtype='uint32')
            LTD_output_weights = np.empty(this_structured_syn_id_count, dtype='float32')
            LTP_output_weights = np.empty(this_structured_syn_id_count, dtype='float32')
            i = 0
            for source_gid in normalized_LTP_delta_weights_dict:
                 for syn_id in syn_ids_by_source_gid_dict[destination_gid][source_gid]:
                     output_syn_ids[i] = syn_id
                     LTP_output_weights[i] = normalized_LTP_delta_weights_dict[source_gid]
                     LTD_output_weights[i] = LTD_delta_weights_dict[source_gid]
                     i += 1
            LTP_output_weights_dict[destination_gid] = {'syn_id': output_syn_ids, synapse_name: LTP_output_weights}
            LTD_output_weights_dict[destination_gid] = {'syn_id': output_syn_ids, synapse_name: LTD_output_weights}

            logger.info('Rank %i; destination: %s; gid %i; generated structured weights for %i inputs in %.2f '
                        's' % (rank, destination, destination_gid, len(output_syn_ids), time.time() - local_time))
            gid_count += 1

        env.comm.barrier()
        if iter_count % write_size == 0:
            if not dry_run:
                append_cell_attributes(output_weights_path, destination, LTD_output_weights_dict,
                                       namespace=LTD_output_weights_namespace, comm=env.comm, io_size=env.io_size,
                                       chunk_size=chunk_size, value_chunk_size=value_chunk_size)
                append_cell_attributes(output_weights_path, destination, LTP_output_weights_dict,
                                       namespace=LTP_output_weights_namespace, comm=env.comm, io_size=env.io_size,
                                       chunk_size=chunk_size, value_chunk_size=value_chunk_size)
                count = env.comm.reduce(len(LTP_output_weights_dict), op=MPI.SUM, root=0)
                env.comm.barrier()

                if rank == 0:
                    logger.info('Destination: %s; appended weights for %i cells' % (destination, count))
                if output_features_path is not None:
                    if output_features_namespace is None:
                        output_features_namespace = '%s Selectivity' % target_selectivity_type_name.title()
                    this_output_features_namespace = '%s %s' % (output_features_namespace, arena_id)
                    append_cell_attributes(output_features_path, destination, output_features_dict,
                                           namespace=this_output_features_namespace)
                    count = env.comm.reduce(len(output_features_dict), op=MPI.SUM, root=0)
                    env.comm.barrier()

                    if rank == 0:
                        logger.info('Destination: %s; appended selectivity features for %i cells' % (destination, count))

            LTP_output_weights_dict.clear()
            LTD_output_weights_dict.clear()
            output_features_dict.clear()
            gc.collect()

        env.comm.barrier()

        if (iter_count >= 10) and debug:
            break

    env.comm.barrier()
    if not dry_run:
        append_cell_attributes(output_weights_path, destination, LTD_output_weights_dict,
                               namespace=LTD_output_weights_namespace, comm=env.comm, io_size=env.io_size,
                               chunk_size=chunk_size, value_chunk_size=value_chunk_size)
        append_cell_attributes(output_weights_path, destination, LTP_output_weights_dict,
                               namespace=LTP_output_weights_namespace, comm=env.comm, io_size=env.io_size,
                               chunk_size=chunk_size, value_chunk_size=value_chunk_size)
        count = comm.reduce(len(LTP_output_weights_dict), op=MPI.SUM, root=0)
        env.comm.barrier()

        if rank == 0:
            logger.info('Destination: %s; appended weights for %i cells' % (destination, count))
        if output_features_path is not None:
            if output_features_namespace is None:
                output_features_namespace = 'Selectivity Features'
            this_output_features_namespace = '%s %s' % (output_features_namespace, arena_id)
            append_cell_attributes(output_features_path, destination, output_features_dict,
                                   namespace=this_output_features_namespace)
            count = env.comm.reduce(len(output_features_dict), op=MPI.SUM, root=0)
            env.comm.barrier()

            if rank == 0:
                logger.info('Destination: %s; appended selectivity features for %i cells' % (destination, count))

    env.comm.barrier()
    global_count = env.comm.gather(gid_count, root=0)
    env.comm.barrier()

    if rank == 0:
        logger.info('Destination: %s; %i ranks assigned structured weights to %i cells in %.2f s' %
                    (destination, env.comm.size, np.sum(global_count), time.time() - start_time))


if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv)+1):])
