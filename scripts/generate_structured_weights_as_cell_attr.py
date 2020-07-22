
import sys, os, time, gc, click, logging, pprint
from collections import defaultdict
import numpy as np
from mpi4py import MPI
import neuroh5
from neuroh5.io import append_cell_attributes, read_population_ranges, \
    scatter_read_cell_attribute_selection, NeuroH5ProjectionGen
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
def main(config, coordinates, field_width, gid, input_features_path, input_features_namespaces, initial_weights_path, output_features_namespace, output_features_path, output_weights_path, reference_weights_path, h5types_path, synapse_name, initial_weights_namespace, output_weights_namespace, reference_weights_namespace, connections_path, destination, sources, non_structured_sources, non_structured_weights_namespace, non_structured_weights_path, arena_id, field_width_scale, max_delta_weight, max_opt_iter, max_weight_decay_fraction, optimize_method, optimize_tol, optimize_grad, peak_rate, reference_weights_are_delta, arena_margin, target_amplitude, io_size, chunk_size, value_chunk_size, cache_size, write_size, verbose, dry_run, plot, show_fig, save_fig):
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

    local_random = np.random.RandomState()

    seed_offset = int(env.model_config['Random Seeds']['GC Structured Weights'])
    spatial_resolution = env.stimulus_config['Spatial Resolution'] # cm

    arena = env.stimulus_config['Arena'][arena_id]
    default_run_vel = arena.properties['default run velocity']  # cm/s

    gid_count = 0
    start_time = time.time()

    target_gid_set = None
    if len(gid) > 0:
        target_gid_set = set(gid)

    all_sources = sources + non_structured_sources
        
    connection_gen_list = [ NeuroH5ProjectionGen(connections_path, source, destination, namespaces=['Synapses'], io_size=io_size, cache_size=cache_size, comm=comm) \
                               for source in all_sources ]

    output_features_dict = {}
    LTP_output_weights_dict = {}
    LTD_output_weights_dict = {}
    for iter_count, attr_gen_package in enumerate(zip_longest(*connection_gen_list)):
        
        local_time = time.time()
        this_gid = attr_gen_package[0][0]
        if not all([attr_gen_items[0] == this_gid for attr_gen_items in attr_gen_package]):
            raise Exception('Rank: %i; destination: %s; this_gid not matched across multiple attribute '
                            'generators: %s' % (rank, destination,
                                                [attr_gen_items[0] for attr_gen_items in attr_gen_package]))
        
        if (target_gid_set is not None) and (this_gid not in target_gid_set):
            continue


        if this_gid is None:
            selection = []
            logger.info('Rank: %i received None' % rank)
        else:
            selection = [this_gid]
            logger.info('Rank: %i received gid %d' % (rank, this_gid))
            local_random.seed(int(this_gid + seed_offset))

        has_structured_weights = False

        dst_input_features_attr_dict = {}
        for input_features_namespace in this_input_features_namespaces:
            input_features_iter = scatter_read_cell_attribute_selection(input_features_path, destination, 
                                                                        namespace=input_features_namespace,
                                                                        mask=set(target_features_attr_names), 
                                                                        selection=selection,
                                                                        io_size=env.io_size, comm=env.comm)
            count = 0
            for gid, attr_dict in input_features_iter:
                dst_input_features_attr_dict[gid] = attr_dict
                count += 1
            if this_gid is not None:
                logger.info('Rank %i: read %s feature data for %i cells in population %s' % (rank, input_features_namespace, count, destination))

        arena_margin_size = 0.
        arena_margin = max(arena_margin, 0.)
        target_selectivity_features_dict = {}
        target_selectivity_config_dict = {}
        target_field_width_dict = {}
        for gid in selection:
            target_selectivity_features_dict[gid] = dst_input_features_attr_dict.get(gid, {})
            target_selectivity_features_dict[gid]['Selectivity Type'] = np.asarray([target_selectivity_type], dtype=np.uint8)
            if len(coordinates) > 0:
                num_fields = len(coordinates)
                target_selectivity_features_dict[gid]['X Offset'] =  np.asarray([x[0] for x in coordinates],
                                                                                dtype=np.float32)
                target_selectivity_features_dict[gid]['Y Offset'] =  np.asarray([x[1] for x in coordinates],
                                                                                dtype=np.float32)
                target_selectivity_features_dict[gid]['Num Fields'] = np.asarray([num_fields], dtype=np.uint8)
            elif 'Num Fields' in target_selectivity_features_dict[gid]:
                num_fields = target_selectivity_features_dict[gid]['Num Fields'][0]
            else:
                num_fields = 0

            if field_width is not None:
                target_selectivity_features_dict[gid]['Field Width'] = np.asarray([field_width]*num_fields, dtype=np.float32)
            elif 'Field Width' in target_selectivity_features_dict[gid]:
                this_field_width = target_selectivity_features_dict[gid]['Field Width']
                target_selectivity_features_dict[gid]['Field Width'] = this_field_width[:num_fields]
            else:
                this_field_width = np.asarray([], dtype=np.float32)

            if peak_rate is not None:
                target_selectivity_features_dict[gid]['Peak Rate'] = np.asarray([peak_rate]*num_fields, dtype=np.float32)

            if num_fields > 0:
                input_cell_config = stimulus.get_input_cell_config(target_selectivity_type,
                                                                   selectivity_type_index,
                                                                   selectivity_attr_dict=target_selectivity_features_dict[gid])
                arena_margin_size = max(arena_margin_size, np.max(input_cell_config.field_width) * arena_margin)
                target_field_width_dict[gid] = input_cell_config.field_width
                target_selectivity_config_dict[gid] = input_cell_config
                has_structured_weights = True

        arena_x, arena_y = stimulus.get_2D_arena_spatial_mesh(arena, spatial_resolution,
                                                              margin=arena_margin_size)
        for gid, input_cell_config in viewitems(target_selectivity_config_dict):
            target_map = np.asarray(input_cell_config.get_rate_map(arena_x, arena_y,
                                                                   scale=field_width_scale),
                                    dtype=np.float32)
            target_selectivity_features_dict[gid]['Arena Rate Map'] = target_map

                
        if not has_structured_weights:
            selection = []
                
        initial_weights_by_syn_id_dict = defaultdict(lambda: dict())
        initial_weights_by_source_gid_dict = defaultdict(lambda: dict())

        if initial_weights_path is not None:
            initial_weights_iter = \
              scatter_read_cell_attribute_selection(initial_weights_path, destination,
                                                    namespace=initial_weights_namespace,
                                                    selection=selection, 
                                                    comm=env.comm, io_size=env.io_size)

            initial_weights_gid_count = 0
            initial_weights_syn_count = 0
            for this_gid, syn_weight_attr_dict in initial_weights_iter:
                syn_ids = syn_weight_attr_dict['syn_id']
                weights = syn_weight_attr_dict[synapse_name]

                for (syn_id, weight) in zip(syn_ids, weights):
                    initial_weights_by_syn_id_dict[this_gid][int(syn_id)] = float(weight)
                initial_weights_gid_count += 1
                initial_weights_syn_count += len(syn_ids)

            if has_structured_weights and (this_gid is not None):
                logger.info('Rank %i: destination: %s; read initial synaptic weights for %i gids and %i syns' %
                            (rank, destination, initial_weights_gid_count, initial_weights_syn_count))

        if len(non_structured_sources) > 0:
            non_structured_weights_by_syn_id_dict = defaultdict(lambda: dict())
            non_structured_weights_by_source_gid_dict = defaultdict(lambda: dict())
        else:
            non_structured_weights_by_syn_id_dict = None
            
        if non_structured_weights_path is not None:
            non_structured_weights_iter = \
                scatter_read_cell_attribute_selection(initial_weights_path, destination,
                                                      namespace=non_structured_weights_namespace,
                                                      selection=selection,
                                                      comm=env.comm, io_size=env.io_size)

            non_structured_weights_gid_count = 0
            non_structured_weights_syn_count = 0
            for this_gid, syn_weight_attr_dict in non_structured_weights_iter:
                syn_ids = syn_weight_attr_dict['syn_id']
                weights = syn_weight_attr_dict[synapse_name]

                for (syn_id, weight) in zip(syn_ids, weights):
                    non_structured_weights_by_syn_id_dict[this_gid][int(syn_id)] = float(weight)
                non_structured_weights_gid_count += 1
                non_structured_weights_syn_count += len(syn_ids)


            if has_structured_weights and (this_gid is not None):
                logger.info('Rank %i: destination: %s; read non-structured synaptic weights for %i gids and %i syns' %
                            (rank, destination, non_structured_weights_gid_count, non_structured_weights_syn_count, ))
            
        reference_weights_by_syn_id_dict = None
        reference_weights_by_source_gid_dict = defaultdict(lambda: dict())
        if reference_weights_path is not None:
            reference_weights_by_syn_id_dict = defaultdict(lambda: dict())
            reference_weights_iter = \
              scatter_read_cell_attribute_selection(reference_weights_path, destination, 
                                                    namespace=reference_weights_namespace,
                                                    selection=selection, 
                                                    comm=env.comm, io_size=env.io_size)
            reference_weights_gid_count = 0

            for this_gid, syn_weight_attr_dict in reference_weights_iter:
                syn_ids = syn_weight_attr_dict['syn_id']
                weights = syn_weight_attr_dict[synapse_name]
            
                for (syn_id, weight) in zip(syn_ids, weights):
                    reference_weights_by_syn_id_dict[this_gid][int(syn_id)] = float(weight)

            if has_structured_weights and (this_gid is not None):
                logger.info('Rank %i: destination: %s; read reference synaptic weights for %i gids' %
                            (rank, destination, reference_weights_gid_count))
            

        syn_count_by_source_gid_dict = defaultdict(int)
        source_gid_set_dict = defaultdict(set)
        syn_ids_by_source_gid_dict = defaultdict(list)
        structured_syn_id_count = 0

        if has_structured_weights:
            for source, (destination_gid, (source_gid_array, conn_attr_dict)) in zip_longest(all_sources, attr_gen_package):
                syn_ids = conn_attr_dict['Synapses']['syn_id']
                count = 0
                this_initial_weights_by_syn_id_dict = None
                this_initial_weights_by_source_gid_dict = None
                this_reference_weights_by_syn_id_dict = None
                this_reference_weights_by_source_gid_dict = None
                this_non_structured_weights_by_syn_id_dict = None
                this_non_structured_weights_by_source_gid_dict = None
                if destination_gid is not None:
                    this_initial_weights_by_syn_id_dict = initial_weights_by_syn_id_dict[destination_gid]
                    this_initial_weights_by_source_gid_dict = initial_weights_by_source_gid_dict[destination_gid]
                    if reference_weights_by_syn_id_dict is not None:
                        this_reference_weights_by_syn_id_dict = reference_weights_by_syn_id_dict[destination_gid]
                        this_reference_weights_by_source_gid_dict = reference_weights_by_source_gid_dict[destination_gid]
                    this_non_structured_weights_by_syn_id_dict = non_structured_weights_by_syn_id_dict[destination_gid]
                    this_non_structured_weights_by_source_gid_dict = non_structured_weights_by_source_gid_dict[destination_gid]

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
                    source_gid_set_dict[source].add(this_source_gid)
                    syn_ids_by_source_gid_dict[this_source_gid].append(this_syn_id)
                    syn_count_by_source_gid_dict[this_source_gid] += 1
                        
                    count += 1
                if source not in non_structured_sources:
                    structured_syn_id_count += len(syn_ids)
                if has_structured_weights and (this_gid is not None):
                    logger.info('Rank %i: destination: %s; gid %i; %d edges from source population %s' %
                                (rank, destination, this_gid, count, source))


        input_rate_maps_by_source_gid_dict = {}
        if len(non_structured_sources) > 0:
            non_structured_input_rate_maps_by_source_gid_dict = {}
        else:
            non_structured_input_rate_maps_by_source_gid_dict = None
        for source in all_sources:
            if has_structured_weights:
                source_gids = list(source_gid_set_dict[source])
            else:
                source_gids = []
            for input_features_namespace in this_input_features_namespaces:
                input_features_iter = scatter_read_cell_attribute_selection(input_features_path, source, 
                                                                            namespace=input_features_namespace,
                                                                            mask=set(source_features_attr_names), 
                                                                            selection=source_gids,
                                                                            comm=env.comm, io_size=env.io_size)
                count = 0
                for gid, attr_dict in input_features_iter:
                    this_selectivity_type = attr_dict['Selectivity Type'][0]
                    this_selectivity_type_name = selectivity_type_index[this_selectivity_type]
                    input_cell_config = stimulus.get_input_cell_config(this_selectivity_type,
                                                                       selectivity_type_index,
                                                                       selectivity_attr_dict=attr_dict)
                    this_arena_rate_map = np.asarray(input_cell_config.get_rate_map(arena_x, arena_y),
                                                     dtype=np.float32)
                    if source in non_structured_sources:
                        non_structured_input_rate_maps_by_source_gid_dict[gid] = this_arena_rate_map
                    else:
                        input_rate_maps_by_source_gid_dict[gid] = this_arena_rate_map
                    count += 1

        if has_structured_weights:

            if is_interactive:
                context.update(locals())

            save_fig_path = None
            if save_fig is not None:
                save_fig_path = '%s/Structured Weights %s %d.png' % (save_fig, destination, this_gid)
                
            normalized_LTP_delta_weights_dict, LTD_delta_weights_dict, arena_LS_map = \
              synapses.generate_structured_weights(target_map=target_selectivity_features_dict[this_gid]['Arena Rate Map'],
                                                initial_weight_dict=this_initial_weights_by_source_gid_dict,
                                                reference_weight_dict=this_reference_weights_by_source_gid_dict,
                                                reference_weights_are_delta=reference_weights_are_delta,
                                                reference_weights_namespace=reference_weights_namespace,
                                                input_rate_map_dict=input_rate_maps_by_source_gid_dict,
                                                non_structured_input_rate_map_dict=non_structured_input_rate_maps_by_source_gid_dict,
                                                non_structured_weights_dict=this_non_structured_weights_by_source_gid_dict,
                                                syn_count_dict=syn_count_by_source_gid_dict,
                                                max_delta_weight=max_delta_weight,
                                                max_opt_iter=max_opt_iter,
                                                max_weight_decay_fraction=max_weight_decay_fraction,
                                                target_amplitude=target_amplitude,
                                                arena_x=arena_x, arena_y=arena_y,
                                                optimize_method=optimize_method,
                                                optimize_tol=optimize_tol,
                                                optimize_grad=optimize_grad,
                                                verbose=verbose, plot=plot, show_fig=show_fig,
                                                save_fig=save_fig_path,
                                                fig_kwargs={'gid': this_gid,
                                                            'field_width': target_field_width_dict[this_gid]})
            gc.collect()

            this_selectivity_dict = target_selectivity_features_dict[this_gid]
            output_features_dict[this_gid] = { fld: this_selectivity_dict[fld]
                                               for fld in ['Selectivity Type',
                                                           'Num Fields',
                                                           'Field Width',
                                                           'Peak Rate',
                                                           'X Offset',
                                                           'Y Offset'] }
            output_features_dict[this_gid]['Arena State Map'] = np.asarray(arena_LS_map.ravel(), dtype=np.float32)
            output_syn_ids = np.empty(structured_syn_id_count, dtype='uint32')
            LTD_output_weights = np.empty(structured_syn_id_count, dtype='float32')
            LTP_output_weights = np.empty(structured_syn_id_count, dtype='float32')
            i = 0
            for source_gid in normalized_LTP_delta_weights_dict:
                for syn_id in syn_ids_by_source_gid_dict[source_gid]:
                    output_syn_ids[i] = syn_id
                    LTP_output_weights[i] = normalized_LTP_delta_weights_dict[source_gid]
                    LTD_output_weights[i] = LTD_delta_weights_dict[source_gid]
                    i += 1
            LTP_output_weights_dict[this_gid] = {'syn_id': output_syn_ids, synapse_name: LTP_output_weights}
            LTD_output_weights_dict[this_gid] = {'syn_id': output_syn_ids, synapse_name: LTD_output_weights}

            logger.info('Rank %i; destination: %s; gid %i; generated structured weights for %i inputs in %.2f '
                        's' % (rank, destination, this_gid, len(output_syn_ids), time.time() - local_time))
            gid_count += 1

        if iter_count % write_size == 0:
            if not dry_run:
                append_cell_attributes(output_weights_path, destination, LTD_output_weights_dict,
                                       namespace=LTD_output_weights_namespace, comm=env.comm, io_size=env.io_size,
                                       chunk_size=chunk_size, value_chunk_size=value_chunk_size)
                append_cell_attributes(output_weights_path, destination, LTP_output_weights_dict,
                                       namespace=LTP_output_weights_namespace, comm=env.comm, io_size=env.io_size,
                                       chunk_size=chunk_size, value_chunk_size=value_chunk_size)
                count = comm.reduce(len(LTP_output_weights_dict), op=MPI.SUM, root=0)
                if rank == 0:
                    logger.info('Destination: %s; appended weights for %i cells' % (destination, count))
                if output_features_path is not None:
                    if output_features_namespace is None:
                        output_features_namespace = '%s Selectivity' % target_selectivity_type_name.title()
                    this_output_features_namespace = '%s %s' % (output_features_namespace, arena_id)
                    append_cell_attributes(output_features_path, destination, output_features_dict,
                                           namespace=this_output_features_namespace)
                    count = comm.reduce(len(output_features_dict), op=MPI.SUM, root=0)
                    if rank == 0:
                        logger.info('Destination: %s; appended selectivity features for %i cells' % (destination, count))

            LTP_output_weights_dict.clear()
            LTD_output_weights_dict.clear()
            output_features_dict.clear()
            gc.collect()

        env.comm.barrier()

    if not dry_run:
        append_cell_attributes(output_weights_path, destination, LTD_output_weights_dict,
                               namespace=LTD_output_weights_namespace, comm=env.comm, io_size=env.io_size,
                               chunk_size=chunk_size, value_chunk_size=value_chunk_size)
        append_cell_attributes(output_weights_path, destination, LTP_output_weights_dict,
                               namespace=LTP_output_weights_namespace, comm=env.comm, io_size=env.io_size,
                               chunk_size=chunk_size, value_chunk_size=value_chunk_size)
        count = comm.reduce(len(LTP_output_weights_dict), op=MPI.SUM, root=0)
        if rank == 0:
            logger.info('Destination: %s; appended weights for %i cells' % (destination, count))
        if output_features_path is not None:
            if output_features_namespace is None:
                output_features_namespace = 'Selectivity Features'
            this_output_features_namespace = '%s %s' % (output_features_namespace, arena_id)
            append_cell_attributes(output_features_path, destination, output_features_dict,
                                   namespace=this_output_features_namespace)
            count = comm.reduce(len(output_features_dict), op=MPI.SUM, root=0)
            if rank == 0:
                logger.info('Destination: %s; appended selectivity features for %i cells' % (destination, count))

    env.comm.barrier()
    global_count = comm.gather(gid_count, root=0)
    if rank == 0:
        logger.info('destination: %s; %i ranks generated structured weights for %i cells in %.2f s' %
                    (destination, comm.size, np.sum(global_count), time.time() - start_time))


if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv)+1):])
