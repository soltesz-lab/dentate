import sys, os, time, gc, click, logging
from collections import defaultdict
import numpy as np
from mpi4py import MPI
import neuroh5
from neuroh5.io import append_cell_attributes, read_population_ranges, bcast_cell_attributes, \
    scatter_read_cell_attributes, read_cell_attribute_selection, NeuroH5ProjectionGen
import dentate
from dentate.env import Env
from dentate import utils, stimulus, synapses
from dentate.utils import *
import h5py

context = Context()

sys_excepthook = sys.excepthook
def mpi_excepthook(type, value, traceback):
    sys_excepthook(type, value, traceback)
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)
sys.excepthook = mpi_excepthook


    

@click.command()
@click.option("--config", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--coordinates", '-c', type=(float, float), default=(None, None))
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
@click.option("--arena-id", '-a', type=str, default='A')
@click.option("--field-width-scale", type=float, default=1.2)
@click.option("--max-delta-weight", type=float, default=4.)
@click.option("--optimize-method", type=str, default='L-BFGS-B')
@click.option("--optimize-tol", type=float, default=1e-4)
@click.option("--optimize-grad", is_flag=True)
@click.option("--peak-rate", type=float)
@click.option("--reference-weights-are-delta", type=bool, default=False)
@click.option("--use-arena-margin", is_flag=True)
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--cache-size", type=int, default=50)
@click.option("--write-size", type=int, default=1)
@click.option("--verbose", "-v", is_flag=True)
@click.option("--dry-run", is_flag=True)
@click.option("--plot", is_flag=True)
@click.option("--show-fig", is_flag=True)
@click.option("--save-fig", type=click.Path(exists=True, file_okay=False, dir_okay=True))
def main(config, coordinates, field_width, gid, input_features_path, input_features_namespaces, initial_weights_path, output_features_namespace, output_features_path, output_weights_path, reference_weights_path, h5types_path, synapse_name, initial_weights_namespace, output_weights_namespace, reference_weights_namespace, connections_path, destination, sources, arena_id, field_width_scale, max_delta_weight, optimize_method, optimize_tol, optimize_grad, peak_rate, reference_weights_are_delta, use_arena_margin, io_size, chunk_size, value_chunk_size, cache_size, write_size, verbose, dry_run, plot, show_fig, save_fig):
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

    this_output_weights_namespace = '%s %s' % (output_weights_namespace, arena_id)
    this_input_features_namespaces = ['%s %s' % (input_features_namespace, arena_id) for input_features_namespace in input_features_namespaces]

    initial_weights_dict = None

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
    
    connection_gen_list = [ NeuroH5ProjectionGen(connections_path, source, destination, namespaces=['Synapses'], comm=comm) \
                               for source in sources ]

    structured_weights_dict = {}
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
            local_random.seed(int(this_gid + seed_offset))

        has_structured_weights = False

        dst_input_features_attr_dict = {}
        for input_features_namespace in this_input_features_namespaces:
            input_features_iter = read_cell_attribute_selection(input_features_path, destination, 
                                                                namespace=input_features_namespace,
                                                                mask=set(target_features_attr_names), 
                                                                comm=env.comm, selection=selection)
            count = 0
            for gid, attr_dict in input_features_iter:
                dst_input_features_attr_dict[gid] = attr_dict
                count += 1
            if rank == 0:
                logger.info('Read %s feature data for %i cells in population %s' % (input_features_namespace, count, destination))

        arena_margin = 0.
        target_selectivity_features_dict = {}
        target_selectivity_config_dict = {}
        target_field_width_dict = {}
        for gid in selection:
            target_selectivity_features_dict[gid] = dst_input_features_attr_dict.get(gid, {})
            target_selectivity_features_dict[gid]['Selectivity Type'] = np.asarray([target_selectivity_type], dtype=np.uint8)

            if coordinates[0] is not None:
                target_selectivity_features_dict[gid]['X Offset'] =  np.asarray([coordinates[0]], dtype=np.float32)
                target_selectivity_features_dict[gid]['Y Offset'] =  np.asarray([coordinates[1]], dtype=np.float32)
                target_selectivity_features_dict[gid]['Num Fields'] = np.asarray([1], dtype=np.uint8)

            num_fields = target_selectivity_features_dict[gid]['Num Fields']
            if field_width is not None:
                target_selectivity_features_dict[gid]['Field Width'] = np.asarray([field_width]*num_fields, dtype=np.float32)

            if peak_rate is not None:
                target_selectivity_features_dict[gid]['Peak Rate'] = np.asarray([peak_rate]*num_fields, dtype=np.float32)

            input_cell_config = stimulus.get_input_cell_config(target_selectivity_type,
                                                               selectivity_type_index,
                                                               selectivity_attr_dict=target_selectivity_features_dict[gid])
            if input_cell_config.num_fields > 0:
                arena_margin = max(arena_margin, np.max(input_cell_config.field_width) / 2.) if use_arena_margin else 0.
                target_field_width_dict[gid] = input_cell_config.field_width
                input_cell_config.field_width *= field_width_scale
                target_selectivity_config_dict[gid] = input_cell_config
                has_structured_weights = True

        arena_x, arena_y = stimulus.get_2D_arena_spatial_mesh(arena, spatial_resolution,
                                                              margin=arena_margin)
        for gid, input_cell_config in viewitems(target_selectivity_config_dict):
            target_map = np.asarray(input_cell_config.get_rate_map(arena_x, arena_y),
                                    dtype=np.float32)
            target_selectivity_features_dict[gid]['Arena Rate Map'] = target_map

                
        if not has_structured_weights:
            selection = []
                
        initial_weights_by_syn_id_dict = {}
        initial_weights_by_source_gid_dict = {}

        if initial_weights_path is not None:
            initial_weights_iter = \
              read_cell_attribute_selection(initial_weights_path, destination,
                                            namespace=initial_weights_namespace,
                                            selection=selection)
            syn_weight_attr_dict = dict(initial_weights_iter)

            syn_ids = syn_weight_attr_dict[target_gid]['syn_id']
            weights = syn_weight_attr_dict[target_gid][synapse_name]

            for (syn_id, weight) in zip(syn_ids, weights):
                initial_weights_by_syn_id_dict[int(syn_id)] = float(weight)

            logger.info('destination: %s; gid %i; read initial synaptic weights for %i synapses' %
                        (destination, this_gid, len(initial_weights_by_syn_id_dict)))
            
        reference_weights_by_syn_id_dict = None
        if reference_weights_path is not None:
            reference_weights_by_syn_id_dict = dict()
            reference_weights_iter = \
              read_cell_attribute_selection(reference_weights_path, destination, namespace=reference_weights_namespace,
                                            selection=selection)
            syn_weight_attr_dict = dict(reference_weights_iter)

            syn_ids = syn_weight_attr_dict[target_gid]['syn_id']
            weights = syn_weight_attr_dict[target_gid][synapse_name]

            for (syn_id, weight) in zip(syn_ids, weights):
                reference_weights_by_syn_id_dict[int(syn_id)] = float(weight)

            logger.info('destination: %s; gid %i; read reference synaptic weights for %i synapses' %
                        (destination, target_gid, len(reference_weights_by_syn_id_dict)))
            
        if reference_weights_by_syn_id_dict is None:
            reference_weights_by_source_gid_dict = None
        else:
            reference_weights_by_source_gid_dict = dict()

        syn_count_by_source_gid_dict = defaultdict(int)
        source_gid_set_dict = defaultdict(set)
        syn_ids_by_source_gid_dict = defaultdict(list)
        structured_syn_id_count = 0

        if has_structured_weights:
            for source, (destination_gid, (source_gid_array, conn_attr_dict)) in zip_longest(sources, attr_gen_package):
                syn_ids = conn_attr_dict['Synapses']['syn_id']
                count = 0
                for i in range(len(source_gid_array)):
                    this_source_gid = source_gid_array[i]
                    this_syn_id = syn_ids[i]
                    this_syn_wgt = initial_weights_by_syn_id_dict.get(this_syn_id, 1.0)
                    source_gid_set_dict[source].add(this_source_gid)
                    syn_ids_by_source_gid_dict[this_source_gid].append(this_syn_id)
                    syn_count_by_source_gid_dict[this_source_gid] += 1
                    if this_source_gid not in initial_weights_by_source_gid_dict:
                        initial_weights_by_source_gid_dict[this_source_gid] = this_syn_wgt
                    if reference_weights_by_source_gid_dict is not None:
                        reference_weights_by_source_gid_dict[this_source_gid] = \
                         reference_weights_by_syn_id_dict[this_syn_id]

                    count += 1
                structured_syn_id_count += len(syn_ids)
                logger.info('Rank %i; destination: %s; gid %i; %d edges from source population %s' %
                            (rank, destination, this_gid, count, source))


        input_rate_maps_by_source_gid_dict = {}
        for source in sources:
            if has_structured_weights:
                source_gids = list(source_gid_set_dict[source])
            else:
                source_gids = []
            if rank == 0:
                logger.info('Reading %s feature data for %i cells in population %s...' % (input_features_namespace, len(source_gids), source))
            for input_features_namespace in this_input_features_namespaces:
                input_features_iter = read_cell_attribute_selection(input_features_path, source, 
                                                                    namespace=input_features_namespace,
                                                                    mask=set(source_features_attr_names), 
                                                                    comm=env.comm, selection=source_gids)
                count = 0
                for gid, attr_dict in input_features_iter:
                    this_selectivity_type = attr_dict['Selectivity Type'][0]
                    this_selectivity_type_name = selectivity_type_index[this_selectivity_type]
                    input_cell_config = stimulus.get_input_cell_config(this_selectivity_type,
                                                                       selectivity_type_index,
                                                                       selectivity_attr_dict=attr_dict)
                    this_arena_rate_map = np.asarray(input_cell_config.get_rate_map(arena_x, arena_y),
                                                     dtype=np.float32)
                    input_rate_maps_by_source_gid_dict[gid] = this_arena_rate_map
                    count += 1
                if rank == 0:
                    logger.info('Read %s feature data for %i cells in population %s' % (input_features_namespace, count, source))

        output_features_dict = {}
        output_weights_dict = {}
        if has_structured_weights:

            if is_interactive:
                context.update(locals())

            save_fig_path = None
            if save_fig is not None:
                save_fig_path = '%s/Structured Weights %s %d.png' % (save_fig, destination, this_gid)
                
            normalized_delta_weights_dict, arena_LS_map = \
              synapses.generate_structured_weights(target_map=target_selectivity_features_dict[this_gid]['Arena Rate Map'],
                                                initial_weight_dict=initial_weights_by_source_gid_dict,
                                                reference_weight_dict=reference_weights_by_source_gid_dict,
                                                reference_weights_are_delta=reference_weights_are_delta,
                                                reference_weights_namespace=reference_weights_namespace,
                                                input_rate_map_dict=input_rate_maps_by_source_gid_dict,
                                                syn_count_dict=syn_count_by_source_gid_dict,
                                                max_delta_weight=max_delta_weight, arena_x=arena_x, arena_y=arena_y,
                                                optimize_method=optimize_method,
                                                optimize_tol=optimize_tol,
                                                optimize_grad=optimize_grad,
                                                verbose=verbose, plot=plot, show_fig=show_fig,
                                                save_fig=save_fig_path,
                                                fig_kwargs={'gid': this_gid,
                                                            'field_width': target_field_width_dict[this_gid]})

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
            output_weights = np.empty(structured_syn_id_count, dtype='float32')
            i = 0
            for source_gid, this_weight in viewitems(normalized_delta_weights_dict):
                for syn_id in syn_ids_by_source_gid_dict[source_gid]:
                    output_syn_ids[i] = syn_id
                    output_weights[i] = this_weight
                    i += 1
            output_weights_dict[this_gid] = {'syn_id': output_syn_ids, synapse_name: output_weights}

            logger.info('Rank %i; destination: %s; gid %i; generated structured weights for %i inputs in %.2f '
                        's' % (rank, destination, this_gid, len(output_syn_ids), time.time() - local_time))
            gid_count += 1

        if iter_count % write_size == 0:
            gc.collect()
            if not dry_run:
                append_cell_attributes(output_weights_path, destination, output_weights_dict,
                                       namespace=this_output_weights_namespace, comm=env.comm, io_size=env.io_size,
                                       chunk_size=chunk_size, value_chunk_size=value_chunk_size)
                count = comm.reduce(len(output_weights_dict), op=MPI.SUM, root=0)
                if rank == 0:
                    logger.info('Destination: %s; appended weights for %i cells' % (destination, count))
                if output_features_path is not None:
                    if output_features_namespace is None:
                        output_features_namespace = '%s Selectivity' % target_selectivity_type.title()
                    this_output_features_namespace = '%s %s' % (output_features_namespace, arena_id)
                    append_cell_attributes(output_features_path, destination, output_features_dict,
                                           namespace=this_output_features_namespace)
                    count = comm.reduce(len(output_features_dict), op=MPI.SUM, root=0)
                    if rank == 0:
                        logger.info('Destination: %s; appended selectivity features for %i cells' % (destination, count))

            output_weights_dict.clear()
            output_features_dict.clear()
            gc.collect()

        env.comm.barrier()

    if not dry_run:
        append_cell_attributes(output_weights_path, destination, output_weights_dict,
                               namespace=this_output_weights_namespace, comm=env.comm, io_size=env.io_size,
                               chunk_size=chunk_size, value_chunk_size=value_chunk_size)
        count = comm.reduce(len(output_weights_dict), op=MPI.SUM, root=0)
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
        logger.info('destination: %s; %i ranks assigned structured weights to %i cells in %.2f s' %
                    (destination, comm.size, np.sum(global_count), time.time() - start_time))


if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv)+1):])
