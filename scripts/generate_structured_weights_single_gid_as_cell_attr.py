import sys, os, time, gc, click, logging, pprint
from collections import defaultdict
from functools import reduce
import numpy as np
from mpi4py import MPI
import neuroh5
from neuroh5.io import append_cell_attributes, read_population_ranges, read_cell_attribute_selection, \
    read_graph_selection
import dentate
from dentate.env import Env
from dentate import utils, stimulus, synapses
from dentate.stimulus import get_2D_arena_spatial_mesh
from dentate.utils import *
import h5py


context = Context()


def get_place_rate_map(x0, y0, width, x, y):
    return np.exp(-((x - x0) / (width / 3. / np.sqrt(2.))) ** 2.) * \
           np.exp(-((y - y0) / (width / 3. / np.sqrt(2.))) ** 2.)


def get_rate_map(x0, y0, field_width, peak_rate, x, y):
    num_fields = len(x0)
    rate_map = np.zeros_like(x, dtype='float32')
    for i in range(num_fields):
        rate_map = np.maximum(rate_map, get_place_rate_map(x0[i], y0[i], field_width[i], x, y))
    return np.multiply(rate_map, peak_rate)


@click.command()
@click.option("--config", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--coordinates", '-c', required=True, type=(float, float), multiple=True)
@click.option("--gid", required=True, type=int)
@click.option("--field-width", default=40.0, type=float)
@click.option("--peak-rate", default=20.0, type=float)
@click.option("--input-features-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--input-features-namespaces", type=str, multiple=True, default=['Place Selectivity', 'Grid Selectivity'])
@click.option("--output-features-namespace", type=str, default='Place Selectivity')
@click.option("--output-weights-path", required=False, type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option("--output-features-path", required=False, type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option("--initial-weights-path", required=False, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--reference-weights-path", required=False, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--h5types-path", required=False, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--synapse-name", type=str, default='AMPA')
@click.option("--initial-weights-namespace", type=str, default='Weights')
@click.option("--reference-weights-namespace", type=str, default='Weights')
@click.option("--output-weights-namespace", type=str, default='Normalized Structured Weights')
@click.option("--reference-weights-are-delta", type=bool, default=False)
@click.option("--connections-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--optimize-method", type=str, default='L-BFGS-B')
@click.option("--destination", '-d', type=str)
@click.option("--sources", '-s', type=str, multiple=True)
@click.option("--arena-id", '-a', type=str, default='A')
@click.option("--max-delta-weight", type=float, default=4.)
@click.option("--field-width-scale", type=float, default=1.2)
@click.option("--max-iter", type=int, default=100)
@click.option("--verbose", "-v", is_flag=True)
@click.option("--dry-run", is_flag=True)
@click.option("--plot", is_flag=True)
def main(config, coordinates, gid, field_width, peak_rate, input_features_path, input_features_namespaces,
         output_features_namespace, output_weights_path, output_features_path, initial_weights_path,
         reference_weights_path, h5types_path, synapse_name, initial_weights_namespace, reference_weights_namespace,
         output_weights_namespace, reference_weights_are_delta, connections_path, optimize_method, destination, sources,
         arena_id, max_delta_weight, field_width_scale, max_iter, verbose, dry_run, plot):
    """
    :param config: str (path to .yaml file)
    :param coordinates: tuple of float
    :param gid: int
    :param field_width: float
    :param peak_rate: float
    :param input_features_path: str (path to .h5 file)
    :param input_features_namespaces: str
    :param output_features_namespace: str
    :param output_weights_path: str (path to .h5 file)
    :param output_features_path: str (path to .h5 file)
    :param initial_weights_path: str (path to .h5 file)
    :param reference_weights_path: str (path to .h5 file)
    :param h5types_path: str (path to .h5 file)
    :param synapse_name: str
    :param initial_weights_namespace: str
    :param output_weights_namespace: str
    :param reference_weights_are_delta: bool
    :param connections_path: str (path to .h5 file)
    :param destination: str (population name)
    :param sources: list of str (population name)
    :param arena_id: str
    :param max_delta_weight: float
    :param field_width_scale: float
    :param max_iter: int
    :param verbose: bool
    :param dry_run: bool
    :param interactive: bool
    :param plot: bool
    """
    utils.config_logging(verbose)
    logger = utils.get_script_logger(__file__)

    env = Env(config_file=config)

    target_gid = gid
    
    if not dry_run:
        if output_weights_path is None:
            raise RuntimeError('Missing required argument: output_weights_path.')
        if not os.path.isfile(output_weights_path):
            if initial_weights_path is not None and os.path.isfile(initial_weights_path):
                input_file_path = initial_weights_path
            elif h5types_path is not None and os.path.isfile(h5types_path):
                input_file_path = h5types_path
            else:
                raise RuntimeError('Missing required source for h5types: either an initial_weights_path or an '
                                   'h5types_path must be provided.')
            with h5py.File(output_weights_path, 'a') as output_file:
                with h5py.File(input_file_path, 'r') as input_file:
                    input_file.copy('/H5Types', output_file)

    this_input_features_namespaces = ['%s %s' % (input_features_namespace, arena_id) for
                                      input_features_namespace in input_features_namespaces]
    features_attr_names = ['Arena Rate Map']
    spatial_resolution = env.stimulus_config['Spatial Resolution'] # cm
    arena = env.stimulus_config['Arena'][arena_id]
    default_run_vel = arena.properties['default run velocity']  # cm/s
    arena_x, arena_y = stimulus.get_2D_arena_spatial_mesh(arena, spatial_resolution)

    dst_input_features = defaultdict(dict)
    num_fields = len(coordinates)
    this_field_width = np.array([field_width] * num_fields, dtype=np.float32)
    this_scaled_field_width = np.array([field_width * field_width_scale] * num_fields, dtype=np.float32)
    this_peak_rate = np.array([peak_rate] * num_fields, dtype=np.float32)
    this_x0 = np.array([x for x, y in coordinates], dtype=np.float32)
    this_y0 = np.array([y for x, y in coordinates], dtype=np.float32)
    this_rate_map = np.asarray(get_rate_map(this_x0, this_y0, this_field_width, this_peak_rate, arena_x, arena_y),
                               dtype=np.float32)
    target_map = np.asarray(get_rate_map(this_x0, this_y0, this_scaled_field_width, this_peak_rate, arena_x, arena_y),
                            dtype=np.float32)
    selectivity_type = env.selectivity_types['place']
    dst_input_features[destination][target_gid] = {
        'Selectivity Type': np.array([selectivity_type], dtype=np.uint8),
        'Num Fields': np.array([num_fields], dtype=np.uint8),
        'Field Width': this_field_width,
        'Peak Rate': this_peak_rate,
        'X Offset': this_x0,
        'Y Offset': this_y0,
        'Arena Rate Map': this_rate_map.ravel()}

    initial_weights_by_syn_id_dict = dict()
    selection = [target_gid]
    if initial_weights_path is not None:
        initial_weights_iter = \
            read_cell_attribute_selection(initial_weights_path, destination, namespace=initial_weights_namespace,
                                          selection=selection)
        syn_weight_attr_dict = dict(initial_weights_iter)

        syn_ids = syn_weight_attr_dict[target_gid]['syn_id']
        weights = syn_weight_attr_dict[target_gid][synapse_name]

        for (syn_id, weight) in zip(syn_ids, weights):
            initial_weights_by_syn_id_dict[int(syn_id)] = float(weight)

        logger.info('destination: %s; gid %i; read initial synaptic weights for %i synapses' %
                    (destination, target_gid, len(initial_weights_by_syn_id_dict)))

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

    source_gid_set_dict = defaultdict(set)
    syn_ids_by_source_gid_dict = defaultdict(list)
    initial_weights_by_source_gid_dict = dict()
    if reference_weights_by_syn_id_dict is None:
        reference_weights_by_source_gid_dict = None
    else:
        reference_weights_by_source_gid_dict = dict()
    (graph, edge_attr_info) = read_graph_selection(file_name=connections_path,
                                                   selection=[target_gid],
                                                   namespaces=['Synapses'])
    syn_id_attr_index = None
    for source, edge_iter in viewitems(graph[destination]):
        if source not in sources:
            continue
        this_edge_attr_info = edge_attr_info[destination][source]
        if 'Synapses' in this_edge_attr_info and \
           'syn_id' in this_edge_attr_info['Synapses']:
            syn_id_attr_index = this_edge_attr_info['Synapses']['syn_id']
        for (destination_gid, edges) in edge_iter:
            assert destination_gid == target_gid
            source_gids, edge_attrs = edges
            syn_ids = edge_attrs['Synapses'][syn_id_attr_index]
            count = 0
            for i in range(len(source_gids)):
                this_source_gid = int(source_gids[i])
                source_gid_set_dict[source].add(this_source_gid)
                this_syn_id = int(syn_ids[i])
                if this_syn_id not in initial_weights_by_syn_id_dict:
                    this_weight = \
                        env.connection_config[destination][source].mechanisms['default'][synapse_name]['weight']
                    initial_weights_by_syn_id_dict[this_syn_id] = this_weight
                syn_ids_by_source_gid_dict[this_source_gid].append(this_syn_id)
                if this_source_gid not in initial_weights_by_source_gid_dict:
                    initial_weights_by_source_gid_dict[this_source_gid] = \
                        initial_weights_by_syn_id_dict[this_syn_id]
                    if reference_weights_by_source_gid_dict is not None:
                        reference_weights_by_source_gid_dict[this_source_gid] = \
                            reference_weights_by_syn_id_dict[this_syn_id]
                count += 1
            logger.info('destination: %s; gid %i; set initial synaptic weights for %d inputs from source population '
                        '%s' % (destination, destination_gid, count, source))

    syn_count_by_source_gid_dict = dict()
    for source_gid in syn_ids_by_source_gid_dict:
        syn_count_by_source_gid_dict[source_gid] = len(syn_ids_by_source_gid_dict[source_gid])

    input_rate_maps_by_source_gid_dict = dict()
    for source in sources:
        source_gids = list(source_gid_set_dict[source])
        for input_features_namespace in this_input_features_namespaces:
            input_features_iter = read_cell_attribute_selection(input_features_path, source, 
                                                                namespace=input_features_namespace,
                                                                mask=set(features_attr_names), 
                                                                selection=source_gids)
            count = 0
            for gid, attr_dict in input_features_iter:
                input_rate_maps_by_source_gid_dict[gid] = attr_dict['Arena Rate Map']
                count += 1
            logger.info('Read %s feature data for %i cells in population %s' %
                        (input_features_namespace, count, source))

    if is_interactive:
        context.update(locals())

    normalized_delta_weights_dict, arena_LS_map = \
        synapses.generate_structured_weights(target_map=target_map,
                                             initial_weight_dict=initial_weights_by_source_gid_dict,
                                             input_rate_map_dict=input_rate_maps_by_source_gid_dict,
                                             syn_count_dict=syn_count_by_source_gid_dict,
                                             max_delta_weight=max_delta_weight, arena_x=arena_x, arena_y=arena_y,
                                             reference_weight_dict=reference_weights_by_source_gid_dict,
                                             reference_weights_are_delta=reference_weights_are_delta,
                                             reference_weights_namespace=reference_weights_namespace,
                                             optimize_method=optimize_method, verbose=verbose, plot=plot)

    output_syn_ids = np.empty(len(initial_weights_by_syn_id_dict), dtype='uint32')
    output_weights = np.empty(len(initial_weights_by_syn_id_dict), dtype='float32')
    i = 0
    for source_gid, this_weight in viewitems(normalized_delta_weights_dict):
        for syn_id in syn_ids_by_source_gid_dict[source_gid]:
            output_syn_ids[i] = syn_id
            output_weights[i] = this_weight
            i += 1
    output_weights_dict = {target_gid: {'syn_id': output_syn_ids,
                                        synapse_name: output_weights}}

    logger.info('destination: %s; gid %i; generated %s for %i synapses' %
                (destination, target_gid, output_weights_namespace, len(output_weights)))

    if not dry_run:
        this_output_weights_namespace = '%s %s' % (output_weights_namespace, arena_id)
        logger.info('Destination: %s; appending %s ...' % (destination, this_output_weights_namespace))
        append_cell_attributes(output_weights_path, destination, output_weights_dict,
                               namespace=this_output_weights_namespace)
        logger.info('Destination: %s; appended %s' % (destination, this_output_weights_namespace))
        output_weights_dict.clear()
        if output_features_path is not None:
            arena_rate_map = np.clip(arena_LS_map, 0., None) * peak_rate
            this_output_features_namespace = '%s %s' % (output_features_namespace, arena_id)
            cell_attr_dict = dst_input_features[destination]
            cell_attr_dict[target_gid]['Arena Rate Map'] = np.asarray(arena_rate_map.ravel(), dtype=np.float32)
            logger.info('Destination: %s; appending %s ...' % (destination, this_output_features_namespace))
            append_cell_attributes(output_features_path, destination, cell_attr_dict,
                                   namespace=this_output_features_namespace)

    if is_interactive:
        context.update(locals())


if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv)+1):],
         standalone_mode=False)
