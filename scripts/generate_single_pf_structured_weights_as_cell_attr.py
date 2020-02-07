import sys, os, time, gc, click, logging, pprint
from collections import defaultdict
from functools import reduce
import numpy as np
from mpi4py import MPI
import neuroh5
from neuroh5.io import append_cell_attributes, read_population_ranges, read_cell_attribute_selection, read_graph_selection
import dentate
from dentate.env import Env
from dentate import utils, stimulus, synapses
from dentate.utils import *
import h5py

def get_place_rate_map(x0, y0, width, x, y):
    return np.exp(-(((x - x0) / (width / 3. / np.sqrt(2.)))) ** 2.) * \
           np.exp(-(((y - y0) / (width / 3. / np.sqrt(2.)))) ** 2.)


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
@click.option("--output-weights-path", required=False, type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option("--output-features-path", required=False, type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option("--weights-path", required=False, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--h5types-path", required=False, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--synapse-name", type=str, default='AMPA')
@click.option("--initial-weights-namespace", type=str, default='Weights')
@click.option("--structured-weights-namespace", type=str, default='Structured Weights')
@click.option("--connections-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--destination", '-d', type=str)
@click.option("--sources", '-s', type=str, multiple=True)
@click.option("--arena-id", '-a', type=str, default='A')
@click.option("--field-width-scale", type=float, default=1.2)
@click.option("--max-iter", type=int, default=10)
@click.option("--verbose", "-v", is_flag=True)
@click.option("--dry-run", is_flag=True)
@click.option("--interactive", is_flag=True)
def main(config, coordinates, gid, field_width, peak_rate, input_features_path, input_features_namespaces,
         output_weights_path, output_features_path, weights_path, h5types_path, synapse_name, initial_weights_namespace,
         structured_weights_namespace, connections_path, destination, sources, arena_id, field_width_scale, max_iter, 
         verbose, dry_run, interactive):
    """

    :param config: str (path to .yaml file)
    :param weights_path: str (path to .h5 file)
    :param initial_weights_namespace: str
    :param structured_weights_namespace: str
    :param connections_path: str (path to .h5 file)
    :param destination: str
    :param sources: list of str
    :param verbose:
    :param dry_run:
    :return:
    """

    utils.config_logging(verbose)
    logger = utils.get_script_logger(__file__)

    env = Env(config_file=config)

    if output_weights_path is None:
        if weights_path is None:
            raise RuntimeError('Output weights path must be specified when weights path is not specified.')
        output_weights_path = weights_path
    
    if (not dry_run):
        if not os.path.isfile(output_weights_path):
            if weights_path is not None:
                input_file  = h5py.File(weights_path,'r')
            elif h5types_path is not None:
                input_file  = h5py.File(h5types_path,'r')
            else:
                raise RuntimeError('h5types input path must be specified when weights path is not specified.')
            output_file = h5py.File(output_weights_path,'w')
            input_file.copy('/H5Types',output_file)
            input_file.close()
            output_file.close()

    
    this_input_features_namespaces = ['%s %s' % (input_features_namespace, arena_id) for input_features_namespace in input_features_namespaces]

    initial_weights_dict = None
    if weights_path is not None:
        logger.info('Reading initial weights data from %s...' % weights_path)
        cell_attributes_dict = read_cell_attribute_selection(weights_path, destination, 
                                                             namespaces=[initial_weights_namespace],
                                                             selection=[gid])
                                                            
        if initial_weights_namespace in cell_attributes_dict:
            initial_weights_iter = cell_attributes_dict[initial_weights_namespace]
            initial_weights_dict = { gid: attr_dict for gid, attr_dict in initial_weights_iter }
        else:
            raise RuntimeError('Initial weights namespace %s was not found in file %s' % (initial_weights_namespace, weights_path))
    
        logger.info('Rank %i; destination: %s; read synaptic weights for %i cells' %
                    (rank, destination, len(initial_weights_dict)))


    features_attr_names = ['Num Fields', 'Field Width', 'Peak Rate', 'X Offset', 'Y Offset', 'Arena Rate Map']
    
    local_random = np.random.RandomState()

    seed_offset = int(env.model_config['Random Seeds']['GC Structured Weights'])
    local_random.seed(int(gid + seed_offset))
    
    spatial_resolution = env.stimulus_config['Spatial Resolution'] # cm

    arena = env.stimulus_config['Arena'][arena_id]
    default_run_vel = arena.properties['default run velocity']  # cm/s

    x, y = stimulus.get_2D_arena_spatial_mesh(arena, spatial_resolution)
    
    plasticity_kernel = lambda x, y, x_loc, y_loc, sx, sy: gauss2d(x-x_loc, y-y_loc, sx=sx, sy=sy)
    plasticity_kernel = np.vectorize(plasticity_kernel, excluded=[2,3,4,5])


    dst_input_features = defaultdict(dict)
    num_fields = len(coordinates)
    this_field_width = np.array([field_width]*num_fields, dtype=np.float32)
    this_peak_rate = np.array([peak_rate]*num_fields, dtype=np.float32)
    this_x0 = np.array([x for x, y in coordinates], dtype=np.float32)
    this_y0 = np.array([y for x, y in coordinates], dtype=np.float32)
    this_rate_map = np.asarray(get_rate_map(this_x0, this_y0, this_field_width, this_peak_rate, x, y),
                               dtype=np.float32)
    selectivity_type = env.selectivity_types['place']
    dst_input_features[destination][gid] = {
        'Selectivity Type': np.array([selectivity_type], dtype=np.uint8),
        'Num Fields': np.array([num_fields], dtype=np.uint8),
        'Field Width': this_field_width,
        'Peak Rate': this_peak_rate,
        'X Offset': this_x0,
        'Y Offset': this_y0,
        'Arena Rate Map': this_rate_map.ravel() }

    selection=[gid]
    structured_weights_dict = {}
    source_syn_dict = defaultdict(lambda: defaultdict(list))
    syn_weight_dict = {}
    if weights_path is not None:
        initial_weights_iter = read_cell_attribute_selection(weights_path, destination, 
                                                                 namespace=initial_weights_namespace, 
                                                                 selection=selection)
        syn_weight_attr_dict = dict(initial_weights_iter)

        syn_ids = syn_weight_attr_dict[gid]['syn_id']
        weights = syn_weight_attr_dict[gid][synapse_name]
                    
        for (syn_id, weight) in zip(syn_ids, weights):
            syn_weight_dict[int(syn_id)] = float(weight) 

        logger.info('destination: %s; gid %i; received synaptic weights for %i synapses' %
                        (destination, gid, len(syn_weight_dict)))

    (graph, edge_attr_info) = read_graph_selection(file_name=connections_path,
                                                   selection=[gid],
                                                   namespaces=['Synapses'])
    syn_id_attr_index = None
    for source, edge_iter in viewitems(graph[destination]):
        this_edge_attr_info = edge_attr_info[destination][source]
        if 'Synapses' in this_edge_attr_info and \
           'syn_id' in this_edge_attr_info['Synapses']:
            syn_id_attr_index = this_edge_attr_info['Synapses']['syn_id']
        for (destination_gid, edges) in edge_iter:
            assert destination_gid == gid
            source_gids, edge_attrs = edges

            syn_ids = edge_attrs['Synapses'][syn_id_attr_index]
            this_source_syn_dict = source_syn_dict[source]
            count = 0
            for i in range(len(source_gids)):
                this_source_gid = source_gids[i]
                this_syn_id = syn_ids[i]
                this_syn_wgt = syn_weight_dict.get(this_syn_id, 0.0)
                this_source_syn_dict[this_source_gid].append((this_syn_id, this_syn_wgt))
                count += 1
            logger.info('destination: %s; gid %i; %d synaptic weights from source population %s' %
                        (destination, gid, count, source))
                    
    src_input_features = defaultdict(dict)
    for source in sources:
        source_gids = list(source_syn_dict[source].keys())
        for input_features_namespace in this_input_features_namespaces:
            input_features_iter = read_cell_attribute_selection(input_features_path, source, 
                                                                namespace=input_features_namespace,
                                                                mask=set(features_attr_names), 
                                                                selection=source_gids)
            this_src_input_features = src_input_features[source]
            count = 0
            for gid, attr_dict in input_features_iter:
                this_src_input_features[gid] = attr_dict
                count += 1
            logger.info('Read %s feature data for %i cells in population %s' % (input_features_namespace, count, source))

    this_syn_weights = \
      synapses.generate_structured_weights(destination_gid,
                                           destination,
                                           synapse_name,
                                           sources,
                                           dst_input_features,
                                           src_input_features,
                                           source_syn_dict,
                                           spatial_mesh=(x,y),
                                           plasticity_kernel=plasticity_kernel,
                                           local_random=local_random,
                                           interactive=interactive)

    assert this_syn_weights is not None
    structured_weights_dict[destination_gid] = this_syn_weights
    logger.info('destination: %s; gid %i; generated structured weights for %i inputs'
                   % (destination, destination_gid, len(this_syn_weights['syn_id'])))
    gc.collect()
    if not dry_run:
            
        logger.info('Destination: %s; appending structured weights...' % (destination))
        this_structured_weights_namespace = '%s %s' % (structured_weights_namespace, arena_id)
        append_cell_attributes(output_weights_path, destination, structured_weights_dict,
                               namespace=this_structured_weights_namespace)
        logger.info('Destination: %s; appended structured weights' % (destination))
        structured_weights_dict.clear()
        if output_features_path is not None:
            output_features_namespace = 'Place Selectivity %s' % arena_id
            cell_attr_dict = dst_input_features[destination]
            logger.info('Destination: %s; appending features...' % (destination))
            append_cell_attributes(output_features_path, destination,
                                   cell_attr_dict, namespace=output_features_namespace)
            
            
        gc.collect()
            
    del(syn_weight_dict)
    del(src_input_features)
    del(dst_input_features)



if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv)+1):])
