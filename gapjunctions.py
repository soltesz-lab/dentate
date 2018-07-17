
"""Procedures related to gap junction connectivity generation. """

import sys, time, string, math, itertools
from collections import defaultdict
import numpy as np
from mpi4py import MPI
from neuron import h
from neuroh5.io import read_population_ranges, read_tree_selection, append_graph


## This logger will inherit its setting from its root logger, dentate,
## which is created in module env
logger = utils.get_module_logger(__name__)


## Compartment weights
## comp_coeff        = [-0.0315,9.4210];
## HIPP_long_weights   = [37.5;112.5;187.5]*linear_coeff(1) + linear_coeff(2);
## HIPP_short_weights  = [25;75;125]*linear_coeff(1) + linear_coeff(2);
## Apical_weights      = [37.5;112.5;187.5;262.5]*linear_coeff(1) + linear_coeff(2);
## Basal_weights       = [25;75;125;175]*linear_coeff(1) + linear_coeff(2);


## Distance-dependent probability (based on Otsuka)
def connection_prob (distance):
    connection_params[0] + (connection_params[1] - connection_params[0]) / (1 + 10. ** ((connection_params[2] - distance) * connection_params[3]))

    
## Scale gap junction strength based on polynomial fit to Amitai distance-dependence
def coupling_strength(distance, cc):
    weights = cc_params[0] * distance ** 2. + cc_params[1] * distance + cc_params[2]
    weights = weights / np.max(weights)
    return cc / np.mean(weights) * weights

## Connections based on weighted distance
## selected = datasample(transpose(1:length(distance)),round(gj_prob(pre_type,post_type)*length(distance)),'Weights',prob,'Replace',false);

def generate_gap_junctions(gj_config, ranstream_gj, gj_probs, gj_distances, gids_a, gids_b,
                           tree_dict_a, tree_dict_b, gj_dict)
    k = round(gj_config.prob * len(gj_distances))
    selected = np.random.choice(np.arange(0, len(gj_distances)), size=k, replace=False, p=gj_probs)
    count = len(selected)
    
    gid_dict = defaultdict(list)
    for i in selected:
        gid_a = gids_a[i]
        gid_b = gids_b[i]
        gid_a_gjs = gid_dict[gid_a] 
        gid_a_gjs.append(gid_b)

    for gid_a,gids_b in gid_dict.iteritems():
        gid_tree_a = tree_dict_a[gid_a]
        for gid_b in gids_b:
            sections =
            positions =
            conductances = 
            gj_dict[gid_a] = ( np.asarray(gids_b, dtype=np.uint32),
                                   { 'Location' : { 'section': np.asarray (sections, dtype=np.uint32),
                                                    'position': np.asarray (positions, dtype=np.float32) },
                                     'Coupling' : { 'conductance' : np.asarray (conductances, dtype=np.float32) } } )
            
    return count

        

def generate_gj_connections(comm, population_dict, gj_config, connection_prob, forest_path,
                            synapse_seed, connectivity_seed, cluster_seed,
                            synapse_namespace, connectivity_namespace, connectivity_path,
                            io_size, chunk_size, value_chunk_size, cache_size,
                            dry_run=False):
    
    """Generates gap junction connectivity based on Euclidean-distance-weighted probabilities.
    :param comm: mpi4py MPI communicator
    :param connection_config: connection configuration object (instance of env.ConnectionGenerator)
    :param connection_prob: ConnectionProb instance
    :param forest_path: location of file with neuronal trees and synapse information
    :param synapse_seed: random seed for synapse partitioning
    :param connectivity_seed: random seed for connectivity generation
    :param cluster_seed: random seed for determining connectivity clustering for repeated connections from the same source
    :param synapse_namespace: namespace of synapse properties
    :param connectivity_namespace: namespace of connectivity attributes
    :param io_size: number of I/O ranks to use for parallel connectivity append
    :param chunk_size: HDF5 chunk size for connectivity file (pointer and index datasets)
    :param value_chunk_size: HDF5 chunk size for connectivity file (value datasets)
    :param cache_size: how many cells to read ahead
    """
        
    rank = comm.rank

    if io_size == -1:
        io_size = comm.size
    if rank == 0:
        logger.info('%i ranks have been allocated' % comm.size)
    sys.stdout.flush()

    start_time = time.time()

    ranstream_gj = np.random.RandomState()

    population_pairs = gj_config['Connection Probabilities'].keys()
    cc_params = np.asarray(gj_config['Coupling Parameters'])
    connection_params = np.asarray(['Connection Parameters'])

    for pp in population_pairs:
        if rank == 0:
            logger.info('%s <-> %s:' % (pp[0], pp[1]))
                           
    gj_config_dict = { pp: (connection_config[destination_population][source_population].synapse_layers,
                                set(connection_config[destination_population][source_population].synapse_locations),
                                set(connection_config[destination_population][source_population].synapse_types),
                                connection_config[destination_population][source_population].synapse_proportions)
                    for pp in population_pairs }
    total_count = 0
    gid_count   = 0
    connection_dict = defaultdict(lambda: {})

    for (i, (pp, gj_config)) in enumerate(gj_config_dict.iteritems()):

        ranstream_gj.seed(gj_seed + i)
        
        population_a = pp[0]
        population_b = pp[1]
        
        coords_a_dict = read_cell_attributes(coords_path, population_a, namespace=coords_namespace)
        coords_b_dict = read_cell_attributes(coords_path, population_b, namespace=coords_namespace)

        coords_a = {}
        for (gid, coords_dict) in coords_a_dict:
            cell_x = coords_dict['X Coordinate'][0]
            cell_y = coords_dict['Y Coordinate'][0]
            cell_z = coords_dict['Z Coordinate'][0]
            coords_a[gid] = (np.asarray([cell_x, cell_y, cell_z]))

        coords_b = []
        for (gid, coords_dict) in coords_b_dict:
            cell_x = coords_dict['X Coordinate'][0]
            cell_y = coords_dict['Y Coordinate'][0]
            cell_z = coords_dict['Z Coordinate'][0]
            coords_b[gid] = (np.asarray([cell_x, cell_y, cell_z]))
            
        dist_dict = make_distance_pairs(coords_a, coords_b)

        gj_prob_dict = filter_by_prob(dist_dict, gj_config.connection_params)

        gj_probs = []
        gj_distances = []
        gids_a = []
        gids_b = []
        for k, v in gj_prob_dict.iteritems():
            if k[0] % rank == 0:
                gids_a.append(k[0])
                gids_b.append(k[1])
                gj_probs.append(v[0])
                gj_distances.append(v[1])

        gj_probs = np.asarray(gj_probs, dtype=np.float32)
        gj_probs = gj_probs / gj_probs.sum()
        gj_distances = np.asarray(gj_distances, dtype=np.float32)
        gids_a = np.asarray(gids_a, dtype=np.uint32)
        gids_b = np.asarray(gids_b, dtype=np.uint32)
                
        tree_dict_a = {}
        selection_a = set(gids_a)
        (tree_iter_a, _) = read_tree_selection(forest_path, population_a, list(gids_a))
        for (gid,tree_dict) in tree_iter_a:
            tree_dict_a[gid] = tree_dict

        
        tree_dict_b = {}
        selection_b = set(gids_b)
        (tree_iter_b, _) = read_tree_selection(forest_path, population_b, list(gids_b))
        for (gid,tree_dict) in tree_iter_b:
            tree_dict_b[gid] = tree_dict

        gj_dict = {}
        count = generate_gap_junctions(ranstream_gj,
                                       gj_probs, gj_distances, gids_a, gids_b,
                                       tree_dict_a, tree_dict_b,
                                       gj_dict)
        
        gj_graph_dict = { pp[0]: { pp[1]: gj_dict } }

        if not dry_run:
            append_graph(connectivity_path, gj_graph_dict, io_size=io_size, comm=comm)

        total_count += count
            
    global_count = comm.gather(total_count, root=0)
    if rank == 0:
        logger.info('%i ranks took %i s to generate %i edges' % (comm.size, time.time() - start_time, np.sum(global_count)))
