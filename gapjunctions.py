
"""Procedures related to gap junction connectivity generation. """

import itertools
from collections import defaultdict
import sys, os.path, string, math
from neuron import h
import numpy as np

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

def generate_gj_connections(comm, population_dict, gj_config, connection_prob, forest_path,
                            synapse_seed, connectivity_seed, cluster_seed,
                            synapse_namespace, connectivity_namespace, connectivity_path,
                            io_size, chunk_size, value_chunk_size, cache_size, write_size=1,
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
    :param write_size: how many cells to write out at the same time
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
    gj_dict = {}

    for (pp, gj_config) in gj_config_dict.iteritems():

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
            
        dist_dict = measure_euclidean_distances(coords_a, coords_b)

        filtered_dist_dict = filter_by_distance(dist_dict, connection_params)

        gids_a, gids_b = 
        
    for gid, synapse_dict in NeuroH5CellAttrGen(forest_path, destination_population, io_size=io_size,
                                                cache_size=cache_size, comm=comm):
        last_time = time.time()
        if gid is None:
            logger.info('Rank %i gid is None' % rank)
        else:
            logger.info('Rank %i received attributes for population: %s, gid: %i' % (rank, destination_population, destination_gid))
            ranstream_gj.seed(destination_gid + connectivity_seed)

            projection_prob_dict = {}
            for source_population in source_populations:
                probs, source_gids, distances_u, distances_v = connection_prob.get_prob(destination_gid, source_population)
                projection_prob_dict[source_population] = (probs, source_gids, distances_u, distances_v)
                if len(distances_u) > 0:
                    max_u_distance = np.max(distances_u)
                    min_u_distance = np.min(distances_u)
                    logger.info('Rank %i has %d possible sources from population %s for destination: %s, gid: %i; max U distance: %f min U distance: %f' % (rank, len(source_gids), source_population, destination_population, destination_gid, max_u_distance, min_u_distance))
                else:
                    logger.info('Rank %i has %d possible sources from population %s for destination: %s, gid: %i' % (rank, len(source_gids), source_population, destination_population, destination_gid))
                    

            
            count = generate_synaptic_connections(rank,
                                                  ranstream_syn,
                                                  ranstream_con,
                                                  cluster_seed+destination_gid,
                                                  destination_gid,
                                                  synapse_dict,
                                                  population_dict,
                                                  projection_synapse_dict,
                                                  projection_prob_dict,
                                                  connection_dict)
            total_count += count
            
            logger.info('Rank %i took %i s to compute %d edges for destination: %s, gid: %i' % (rank, time.time() - last_time, count, destination_population, destination_gid))
            sys.stdout.flush()

        if gid_count % write_size == 0:
            last_time = time.time()
            if len(connection_dict) > 0:
                projection_dict = { destination_population: connection_dict }
            else:
                projection_dict = {}
            if not dry_run:
                append_graph(connectivity_path, projection_dict, io_size=io_size, comm=comm)
            if rank == 0:
                if connection_dict:
                    for (prj, prj_dict) in  connection_dict.iteritems():
                        logger.info("%s: %s" % (prj, str(prj_dict.keys())))
                    logger.info('Appending connectivity for %i projections took %i s' % (len(connection_dict), time.time() - last_time))
            projection_dict.clear()
            connection_dict.clear()
            gc.collect()
            
        gid_count += 1

    last_time = time.time()
    if len(connection_dict) > 0:
        projection_dict = { destination_population: connection_dict }
    else:
        projection_dict = {}
    if not dry_run:
        append_graph(connectivity_path, projection_dict, io_size=io_size, comm=comm)
    if rank == 0:
        if connection_dict:
            for (prj, prj_dict) in  connection_dict.iteritems():
                logger.info("%s: %s" % (prj, str(prj_dict.keys())))
                logger.info('Appending connectivity for %i projections took %i s' % (len(connection_dict), time.time() - last_time))

    global_count = comm.gather(total_count, root=0)
    if rank == 0:
        logger.info('%i ranks took %i s to generate %i edges' % (comm.size, time.time() - start_time, np.sum(global_count)))




