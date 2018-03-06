
##
## Classes and procedures related to neuronal connectivity generation.
##

import sys, time, gc
import itertools
from collections import defaultdict
import numpy as np
import rbf
from rbf.interpolate import RBFInterpolant
import rbf.basis
from mpi4py import MPI
from neuroh5.io import NeuroH5CellAttrGen, bcast_cell_attributes, read_population_ranges, append_graph
import click
import utils
import logging
logger = logging.getLogger(__name__)

def list_index (element, lst):
    try:
        index_element = lst.index(element)
        return index_element
    except ValueError:
        return None


def random_choice_w_replacement(ranstream,n,p):
    return ranstream.multinomial(n,p.ravel())

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
    

    
class ConnectionProb(object):
    """An object of this class will instantiate functions that describe
    the connection probabilities for each presynaptic population. These
    functions can then be used to get the distribution of connection
    probabilities across all possible source neurons, given the soma
    coordinates of a destination (post-synaptic) neuron.
    """
    def __init__(self, destination_population, soma_coords, soma_distances, extent, nstdev = 5., res=5):
        """
        Warning: This method does not produce an absolute probability. It must be normalized so that the total area
        (volume) under the distribution is 1 before sampling.
        :param destination_population: post-synaptic population name
        :param soma_distances: a dictionary that contains per-population dicts of u, v distances of cell somas
        :param extent: dict: {source: 'width': (tuple of float), 'offset': (tuple of float)}
        """
        self.destination_population = destination_population
        self.soma_coords = soma_coords
        self.soma_distances = soma_distances
        self.p_dist = {}
        self.width  = {}
        self.offset = {}
        self.sigma  = {}

        
        for source_population in extent:
            extent_width  = extent[source_population]['width']
            if extent[source_population].has_key('offset'):
                extent_offset = extent[source_population]['offset']
            else:
                extent_offset = None
            self.width[source_population] = {'u': extent_width[0], 'v': extent_width[1]}
            self.sigma[source_population] = {axis: self.width[source_population][axis] / nstdev / np.sqrt(2.) for axis in self.width[source_population]}
            if extent_offset is None:
                self.offset[source_population] = {'u': 0., 'v': 0.}
            else:
                self.offset[source_population] = {'u': extent_offset[0], 'v': extent_offset[1]}
            self.p_dist[source_population] = (lambda source_population: np.vectorize(lambda distance_u, distance_v:
                                                np.exp(-(((abs(distance_u - self.offset[source_population]['u'])) /
                                                            self.sigma[source_population]['u'])**2. +
                                                            ((abs(distance_v - self.offset[source_population]['v'])) /
                                                            self.sigma[source_population]['v'])**2.)), otypes=[float]))(source_population)
    
    def filter_by_distance(self, destination_gid, source_population):
        """
        Given the id of a target neuron, returns the distances along u and v
        and the gids of source neurons whose axons potentially contact the target neuron.
        :param destination_gid: int
        :param source_population: string
        :return: tuple of array of int
        """
        destination_coords = self.soma_coords[self.destination_population][destination_gid]
        destination_distances = self.soma_distances[self.destination_population][destination_gid]
        
        source_soma_coords = self.soma_coords[source_population]
        source_soma_distances = self.soma_distances[source_population]

        destination_u, destination_v, destination_l  = destination_coords
        destination_distance_u, destination_distance_v = destination_distances
        
        distance_u_lst = []
        distance_v_lst = []
        source_u_lst   = []
        source_v_lst   = []
        source_gid_lst = []

        for (source_gid, coords) in source_soma_coords.iteritems():

            source_u, source_v, source_l = coords

            source_distance_u, source_distance_v  = source_soma_distances[source_gid]

            distance_u = abs(destination_distance_u - source_distance_u)
            distance_v = abs(destination_distance_v - source_distance_v)
            
            source_width = self.width[source_population]
            source_offset = self.offset[source_population]
                #print 'source_gid: %u destination u = %f destination v = %f source u = %f source v = %f source_distance_u = %f source_distance_v = %g' % (source_gid, destination_u, destination_v, source_u, source_v, source_distance_u, source_distance_v)
            if ((distance_u <= source_width['u'] / 2. + source_offset['u']) &
                (distance_v <= source_width['v'] / 2. + source_offset['v'])):
                source_u_lst.append(source_u)
                source_v_lst.append(source_v)
                distance_u_lst.append(distance_u)
                distance_v_lst.append(distance_v)
                source_gid_lst.append(source_gid)

        return destination_u, destination_v, np.asarray(source_u_lst), np.asarray(source_v_lst), np.asarray(distance_u_lst), np.asarray(distance_v_lst), np.asarray(source_gid_lst, dtype=np.uint32)

    
    def get_prob(self, destination_gid, source, plot=False):
        """
        Given the soma coordinates of a destination neuron and a population source, return an array of connection 
        probabilities and an array of corresponding source gids.
        :param destination_gid: int
        :param source: string
        :param plot: bool
        :return: array of float, array of int
        """
        destination_u, destination_v, source_u, source_v, distance_u, distance_v, source_gid = self.filter_by_distance(destination_gid, source)
        p = self.p_dist[source](distance_u, distance_v)
        psum = np.sum(p)
        if psum > 0.:
            p1 = softmax(p)
        else:
            p1 = p
        assert((p1 >= 0.).all() and (p1 <= 1.).all())
        if plot:
            plt.scatter(distance_u, distance_v, c=p1)
            plt.title(source+' -> '+target)
            plt.xlabel('Septotemporal distance (um)')
            plt.ylabel('Transverse distance (um)')
            plt.show()
            plt.close()
        return p1, source_gid, distance_u, distance_v


def get_volume_distances (ip_vol, res=2, step=1, verbose=False):
    if verbose:
        logger.setLevel(logging.INFO)

    if verbose:
        logger.info('Resampling volume...')
    U, V, L = ip_vol._resample_uvl(res, res, res)

    if verbose:
        logger.info('Computing U distances...')
    ldist_u, obs_dist_u = ip_vol.point_distance(U, V, L, axis=0)

    obs_uvl = np.array([np.concatenate(obs_dist_u[0]), \
                        np.concatenate(obs_dist_u[1]), \
                        np.concatenate(obs_dist_u[2])]).T
    sample_inds = np.arange(0, obs_uvl.shape[0]-1, step)
    obs_u = obs_uvl[sample_inds,:]
    distances_u = np.concatenate(ldist_u)[sample_inds]
    #self.dist_u = RBFInterpolant(,distances_u[sample_inds],order=1,basis=rbf.basis.phs3,extrapolate=True)
    #del ldist_u, obs_dist_u
        
    if verbose:
        logger.info('Computing V distances...')
    ldist_v, obs_dist_v = ip_vol.point_distance(U, V, L, axis=1)
    obs_uvl = np.array([np.concatenate(obs_dist_v[0]), \
                        np.concatenate(obs_dist_v[1]), \
                        np.concatenate(obs_dist_v[2])]).T
    sample_inds = np.arange(0, obs_uvl.shape[0]-1, step)
    obs_v = obs_uvl[sample_inds,:]    
    distances_v = np.concatenate(ldist_v)[sample_inds]
    #self.dist_v = RBFInterpolant(obs_uvl[sample_inds,:],distances_v[sample_inds],order=1,basis=rbf.basis.phs3,extrapolate=True)
    #del ldist_v, obs_dist_v

    return (distances_u, obs_u, distances_v, obs_v)



        
def get_soma_distances(comm, dist_u, dist_v, soma_coords):
    rank = comm.rank
    size = comm.size

    soma_distances = {}
    for pop, coords_dict in soma_coords.iteritems():
        local_dist_dict = {}
        for gid, coords in coords_dict.iteritems():
            if gid % size == rank:
                soma_u, soma_v, soma_l = coords
                uvl_obs = np.array([soma_u,soma_v,soma_l]).reshape(1,3)
                distance_u = dist_u(uvl_obs)
                distance_v = dist_v(uvl_obs)
                local_dist_dict[gid] = (distance_u, distance_v)
        dist_dicts = comm.allgather(local_dist_dict)
        combined_dist_dict = {}
        for dist_dict in dist_dicts:
            for k, v in dist_dict.iteritems():
                combined_dist_dict[k] = v
        soma_distances[pop] = combined_dist_dict
    return soma_distances



def choose_synapse_projection (ranstream_syn, syn_layer, swc_type, syn_type, population_dict, projection_synapse_dict):
    """Given a synapse projection, SWC synapse location, and synapse
    type, chooses a projection from the given projection dictionary
    based on 1) whether the projection properties match the given
    synapse properties and 2) random choice between all the projections that satisfy the given criteria.
    :param ranstream_syn: random state object
    :param syn_layer: synapse layer
    :param swc_type: SWC location for synapse (soma, axon, apical, basal)
    :param syn_type: synapse type (excitatory, inhibitory, neuromodulatory)
    :param population_dict: mapping of population names to population indices
    :param projection_synapse_dict: mapping of projection names to a tuple of the form: <syn_layer, swc_type, syn_type, syn_proportion>
    """
    ivd = { v:k for k,v in population_dict.iteritems() }
    projection_lst = []
    projection_prob_lst = []
    for k, v in projection_synapse_dict.iteritems():
        if (syn_type in v[2]) and (swc_type in v[1]):
            ord_index = list_index(syn_layer, v[0])
            if ord_index is not None:
                projection_lst.append(population_dict[k])
                projection_prob_lst.append(v[3][ord_index])
    if len(projection_lst) > 1:
       candidate_projections = np.asarray(projection_lst)
       candidate_probs       = np.asarray(projection_prob_lst)
       projection            = ranstream_syn.choice(candidate_projections, 1, p=candidate_probs)[0]
    elif len(projection_lst) > 0:
       projection = projection_lst[0]
    else:
       projection = None

    if projection is not None:
        return ivd[projection]
    else:
        return None

 
def generate_synaptic_connections(ranstream_syn,
                                  ranstream_con,
                                  destination_gid,
                                  synapse_dict,
                                  population_dict,
                                  projection_synapse_dict,
                                  projection_prob_dict,
                                  connection_dict,
                                  random_choice=random_choice_w_replacement):
    """Given a set of synapses for a particular gid, projection configuration, projection and connection probability dictionaries,
    generates a set of possible connections for each synapse. The procedure first assigns each synapse to a projection, using 
    the given proportions of each synapse type, and then chooses source gids for each synapse using the given projection probability dictionary.
    :param ranstream_syn: random stream for the synapse partitioning step
    :param ranstream_con: random stream for the choosing source gids step
    :param destination_gid: destination gid
    :param synapse_dict: synapse configurations, a dictionary with fields: 1) syn_ids (synapse ids) 2) syn_types (excitatory, inhibitory, etc).,
                        3) swc_types (SWC types(s) of synapse location in the neuronal morphological structure 3) syn_layers (synapse layer placement)
    :param population_dict: mapping of population names to population indices
    :param projection_synapse_dict: mapping of projection names to a tuple of the form: <syn_layer, swc_type, syn_type, syn_proportion>
    :param projection_prob_dict: mapping of presynaptic population names to sets of source probabilities and source gids
    :param connection_dict: output connection dictionary
    :param random_choice: random choice procedure (default uses np.ranstream.multinomial)
    """
    ## assign each synapse to a projection
    synapse_prj_partition = defaultdict(list)
    for (syn_id,syn_type,swc_type,syn_layer) in itertools.izip(synapse_dict['syn_ids'],
                                                               synapse_dict['syn_types'],
                                                               synapse_dict['swc_types'],
                                                               synapse_dict['syn_layers']):
        projection = choose_synapse_projection(ranstream_syn, syn_layer, swc_type, syn_type,
                                               population_dict, projection_synapse_dict)
        if projection is None:
            print 'Projection is none for syn_type = ', syn_type, 'swc_type = ', swc_type, ' syn_layer = ', syn_layer
            print projection_synapse_dict
        assert(projection is not None)
        synapse_prj_partition[projection].append(syn_id)

    ## Choose source connections based on distance-weighted probability
    count = 0
    for projection, syn_ids in synapse_prj_partition.iteritems():
        count += len(syn_ids)
        source_probs, source_gids, distances_u, distances_v = projection_prob_dict[projection]
        if len(source_gids) > 0:
            source_gid_counts = random_choice(ranstream_con,len(syn_ids),source_probs)
            uv_distance_sums = np.add(distances_u, distances_v, dtype=np.float32)
            distances = np.repeat(uv_distance_sums, source_gid_counts)
            gid_dict = connection_dict[projection]
            gid_dict[destination_gid] = (np.repeat(source_gids, source_gid_counts),
                                             { 'Synapses' : { 'syn_id': np.asarray (syn_ids, dtype=np.uint32) },
                                                   'Connections' : { 'distance': distances }
                                             } )
            connection_dict[projection] = gid_dict

        
    ## If any projection does not have connections associated with it, create empty entries
    ## This is necessary for the parallel graph append operation, which performs a separate append for each projection,
    ## and therefore needs all ranks to participate in it.
    for projection in projection_synapse_dict.iterkeys():
        if not connection_dict.has_key(projection):
            gid_dict = connection_dict[projection]
            gid_dict[destination_gid] = ( np.asarray ([], dtype=np.uint32),
                                              { 'Synapses' : { 'syn_id': np.asarray ([], dtype=np.uint32) },
                                                    'Connections' : { 'distance': np.asarray ([], dtype=np.float32) }
                                              } )
            connection_dict[projection] = gid_dict

    return count


def generate_uv_distance_connections(comm, population_dict, connection_config, connection_prob, forest_path,
                                     synapse_seed, synapse_namespace, 
                                     connectivity_seed, connectivity_namespace, connectivity_path,
                                     io_size, chunk_size, value_chunk_size, cache_size, write_size=1,
                                     verbose=False):
    """Generates connectivity based on U, V distance-weighted probabilities.
    :param comm: mpi4py MPI communicator
    :param connection_config: connection configuration object (instance of env.ConnectionGenerator)
    :param connection_prob: ConnectionProb instance
    :param forest_path: location of file with neuronal trees and synapse information
    :param synapse_seed: random seed for synapse partitioning
    :param synapse_namespace: namespace of synapse properties
    :param connectivity_seed: random seed for connectivity generation
    :param connectivity_namespace: namespace of connectivity attributes
    :param io_size: number of I/O ranks to use for parallel connectivity append
    :param chunk_size: HDF5 chunk size for connectivity file (pointer and index datasets)
    :param value_chunk_size: HDF5 chunk size for connectivity file (value datasets)
    :param cache_size: how many cells to read ahead
    :param write_size: how many cells to write out at the same time
    """
    if verbose:
        logger.setLevel(logging.INFO)
        
    rank = comm.rank

    if io_size == -1:
        io_size = comm.size
    if rank == 0:
        logger.info('%i ranks have been allocated' % comm.size)
    sys.stdout.flush()

    start_time = time.time()

    ranstream_syn = np.random.RandomState()
    ranstream_con = np.random.RandomState()

    destination_population = connection_prob.destination_population
    
    source_populations = connection_config[destination_population].keys()

    for source_population in source_populations:
        if rank == 0:
            logger.info('%s -> %s:' % (source_population, destination_population))
            logger.info(str(connection_config[destination_population][source_population]))
                           
    projection_synapse_dict = {source_population: (connection_config[destination_population][source_population].synapse_layers,
                                                   set(connection_config[destination_population][source_population].synapse_locations),
                                                   set(connection_config[destination_population][source_population].synapse_types),
                                                   connection_config[destination_population][source_population].synapse_proportions)
                                for source_population in source_populations}
    total_count = 0
    gid_count   = 0
    connection_dict = defaultdict(lambda: {})
    for destination_gid, synapse_dict in NeuroH5CellAttrGen(forest_path, destination_population, io_size=io_size,
                                                            cache_size=cache_size, namespace=synapse_namespace, comm=comm):
        last_time = time.time()
        if destination_gid is None:
            logger.info('Rank %i destination gid is None' % rank)
        else:
            logger.info('Rank %i received attributes for destination: %s, gid: %i' % (rank, destination_population, destination_gid))
            ranstream_con.seed(destination_gid + connectivity_seed)
            ranstream_syn.seed(destination_gid + synapse_seed)

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
                    

            
            count = generate_synaptic_connections(ranstream_syn,
                                                   ranstream_con,
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
            append_graph(connectivity_path, projection_dict, io_size=io_size, comm=comm)
            if rank == 0:
                if connection_dict:
                    for (prj, prj_dict) in  connection_dict.iteritems():
                        logger.info("%s: %s" % (prj, str(prj_dict.keys())))
                    logger.info('Appending connectivity for %i projections took %i s' % (len(connection_dict), time.time() - last_time))
            connection_dict.clear()
            gc.collect()
            
        gid_count += 1

    global_count = comm.gather(total_count, root=0)
    if rank == 0:
        logger.info('%i ranks took %i s to generate %i edges' % (comm.size, time.time() - start_time, np.sum(global_count)))


