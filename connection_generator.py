import sys, time, gc
import itertools
import numpy as np
import itertools
from collections import defaultdict
from mpi4py import MPI
from neuroh5.io import NeuroH5CellAttrGen, bcast_cell_attributes, read_population_ranges, append_graph
import click
import utils

def list_index (element, lst):
    try:
        index_element = lst.index(element)
        return index_element
    except ValueError:
        return None

class ConnectionProb(object):
    """An object of this class will instantiate functions that describe
    the connection probabilities for each presynaptic population. These
    functions can then be used to get the distribution of connection
    probabilities across all possible source neurons, given the soma
    coordinates of a destination (post-synaptic) neuron.
    """
    def __init__(self, destination_population, soma_coords, ip_surface, extent, nstdev = 5.):
        """
        Warning: This method does not produce an absolute probability. It must be normalized so that the total area
        (volume) under the distribution is 1 before sampling.
        :param destination_population: post-synaptic population name
        :param soma_coords: a dictionary that contains per-population dicts of u, v coordinates of cell somas
        :param ip_surface: surface interpolation function, generated by e.g. BSplineSurface
        :param extent: dict: {source: 'width': (tuple of float), 'offset': (tuple of float)}
        """
        self.destination_population = destination_population
        self.soma_coords = soma_coords
        self.ip_surface  = ip_surface
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
                                                np.exp(-(((abs(distance_u) - self.offset[source_population]['u']) /
                                                            self.sigma[source_population]['u'])**2. +
                                                            ((abs(distance_v) - self.offset[source_population]['v']) /
                                                            self.sigma[source_population]['v'])**2.))))(source_population)
            

    def filter_by_arc_lengths(self, destination_gid, source_population, npoints=100):
        """
        Given the id of a target neuron, returns the arc distances along u and v
        and the gids of source neurons whose axons potentially contact the target neuron.
        :param destination_gid: int
        :param source_population: string
        :param npoints: number of points for arc length approximation [default: 250]
        :return: tuple of array of int
        """
        destination_coords = self.soma_coords[self.destination_population][destination_gid]
        destination_u = destination_coords['U Coordinate']
        destination_v = destination_coords['V Coordinate']
        source_soma_coords = self.soma_coords[source_population]
        
        source_distance_u_lst = []
        source_distance_v_lst = []
        source_gid_lst        = []

        for (source_gid, coords_dict) in source_soma_coords.iteritems():

            source_u = coords_dict['U Coordinate']
            source_v = coords_dict['V Coordinate']
            
            U = np.linspace(destination_u, source_u, npoints)
            V = np.linspace(destination_v, source_v, npoints)

            source_distance_u = self.ip_surface.arc_length(U, destination_v)
            source_distance_v = self.ip_surface.arc_length(destination_u, V)

            source_width = self.width[source_population]
            source_offset = self.offset[source_population]
            if ((source_distance_u <= source_width['u'] / 2. + source_offset['u']) &
                (source_distance_v <= source_width['v'] / 2. + source_offset['v'])):
                source_distance_u_lst.append(source_distance_u)
                source_distance_v_lst.append(source_distance_v)
                source_gid_lst.append(source_gid)

        return np.asarray(source_distance_u_lst), np.asarray(source_distance_v_lst), np.asarray(source_gid_lst)

    def get_prob(self, destination_gid, source, plot=False):
        """
        Given the soma coordinates of a destination neuron and a population source, return an array of connection 
        probabilities and an array of corresponding source gids.
        :param destination: string
        :param source: string
        :param destination_gid: int
        :param soma_coords: nested dict of array
        :param distance_U: array of float
        :param distance_V: array of float
        :param plot: bool
        :return: array of float, array of int
        """
        source_distance_u, source_distance_v, source_gid = self.filter_by_arc_lengths(destination_gid, source)
        p = self.p_dist[source](source_distance_u, source_distance_v)
        p /= np.sum(p)
        if plot:
            plt.scatter(source_distance_u, source_distance_v, c=p)
            plt.title(source+' -> '+target)
            plt.xlabel('Septotemporal distance (um)')
            plt.ylabel('Transverse distance (um)')
            plt.show()
            plt.close()
        return p, source_gid

    
def choose_synapse_projection (ranstream_syn, syn_layer, swc_type, syn_type, population_dict, projection_synapse_dict):
    """Given a synapse projection, SWC synapse location, and synapse
    type, chooses a projection from the given projection dictionary
    based on 1) whether the projection properties match the given
    synapse properties and 2) random choice between all the projections that satisfy the given criteria.
    :param ranstream_syn: random state object
    :param syn_layer: synapse layer
    :param swc_type: SWC location for synapse (soma, axon, apical, basal)
    :param syn_type: synapse type (excitatory, inhibitory, neuromodulatory)
    :param projection_synapse_dict:
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
                break
    if len(projection_lst) > 1:
       candidate_projections = np.asarray(projection_lst)
       candidate_probs       = np.asarray(projection_prob_lst)
       projection            = ranstream_syn.choice(candidate_projections, 1, p=candidate_probs)
    elif len(projection_lst) > 0:
       projection = projection_lst[0]
    else:
       projection = None

    if projection is not None:
        return ivd[projection]
    else:
        return None

 
def generate_synaptic_connections(ranstream_syn, ranstream_con, population_dict, synapse_dict, projection_synapse_dict, projection_prob_dict):
    """
    :param ranstream_syn:
    :param ranstream_con:
    :param synapse_dict:
    :param projection_synapse_dict:
    :param projection_prob_dict:
    """
    synapse_prj_partition = defaultdict(list)
    for (syn_id,syn_type,swc_type,syn_layer) in itertools.izip(synapse_dict['syn_ids'],
                                                               synapse_dict['syn_types'],
                                                               synapse_dict['swc_types'],
                                                               synapse_dict['syn_layers']):
        projection = choose_synapse_projection(ranstream_syn, syn_layer, swc_type, syn_type,
                                               population_dict, projection_synapse_dict)
        assert(projection is not None)
        synapse_prj_partition[projection].append(syn_id)

    prj_dict = {}
    
    for projection, syn_ids in synapse_prj_partition.iteritems():
        source_probs, source_gids = projection_prob_dict[projection]
        prj_dict[projection] = ( ranstream_con.choice(source_gids, len(syn_ids), p=source_probs),
                                 { 'Synapses' : { 'syn_id': syn_ids } } )
        
    return prj_dict


def generate_uv_distance_connections(comm, population_dict, connection_config, connection_prob, forest_path,
                                     synapse_seed, synapse_namespace, 
                                     connectivity_seed, connectivity_namespace, connectivity_path,
                                     io_size, chunk_size, value_chunk_size, cache_size):
    """
    :param comm:
    :param connection_config:
    :param connection_prob:
    :param forest_path:
    :param synapse_seed:
    :param synapse_namespace:
    :param connectivity_seed:
    :param connectivity_namespace:
    :param io_size:
    :param chunk_size:
    :param value_chunk_size:
    :param cache_size:
    """
    rank = comm.rank

    if io_size == -1:
        io_size = comm.size
    if rank == 0:
        print '%i ranks have been allocated' % comm.size
    sys.stdout.flush()

    start_time = time.time()

    ranstream_syn = np.random.RandomState()
    ranstream_con = np.random.RandomState()

    destination_population = connection_prob.destination_population
    
    source_populations = connection_config[destination_population].keys()

    for source_population in source_populations:
        print('%s -> %s:' % (source_population, destination_population), connection_config[destination_population][source_population])
                           
    projection_synapse_dict = {source_population: (connection_config[destination_population][source_population].synapse_layers,
                                                   set(connection_config[destination_population][source_population].synapse_locations),
                                                   set(connection_config[destination_population][source_population].synapse_types),
                                                   connection_config[destination_population][source_population].synapse_proportions)
                                for source_population in source_populations}

    print projection_synapse_dict
    count = 0
    for destination_gid, attributes_dict in NeuroH5CellAttrGen(comm, forest_path, destination_population, io_size=io_size,
                                                               cache_size=cache_size, namespace=synapse_namespace):
        last_time = time.time()
        
        connection_dict = defaultdict(lambda: {})
        if destination_gid is None:
            print 'Rank %i destination gid is None' % rank
        else:
            print 'Rank %i received attributes for destination: %s, gid: %i' % (rank, destination_population, destination_gid)
            ranstream_con.seed(destination_gid + connectivity_seed)
            ranstream_syn.seed(destination_gid + synapse_seed)

            synapse_dict = attributes_dict[synapse_namespace]

            projection_prob_dict = {}
            for source_population in source_populations:
                print ('destination %d: source %s:' % (destination_gid, source_population))
                probs, source_gids = connection_prob.get_prob(destination_gid, source_population)
                print ('source %s: len(source_gids) = %d' % (source_population, len(source_gids)))
                projection_prob_dict[source_population] = (probs, source_gids)

            connection_dict[destination_gid] = generate_synaptic_connections(ranstream_syn,
                                                                             ranstream_con,
                                                                             population_dict,
                                                                             synapse_dict,
                                                                             projection_synapse_dict,
                                                                             projection_prob_dict)

            for prj_dict in connection_dict[destination_gid].itervalues():
                for edge_dict in prj_dict.itervalues():
                    count += len(edge_dict[1]['Synapses']['syn_id'])
            
            print 'Rank %i took %i s to compute %d edges for destination: %s, gid: %i' % (rank, time.time() - last_time, count, destination_population, destination_gid)
            sys.stdout.flush()

    last_time = time.time()
    append_graph(comm, connectivity_path, {destination_population: connection_dict}, io_size)
    if rank == 0:
        print 'Appending connectivity for destination: %s took %i s' % (destination, time.time() - last_time)
    sys.stdout.flush()
    del connection_dict
    gc.collect()

    global_count = comm.gather(count, root=0)
    if rank == 0:
        print '%i ranks took took %i s to compute connectivity for %i cells' % (comm.size, time.time() - start_time,
                                                                                np.sum(global_count))


