
import itertools
import numpy as np
from mpi4py import MPI
from neuroh5.io import NeuroH5CellAttrGen, bcast_cell_attributes, read_population_ranges, append_graph
from bspline_surface import BSplineSurface
import click
import utils

try:
    import mkl
    mkl.set_num_threads(1)
except:
    pass


class ConnectionProb(object):
    """
    An object of this class will instantiate customized vectorized functions describing the connection probabilities
    for each presynaptic population. These functions can then be used to get the distribution of connection 
    probabilities across all possible source neurons, given the soma coordinates of a target neuron. 
    """
    def __init__(self, extent, nstdev = 5.):
        """
        Warning: This method does not produce an absolute probability. It must be normalized so that the total area
        (volume) under the distribution is 1 before sampling.
        :param extent: dict: {source: 'width': (tuple of float), 'offset': (tuple of float)}
        """
        self.p_dist = {}
        self.width  = {}
        self.offset = {}
        self.sigma  = {}
        for source in extent:
            extent_width  = extent[source]['width']
            if extent[source].has_key('offset'):
                extent_offset = extent[source]['offset']
            else:
                extent_offset = None
            self.width[source] = {'u': extent_width[0], 'v': extent_width[1]}
            self.sigma[source] = {axis: self.width[source][axis] / nstdev / np.sqrt(2.) for axis in self.width[source]}
            if extent_offset is None:
                self.offset[source] = {'u': 0., 'v': 0.}
            else:
                self.offset[source] = {'u': extent_offset[source][0], 'v': axon_offset[source][1]}


    def filter_by_arc_lengths_uv(self, destination, source, destination_gid, soma_coords, ip_surface):
        """
        Given the coordinates of a target neuron, returns the arc distances along u and v
        and the gids of source neurons whose axons potentially contact the target neuron.
        :param target: string
        :param source: string
        :param target_gid: int
        :param soma_coords: nested dict of array
        :param ip_surface: surface interpolation function
        :return: tuple of array of int
        """
        soma_uv = soma_coords[destination][destination_gid]['u_index']
        target_index_u = soma_uv['u_index']
        target_index_v = soma_uv['v_index']
        source_distance_u = []
        source_distance_v = []
        source_gid = []
        for source_gid in soma_coords[source]:
            this_source_distance_u, this_source_distance_v = \
                self.get_approximate_arc_distances(target_index_u, target_index_v,
                                                   soma_coords[source][this_source_gid]['u_index'],
                                                   soma_coords[source][this_source_gid]['v_index'], distance_U,
                                                   distance_V)
            if ((np.abs(this_source_distance_u) <= self.width[source]['u'] / 2. + self.offset[source]['u']) &
                    (np.abs(this_source_distance_v) <= self.width[source]['v'] / 2. + self.offset[source]['v'])):
                source_distance_u.append(this_source_distance_u)
                source_distance_v.append(this_source_distance_v)
                source_gid.append(this_source_gid)

        return np.array(source_distance_u), np.array(source_distance_v), np.array(source_gid)

    def get_p(self, target, source, target_gid, soma_coords, ip_surface, plot=False):
        """
        Given the soma coordinates of a target neuron and a population source, return an array of connection 
        probabilities and an array of corresponding source gids.
        :param target: str
        :param source: str
        :param target_gid: int
        :param soma_coords: nested dict of array
        :param distance_U: array of float
        :param distance_V: array of float
        :param plot: bool
        :return: array of float, array of int
        """
        source_distance_u, source_distance_v, source_gid = self.filter_by_soma_coords(target, source, target_gid,
                                                                                      soma_coords, ip_surface)
        p = self.p_dist[source](source_distance_u, source_distance_v)
        p /= np.sum(p)
        if plot:
            plt.scatter(source_distance_u, source_distance_v, c=p)
            plt.title(source+' -> '+target)
            plt.xlabel('Septotemporal distance (um)')
            plt.ylabel('Tranverse distance (um)')
            plt.show()
            plt.close()
        return p, source_gid


def filter_sources(target, layer, swc_type, syn_type):
    """
    
    :param target: str 
    :param layer: int
    :param swc_type: int
    :param syn_type: int
    :return: list
    """
    source_list = []
    proportion_list = []
    for source in layers[target]:
        for i, this_layer in enumerate(layers[target][source]):
            if this_layer == layer:
                if swc_types[target][source][i] == swc_type:
                    if syn_types[target][source][i] == syn_type:
                        source_list.append(source)
                        proportion_list.append(proportions[target][source][i])
    if proportion_list and np.sum(proportion_list) != 1.:
        raise Exception('Proportions of synapses to target: %s, layer: %i, swc_type: %i, '
                        'syn_type: %i do not sum to 1' % (target, layer, swc_type, syn_type))
    return source_list, proportion_list


def generate_uv_distance_connections(comm,
                                     destination, forest_path,
                                     synapse_layers, synapse_types,
                                     synapse_locations, synapse_namespace, 
                                     coords_path, coords_namespace,
                                     connectivity_seed, connectivity_namespace,
                                     io_size, chunk_size, value_chunk_size, cache_size):
    """

    :param forest_path:
    :param connectivity_namespace:
    :param coords_path:
    :param coords_namespace:
    :param io_size:
    :param chunk_size:
    :param value_chunk_size:
    :param cache_size:
    """
    rank = comm.rank  # The process ID (integer 0-3 for 4-process run)

    if io_size == -1:
        io_size = comm.size
    if rank == 0:
        print '%i ranks have been allocated' % comm.size
    sys.stdout.flush()

    start_time = time.time()

    soma_coords = {}
    source_populations = synapse_layers[destination].keys()
    for source in source_populations:
        soma_coords[source] = bcast_cell_attributes(comm, 0, coords_path, population,
                                                    namespace=coords_namespace)

    syn_layer_set, syn_location_set, syn_type_set = set(), set(), set()
    for source in source_populations:
        syn_layer_set.update(synapse_layers[destination][source])
        syn_location_set.update(syn_locations[destination][source])
        syn_type_set.update(syn_types[destination][source])

    count = 0
    for destination_gid, attributes_dict in NeuroH5CellAttrGen(comm, forest_path, destination, io_size=io_size,
                                                               cache_size=cache_size, namespace=synapse_namespace):
        last_time = time.time()
        connection_dict = {}
        p_dict = {}
        source_gid_dict = {}
        if destination_gid is None:
            print 'Rank %i destination gid is None' % rank
        else:
            print 'Rank %i received attributes for destination: %s, gid: %i' % (rank, destination, destination_gid)
            synapse_dict = attributes_dict[synapse_namespace]
            connection_dict[destination_gid] = {}
            local_np_random.seed(destination_gid + connectivity_seed)

            ## Candidate presynaptic cells to be connected to destination_gid
            candidates = 
            
            connection_dict[destination_gid]['source_gid'] = np.array([], dtype='uint32')
            connection_dict[destination_gid]['syn_id']     = np.array([], dtype='uint32')

        
            for layer in layer_set:
                for swc_type in swc_type_set:
                    for syn_type in syn_type_set:
                        sources, this_proportions = filter_sources(destination, layer, swc_type, syn_type)
                        if sources:
                            if rank == 0 and count == 0:
                                source_list_str = '[' + ', '.join(['%s' % xi for xi in sources]) + ']'
                                print 'Connections to destination: %s in layer: %i ' \
                                    '(swc_type: %i, syn_type: %i): %s' % \
                                    (destination, layer, swc_type, syn_type, source_list_str)
                            p, source_gid = np.array([]), np.array([])
                            for source, this_proportion in zip(sources, this_proportions):
                                if source not in source_gid_dict:
                                    this_p, this_source_gid = p_connect.get_p(destination, source, destination_gid, soma_coords,
                                                                              distance_U, distance_V)
                                    source_gid_dict[source] = this_source_gid
                                    p_dict[source] = this_p
                                else:
                                    this_source_gid = source_gid_dict[source]
                                    this_p = p_dict[source]
                                p = np.append(p, this_p * this_proportion)
                                source_gid = np.append(source_gid, this_source_gid)
                            syn_indexes = filter_synapses(synapse_dict, layer, swc_type, syn_type)
                            connection_dict[destination_gid]['syn_id'] = \
                                np.append(connection_dict[destination_gid]['syn_id'],
                                          synapse_dict['syn_id'][syn_indexes]).astype('uint32', copy=False)
                            this_source_gid = local_np_random.choice(source_gid, len(syn_indexes), p=p)
                            connection_dict[destination_gid]['source_gid'] = \
                                np.append(connection_dict[destination_gid]['source_gid'],
                                          this_source_gid).astype('uint32', copy=False)
            count += 1
            print 'Rank %i took %i s to compute connectivity for destination: %s, gid: %i' % (rank, time.time() - last_time,
                                                                                         destination, destination_gid)
            sys.stdout.flush()
        last_time = time.time()
        append_graph(comm, connectivity_path, source, destination, connection_dict, io_size=io_size)
        if rank == 0:
            print 'Appending connectivity for destination: %s took %i s' % (destination, time.time() - last_time)
        sys.stdout.flush()
        del connection_dict
        del p_dict
        del source_gid_dict
        gc.collect()

    global_count = comm.gather(count, root=0)
    if rank == 0:
        print '%i ranks took took %i s to compute connectivity for %i cells' % (comm.size, time.time() - start_time,
                                                                                np.sum(global_count))


