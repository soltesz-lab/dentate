
##
## Classes and procedures related to neuronal connectivity generation.
##

import sys, time, gc, numbers, itertools
from collections import defaultdict
import numpy as np
from scipy.stats import norm
import rbf
from rbf.interpolate import RBFInterpolant
import rbf.basis
from mpi4py import MPI
from neuroh5.io import NeuroH5CellAttrGen, bcast_cell_attributes, read_population_ranges, append_graph
import click, logging
import dentate.utils
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


def make_random_clusters(centers, n_samples_per_center, n_features=2, cluster_std=1.0,
                         center_ids=None, center_box=(-10.0, 10.0), random_seed=None):
    """Generate isotropic Gaussian blobs for clustering.

    Parameters
    ----------
    centers : int or array of shape [n_centers, n_features]
        The number of centers to generate, or the fixed center locations.
    n_samples_per_center : int array
        Number of points for each cluster.
    n_features : int, optional (default=2)
        The number of features for each sample.
    cluster_std : float or sequence of floats, optional (default=1.0)
        The standard deviation of the clusters.
    center_ids : array of integer center ids, if None then centers will be numbered 0 .. n_centers-1
    center_box : pair of floats (min, max), optional (default=(-10.0, 10.0))
        The bounding box for each cluster center when centers are
        generated at random.
    random_seed : int or None, optional (default=None)
        If int, random_seed is the seed used by the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    Returns
    -------
    X : array of shape [n_samples, n_features]
        The generated samples.
    y : array of shape [n_samples]
        The integer labels for cluster membership of each sample.
    Examples
    --------
    >>> X, y = make_random_clusters (centers=6, n_samples_per_center=np.array([1,3,10,15,7,9]), n_features=1, \
                                     center_ids=np.array([10,13,21,25,27,29]).reshape(-1,1), cluster_std=1.0, \
                                     center_box=(-10.0, 10.0))
    >>> print(X.shape)
    (45, 1)
    >>> y
    array([10, 13, 13, 13, ..., 29, 29, 29])
    """
    rng = np.random.RandomState(random_seed)

    if isinstance(centers, numbers.Integral):
        centers = np.sort(rng.uniform(center_box[0], center_box[1], \
                                      size=(centers, n_features)), axis=0)
    else:
        assert(isinstance(centers, np.ndarray))
        n_features = centers.shape[1]

    if center_ids is None:
        center_ids = np.arange(0, centers.shape[0])
        
    if isinstance(cluster_std, numbers.Real):
        cluster_std = np.ones(len(centers)) * cluster_std

    X = []
    y = []

    n_centers = centers.shape[0]

    for i, (cid, n, std) in enumerate(itertools.izip(center_ids, n_samples_per_center, cluster_std)):
        if n > 0:
            X.append(centers[i] + rng.normal(scale=std, size=(n, n_features)))
            y += [cid] * n

    X = np.concatenate(X)
    y = np.array(y)

    return X, y


def random_clustered_shuffle(centers, n_samples_per_center, center_ids=None, cluster_std=1.0, center_box=(-1.0, 1.0), random_seed=None):
    """Generates a Gaussian random clustering given a number of cluster
    centers, samples per each center, optional integer center ids, and
    cluster standard deviation.

    Parameters
    ----------
    centers : int or array of shape [n_centers]
        The number of centers to generate, or the fixed center locations.
    n_samples_per_center : int array
        Number of points for each cluster.
    cluster_std : float or sequence of floats, optional (default=1.0)
        The standard deviation of the clusters.
    center_ids : array of integer center ids, if None then centers will be numbered 0 .. n_centers-1
    random_seed : int or None, optional (default=None)
        If int, random_seed is the seed used by the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    >>> x = random_clustered_shuffle(centers=6,center_ids=np.array([10,13,21,25,27,29]).reshape(-1,1), \
                                     n_samples_per_center=np.array([1,3,10,15,7,9]))
    >>> array([10, 13, 13, 25, 13, 29, 21, 25, 27, 21, 27, 29, 25, 25, 25, 21, 29,
               27, 25, 21, 29, 25, 25, 25, 25, 29, 21, 25, 21, 29, 29, 29, 21, 25,
               29, 21, 27, 27, 21, 27, 25, 21, 25, 27, 25])
    """

    if isinstance(centers, numbers.Integral):
        n_centers = centers
    else:
        assert(isinstance(centers, np.ndarray))
        n_centers = len(centers)
    
    X, y = make_random_clusters (centers, n_samples_per_center, n_features=1, \
                                 center_ids=center_ids, cluster_std=cluster_std, center_box=center_box, \
                                 random_seed=random_seed)
    s = np.argsort(X,axis=0).ravel()
    return y[s].ravel()


    
class ConnectionProb(object):
    """An object of this class will instantiate functions that describe
    the connection probabilities for each presynaptic population. These
    functions can then be used to get the distribution of connection
    probabilities across all possible source neurons, given the soma
    coordinates of a destination (post-synaptic) neuron.
    """
    def __init__(self, destination_population, soma_coords, soma_distances, extent, sigma = 0.4, res=5):
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
        self.scale_factor  = {}

        
        for source_population in extent:
            extent_width  = extent[source_population]['width']
            if extent[source_population].has_key('offset'):
                extent_offset = extent[source_population]['offset']
            else:
                extent_offset = None
            self.width[source_population] = {'u': extent_width[0], 'v': extent_width[1]}
            self.scale_factor[source_population] = { axis: self.width[source_population][axis] for axis in self.width[source_population] }
            if extent_offset is None:
                self.offset[source_population] = {'u': 0., 'v': 0.}
            else:
                self.offset[source_population] = {'u': extent_offset[0], 'v': extent_offset[1]}
            self.p_dist[source_population] = (lambda source_population: np.vectorize(lambda distance_u, distance_v: \
                                                       (norm.pdf(np.abs(distance_u - self.offset[source_population]['u']) / self.scale_factor[source_population]['u'], scale=sigma) * \
                                                        norm.pdf(np.abs(distance_v - self.offset[source_population]['v']) / self.scale_factor[source_population]['v'], scale=sigma)), \
                                                        otypes=[float]))(source_population)

    
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

    
    def get_prob(self, destination_gid, source):
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
        assert((p >= 0.).all() and (p <= 1.).all())
        if psum > 0.:
            pn = softmax(p)
        else:
            pn = p
        return pn.ravel(), source_gid.ravel(), distance_u.ravel(), distance_v.ravel()


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
        
    if verbose:
        logger.info('Computing V distances...')
    ldist_v, obs_dist_v = ip_vol.point_distance(U, V, L, axis=1)
    obs_uvl = np.array([np.concatenate(obs_dist_v[0]), \
                        np.concatenate(obs_dist_v[1]), \
                        np.concatenate(obs_dist_v[2])]).T
    sample_inds = np.arange(0, obs_uvl.shape[0]-1, step)
    obs_v = obs_uvl[sample_inds,:]    
    distances_v = np.concatenate(ldist_v)[sample_inds]

    return (distances_u, obs_u, distances_v, obs_v)



        
def get_soma_distances(comm, dist_u, dist_v, soma_coords, combined=False):
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
        if combined:
            dist_dicts = comm.allgather(local_dist_dict)
            combined_dist_dict = {}
            for dist_dict in dist_dicts:
                for k, v in dist_dict.iteritems():
                    combined_dist_dict[k] = v
            soma_distances[pop] = combined_dist_dict
        else:
            soma_distances[pop] = local_dist_dict

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

 
def generate_synaptic_connections(rank,
                                  ranstream_syn,
                                  ranstream_con,
                                  cluster_seed,
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
            logger.error('Projection is none for syn_type = %s swc_type = %s syn_layer = %s' % (str(syn_type), str(swc_type), str(syn_layer)))
            print projection_synapse_dict
        assert(projection is not None)
        synapse_prj_partition[projection].append(syn_id)

    ## Choose source connections based on distance-weighted probability
    count = 0
    for projection, syn_ids in synapse_prj_partition.iteritems():
        count += len(syn_ids)
        source_probs, source_gids, distances_u, distances_v = projection_prob_dict[projection]
        distance_dict = { source_gid: distance_u+distance_v \
                          for (source_gid,distance_u,distance_v) in itertools.izip(source_gids, \
                                                                                   distances_u, \
                                                                                   distances_v) }
        if len(source_gids) > 0:
            source_gid_counts = random_choice(ranstream_con,len(syn_ids),source_probs)
            uv_distance_sums = np.add(distances_u, distances_v, dtype=np.float32)
            gid_dict = connection_dict[projection]
            source_vertices = np.asarray(random_clustered_shuffle(len(source_gids), source_gid_counts, center_ids=source_gids, \
                                                                    cluster_std=2.0, random_seed=cluster_seed),dtype=np.uint32)
            distances = np.asarray([ distance_dict[gid] for gid in source_vertices ], dtype=np.float32).reshape(-1,)
            gid_dict[destination_gid] = ( source_vertices,
                                              { 'Synapses' : { 'syn_id': np.asarray (syn_ids, dtype=np.uint32) },
                                                'Connections' : { 'distance': distances }
                                              } )
            connection_dict[projection] = gid_dict
        else:
            logger.warning('Rank %i: source gid list is empty for gid: %i projection: %s len(syn_ids): %i' % (rank, destination_gid, projection, len(syn_ids)))

        
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
                                     connectivity_seed, cluster_seed, connectivity_namespace, connectivity_path,
                                     io_size, chunk_size, value_chunk_size, cache_size, write_size=1,
                                     verbose=False, dry_run=False):
    """Generates connectivity based on U, V distance-weighted probabilities.
    :param comm: mpi4py MPI communicator
    :param connection_config: connection configuration object (instance of env.ConnectionGenerator)
    :param connection_prob: ConnectionProb instance
    :param forest_path: location of file with neuronal trees and synapse information
    :param synapse_seed: random seed for synapse partitioning
    :param synapse_namespace: namespace of synapse properties
    :param connectivity_seed: random seed for connectivity generation
    :param cluster_seed: random seed for determining connectivity clustering for repeated connections from the same source
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
    projection_dict = {}
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


