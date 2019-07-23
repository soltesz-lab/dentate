"""Classes and procedures related to neuronal connectivity graph analysis. """

import gc, math, time
from collections import defaultdict, ChainMap
import numpy as np
from dentate.utils import get_module_logger, list_find_all, range, str, zip, viewitems
from dentate.utils import Struct, add_bins, update_bins, finalize_bins
from neuroh5.io import NeuroH5ProjectionGen, bcast_cell_attributes, read_cell_attributes, read_population_names, read_population_ranges, read_projection_names

## This logger will inherit its setting from its root logger, dentate,
## which is created in module env
logger = get_module_logger(__name__)



def vertex_metrics(connectivity_path, coords_path, vertex_metrics_namespace, distances_namespace, destination, sources, bin_size = 50., metric='Indegree'):
    """
    Obtain vertex metrics with respect to septo-temporal position (longitudinal and transverse arc distances to reference points).

    :param connectivity_path:
    :param coords_path:
    :param distances_namespace: 
    :param destination: 
    :param source: 
    :param bin_size: 
    :param metric: 

    """

    (population_ranges, _) = read_population_ranges(coords_path)

    destination_start = population_ranges[destination][0]
    destination_count = population_ranges[destination][1]

    if sources == ():
        sources = []
        for (src, dst) in read_projection_names(connectivity_path):
            if dst == destination:
                sources.append(src)
    
    degrees_dict = {}
    with h5py.File(connectivity_path, 'r') as f:
        for source in sources:
            degrees_dict[source] = f['Nodes'][vertex_metrics_namespace]['%s %s -> %s' % (metric, source, destination)]['Attribute Value'][0:destination_count]
            
    for source in sources:
        logger.info('projection: %s -> %s: max: %i min: %i mean: %i stdev: %i (%d units)' % \
                        (source, destination, \
                         np.max(degrees_dict[source]), \
                         np.min(degrees_dict[source]), \
                         np.mean(degrees_dict[source]), \
                         np.std(degrees_dict[source]), \
                         len(degrees_dict[source])))

    if metric == 'Indegree':
        distances = read_cell_attributes(coords_path, destination, namespace=distances_namespace)
        soma_distances = { k: (v['U Distance'][0], v['V Distance'][0]) for (k,v) in distances }
        del distances
    elif metric == 'Outdegree':
        distances = read_cell_attributes(coords_path, sources[0], namespace=distances_namespace)
        soma_distances = { k: (v['U Distance'][0], v['V Distance'][0]) for (k,v) in distances }
        del distances
        
    gids = sorted(soma_distances.keys())
    distance_U = np.asarray([ soma_distances[gid][0] for gid in gids ])
    distance_V = np.asarray([ soma_distances[gid][1] for gid in gids ])

    return (distance_U, distance_V, degrees_dict)


def vertex_distribution(connectivity_path, coords_path, distances_namespace, destination, sources, 
                        bin_size=20.0, cache_size=100, comm=None):
    """
    Obtain spatial histograms of source vertices connecting to a given destination population.

    :param connectivity_path:
    :param coords_path:
    :param distances_namespace: 
    :param destination: 
    :param source: 

    """

    if comm is None:
        comm = MPI.COMM_WORLD

    rank = comm.Get_rank()
        
    (population_ranges, _) = read_population_ranges(coords_path)

    destination_start = population_ranges[destination][0]
    destination_count = population_ranges[destination][1]

    if rank == 0:
        logger.info('reading %s distances...' % destination)
        
    destination_soma_distances = bcast_cell_attributes(coords_path, destination, namespace=distances_namespace, comm=comm, root=0)
    

    destination_soma_distance_U = {}
    destination_soma_distance_V = {}
    for k,v in destination_soma_distances:
        destination_soma_distance_U[k] = v['U Distance'][0]
        destination_soma_distance_V[k] = v['V Distance'][0]

    del(destination_soma_distances)

    if sources == ():
        sources = []
        for (src, dst) in read_projection_names(connectivity_path):
            if dst == destination:
                sources.append(src)

    source_soma_distances = {}
    for s in sources:
        if rank == 0:
            logger.info('reading %s distances...' % s)
        source_soma_distances[s] = bcast_cell_attributes(coords_path, s, namespace=distances_namespace, comm=comm, root=0)

    
    source_soma_distance_U = {}
    source_soma_distance_V = {}
    for s in sources:
        this_source_soma_distance_U = {}
        this_source_soma_distance_V = {}
        for k,v in source_soma_distances[s]:
            this_source_soma_distance_U[k] = v['U Distance'][0]
            this_source_soma_distance_V[k] = v['V Distance'][0]
        source_soma_distance_U[s] = this_source_soma_distance_U
        source_soma_distance_V[s] = this_source_soma_distance_V
    del(source_soma_distances)

    logger.info('reading connections %s -> %s...' % (str(sources), destination))
    gg = [ NeuroH5ProjectionGen (connectivity_path, source, destination, cache_size=cache_size, comm=comm) for source in sources ]

    dist_bins = defaultdict(dict)
    dist_u_bins = defaultdict(dict)
    dist_v_bins = defaultdict(dict)
    
    for prj_gen_tuple in zip_longest(*gg):
        destination_gid = prj_gen_tuple[0][0]
        if not all([prj_gen_elt[0] == destination_gid for prj_gen_elt in prj_gen_tuple]):
            raise RuntimeError('destination %s: destination_gid %i not matched across multiple projection generators: '
                               '%s' % (destination, destination_gid, [prj_gen_elt[0] for prj_gen_elt in prj_gen_tuple]))

        if destination_gid is not None:
            for (source, (this_destination_gid,rest)) in zip_longest(sources, prj_gen_tuple):
                this_source_soma_distance_U = source_soma_distance_U[source]
                this_source_soma_distance_V = source_soma_distance_V[source]
                this_dist_bins = dist_bins[source]
                this_dist_u_bins = dist_u_bins[source]
                this_dist_v_bins = dist_v_bins[source]
                (source_indexes, attr_dict) = rest
                dst_U = destination_soma_distance_U[destination_gid]
                dst_V = destination_soma_distance_V[destination_gid]
                for source_gid in source_indexes:
                    dist_u = dst_U - this_source_soma_distance_U[source_gid]
                    dist_v = dst_V - this_source_soma_distance_V[source_gid]
                    dist = abs(dist_u) + abs(dist_v)
                
                    update_bins(this_dist_bins, bin_size, dist)
                    update_bins(this_dist_u_bins, bin_size, dist_u)
                    update_bins(this_dist_v_bins, bin_size, dist_v)

    add_bins_op = MPI.Op.Create(add_bins, commute=True)
    for source in sources:
        dist_bins[source] = comm.reduce(dist_bins[source], op=add_bins_op)
        dist_u_bins[source] = comm.reduce(dist_u_bins[source], op=add_bins_op)
        dist_v_bins[source] = comm.reduce(dist_v_bins[source], op=add_bins_op)

    dist_hist_dict = defaultdict(dict)
    dist_u_hist_dict = defaultdict(dict)
    dist_v_hist_dict = defaultdict(dict)
    
    if rank == 0:
        for source in sources:
            dist_hist_dict[destination][source] = finalize_bins(dist_bins[source], bin_size)
            dist_u_hist_dict[destination][source] = finalize_bins(dist_u_bins[source], bin_size)
            dist_v_hist_dict[destination][source] = finalize_bins(dist_v_bins[source], bin_size)


    return {'Total distance': dist_hist_dict,
            'U distance': dist_u_hist_dict,
            'V distance': dist_v_hist_dict }


def spatial_bin_graph(connectivity_path, coords_path, distances_namespace, destination, sources, extents,
                      bin_size=20.0, cache_size=100, comm=None):
    """
    Obtain reduced graphs of the specified projections by binning nodes according to their spatial position.

    :param connectivity_path:
    :param coords_path:
    :param distances_namespace: 
    :param destination: 
    :param source: 

    """

    import networkx as nx
    
    if comm is None:
        comm = MPI.COMM_WORLD

    rank = comm.Get_rank()
        
    (population_ranges, _) = read_population_ranges(coords_path)

    destination_start = population_ranges[destination][0]
    destination_count = population_ranges[destination][1]

    if rank == 0:
        logger.info('reading %s distances...' % destination)
        
    destination_soma_distances = bcast_cell_attributes(coords_path, destination, namespace=distances_namespace, comm=comm, root=0)

    ((u_min, u_max), (v_min, v_max)) = extents
    u_bins = np.arange(u_min, u_max, bin_size)
    v_bins = np.arange(v_min, v_max, bin_size)

    dest_u_bins = {}
    dest_v_bins = {}
    destination_soma_distance_U = {}
    destination_soma_distance_V = {}
    for k,v in destination_soma_distances:
        dist_u = v['U Distance'][0]
        dist_v = v['V Distance'][0]
        dest_u_bins[k] = np.searchsorted(u_bins, dist_u, side='left')
        dest_v_bins[k] = np.searchsorted(v_bins, dist_v, side='left')
        destination_soma_distance_U[k] = dist_u
        destination_soma_distance_V[k] = dist_v

    del(destination_soma_distances)

    if (sources == ()) or (sources == []) or (sources is None):
        sources = []
        for (src, dst) in read_projection_names(connectivity_path):
            if dst == destination:
                sources.append(src)

    source_soma_distances = {}
    for s in sources:
        if rank == 0:
            logger.info('reading %s distances...' % s)
        source_soma_distances[s] = bcast_cell_attributes(coords_path, s, namespace=distances_namespace, comm=comm, root=0)

    
    source_u_bins = {}
    source_v_bins = {}
    source_soma_distance_U = {}
    source_soma_distance_V = {}
    for s in sources:
        this_source_soma_distance_U = {}
        this_source_soma_distance_V = {}
        this_source_u_bins = {}
        this_source_v_bins = {}
        for k,v in source_soma_distances[s]:
            dist_u = v['U Distance'][0]
            dist_v = v['V Distance'][0]
            this_source_u_bins[k] = np.searchsorted(u_bins, dist_u, side='left')
            this_source_v_bins[k] = np.searchsorted(v_bins, dist_v, side='left')
            this_source_soma_distance_U[k] = dist_u
            this_source_soma_distance_V[k] = dist_v
        source_soma_distance_U[s] = this_source_soma_distance_U
        source_soma_distance_V[s] = this_source_soma_distance_V
        source_u_bins[s] = this_source_u_bins
        source_v_bins[s] = this_source_v_bins
    del(source_soma_distances)

    logger.info('reading connections %s -> %s...' % (str(sources), destination))
    gg = [ NeuroH5ProjectionGen (connectivity_path, source, destination, cache_size=cache_size, comm=comm) for source in sources ]

    dist_bins = defaultdict(dict)
    dist_u_bins = defaultdict(dict)
    dist_v_bins = defaultdict(dict)

    
    local_u_bin_graph = defaultdict(dict)
    local_v_bin_graph = defaultdict(dict)
    
    for prj_gen_tuple in zip_longest(*gg):
        destination_gid = prj_gen_tuple[0][0]
        if not all([prj_gen_elt[0] == destination_gid for prj_gen_elt in prj_gen_tuple]):
            raise RuntimeError('destination %s: destination_gid %i not matched across multiple projection generators: '
                               '%s' % (destination, destination_gid, [prj_gen_elt[0] for prj_gen_elt in prj_gen_tuple]))

        if destination_gid is not None:
            dest_u_bin = dest_u_bins[destination_gid]
            dest_v_bin = dest_v_bins[destination_gid]
            for (source, (this_destination_gid,rest)) in zip_longest(sources, prj_gen_tuple):
                this_source_u_bins = source_u_bins[source]
                this_source_v_bins = source_v_bins[source]
                (source_indexes, attr_dict) = rest
                source_u_bin_dict = defaultdict(int)
                source_b_bin_dict = defaultdict(int)
                for source_gid in source_indexes:
                    source_u_bin = this_source_u_bins[source_gid]
                    source_v_bin = this_source_v_bins[source_gid]
                    source_u_bin_dict[source_u_bin] += 1
                    source_v_bin_dict[source_v_bin] += 1
                local_u_bin_graph[dest_u_bin][source] = source_u_bin_dict
                local_v_bin_graph[dest_v_bin][source] = source_v_bin_dict
                    
    u_bin_edges = { destination: dict(ChainMap(*comm.gather(dict(local_u_bin_graph), root=0))) }
    v_bin_edges = { destination: dict(ChainMap(*comm.gather(dict(local_v_bin_graph), root=0))) }

    nu = len(u_bins)
    u_bin_graph = nx.Graph()
    for pop in [destination]+sources:
        for i in range(nu):
            u_bin_graph.add_node((pop, i))


    for i, sources in viewitems(u_bin_edges[destination]):
        for source, ids in viewitems(sources):
            u_bin_graph.add_edges_from([((source, j), (destination, i)) for j in ids])

    nv = len(v_bins)
    v_bin_graph = nx.Graph()
    for pop in [destination]+sources:
        for i in range(nv):
            v_bin_graph.add_node((pop, i))

    for i, sources in viewitems(v_bin_edges[destination]):
        for source, ids in viewitems(sources):
            v_bin_graph.add_edges_from([((source, j), (destination, i)) for j in ids])
    
    return (u_bin_graph, v_bin_graph)

