
import itertools, math
from collections import defaultdict
import numpy as np
from scipy import signal, interpolate
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.tri as tri
from matplotlib import gridspec, mlab, rcParams
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpi4py import MPI
import h5py
from neuroh5.io import read_population_ranges, read_population_names, read_cell_attributes, NeuroH5CellAttrGen, NeuroH5ProjectionGen, read_trees
import dentate
from dentate import spikedata, statedata, stimulus
from dentate.env import Env

color_list = ["#00FF00", "#0000FF", "#FF0000", "#01FFFE", "#FFA6FE",
              "#FFDB66", "#006401", "#010067", "#95003A", "#007DB5", "#FF00F6", "#FFEEE8", "#774D00",
              "#90FB92", "#0076FF", "#D5FF00", "#FF937E", "#6A826C", "#FF029D", "#FE8900", "#7A4782",
              "#7E2DD2", "#85A900", "#FF0056", "#A42400", "#00AE7E", "#683D3B", "#BDC6FF", "#263400",
              "#BDD393", "#00B917", "#9E008E", "#001544", "#C28C9F", "#FF74A3", "#01D0FF", "#004754",
              "#E56FFE", "#788231", "#0E4CA1", "#91D0CB", "#BE9970", "#968AE8", "#BB8800", "#43002C",
              "#DEFF74", "#00FFC6", "#FFE502", "#620E00", "#008F9C", "#98FF52", "#7544B1", "#B500FF",
              "#00FF78", "#FF6E41", "#005F39", "#6B6882", "#5FAD4E", "#A75740", "#A5FFD2", "#FFB167", 
              "#009BFF", "#E85EBE"]

selectivity_type_dict = {'MPP': stimulus.selectivity_grid, 'LPP': stimulus.selectivity_place_field}




def show_figure():
    try:
        plt.show(block=False)
    except:
        plt.show()


def ifilternone(iterable):
    for x in iterable:
        if not (x is None):
            yield x
            
def flatten(iterables):
    return (elem for iterable in ifilternone(iterables) for elem in iterable)

def plot_graph(x, y, z, start_idx, end_idx, edge_scalars=None, **kwargs):
    """ Shows graph edges using Mayavi

        Parameters
        -----------
        x: ndarray
            x coordinates of the points
        y: ndarray
            y coordinates of the points
        z: ndarray
            z coordinates of the points
        edge_scalars: ndarray, optional
            optional data to give the color of the edges.
        kwargs:
            extra keyword arguments are passed to quiver3d.
    """
    from mayavi import mlab
    vec = mlab.quiver3d(x[start_idx],
                        y[start_idx],
                        z[start_idx],
                        x[end_idx] - x[start_idx],
                        y[end_idx] - y[start_idx],
                        z[end_idx] - z[start_idx],
                        scalars=edge_scalars,
                        mode='2ddash',
                        scale_factor=1,
                        **kwargs)
    if edge_scalars is not None:
        vec.glyph.color_mode = 'color_by_scalar'
    return vec


def make_geometric_graph(x, y, z, edges):
    """ Builds a NetworkX graph with xyz node coordinates and the node indices
        of the end nodes.

        Parameters
        -----------
        x: ndarray
            x coordinates of the points
        y: ndarray
            y coordinates of the points
        z: ndarray
            z coordinates of the points
        edges: the (2, N) array returned by compute_delaunay_edges()
            containing node indices of the end nodes. Weights are applied to
            the edges based on their euclidean length for use by the MST
            algorithm.

        Returns
        ---------
        g: A NetworkX undirected graph

        Notes
        ------
        We don't bother putting the coordinates into the NX graph.
        Instead the graph node is an index to the column.
    """
    import networkx as nx
    xyz = np.array((x, y, z))
    def euclidean_dist(i, j):
        d = xyz[:,i] - xyz[:,j]
        return np.sqrt(np.dot(d, d))

    g = nx.Graph()
    for i, j in edges:
        g.add_edge(i, j, weight=euclidean_dist(i, j))
    return g


def update_bins(bins, binsize, x):
    i = math.floor(x / binsize)
    if i in bins:
        bins[i] += 1
    else:
        bins[i] = 1

        
def finalize_bins(bins, binsize):
    imin = int(min(bins.keys()))
    imax = int(max(bins.keys()))
    a = [0] * (imax - imin + 1)
    b = [binsize * k for k in range(imin, imax + 1)]
    for i in range(imin, imax + 1):
        if i in bins:
            a[i - imin] = bins[i]
    return np.asarray(a), np.asarray(b)


def plot_vertex_metrics(connectivity_path, coords_path, vertex_metrics_namespace, distances_namespace, destination, sources,
                        metric='Indegree', normed = False, fontSize=14, showFig = True, saveFig = False, verbose = False):
    """
    Plot vertex metric with respect to septo-temporal position (longitudinal and transverse arc distances to reference points).

    :param connectivity_path:
    :param coords_path:
    :param distances_namespace: 
    :param destination_pop: 

    """

    dx = 100
    dy = 100
    
    (population_ranges, _) = read_population_ranges(coords_path)

    destination_start = population_ranges[destination][0]
    destination_count = population_ranges[destination][1]

    with h5py.File(connectivity_path, 'r') as f:
        degrees_lst = []
        for source in sources:
            degrees_lst.append(f['Nodes'][vertex_metrics_namespace]['%s %s -> %s' % (metric, source, destination)]['Attribute Value'][0:destination_count])
        degrees = np.sum(degrees_lst, axis=0)
            
    if verbose:
        print 'read degrees (%i elements)' % len(degrees)
        print 'max: %i min: %i mean: %i stdev: %i' % (np.max(degrees), np.min(degrees), np.mean(degrees), np.std(degrees))
        
    distances = read_cell_attributes(coords_path, destination, namespace=distances_namespace)
    
    soma_distances = { k: (v['U Distance'][0], v['V Distance'][0]) for (k,v) in distances }
    del distances
    if verbose:
        print 'read distances (%i elements)' % len(soma_distances.keys())
    
    fig = plt.figure(1, figsize=plt.figaspect(1.) * 2.)
    ax = plt.gca()

    distance_U = np.asarray([ soma_distances[v][0] for v in range(0,len(degrees)) ])
    distance_V = np.asarray([ soma_distances[v][1] for v in range(0,len(degrees)) ])

    if normed:
        (H1, xedges, yedges) = np.histogram2d(distance_U, distance_V, bins=[dx, dy], weights=degrees, normed=normed)
        (H2, xedges, yedges) = np.histogram2d(distance_U, distance_V, bins=[dx, dy])
        H = np.zeros(H1.shape)
        nz = np.where(H2 > 0.0)
        H[nz] = np.divide(H1[nz], H2[nz])
        H[nz] = np.divide(H[nz], np.max(H[nz]))
    else:
        (H, xedges, yedges) = np.histogram2d(distance_U, distance_V, bins=[dx, dy], weights=degrees)
        
    if verbose:
        print 'Plotting in-degree distribution...'

    x_min = np.min(distance_U)
    x_max = np.max(distance_U)
    y_min = np.min(distance_V)
    y_max = np.max(distance_V)

    ax.axis([x_min, x_max, y_min, y_max])

    X, Y = np.meshgrid(xedges, yedges)
    pcm = ax.pcolormesh(X, Y, H.T)
    
    ax.set_xlabel('Arc distance (septal - temporal) (um)', fontsize=fontSize)
    ax.set_ylabel('Arc distance (supra - infrapyramidal)  (um)', fontsize=fontSize)
    ax.set_title('%s distribution for destination: %s sources: %s' % (metric, destination, ', '.join(sources)), fontsize=fontSize)
    ax.set_aspect('equal')
    fig.colorbar(pcm, ax=ax, shrink=0.5, aspect=20)
    
    if saveFig: 
        if isinstance(saveFig, basestring):
            filename = saveFig
        else:
            filename = destination+' %s.png' % metric
            plt.savefig(filename)

    if showFig:
        show_figure()
    
    return ax


def plot_vertex_dist(connectivity_path, coords_path, distances_namespace, destination, source,
                        bin_size=20.0, fontSize=14, showFig = True, saveFig = False, verbose = False):
    """
    Plot vertex distribution with respect to septo-temporal distance

    :param connectivity_path:
    :param coords_path:
    :param distances_namespace: 
    :param destination: 
    :param source: 

    """
    
    (population_ranges, _) = read_population_ranges(coords_path)

    destination_start = population_ranges[destination][0]
    destination_count = population_ranges[destination][1]

    source_soma_distances = read_cell_attributes(coords_path, source, namespace=distances_namespace)
    destination_soma_distances = read_cell_attributes(coords_path, destination, namespace=distances_namespace)

    source_soma_distance_U = {}
    source_soma_distance_V = {}
    destination_soma_distance_U = {}
    destination_soma_distance_V = {}
    for k,v in source_soma_distances:
        source_soma_distance_U[k] = v['U Distance'][0]
        source_soma_distance_V[k] = v['V Distance'][0]
    for k,v in destination_soma_distances:
        destination_soma_distance_U[k] = v['U Distance'][0]
        destination_soma_distance_V[k] = v['V Distance'][0]

    del(source_soma_distances)
    del(destination_soma_distances)
                
    g = NeuroH5ProjectionGen (connectivity_path, source, destination, cache_size=50)
    dist_bins = {}
    dist_u_bins = {}
    count = 0
    min_dist = float('inf')
    max_dist = 0.0
    max_dist_u = 0.0
    for (destination_gid,rest) in g:
        if destination_gid is not None:
            (source_indexes, attr_dict) = rest
            for source_gid in source_indexes:
                dist_u = abs(destination_soma_distance_U[destination_gid] - source_soma_distance_U[source_gid]) 
                dist = abs(destination_soma_distance_U[destination_gid] - source_soma_distance_U[source_gid]) + \
                       abs(destination_soma_distance_V[destination_gid] - source_soma_distance_V[source_gid])
                if verbose:
                    print '%i: dist_u = %f' % (destination_gid, destination_soma_distance_U[destination_gid])
                    print '%i: dist_u = %f' % (source_gid, source_soma_distance_U[source_gid])
                    print '%i: %i -> %i: dist = %f; dist_u = %f' % (count, source_gid, destination_gid, dist, dist_u)
                min_dist = min(min_dist, dist)
                max_dist = max(max_dist, dist)
                max_dist_u = max(max_dist_u, dist_u)
                update_bins(dist_bins, bin_size, dist)
                update_bins(dist_u_bins, bin_size, dist_u)
            count = count + 1
    dist_histoCount, dist_bin_edges = finalize_bins(dist_bins, bin_size)
    dist_u_histoCount, dist_u_bin_edges = finalize_bins(dist_u_bins, bin_size)
    if verbose:
        print 'min dist = %f; max dist = %f; max dist u = %f' % (min_dist, max_dist, max_dist_u)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.bar(dist_bin_edges, dist_histoCount, width=bin_size)
    ax2.bar(dist_u_bin_edges, dist_u_histoCount, width=bin_size)

    if saveFig: 
        if isinstance(saveFig, basestring):
            filename = saveFig
        else:
            filename = '%s to %s.png' % (source, destination)
            plt.savefig(filename)

    if showFig:
        show_figure()
    

    

def plot_tree_metrics(forest_path, coords_path, population, metric_namespace='Tree Measurements', distances_namespace='Arc Distances', 
                       metric='dendrite_length', metric_index=0, fontSize=14, showFig = True, saveFig = False, verbose = False):
    """
    Plot tree length or area with respect to septo-temporal position (longitudinal and transverse arc distances).

    :param forest_path:
    :param coords_path:
    :param distances_namespace: 
    :param measures_namespace: 
    :param population: 

    """

    dx = 50
    dy = 50
    
        
    soma_distances = read_cell_attributes(coords_path, population, namespace=distances_namespace)
    
    tree_metrics = { k: v[metric][metric_index] for (k,v) in read_cell_attributes(forest_path, population, namespace=metric_namespace) }
        
    fig = plt.figure(1, figsize=plt.figaspect(1.) * 2.)
    ax = plt.gca()

    distance_U = {}
    distance_V = {}
    for k,v in soma_distances:
        distance_U[k] = v['U Distance'][0]
        distance_V[k] = v['V Distance'][0]
    
    sorted_keys = sorted(tree_metrics.keys())
    tree_metrics_array = np.array([tree_metrics[k] for k in sorted_keys])
    tree_metric_stats = (np.min(tree_metrics_array), np.max(tree_metrics_array), np.mean(tree_metrics_array))
    print ('min: %f max: %f mean: %f' % (tree_metric_stats))
    
    distance_U_array = np.array([distance_U[k] for k in sorted_keys])
    distance_V_array = np.array([distance_V[k] for k in sorted_keys])

    x_min = np.min(distance_U_array)
    x_max = np.max(distance_U_array)
    y_min = np.min(distance_V_array)
    y_max = np.max(distance_V_array)

    (H, xedges, yedges) = np.histogram2d(distance_U_array, distance_V_array, \
                                         bins=[dx, dy], weights=tree_metrics_array, normed=True)


    ax.axis([x_min, x_max, y_min, y_max])

    X, Y = np.meshgrid(xedges, yedges)
    pcm = ax.pcolormesh(X, Y, H.T)
    
    ax.set_xlabel('Arc distance (septal - temporal) (um)', fontsize=fontSize)
    ax.set_ylabel('Arc distance (supra - infrapyramidal)  (um)', fontsize=fontSize)
    ax.set_title('%s distribution for population: %s' % (metric, population), fontsize=fontSize)
    ax.set_aspect('equal')
    fig.colorbar(pcm, ax=ax, shrink=0.5, aspect=20)
    
    if saveFig: 
        if isinstance(saveFig, basestring):
            filename = saveFig
        else:
            filename = population+' %s.png' % metric
            plt.savefig(filename)

    if showFig:
        show_figure()
    
    return ax


def plot_positions(coords_path, population, distances_namespace='Arc Distances', 
                    fontSize=14, showFig = True, saveFig = False, verbose = False):
    """
    Plot septo-temporal position (longitudinal and transverse arc distances).

    :param coords_path:
    :param distances_namespace: 
    :param population: 

    """

    dx = 50
    dy = 50
    
        
    soma_distances = read_cell_attributes(coords_path, population, namespace=distances_namespace)
    
        
    fig = plt.figure(1, figsize=plt.figaspect(1.) * 2.)
    ax = plt.gca()

    distance_U = {}
    distance_V = {}
    for k,v in soma_distances:
        distance_U[k] = v['U Distance'][0]
        distance_V[k] = v['V Distance'][0]
    
    distance_U_array = np.asarray([distance_U[k] for k in sorted(distance_U.keys())])
    distance_V_array = np.asarray([distance_V[k] for k in sorted(distance_V.keys())])

    x_min = np.min(distance_U_array)
    x_max = np.max(distance_U_array)
    y_min = np.min(distance_V_array)
    y_max = np.max(distance_V_array)

    (H, xedges, yedges) = np.histogram2d(distance_U_array, distance_V_array, bins=[dx, dy])


    ax.axis([x_min, x_max, y_min, y_max])

    X, Y = np.meshgrid(xedges, yedges)
    pcm = ax.pcolormesh(X, Y, H.T)
    
    ax.set_xlabel('Arc distance (septal - temporal) (um)', fontsize=fontSize)
    ax.set_ylabel('Arc distance (supra - infrapyramidal)  (um)', fontsize=fontSize)
    ax.set_title('Position distribution for population: %s' % (population), fontsize=fontSize)
    ax.set_aspect('equal')
    fig.colorbar(pcm, ax=ax, shrink=0.5, aspect=20)
    
    if saveFig: 
        if isinstance(saveFig, basestring):
            filename = saveFig
        else:
            filename = population+' Positions.png' 
            plt.savefig(filename)

    if showFig:
        show_figure()
    
    return ax


def plot_reindex_positions(coords_path, population, distances_namespace='Arc Distances',
                           reindex_namespace='Tree Reindex', reindex_attribute='New Cell Index', 
                           fontSize=14, showFig = True, saveFig = False, verbose = False):
    """
    Plot septo-temporal position (longitudinal and transverse arc distances).

    :param coords_path:
    :param distances_namespace: 
    :param population: 

    """

    dx = 50
    dy = 50
    
        
    soma_distances = read_cell_attributes(coords_path, population, namespace=distances_namespace)
    cell_reindex = read_cell_attributes(coords_path, population, namespace=reindex_namespace)
    cell_reindex_dict = {}
    for k,v in cell_reindex:
        cell_reindex_dict[k] = v[reindex_attribute][0]
        
    fig = plt.figure(1, figsize=plt.figaspect(1.) * 2.)
    ax = plt.gca()

    distance_U = {}
    distance_V = {}
    for k,v in soma_distances:
        if cell_reindex_dict.has_key(k):
            distance_U[k] = v['U Distance'][0]
            distance_V[k] = v['V Distance'][0]
        
        
    distance_U_array = np.asarray([distance_U[k] for k in sorted(distance_U.keys())])
    distance_V_array = np.asarray([distance_V[k] for k in sorted(distance_V.keys())])

    x_min = np.min(distance_U_array)
    x_max = np.max(distance_U_array)
    y_min = np.min(distance_V_array)
    y_max = np.max(distance_V_array)

    (H, xedges, yedges) = np.histogram2d(distance_U_array, distance_V_array, bins=[dx, dy])


    ax.axis([x_min, x_max, y_min, y_max])

    X, Y = np.meshgrid(xedges, yedges)
    pcm = ax.pcolormesh(X, Y, H.T)
    
    ax.set_xlabel('Arc distance (septal - temporal) (um)', fontsize=fontSize)
    ax.set_ylabel('Arc distance (supra - infrapyramidal)  (um)', fontsize=fontSize)
    ax.set_title('Position distribution for population: %s' % (population), fontsize=fontSize)
    ax.set_aspect('equal')
    fig.colorbar(pcm, ax=ax, shrink=0.5, aspect=20)
    
    if saveFig: 
        if isinstance(saveFig, basestring):
            filename = saveFig
        else:
            filename = population+' Reindex Positions.png' 
            plt.savefig(filename)

    if showFig:
        show_figure()
    
    return ax


def plot_coords_in_volume(population, coords_path, coords_namespace, config, scale=15., subvol=True, verbose=False):
    
    env = Env(configFile=config)

    rotate = env.geometry['Rotation']
    min_extents = env.geometry['Parametric Surface']['Minimum Extent']
    max_extents = env.geometry['Parametric Surface']['Maximum Extent']

    pop_max_extent = None
    pop_min_extent = None
    for ((layer_name,max_extent),(_,min_extent)) in itertools.izip(max_extents.iteritems(),min_extents.iteritems()):

        layer_count = env.geometry['Cell Layer Counts'][population][layer_name]
        if layer_count > 0:
            if pop_max_extent is None:
                pop_max_extent = np.asarray(max_extent)
            else:
                pop_max_extent = np.maximum(pop_max_extent, np.asarray(max_extent))
            if pop_min_extent is None:
                pop_min_extent = np.asarray(min_extent)
            else:
                pop_min_extent = np.minimum(pop_min_extent, np.asarray(min_extent))
    
    if verbose:
        print('Reading coordinates...')

    coords = read_cell_attributes(coords_path, population, namespace=coords_namespace)

    
    xcoords = []
    ycoords = []
    zcoords = []
    for (k,v) in coords:
        xcoords.append(v['X Coordinate'][0])
        ycoords.append(v['Y Coordinate'][0])
        zcoords.append(v['Z Coordinate'][0])
        
    pts = np.concatenate((np.asarray(xcoords).reshape(-1,1), \
                          np.asarray(ycoords).reshape(-1,1), \
                          np.asarray(zcoords).reshape(-1,1)),axis=1)

    if verbose:
        print('Creating volume...')

    import DG_volume

    if subvol:
        subvol = DG_volume.make_volume (pop_min_extent[2], pop_max_extent[2], rotate=rotate)
    else:
        vol = DG_volume.make_volume (-3.95, 3.0, rotate=rotate)
    
    if verbose:
        print('Plotting volume...')

    from mayavi import mlab
    if subvol:
        subvol.mplot_surface(color=(0, 0.4, 0), opacity=0.33)
    else:
        vol.mplot_surface(color=(0, 1, 0), opacity=0.33)
    
    if verbose:
        print('Plotting coordinates in volume...')

    mlab.points3d(*pts.T, color=(1, 1, 0), scale_factor=scale)
    
    mlab.show()


    
def plot_trees_in_volume(population, forest_path, config, width=3., sample=0.05, subvol=True, verbose=False):
    
    env = Env(configFile=config)

    rotate = env.geometry['Rotation']

    min_extents = env.geometry['Parametric Surface']['Minimum Extent']
    max_extents = env.geometry['Parametric Surface']['Maximum Extent']

    pop_max_extent = None
    pop_min_extent = None
    for ((layer_name,max_extent),(_,min_extent)) in itertools.izip(max_extents.iteritems(),min_extents.iteritems()):

        layer_count = env.geometry['Cell Layer Counts'][population][layer_name]
        if layer_count > 0:
            if pop_max_extent is None:
                pop_max_extent = np.asarray(max_extent)
            else:
                pop_max_extent = np.maximum(pop_max_extent, np.asarray(max_extent))
            if pop_min_extent is None:
                pop_min_extent = np.asarray(min_extent)
            else:
                pop_min_extent = np.minimum(pop_min_extent, np.asarray(min_extent))
    

    if verbose:
        print('Creating volume...')

    import DG_volume

    if subvol:
        subvol = DG_volume.make_volume (pop_min_extent[2], pop_max_extent[2], rotate=rotate)
    else:
        vol = DG_volume.make_volume (-3.95, 3.0, rotate=rotate)
    

    import networkx as nx
    from mayavi import mlab
    
    if verbose:
        print('Plotting trees in volume...')

    (trees, _) = read_trees(forest_path, population)
    print trees
    for (_,tree_dict) in trees:

        s = np.random.random_sample()
        if s <= sample:
            xcoords = tree_dict['x']
            ycoords = tree_dict['y']
            zcoords = tree_dict['z']
            swc_type = tree_dict['swc_type']

            secnodes = tree_dict['section_topology']['nodes']
            src      = tree_dict['section_topology']['src']
            dst      = tree_dict['section_topology']['dst']

            dend_idxs = np.where(swc_type == 4)[0]
            dend_idx_set = set(dend_idxs.flat)

            edges = []
            for sec, nodes in secnodes.iteritems():
                for i in xrange(1, len(nodes)):
                    srcnode = nodes[i-1]
                    dstnode = nodes[i]
                    if ((srcnode in dend_idx_set) and (dstnode in dend_idx_set)):
                        edges.append((srcnode, dstnode))
            for (s,d) in itertools.izip(src,dst):
                srcnode = secnodes[s][-1]
                dstnode = secnodes[d][0]
                if ((srcnode in dend_idx_set) and (dstnode in dend_idx_set)):
                    edges.append((srcnode, dstnode))


            x = xcoords[dend_idxs].reshape(-1,)
            y = ycoords[dend_idxs].reshape(-1,)
            z = zcoords[dend_idxs].reshape(-1,)

            # Make a NetworkX graph out of our point and edge data
            g = make_geometric_graph(x, y, z, edges)

            # Compute minimum spanning tree using networkx
            # nx.mst returns an edge generator
            edges = nx.minimum_spanning_tree(g).edges(data=True)
            start_idx, end_idx, _ = np.array(list(edges)).T
            start_idx = start_idx.astype(np.int)
            end_idx   = end_idx.astype(np.int)

            # Plot this with Mayavi
            plot_graph(x, y, z, start_idx, end_idx,
                        edge_scalars=z[start_idx],
                        opacity=0.8,
                        colormap='summer',
                        line_width=width)
        else:
            g = None
            edges = []
            
    if verbose:
        print('Plotting volume...')
    if subvol:
        subvol.mplot_surface(color=(0, 0.4, 0), opacity=0.33)
    else:
        vol.mplot_surface(color=(0, 1, 0), opacity=0.33)
    mlab.show()



def plot_population_density(population, soma_coords, distances_namespace, max_u, max_v, bin_size=100., showFig = True, saveFig = False, verbose=True):
    """

    :param population: str
    :param soma_coords: dict of array
    :param u: array
    :param v: array
    :param distance_U: array
    :param distance_V: array
    :param max_u: float: u_distance
    :param max_v: float: v_distance
    :param bin_size: float
    :return:
    """
    fig1 = plt.figure()
    ax = fig1.add_subplot(111, projection='3d')
    pop_size = len(soma_coords[population]['x'])
    indexes = random.sample(range(pop_size), min(pop_size, 5000))
    ax.scatter(soma_coords[population]['x'][indexes], soma_coords[population]['y'][indexes],
               soma_coords[population]['z'][indexes], alpha=0.1, linewidth=0)
    scaling = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    ax.auto_scale_xyz(*[[np.min(scaling), np.max(scaling)]] * 3)
    ax.set_xlabel('X (um)')
    ax.set_ylabel('Y (um)')
    ax.set_zlabel('Z (um)')

    step_sizes = [int(max_u / bin_size), int(max_v / bin_size)]
    plt.figure(figsize=plt.figaspect(1.)*2.)
    population_indexes_u = get_array_index(u, soma_coords[population]['u'])
    population_indexes_v = get_array_index(v, soma_coords[population]['v'])
    H, u_edges, v_edges = np.histogram2d(distance_U[population_indexes_u, population_indexes_v],
                                         distance_V[population_indexes_u, population_indexes_v], step_sizes)
    H = np.rot90(H)
    H = np.flipud(H)
    Hmasked = np.ma.masked_where(H == 0, H)
    ax = plt.gca()
    pcm = ax.pcolormesh(u_edges, v_edges, Hmasked)
    ax.set_xlabel('Arc distance (septal - temporal) (um)')
    ax.set_ylabel('Arc distance (supra - infrapyramidal)  (um)')
    ax.set_title(population)
    ax.set_aspect('equal', 'box')
    clean_axes(ax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    cbar = plt.colorbar(pcm, cax=cax)
    cbar.ax.set_ylabel('Counts')

    if saveFig: 
        if isinstance(saveFig, basestring):
            filename = saveFig
        else:
            filename = distances_namespace+' '+'density.png'
            plt.savefig(filename)

    if showFig:
        show_figure()

    return ax


## Plot intracellular state trace 
def plot_intracellular_state (input_path, namespace_id, include = ['eachPop'], timeRange = None, timeVariable='t', variable='v', maxUnits = 1, unitNo = None,
                              orderInverse = False, labels = None, lw = 3, marker = '|', figSize = (15,8), fontSize = 14, saveFig = None, 
                              showFig = True, query = False, verbose = False): 
    ''' 
    Line plot of intracellular state variable (default: v). Returns the figure handle.

    input_path: file with spike data
    namespace_id: attribute namespace for spike events
    timeRange ([start:stop]): Time range of spikes shown; if None shows all (default: None)
    timeVariable: Name of variable containing spike times (default: 't')
    variable: Name of state variable (default: 'v')
    maxUnits (int): maximum number of units from each population that will be plotted  (default: 1)
    orderInverse (True|False): Invert the y-axis order (default: False)
    labels = ('legend', 'overlay'): Show population labels in a legend or overlayed on one side of raster (default: 'legend')
    lw (integer): Line width for each spike (default: 3)
    marker (char): Marker for each spike (default: '|')
    fontSize (integer): Size of text font (default: 14)
    figSize ((width, height)): Size of figure (default: (15,8))
    saveFig (None|True|'fileName'): File name where to save the figure (default: None)
    showFig (True|False): Whether to show the figure or not (default: True)
    '''

    comm = MPI.COMM_WORLD

    (population_ranges, N) = read_population_ranges(input_path)
    population_names  = read_population_names(input_path)

    pop_num_cells = {}
    for k in population_names:
        pop_num_cells[k] = population_ranges[k][1]

    # Replace 'eachPop' with list of populations
    if 'eachPop' in include: 
        include.remove('eachPop')
        for pop in population_names:
            include.append(pop)

    data = statedata.read_state (comm, input_path, include, namespace_id, timeVariable=timeVariable,
                                 variable=variable, timeRange=timeRange, verbose=verbose,
                                 maxUnits = maxUnits, unitNo = unitNo, query = query)

    if query:
        return

    states     = data['states']
    
    pop_colors = { pop_name: color_list[ipop%len(color_list)] for ipop, pop_name in enumerate(states.keys()) }
    
    stplots = []
    
    fig, ax1 = plt.subplots(figsize=figSize,sharex='all',sharey='all')
        
    for (pop_name, pop_states) in states.iteritems():
        
        for (gid, cell_states) in pop_states.iteritems():

            if verbose:
                print('Creating state plot for gid %i...' % gid)
            stplots.append(ax1.plot(cell_states[0], cell_states[1], linewidth=lw, marker=marker, c=pop_colors[pop_name], alpha=0.5, label=pop_name))
            
    ax1.set_xlim(timeRange)

    ax1.set_xlabel('Time (ms)', fontsize=fontSize)
    ax1.set_ylabel(variable, fontsize=fontSize)
    ax1.set_xlim(timeRange)
    
    # Add legend
    pop_labels = pop_name
    
    if labels == 'legend':
        legend_labels = pop_labels
        lgd = plt.legend(stplots, legend_labels, fontsize=fontSize, scatterpoints=1, markerscale=5.,
                         loc='upper right', bbox_to_anchor=(1.2, 1.0))
        ## From https://stackoverflow.com/questions/30413789/matplotlib-automatic-legend-outside-plot
        ## draw the legend on the canvas to assign it real pixel coordinates:
        plt.gcf().canvas.draw()
        ## transformation from pixel coordinates to Figure coordinates:
        transfig = plt.gcf().transFigure.inverted()
        ## Get the legend extents in pixels and convert to Figure coordinates.
        ## Pull out the farthest extent in the x direction since that is the canvas direction we need to adjust:
        lgd_pos = lgd.get_window_extent()
        lgd_coord = transfig.transform(lgd_pos)
        lgd_xmax = lgd_coord[1, 0]
        ## Do the same for the Axes:
        ax_pos = plt.gca().get_window_extent()
        ax_coord = transfig.transform(ax_pos)
        ax_xmax = ax_coord[1, 0]
        ## Adjust the Figure canvas using tight_layout for
        ## Axes that must move over to allow room for the legend to fit within the canvas:
        shift = 1 - (lgd_xmax - ax_xmax)
        plt.gcf().tight_layout(rect=(0, 0, shift, 1))
        
    if orderInverse:
        plt.gca().invert_yaxis()

    # save figure
    if saveFig: 
        if isinstance(saveFig, basestring):
            filename = saveFig
        else:
            filename = namespace_id+' '+'state.png'
            plt.savefig(filename)
                
    # show fig 
    if showFig:
        show_figure()
    
    return fig


## Plot spike raster
def plot_spike_raster (input_path, namespace_id, include = ['eachPop'], timeRange = None, timeVariable='t', maxSpikes = int(1e6),
                       orderInverse = False, labels = 'legend', popRates = False,
                       spikeHist = None, spikeHistBin = 5, lw = 3, marker = '|', figSize = (15,8), fontSize = 14, saveFig = None, 
                       showFig = True, verbose = False): 
    ''' 
    Raster plot of network spike times. Returns the figure handle.

    input_path: file with spike data
    namespace_id: attribute namespace for spike events
    timeRange ([start:stop]): Time range of spikes shown; if None shows all (default: None)
    timeVariable: Name of variable containing spike times (default: 't')
    maxSpikes (int): maximum number of spikes that will be plotted  (default: 1e6)
    orderInverse (True|False): Invert the y-axis order (default: False)
    labels = ('legend', 'overlay'): Show population labels in a legend or overlayed on one side of raster (default: 'legend')
    popRates = (True|False): Include population rates (default: False)
    spikeHist (None|'overlay'|'subplot'): overlay line over raster showing spike histogram (spikes/bin) (default: False)
    spikeHistBin (int): Size of bin in ms to use for histogram (default: 5)
    lw (integer): Line width for each spike (default: 3)
    marker (char): Marker for each spike (default: '|')
    fontSize (integer): Size of text font (default: 14)
    figSize ((width, height)): Size of figure (default: (15,8))
    saveFig (None|True|'fileName'): File name where to save the figure (default: None)
    showFig (True|False): Whether to show the figure or not (default: True)
    '''

    comm = MPI.COMM_WORLD

    (population_ranges, N) = read_population_ranges(input_path)
    population_names  = read_population_names(input_path)

    pop_num_cells = {}
    for k in population_names:
        pop_num_cells[k] = population_ranges[k][1]

    # Replace 'eachPop' with list of populations
    if 'eachPop' in include: 
        include.remove('eachPop')
        for pop in population_names:
            include.append(pop)

    spkdata = spikedata.read_spike_events (input_path, include, namespace_id, timeVariable=timeVariable, timeRange=timeRange, verbose=verbose)

    spkpoplst        = spkdata['spkpoplst']
    spkindlst        = spkdata['spkindlst']
    spktlst          = spkdata['spktlst']
    num_cell_spks    = spkdata['num_cell_spks']
    pop_active_cells = spkdata['pop_active_cells']
    tmin             = spkdata['tmin']
    tmax             = spkdata['tmax']
    
    timeRange = [tmin, tmax]

    # Calculate spike histogram if requested
    if spikeHist:
        all_spkts = np.concatenate(spktlst, axis=0)
        histoCount, bin_edges = np.histogram(all_spkts, bins = np.arange(timeRange[0], timeRange[1], spikeHistBin))
        histoT = bin_edges[:-1]+spikeHistBin/2

    maxN = 0
    minN = N
    if popRates:
        avg_rates = {}
        tsecs = (timeRange[1]-timeRange[0])/1e3 
        for i,pop_name in enumerate(spkpoplst):
            pop_num = len(pop_active_cells[pop_name])
            maxN = max(maxN, max(pop_active_cells[pop_name]))
            minN = min(minN, min(pop_active_cells[pop_name]))
            if pop_num > 0:
                if num_cell_spks[pop_name] == 0:
                    avg_rates[pop_name] = 0
                else:
                    avg_rates[pop_name] = num_cell_spks[pop_name] / pop_num / tsecs
        
    
    pop_colors = { pop_name: color_list[ipop%len(color_list)] for ipop, pop_name in enumerate(spkpoplst) }
    
    # Plot spikes
    fig, ax1 = plt.subplots(figsize=figSize)

    if verbose:
        print('Creating raster plot...')

    sctplots = []
    
    if spikeHist is None:

        for (pop_name, pop_spkinds, pop_spkts) in itertools.izip (spkpoplst, spkindlst, spktlst):

            if maxSpikes is not None:
                if int(maxSpikes) < len(pop_spkinds):
                    if verbose:
                        print('  Displaying only randomly sampled %i out of %i spikes for population %s' % (maxSpikes, len(pop_spkts), pop_name))
                    sample_inds = np.random.randint(0, len(pop_spkinds)-1, size=int(maxSpikes))
                    pop_spkts   = pop_spkts[sample_inds]
                    pop_spkinds = pop_spkinds[sample_inds]

            sctplots.append(ax1.scatter(pop_spkts, pop_spkinds, s=10, linewidths=lw, marker=marker, c=pop_colors[pop_name], alpha=0.5, label=pop_name))
        
        ax1.set_xlim(timeRange)

        ax1.set_xlabel('Time (ms)', fontsize=fontSize)
        ax1.set_ylabel('Cell Index', fontsize=fontSize)
        ax1.set_xlim([tmin, tmax])
        ax1.set_ylim(minN-1, maxN+1)
        
    elif spikeHist == 'subplot':

        gs = gridspec.GridSpec(2, 1, height_ratios=[2,1])
        ax1=plt.subplot(gs[0])
        
        for (pop_name, pop_spkinds, pop_spkts) in itertools.izip (spkpoplst, spkindlst, spktlst):

            if maxSpikes is not None:
                if int(maxSpikes) < len(pop_spkinds):
                    if verbose:
                        print('  Displaying only randomly sampled %i out of %i spikes for population %s' % (maxSpikes, len(pop_spkts), pop_name))
                    sample_inds = np.random.randint(0, len(pop_spkinds)-1, size=int(maxSpikes))
                    pop_spkts   = pop_spkts[sample_inds]
                    pop_spkinds = pop_spkinds[sample_inds]

            sctplots.append(ax1.scatter(pop_spkts, pop_spkinds, s=10, linewidths=lw, marker=marker, c=pop_colors[pop_name], alpha=0.5, label=pop_name))
            
        ax1.set_xlim(timeRange)

        ax1.set_xlabel('Time (ms)', fontsize=fontSize)
        ax1.set_ylabel('Cell Index', fontsize=fontSize)
        ax1.set_xlim(timeRange)
        ax1.set_ylim(minN-1, maxN+1)

        # Add legend
        if popRates:
            pop_labels = [pop_name + ' (%i active; %.3g Hz)' % (len(pop_active_cells[pop_name]), avg_rates[pop_name]) for pop_name in spkpoplst if pop_name in avg_rates]
        else:
            pop_labels = [pop_name + ' (%i active)' % (len(pop_active_cells[pop_name]))]
            
        if labels == 'legend':
            legend_labels = pop_labels
            lgd = plt.legend(sctplots, legend_labels, fontsize=fontSize, scatterpoints=1, markerscale=5.,
                             loc='upper right', bbox_to_anchor=(1.2, 1.0))
            ## From https://stackoverflow.com/questions/30413789/matplotlib-automatic-legend-outside-plot
            ## draw the legend on the canvas to assign it real pixel coordinates:
            plt.gcf().canvas.draw()
            ## transformation from pixel coordinates to Figure coordinates:
            transfig = plt.gcf().transFigure.inverted()
            ## Get the legend extents in pixels and convert to Figure coordinates.
            ## Pull out the farthest extent in the x direction since that is the canvas direction we need to adjust:
            lgd_pos = lgd.get_window_extent()
            lgd_coord = transfig.transform(lgd_pos)
            lgd_xmax = lgd_coord[1, 0]
            ## Do the same for the Axes:
            ax_pos = plt.gca().get_window_extent()
            ax_coord = transfig.transform(ax_pos)
            ax_xmax = ax_coord[1, 0]
            ## Adjust the Figure canvas using tight_layout for
            ## Axes that must move over to allow room for the legend to fit within the canvas:
            shift = 1 - (lgd_xmax - ax_xmax)
            plt.gcf().tight_layout(rect=(0, 0, shift, 1))

            
        
        # Plot spike hist
        if spikeHist == 'overlay':
            ax2 = ax1.twinx()
            ax2.plot (histoT, histoCount, linewidth=0.5)
            ax2.set_ylabel('Spike count', fontsize=fontSize) # add yaxis label in opposite side
            ax2.set_xlim(timeRange)
        elif spikeHist == 'subplot':
            ax2=plt.subplot(gs[1])
            ax2.plot (histoT, histoCount, linewidth=1.0)
            ax2.set_xlabel('Time (ms)', fontsize=fontSize)
            ax2.set_ylabel('Spike count', fontsize=fontSize)
            ax2.set_xlim(timeRange)

        if orderInverse:
            plt.gca().invert_yaxis()

        # save figure
        if saveFig: 
            if isinstance(saveFig, basestring):
                filename = saveFig
            else:
                filename = namespace_id+' '+'raster.png'
                plt.savefig(filename)
                
    # show fig 
    if showFig:
        show_figure()
    
    return fig


## Plot spike rates
def plot_spike_rates (input_path, namespace_id, include = ['eachPop'], timeRange = None, timeVariable='t', orderInverse = False, labels = 'legend', 
                      spikeRateBin = 25.0, sigma = 0.05, lw = 3, marker = '|', figSize = (15,8), fontSize = 14, saveFig = None, showFig = True, verbose = False): 
    ''' 
    Plot of network firing rates. Returns the figure handle.

    input_path: file with spike data
    namespace_id: attribute namespace for spike events
    timeRange ([start:stop]): Time range of spikes shown; if None shows all (default: None)
    timeVariable: Name of variable containing spike times (default: 't')
    orderInverse (True|False): Invert the y-axis order (default: False)
    labels = ('legend', 'overlay'): Show population labels in a legend or overlayed on one side of raster (default: 'legend')
    spikeRateBin (int): Size of bin in ms to use for rate computation (default: 5)
    lw (integer): Line width for each spike (default: 3)
    marker (char): Marker for each spike (default: '|')
    fontSize (integer): Size of text font (default: 14)
    figSize ((width, height)): Size of figure (default: (15,8))
    saveFig (None|True|'fileName'): File name where to save the figure (default: None)
    showFig (True|False): Whether to show the figure or not (default: True)
    '''

    comm = MPI.COMM_WORLD

    (population_ranges, N) = read_population_ranges(comm, input_path)
    population_names  = read_population_names(comm, input_path)

    pop_num_cells = {}
    for k in population_names:
        pop_num_cells[k] = population_ranges[k][1]

    # Replace 'eachPop' with list of populations
    if 'eachPop' in include: 
        include.remove('eachPop')
        for pop in population_names:
            include.append(pop)

    spkdata = spikedata.read_spike_events (comm, input_path, include, namespace_id, timeVariable=timeVariable,
                                           timeRange=timeRange, verbose=verbose)

    spkpoplst        = spkdata['spkpoplst']
    spkindlst        = spkdata['spkindlst']
    spktlst          = spkdata['spktlst']
    num_cell_spks    = spkdata['num_cell_spks']
    pop_active_cells = spkdata['pop_active_cells']
    tmin             = spkdata['tmin']
    tmax             = spkdata['tmax']

    timeRange = [tmin, tmax]

    # Calculate binned spike rates
    
    if verbose:
        print('Calculating spike rates...')

    time_bins  = np.arange(timeRange[0], timeRange[1], spikeRateBin)

    spkrate_dict = {}
    for subset, spkinds, spkts in itertools.izip(spkpoplst, spkindlst, spktlst):
        spkdict = spikedata.make_spike_dict(spkinds, spkts)
        rate_bin_dict = spikedata.spike_bin_rates(spkdict, time_bins, t_start=timeRange[0], t_stop=timeRange[1], sigma=sigma)
        i = 0
        rate_dict = {}
        for ind, (count_bins, rate_bins) in rate_bin_dict.iteritems():
            rates       = np.asarray(rate_bins)
            peak        = np.mean(rates[np.where(rates >= np.percentile(rates, 90.))[0]])
            peak_index  = np.where(rates == np.max(rates))[0][0]
            rate_dict[i] = { 'rate': rates, 'peak': peak, 'peak index': peak_index }
            i = i+1
        spkrate_dict[subset] = rate_dict
        if verbose:
            print('Calculated spike rates for %i cells in population %s' % (len(rate_dict), subset))

                    
    # Plot spikes
    fig, ax1 = plt.subplots(figsize=figSize)

    if verbose:
        print('Creating rate plot...')

    for (iplot, subset) in enumerate(spkpoplst):

        pop_rates = spkrate_dict[subset]
        
        peak_lst = []
        for ind, rate_dict in pop_rates.iteritems():
            rate       = rate_dict['rate']
            peak_index = rate_dict['peak index']
            peak_lst.append(peak_index)

        ind_peak_lst = list(enumerate(peak_lst))
        del(peak_lst)
        ind_peak_lst.sort(key=lambda (i, x): x, reverse=orderInverse)

        rate_lst = [ pop_rates[i]['rate'] for i, _ in ind_peak_lst ]
        del(ind_peak_lst)
        
        rate_matrix = np.matrix(rate_lst)
        del(rate_lst)

        color = color_list[iplot%len(color_list)]

        plt.subplot(len(spkpoplst),1,iplot+1)  # if subplot, create new subplot
        plt.title (str(subset), fontsize=fontSize)

        im = plt.imshow(rate_matrix, origin='lower', aspect='auto', #interpolation='bicubic',
                        extent=[timeRange[0], timeRange[1], 0, rate_matrix.shape[0]], cmap=cm.jet)

        if iplot == 0: 
            plt.ylabel('Relative Cell Index', fontsize=fontSize)
        if iplot == len(spkpoplst)-1:
            plt.xlabel('Time (ms)', fontsize=fontSize)

        cbar = plt.colorbar(im)
        cbar.ax.set_ylabel('Firing Rate (Hz)', fontsize=fontSize)
        

                
    # show fig 
    if showFig:
        show_figure()
    
    return fig

## Plot spike histogram
def plot_spike_histogram (input_path, namespace_id, include = ['eachPop'], timeVariable='t', timeRange = None, 
                          popRates = False, binSize = 5., smooth = 0, quantity = 'rate',
                          figSize = (15,8), overlay=True, graphType='bar',
                          fontSize = 14, lw = 3, saveFig = None, showFig = True, verbose = False): 
    ''' 
    Plots spike histogram. Returns figure handle.

        - input_path: file with spike data
        - namespace_id: attribute namespace for spike events
        - include (['eachPop'|<population name>]): List of data series to include. 
            (default: ['eachPop'] - expands to the name of each population)
        - timeVariable: Name of variable containing spike times (default: 't')
        - timeRange ([start:stop]): Time range of spikes shown; if None shows all (default: None)
        - binSize (int): Size in ms of each bin (default: 5)
        - overlay (True|False): Whether to overlay the data lines or plot in separate subplots (default: True)
        - graphType ('line'|'bar'): Type of graph to use (line graph or bar plot) (default: 'line')
        - quantity ('rate'|'count'): Quantity of y axis (firing rate in Hz, or spike count) (default: 'rate')
        - figSize ((width, height)): Size of figure (default: (10,8))
        - fontSize (integer): Size of text font (default: 14)
        - lw (integer): Line width for each spike (default: 3)
        - saveFig (None|True|'fileName'): File name where to save the figure;
            if set to True uses filename from simConfig (default: None)
        - showFig (True|False): Whether to show the figure or not (default: True)
    '''
    comm = MPI.COMM_WORLD

    (population_ranges, N) = read_population_ranges(comm, input_path)
    population_names  = read_population_names(comm, input_path)

    pop_num_cells = {}
    for k in population_names:
        pop_num_cells[k] = population_ranges[k][1]

    
    # Replace 'eachPop' with list of populations
    if 'eachPop' in include: 
        include.remove('eachPop')
        for pop in population_names:
            include.append(pop)

    spkdata = spikedata.read_spike_events (comm, input_path, include, namespace_id, timeVariable=timeVariable,
                                           timeRange=timeRange, verbose=verbose, maxSpikes = 1e6)

    spkpoplst        = spkdata['spkpoplst']
    spkindlst        = spkdata['spkindlst']
    spktlst          = spkdata['spktlst']
    num_cell_spks    = spkdata['num_cell_spks']
    pop_active_cells = spkdata['pop_active_cells']
    tmin             = spkdata['tmin']
    tmax             = spkdata['tmax']

    timeRange = [tmin, tmax]

    avg_rates = {}
    maxN = 0
    minN = N
    if popRates:
        tsecs = (timeRange[1]-timeRange[0])/1e3 
        for i,pop_name in enumerate(spkpoplst):
            pop_num = len(pop_active_cells[pop_name])
            maxN = max(maxN, max(pop_active_cells[pop_name]))
            minN = min(minN, min(pop_active_cells[pop_name]))
            if pop_num > 0:
                if num_cell_spks[pop_name] == 0:
                    avg_rates[pop_name] = 0
                else:
                    avg_rates[pop_name] = num_cell_spks[pop_name] / pop_num / tsecs
            
    # Y-axis label
    if quantity == 'rate':
        yaxisLabel = 'Mean cell firing rate (Hz)'
    elif quantity == 'count':
        yaxisLabel = 'Spike count'
    elif quantity == 'active':
        yaxisLabel = 'Active cell count'
    else:
        print 'Invalid quantity value %s', (quantity)
        return

    # create fig
    fig, axes = plt.subplots(len(spkpoplst), 1, figsize=figSize, sharex=True)
        
            
    if verbose:
        print('Calculating spike rates...')
        

    time_bins  = np.arange(timeRange[0], timeRange[1], binSize)

    
    hist_dict = {}
    if quantity == 'rate':
        for subset, spkinds, spkts in itertools.izip(spkpoplst, spkindlst, spktlst):
            spkdict = spikedata.make_spike_dict(spkinds, spkts)
            rate_bin_dict = spikedata.spike_bin_rates(spkdict, time_bins, t_start=timeRange[0], t_stop=timeRange[1])
            del(spkdict)
            bin_dict      = defaultdict(lambda: {'rates':0.0, 'counts':0, 'active': 0})
            for (ind, (counts, rates)) in rate_bin_dict.iteritems():
                for ibin in xrange(0, time_bins.size):
                    if counts[ibin-1] > 0:
                        d = bin_dict[ibin]
                        d['rates']  += rates[ibin-1]
                        d['counts'] += counts[ibin-1]
                    d['active'] += 1
            hist_dict[subset] = bin_dict
            if verbose:
                print('Calculated spike rates for %i cells in population %s' % (len(rate_bin_dict), subset))
    else:
        for subset, spkinds, spkts in itertools.izip(spkpoplst, spkindlst, spktlst):
            spkdict = spikedata.make_spike_dict(spkinds, spkts)
            count_bin_dict = spikedata.spike_bin_counts(spkdict, time_bins)
            del(spkdict)
            bin_dict      = defaultdict(lambda: {'counts':0, 'active': 0})
            for (ind, counts) in count_bin_dict.iteritems():
                for ibin in xrange(0, time_bins.size):
                    if counts[ibin-1] > 0:
                        d = bin_dict[ibin]
                        d['counts'] += counts[ibin-1]
                        d['active'] += 1
            hist_dict[subset] = bin_dict
            if verbose:
                print('Calculated spike counts for %i cells in population %s' % (len(count_bin_dict), subset))
        
            
    del spkindlst, spktlst

    if verbose:
        print('Plotting spike histogram...')

    # Plot separate line for each entry in include
    for iplot, subset in enumerate(spkpoplst):

        histoT = time_bins+binSize/2
        bin_dict = hist_dict[subset]

        if quantity=='rate':
            histoCount = np.asarray([bin_dict[ibin]['rates'] / bin_dict[ibin]['active'] for ibin in xrange(0, time_bins.size)])
        elif quantity=='active':
            histoCount = np.asarray([bin_dict[ibin]['active'] for ibin in xrange(0, time_bins.size)])
        else:
            histoCount = np.asarray([bin_dict[ibin]['counts'] for ibin in xrange(0, time_bins.size)])

        del bin_dict
        del hist_dict[subset]
        
        color = color_list[iplot%len(color_list)]

        if not overlay:
            if popRates:
                label = str(subset)  + ' (%i active; %.3g Hz)' % (len(pop_active_cells[subset]), avg_rates[subset])
            else:
                label = str(subset)  + ' (%i active)' % (len(pop_active_cells[subset]))
            plt.subplot(len(spkpoplst),1,iplot+1)
            plt.title (label, fontsize=fontSize)
            
        if smooth:
            hsignal = signal.savgol_filter(histoCount, window_length=2*(len(histoCount)/16) + 1, polyorder=smooth) 
        else:
            hsignal = histoCount
        
        if graphType == 'line':
            plt.plot (histoT, hsignal, linewidth=lw, color = color)
        elif graphType == 'bar':
            plt.bar(histoT, hsignal, width = binSize, color = color)

        if iplot == 0:
            plt.ylabel(yaxisLabel, fontsize=fontSize)
        if iplot == len(spkpoplst)-1:
            plt.xlabel('Time (ms)', fontsize=fontSize)
        else:
            plt.tick_params(labelbottom='off')

        #axes[iplot].xaxis.set_visible(False)
            
        plt.xlim(timeRange)

    if len(spkpoplst) < 5:  # if apply tight_layout with many subplots it inverts the y-axis
        try:
            plt.tight_layout()
        except:
            pass

    # Add legend
    if overlay:
        for i,subset in enumerate(spkpoplst):
            plt.plot(0,0,color=color_list[i%len(color_list)],label=str(subset))
        plt.legend(fontsize=fontSize, bbox_to_anchor=(1.04, 1), loc=2, borderaxespad=0.)
        maxLabelLen = min(10,max([len(str(l)) for l in include]))
        plt.subplots_adjust(right=(0.9-0.012*maxLabelLen))


    if saveFig: 
        if isinstance(saveFig, basestring):
            filename = saveFig
        else:
            filename = namespace_id+' '+'histogram.png'
        plt.savefig(filename)

    if showFig:
        show_figure()

    return fig



## Plot spike distribution per cell
def plot_spike_distribution_per_cell (input_path, namespace_id, include = ['eachPop'], timeVariable='t', timeRange = None, 
                                      overlay=True, quantity = 'rate', figSize = (15,8),
                                      fontSize = 14, lw = 3, saveFig = None, showFig = True, verbose = False): 
    ''' 
    Plots distributions of spike rate/count. Returns figure handle.

        - input_path: file with spike data
        - namespace_id: attribute namespace for spike events
        - include (['eachPop'|<population name>]): List of data series to include. 
            (default: ['eachPop'] - expands to the name of each population)
        - timeVariable: Name of variable containing spike times (default: 't')
        - timeRange ([start:stop]): Time range of spikes shown; if None shows all (default: None)
        - overlay (True|False): Whether to overlay the data lines or plot in separate subplots (default: True)
        - quantity ('rate'|'count'): Quantity of y axis (firing rate in Hz, or spike count) (default: 'rate')
        - figSize ((width, height)): Size of figure (default: (10,8))
        - fontSize (integer): Size of text font (default: 14)
        - lw (integer): Line width for each spike (default: 3)
        - saveFig (None|True|'fileName'): File name where to save the figure;
            if set to True uses filename from simConfig (default: None)
        - showFig (True|False): Whether to show the figure or not (default: True)
    '''
    comm = MPI.COMM_WORLD


    (population_ranges, N) = read_population_ranges(comm, input_path)
    population_names  = read_population_names(comm, input_path)

    pop_num_cells = {}
    for k in population_names:
        pop_num_cells[k] = population_ranges[k][1]

    
    # Replace 'eachPop' with list of populations
    if 'eachPop' in include: 
        include.remove('eachPop')
        for pop in population_names:
            include.append(pop)

    spkdata = spikedata.read_spike_events (comm, input_path, include, namespace_id, timeVariable=timeVariable,
                                           timeRange=timeRange, verbose=verbose)

    spkpoplst        = spkdata['spkpoplst']
    spkindlst        = spkdata['spkindlst']
    spktlst          = spkdata['spktlst']
    num_cell_spks    = spkdata['num_cell_spks']
    pop_active_cells = spkdata['pop_active_cells']
    tmin             = spkdata['tmin']
    tmax             = spkdata['tmax']

    timeRange = [tmin, tmax]
            
    # Y-axis label
    if quantity == 'rate':
        yaxisLabel = 'Cell firing rate (Hz)'
    elif quantity == 'count':
        yaxisLabel = 'Spike count'
    else:
        print 'Invalid quantity value %s', (quantity)
        return

    # create fig
    fig, axes = plt.subplots(len(spkpoplst), 1, figsize=figSize, sharex=True)

    if verbose:
        print('Plotting spike distribution...')
        
    # Plot separate line for each entry in include
    for iplot, subset in enumerate(spkpoplst):

        spkts    = spktlst[iplot]
        spkinds  = spkindlst[iplot]

        u, counts = np.unique(spkinds, return_counts=True)
        if quantity == 'rate':
            rate_dict = spike_rates(spkinds, spkts)
            rates = [rate_dict[ind] for ind in u]

        color = color_list[iplot%len(color_list)]

        if not overlay:
            label = str(subset)  + ' (%i active)' % (len(pop_active_cells[subset]))
            plt.subplot(len(spkpoplst),1,iplot+1)
            plt.title (label, fontsize=fontSize)
            
        if quantity == 'rate':
            y = rates
        elif quantity == 'count':
            y = counts

        plt.plot(u,y)
        
        if iplot == 0:
            plt.ylabel(yaxisLabel, fontsize=fontSize)
        if iplot == len(spkpoplst)-1:
            plt.xlabel('Cell index', fontsize=fontSize)
        else:
            plt.tick_params(labelbottom='off')


    if len(spkpoplst) < 5:  # if apply tight_layout with many subplots it inverts the y-axis
        try:
            plt.tight_layout()
        except:
            pass

    # Add legend
    if overlay:
        for i,subset in enumerate(spkpoplst):
            plt.plot(0,0,color=color_list[i%len(color_list)],label=str(subset))
        plt.legend(fontsize=fontSize, bbox_to_anchor=(1.04, 1), loc=2, borderaxespad=0.)
        maxLabelLen = min(10,max([len(str(l)) for l in include]))
        plt.subplots_adjust(right=(0.9-0.012*maxLabelLen))


    if saveFig: 
        if isinstance(saveFig, basestring):
            filename = saveFig
        else:
            filename = namespace_id+' '+'distribution.png'
        plt.savefig(filename)

    if showFig:
        show_figure()

    return fig


## Plot spike distribution per time
def plot_spike_distribution_per_time (input_path, namespace_id, include = ['eachPop'],
                                      timeBinSize = 50.0, binCount = 10,
                                      timeVariable='t', timeRange = None, 
                                      overlay=True, quantity = 'rate', alpha_fill = 0.2, figSize = (15,8),
                                      fontSize = 14, lw = 3, saveFig = None, showFig = True, verbose = False): 
    ''' 
    Plots distributions of spike rate/count. Returns figure handle.

        - input_path: file with spike data
        - namespace_id: attribute namespace for spike events
        - include (['eachPop'|<population name>]): List of data series to include. 
            (default: ['eachPop'] - expands to the name of each population)
        - timeVariable: Name of variable containing spike times (default: 't')
        - timeRange ([start:stop]): Time range of spikes shown; if None shows all (default: None)
        - overlay (True|False): Whether to overlay the data lines or plot in separate subplots (default: True)
        - quantity ('rate'|'count'): Units of x axis (firing rate in Hz, or spike count) (default: 'rate')
        - figSize ((width, height)): Size of figure (default: (10,8))
        - fontSize (integer): Size of text font (default: 14)
        - lw (integer): Line width for each spike (default: 3)
        - saveFig (None|True|'fileName'): File name where to save the figure;
            if set to True uses filename from simConfig (default: None)
        - showFig (True|False): Whether to show the figure or not (default: True)
    '''
    comm = MPI.COMM_WORLD


    (population_ranges, N) = read_population_ranges(comm, input_path)
    population_names  = read_population_names(comm, input_path)

    pop_num_cells = {}
    for k in population_names:
        pop_num_cells[k] = population_ranges[k][1]

    
    # Replace 'eachPop' with list of populations
    if 'eachPop' in include: 
        include.remove('eachPop')
        for pop in population_names:
            include.append(pop)

    spkdata = spikedata.read_spike_events (comm, input_path, include, namespace_id, timeVariable=timeVariable,
                                           timeRange=timeRange, verbose=verbose)

    spkpoplst        = spkdata['spkpoplst']
    spkindlst        = spkdata['spkindlst']
    spktlst          = spkdata['spktlst']
    num_cell_spks    = spkdata['num_cell_spks']
    pop_active_cells = spkdata['pop_active_cells']
    tmin             = spkdata['tmin']
    tmax             = spkdata['tmax']

    timeRange = [tmin, tmax]
            
    # Y-axis label
    if quantity == 'rate':
        xaxisLabel = 'Cell firing rate (Hz)'
    elif quantity == 'count':
        xaxisLabel = 'Spike count'
    else:
        print 'Invalid quantity value %s', (quantity)
        return

    # create fig
    fig, axes = plt.subplots(len(spkpoplst), 1, figsize=figSize, sharex=True)

    # Plot separate line for each entry in include
    for iplot, subset in enumerate(spkpoplst):

        spkts         = spktlst[iplot]
        spkinds       = spkindlst[iplot]
        bins          = np.arange(timeRange[0], timeRange[1], timeBinSize)
        spkdict       = spikedata.make_spike_dict(spkinds, spkts)
        rate_bin_dict = spikedata.spike_bin_rates(spkdict, bins, t_start=timeRange[0], t_stop=timeRange[1])
        max_count     = np.zeros(bins.size-1)
        max_rate      = np.zeros(bins.size-1)
        bin_dict      = defaultdict(lambda: {'counts': [], 'rates': []})
        for ind, (count_bins, rate_bins) in rate_bin_dict.iteritems():
            counts     = count_bins
            rates      = rate_bins
            for ibin in xrange(1, bins.size+1):
                if counts[ibin-1] > 0:
                    d = bin_dict[ibin]
                    d['counts'].append(counts[ibin-1])
                    d['rates'].append(rates[ibin-1])
            max_count  = np.maximum(max_count, np.asarray(count_bins))
            max_rate   = np.maximum(max_rate, np.asarray(rate_bins))

        histlst  = []
        for ibin in sorted(bin_dict.keys()):
            d = bin_dict[ibin]
            counts = d['counts']
            rates = d['rates']
            if quantity == 'rate':
                histoCount, bin_edges = np.histogram(np.asarray(rates), bins = binCount, range=(0.0, float(max_rate[ibin-1])))
            else:
                histoCount, bin_edges = np.histogram(np.asarray(counts), bins = binCount, range=(0.0, float(max_count[ibin-1])))
            histlst.append(histoCount)

            
        bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
        
        if not overlay:
            label = str(subset)  + ' (%i active)' % (len(pop_active_cells[subset]))
            plt.subplot(len(spkpoplst),1,iplot+1)
            plt.title (label, fontsize=fontSize)

        hist_mean = []
        hist_std  = []
        for i in xrange(0, binCount):
            binvect = np.asarray([hist[i] for hist in histlst])
            hist_mean.append(np.mean(binvect))
            hist_std.append(np.std(binvect))
            
        color = color_list[iplot%len(color_list)]

        ymin = np.asarray(hist_mean) - hist_std
        ymax = np.asarray(hist_mean) + hist_std

        x = np.linspace(bin_centers.min(),bin_centers.max(),100)
        y_smooth    = np.clip(interpolate.spline(bin_centers, hist_mean, x), 0, None)
        ymax_smooth = np.clip(interpolate.spline(bin_centers, ymax, x), 0, None)
        ymin_smooth = np.clip(interpolate.spline(bin_centers, ymin, x), 0, None)
        plt.plot(x, y_smooth, color=color)
        plt.fill_between(x, ymax_smooth, ymin_smooth, color=color, alpha=alpha_fill)
        
        if iplot == 0:
            plt.ylabel('Cell Count', fontsize=fontSize)
        if iplot == len(spkpoplst)-1:
            plt.xlabel(xaxisLabel, fontsize=fontSize)
        else:
            plt.tick_params(labelbottom='off')
        plt.autoscale(enable=True, axis='both', tight=True)
        plt.xticks(np.linspace(bin_centers.min(),bin_centers.max(),binCount))

    if len(spkpoplst) < 5:  # if apply tight_layout with many subplots it inverts the y-axis
        try:
            plt.tight_layout()
        except:
            pass

    # Add legend
    if overlay:
        for i,subset in enumerate(spkpoplst):
            plt.plot(0,0,color=color_list[i%len(color_list)],label=str(subset))
        plt.legend(fontsize=fontSize, bbox_to_anchor=(1.04, 1), loc=2, borderaxespad=0.)
        maxLabelLen = min(10,max([len(str(l)) for l in include]))
        plt.subplots_adjust(right=(0.9-0.012*maxLabelLen))


    if saveFig: 
        if isinstance(saveFig, basestring):
            filename = saveFig
        else:
            filename = namespace_id+' '+'distribution.png'
        plt.savefig(filename)

    if showFig:
        show_figure()

    return fig


## Plot spatial information distribution
def plot_spatial_information (spike_input_path, spike_namespace_id, 
                              trajectory_path, trajectory_id, include = ['eachPop'],
                              positionBinSize = 5.0, binCount = 50,
                              timeVariable='t', timeRange = None, 
                              alpha_fill = 0.2, figSize = (15,8), overlay = False,
                              fontSize = 14, lw = 3, loadData = None, saveData = None,
                              saveFig = None, showFig = True, verbose = False): 
    ''' 
    Plots distributions of spatial information per cell. Returns figure handle.

        - input_path: file with spike data
        - namespace_id: attribute namespace for spike events
        - include (['eachPop'|<population name>]): List of data series to include. 
            (default: ['eachPop'] - expands to the name of each population)
        - timeVariable: Name of variable containing spike times (default: 't')
        - timeRange ([start:stop]): Time range of spikes shown; if None shows all (default: None)
        - overlay (True|False): Whether to overlay the data lines or plot in separate subplots (default: True)
        - quantity ('rate'|'count'): Units of x axis (firing rate in Hz, or spike count) (default: 'rate')
        - figSize ((width, height)): Size of figure (default: (10,8))
        - fontSize (integer): Size of text font (default: 14)
        - lw (integer): Line width for each spike (default: 3)
        - saveFig (None|True|'fileName'): File name where to save the figure;
            if set to True uses filename from simConfig (default: None)
        - showFig (True|False): Whether to show the figure or not (default: True)
    '''
    comm = MPI.COMM_WORLD

    trajectory = stimulus.read_trajectory (comm, trajectory_path, trajectory_id, verbose=False)

    (population_ranges, N) = read_population_ranges(comm, spike_input_path)
    population_names  = read_population_names(comm, spike_input_path)

    pop_num_cells = {}
    for k in population_names:
        pop_num_cells[k] = population_ranges[k][1]

    
    # Replace 'eachPop' with list of populations
    if 'eachPop' in include: 
        include.remove('eachPop')
        for pop in population_names:
            include.append(pop)

    if loadData is None:
        spkdata = spikedata.read_spike_events (comm, spike_input_path, include, spike_namespace_id,
                                               timeVariable=timeVariable, timeRange=timeRange,
                                               verbose=verbose)

        spkpoplst        = spkdata['spkpoplst']
        spkindlst        = spkdata['spkindlst']
        spktlst          = spkdata['spktlst']
        num_cell_spks    = spkdata['num_cell_spks']
        pop_active_cells = spkdata['pop_active_cells']
        tmin             = spkdata['tmin']
        tmax             = spkdata['tmax']

        timeRange = [tmin, tmax]
    else:
        spkpoplst = include
            
    # create fig
    fig, axes = plt.subplots(len(spkpoplst), 1, figsize=figSize, sharex=True)

    histlst = []
    # Plot separate line for each entry in include
    for iplot, subset in enumerate(spkpoplst):

        if loadData:
            MI_dict = read_cell_attributes(loadData[iplot], subset, namespace='Spatial Mutual Information')
        else:
            spkts         = spktlst[iplot]
            spkinds       = spkindlst[iplot]
            spkdict       = spikedata.make_spike_dict(spkinds, spkts)
            if saveData:
                if isinstance(saveData, basestring):
                    filename = saveData
                else:
                    filename = spike_namespace_id+' '+subset
            else:
                filename = False
                MI_dict       = spikedata.spatial_information(trajectory, spkdict, timeRange, positionBinSize, saveData=filename)

        MI_lst  = []
        for ind in sorted(MI_dict.keys()):
            MI = MI_dict[ind]
            MI_lst.append(MI)
        del(MI_dict)

        MI_array = np.asarray(MI_lst, dtype=np.float32)
        del(MI_lst)
        
        if not overlay:
            if loadData:
                label = str(subset)  + ' (mean MI %.2f bits)' % (np.mean(MI_array))
            else:
                label = str(subset)  + ' (%i active; mean MI %.2f bits)' % (len(pop_active_cells[subset]),np.mean(MI_array))
            plt.subplot(len(spkpoplst),1,iplot+1)
            plt.title (label, fontsize=fontSize)
            
        color = color_list[iplot%len(color_list)]

        #MI_hist, bin_edges = np.histogram(MI_array, bins = binCount)
        #bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
        #plt.bar(bin_centers, MI_hist, color=color, width=0.3*(np.mean(np.diff(bin_edges))))

        n, bins, patches = plt.hist(MI_array, bins=binCount, alpha=0.75, rwidth=1, color=color)
        plt.xticks(fontsize=fontSize)
                   
        if iplot == 0:
            plt.ylabel('Cell Index', fontsize=fontSize)
        if iplot == len(spkpoplst)-1:
            plt.xlabel('Mutual Information [bits]', fontsize=fontSize)
        else:
            plt.tick_params(labelbottom='off')
        plt.autoscale(enable=True, axis='both', tight=True)

    if len(spkpoplst) < 5:  # if apply tight_layout with many subplots it inverts the y-axis
        try:
            plt.tight_layout()
        except:
            pass

    # Add legend
    if overlay:
        for i,subset in enumerate(spkpoplst):
            plt.plot(0,0,color=color_list[i%len(color_list)],label=str(subset))
        plt.legend(fontsize=fontSize, bbox_to_anchor=(1.04, 1), loc=2, borderaxespad=0.)
        maxLabelLen = min(10,max([len(str(l)) for l in include]))
        plt.subplots_adjust(right=(0.9-0.012*maxLabelLen))


    if saveFig: 
        if isinstance(saveFig, basestring):
            filename = saveFig
        else:
            filename = namespace_id+' '+'information.png'
        plt.savefig(filename)

    if showFig:
        show_figure()

    return fig


def plot_place_fields (spike_input_path, spike_namespace_id, 
                       trajectory_path, trajectory_id, include = ['eachPop'],
                       positionBinSize = 5.0, binCount = 50,
                       timeVariable='t', timeRange = None, 
                       alpha_fill = 0.2, figSize = (15,8), overlay = False,
                       fontSize = 14, lw = 3, loadData = None, saveData = None,
                       saveFig = None, showFig = True, verbose = False): 
    ''' 
    Plots distributions of place fields per cell. Returns figure handle.

        - input_path: file with spike data
        - namespace_id: attribute namespace for spike events
        - include (['eachPop'|<population name>]): List of data series to include. 
            (default: ['eachPop'] - expands to the name of each population)
        - timeVariable: Name of variable containing spike times (default: 't')
        - timeRange ([start:stop]): Time range of spikes shown; if None shows all (default: None)
        - overlay (True|False): Whether to overlay the data lines or plot in separate subplots (default: True)
        - quantity ('rate'|'count'): Units of x axis (firing rate in Hz, or spike count) (default: 'rate')
        - figSize ((width, height)): Size of figure (default: (10,8))
        - fontSize (integer): Size of text font (default: 14)
        - lw (integer): Line width for each spike (default: 3)
        - saveFig (None|True|'fileName'): File name where to save the figure;
            if set to True uses filename from simConfig (default: None)
        - showFig (True|False): Whether to show the figure or not (default: True)
    '''
    comm = MPI.COMM_WORLD

    trajectory = stimulus.read_trajectory (comm, trajectory_path, trajectory_id, verbose=False)

    (population_ranges, N) = read_population_ranges(comm, spike_input_path)
    population_names  = read_population_names(comm, spike_input_path)

    pop_num_cells = {}
    for k in population_names:
        pop_num_cells[k] = population_ranges[k][1]

    
    # Replace 'eachPop' with list of populations
    if 'eachPop' in include: 
        include.remove('eachPop')
        for pop in population_names:
            include.append(pop)

    if loadData is None:
        spkdata = spikedata.read_spike_events (comm, spike_input_path, include, spike_namespace_id,
                                               timeVariable=timeVariable, timeRange=timeRange,
                                               verbose=verbose)

        spkpoplst        = spkdata['spkpoplst']
        spkindlst        = spkdata['spkindlst']
        spktlst          = spkdata['spktlst']
        num_cell_spks    = spkdata['num_cell_spks']
        pop_active_cells = spkdata['pop_active_cells']
        tmin             = spkdata['tmin']
        tmax             = spkdata['tmax']

        timeRange = [tmin, tmax]
    else:
        spkpoplst = include
            
    # create fig
    fig, axes = plt.subplots(len(spkpoplst), 1, figsize=figSize, sharex=True)

    histlst = []
    # Plot separate line for each entry in include
    for iplot, subset in enumerate(spkpoplst):

        if loadData:
            rate_bin_dict = read_cell_attributes(loadData[iplot], subset, namespace='Instantaneous Rate')
        else:
            spkts         = spktlst[iplot]
            spkinds       = spkindlst[iplot]
            spkdict       = spikedata.make_spike_dict(spkinds, spkts)
            if saveData:
                if isinstance(saveData, basestring):
                    filename = saveData
                else:
                    filename = spike_namespace_id+' '+subset
            else:
                filename = False
        PF_dict  = spikedata.place_fields(rate_bin_dict, timeRange)

        PF_lst  = []
        for ind in sorted(PF_dict.keys()):
            PF = PF_dict[ind]
            PF_lst.append(PF)
        del(PF_dict)

        PF_array = np.asarray(PF_lst, dtype=np.float32)
        del(PF_lst)
        
        if not overlay:
            if loadData:
                label = str(subset)  + ' (mean %i place fields)' % (np.mean(PF_array))
            else:
                label = str(subset)  + ' (%i active; mean %i place fields)' % (len(pop_active_cells[subset]),np.mean(PF_array))
            plt.subplot(len(spkpoplst),1,iplot+1)
            plt.title (label, fontsize=fontSize)
            
        color = color_list[iplot%len(color_list)]

        n, bins, patches = plt.hist(PF_array, bins=binCount, alpha=0.75, rwidth=1, color=color)
        plt.xticks(fontsize=fontSize)
                   
        if iplot == 0:
            plt.ylabel('Cell Index', fontsize=fontSize)
        if iplot == len(spkpoplst)-1:
            plt.xlabel('# Place fields', fontsize=fontSize)
        else:
            plt.tick_params(labelbottom='off')
        plt.autoscale(enable=True, axis='both', tight=True)

    if len(spkpoplst) < 5:  # if apply tight_layout with many subplots it inverts the y-axis
        try:
            plt.tight_layout()
        except:
            pass

    # Add legend
    if overlay:
        for i,subset in enumerate(spkpoplst):
            plt.plot(0,0,color=color_list[i%len(color_list)],label=str(subset))
        plt.legend(fontsize=fontSize, bbox_to_anchor=(1.04, 1), loc=2, borderaxespad=0.)
        maxLabelLen = min(10,max([len(str(l)) for l in include]))
        plt.subplots_adjust(right=(0.9-0.012*maxLabelLen))


    if saveFig: 
        if isinstance(saveFig, basestring):
            filename = saveFig
        else:
            filename = namespace_id+' '+'information.png'
        plt.savefig(filename)

    if showFig:
        show_figure()

    return fig




def plot_rate_PSD (input_path, namespace_id, include = ['eachPop'], timeRange = None, timeVariable='t', 
                   binSize = 5, Fs = 200, nperseg = 128, smooth = 0, overlay = True,
                   figSize = (8,8), fontSize = 14, lw = 3, saveFig = None, showFig = True, verbose = False): 
    ''' 
    Plots firing rate power spectral density (PSD). Returns figure handle.
        - input_path: file with spike data
        - namespace_id: attribute namespace for spike events
        - include (['eachPop'|<population name>]): List of data series to include. 
            (default: ['eachPop'] - expands to the name of each population)
        - timeRange ([start:stop]): Time range of spikes shown; if None shows all (default: None)
        - timeVariable: Name of variable containing spike times (default: 't')
        - binSize (int): Size in ms of each bin (default: 5)
        - Fs (float): sampling frequency
        - nperseg (int): Length of each segment. 
        - overlay (True|False): Whether to overlay the data lines or plot in separate subplots (default: True)
        - figSize ((width, height)): Size of figure (default: (8,8))
        - fontSize (integer): Size of text font (default: 14)
        - lw (integer): Line width for each spike (default: 3)
        - saveFig (None|True|'fileName'): File name where to save the figure;
            if set to True uses filename from simConfig (default: None)
        - showFig (True|False): Whether to show the figure or not (default: True)

    '''
    comm = MPI.COMM_WORLD

    (population_ranges, N) = read_population_ranges(comm, input_path)
    population_names  = read_population_names(comm, input_path)

    pop_num_cells = {}
    for k in population_names:
        pop_num_cells[k] = population_ranges[k][1]

    
    # Replace 'eachPop' with list of populations
    if 'eachPop' in include: 
        include.remove('eachPop')
        for pop in population_names:
            include.append(pop)

    spkdata = spikedata.read_spike_events (comm, input_path, include, namespace_id, timeVariable=timeVariable, 
                                           timeRange=timeRange, verbose=verbose)

    spkpoplst        = spkdata['spkpoplst']
    spkindlst        = spkdata['spkindlst']
    spktlst          = spkdata['spktlst']
    num_cell_spks    = spkdata['num_cell_spks']
    pop_active_cells = spkdata['pop_active_cells']
    tmin             = spkdata['tmin']
    tmax             = spkdata['tmax']

    timeRange = [tmin, tmax]

    # create fig
    fig, ax1 = plt.subplots(figsize=figSize)

    if verbose:
        print('Plotting firing rate power spectral density (PSD) ...')

    psds = []
    # Plot separate line for each entry in include
    for iplot, subset in enumerate(spkpoplst):

        spkts = spktlst[iplot]

        histoCount, bin_edges = np.histogram(spkts, bins = np.arange(timeRange[0], timeRange[1], binSize))

        if smooth:
            hsignal = signal.savgol_filter(histoCount * (1000.0 / binSize) / (len(pop_active_cells[subset])),
                                           window_length=nperseg/2 + 1, polyorder=smooth) # smoothen and convert to firing rate
        else:
            hsignal = histoCount * (1000.0 / binSize) / (len(pop_active_cells[subset])) # convert to firing rate
            
        win = signal.get_window('hanning',nperseg)
        freqs, psd = signal.welch(hsignal, fs=Fs, nperseg=nperseg, noverlap=(nperseg // 2),
                                  scaling='density', window=win)
        
        psd = 10*np.log10(psd)
        peak_index  = np.where(psd == np.max(psd))[0]
        
        color = color_list[iplot%len(color_list)]

        if not overlay:
            label = str(subset)
            plt.subplot(len(spkpoplst),1,iplot+1)
            plt.title ('%s (peak: %.3g Hz)' % (label, freqs[peak_index]), fontsize=fontSize)

        plt.plot(freqs, psd, linewidth=lw, color=color)
        
        if iplot == 0:
            plt.ylabel('Power Spectral Density (dB/Hz)', fontsize=fontSize) # add yaxis in opposite side
        if iplot == len(spkpoplst)-1:
            plt.xlabel('Frequency (Hz)', fontsize=fontSize)
        plt.xlim([0, (Fs/2)-1])

        psds.append(psd)
        
    if len(spkpoplst) < 5:  # if apply tight_layout with many subplots it inverts the y-axis
        try:
            plt.tight_layout()
        except:
            pass

    # save figure
    if saveFig: 
        if isinstance(saveFig, basestring):
            filename = saveFig
        else:
            filename = namespace_id+'_'+'ratePSD.png'
        plt.savefig(filename)

    # show fig 
    if showFig:
        show_figure()

    return fig, psds



def plot_stimulus_rate (input_path, namespace_id, include, trajectory_id=None,
                        figSize = (8,8), fontSize = 14, saveFig = None, showFig = True,
                        verbose = False): 
    ''' 

        - input_path: file with stimulus data
        - namespace_id: attribute namespace for stimulus
        - include (['eachPop'|<population name>]): List of data series to include. 
            (default: ['eachPop'] - expands to the name of each population)
        - figSize ((width, height)): Size of figure (default: (8,8))
        - fontSize (integer): Size of text font (default: 14)
        - lw (integer): Line width for each spike (default: 3)
        - saveFig (None|True|'fileName'): File name where to save the figure;
            if set to True uses filename from simConfig (default: None)
        - showFig (True|False): Whether to show the figure or not (default: True)

    '''
    comm = MPI.COMM_WORLD

    fig, axes = plt.subplots(1, len(include), figsize=figSize)

    if trajectory_id is not None:
        trajectory = stimulus.read_trajectory (comm, input_path, trajectory_id, verbose=verbose)
        (_, _, _, t)  = trajectory
    else:
        t = None
        
    M = 0
    for iplot, population in enumerate(include):
        rate_lst = []
        if verbose:
            print 'Reading vector stimulus data from namespace %s for population %s...' % (namespace_id, population )
        for (gid, rate, _, _) in stimulus.read_stimulus(comm, input_path, namespace_id, population):
            if np.max(rate) > 0.:
                rate_lst.append(rate)

        M = max(M, len(rate_lst))
        N = len(rate_lst)
        rate_matrix = np.matrix(rate_lst)
        del(rate_lst)

        if verbose:
            print 'Plotting stimulus data for population %s...' % population 

        if t is None:
            extent=[0, len(rate), 0, N]
        else:
            extent=[t[0], t[-1], 0, N]
            
        if len(include) > 1:
            axes[iplot].set_title(population, fontsize=fontSize)
            axes[iplot].imshow(rate_matrix, origin='lower', aspect='auto', cmap=cm.coolwarm, extent=extent)
            axes[iplot].set_xlim([extent[0], extent[1]])
            axes[iplot].set_ylim(-1, N+1)
            
        else:
            axes.set_title(population, fontsize=fontSize)
            axes.imshow(rate_matrix, origin='lower', aspect='auto', cmap=cm.coolwarm, extent=extent)
            axes.set_xlim([extent[0], extent[1]])
            axes.set_ylim(-1, N+1)    
            

    axes.set_xlabel('Time (ms)', fontsize=fontSize)
    axes.set_ylabel('Input #', fontsize=fontSize)
    
    # save figure
    if saveFig: 
        if isinstance(saveFig, basestring):
            filename = saveFig
        else:
            filename = namespace_id+'_'+'ratemap.png'
        plt.savefig(filename)

    # show fig 
    if showFig:
        show_figure()


        
def plot_stimulus_spatial_rate_map (input_path, coords_path, stimulus_namespace, distances_namespace, include,
                                    normed = False, figSize = (8,8), fontSize = 14, saveFig = None, showFig = True,
                                    verbose = False): 
    ''' 

        - input_path: file with stimulus data
        - stimulus_namespace: attribute namespace for stimulus
        - distances_namespace: attribute namespace for longitudinal and transverse distances
        - include (['eachPop'|<population name>]): List of data series to include. 
            (default: ['eachPop'] - expands to the name of each population)
        - figSize ((width, height)): Size of figure (default: (8,8))
        - fontSize (integer): Size of text font (default: 14)
        - lw (integer): Line width for each spike (default: 3)
        - saveFig (None|True|'fileName'): File name where to save the figure;
            if set to True uses filename from simConfig (default: None)
        - showFig (True|False): Whether to show the figure or not (default: True)

    '''
    comm = MPI.COMM_WORLD

    fig, axes = plt.subplots(1, len(include), figsize=figSize)

    for iplot, population in enumerate(include):
        rate_sum_dict = {}
        if verbose:
            print 'Reading vector stimulus data for population %s...' % population 
        for (gid, rate, _, _) in stimulus.read_stimulus(comm, input_path, stimulus_namespace, population):
            rate_sum_dict[gid] = np.sum(rate)
        
        if verbose:
            print 'read rates (%i elements)' % len(rate_sum_dict.keys())

        distances = read_cell_attributes(coords_path, population, namespace=distances_namespace, comm=comm)
    
        soma_distances = { k: (v['U Distance'][0], v['V Distance'][0]) for (k,v) in distances }
        del distances
        
        if verbose:
            print 'read distances (%i elements)' % len(soma_distances.keys())

        distance_U = np.asarray([ soma_distances[gid][0] for gid in rate_sum_dict.keys() ])
        distance_V = np.asarray([ soma_distances[gid][1] for gid in rate_sum_dict.keys() ])
        rate_sums  = np.asarray([ rate_sum_dict[gid] for gid in rate_sum_dict.keys() ])

        x_min = np.min(distance_U)
        x_max = np.max(distance_U)
        y_min = np.min(distance_V)
        y_max = np.max(distance_V)

        (H, xedges, yedges) = np.histogram2d(distance_U, distance_V, bins=[250, 100], weights=rate_sums, normed=normed)
    
        if verbose:
            print 'Plotting stimulus spatial distribution...'

        X, Y = np.meshgrid(xedges, yedges)
        if (len(include) > 1):
            pcm = axes[iplot].pcolormesh(X, Y, H.T)

            axes[iplot].axis([x_min, x_max, y_min, y_max])
            axes[iplot].set_aspect('equal')
            
            axes[iplot].set_xlabel('Arc distance (septal - temporal) (um)', fontsize=fontSize)
            axes[iplot].set_ylabel('Arc distance (supra - infrapyramidal)  (um)', fontsize=fontSize)
            fig.colorbar(pcm, ax=axes[iplot], shrink=0.5, aspect=20)
            
        else:
            pcm = axes.pcolormesh(X, Y, H.T)

            axes.axis([x_min, x_max, y_min, y_max])
            axes.set_aspect('equal')
    
            axes.set_xlabel('Arc distance (septal - temporal) (um)', fontsize=fontSize)
            axes.set_ylabel('Arc distance (supra - infrapyramidal)  (um)', fontsize=fontSize)
            fig.colorbar(pcm, ax=axes, shrink=0.5, aspect=20)

    # save figure
    if saveFig: 
        if isinstance(saveFig, basestring):
            filename = saveFig
        else:
            filename = stimulus_namespace+' '+'spatial ratemap.png'
        plt.savefig(filename)

    # show fig 
    if showFig:
        show_figure()

        
        

        
## Plot spike auto-correlation
def plot_spike_histogram_autocorr (input_path, namespace_id, include = ['eachPop'], timeRange = None, timeVariable='t', binSize = 25, graphType = 'matrix', lag=1,
                                   maxCells = None, xlim = None, lw = 3, marker = '|', figSize = (15,8), fontSize = 14, saveFig = None, showFig = True, verbose = False): 
    ''' 
    Plot of spike histogram correlations. Returns the figure handle.

    input_path: file with spike data
    namespace_id: attribute namespace for spike events
    timeRange ([start:stop]): Time range of spikes shown; if None shows all (default: None)
    timeVariable: Name of variable containing spike times (default: 't')
    binSize (int): Size of bin in ms to use for spike count and rate computations (default: 5)
    lw (integer): Line width for each spike (default: 3)
    marker (char): Marker for each spike (default: '|')
    fontSize (integer): Size of text font (default: 14)
    figSize ((width, height)): Size of figure (default: (15,8))
    saveFig (None|True|'fileName'): File name where to save the figure (default: None)
    showFig (True|False): Whether to show the figure or not (default: True)
    '''

    comm = MPI.COMM_WORLD

    (population_ranges, N) = read_population_ranges(comm, input_path)
    population_names  = read_population_names(comm, input_path)

    pop_num_cells = {}
    for k in population_names:
        pop_num_cells[k] = population_ranges[k][1]

    # Replace 'eachPop' with list of populations
    if 'eachPop' in include: 
        include.remove('eachPop')
        for pop in population_names:
            include.append(pop)

    spkdata = spikedata.read_spike_events (comm, input_path, include, namespace_id, timeVariable=timeVariable,
                                           timeRange=timeRange, verbose=verbose)

    spkpoplst        = spkdata['spkpoplst']
    spkindlst        = spkdata['spkindlst']
    spktlst          = spkdata['spktlst']
    num_cell_spks    = spkdata['num_cell_spks']
    pop_active_cells = spkdata['pop_active_cells']
    tmin             = spkdata['tmin']
    tmax             = spkdata['tmax']
    
    if verbose:
        print('Calculating spike correlations...')

    corr_dict = spikedata.histogram_autocorrelation(spkdata, binSize=binSize, maxElems=maxCells, lag=lag)
        
    # Plot spikes
    fig, axes = plt.subplots(len(spkpoplst), 1, figsize=figSize, sharex=True)

    if verbose:
        print('Creating autocorrelation plots...')

    X_max = None
    X_min = None
    for (iplot, subset) in enumerate(spkpoplst):

        pop_corr = corr_dict[subset]
        
        if len(spkpoplst) > 1:
            axes[iplot].set_title (str(subset), fontsize=fontSize)
        else:
            axes.set_title (str(subset), fontsize=fontSize)

        if graphType == 'matrix':
            im = axes[iplot].imshow(pop_corr, origin='lower', aspect='auto', interpolation='none', cmap=cm.jet)
            cbar = plt.colorbar(im)
            cbar.ax.set_ylabel('Correlation Coefficient', fontsize=fontSize)
        elif graphType == 'histogram':
            histoCount, bin_edges = np.histogram(pop_corr, bins = 100)
            corrBinSize = bin_edges[1] - bin_edges[0]
            histoX = bin_edges[:-1]+corrBinSize/2
            color = color_list[iplot%len(color_list)]
            if len(spkpoplst) > 1:
                b = axes[iplot].bar(histoX, histoCount, width = corrBinSize, color = color)
            else:
                b = axes.bar(histoX, histoCount, width = corrBinSize, color = color)
            if X_max is None:
                X_max = bin_edges[-1]
            else:
                X_max = max(X_max, bin_edges[-1])
            if X_min is None:
                X_min = bin_edges[0]
            else:
                X_min = max(X_min, bin_edges[0])
                
            if len(spkpoplst) > 1:
                axes[iplot].set_xlim([X_min, X_max])
            else:
                axes.set_xlim([X_min, X_max])
        else:
            im = axes[iplot].imshow(pop_corr, origin='lower', aspect='auto', interpolation='none', cmap=cm.jet)
            cbar = plt.colorbar(im)
            cbar.ax.set_ylabel('Correlation Coefficient', fontsize=fontSize)

        if graphType == 'matrix':
            if iplot == 0:
                axes[iplot].ylabel('Relative Cell Index', fontsize=fontSize)
            if iplot == len(spkpoplst)-1:
                axes[iplot].xlabel('Relative Cell Index', fontsize=fontSize)

                
    # show fig 
    if showFig:
        show_figure()
    
    return fig


## Plot spike cross-correlation
def plot_spike_histogram_corr (input_path, namespace_id, include = ['eachPop'], timeRange = None, timeVariable='t', binSize = 25, graphType = 'matrix',
                               maxCells = None, lw = 3, marker = '|', figSize = (15,8), fontSize = 14, saveFig = None, showFig = True, verbose = False): 
    ''' 
    Plot of spike histogram correlations. Returns the figure handle.

    input_path: file with spike data
    namespace_id: attribute namespace for spike events
    timeRange ([start:stop]): Time range of spikes shown; if None shows all (default: None)
    timeVariable: Name of variable containing spike times (default: 't')
    binSize (int): Size of bin in ms to use for spike count and rate computations (default: 5)
    lw (integer): Line width for each spike (default: 3)
    marker (char): Marker for each spike (default: '|')
    fontSize (integer): Size of text font (default: 14)
    figSize ((width, height)): Size of figure (default: (15,8))
    saveFig (None|True|'fileName'): File name where to save the figure (default: None)
    showFig (True|False): Whether to show the figure or not (default: True)
    '''

    comm = MPI.COMM_WORLD

    (population_ranges, N) = read_population_ranges(comm, input_path)
    population_names  = read_population_names(comm, input_path)

    pop_num_cells = {}
    for k in population_names:
        pop_num_cells[k] = population_ranges[k][1]

    # Replace 'eachPop' with list of populations
    if 'eachPop' in include: 
        include.remove('eachPop')
        for pop in population_names:
            include.append(pop)

    spkdata = spikedata.read_spike_events (comm, input_path, include, namespace_id, timeVariable=timeVariable,
                                           timeRange=timeRange, verbose=verbose)

    spkpoplst        = spkdata['spkpoplst']
    spkindlst        = spkdata['spkindlst']
    spktlst          = spkdata['spktlst']
    num_cell_spks    = spkdata['num_cell_spks']
    pop_active_cells = spkdata['pop_active_cells']
    tmin             = spkdata['tmin']
    tmax             = spkdata['tmax']
    
    if verbose:
        print('Calculating spike correlations...')

    corr_dict = spikedata.histogram_correlation(spkdata, binSize=binSize, maxElems=maxCells)
        
    # Plot spikes
    fig, axes = plt.subplots(len(spkpoplst), 1, figsize=figSize, sharex=True)

    if verbose:
        print('Creating correlation plots...')

    X_max = None
    X_min = None
    for (iplot, subset) in enumerate(spkpoplst):

        pop_corr = corr_dict[subset]

        if len(spkpoplst) > 1:
            axes[iplot].set_title (str(subset), fontsize=fontSize)
        else:
            axes.set_title (str(subset), fontsize=fontSize)
            
        if graphType == 'matrix':
            im = axes[iplot].imshow(pop_corr, origin='lower', aspect='auto', interpolation='none', cmap=cm.jet)
            cbar = plt.colorbar(im)
            cbar.ax.set_ylabel('Correlation Coefficient', fontsize=fontSize)
        elif graphType == 'histogram':
            np.fill_diagonal(pop_corr, 0.)
            mean_corr = np.apply_along_axis(lambda y: np.mean(y), 1, pop_corr)
            histoCount, bin_edges = np.histogram(mean_corr, bins = 100)
            corrBinSize = bin_edges[1] - bin_edges[0]
            histoX = bin_edges[:-1]+corrBinSize/2
            color = color_list[iplot%len(color_list)]
            if len(spkpoplst) > 1:
                b = axes[iplot].bar(histoX, histoCount, width = corrBinSize, color = color)
            else:
                b = axes.bar(histoX, histoCount, width = corrBinSize, color = color)
            if X_max is None:
                X_max = bin_edges[-1]
            else:
                X_max = max(X_max, bin_edges[-1])
            if X_min is None:
                X_min = bin_edges[0]
            else:
                X_min = max(X_min, bin_edges[0])

                
            if len(spkpoplst) > 1:
                axes[iplot].set_xlim([-0.5, 0.5])
            else:
                axes.set_xlim([-0.5, 0.5])
        else:
            im = axes[iplot].imshow(pop_corr, origin='lower', aspect='auto', interpolation='none', cmap=cm.jet)
            cbar = plt.colorbar(im)
            cbar.ax.set_ylabel('Correlation Coefficient', fontsize=fontSize)

        if graphType == 'matrix':
            if iplot == 0: 
                axes[iplot].ylabel('Relative Cell Index', fontsize=fontSize)
            if iplot == len(spkpoplst)-1:
                axes[iplot].xlabel('Relative Cell Index', fontsize=fontSize)

                
    # show fig 
    if showFig:
        show_figure()
    
    return fig
