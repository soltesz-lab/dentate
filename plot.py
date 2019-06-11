import numbers, os, copy
from collections import defaultdict
from scipy import interpolate, signal
import numpy as np
from mpi4py import MPI
import h5py
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.colors import BoundaryNorm
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

import dentate.statedata as statedata
from dentate.cells import default_ordered_sec_types, get_distance_to_node
from dentate.env import Env
from dentate.synapses import get_syn_filter_dict, get_syn_mech_param
from dentate.utils import get_module_logger, Struct, add_bins, update_bins, finalize_bins
from dentate.utils import make_geometric_graph, viewitems, zip_longest, old_div
from neuroh5.io import NeuroH5ProjectionGen, bcast_cell_attributes, read_cell_attributes, read_population_names, read_population_ranges, read_projection_names, read_tree_selection

try:
    import dentate.spikedata as spikedata
except ImportError as e:
    print(('dentate.plot: problem importing module required by dentate.spikedata:', e))
try:
    import dentate.stimulus as stimulus
except ImportError as e:
    print(('dentate.plot: problem importing module required by dentate.stimulus:', e))
try:
    from dentate.geometry import DG_volume, measure_distance_extents
except ImportError as e:
    print(('dentate.plot: problem importing module required by dentate.geometry:', e))

# This logger will inherit its settings from the root logger, created in dentate.env
logger = get_module_logger(__name__)

# Default figure configuration
default_fig_options = Struct(figFormat='png', lw=3, figSize=(15,8), fontSize=14, saveFig=None, showFig=True,
                             colormap=cm.jet, saveFigDir=None)

color_list = ["#009BFF", "#E85EBE", "#00FF00", "#0000FF", "#FF0000", "#01FFFE", "#FFA6FE", 
              "#FFDB66", "#006401", "#010067", "#95003A", "#007DB5", "#FF00F6", "#FFEEE8", "#774D00",
              "#90FB92", "#0076FF", "#D5FF00", "#FF937E", "#6A826C", "#FF029D", "#FE8900", "#7A4782",
              "#7E2DD2", "#85A900", "#FF0056", "#A42400", "#00AE7E", "#683D3B", "#BDC6FF", "#263400",
              "#BDD393", "#00B917", "#9E008E", "#001544", "#C28C9F", "#FF74A3", "#01D0FF", "#004754",
              "#E56FFE", "#788231", "#0E4CA1", "#91D0CB", "#BE9970", "#968AE8", "#BB8800", "#43002C",
              "#DEFF74", "#00FFC6", "#FFE502", "#620E00", "#008F9C", "#98FF52", "#7544B1", "#B500FF",
              "#00FF78", "#FF6E41", "#005F39", "#6B6882", "#5FAD4E", "#A75740", "#A5FFD2", "#FFB167"]

rainbow_color_list = ["#9400D3", "#4B0082", "#00FF00", "#FFFF00", "#FF7F00", "#FF0000"]

raster_color_list = ['#8dd3c7', '#ffed6f', '#bebada', '#fb8072', '#80b1d3', '#fdb462',
                    '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd', '#ccebc5']


def hex2rgb(hexcode):
    return tuple([ float(b)/255.0 for b in map(ord,hexcode[1:].decode('hex')) ])

mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.size'] = 14.
mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['text.usetex'] = False


def show_figure():
    try:
        plt.show(block=False)
    except:
        plt.show()


def save_figure(file_name_prefix, fig=None, **kwargs):
    """

    :param file_name_prefix:
    :param fig: :class:'plt.Figure'
    :param kwargs: dict
    """
    fig_options = copy.copy(default_fig_options)
    fig_options.update(kwargs)
    fig_file_path = '%s.%s' % (file_name_prefix, fig_options.figFormat)
    if fig_options.saveFigDir is not None:
        fig_file_path = '%s/%s' % (fig_options.saveFigDir, fig_file_path)
    if fig is not None:
        fig.savefig(fig_file_path)
    else:
        plt.savefig(fig_file_path)


def plot_graph(x, y, z, start_idx, end_idx, edge_scalars=None, edge_color=None, **kwargs):
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
    if edge_color is not None:
        kwargs['color'] = edge_color
    mlab.points3d(x[0],y[0],z[0],
                  mode='cone',
                  scale_factor=50,
                  **kwargs)
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



def plot_vertex_metrics(env, connectivity_path, coords_path, vertex_metrics_namespace, distances_namespace, destination, sources, bin_size = 50., metric='Indegree', normed = False, graph_type = 'histogram2d', **kwargs):
    """
    Plot vertex metric with respect to septo-temporal position (longitudinal and transverse arc distances to reference points).

    :param connectivity_path:
    :param coords_path:
    :param distances_namespace: 
    :param destination_pop: 

    """

    fig_options = copy.copy(default_fig_options)
    fig_options.update(kwargs)

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

    distance_x_min = np.min(distance_U)
    distance_x_max = np.max(distance_U)
    distance_y_min = np.min(distance_V)
    distance_y_max = np.max(distance_V)

    ((x_min, x_max), (y_min, y_max)) = measure_distance_extents(env)

    dx = int(old_div((distance_x_max - distance_x_min), bin_size))
    dy = int(old_div((distance_y_max - distance_y_min), bin_size))

    for source, degrees in viewitems(degrees_dict):
        
        fig = plt.figure(figsize=fig_options.figSize)
        ax = plt.gca()
        ax.axis([x_min, x_max, y_min, y_max])

        if graph_type == 'histogram1d':
            bins_U = np.linspace(x_min, x_max, dx)
            bins_V = np.linspace(y_min, y_max, dy)
            hist_vals_U, bin_edges_U = np.histogram(distance_U, bins = bins_U, weights=degrees)
            hist_vals_V, bin_edges_V = np.histogram(distance_V, bins = bins_V, weights=degrees)
            gs  = gridspec.GridSpec(3, 1, height_ratios=[2,1,2])
            ax1 = plt.subplot(gs[0])
            ax1.plot (bin_edges_U[:-1], hist_vals_U)
            ax1.set_title('%s distribution for destination: %s source: %s' % (metric, destination, source), fontsize=fig_options.fontSize)
            ax2 = plt.subplot(gs[2])
            ax2.plot (bin_edges_V[:-1], hist_vals_V)
            ax1.set_xlabel('Arc distance (septal - temporal) (um)', fontsize=fig_options.fontSize)
            ax2.set_xlabel('Arc distance (supra - infrapyramidal)  (um)', fontsize=fig_options.fontSize)
            ax1.set_ylabel('Number of edges', fontsize=fig_options.fontSize)
            ax2.set_ylabel('Number of edges', fontsize=fig_options.fontSize)
            ax1.tick_params(labelsize=fig_options.fontSize)
            ax2.tick_params(labelsize=fig_options.fontSize)
            plt.subplot(gs[1]).remove()
        elif graph_type == 'histogram2d':
            if normed:
                (H1, xedges, yedges) = np.histogram2d(distance_U, distance_V, bins=[dx, dy], weights=degrees, normed=normed)
                (H2, xedges, yedges) = np.histogram2d(distance_U, distance_V, bins=[dx, dy])
                H = np.zeros(H1.shape)
                nz = np.where(H2 > 0.0)
                H[nz] = np.divide(H1[nz], H2[nz])
                H[nz] = np.divide(H[nz], np.max(H[nz]))
            else:
                (H, xedges, yedges) = np.histogram2d(distance_U, distance_V, bins=[dx, dy], weights=degrees)
                 
            X, Y = np.meshgrid(xedges, yedges)
            pcm = ax.pcolormesh(X, Y, H.T)
            cb = fig.colorbar(pcm, ax=ax, shrink=0.5, aspect=20)
            cb.ax.tick_params(labelsize=fig_options.fontSize)
        else:
            raise ValueError('Unknown graph type %s' % graph_type)

        ax.tick_params(labelsize=fig_options.fontSize)
        ax.set_xlabel('Arc distance (septal - temporal) (um)', fontsize=fig_options.fontSize)
        ax.set_ylabel('Arc distance (supra - infrapyramidal)  (um)', fontsize=fig_options.fontSize)
        ax.set_title('%s distribution for destination: %s source: %s' % (metric, destination, source), fontsize=fig_options.fontSize)
        ax.set_aspect('equal')
    
        if fig_options.saveFig:
            if isinstance(fig_options.saveFig, str):
                filename = fig_options.saveFig
            else:
                filename = '%s to %s %s %s.%s' % (source, destination, metric, graph_type, fig_options.figFormat)
                plt.savefig(filename)

        if fig_options.showFig:
            show_figure()
    



def plot_vertex_dist(connectivity_path, coords_path, distances_namespace, destination, sources, 
                        bin_size=20.0, cache_size=100, comm=None, **kwargs):
    """
    Plot vertex distribution with respect to septo-temporal distance

    :param connectivity_path:
    :param coords_path:
    :param distances_namespace: 
    :param destination: 
    :param source: 

    """
    fig_options = copy.copy(default_fig_options)
    fig_options.update(kwargs)

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
            raise Exception('destination %s: destination_gid %i not matched across multiple projection generators: %s' %
                            (destination, destination_gid, [prj_gen_elt[0] for prj_gen_elt in prj_gen_tuple]))

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
                    
    if rank == 0:
        for source in sources:
            dist_hist_vals, dist_bin_edges = finalize_bins(dist_bins[source], bin_size)
            dist_u_hist_vals, dist_u_bin_edges = finalize_bins(dist_u_bins[source], bin_size)
            dist_v_hist_vals, dist_v_bin_edges = finalize_bins(dist_v_bins[source], bin_size)

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10,6))
            fig.suptitle('Distribution of connection distances for projection %s -> %s' % (source, destination), fontsize=fig_options.fontSize)

            ax1.bar(dist_bin_edges, dist_hist_vals, width=bin_size)
            ax1.set_xlabel('Total distance (um)', fontsize=fig_options.fontSize)
            ax1.set_ylabel('Number of connections', fontsize=fig_options.fontSize)
        
            ax2.bar(dist_u_bin_edges, dist_u_hist_vals, width=bin_size)
            ax2.set_xlabel('Septal - temporal (um)', fontsize=fig_options.fontSize)
            
            ax3.bar(dist_v_bin_edges, dist_v_hist_vals, width=bin_size)
            ax3.set_xlabel('Supra - infrapyramidal (um)', fontsize=fig_options.fontSize)

            ax1.tick_params(labelsize=fig_options.fontSize)
            ax2.tick_params(labelsize=fig_options.fontSize)
            ax3.tick_params(labelsize=fig_options.fontSize)
            
            if fig_options.saveFig:
                if isinstance(fig_options.saveFig, str):
                    filename = fig_options.saveFig
                else:
                    filename = 'Connection distance %s to %s.%s' % (source, destination, fig_options.figFormat)
                    plt.savefig(filename)
                    
            if fig_options.showFig:
                show_figure()
                
    comm.barrier()

    

def plot_single_vertex_dist(env, connectivity_path, coords_path, distances_namespace, target_gid,
                            destination, source, extent_type='local', direction='in',
                            bin_size=20.0, normed=False, comm=None, **kwargs):
    """
    Plot vertex distribution with respect to septo-temporal distance for a single postsynaptic cell

    :param connectivity_path:
    :param coords_path:
    :param distances_namespace: 
    :param destination_gid: 
    :param destination: 
    :param source: 

    """
    fig_options = copy.copy(default_fig_options)
    fig_options.update(kwargs)
    
    (population_ranges, _) = read_population_ranges(coords_path)

    destination_start = population_ranges[destination][0]
    destination_count = population_ranges[destination][1]

    if comm is None:
        comm = MPI.COMM_WORLD

    rank = comm.Get_rank()

    source_soma_distances = bcast_cell_attributes(coords_path, source, namespace=distances_namespace, comm=comm, root=0)
    destination_soma_distances = bcast_cell_attributes(coords_path, destination, namespace=distances_namespace, comm=comm, root=0)

    ((total_x_min,total_x_max),(total_y_min,total_y_max)) = measure_distance_extents(env)

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
                
    g = NeuroH5ProjectionGen (connectivity_path, source, destination, comm=comm, cache_size=20)

    dist_bins = {}

    if direction == 'in':
        for (destination_gid,rest) in g:
            if destination_gid == target_gid:
                (source_indexes, attr_dict) = rest
                for source_gid in source_indexes:
                    dist_u = source_soma_distance_U[source_gid]
                    dist_v = source_soma_distance_V[source_gid]
                    update_bins(dist_bins, bin_size, dist_u, dist_v)
                break
    elif direction == 'out':
        for (destination_gid,rest) in g:
            if rest is not None:
                (source_indexes, attr_dict) = rest
                for source_gid in source_indexes:
                    if source_gid == target_gid:
                        dist_u = destination_soma_distance_U[destination_gid]
                        dist_v = destination_soma_distance_V[destination_gid]
                        update_bins(dist_bins, bin_size, dist_u, dist_v)
    else:
        raise RuntimeError('Unknown direction type %s' % str(direction))

    add_bins_op = MPI.Op.Create(add_bins, commute=True)
    dist_bins = comm.reduce(dist_bins, op=add_bins_op)

    if rank == 0:

        dist_hist_vals, dist_u_bin_edges, dist_v_bin_edges = finalize_bins(dist_bins, bin_size)

        dist_x_min = dist_u_bin_edges[0]
        dist_x_max = dist_u_bin_edges[-1]
        dist_y_min = dist_v_bin_edges[0]
        dist_y_max = dist_v_bin_edges[-1]
        
        if extent_type == 'local':
            x_min = dist_x_min
            x_max = dist_x_max
            y_min = dist_y_min
            y_max = dist_y_max
        elif extent_type == 'global':
            x_min = total_x_min
            x_max = total_x_max
            y_min = total_y_min
            y_max = total_y_max
        else:
            raise RuntimeError('Unknown extent type %s' % str(extent_type))

        X, Y = np.meshgrid(dist_u_bin_edges, dist_v_bin_edges)

        fig = plt.figure(figsize=fig_options.figSize)

        ax = plt.gca()
        ax.axis([x_min, x_max, y_min, y_max])

        if direction == 'in':
            ax.plot(destination_soma_distance_U[target_gid], \
                    destination_soma_distance_V[target_gid], \
                    'r+', markersize=12, mew=3)
        elif direction == 'out':
            ax.plot(source_soma_distance_U[target_gid], \
                    source_soma_distance_V[target_gid], \
                    'r+', markersize=12, mew=3)
        else:
            raise RuntimeError('Unknown direction type %s' % str(direction))

        H = np.array(dist_hist_vals.todense())
        if normed:
            H = np.divide(H.astype(float), float(np.max(H)))
        pcm_boundaries = np.arange(0, np.max(H), .1)
        cmap_pls = plt.cm.get_cmap('PuBu',len(pcm_boundaries))
        pcm_colors = list(cmap_pls(np.arange(len(pcm_boundaries))))
        pcm_cmap = mpl.colors.ListedColormap(pcm_colors[:-1], "")
        pcm_cmap.set_under(pcm_colors[0], alpha=0.0)
        
        pcm = ax.pcolormesh(X, Y, H.T, cmap=pcm_cmap)

#        pcm = ax.pcolormesh(X, Y, H.T, cmap=pcm_cmap,
#                            norm = mpl.colors.BoundaryNorm(pcm_boundaries, ncolors=len(pcm_boundaries)-1,
#                                                            clip=False))

        clb_label = 'Normalized number of connections' if normed else 'Number of connections'
        clb = fig.colorbar(pcm, ax=ax, shrink=0.5, label=clb_label)
        clb.ax.tick_params(labelsize=fig_options.fontSize)
    
        ax.set_aspect('equal')
        ax.set_facecolor(pcm_colors[0])
        ax.tick_params(labelsize=fig_options.fontSize)
        ax.set_xlabel('Arc distance (septal - temporal) (um)', fontsize=fig_options.fontSize)
        ax.set_ylabel('Arc distance (supra - infrapyramidal)  (um)', fontsize=fig_options.fontSize)
        ax.set_title('Connectivity distribution (%s) of %s to %s for gid: %i' % (direction, source, destination, target_gid), \
                    fontsize=fig_options.fontSize)
        

        if fig_options.showFig:
            show_figure()

        if fig_options.saveFig:
            if isinstance(fig_options.saveFig, str):
                filename = fig_options.saveFig
            else:
                filename = 'Connection distance %s %s to %s gid %i.%s' % (direction, source, destination, target_gid, fig_options.figFormat)
                plt.savefig(filename)
    
    

def plot_tree_metrics(env, forest_path, coords_path, population, metric_namespace='Tree Measurements', distances_namespace='Arc Distances', metric='dendrite_length', metric_index=0, percentile=None, **kwargs):
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

    fig_options = copy.copy(default_fig_options)
    fig_options.update(kwargs)
        
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
    print(('min: %f max: %f mean: %f' % (tree_metric_stats)))

    if percentile is not None:
        percentile_value = np.percentile(tree_metrics_array, percentile)
        print('%f percentile value: %f' % (percentile, percentile_value))
        sample = np.where(tree_metrics_array >= percentile_value)
        tree_metrics_array = tree_metrics_array[sample]
        sorted_keys = np.asarray(sorted_keys)[sample]
        
    
    distance_U_array = np.array([distance_U[k] for k in sorted_keys])
    distance_V_array = np.array([distance_V[k] for k in sorted_keys])

    ((x_min, x_max), (y_min, y_max)) = measure_distance_extents(env)

    (H, xedges, yedges) = np.histogram2d(distance_U_array, distance_V_array, \
                                         bins=[dx, dy], weights=tree_metrics_array)


    ax.axis([x_min, x_max, y_min, y_max])

    X, Y = np.meshgrid(xedges, yedges)
    pcm = ax.pcolormesh(X, Y, H.T)
    
    ax.set_xlabel('Arc distance (septal - temporal) (um)', fontsize=fig_options.fontSize)
    ax.set_ylabel('Arc distance (supra - infrapyramidal)  (um)', fontsize=fig_options.fontSize)
    ax.set_title('%s distribution for population: %s' % (metric, population), fontsize=fig_options.fontSize)
    ax.set_aspect('equal')
    fig.colorbar(pcm, ax=ax, shrink=0.5, aspect=20)
    
    if fig_options.saveFig:
        if isinstance(fig_options.saveFig, str):
            filename = fig_options.saveFig
        else:
            filename = population+' %s.%s' % (metric, fig_options.figFormat)
            plt.savefig(filename)

    if fig_options.showFig:
        show_figure()
    
    return ax


def plot_positions(env, label, distances, bin_size=50., graph_type ='kde', **kwargs):
    """
    Plot septo-temporal position (longitudinal and transverse arc distances).

    :param label: 
    :param distances: 

    """
        
    fig_options = copy.copy(default_fig_options)
    fig_options.update(kwargs)

    fig = plt.figure(1, figsize=plt.figaspect(1.) * 2.)
    ax = plt.gca()

    distance_U = {}
    distance_V = {}
    for k,v in distances:
        distance_U[k] = v['U Distance'][0]
        distance_V[k] = v['V Distance'][0]
    
    distance_U_array = np.asarray([distance_U[k] for k in sorted(distance_U.keys())])
    distance_V_array = np.asarray([distance_V[k] for k in sorted(distance_V.keys())])

    distance_x_min = np.min(distance_U_array)
    distance_x_max = np.max(distance_U_array)
    distance_y_min = np.min(distance_V_array)
    distance_y_max = np.max(distance_V_array)

    ((x_min, x_max), (y_min, y_max)) = measure_distance_extents(env)
    ax.axis([x_min, x_max, y_min, y_max])

    dx = int(old_div((distance_x_max - distance_x_min), bin_size))
    dy = int(old_div((distance_y_max - distance_y_min), bin_size))
    if graph_type == 'histogram1d':
        bins_U = np.linspace(x_min, x_max, dx)
        bins_V = np.linspace(y_min, y_max, dy)
        hist_vals_U, bin_edges_U = np.histogram(distance_U_array, bins = bins_U)
        hist_vals_V, bin_edges_V = np.histogram(distance_V_array, bins = bins_V)
        gs  = gridspec.GridSpec(2, 1, height_ratios=[2,1])
        ax1 = plt.subplot(gs[0])
        ax1.bar (bin_edges_U[:-1], hist_vals_U, width=dx)
        ax1.set_title('Position distribution for %s' % (label), fontsize=fig_options.fontSize)
        ax2 = plt.subplot(gs[1])
        ax2.bar (bin_edges_V[:-1], hist_vals_V, width=dy)
        ax1.set_xlabel('Arc distance (septal - temporal) (um)', fontsize=fig_options.fontSize)
        ax2.set_xlabel('Arc distance (supra - infrapyramidal)  (um)', fontsize=fig_options.fontSize)
        ax1.set_ylabel('Number of cells', fontsize=fig_options.fontSize)
        ax2.set_ylabel('Number of cells', fontsize=fig_options.fontSize)
    elif graph_type == 'histogram2d':
        (H, xedges, yedges) = np.histogram2d(distance_U_array, distance_V_array, bins=[dx, dy])
        X, Y = np.meshgrid(xedges, yedges)
        Hint = H[:-1, :-1]
        levels = MaxNLocator(nbins=150).tick_values(Hint.min(), Hint.max())
        cmap = plt.get_cmap('jet')
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        p = ax.contourf(X[:-1,:-1] + old_div(bin_size,2), Y[:-1,:-1]+old_div(bin_size,2), H.T, levels=levels, cmap=cmap)
        fig.colorbar(p, ax=ax, shrink=0.5, aspect=20)
    elif graph_type == 'kde':
        X, Y, Z    = kde_scipy(distance_U_array, distance_V_array, bin_size)
        p    = ax.imshow(Z, origin='lower', aspect='auto', extent=[x_min, x_max, y_min, y_max])
        fig.colorbar(p, ax=ax, shrink=0.5, aspect=20)
    else:
        raise ValueError('Unknown graph type %s' % graph_type)
    ax.set_xlabel('Arc distance (septal - temporal) (um)', fontsize=fig_options.fontSize)
    ax.set_ylabel('Arc distance (supra - infrapyramidal)  (um)', fontsize=fig_options.fontSize)
    ax.set_title('Position distribution for %s' % (label), fontsize=fig_options.fontSize)
    ax.set_aspect('equal')
    
    if fig_options.saveFig:
        if isinstance(fig_options.saveFig, str):
            filename = fig_options.saveFig
        else:
            filename = '%s Positions.%s' % (label, fig_options.figFormat)
            plt.savefig(filename)

    if fig_options.showFig:
        show_figure()
    
    return ax


def plot_coordinates(coords_path, population, namespace, index = 0, graph_type = 'scatter', bin_size = 0.01, xyz = False, **kwargs):
    """
    Plot coordinates

    :param coords_path:
    :param namespace: 
    :param population: 

    """
    fig_options = copy.copy(default_fig_options)
    fig_options.update(kwargs)
        
    soma_coords = read_cell_attributes(coords_path, population, namespace=namespace)
    
        
    fig = plt.figure(1, figsize=plt.figaspect(1.) * 2.)
    ax = plt.gca()

    coord_U = {}
    coord_V = {}
    if xyz:
        for k,v in soma_coords:
            coord_U[k] = v['X Coordinate'][index]
            coord_V[k] = v['Y Coordinate'][index]
    else:
        for k,v in soma_coords:
            coord_U[k] = v['U Coordinate'][index]
            coord_V[k] = v['V Coordinate'][index]
    
    coord_U_array = np.asarray([coord_U[k] for k in sorted(coord_U.keys())])
    coord_V_array = np.asarray([coord_V[k] for k in sorted(coord_V.keys())])

    x_min = np.min(coord_U_array)
    x_max = np.max(coord_U_array)
    y_min = np.min(coord_V_array)
    y_max = np.max(coord_V_array)

    dx = int(old_div((x_max - x_min), bin_size))
    dy = int(old_div((y_max - y_min), bin_size))

    if graph_type == 'scatter':
        ax.scatter(coord_U_array, coord_V_array, alpha=0.1, linewidth=0)
        ax.axis([x_min, x_max, y_min, y_max])
    elif graph_type == 'histogram2d':
        (H, xedges, yedges) = np.histogram2d(coord_U_array, coord_V_array, bins=[dx, dy])
        X, Y = np.meshgrid(xedges, yedges)
        Hint = H[:-1, :-1]
        levels = MaxNLocator(nbins=25).tick_values(Hint.min(), Hint.max())
        cmap = plt.get_cmap('jet')
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        p = ax.contourf(X[:-1,:-1] + old_div(bin_size,2), Y[:-1,:-1]+old_div(bin_size,2), H.T, levels=levels, cmap=cmap)
        fig.colorbar(p, ax=ax, shrink=0.5, aspect=20)
    else:
        raise ValueError('Unknown graph type %s' % graph_type)

    if xyz:
        ax.set_xlabel('X coordinate (um)', fontsize=fig_options.fontSize)
        ax.set_ylabel('Y coordinate (um)', fontsize=fig_options.fontSize)
    else:
        ax.set_xlabel('U coordinate (septal - temporal)', fontsize=fig_options.fontSize)
        ax.set_ylabel('V coordinate (supra - infrapyramidal)', fontsize=fig_options.fontSize)
        
    ax.set_title('Coordinate distribution for population: %s' % (population), fontsize=fig_options.fontSize)
    
    if fig_options.saveFig:
        if isinstance(fig_options.saveFig, str):
            filename = fig_options.saveFig
        else:
            filename = population+' Coordinates.%s' % fig_options.figFormat
            plt.savefig(filename)

    if fig_options.showFig:
        show_figure()
    
    return ax


def plot_projected_coordinates(coords_path, population, namespace, index = 0, graph_type = 'scatter', bin_size = 10.0, project = 3.1, rotate = None, **kwargs):
    """
    Plot coordinates

    :param coords_path:
    :param namespace: 
    :param population: 

    """
    
    fig_options = copy.copy(default_fig_options)
    fig_options.update(kwargs)

    soma_coords = read_cell_attributes(coords_path, population, namespace=namespace)
    
        
    fig = plt.figure(1, figsize=plt.figaspect(1.) * 2.)
    ax = plt.gca()

    coord_X = {}
    coord_Y = {}
    for k,v in soma_coords:
        ucoord = v['U Coordinate'][index]
        vcoord = v['V Coordinate'][index]
        xyz = DG_volume (ucoord, vcoord, project, rotate=rotate)
        coord_X[k] = xyz[0,0]
        coord_Y[k] = xyz[0,1]
    
    coord_X_array = np.asarray([coord_X[k] for k in sorted(coord_X.keys())])
    coord_Y_array = np.asarray([coord_Y[k] for k in sorted(coord_Y.keys())])

    x_min = np.min(coord_X_array)
    x_max = np.max(coord_X_array)
    y_min = np.min(coord_Y_array)
    y_max = np.max(coord_Y_array)

    dx = int(old_div((x_max - x_min), bin_size))
    dy = int(old_div((y_max - y_min), bin_size))

    if graph_type == 'scatter':
        ax.scatter(coord_X_array, coord_Y_array, alpha=0.1, linewidth=0)
        ax.axis([x_min, x_max, y_min, y_max])
    elif graph_type == 'histogram2d':
        (H, xedges, yedges) = np.histogram2d(coord_X_array, coord_Y_array, bins=[dx, dy])
        X, Y = np.meshgrid(xedges, yedges)
        Hint = H[:-1, :-1]
        levels = MaxNLocator(nbins=25).tick_values(Hint.min(), Hint.max())
        cmap = plt.get_cmap('jet')
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        p = ax.contourf(X[:-1,:-1] + old_div(bin_size,2), Y[:-1,:-1]+old_div(bin_size,2), H.T, levels=levels, cmap=cmap)
        fig.colorbar(p, ax=ax, shrink=0.5, aspect=20)
    else:
        raise ValueError('Unknown graph type %s' % graph_type)

    ax.set_xlabel('X coordinate (um)', fontsize=fig_options.fontSize)
    ax.set_ylabel('Y coordinate (um)', fontsize=fig_options.fontSize)
        
    ax.set_title('Coordinate distribution for population: %s' % (population), fontsize=fig_options.fontSize)
    
    if fig_options.saveFig:
        if isinstance(fig_options.saveFig, str):
            filename = fig_options.saveFig
        else:
            filename = population+' Coordinates.%s' % fig_options.figFormat
            plt.savefig(filename)

    if fig_options.showFig:
        show_figure()
    
    return ax


def plot_reindex_positions(env, coords_path, population, distances_namespace='Arc Distances',
                           reindex_namespace='Tree Reindex', reindex_attribute='New Cell Index', 
                           **kwargs):
    """
    Plot septo-temporal position (longitudinal and transverse arc distances).

    :param coords_path:
    :param distances_namespace: 
    :param population: 

    """

    dx = 50
    dy = 50

    fig_options = copy.copy(default_fig_options)
    fig_options.update(kwargs)
        
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
        if k in cell_reindex_dict:
            distance_U[k] = v['U Distance'][0]
            distance_V[k] = v['V Distance'][0]
        
        
    distance_U_array = np.asarray([distance_U[k] for k in sorted(distance_U.keys())])
    distance_V_array = np.asarray([distance_V[k] for k in sorted(distance_V.keys())])

    ((x_min, x_max), (y_min, y_max)) = measure_distance_extents(env)

    (H, xedges, yedges) = np.histogram2d(distance_U_array, distance_V_array, bins=[dx, dy])

    ax.axis([x_min, x_max, y_min, y_max])

    X, Y = np.meshgrid(xedges, yedges)
    pcm = ax.pcolormesh(X, Y, H.T)
    
    ax.set_xlabel('Arc distance (septal - temporal) (um)', fontsize=fig_options.fontSize)
    ax.set_ylabel('Arc distance (supra - infrapyramidal)  (um)', fontsize=fig_options.fontSize)
    ax.set_title('Position distribution for population: %s' % (population), fontsize=fig_options.fontSize)
    ax.set_aspect('equal')
    fig.colorbar(pcm, ax=ax, shrink=0.5, aspect=20)
    
    if fig_options.saveFig:
        if isinstance(fig_options.saveFig, str):
            filename = fig_options.saveFig
        else:
            filename = population+' Reindex Positions.%s' % fig_options.figFormat
            plt.savefig(filename)

    if fig_options.showFig:
        show_figure()
    
    return ax


def plot_coords_in_volume(populations, coords_path, coords_namespace, config, scale=25., subvol=False):
    
    env = Env(config_file=config)

    rotate = env.geometry['Parametric Surface']['Rotation']
    min_extents = env.geometry['Parametric Surface']['Minimum Extent']
    max_extents = env.geometry['Parametric Surface']['Maximum Extent']

    layer_min_extent = None
    layer_max_extent = None
    for ((layer_name,max_extent),(_,min_extent)) in zip(viewitems(max_extents),viewitems(min_extents)):
        if layer_min_extent is None:
            layer_min_extent = np.asarray(min_extent)
        else:
            layer_min_extent = np.minimum(layer_min_extent, np.asarray(min_extent))
        if layer_max_extent is None:
            layer_max_extent = np.asarray(max_extent)
        else:
            layer_max_extent = np.maximum(layer_max_extent, np.asarray(max_extent))

    logger.info(("Layer minimum extents: %s" % (str(layer_min_extent))))
    logger.info(("Layer maximum extents: %s" % (str(layer_max_extent))))
    logger.info('Reading coordinates...')

    pop_min_extent = None
    pop_max_extent = None

    xcoords = []
    ycoords = []
    zcoords = []
    for population in populations:
        coords = read_cell_attributes(coords_path, population, namespace=coords_namespace)

        for (k,v) in coords:
            xcoords.append(v['X Coordinate'][0])
            ycoords.append(v['Y Coordinate'][0])
            zcoords.append(v['Z Coordinate'][0])

        if pop_min_extent is None:
            pop_min_extent = np.asarray(env.geometry['Cell Layers']['Minimum Extent'][population])
        else:
            pop_min_extent = np.minimum(pop_min_extent, np.asarray(env.geometry['Cell Layers']['Minimum Extent'][population]))

        if pop_max_extent is None:
            pop_max_extent = np.asarray(env.geometry['Cell Layers']['Maximum Extent'][population])
        else:
            pop_max_extent = np.minimum(pop_max_extent, np.asarray(env.geometry['Cell Layers']['Maximum Extent'][population]))

    pts = np.concatenate((np.asarray(xcoords).reshape(-1,1), \
                          np.asarray(ycoords).reshape(-1,1), \
                          np.asarray(zcoords).reshape(-1,1)),axis=1)

    from mayavi import mlab
    

    mlab.points3d(*pts.T, color=(1, 1, 0), scale_factor=scale)

    logger.info('Constructing volume...')

    from dentate.geometry import make_volume

    if subvol:
        subvol = make_volume ((pop_min_extent[0], pop_max_extent[0]), \
                              (pop_min_extent[1], pop_max_extent[1]), \
                              (pop_min_extent[2], pop_max_extent[2]), \
                              resolution=[20, 20, 3], \
                              rotate=rotate)
    else:
        vol = make_volume ((layer_min_extent[0], layer_max_extent[0]), \
                           (layer_min_extent[1], layer_max_extent[1]), \
                           (layer_min_extent[2], layer_max_extent[2]), \
                           resolution=[20, 20, 3], \
                           rotate=rotate)


    logger.info('Plotting volume...')

    if subvol:
        subvol.mplot_surface(color=(0, 0.4, 0), opacity=0.33)
    else:
        vol.mplot_surface(color=(0, 1, 0), opacity=0.33)
    
    mlab.show()


def plot_trees_in_volume(population, forest_path, config, line_width=1., sample=0.05, coords_path=None, distances_namespace='Arc Distances', longitudinal_extent=None, volume='full', color_edge_scalars=True, volume_opacity=0.1):
    
    env = Env(config_file=config)

    rotate = env.geometry['Parametric Surface']['Rotation']

    pop_min_extent = np.asarray(env.geometry['Cell Layers']['Minimum Extent'][population])
    pop_max_extent = np.asarray(env.geometry['Cell Layers']['Maximum Extent'][population])

    min_extents = env.geometry['Parametric Surface']['Minimum Extent']
    max_extents = env.geometry['Parametric Surface']['Maximum Extent']
    layer_min_extent = None
    layer_max_extent = None
    for ((layer_name,max_extent),(_,min_extent)) in zip(viewitems(max_extents),viewitems(min_extents)):
        if layer_min_extent is None:
            layer_min_extent = np.asarray(min_extent)
        else:
            layer_min_extent = np.minimum(layer_min_extent, np.asarray(min_extent))
        if layer_max_extent is None:
            layer_max_extent = np.asarray(max_extent)
        else:
            layer_max_extent = np.maximum(layer_max_extent, np.asarray(max_extent))

    logger.info(("Layer minimum extents: %s" % (str(layer_min_extent))))
    logger.info(("Layer maximum extents: %s" % (str(layer_max_extent))))

    (population_ranges, _) = read_population_ranges(forest_path)

    population_start = population_ranges[population][0]
    population_count = population_ranges[population][1]

    import networkx as nx
    from mayavi import mlab
    
    if longitudinal_extent is None:
        #(trees, _) = NeuroH5TreeGen(forest_path, population)
        if isinstance(sample, numbers.Real):
            s = np.random.random_sample((population_count,))
            selection = np.where(s <= sample) + population_start
        else:
            selection = list(sample)
    else:
        print('Reading distances...')
        distances = read_cell_attributes(coords_path, population, namespace=distances_namespace)
    
        soma_distances = { k: (v['U Distance'][0], v['V Distance'][0]) for (k,v) in distances }
        del distances

        lst = []
        for k, v in viewitems(soma_distances):
            if v[0] >= longitudinal_extent[0] and v[0] <= longitudinal_extent[1]:
                lst.append(k)
        sample_range = np.asarray(lst)
                          
        if isinstance(sample, numbers.Real):
            s = np.random.random_sample(sample_range.shape)
            selection = sample_range[np.where(s <= sample)]
        else:
            raise RuntimeError('Sample must be a real number')

    print('%d trees selected from population %s' % (len(selection), population))
    (tree_iter, _) = read_tree_selection(forest_path, population, selection=selection.tolist())
   
    mlab.figure(bgcolor=(0,0,0))
    for (gid,tree_dict) in tree_iter:

        logger.info('plotting tree %i' % gid)
        xcoords = tree_dict['x']
        ycoords = tree_dict['y']
        zcoords = tree_dict['z']
        swc_type = tree_dict['swc_type']
        layer    = tree_dict['layer']
        secnodes = tree_dict['section_topology']['nodes']
        src      = tree_dict['section_topology']['src']
        dst      = tree_dict['section_topology']['dst']

        dend_idxs = np.where(swc_type == 4)[0]
        dend_idx_set = set(dend_idxs.flat)

        edges = []
        for sec, nodes in viewitems(secnodes):
            for i in range(1, len(nodes)):
                srcnode = nodes[i-1]
                dstnode = nodes[i]
                if ((srcnode in dend_idx_set) and (dstnode in dend_idx_set)):
                    edges.append((srcnode, dstnode))
        for (s,d) in zip(src,dst):
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
        if color_edge_scalars:
            edge_scalars = z[start_idx]
            edge_color = None
        else:
            edge_scalars = None
            edge_color = hex2rgb(rainbow_color_list[gid%len(rainbow_color_list)])
                                        
        # Plot this with Mayavi
        plot_graph(x, y, z, start_idx, end_idx, edge_scalars=edge_scalars, edge_color=edge_color, \
                       opacity=0.8, colormap='summer', line_width=line_width)

            
    from dentate.geometry import make_volume

    if volume == 'none':
        pass
    elif volume == 'subvol':
        logger.indo('Creating volume...')
        vol = make_volume ((pop_min_extent[0], pop_max_extent[0]), \
                               (pop_min_extent[1], pop_max_extent[1]), \
                               (pop_min_extent[2], pop_max_extent[2]), \
                               rotate=rotate)

        logger.info('Plotting volume...')
        vol.mplot_surface(color=(0, 0.4, 0), opacity=volume_opacity)
    elif volume == 'full':
        logger.info('Creating volume...')
        vol = make_volume ((layer_min_extent[0], layer_max_extent[0]), \
                              (layer_min_extent[1], layer_max_extent[1]), \
                              (layer_min_extent[2], layer_max_extent[2]), \
                              rotate=rotate)
        logger.info('Plotting volume...')
        vol.mplot_surface(color=(0, 1, 0), opacity=volume_opacity)
    else:
        raise ValueError('Unknown volume plot type %s' % volume)        

    mlab.gcf().scene.x_plus_view()
    mlab.savefig('%s_trees_in_volume.tiff' % population, magnification=10)
    mlab.show()


def plot_population_density(population, soma_coords, distances_namespace, max_u, max_v, bin_size=100., **kwargs):
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

    fig_options = copy.copy(default_fig_options)
    fig_options.update(kwargs)
    
    fig1 = plt.figure()
    ax = fig1.add_subplot(111, projection='3d')
    pop_size = len(soma_coords[population]['x'])
    indexes = random.sample(list(range(pop_size)), min(pop_size, 5000))
    ax.scatter(soma_coords[population]['x'][indexes], soma_coords[population]['y'][indexes],
               soma_coords[population]['z'][indexes], alpha=0.1, linewidth=0)
    scaling = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    ax.auto_scale_xyz(*[[np.min(scaling), np.max(scaling)]] * 3)
    ax.set_xlabel('X (um)')
    ax.set_ylabel('Y (um)')
    ax.set_zlabel('Z (um)')

    step_sizes = [int(old_div(max_u, bin_size)), int(old_div(max_v, bin_size))]
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

    if fig_options.saveFig:
        if isinstance(fig_options.saveFig, str):
            filename = fig_options.saveFig
        else:
            filename = distances_namespace+' '+'density.%s' % fig_options.figFormat
            plt.savefig(filename)

    if fig_options.showFig:
        show_figure()

    return ax


def plot_lfp(config, input_path, time_range = None, compute_psd=False, window_size=1024, frequency_range=(0, 400.), overlap=0.5, **kwargs):
    '''
    Line plot of LFP state variable (default: v). Returns figure handle.

    config: path to model configuration file
    input_path: file with LFP trace data
    time_range ([start:stop]): Time range of spikes shown; if None shows all (default: None)
    '''
    fig_options = copy.copy(default_fig_options)
    fig_options.update(kwargs)

    env = Env(config_file=config)

    nrows = len(env.lfpConfig)
    if compute_psd:
        ncols = 2
    else:
        ncols = 1
        
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=fig_options.figSize, squeeze=False)
    for iplot, (lfp_label, lfp_config_dict) in enumerate(viewitems(env.lfpConfig)):
        namespace_id = "Local Field Potential %s" % str(lfp_label)
        import h5py
        infile = h5py.File(input_path)

        logger.info('plot_lfp: reading data for %s...' % namespace_id)
        if time_range is None:
            t = infile[namespace_id]['t']
            v = infile[namespace_id]['v']
        else:
            tlst = []
            vlst = []
            for (t,v) in zip(infile[namespace_id]['t'], infile[namespace_id]['v']):
                if time_range[0] <= t <= time_range[1]:
                    tlst.append(t)
                    vlst.append(v)
            t = np.asarray(tlst)
            v = np.asarray(vlst)

        dt = lfp_config_dict['dt']

        if compute_psd:
            Fs = 1000. / dt

            nperseg    = window_size
            win        = signal.get_window('hanning', nperseg)
            noverlap   = int(overlap * nperseg)
            
            freqs, psd = signal.welch(v, fs=Fs, scaling='density', nperseg=nperseg, noverlap=noverlap,
                                      window=win, return_onesided=True)
            
            freqinds = np.where((freqs >= frequency_range[0]) & (freqs <= frequency_range[1]))

            freqs = freqs[freqinds]
            psd = psd[freqinds]
            if np.all(psd):
                psd = 10. * np.log10(psd)

            peak_index = np.where(psd == np.max(psd))[0]


        axes[iplot, 0].set_title('%s' % (namespace_id), fontsize=fig_options.fontSize)
        axes[iplot, 0].plot(t, v, label=lfp_label, linewidth=fig_options.lw)
        axes[iplot, 0].set_xlabel('Time (ms)', fontsize=fig_options.fontSize)
        axes[iplot, 0].set_ylabel('Field Potential (mV)', fontsize=fig_options.fontSize)
        if compute_psd:
            axes[iplot, 1].plot(freqs, psd, linewidth=fig_options.lw)
            axes[iplot, 1].set_xlabel('Frequency (Hz)', fontsize=fig_options.fontSize)
            axes[iplot, 1].set_ylabel('Power Spectral Density (dB/Hz)', fontsize=fig_options.fontSize)
            axes[iplot, 1].set_title('PSD %s (peak: %.3g Hz)' % (namespace_id, freqs[peak_index]), fontsize=fig_options.fontSize)            
            
    # save figure
    if fig_options.saveFig:
        if isinstance(fig_options.saveFig, str):
            filename = fig_options.saveFig
        else:
            filename = namespace_id+'.%s' % fig_options.figFormat
            plt.savefig(filename)
                
    # show fig
    if fig_options.showFig:
        show_figure()

    return fig
        


## Plot intracellular state trace 
def plot_intracellular_state (input_path, namespace_id, include = ['eachPop'], time_range = None, time_variable='t', variable='v', max_units = 1, unit_no = None, query = False, labels = None, marker = '|', **kwargs): 
    ''' 
    Line plot of intracellular state variable (default: v). Returns the figure handle.

    input_path: file with spike data
    namespace_id: attribute namespace for spike events
    time_range ([start:stop]): Time range of spikes shown; if None shows all (default: None)
    time_variable: Name of variable containing spike times (default: 't')
    variable: Name of state variable (default: 'v')
    max_units (int): maximum number of units from each population that will be plotted  (default: 1)
    labels = ('legend', 'overlay'): Show population labels in a legend or overlayed on one side of raster (default: 'legend')
    marker (char): Marker for each spike (default: '|')
    '''

    fig_options = copy.copy(default_fig_options)
    fig_options.update(kwargs)

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

    data = statedata.read_state (input_path, include, namespace_id, time_variable=time_variable,
                                 variable=variable, time_range=time_range, 
                                 max_units = max_units, unit_no = unit_no, query = query)

    if query:
        return

    states     = data['states']
    
    pop_colors = { pop_name: color_list[ipop%len(color_list)] for ipop, pop_name in enumerate(states) }
    
    stplots = []
    
    fig, ax1 = plt.subplots(figsize=fig_options.figSize,sharex='all',sharey='all')
        
    for (pop_name, pop_states) in viewitems(states):
        
        for (gid, cell_states) in viewitems(pop_states):
            
            stplots.append(ax1.plot(cell_states[0], cell_states[1], linewidth=fig_options.lw, marker=marker, c=pop_colors[pop_name], alpha=0.5, label=pop_name))
            

    ax1.set_xlabel('Time (ms)', fontsize=fig_options.fontSize)
    ax1.set_ylabel(variable, fontsize=fig_options.fontSize)
    #ax1.set_xlim(time_range)
    
    # Add legend
    pop_labels = pop_name
    
    if labels == 'legend':
        legend_labels = pop_labels
        lgd = plt.legend(stplots, legend_labels, fontsize=fig_options.fontSize, scatterpoints=1, markerscale=5.,
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
        
    # save figure
    if fig_options.saveFig:
        if isinstance(fig_options.saveFig, str):
            filename = fig_options.saveFig
        else:
            filename = namespace_id+' '+'state.%s' % fig_options.figFormat
            plt.savefig(filename)
                
    # show fig 
    if fig_options.showFig:
        show_figure()
    
    return fig


## Plot spike raster
def plot_spike_raster (input_path, namespace_id, include = ['eachPop'], time_range = None, time_variable='t', max_spikes = int(1e6), labels = 'legend', pop_rates = True, spike_hist = None, spike_hist_bin = 5, marker='|', **kwargs):
    ''' 
    Raster plot of network spike times. Returns the figure handle.

    input_path: file with spike data
    namespace_id: attribute namespace for spike events
    time_range ([start:stop]): Time range of spikes shown; if None shows all (default: None)
    time_variable: Name of variable containing spike times (default: 't')
    max_spikes (int): maximum number of spikes that will be plotted  (default: 1e6)
    labels = ('legend', 'overlay'): Show population labels in a legend or overlayed on one side of raster (default: 'legend')
    pop_rates = (True|False): Include population rates (default: False)
    spike_hist (None|'overlay'|'subplot'): overlay line over raster showing spike histogram (spikes/bin) (default: False)
    spike_hist_bin (int): Size of bin in ms to use for histogram (default: 5)
    marker (char): Marker for each spike (default: '|')
    '''

    fig_options = copy.copy(default_fig_options)
    fig_options.update(kwargs)

    (population_ranges, N) = read_population_ranges(input_path)
    population_names  = read_population_names(input_path)

    total_num_cells = 0
    pop_num_cells = {}
    pop_start_inds = {}
    for k in population_names:
        pop_start_inds[k] = population_ranges[k][0]
        pop_num_cells[k] = population_ranges[k][1]
        total_num_cells += population_ranges[k][1]

    include = list(include)
    # Replace 'eachPop' with list of populations
    if 'eachPop' in include: 
        include.remove('eachPop')
        for pop in population_names:
            include.append(pop)
            
    # sort according to start index        
    include.sort(key=lambda x: pop_start_inds[x])
    
    spkdata = spikedata.read_spike_events (input_path, include, namespace_id, spike_train_attr_name=time_variable, time_range=time_range)

    spkpoplst        = spkdata['spkpoplst']
    spkindlst        = spkdata['spkindlst']
    spktlst          = spkdata['spktlst']
    num_cell_spks    = spkdata['num_cell_spks']
    pop_active_cells = spkdata['pop_active_cells']
    tmin             = spkdata['tmin']
    tmax             = spkdata['tmax']
    fraction_active  = { pop_name: float(len(pop_active_cells[pop_name])) / float(pop_num_cells[pop_name]) for pop_name in include }
    
    time_range = [tmin, tmax]

    # Calculate spike histogram if requested
    if spike_hist:
        all_spkts = np.concatenate(spktlst, axis=0)
        sphist_y, bin_edges = np.histogram(all_spkts, bins = np.arange(time_range[0], time_range[1], spike_hist_bin))
        sphist_x = bin_edges[:-1]+old_div(spike_hist_bin,2)

    maxN = 0
    minN = N

    avg_rates = {}
    tsecs = old_div((time_range[1]-time_range[0]),1e3) 
    for i,pop_name in enumerate(spkpoplst):
        pop_num = len(pop_active_cells[pop_name])
        maxN = max(maxN, max(pop_active_cells[pop_name]))
        minN = min(minN, min(pop_active_cells[pop_name]))
        if pop_num > 0:
            if num_cell_spks[pop_name] == 0:
                avg_rates[pop_name] = 0
            else:
                avg_rates[pop_name] = old_div(num_cell_spks[pop_name], old_div(pop_num, tsecs))
        
    
    pop_colors = { pop_name: color_list[ipop%len(raster_color_list)] for ipop, pop_name in enumerate(spkpoplst) }

    pop_spk_dict = { pop_name: (pop_spkinds, pop_spkts) for (pop_name, pop_spkinds, pop_spkts) in zip(spkpoplst, spkindlst, spktlst) }

    if spike_hist is None:
        fig, axes = plt.subplots(nrows=len(spkpoplst), sharex=True, figsize=fig_options.figSize)
    elif spike_hist == 'subplot':
        fig, axes = plt.subplots(nrows=len(spkpoplst)+1, sharex=True, figsize=fig_options.figSize,
                                 gridspec_kw={'height_ratios': [1]*len(spkpoplst) + [2]})
    fig.suptitle ('DG Spike Raster', fontsize=fig_options.fontSize)

    sctplots = []
    
    for i, pop_name in enumerate(include):
        pop_spkinds, pop_spkts = pop_spk_dict[pop_name]

        if max_spikes is not None:
            if int(max_spikes) < len(pop_spkinds):
               logger.info(('  Displaying only randomly sampled %i out of %i spikes for population %s' % (max_spikes, len(pop_spkts), pop_name)))
               sample_inds = np.random.randint(0, len(pop_spkinds)-1, size=int(max_spikes))
               pop_spkts   = pop_spkts[sample_inds]
               pop_spkinds = pop_spkinds[sample_inds]

        sct = axes[i].scatter(pop_spkts, pop_spkinds, s=10, linewidths=fig_options.lw, marker=marker, c=pop_colors[pop_name], alpha=0.5, label=pop_name)
        axes[i].spines["top"].set_visible(False)
        axes[i].spines["bottom"].set_visible(False)
        axes[i].spines["left"].set_visible(False)
        axes[i].spines["right"].set_visible(False)
        sctplots.append(sct)

        N = pop_num_cells[pop_name]
        S = pop_start_inds[pop_name]
        axes[i].set_ylim(S, S+N-1)
        
    lgd_info = [(100. * fraction_active[pop_name], avg_rates[pop_name]) for pop_name in spkpoplst if pop_name in avg_rates]
            
    # set raster plot y tick labels to the middle of the index range for each population
    for pop_name, a in zip_longest(include, fig.axes[:-1]):
        maxN = max(pop_active_cells[pop_name])
        minN = min(pop_active_cells[pop_name])
        loc = pop_start_inds[pop_name] + 0.5 * (maxN - minN)
        yaxis = a.get_yaxis()
        yaxis.set_ticks([loc])
        yaxis.set_ticklabels([pop_name])
        yaxis.set_tick_params(length=0)
        a.get_xaxis().set_tick_params(length=0)
        
    # Plot spike histogram
    pch = interpolate.pchip(sphist_x, sphist_y)
    res_npts = int((sphist_x.max() - sphist_x.min()))
    sphist_x_res = np.linspace(sphist_x.min(), sphist_x.max(), res_npts, endpoint=True)
    sphist_y_res = pch(sphist_x_res)

    if spike_hist == 'overlay':
        ax2 = axes[-1].twinx()
        ax2.plot (sphist_x_res, sphist_y_res, linewidth=0.5)
        ax2.set_ylabel('Spike count', fontsize=fig_options.fontSize) # add yaxis label in opposite side
        ax2.set_xlim(time_range)
    elif spike_hist == 'subplot':
        ax2=axes[-1]
        ax2.plot (sphist_x_res, sphist_y_res, linewidth=1.0)
        ax2.set_xlabel('Time (ms)', fontsize=fig_options.fontSize)
        ax2.set_ylabel('Spikes', fontsize=fig_options.fontSize)
        ax2.set_xlim(time_range)
        
#    locator=MaxNLocator(prune='both', nbins=10)
#    ax2.xaxis.set_major_locator(locator)
    
    if labels == 'legend':
        # Shrink axes by 15%
        for ax in axes:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        if pop_rates:
            lgd_labels = [ '%s (%.02f%% active; %.3g Hz)' % (pop_name, info[0], info[1]) for pop_name, info in zip_longest(include, lgd_info) ]
        else:
            lgd_labels = [ '%s (%.02f%% active)' % (pop_name, info[0]) for pop_name, info in zip_longest(include, lgd_info) ]
        # Add legend
        lgd = fig.legend(sctplots, lgd_labels, loc = 'center right', 
                         fontsize='small', scatterpoints=1, markerscale=5.,
                         bbox_to_anchor=(1.002, 0.5), bbox_transform=plt.gcf().transFigure)
        fig.artists.append(lgd)
       
    elif labels == 'overlay':
        if pop_rates:
            lgd_labels = [ '%s (%.02f%% active; %.3g Hz)' % (pop_name, info[0], info[1]) for pop_name, info in zip_longest(include, lgd_info) ]
        else:
            lgd_labels = [ '%s (%.02f%% active)' % (pop_name, info[0]) for pop_name, info in zip_longest(include, lgd_info) ]
        for i, (pop_name, lgd_info) in enumerate(zip(spkpoplst, lgd_info)):
                at = AnchoredText(pop_name + ' ' + lgd_label,
                                  loc='upper right', borderpad=0.01, prop=dict(size=fig_options.fontSize))
                axes[i].add_artist(at)
        max_label_len = max([len(l) for l in lgd_labels])
        
    elif labels == 'yticks':
        for pop_name, info, a in zip_longest(include, lgd_info, fig.axes[:-1]):
            if pop_rates:
                label = '%.02f%%\n%.2g Hz' % (info[0], info[1])
            else:
                label = '%.02f%%\n' % (info[0])

            maxN = max(pop_active_cells[pop_name])
            minN = min(pop_active_cells[pop_name])
            loc = pop_start_inds[pop_name] + 0.5 * (maxN - minN)
            a.set_yticks([loc, loc])
            a.set_yticklabels([pop_name, label])
            yticklabels = a.get_yticklabels()
            # Create offset transform in x direction
            dx = -66/72.; dy = 0/72. 
            offset = mpl.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
            # apply offset transform to labels.
            yticklabels[0].set_transform(yticklabels[0].get_transform() + offset)
            dx = -55/72.; dy = 0/72. 
            offset = mpl.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
            yticklabels[1].set_ha('left')    
            yticklabels[1].set_transform(yticklabels[1].get_transform() + offset)

            
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    
    # save figure
    if fig_options.saveFig:
       if isinstance(fig_options.saveFig, str):
           filename = fig_options.saveFig
       else:
           filename = namespace_id+' '+'raster.%s' % fig_options.figFormat
           plt.savefig(filename)
                
    # show fig 
    if fig_options.showFig:
        show_figure()
    
    return fig


    
    
def update_spatial_rasters(frame, scts, timebins, data, distances_U_dict, distances_V_dict, lgd):
    if frame > 0:
        t0 = timebins[frame]
        t1 = timebins[frame+1]
        for p, (pop_name, spkinds, spkts) in enumerate(data):
            distances_U = distances_U_dict[pop_name]
            distances_V = distances_V_dict[pop_name]
            rinds = np.where(np.logical_and(spkts >= t0, spkts <= t1))
            cinds = spkinds[rinds]
            x = np.asarray([distances_U[ind] for ind in cinds])
            y = np.asarray([distances_V[ind] for ind in cinds])
            scts[p].set_data(x, y)
            scts[p].set_label(pop_name)
            scts[-1].set_text('t = %f ms' % t1)
    return scts


def init_spatial_rasters(ax, timebins, data, range_U_dict, range_V_dict, distances_U_dict, distances_V_dict, lgd, marker, pop_colors, **kwargs):

    fig_options = copy.copy(default_fig_options)
    fig_options.update(kwargs)

    scts = []
    t0 = timebins[0]
    t1 = timebins[1]
    min_U = None
    min_V = None
    max_U = None
    max_V = None
    for (pop_name, spkinds, spkts) in data:
        distances_U = distances_U_dict[pop_name]
        distances_V = distances_V_dict[pop_name]
        rinds = np.where(np.logical_and(spkts >= t0, spkts <= t1))
        cinds = spkinds[rinds]
        x = np.asarray([distances_U[ind] for ind in cinds])
        y = np.asarray([distances_V[ind] for ind in cinds])
        #scts.append(ax.scatter(x, y, linewidths=options.lw, marker=marker, c=pop_colors[pop_name], alpha=0.5, label=pop_name))
        scts = scts + plt.plot([], [], marker, animated=True, alpha=0.5)
        if min_U is None:
            min_U = range_U_dict[pop_name][0]
        else:
            min_U = min(min_U, range_U_dict[pop_name][0])
        if min_V is None:
            min_V = range_V_dict[pop_name][0]
        else:
            min_V = min(min_V, range_V_dict[pop_name][0])
        if max_U is None:
            max_U = range_U_dict[pop_name][1]
        else:
            max_U = max(max_U, range_U_dict[pop_name][1])
        if max_V is None:
            max_V = range_V_dict[pop_name][1]
        else:
            max_V = max(max_V, range_V_dict[pop_name][1])
    ax.set_xlim((min_U, max_U))
    ax.set_ylim((min_V, max_V))
    
    return scts + [lgd(scts), plt.text(0.05, 0.95, 't = %f ms' % t0, fontsize=fig_options.fontSize, transform=ax.transAxes)]
        
spatial_raster_aniplots = []
        
## Plot spike raster
def plot_spatial_spike_raster (input_path, namespace_id, coords_path, distances_namespace='Arc Distances',
                               include = ['eachPop'], time_step = 5.0, time_range = None, time_variable='t', max_spikes = int(1e6), marker = 'o', **kwargs): 
    ''' 
    Spatial raster plot of network spike times. Returns the figure handle.

    input_path: file with spike data
    namespace_id: attribute namespace for spike events
    time_range ([start:stop]): Time range of spikes shown; if None shows all (default: None)
    time_variable: Name of variable containing spike times (default: 't')
    max_spikes (int): maximum number of spikes that will be plotted  (default: 1e6)
    labels = ('legend', 'overlay'): Show population labels in a legend or overlayed on one side of raster (default: 'legend')
    marker (char): Marker for each spike (default: '|')
    '''
    fig_options = copy.copy(default_fig_options)
    fig_options.update(kwargs)

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

    distance_U_dict = {}
    distance_V_dict = {}
    range_U_dict = {}
    range_V_dict = {}
    for population in include:
        distances = read_cell_attributes(coords_path, population, namespace=distances_namespace)
    
        soma_distances = { k: (v['U Distance'][0], v['V Distance'][0]) for (k,v) in distances }
        del distances
        
        logger.info('read distances (%i elements)' % len(soma_distances.keys()))
        distance_U_array = np.asarray([soma_distances[gid][0] for gid in soma_distances])
        distance_V_array = np.asarray([soma_distances[gid][1] for gid in soma_distances])

        U_min = np.min(distance_U_array)
        U_max = np.max(distance_U_array)
        V_min = np.min(distance_V_array)
        V_max = np.max(distance_V_array)

        range_U_dict[population] = (U_min, U_max)
        range_V_dict[population] = (V_min, V_max)
        
        distance_U = { gid: soma_distances[gid][0] for gid in soma_distances }
        distance_V = { gid: soma_distances[gid][1] for gid in soma_distances }

        distance_U_dict[population] = distance_U
        distance_V_dict[population] = distance_V
        
    spkdata = spikedata.read_spike_events (input_path, include, namespace_id, spike_train_attr_name=time_variable, time_range=time_range)

    spkpoplst        = spkdata['spkpoplst']
    spkindlst        = spkdata['spkindlst']
    spktlst          = spkdata['spktlst']
    num_cell_spks    = spkdata['num_cell_spks']
    pop_active_cells = spkdata['pop_active_cells']
    tmin             = spkdata['tmin']
    tmax             = spkdata['tmax']
    
    time_range = [tmin, tmax]
    
    pop_colors = { pop_name: color_list[ipop%len(color_list)] for ipop, pop_name in enumerate(spkpoplst) }
    
    # Plot spikes
    fig, ax = plt.subplots(figsize=fig_options.figSize)

    pop_labels = [ pop_name for pop_name in spkpoplst ]
    legend_labels = pop_labels
    lgd = lambda objs: plt.legend(objs, legend_labels, fontsize=fig_options.fontSize, scatterpoints=1, markerscale=2., \
                                    loc='upper right', bbox_to_anchor=(0.95, 0.95))
    
    timebins = np.linspace(tmin, tmax, old_div((tmax-tmin), time_step))
    
    data = list(zip (spkpoplst, spkindlst, spktlst))
    scts = init_spatial_rasters(ax, timebins, data, range_U_dict, range_V_dict, distance_U_dict, distance_V_dict, lgd, marker, pop_colors)
    ani = FuncAnimation(fig, func=update_spatial_rasters, frames=list(range(0, len(timebins)-1)), \
                        blit=True, repeat=False, init_func=lambda: scts, fargs=(scts, timebins, data, distance_U_dict, distance_V_dict, lgd))
    spatial_raster_aniplots.append(ani)

    # show fig 
    if fig_options.showFig:
        show_figure()
    
    return fig


def plot_network_clamp (input_path, spike_namespace, intracellular_namespace, unit_no, include='eachPop', time_range = None, time_variable='t', intracellular_variable='v', labels = 'legend', pop_rates = True, spike_hist = None, spike_hist_bin = 5, marker = ',', **kwargs): 
    ''' 
    Raster plot of target cell intracellular trace + spike raster of presynaptic inputs. Returns the figure handle.

    input_path: file with spike data
    spike_namespace: attribute namespace for spike events
    intracellular_namespace: attribute namespace for intracellular trace
    time_range ([start:stop]): Time range of spikes shown; if None shows all (default: None)
    time_variable: Name of variable containing spike times (default: 't')
    labels = ('legend', 'overlay'): Show population labels in a legend or overlayed on one side of raster (default: 'legend')
    pop_rates = (True|False): Include population rates (default: False)
    spike_hist (None|'overlay'|'subplot'): overlay line over raster showing spike histogram (spikes/bin) (default: False)
    spike_hist_bin (int): Size of bin in ms to use for histogram (default: 5)
    marker (char): Marker for each spike (default: '|')
    '''
    fig_options = copy.copy(default_fig_options)
    fig_options.update(kwargs)

    (population_ranges, N) = read_population_ranges(input_path)
    population_names  = read_population_names(input_path)

    popName = None
    pop_num_cells = {}
    pop_start_inds = {}
    for k in population_names:
        pop_start_inds[k] = population_ranges[k][0]
        pop_range = population_ranges[k]
        pop_num_cells[k] = pop_range[1]
        if (unit_no >= pop_range[0]) and (unit_no < pop_range[0] + pop_range[1]):
           popName = k

    include = list(include)
    # Replace 'eachPop' with list of populations
    if 'eachPop' in include: 
        include.remove('eachPop')
        for pop in population_names:
            include.append(pop)

    # sort according to start index        
    include.sort(key=lambda x: pop_start_inds[x])
    include.reverse()

    spkdata = spikedata.read_spike_events (input_path, include, spike_namespace, \
                                           spike_train_attr_name=time_variable, time_range=time_range)
    indata  = statedata.read_state (input_path, [popName], intracellular_namespace, time_variable=time_variable, \
                                    variable=intracellular_variable, time_range=time_range, unit_no = [unit_no])

    spkpoplst        = spkdata['spkpoplst']
    spkindlst        = spkdata['spkindlst']
    spktlst          = spkdata['spktlst']
    num_cell_spks    = spkdata['num_cell_spks']
    pop_active_cells = spkdata['pop_active_cells']
    tmin             = spkdata['tmin']
    tmax             = spkdata['tmax']
    
    time_range = [tmin, tmax]


    histo_dict = {}
    # Calculate spike histogram if requested
    if spike_hist:
        all_spkts = np.concatenate(spktlst, axis=0)
        sphist_y, bin_edges = np.histogram(all_spkts, bins = np.arange(time_range[0], time_range[1], spike_hist_bin))
        sphist_x = bin_edges[:-1]+old_div(spike_hist_bin,2)

        
    maxN = 0
    minN = N

    avg_rates = {}
    tsecs = old_div((time_range[1]-time_range[0]),1e3) 
    for i,pop_name in enumerate(spkpoplst):
        pop_num = len(pop_active_cells[pop_name])
        maxN = max(maxN, max(pop_active_cells[pop_name]))
        minN = min(minN, min(pop_active_cells[pop_name]))
        if pop_num > 0:
            if num_cell_spks[pop_name] == 0:
                avg_rates[pop_name] = 0
            else:
                avg_rates[pop_name] = old_div(num_cell_spks[pop_name], old_div(pop_num, tsecs))
        
    
    pop_colors = { pop_name: color_list[ipop%len(color_list)] for ipop, pop_name in enumerate(spkpoplst) }

    pop_spk_dict = { pop_name: (pop_spkinds, pop_spkts) for (pop_name, pop_spkinds, pop_spkts) in zip(spkpoplst, spkindlst, spktlst) }
    
    # Plot spikes
    if spike_hist is None:
        fig, axes = plt.subplots(nrows=len(spkpoplst)+1, sharex=True, figsize=fig_options.figSize,
                                 gridspec_kw={'height_ratios': [1]*len(spkpoplst) + [2]})
    elif spike_hist == 'subplot':
        fig, axes = plt.subplots(nrows=len(spkpoplst)+2, sharex=True, figsize=fig_options.figSize,
                                 gridspec_kw={'height_ratios': [1]*len(spkpoplst) + [2, 2]})

    sctplots = []
    
    for i, pop_name in enumerate(include):
        pop_spkinds, pop_spkts = pop_spk_dict[pop_name]

        sctplots.append(axes[i].scatter(pop_spkts, pop_spkinds, s=10, linewidths=fig_options.lw, marker=marker, c=pop_colors[pop_name], alpha=0.5, label=pop_name))

        N = pop_num_cells[pop_name]
        S = pop_start_inds[pop_name]
        axes[i].set_ylim(S, S+N-1)
        
    axes[0].set_xlim(time_range)

    axes[0].set_xlabel('Time (ms)', fontsize=fig_options.fontSize)
    axes[0].set_ylabel('Cell Index', fontsize=fig_options.fontSize)

    axes[0].tick_params(axis='x', which='major', labelsize=fig_options.fontSize)
    axes[0].tick_params(axis='x', which='minor', labelsize=fig_options.fontSize)

    fig.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in fig.axes[:-2]], visible=False)

    # set raster plot ticks to the end of the index range for each population
    for i, pop_name in enumerate(include):
        a = fig.axes[i]
        start, end = a.get_ylim()
        a.get_yaxis().set_ticks([end])

    # set raster plot ticks to start and end of index range for first population
    a = fig.axes[len(include)-1]
    start, end = a.get_ylim()
    a.get_yaxis().set_ticks([start, end])
        
    if pop_rates:
        lgd_labels = [pop_name + ' (%i active; %.3g Hz)' % (len(pop_active_cells[pop_name]), avg_rates[pop_name]) for pop_name in spkpoplst if pop_name in avg_rates]
    else:
        lgd_labels = [pop_name + ' (%i active)' % (len(pop_active_cells[pop_name]))  for pop_name in spkpoplst if pop_name in avg_rates]

    # Plot spike histogram
    pch = interpolate.pchip(sphist_x, sphist_y)
    res_npts = int((sphist_x.max() - sphist_x.min()))
    sphist_x_res = np.linspace(sphist_x.min(), sphist_x.max(), res_npts, endpoint=True)
    sphist_y_res = pch(sphist_x_res)

    if spike_hist == 'overlay':
        ax2 = axes[-2].twinx()
        ax2.plot (sphist_x_res, sphist_y_res, linewidth=0.5)
        ax2.set_ylabel('Spike count', fontsize=fig_options.fontSize) # add yaxis label in opposite side
        ax2.set_xlim(time_range)
    elif spike_hist == 'subplot':
        ax2=axes[-2]
        ax2.plot (sphist_x_res, sphist_y_res, linewidth=1.0)
        ax2.set_xlabel('Time (ms)', fontsize=fig_options.fontSize)
        ax2.set_ylabel('Spike count', fontsize=fig_options.fontSize)
        ax2.set_xlim(time_range)
            
    # Plot intracellular state
    ax3 = axes[-1]
    ax3.set_xlabel('Time (ms)', fontsize=fig_options.fontSize)
    ax3.set_ylabel(intracellular_variable, fontsize=fig_options.fontSize)
    ax3.set_xlim(time_range)

    states = indata['states']
    stplots = []
    for (pop_name, pop_states) in viewitems(states):
        for (gid, cell_states) in viewitems(pop_states):
            st_x, st_y = cell_states
            pch = interpolate.pchip(st_x, st_y)
            res_npts = int((st_x.max() - st_x.min()))
            st_x_res = np.linspace(st_x.min(), st_x.max(), res_npts, endpoint=True)
            st_y_res = pch(st_x_res)
            stplots.append(ax3.plot(st_x_res, st_y_res, linewidth=fig_options.lw, marker=marker, alpha=0.5, label=pop_name))

    if labels == 'legend':
       # Shrink axes by 15%
       for ax in axes:
           box = ax.get_position()
           ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
       # Add legend
       lgd = fig.legend(sctplots, lgd_labels, loc = 'center right', 
                        fontsize='small', scatterpoints=1, markerscale=5.,
                        bbox_to_anchor=(1.002, 0.5), bbox_transform=plt.gcf().transFigure)
       fig.artists.append(lgd)
       
    elif labels == 'overlay':
        for i, (pop_name, lgd_label) in enumerate(zip(spkpoplst, lgd_labels)):
                at = AnchoredText(lgd_label, loc='upper right', borderpad=0.01, prop=dict(size=fig_options.fontSize))
                axes[i].add_artist(at)
        max_label_len = max([len(l) for l in lgd_labels])
            
    # save figure
    if fig_options.saveFig:
        if isinstance(fig_options.saveFig, str):
            filename = fig_options.saveFig
        else:
            filename = 'Network Clamp %s %i.%s' % (popName, unit_no, fig_options.figFormat)
            plt.savefig(filename)
                
    # show fig 
    if fig_options.showFig:
        show_figure()
    
    return fig


def plot_spike_rates (input_path, namespace_id, include = ['eachPop'], time_range = None, time_variable='t', meansub=False, max_units = None, labels = 'legend', bin_size = 100., progress=False, **kwargs):
    ''' 
    Plot of network firing rates. Returns the figure handle.

    input_path: file with spike data
    namespace_id: attribute namespace for spike events
    time_range ([start:stop]): Time range of spikes shown; if None shows all (default: None)
    time_variable: Name of variable containing spike times (default: 't')
    labels = ('legend', 'overlay'): Show population labels in a legend or overlayed on one side of raster (default: 'legend')
    '''
    fig_options = copy.copy(default_fig_options)
    fig_options.update(kwargs)

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

    spkdata = spikedata.read_spike_events (input_path, include, namespace_id, spike_train_attr_name=time_variable,
                                           time_range=time_range)

    spkpoplst        = spkdata['spkpoplst']
    spkindlst        = spkdata['spkindlst']
    spktlst          = spkdata['spktlst']
    num_cell_spks    = spkdata['num_cell_spks']
    pop_active_cells = spkdata['pop_active_cells']
    tmin             = spkdata['tmin']
    tmax             = spkdata['tmax']

    time_range = [tmin, tmax]
    time_bins  = np.arange(time_range[0], time_range[1], bin_size)

    spkrate_dict = {}
    for subset, spkinds, spkts in zip(spkpoplst, spkindlst, spktlst):
        spkdict = spikedata.make_spike_dict(spkinds, spkts)
        sdf_dict = spikedata.spike_density_estimate(subset, spkdict, time_bins, progress=progress)
        i = 0
        rate_dict = {}
        for ind, dct in viewitems(sdf_dict):
            rates       = np.asarray(dct['rate'], dtype=np.float32)
            meansub_rates = rates - np.mean(rates)
            peak        = np.mean(rates[np.where(rates >= np.percentile(rates, 90.))[0]])
            peak_index  = np.where(rates == np.max(rates))[0][0]
            rate_dict[i] = { 'rate': rates, 'meansub': meansub_rates, 'peak': peak, 'peak index': peak_index }
            i = i+1
            if max_units is not None:
                if i >= max_units:
                    break
        spkrate_dict[subset] = rate_dict
        logger.info(('Calculated spike rates for %i cells in population %s' % (len(rate_dict), subset)))

                    
    # Plot spikes
    fig, ax1 = plt.subplots(figsize=fig_options.figSize)


    for (iplot, subset) in enumerate(spkpoplst):

        pop_rates = spkrate_dict[subset]
        
        peak_lst = []
        for ind, rate_dict in viewitems(pop_rates):
            rate       = rate_dict['rate']
            peak_index = rate_dict['peak index']
            peak_lst.append(peak_index)

        ind_peak_lst = list(enumerate(peak_lst))
        del(peak_lst)
        ind_peak_lst.sort(key=lambda i_x: i_x[1])

        if meansub:
            rate_lst = [ pop_rates[i]['meansub'] for i, _ in ind_peak_lst ]
        else:
            rate_lst = [ pop_rates[i]['rate'] for i, _ in ind_peak_lst ]
        del(ind_peak_lst)

        rate_matrix = np.matrix(rate_lst, dtype=np.float32)
        del(rate_lst)

        color = color_list[iplot%len(color_list)]

        plt.subplot(len(spkpoplst),1,iplot+1)  # if subplot, create new subplot
        if meansub:
            plt.title ('%s Mean-subtracted Instantaneous Firing Rate' % str(subset), fontsize=fig_options.fontSize)
        else:
            plt.title ('%s Instantaneous Firing Rate' % str(subset), fontsize=fig_options.fontSize)

        im = plt.imshow(rate_matrix, origin='upper', aspect='auto', interpolation='none',
                        extent=[time_range[0], time_range[1], 0, rate_matrix.shape[0]], cmap=fig_options['colormap'])

        im.axes.tick_params(labelsize=fig_options.fontSize)
        
        if iplot == 0: 
            plt.ylabel('Relative Cell Index', fontsize=fig_options.fontSize)
        if iplot == len(spkpoplst)-1:
            plt.xlabel('Time (ms)', fontsize=fig_options.fontSize)

        cbar = plt.colorbar(im)
        cbar.ax.set_ylabel('Firing Rate (Hz)', fontsize=fig_options.fontSize)
        cbar.ax.tick_params(labelsize=fig_options.fontSize)

    if fig_options.saveFig:
        if isinstance(fig_options.saveFig, str):
            filename = fig_options.saveFig
        else:
            if meansub:
                filename = '%s meansub firing rate.%s' % (namespace_id, fig_options.figFormat)
            else:
                filename = '%s firing rate.%s' % (namespace_id, fig_options.figFormat)
        plt.savefig(filename)
                
    # show fig 
    if fig_options.showFig:
        show_figure()
    
    return fig

def plot_spike_histogram (input_path, namespace_id, include = ['eachPop'], time_variable='t', time_range = None, 
                          pop_rates = False, bin_size = 5., smooth = 0, quantity = 'rate', progress = False,
                          overlay=True, graph_type='bar', **kwargs):
    ''' 
    Plots spike histogram. Returns figure handle.

        - input_path: file with spike data
        - namespace_id: attribute namespace for spike events
        - include (['eachPop'|<population name>]): List of data series to include. 
            (default: ['eachPop'] - expands to the name of each population)
        - time_variable: Name of variable containing spike times (default: 't')
        - time_range ([start:stop]): Time range of spikes shown; if None shows all (default: None)
        - bin_size (int): Size in ms of each bin (default: 5)
        - overlay (True|False): Whether to overlay the data lines or plot in separate subplots (default: True)
        - graph_type ('line'|'bar'): Type of graph to use (line graph or bar plot) (default: 'line')
        - quantity ('rate'|'count'): Quantity of y axis (firing rate in Hz, or spike count) (default: 'rate')
    '''
    fig_options = copy.copy(default_fig_options)
    fig_options.update(kwargs)

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
        include.reverse()
        
    spkdata = spikedata.read_spike_events (input_path, include, namespace_id, spike_train_attr_name=time_variable,
                                           time_range=time_range)

    spkpoplst        = spkdata['spkpoplst']
    spkindlst        = spkdata['spkindlst']
    spktlst          = spkdata['spktlst']
    num_cell_spks    = spkdata['num_cell_spks']
    pop_active_cells = spkdata['pop_active_cells']
    tmin             = spkdata['tmin']
    tmax             = spkdata['tmax']

    time_range = [tmin, tmax]

    avg_rates = {}
    maxN = 0
    minN = N
    if pop_rates:
        tsecs = old_div((time_range[1]-time_range[0]),1e3) 
        for i,pop_name in enumerate(spkpoplst):
            pop_num = len(pop_active_cells[pop_name])
            maxN = max(maxN, max(pop_active_cells[pop_name]))
            minN = min(minN, min(pop_active_cells[pop_name]))
            if pop_num > 0:
                if num_cell_spks[pop_name] == 0:
                    avg_rates[pop_name] = 0
                else:
                    avg_rates[pop_name] = old_div(num_cell_spks[pop_name], old_div(pop_num, tsecs))
            
    # Y-axis label
    if quantity == 'rate':
        yaxisLabel = 'Mean cell firing rate (Hz)'
    elif quantity == 'count':
        yaxisLabel = 'Spike count'
    elif quantity == 'active':
        yaxisLabel = 'Active cell count'
    else:
        print('Invalid quantity value %s' % str(quantity))
        return

    # create fig
    fig, axes = plt.subplots(len(spkpoplst), 1, figsize=fig_options.figSize, sharex=True)
        
    time_bins  = np.arange(time_range[0], time_range[1], bin_size)

    
    hist_dict = {}
    if quantity == 'rate':
        for subset, spkinds, spkts in zip(spkpoplst, spkindlst, spktlst):
            spkdict = spikedata.make_spike_dict(spkinds, spkts)
            sdf_dict = spikedata.spike_density_estimate(subset, spkdict, time_bins, progress=progress)
            bin_dict = defaultdict(lambda: {'rates':0.0, 'active': 0})
            for (ind, dct) in viewitems(sdf_dict):
                rate = dct['rate']
                for ibin in range(0, len(time_bins)):
                    d = bin_dict[ibin]
                    bin_rate = rate[ibin]
                    d['rates']  += bin_rate
                    d['active'] += 1
            hist_dict[subset] = bin_dict
            logger.info(('Calculated spike rates for %i cells in population %s' % (len(sdf_dict), subset)))
    else:
        for subset, spkinds, spkts in zip(spkpoplst, spkindlst, spktlst):
            spkdict = spikedata.make_spike_dict(spkinds, spkts)
            count_bin_dict = spikedata.spike_bin_counts(spkdict, time_bins)
            bin_dict      = defaultdict(lambda: {'counts':0, 'active': 0})
            for (ind, counts) in viewitems(count_bin_dict):
                for ibin in range(0, len(time_bins)-1):
                    d = bin_dict[ibin]
                    d['counts'] += counts[ibin]
                    d['active'] += 1
            hist_dict[subset] = bin_dict
            logger.info(('Calculated spike counts for %i cells in population %s' % (len(count_bin_dict), subset)))
        
            
    del spkindlst, spktlst

    # Plot separate line for each entry in include
    for iplot, subset in enumerate(spkpoplst):

        hist_x = time_bins+old_div(bin_size,2)
        bin_dict = hist_dict[subset]

        if quantity=='rate':
            hist_y = np.asarray([old_div(bin_dict[ibin]['rates'], bin_dict[ibin]['active'])  if bin_dict[ibin]['active'] > 0 else 0.
                                     for ibin in range(0, len(time_bins))])
        elif quantity=='active':
            hist_y = np.asarray([bin_dict[ibin]['active'] for ibin in range(0, len(time_bins))])
        else:
            hist_y = np.asarray([bin_dict[ibin]['counts'] for ibin in range(0, len(time_bins))])

        del bin_dict
        del hist_dict[subset]
        
        color = color_list[iplot%len(color_list)]

        if not overlay:
            if pop_rates:
                label = str(subset)  + ' (%i active; %.3g Hz)' % (len(pop_active_cells[subset]), avg_rates[subset])
            else:
                label = str(subset)  + ' (%i active)' % (len(pop_active_cells[subset]))

        ax = plt.subplot(len(spkpoplst),1,(iplot+1))
        plt.title (label, fontsize=fig_options.fontSize)
        ax.tick_params(labelsize=fig_options.fontSize)
        #axes[iplot].xaxis.set_visible(False)
            
        if smooth:
            hsignal = signal.savgol_filter(hist_y, window_length=2*(old_div(len(hist_y),16)) + 1, polyorder=smooth) 
        else:
            hsignal = hist_y
        
        if graph_type == 'line':
            ax.plot (hist_x, hsignal, linewidth=fig_options.lw, color = color)
        elif graph_type == 'bar':
            ax.bar(hist_x, hsignal, width = bin_size, color = color)

        if iplot == 0:
            ax.set_ylabel(yaxisLabel, fontsize=fig_options.fontSize)
        if iplot == len(spkpoplst)-1:
            ax.set_xlabel('Time (ms)', fontsize=fig_options.fontSize)
        else:
            ax.tick_params(labelbottom='off')

            
        ax.set_xlim(time_range)


    plt.tight_layout()

    # Add legend
    if overlay:
        for i,subset in enumerate(spkpoplst):
            plt.plot(0,0,color=color_list[i%len(color_list)],label=str(subset))
        plt.legend(fontsize=fig_options.fontSize, bbox_to_anchor=(1.04, 1), loc=2, borderaxespad=0.)
        maxLabelLen = min(10,max([len(str(l)) for l in include]))
        plt.subplots_adjust(right=(0.9-0.012*maxLabelLen))


    if fig_options.saveFig:
        if isinstance(fig_options.saveFig, str):
            filename = fig_options.saveFig
        else:
            filename = namespace_id+' '+'histogram.%s' % fig_options.figFormat
        plt.savefig(filename)

    if fig_options.showFig:
        show_figure()

    return fig


def plot_spike_distribution_per_cell (input_path, namespace_id, include = ['eachPop'], time_variable='t', time_range = None, overlay=True, quantity = 'rate', graph_type = 'point', **kwargs):
    ''' 
    Plots distributions of spike rate/count. Returns figure handle.

        - input_path: file with spike data
        - namespace_id: attribute namespace for spike events
        - include (['eachPop'|<population name>]): List of data series to include. 
            (default: ['eachPop'] - expands to the name of each population)
        - time_variable: Name of variable containing spike times (default: 't')
        - time_range ([start:stop]): Time range of spikes shown; if None shows all (default: None)
        - overlay (True|False): Whether to overlay the data lines or plot in separate subplots (default: True)
        - quantity ('rate'|'count'): Quantity of y axis (firing rate in Hz, or spike count) (default: 'rate')
    '''
    fig_options = copy.copy(default_fig_options)
    fig_options.update(kwargs)

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

    spkdata = spikedata.read_spike_events (input_path, include, namespace_id, spike_train_attr_name=time_variable,
                                           time_range=time_range)

    spkpoplst        = spkdata['spkpoplst']
    spkindlst        = spkdata['spkindlst']
    spktlst          = spkdata['spktlst']
    num_cell_spks    = spkdata['num_cell_spks']
    pop_active_cells = spkdata['pop_active_cells']
    tmin             = spkdata['tmin']
    tmax             = spkdata['tmax']

    time_range = [tmin, tmax]
            
    if quantity == 'rate':
        quantityLabel = 'Cell firing rate (Hz)'
    elif quantity == 'count':
        quantityLabel = 'Spike count'
    else:
        print('Invalid quantity value %s' % str(quantity))
        return


    # create fig
    fig, axes = plt.subplots(len(spkpoplst), 1, figsize=fig_options.figSize, sharex=True)

        
    # Plot separate line for each entry in include
    for iplot, subset in enumerate(spkpoplst):

        spkts    = spktlst[iplot]
        spkinds  = spkindlst[iplot]

        u, counts = np.unique(spkinds, return_counts=True)
        sorted_count_idxs = np.argsort(counts)[::-1]
        if quantity == 'rate':
            spkdict = spikedata.make_spike_dict(spkinds, spkts)
            rate_dict = spikedata.spike_rates(spkdict)
            rates = np.asarray([rate_dict[ind] for ind in u if rate_dict[ind] > 0])
            sorted_rate_idxs = np.argsort(rates)[::-1]
            
        color = color_list[iplot%len(color_list)]

        if not overlay:
            label = str(subset)  + ' (%i active)' % (len(pop_active_cells[subset]))
            plt.subplot(len(spkpoplst),1,iplot+1)
            plt.title (label, fontsize=fig_options.fontSize)
            
        if quantity == 'rate':
            x = u[sorted_rate_idxs]
            y = rates[sorted_rate_idxs]
        elif quantity == 'count':
            x = u[sorted_count_idxs]
            y = counts[sorted_count_idxs]
        else:
            raise ValueError('plot_spike_distribution_per_cell: unrecognized quantity: %s' % str(quantity))

        if graph_type == 'point':
            plt.plot(x,y,'o')
            yaxisLabel = quantityLabel
            xaxisLabel = 'Cell index'
        elif graph_type == 'histogram':
            hist_y, bin_edges = np.histogram(np.asarray(y), bins = 40)
            bin_size = bin_edges[1] - bin_edges[0]
            hist_X = bin_edges[:-1]+old_div(bin_size,2)
            b = plt.bar(hist_X, hist_y, width=bin_size)
            yaxisLabel = 'Cell count'
            xaxisLabel = quantityLabel
        else:
            raise ValueError('plot_spike_distribution_per_cell: unrecognized graph type: %s' % str(graph_type))
            
        
        if iplot == 0:
            plt.ylabel(yaxisLabel, fontsize=fig_options.fontSize)
        if iplot == len(spkpoplst)-1:
            plt.xlabel(xaxisLabel, fontsize=fig_options.fontSize)


    if len(spkpoplst) < 5:  # if apply tight_layout with many subplots it inverts the y-axis
        try:
            plt.tight_layout()
        except:
            pass

    # Add legend
    if overlay:
        for i,subset in enumerate(spkpoplst):
            plt.plot(0,0,color=color_list[i%len(color_list)],label=str(subset))
        plt.legend(fontsize=fig_options.fontSize, bbox_to_anchor=(1.04, 1), loc=2, borderaxespad=0.)
        maxLabelLen = min(10,max([len(str(l)) for l in include]))
        plt.subplots_adjust(right=(0.9-0.012*maxLabelLen))

    if fig_options.saveFig:
        if isinstance(fig_options.saveFig, str):
            filename = fig_options.saveFig
        else:
            filename = namespace_id+' '+'distribution.%s' % fig_options.figFormat
        plt.savefig(filename)

    if fig_options.showFig:
        show_figure()

    return fig


def plot_spike_distribution_per_time (input_path, namespace_id, include = ['eachPop'],
                                      time_bin_size = 50.0, binCount = 10,
                                      time_variable='t', time_range = None, 
                                      overlay=True, quantity = 'rate', alpha_fill = 0.2, **kwargs):
    ''' 
    Plots distributions of spike rate/count. Returns figure handle.

        - input_path: file with spike data
        - namespace_id: attribute namespace for spike events
        - include (['eachPop'|<population name>]): List of data series to include. 
            (default: ['eachPop'] - expands to the name of each population)
        - time_variable: Name of variable containing spike times (default: 't')
        - time_range ([start:stop]): Time range of spikes shown; if None shows all (default: None)
        - overlay (True|False): Whether to overlay the data lines or plot in separate subplots (default: True)
        - quantity ('rate'|'count'): Units of x axis (firing rate in Hz, or spike count) (default: 'rate')
    '''

    fig_options = copy.copy(default_fig_options)
    fig_options.update(kwargs)

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

    spkdata = spikedata.read_spike_events (input_path, include, namespace_id, spike_train_attr_name=time_variable,
                                           time_range=time_range)

    spkpoplst        = spkdata['spkpoplst']
    spkindlst        = spkdata['spkindlst']
    spktlst          = spkdata['spktlst']
    num_cell_spks    = spkdata['num_cell_spks']
    pop_active_cells = spkdata['pop_active_cells']
    tmin             = spkdata['tmin']
    tmax             = spkdata['tmax']

    time_range = [tmin, tmax]
            
    # Y-axis label
    if quantity == 'rate':
        xaxisLabel = 'Cell firing rate (Hz)'
    elif quantity == 'count':
        xaxisLabel = 'Spike count'
    else:
        print('Invalid quantity value %s' % str(quantity))
        return

    # create fig
    fig, axes = plt.subplots(len(spkpoplst), 1, figsize=fig_options.figSize, sharex=True)

    # Plot separate line for each entry in include
    for iplot, subset in enumerate(spkpoplst):

        spkts         = spktlst[iplot]
        spkinds       = spkindlst[iplot]
        time_bins     = np.arange(time_range[0], time_range[1], time_bin_size)
        spkdict       = spikedata.make_spike_dict(spkinds, spkts)
        sdf_dict      = spikedata.spike_density_estimate(subset, spkdict, time_bins, return_counts=True)
        max_rate      = np.zeros(time_bins.size-1)
        max_count     = np.zeros(time_bins.size-1)
        bin_dict      = defaultdict(lambda: {'counts': [], 'rates': []})
        for ind, dct in viewitems(sdf_dict):
            rate      = dct['rate']
            count     = dct['count']
            for ibin in range(1, bins.size+1):
                if counts[ibin-1] > 0:
                    d = bin_dict[ibin]
                    d['counts'].append(count[ibin-1])
                    d['rates'].append(rate[ibin-1])
            max_count  = np.maximum(max_count, count)
            max_rate   = np.maximum(max_rate, rate)

        histlst  = []
        for ibin in sorted(bin_dict.keys()):
            d = bin_dict[ibin]
            counts = d['counts']
            rates = d['rates']
            if quantity == 'rate':
                hist_y, bin_edges = np.histogram(np.asarray(rates), bins = binCount, range=(0.0, float(max_rate[ibin-1])))
            else:
                hist_y, bin_edges = np.histogram(np.asarray(counts), bins = binCount, range=(0.0, float(max_count[ibin-1])))
            histlst.append(hist_y)

            
        bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
        
        if not overlay:
            label = str(subset)  + ' (%i active)' % (len(pop_active_cells[subset]))
            plt.subplot(len(spkpoplst),1,iplot+1)
            plt.title (label, fontsize=fig_options.fontSize)

        hist_mean = []
        hist_std  = []
        for i in range(0, binCount):
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
            plt.ylabel('Cell Count', fontsize=fig_options.fontSize)
        if iplot == len(spkpoplst)-1:
            plt.xlabel(xaxisLabel, fontsize=fig_options.fontSize)
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
        plt.legend(fontsize=fig_options.fontSize, bbox_to_anchor=(1.04, 1), loc=2, borderaxespad=0.)
        maxLabelLen = min(10,max([len(str(l)) for l in include]))
        plt.subplots_adjust(right=(0.9-0.012*maxLabelLen))


    if fig_options.saveFig:
        if isinstance(fig_options.saveFig, str):
            filename = fig_options.saveFig
        else:
            filename = namespace_id+' '+'distribution.%s' % fig_options.figFormat
        plt.savefig(filename)

    if fig_options.showFig:
        show_figure()

    return fig


def plot_spatial_information(spike_input_path, spike_namespace_id, trajectory_path, arena_id, trajectory_id,
                             populations=None, position_bin_size=5.0, spike_train_attr_name='t', time_range=None,
                             alpha_fill=0.2, output_file_path=None, plot_dir_path=None, **kwargs):
    """
    Plots distributions of spatial information per cell. Returns figure handle.

    :param spike_input_path: str (path to file)
    :param spike_namespace_id: str
    :param trajectory_path: str (path to file)
    :param arena_id: str
    :param trajectory_id: str
    :param populations: list of str
    :param position_bin_size: float
    :param spike_train_attr_name: str
    :param time_range: list of float
    :param alpha_fill: float
    :param output_file_path: str (path to file)
    :param plot_dir_path: str (path to dir)
    :param kwargs: dict
    :return: :class:'plt.Figure'
    """
    fig_options = copy.copy(default_fig_options)
    fig_options.update(kwargs)

    trajectory = stimulus.read_trajectory(trajectory_path, arena_id, trajectory_id)

    (population_ranges, N) = read_population_ranges(spike_input_path)
    population_names = read_population_names(spike_input_path)

    pop_num_cells = {}
    for k in population_names:
        pop_num_cells[k] = population_ranges[k][1]

    if populations is None:
        populations = list(population_names)

    this_spike_namespace = '%s %s %s' % (spike_namespace_id, arena_id, trajectory_id)

    spkdata = spikedata.read_spike_events(spike_input_path, populations, this_spike_namespace,
                                          spike_train_attr_name=spike_train_attr_name, time_range=time_range)

    spkpoplst = spkdata['spkpoplst']
    spkindlst = spkdata['spkindlst']
    spktlst = spkdata['spktlst']
    num_cell_spks = spkdata['num_cell_spks']
    pop_active_cells = spkdata['pop_active_cells']
    tmin = spkdata['tmin']
    tmax = spkdata['tmax']

    time_range = [tmin, tmax]

    # create fig
    fig, axes = plt.subplots(len(spkpoplst), 1, figsize=fig_options.figSize, sharex=True)

    if output_file_path is not None and not os.path.isfile(output_file_path):
        input_file = h5py.File(spike_input_path, 'r')
        output_file = h5py.File(output_file_path, 'w')
        input_file.copy('/H5Types', output_file)
        input_file.close()
        output_file.close()

    histlst = []
    # Plot separate line for each entry in include
    for iplot, subset in enumerate(spkpoplst):

        spkts = spktlst[iplot]
        spkinds = spkindlst[iplot]
        spkdict = spikedata.make_spike_dict(spkinds, spkts)
        MI_dict = spikedata.spatial_information(subset, trajectory, spkdict, time_range, position_bin_size,
                                                arena_id=arena_id, trajectory_id=trajectory_id,
                                                output_file_path=output_file_path, **kwargs)

        MI_lst = []
        for ind in sorted(MI_dict.keys()):
            MI = MI_dict[ind]
            MI_lst.append(MI)
        del MI_dict

        MI_array = np.asarray(MI_lst, dtype=np.float32)
        del MI_lst

        label = str(subset) + ' (%i active; mean MI %.2f bits)' % (len(pop_active_cells[subset]), np.mean(MI_array))
        plt.subplot(len(spkpoplst), 1, iplot + 1)
        plt.title(label, fontsize=fig_options.fontSize)

        color = color_list[iplot % len(color_list)]

        MI_hist, bin_edges = np.histogram(MI_array, bins='auto')
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        plt.bar(bin_centers, MI_hist, color=color, width=0.3 * (np.mean(np.diff(bin_edges))))

        plt.xticks(fontsize=fig_options.fontSize)

        if iplot == 0:
            plt.ylabel('Cell count', fontsize=fig_options.fontSize)
        if iplot == len(spkpoplst) - 1:
            plt.xlabel('Mutual information [bits]', fontsize=fig_options.fontSize)
        else:
            plt.tick_params(labelbottom='off')
        plt.autoscale(enable=True, axis='both', tight=True)

    if len(spkpoplst) < 5:  # if apply tight_layout with many subplots it inverts the y-axis
        try:
            plt.tight_layout()
        except:
            pass

    # Add legend
    for i, subset in enumerate(spkpoplst):
        plt.plot(0, 0, color=color_list[i % len(color_list)], label=str(subset))
    plt.legend(fontsize=fig_options.fontSize, bbox_to_anchor=(1.04, 1), loc=2, borderaxespad=0.)
    maxLabelLen = min(10, max([len(str(l)) for l in populations]))
    plt.subplots_adjust(right=(0.9 - 0.012 * maxLabelLen))

    if fig_options.saveFig is not None:
        fig_file_path = '%s spatial mutual information %s %s.%s' % \
                        (str(fig_options.saveFig), arena_id, trajectory_id, fig_options.figFormat)
        if plot_dir_path is not None:
            fig_file_path = '%s/%s' % (plot_dir_path, fig_file_path)
        plt.savefig(fig_file_path)

    if fig_options.showFig:
        show_figure()

    return fig


def plot_place_cells(features_path, population, nfields=1, to_plot=100, **kwargs):

    fig_options = copy.copy(default_fig_options)
    fig_options.update(kwargs)

    attr_gen = read_cell_attributes(features_path, population, namespace='Place Selectivity')
    place_cells = {}
    for (gid, cell_attributes) in attr_gen:
        place_cells[gid] = cell_attributes

    cells_to_plot = []
    N = 0
    for gid in place_cells:
        cell_features = place_cells[gid]
        if N == to_plot:
            break
        if cell_features['Num Fields'][0] == nfields:
            N += 1
            nx, ny = cell_features['Nx'][0], cell_features['Ny'][0]
            cells_to_plot.append(cell_features['Rate Map'].reshape(nx, ny))

    axes_dim = int(np.round(np.sqrt(to_plot)))
    fig, axes = plt.subplots(axes_dim, axes_dim)
    for i in range(len(cells_to_plot)):
        img = axes[i%axes_dim, old_div(i,axes_dim)].imshow(cells_to_plot[i], cmap='viridis')
        plt.colorbar(img, ax=axes[i%axes_dim, old_div(i,axes_dim)])
 
    if fig_options.saveFig:
        if isinstance(fig_options.saveFig, str):
            title = fig_options.saveFig
        else:
            title = 'Place-Fields.png'
        plt.savefig(title)

    if fig_options.showFig:
        plt.show()


def plot_place_fields(spike_input_path, spike_namespace_id, trajectory_path, arena_id, trajectory_id, populations=None,
                      bin_size=10.0, min_pf_width=10., spike_train_attr_name='t', time_range=None, alpha_fill=0.2,
                      overlay=False, output_file_path=None, plot_dir_path=None, **kwargs):
    """
    Plots distributions of place fields per cell. Returns figure handle.
    :param spike_input_path: str (path to file)
    :param spike_namespace_id: str
    :param trajectory_path: str (path to file)
    :param arena_id: str
    :param trajectory_id: str
    :param populations: list of str
    :param bin_size: float
    :param min_pf_width: float
    :param spike_train_attr_name: str
    :param time_range: list of float
    :param alpha_fill: float
    :param overlay: bool
    :param output_file_path: str (path to file)
    :param plot_dir_path: str (path to dir)
    :param kwargs: dict
    :return: :class:'plt.Figure'
    """
    fig_options = copy.copy(default_fig_options)
    fig_options.update(kwargs)

    trajectory = stimulus.read_trajectory(trajectory_path, arena_id, trajectory_id)

    (population_ranges, N) = read_population_ranges(spike_input_path)
    population_names = read_population_names(spike_input_path)

    pop_num_cells = {}
    for k in population_names:
        pop_num_cells[k] = population_ranges[k][1]

    if populations is None:
        populations = list(population_names)

    this_spike_namespace = '%s %s %s' % (spike_namespace_id, arena_id, trajectory_id)

    spkdata = spikedata.read_spike_events(spike_input_path, populations, this_spike_namespace,
                                          spike_train_attr_name=spike_train_attr_name, time_range=time_range)

    spkpoplst = spkdata['spkpoplst']
    spkindlst = spkdata['spkindlst']
    spktlst = spkdata['spktlst']
    num_cell_spks = spkdata['num_cell_spks']
    pop_active_cells = spkdata['pop_active_cells']
    tmin = spkdata['tmin']
    tmax = spkdata['tmax']
    
    time_range = [tmin, tmax]
    time_bins  = np.arange(time_range[0], time_range[1], bin_size)
            
    # create fig
    fig = plt.figure(figsize=fig_options.figSize)
    gs  = gridspec.GridSpec(len(spkpoplst), 3, height_ratios=[1 for name in spkpoplst ], width_ratios=[1,4,3])

    if output_file_path is not None and not os.path.isfile(output_file_path):
        input_file = h5py.File(spike_input_path, 'r')
        output_file = h5py.File(output_file_path, 'w')
        input_file.copy('/H5Types', output_file)
        input_file.close()
        output_file.close()

    histlst = []
    # Plot separate line for each entry in include
    for iplot, subset in enumerate(spkpoplst):

        spkts         = spktlst[iplot]
        spkinds       = spkindlst[iplot]
        spkdict       = spikedata.make_spike_dict(spkinds, spkts)

        rate_bin_dict = spikedata.spike_density_estimate(subset, spkdict, time_bins, arena_id=arena_id,
                                                         trajectory_id=trajectory_id,
                                                         output_file_path=output_file_path, **kwargs)
        PF_dict = spikedata.place_fields(subset, bin_size, rate_bin_dict, trajectory, arena_id=arena_id,
                                          trajectory_id=trajectory_id, output_file_path=output_file_path,
                                          min_pf_width=min_pf_width, **kwargs)
        
        PF_count_lst  = []
        PF_infield_rate_lst = []
        PF_field_width_lst = []
        for ind in sorted(PF_dict.keys()):
            PF = PF_dict[ind]
            PF_count_lst.append(PF['pf_count'])
            if PF['pf_count'] > 0:
                PF_field_width_lst.append(PF['pf_mean_width'])
                PF_infield_rate_lst.append(PF['pf_mean_rate'])
                
        del(PF_dict)

        if len(PF_count_lst) > 0:
            PF_count_array = np.concatenate(PF_count_lst)
        else:
            PF_count_array = np.asarray([], dtype=np.float32)
        PF_infield_rate_array = np.concatenate(PF_infield_rate_lst)
        PF_field_width_array = np.concatenate(PF_field_width_lst)
        del(PF_count_lst)
        del(PF_infield_rate_lst)
        del(PF_field_width_lst)
        
        if not overlay:
            label = str(subset) + ' (%i active; mean %.02f place fields)' % \
                    (len(pop_active_cells[subset]), np.mean(PF_count_array))
            plt.subplot(len(spkpoplst),1,iplot+1)
            plt.title(label, fontsize=fig_options.fontSize)
            
        color = color_list[iplot%len(color_list)]

        ax1 = plt.subplot(gs[iplot*3])
        plt.setp([ax1], title='%s Place Fields' % subset)
        
        PF_unique_count = np.unique(PF_count_array)
        if len(PF_unique_count) > 1:
            dmin = np.diff(PF_unique_count).min()
            left_of_first_bin = PF_count_array.min() - float(dmin)/2
            right_of_last_bin = PF_count_array.max() + float(dmin)/2
            bins = np.arange(left_of_first_bin, right_of_last_bin + dmin, dmin)
            PF_count_hist, bin_edges = np.histogram(PF_count_array, bins=bins)
        else:
            PF_count_hist, bin_edges = np.histogram(PF_count_array, bins='auto')
        bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
        ax1.bar(bin_centers, PF_count_hist, color=color, width=0.3*(np.mean(np.diff(bin_edges))))
        ax1.set_xticks(bin_centers)
        ax1.tick_params(axis="x", labelsize=fig_options.fontSize)
        ax1.tick_params(axis="y", labelsize=fig_options.fontSize)

        ax2 = plt.subplot(gs[iplot*3 + 1])
        PF_field_width_hist, bin_edges = np.histogram(PF_field_width_array)
        bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
        ax2.bar(bin_centers, PF_field_width_hist, color=color, width=0.3*(np.mean(np.diff(bin_edges))))
        ax2.set_xticks(bin_centers)
        ax2.xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
        ax2.tick_params(axis="x", labelsize=fig_options.fontSize)
        ax2.tick_params(axis="y", labelsize=fig_options.fontSize)
        
        ax3 = plt.subplot(gs[iplot*3 + 2])
        PF_infield_rate_hist, bin_edges = np.histogram(PF_infield_rate_array)
        bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
        ax3.bar(bin_centers, PF_infield_rate_hist, color=color, width=0.3*(np.mean(np.diff(bin_edges))))
        ax3.set_xticks(bin_centers)
        ax3.xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
        ax3.tick_params(axis="x", labelsize=fig_options.fontSize)
        ax3.tick_params(axis="y", labelsize=fig_options.fontSize)
        
        if iplot == 0:
            ax1.set_ylabel('Cell Index', fontsize=fig_options.fontSize)
        if iplot == len(spkpoplst)-1:
            ax1.set_xlabel('Number of place fields', fontsize=fig_options.fontSize)
            ax2.set_xlabel('Mean field width [cm]', fontsize=fig_options.fontSize)
            ax3.set_xlabel('In-field mean firing rate [Hz]', fontsize=fig_options.fontSize)

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
        plt.legend(fontsize=fig_options.fontSize, bbox_to_anchor=(1.04, 1), loc=2, borderaxespad=0.)
        maxLabelLen = min(10,max([len(str(l)) for l in populations]))
        plt.subplots_adjust(right=(0.9-0.012*maxLabelLen))

    if fig_options.saveFig is not None:
        save_figure('%s place fields %s %s' % (str(fig_options.saveFig), arena_id, trajectory_id), **fig_options())

    if fig_options.showFig:
        show_figure()

    return fig



def plot_spike_PSD (input_path, namespace_id, include = ['eachPop'], time_range = None, time_variable='t', 
                    bin_size = 1., window_size = 1024, smooth = 0, frequency_range=(0, 100.), overlap=0.5,
                    overlay = True, **kwargs):
    ''' 
    Plots spike train power spectral density (PSD). Returns figure handle.
        - input_path: file with spike data
        - namespace_id: attribute namespace for spike events
        - include (['eachPop'|<population name>]): List of data series to include. 
            (default: ['eachPop'] - expands to the name of each population)
        - time_range ([start:stop]): Time range of spikes shown; if None shows all (default: None)
        - time_variable: Name of variable containing spike times (default: 't')
        - bin_size (int): Size in ms of each bin (default: 1)
        - Fs (float): sampling frequency
        - nperseg (int): Length of each segment. 
        - overlay (True|False): Whether to overlay the data lines or plot in separate subplots (default: True)

    '''

    fig_options = copy.copy(default_fig_options)
    fig_options.update(kwargs)

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

    spkdata = spikedata.read_spike_events (input_path, include, namespace_id, spike_train_attr_name=time_variable, 
                                           time_range=time_range)

    spkpoplst        = spkdata['spkpoplst']
    spkindlst        = spkdata['spkindlst']
    spktlst          = spkdata['spktlst']
    num_cell_spks    = spkdata['num_cell_spks']
    pop_active_cells = spkdata['pop_active_cells']
    tmin             = spkdata['tmin']
    tmax             = spkdata['tmax']

    time_range = [tmin, tmax]

    # create fig
    fig, ax1 = plt.subplots(figsize=fig_options.figSize)

    time_bins  = np.arange(time_range[0], time_range[1], bin_size)
    nperseg    = window_size
    win        = signal.get_window('hanning', nperseg)
 
    psds = []
    # Plot separate line for each entry in include
    for iplot, (subset, spkinds, spkts) in enumerate(zip(spkpoplst, spkindlst, spktlst)):

        spk_count, bin_edges = np.histogram(spkts, bins=time_bins)

        if smooth:
            # smoothen firing rate histogram
            hsignal = signal.savgol_filter(spk_count, window_length=5, polyorder=smooth, mode='nearest')
        else:
            hsignal = spk_count

        Fs = 1000. / bin_size

        noverlap = int(overlap * nperseg)
        freqs, psd = signal.welch(hsignal, fs=Fs, scaling='density', nperseg=nperseg, noverlap=noverlap, return_onesided=True)
        freqinds = np.where((freqs >= frequency_range[0]) & (freqs <= frequency_range[1]))

        freqs = freqs[freqinds]
        psd = psd[freqinds]

        if np.all(psd):
            psd = 10. * np.log10(psd)

        min_freq = np.min(freqs)
        max_freq = np.max(freqs)

        peak_index = np.where(psd == np.max(psd))[0]
        
        color = color_list[iplot%len(color_list)]

        if not overlay:
            label = str(subset)
            plt.subplot(len(spkpoplst),1,iplot+1)
            plt.title ('%s (peak: %.3g Hz)' % (label, freqs[peak_index]), fontsize=fig_options.fontSize)

        plt.plot(freqs, psd, linewidth=fig_options.lw, color=color)
        
        if iplot == 0:
            plt.ylabel('Power Spectral Density (dB/Hz)', fontsize=fig_options.fontSize) # add yaxis in opposite side
        if iplot == len(spkpoplst)-1:
            plt.xlabel('Frequency (Hz)', fontsize=fig_options.fontSize)
        plt.xlim([0, np.max(freqs)])

        psds.append(psd)
        
    if len(spkpoplst) < 5:  # if apply tight_layout with many subplots it inverts the y-axis
        try:
            plt.tight_layout()
        except:
            pass

    # save figure
    if fig_options.saveFig:
        if isinstance(fig_options.saveFig, str):
            filename = fig_options.saveFig
        else:
            filename = namespace_id+' '+'rate PSD.%s' % fig_options.figFormat
        plt.savefig(filename)

    # show fig 
    if fig_options.showFig:
        show_figure()

    return fig, psds


def plot_selectivity_metrics (env, coords_path, features_path, distances_namespace, population='MPP',
                              selectivity_type = 'grid', bin_size=250., metric='spacing', normed=False,
                              graph_type = 'histogram2d', **kwargs):

    """
    :param env:
    :param coords_path:
    :param features_path:
    :param distances_namespace:
    :param population:
    :param selectivity_type:
    :param bin_size:
    :param metric:
    :param normed:
    :param graph_type:
    """
    fig_options = copy.copy(default_fig_options)
    fig_options.update(kwargs)

    if selectivity_type == 'grid':
        input_selectivity_namespace = 'Grid Selectivity'
        selectivity_type_label = 'grid input'
    elif selectivity_type == 'place':
        input_selectivity_namespace = 'Place Selectivity'
        selectivity_type_label = 'spatial input'
    if metric == 'spacing' and selectivity_type == 'grid':
        attribute = 'Grid Spacing'
        cbar_label = 'Mean grid spacing (cm)'
        feature_label = 'grid spacing'
    elif metric == 'spacing' and selectivity_type == 'place':
        attribute = 'Field Width'
        cbar_label = 'Mean field width (cm)'
        feature_label = 'spatial field width'
    if metric == 'num-fields':
        if selectivity_type == 'grid':
            return
        elif selectivity_type == 'place':
            attribute = 'Num Fields'
            cbar_label = 'Mean number of spatial fields'
            feature_label = 'number of spatial fields'
    if metric == 'orientation' and selectivity_type == 'grid':
        attribute = 'Grid Orientation'
        cbar_label = 'Mean grid orientation (rad)'
        feature_label = 'grid orientation'
    elif metric == 'orientation' and selectivity_type == 'place':
        return 
    
    attr_gen = read_cell_attributes(features_path, population, input_selectivity_namespace)
    attr_dict = {}
    for (gid, features_dict) in attr_gen:
        attr_dict[gid] = features_dict[attribute]
    del attr_gen
    present_gids = list(attr_dict.keys())

    distances = read_cell_attributes(coords_path, population, distances_namespace)
    soma_distances = { k: (v['U Distance'][0], v['V Distance'][0]) for (k,v) in distances}
    del distances

    distance_U, distance_V = [], []
    attr_lst = []
    for gid in present_gids:
        distance_U.append(soma_distances[gid][0])
        distance_V.append(soma_distances[gid][1])
        attr_mean = np.mean(attr_dict[gid])
        attr_lst.append(attr_mean)

    distance_U = np.asarray(distance_U, dtype='float32')
    distance_V = np.asarray(distance_V, dtype='float32')

    distance_x_min = np.min(distance_U)
    distance_x_max = np.max(distance_U)
    distance_y_min = np.min(distance_V)
    distance_y_max = np.max(distance_V)
 
    ((x_min, x_max), (y_min, y_max)) = measure_distance_extents(env)

    dx = int(old_div((distance_x_max - distance_x_min), bin_size))
    dy = int(old_div((distance_y_max - distance_y_min), bin_size))

    fig = plt.figure(figsize=plt.figaspect(1.) * 2.)
    ax = plt.gca()
    ax.axis([x_min, x_max, y_min, y_max])
        
    (H1, xedges, yedges) = np.histogram2d(distance_U, distance_V, bins=[dx, dy], weights=attr_lst, normed=normed)
    (H2, xedges, yedges) = np.histogram2d(distance_U, distance_V, bins=[dx, dy])
    zeros = np.where(H2 == 0.0)
    H = np.zeros(H1.shape)
    nz = np.where(H2 > 0.0)
    H[nz] = np.divide(H1[nz], H2[nz])
    H[zeros] = None
    if normed:
        H[nz] = np.divide(H[nz], np.max(H[nz]))

    X, Y = np.meshgrid(xedges, yedges)
    pcm = ax.pcolormesh(X, Y, H.T, cmap='jet')
    cbar = fig.colorbar(pcm, ax=ax, shrink=0.48, aspect=20)
    cbar.set_label(cbar_label, rotation=270., labelpad=20.)
    
    ax.set_ylabel('Transverse distance (um)', fontsize=fig_options.fontSize)
    ax.set_xlabel('Longitudinal distance (um)\n\nBin size: %i x %i um' % (bin_size, bin_size), fontsize=fig_options.fontSize)
    ax.set_title('%s %s: %s' % (population, selectivity_type_label, feature_label), fontsize=fig_options.fontSize)
    ax.set_aspect('equal')
    
    if fig_options.saveFig:
        if isinstance(fig_options.saveFig, str):
            filename = fig_options.saveFig
        else:
            filename = '%s-%s-%s.%s' % (population, selectivity_type, metric, fig_options.figFormat)
        plt.savefig(filename)

    if fig_options.showFig:
        show_figure()


def plot_stimulus_rate(input_path, namespace_id, population, arena_id=None, trajectory_id=None, **kwargs):
    """

        - input_path: file with stimulus data
        - namespace_id: attribute namespace for stimulus
        - population: str name of a valid cell population
    """

    fig_options = copy.copy(default_fig_options)
    fig_options.update(kwargs)

    if trajectory_id is not None and arena_id is not None:
        trajectory = stimulus.read_trajectory(input_path, arena_id, trajectory_id)
        (_, _, _, t)  = trajectory
    else:
        t = None

    M = 0
    if (arena_id is None):
        ns = namespace_id
    else:
        ns = '%s %s' % (namespace_id, arena_id)
        
    logger.info('Reading feature data from namespace %s for population %s...' % (ns, population ))
    fig, axes = plt.subplots(2, 5)
    for module in range(0, 10):
        rate_lst = []
        for (gid, rate) in stimulus.read_feature(input_path, ns, population, module=module):
            if np.max(rate) > 0.:
                rate_lst.append(rate)
        col = module % 5
        row = old_div(module, 5)
        M = max(M, len(rate_lst))
        N = len(rate_lst)
        rate_matrix = np.matrix(rate_lst)
        del(rate_lst)

        if t is None:
            extent=[0, len(rate), 0, N]
        else:
            extent=[t[0], t[-1], 0, N]
        title = 'Module: %i' % module
        axes[row][col].set_title(title, fontsize=fig_options.fontSize)
        img = axes[row][col].imshow(rate_matrix, origin='upper', aspect='auto', cmap=cm.coolwarm,
                                    extent=extent)
        #axes[row][col].set_xlim([extent[0], extent[1]])
        #axes[row][col].set_ylim(-1, N+1)
        if col == 0:
            axes[row][col].set_ylabel('Cell index', fontsize=fig_options.fontSize)
        if row == 1:
            axes[row][col].set_xlabel('Time (ms)', fontsize=fig_options.fontSize)

    cax, kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
    cbar = plt.colorbar(img, cax=cax, **kw)
    cbar.set_label('Firing rate (Hz)', rotation=270., labelpad=20.)

    fig.suptitle(population, fontsize=fig_options.fontSize)

    plt.show()
    
    # save figure
    if fig_options.saveFig:
        if isinstance(fig_options.saveFig, str):
            filename = fig_options.saveFig
        else:
            filename = namespace_id+'_'+'ratemap.%s' % fig_options.figFormat
        plt.savefig(filename)

    # show fig
    if fig_options.showFig:
        show_figure()


def plot_stimulus_spatial_rate_map(env, input_path, coords_path, arena_id, trajectory_id, stimulus_namespace, distances_namespace, include, bin_size = 100., from_spikes = True, **kwargs):
    """
        - input_path: path to file with stimulus data (str)
        - coords_path: path to file with cell position coordinates (str)
        - trajectory_id: identifier for spatial trajectory (int)
        - stimulus_namespace: attribute namespace for stimulus (str)
        - distances_namespace: attribute namespace for longitudinal and transverse distances (str)
        - include (['eachPop'|<population name>]): List of data series to include. 
            (default: ['eachPop'] - expands to the name of each population)
        - bin_size: length of square edge for 2D histogram (float)
        - fromSpikes: bool; whether to compute rate maps from stored spikes, or from target function
    """

    fig_options = copy.copy(default_fig_options)
    fig_options.update(kwargs)

    _, _, _, t = stimulus.read_trajectory(input_path, arena_id, trajectory_id)
    dt = float(t[1] - t[0]) / 1000. # ms -> s
    T  = float(t[-1] - t[0]) / 1000. # ms -> s

    if (arena_id is None) or (trajectory_id is None):
        ns = stimulus_namespace
    else:
        ns = '%s %s %s' % (stimulus_namespace, arena_id, trajectory_id)

    for iplot, population in enumerate(include):
   
        spiketrain_dict = {}
        logger.info('Reading stimulus data for population %s...' % population) 

        for (gid, rate, spiketrain, _) in stimulus.read_stimulus(input_path, ns, population): 
            if from_spikes:
                spiketrain_dict[gid] = len(spiketrain)
            else:
                spiketrain_dict[gid] = np.mean(rate) #np.sum(rate * dt)

        present_gids = list(spiketrain_dict.keys())
        
        logger.info('read rates (%i elements)' % len(present_gids))

        distances = read_cell_attributes(coords_path, population, namespace=distances_namespace)
        soma_distances = { k: (v['U Distance'][0], v['V Distance'][0]) for (k,v) in distances }
        del distances

        distance_U, distance_V = [], []
        spikes    = []
        for gid in present_gids:
            distance_U.append(soma_distances[gid][0])
            distance_V.append(soma_distances[gid][1])
            spikes.append(spiketrain_dict[gid])
        distance_U = np.asarray(distance_U, dtype='float32')
        distance_V = np.asarray(distance_V, dtype='float32')
        
        distance_x_min = np.min(distance_U)
        distance_x_max = np.max(distance_U)
        distance_y_min = np.min(distance_V)
        distance_y_max = np.max(distance_V)
        
        logger.info('read distances (%i elements)' % len(soma_distances.keys()))

        ((x_min, x_max), (y_min, y_max)) = measure_distance_extents(env)

        dx = int(old_div((distance_x_max - distance_x_min), bin_size))
        dy = int(old_div((distance_y_max - distance_y_min), bin_size))

        (H1, xedges, yedges)  = np.histogram2d(distance_U, distance_V, bins=[dx, dy], weights=spikes)
        (H2, xedges, yedges)  = np.histogram2d(distance_U, distance_V, bins=[dx, dy])
        nz = np.where(H2 > 0.0)
        zeros = np.where(H2 == 0.0)

        H = np.zeros_like(H1)
        H[nz] = np.divide(H1[nz], H2[nz])
        if from_spikes:
            H = np.divide(H, T)
        H[zeros] = None

        X, Y = np.meshgrid(xedges, yedges)
        fig = plt.figure(figsize=plt.figaspect(1.) * 2.)
        axes = plt.gca()
        pcm = axes.pcolormesh(X, Y, H.T)
        axes.axis([x_min, x_max, y_min, y_max])
        axes.set_aspect('equal')

        if from_spikes:
            title = '%s input firing rate\nTrajectory: %s %s' % (population, arena_id, trajectory_id)
        else:
            title = '%s expected input firing rate' % population
        axes.set_title(title, fontsize=fig_options.fontSize)
        axes.set_xlabel('Longitudinal distance (um)\n\nBin size: %i x %i um' % (bin_size, bin_size), fontsize=fig_options.fontSize)
        axes.set_ylabel('Transverse distance (um)', fontsize=fig_options.fontSize)
        cbar = fig.colorbar(pcm, ax=axes, shrink=0.48, aspect=20)
        cbar.set_label('Mean input firing rate (Hz)', rotation=270., labelpad=20.)

        # save figure
        if fig_options.saveFig:
            if isinstance(fig_options.saveFig, str):
                filename = fig_options.saveFig
            else:
                filename = '%s %s spatial ratemap.%s' % (population, stimulus_namespace, fig_options.figFormat)
            plt.savefig(filename)

        # show fig
        if fig_options.showFig:
            show_figure()


def plot_spike_histogram_autocorr (input_path, namespace_id, include = ['eachPop'], time_range = None, time_variable='t',
                                   bin_size = 25, graph_type = 'matrix', lag=1, max_cells = None, xlim = None, 
                                   marker = '|', **kwargs):
    """
    Plot of spike histogram correlations. Returns the figure handle.

    input_path: file with spike data
    namespace_id: attribute namespace for spike events
    time_range ([start:stop]): Time range of spikes shown; if None shows all (default: None)
    time_variable: Name of variable containing spike times (default: 't')
    bin_size (int): Size of bin in ms to use for spike count and rate computations (default: 5)
    marker (char): Marker for each spike (default: '|')
    """

    fig_options = copy.copy(default_fig_options)
    fig_options.update(kwargs)

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

    spkdata = spikedata.read_spike_events (input_path, include, namespace_id, spike_train_attr_name=time_variable,
                                           time_range=time_range)

    spkpoplst        = spkdata['spkpoplst']
    spkindlst        = spkdata['spkindlst']
    spktlst          = spkdata['spktlst']
    num_cell_spks    = spkdata['num_cell_spks']
    pop_active_cells = spkdata['pop_active_cells']
    tmin             = spkdata['tmin']
    tmax             = spkdata['tmax']
    
    corr_dict = spikedata.histogram_autocorrelation(spkdata, bin_size=bin_size, max_elems=max_cells, lag=lag)
        
    # Plot spikes
    fig, axes = plt.subplots(len(spkpoplst), 1, figsize=fig_options.figSize, sharex=True)

    X_max = None
    X_min = None
    for (iplot, subset) in enumerate(spkpoplst):

        pop_corr = corr_dict[subset]
        
        if len(spkpoplst) > 1:
            axes[iplot].set_title (str(subset), fontsize=fig_options.fontSize)
        else:
            axes.set_title (str(subset), fontsize=fig_options.fontSize)

        if graph_type == 'matrix':
            im = axes[iplot].imshow(pop_corr, origin='lower', aspect='auto', interpolation='none', cmap=fig_options['colormap'])
            cbar = plt.colorbar(im)
            cbar.ax.set_ylabel('Correlation Coefficient', fontsize=fig_options.fontSize)
        elif graph_type == 'histogram':
            hist_y, bin_edges = np.histogram(pop_corr, bins = 100)
            corr_bin_size = bin_edges[1] - bin_edges[0]
            hist_X = bin_edges[:-1]+old_div(corr_bin_size,2)
            color = color_list[iplot%len(color_list)]
            if len(spkpoplst) > 1:
                b = axes[iplot].bar(hist_X, hist_y, width = corr_bin_size, color = color)
            else:
                b = axes.bar(hist_X, hist_y, width = corr_bin_size, color = color)
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
            im = axes[iplot].imshow(pop_corr, origin='lower', aspect='auto', interpolation='none', cmap=fig_options['colormap'])
            cbar = plt.colorbar(im)
            cbar.ax.set_ylabel('Correlation Coefficient', fontsize=fig_options.fontSize)

        if graph_type == 'matrix':
            if iplot == 0:
                axes[iplot].ylabel('Relative Cell Index', fontsize=fig_options.fontSize)
            if iplot == len(spkpoplst)-1:
                axes[iplot].xlabel('Relative Cell Index', fontsize=fig_options.fontSize)

                
    # show fig 
    if fig_options.showFig:
        show_figure()
    
    return fig


## Plot spike cross-correlation
def plot_spike_histogram_corr (input_path, namespace_id, include = ['eachPop'], time_range = None, time_variable='t', bin_size = 25, graph_type = 'matrix', max_cells = None, marker = '|', **kwargs): 
    ''' 
    Plot of spike histogram correlations. Returns the figure handle.

    input_path: file with spike data
    namespace_id: attribute namespace for spike events
    time_range ([start:stop]): Time range of spikes shown; if None shows all (default: None)
    time_variable: Name of variable containing spike times (default: 't')
    bin_size (int): Size of bin in ms to use for spike count and rate computations (default: 5)
    marker (char): Marker for each spike (default: '|')
    '''

    fig_options = copy.copy(default_fig_options)
    fig_options.update(kwargs)

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

    spkdata = spikedata.read_spike_events (input_path, include, namespace_id, spike_train_attr_name=time_variable,
                                           time_range=time_range)

    spkpoplst        = spkdata['spkpoplst']
    spkindlst        = spkdata['spkindlst']
    spktlst          = spkdata['spktlst']
    num_cell_spks    = spkdata['num_cell_spks']
    pop_active_cells = spkdata['pop_active_cells']
    tmin             = spkdata['tmin']
    tmax             = spkdata['tmax']
    
    corr_dict = spikedata.histogram_correlation(spkdata, bin_size=bin_size, max_elems=max_cells)
        
    # Plot spikes
    fig, axes = plt.subplots(len(spkpoplst), 1, figsize=fig_options.figSize, sharex=True)

    X_max = None
    X_min = None
    for (iplot, subset) in enumerate(spkpoplst):

        pop_corr = corr_dict[subset]

        if len(spkpoplst) > 1:
            axes[iplot].set_title (str(subset), fontsize=fig_options.fontSize)
        else:
            axes.set_title (str(subset), fontsize=fig_options.fontSize)
            
        if graph_type == 'matrix':
            im = axes[iplot].imshow(pop_corr, origin='lower', aspect='auto', interpolation='none', cmap=fig_options['colormap'])
            cbar = plt.colorbar(im)
            cbar.ax.set_ylabel('Correlation Coefficient', fontsize=fig_options.fontSize)
        elif graph_type == 'histogram':
            np.fill_diagonal(pop_corr, 0.)
            mean_corr = np.apply_along_axis(lambda y: np.mean(y), 1, pop_corr)
            hist_y, bin_edges = np.histogram(mean_corr, bins = 100)
            corr_bin_size = bin_edges[1] - bin_edges[0]
            hist_X = bin_edges[:-1]+old_div(corr_bin_size,2)
            color = color_list[iplot%len(color_list)]
            if len(spkpoplst) > 1:
                b = axes[iplot].bar(hist_X, hist_y, width = corr_bin_size, color = color)
            else:
                b = axes.bar(hist_X, hist_y, width = corr_bin_size, color = color)
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
            im = axes[iplot].imshow(pop_corr, origin='lower', aspect='auto', interpolation='none', cmap=fig_options['colormap'])
            cbar = plt.colorbar(im)
            cbar.ax.set_ylabel('Correlation Coefficient', fontsize=fig_options.fontSize)

        if graph_type == 'matrix':
            if iplot == 0: 
                axes[iplot].ylabel('Relative Cell Index', fontsize=fig_options.fontSize)
            if iplot == len(spkpoplst)-1:
                axes[iplot].xlabel('Relative Cell Index', fontsize=fig_options.fontSize)

                
    # show fig 
    if fig_options.showFig:
        show_figure()
    
    return fig


def plot_synaptic_attribute_distribution(cell, env, syn_name, param_name, filters=None, from_mech_attrs=True,
                                         from_target_attrs=False, export=None, overwrite=False, description=None,
                                         scale_factor=1., param_label=None, ylabel='Peak conductance', yunits='uS',
                                         svg_title=None, show=True, sec_types=None, data_dir='data'):
    """
    Plots values of synapse attributes found in point processes and NetCons of a Hoc Cell. No simulation is required;
    this method just takes a fully specified cell and plots the relationship between distance and the specified synaptic
    parameter.

    Note: exported files can be plotted using plot_syn_attr_from_file; give syn_name as the input parameter instead of
    mech_name.

    :param cell: :class:'BiophysCell'
    :param env: :class:'Env'
    :param syn_name: str
    :param param_name: str
    :param filters: dict (ex. syn_indexes, layers, syn_types) with str values
    :param from_mech_attrs: bool
    :param from_target_attrs: bool
    :param export: str (name of hdf5 file for export)
    :param overwrite: bool (whether to overwrite or append to potentially existing hdf5 file)
    :param description: str (to be saved in hdf5 file as a descriptor of this session)
    :param scale_factor: float
    :param param_label: str
    :param ylabel: str
    :param yunits: str
    :param svg_title: str
    :param show: bool (whether to show the plot, or simply save the hdf5 file)
    :param sec_types: list or str
    :param data_dir: str
    :return:
    """
    if svg_title is not None:
        remember_font_size = mpl.rcParams['font.size']
        mpl.rcParams['font.size'] = 20
    if sec_types is None or (isinstance(sec_types, str) and sec_types == 'dend'):
        sec_types = ['basal', 'trunk', 'apical', 'tuft']
    elif isinstance(sec_types, str) and sec_types == 'all':
        sec_types = default_ordered_sec_types
    elif not all(sec_type in default_ordered_sec_types for sec_type in sec_types):
        raise ValueError('plot_synaptic_attribute_distribution: unrecognized sec_types: %s' % str(sec_types))
    sec_types_list = [sec_type for sec_type in sec_types if sec_type in cell.nodes and len(cell.nodes[sec_type]) > 0]
    attr_types = []
    if from_mech_attrs:
        attr_types.append('mech_attrs')
    if from_target_attrs:
        attr_types.append('target_attrs')
    if len(attr_types) == 0:
        raise Exception('plot_synaptic_attribute_distribution: both from_mech_attrs and from_target_attrs cannot be '
                        'False')
    distances = {attr_type: defaultdict(list) for attr_type in attr_types}
    attr_vals = {attr_type: defaultdict(list) for attr_type in attr_types}
    num_colors = 10
    color_x = np.linspace(0., 1., num_colors)
    colors = [cm.Set1(x) for x in color_x]
    syn_attrs = env.synapse_attributes
    gid = cell.gid
    for sec_type in sec_types_list:
        if len(cell.nodes[sec_type]) > 0:
            for node in cell.nodes[sec_type]:
                if filters is not None:
                    converted_filters = get_syn_filter_dict(env, filters, convert=True)
                else:
                    converted_filters = {}
                syns = syn_attrs.filter_synapses(gid, syn_sections=[node.index], **converted_filters)
                for syn_id, syn in viewitems(syns):
                    # TODO: figure out what to do with spine synapses that are not inserted into a branch node
                    if from_mech_attrs:
                        this_param_val = syn_attrs.get_mech_attrs(gid, syn_id, syn_name, throw_error=False)
                        if this_param_val is not None and param_name in this_param_val:
                            attr_vals['mech_attrs'][sec_type].append(this_param_val[param_name] * scale_factor)
                            syn_loc = syn.syn_loc
                            distances['mech_attrs'][sec_type].append(
                                get_distance_to_node(cell, cell.tree.root, node, syn_loc))
                            if sec_type == 'basal':
                                distances['mech_attrs'][sec_type][-1] *= -1
                    if from_target_attrs:
                        if syn_attrs.has_netcon(cell.gid, syn_id, syn_name):
                            this_nc = syn_attrs.get_netcon(cell.gid, syn_id, syn_name)
                            attr_vals['target_attrs'][sec_type].append(
                                get_syn_mech_param(syn_name, syn_attrs.syn_param_rules, param_name,
                                                   mech_names=syn_attrs.syn_mech_names, nc=this_nc) * scale_factor)
                            syn_loc = syn.syn_loc
                            distances['target_attrs'][sec_type].append(
                                get_distance_to_node(cell, cell.tree.root, node, syn_loc))
                            if sec_type == 'basal':
                                distances['target_attrs'][sec_type][-1] *= -1

    if export is not None:
        export_file_path = data_dir + '/' + export
        if overwrite:
            if os.path.isfile(export_file_path):
                os.remove(export_file_path)

    for attr_type in attr_types:
        if len(attr_vals[attr_type]) == 0 and export is not None:
            print('Not exporting to %s; mechanism: %s %s parameter: %s not found in any sec_type' % \
                  (export, syn_name, attr_type, param_name))
            # return
    xmax0 = 0.1
    xmin0 = 0.
    maxval, minval = 0., 0.
    fig, axarr = plt.subplots(ncols=len(attr_types), sharey=True)
    for i, attr_type in enumerate(attr_types):
        if len(attr_types) == 1:
            axes = axarr
        else:
            axes = axarr[i]
        for j, sec_type in enumerate(attr_vals[attr_type]):
            if len(attr_vals[attr_type][sec_type]) != 0:
                axes.scatter(distances[attr_type][sec_type], attr_vals[attr_type][sec_type], color=colors[j],
                             label=sec_type, alpha=0.5, s=10.)
                if maxval is None:
                    maxval = max(attr_vals[attr_type][sec_type])
                else:
                    maxval = max(maxval, max(attr_vals[attr_type][sec_type]))
                if minval is None:
                    minval = min(attr_vals[attr_type][sec_type])
                else:
                    minval = min(minval, min(attr_vals[attr_type][sec_type]))
                xmax0 = max(xmax0, max(distances[attr_type][sec_type]))
                xmin0 = min(xmin0, min(distances[attr_type][sec_type]))
        axes.legend(loc='best', scatterpoints=1, frameon=False, framealpha=0.5, fontsize=mpl.rcParams['font.size'])
    xmin = xmin0 - 0.01 * (xmax0 - xmin0)
    xmax = xmax0 + 0.01 * (xmax0 - xmin0)
    for i, attr_type in enumerate(attr_types):
        if len(attr_types) == 1:
            axes = axarr
        else:
            axes = axarr[i]
        axes.set_xlabel('Distance to soma (um)')
        axes.set_xlim(xmin, xmax)
        axes.set_ylabel(ylabel + ' (' + yunits + ')')
        if (maxval is not None) and (minval is not None):
            buffer = 0.01 * (maxval - minval)
            axes.set_ylim(minval - buffer, maxval + buffer)
        axes.set_title(attr_type, fontsize=mpl.rcParams['font.size'])
    clean_axes(axarr)
    fig.tight_layout()
    if param_label is not None:
        fig.suptitle(param_label, fontsize=mpl.rcParams['font.size'])
    else:
        syn_mech_name = syn_attrs.syn_mech_names[syn_name]
        fig.suptitle('%s; %s; %s' % (syn_name, syn_mech_name, param_name), fontsize=mpl.rcParams['font.size'])
    if not svg_title is None:
        if param_label is not None:
            svg_title = svg_title + ' - ' + param_label + '.svg'
        else:
            svg_title = svg_title + ' - ' + syn_name + '_' + param_name + ' distribution.svg'
        fig.set_size_inches(5.27, 4.37)
        fig.savefig(data_dir + svg_title, format='svg', transparent=True)
    if show:
        plt.show()
    plt.close()
    if svg_title is not None:
        mpl.rcParams['font.size'] = remember_font_size

    if export is not None:
        f = h5py.File(export_file_path, 'a')
        if 'mech_file_path' in f.attrs:
            if not (f.attrs['mech_file_path'] == '{}'.format(cell.mech_file_path)):
                raise Exception('Specified mechanism filepath {} does not match the mechanism filepath '
                                'of the cell {}'.format(f.attrs['mech_file_path'], cell.mech_file_path))
        else:
            f.attrs['mech_file_path'] = '{}'.format(cell.mech_file_path)
        filetype = 'plot_syn_param'
        if filetype not in f:
            f.create_group(filetype)
        if not f[filetype].attrs.__contains__('mech_attrs'):
            f[filetype].attrs.create('mech_attrs', False)
        if not f[filetype].attrs.__contains__('target_attrs'):
            f[filetype].attrs.create('target_attrs', False)
        if from_mech_attrs and f[filetype].attrs['mech_attrs'] == False:
            f[filetype].attrs['mech_attrs'] = True
        if from_target_attrs and f[filetype].attrs['target_attrs'] == False:
            f[filetype].attrs['target_attrs'] = True
        if len(f[filetype]) == 0:
            session_id = '0'
        else:
            session_id = str(len(f[filetype]))
        f[filetype].create_group(session_id)
        if description is not None:
            f[filetype][session_id].attrs['description'] = description
        f[filetype][session_id].create_group(syn_name)
        f[filetype][session_id][syn_name].create_group(param_name)
        if param_label is not None:
            f[filetype][session_id][syn_name][param_name].attrs['param_label'] = param_label
        f[filetype][session_id][syn_name][param_name].attrs['gid'] = cell.gid
        if svg_title is not None:
            f[filetype][session_id][syn_name][param_name].attrs['svg_title'] = svg_title
        for attr_type in attr_types:
            f[filetype][session_id][syn_name][param_name].create_group(attr_type)
            for sec_type in attr_vals[attr_type]:
                f[filetype][session_id][syn_name][param_name][attr_type].create_group(sec_type)
                f[filetype][session_id][syn_name][param_name][attr_type][sec_type].create_dataset('values',
                                                                                    data=attr_vals[attr_type][sec_type])
                f[filetype][session_id][syn_name][param_name][attr_type][sec_type].create_dataset('distances',
                                                                                    data=distances[attr_type][sec_type])
        f.close()


def plot_syn_attr_from_file(syn_name, param_name, filename, descriptions=None, param_label=None,
                            ylabel='Conductance density', yunits='pS/um2', svg_title=None, data_dir='data'):
    """
    Takes in a list of files, and superimposes plots of distance vs. the provided mechanism parameter for all sec_types
    found in each file.
    :param syn_name: str
    :param param_name: str
    :param filename: str
    :param descriptions: list of str (descriptions of each session). If None, then plot all session_ids
    :param param_label: str
    :param ylabel: str
    :param yunits: str
    :param svg_title: str
    :param data_dir: str (path)
    """
    if svg_title is not None:
        remember_font_size = mpl.rcParams['font.size']
        mpl.rcParams['font.size'] = 20
    markers = mlines.Line2D.filled_markers
    marker_dict = {}
    num_colors = 10
    color_x = np.linspace(0., 1., num_colors)
    colors = [cm.Set1(x) for x in color_x]
    max_param_val, min_param_val = 0., 0.
    max_dist, min_dist = 0.1, 0.
    file_path = data_dir + '/' + filename
    found = False
    if os.path.isfile(file_path):
        with h5py.File(file_path, 'r') as f:
            filetype = 'plot_syn_param'
            if filetype not in f:
                raise Exception('The file {} has the incorrect filetype; it is not plot_syn_param'.format(file))
            attr_types = []
            if f[filetype].attrs['mech_attrs']:
                attr_types.append('mech_attrs')
            if f[filetype].attrs['target_attrs']:
                attr_types.append('target_attrs')
            fig, axarr = plt.subplots(ncols=len(attr_types), sharey=True)
            for s, session_id in enumerate(f[filetype]):
                if f[filetype][session_id].attrs.__contains__('description'):
                    description = f[filetype][session_id].attrs['description']
                    if descriptions is not None and description not in descriptions:
                        continue
                else:
                    description = None
                if syn_name in f[filetype][session_id] and param_name is not None and \
                        param_name in f[filetype][session_id][syn_name]:
                    found = True
                    if param_label is None and 'param_label' in f[filetype][session_id][syn_name][param_name].attrs:
                        param_label = f[filetype][session_id][syn_name][param_name].attrs['param_label']
                    for i, attr_type in enumerate(attr_types):
                        if len(attr_types) == 1:
                            axes = axarr
                        else:
                            axes = axarr[i]
                        if attr_type not in f[filetype][session_id][syn_name][param_name]:
                            continue
                        for j, sec_type in enumerate(f[filetype][session_id][syn_name][param_name][attr_type]):
                            if sec_type not in marker_dict:
                                m = len(marker_dict)
                                marker_dict[sec_type] = markers[m]
                            marker = marker_dict[sec_type]
                            distances = f[filetype][session_id][syn_name][param_name][attr_type][sec_type]['distances'][:]
                            param_vals = f[filetype][session_id][syn_name][param_name][attr_type][sec_type]['values'][:]
                            if description is None:
                                label = sec_type + ' session' + session_id
                            else:
                                label = sec_type + ' ' + description
                            axes.scatter(distances, param_vals, color=colors[s], label=label, alpha=0.25,
                                         marker=marker, s=10.)
                            if max_param_val is None:
                                max_param_val = max(param_vals)
                            else:
                                max_param_val = max(max_param_val, max(param_vals))
                            if min_param_val is None:
                                min_param_val = min(param_vals)
                            else:
                                min_param_val = min(min_param_val, min(param_vals))
                            if max_dist is None:
                                max_dist = max(distances)
                            else:
                                max_dist = max(max_dist, max(distances))
                            if min_dist is None:
                                min_dist = min(distances)
                            else:
                                min_dist = min(min_dist, min(distances))
            if not found:
                raise Exception('Specified synaptic mechanism: %s parameter: %s not found in the provided file: %s' %
                                (syn_name, param_name, file))
            min_dist = min(0., min_dist)
            xmin = min_dist - 0.01 * (max_dist - min_dist)
            xmax = max_dist + 0.01 * (max_dist - min_dist)
            for i, attr_type in enumerate(attr_types):
                if len(attr_types) == 1:
                    axes = axarr
                else:
                    axes = axarr[i]
                axes.legend(loc='best', scatterpoints=1, frameon=False, framealpha=0.5,
                            fontsize=mpl.rcParams['font.size'])
                axes.set_xlabel('Distance to soma (um)')
                axes.set_xlim(xmin, xmax)
                axes.set_ylabel(ylabel + ' (' + yunits + ')')
                if (max_param_val is not None) and (min_param_val is not None):
                    buffer = 0.1 * (max_param_val - min_param_val)
                    axes.set_ylim(min_param_val - buffer, max_param_val + buffer)
                if param_label is not None:
                    axes.set_title(attr_types[i], fontsize=mpl.rcParams['font.size'])
                clean_axes(axes)
                axes.tick_params(direction='out')
            fig.suptitle(param_label, fontsize=mpl.rcParams['font.size'])
            if not svg_title is None:
                if param_label is not None:
                    svg_title = svg_title + ' - ' + param_label + '.svg'
                elif param_name is None:
                    svg_title = svg_title + ' - ' + syn_name + '_' + ' distribution.svg'
                else:
                    svg_title = svg_title + ' - ' + syn_name + '_' + param_name + ' distribution.svg'
                fig.set_size_inches(5.27, 4.37)
                fig.savefig(data_dir + svg_title, format='svg', transparent=True)
            plt.show()
            plt.close()
            if svg_title is not None:
                mpl.rcParams['font.size'] = remember_font_size


def plot_mech_param_distribution(cell, mech_name, param_name, export=None, overwrite=False, scale_factor=10000.,
                                 param_label=None, description=None, ylabel='Conductance density', yunits='pS/um2',
                                 svg_title=None, show=True, sec_types=None, data_dir='data'):
    """
    Takes a cell as input rather than a file. No simulation is required, this method just takes a fully specified cell
    and plots the relationship between distance and the specified mechanism parameter for all segments in sections of
    the provided sec_types (defaults to just dendritic sec_types). Used while debugging specification of mechanism
    parameters.
    :param cell: :class:'BiophysCell'
    :param mech_name: str
    :param param_name: str
    :param export: str (name of hdf5 file for export)
    :param overwrite: bool (whether to overwrite or append to potentially existing hdf5 file)
    :param scale_factor: float
    :param param_label: str
    :param description: str
    :param ylabel: str
    :param yunits: str
    :param svg_title: str
    :param show: bool
    :param sec_types: list or str
    :param data_dir: str (path)
    """
    if svg_title is not None:
        remember_font_size = mpl.rcParams['font.size']
        mpl.rcParams['font.size'] = 20
    if sec_types is None or (isinstance(sec_types, str) and sec_types == 'dend'):
        sec_types = ['basal', 'trunk', 'apical', 'tuft']
    elif isinstance(sec_types, str) and sec_types == 'all':
        sec_types = default_ordered_sec_types
    elif not all(sec_type in default_ordered_sec_types for sec_type in sec_types):
        raise ValueError('plot_mech_param_distribution: unrecognized sec_types: %s' % str(sec_types))
    maxval, minval = 1., 0.
    distances = defaultdict(list)
    param_vals = defaultdict(list)
    sec_types_list = [sec_type for sec_type in sec_types if sec_type in cell.nodes and len(cell.nodes[sec_type]) > 0]
    num_colors = 10
    color_x = np.linspace(0., 1., num_colors)
    colors = [cm.Set1(x) for x in color_x]
    for sec_type in sec_types_list:
        if len(cell.nodes[sec_type]) > 0:
            for branch in cell.nodes[sec_type]:
                for seg in [seg for seg in branch.sec if hasattr(seg, mech_name)]:
                    distances[sec_type].append(get_distance_to_node(cell, cell.tree.root, branch, seg.x))
                    if sec_type == 'basal':
                        distances[sec_type][-1] *= -1
                    param_vals[sec_type].append(getattr(getattr(seg, mech_name), param_name) * scale_factor)

    if export is not None:
        export_file_path = data_dir + '/' + export
        if overwrite:
            if os.path.isfile(export_file_path):
                os.remove(export_file_path)

    if len(param_vals) == 0 and export is not None:
        print('Not exporting to %s; mechanism: %s parameter: %s not found in any sec_type' % \
              (export, mech_name, param_name))
        return
    fig, axes = plt.subplots(1)
    max_param_val, min_param_val = 0.1, 0.
    xmax0, xmin0 = 0.1, 0.
    for i, sec_type in enumerate(param_vals):
        axes.scatter(distances[sec_type], param_vals[sec_type], color=colors[i], label=sec_type, alpha=0.5)
        if maxval is None:
            maxval = max(param_vals[sec_type])
        else:
            maxval = max(maxval, max(param_vals[sec_type]))
        if minval is None:
            minval = min(param_vals[sec_type])
        else:
            minval = min(minval, min(param_vals[sec_type]))
        xmax0 = max(xmax0, max(distances[sec_type]))
        xmin0 = min(xmin0, min(distances[sec_type]))
    axes.set_xlabel('Distance to soma (um)')
    xmin = xmin0 - 0.01 * (xmax0 - xmin0)
    xmax = xmax0 + 0.01 * (xmax0 - xmin0)
    axes.set_xlim(xmin, xmax)
    axes.set_ylabel(ylabel + ' (' + yunits + ')')
    if (maxval is not None) and (minval is not None):
        buffer = 0.01 * (maxval - minval)
        axes.set_ylim(minval - buffer, maxval + buffer)
    if param_label is not None:
        axes.set_title(param_label, fontsize=mpl.rcParams['font.size'])
    axes.legend(loc='best', scatterpoints=1, frameon=False, framealpha=0.5, fontsize=mpl.rcParams['font.size'])
    clean_axes(axes)

    if svg_title is not None:
        if param_label is not None:
            svg_title = svg_title + ' - ' + param_label + '.svg'
        else:
            svg_title = svg_title + ' - ' + mech_name + '_' + param_name + ' distribution.svg'
        fig.set_size_inches(5.27, 4.37)
        fig.savefig(data_dir + svg_title, format='svg', transparent=True)
    if show:
        plt.show()
    plt.close()
    if svg_title is not None:
        mpl.rcParams['font.size'] = remember_font_size

    if export is not None:
        f = h5py.File(export_file_path, 'a')
        if 'mech_file_path' in f.attrs:
            if cell.mech_file_path is None or not f.attrs['mech_file_path'] == cell.mech_file_path:
                raise ValueError('plot_mech_param_distribution: provided mech_file_path: %s does not match the '
                                'mech_file_path of %s cell %i: %s' %
                                (f.attrs['mech_file_path'], cell.pop_name, cell.gid, cell.mech_file_path))
        elif cell.mech_file_path is not None:
            f.attrs['mech_file_path'] = cell.mech_file_path
        filetype = 'plot_mech_param'
        if filetype not in f:
            f.create_group(filetype)
        if len(f[filetype]) == 0:
            session_id = '0'
        else:
            session_id = str(len(f[filetype]))
        f[filetype].create_group(session_id)
        if description is not None:
            f[filetype][session_id].attrs['description'] = description
        f[filetype][session_id].create_group(mech_name)
        f[filetype][session_id][mech_name].create_group(param_name)
        if param_label is not None:
            f[filetype][session_id][mech_name][param_name].attrs['param_label'] = param_label
        f[filetype][session_id][mech_name][param_name].attrs['gid'] = cell.gid
        if svg_title is not None:
            f[filetype][session_id][mech_name][param_name].attrs['svg_title'] = svg_title
        for sec_type in param_vals:
            f[filetype][session_id][mech_name][param_name].create_group(sec_type)
            f[filetype][session_id][mech_name][param_name][sec_type].create_dataset('values',
                                                                                    data=param_vals[sec_type])
            f[filetype][session_id][mech_name][param_name][sec_type].create_dataset('distances',
                                                                                    data=distances[sec_type])
        f.close()


def plot_cable_param_distribution(cell, mech_name, export=None, overwrite=False, scale_factor=1., param_label=None,
                                  description=None, ylabel='Specific capacitance', yunits='uF/cm2', svg_title=None,
                                  show=True, data_dir='data', sec_types=None):
    """
    Takes a cell as input rather than a file. No simulation is required, this method just takes a fully specified cell
    and plots the relationship between distance and the specified mechanism parameter for all dendritic segments. Used
    while debugging specification of mechanism parameters.
    :param cell: :class:'BiophysCell'
    :param mech_name: str
    :param param_name: str
    :param export: str (name of hdf5 file for export)
    :param overwrite: bool (whether to overwrite or append to potentially existing hdf5 file)
    :param scale_factor: float
    :param param_label: str
    :param ylabel: str
    :param yunits: str
    :param svg_title: str
    :param data_dir: str (path)
    :param sec_types: list of str
    """
    if svg_title is not None:
        remember_font_size = mpl.rcParams['font.size']
        mpl.rcParams['font.size'] = 20
    if sec_types is None or (isinstance(sec_types, str) and sec_types == 'dend'):
        sec_types = ['basal', 'trunk', 'apical', 'tuft']
    elif isinstance(sec_types, str) and sec_types == 'all':
        sec_types = default_ordered_sec_types
    elif not all(sec_type in default_ordered_sec_types for sec_type in sec_types):
        raise ValueError('plot_synaptic_attribute_distribution: unrecognized sec_types: %s' % str(sec_types))
    sec_types_list = [sec_type for sec_type in sec_types if sec_type in cell.nodes and len(cell.nodes[sec_type]) > 0]
    fig, axes = plt.subplots(1)
    maxval, minval = 1., 0.
    distances = defaultdict(list)
    param_vals = defaultdict(list)
    num_colors = len(sec_types_list)
    color_x = np.linspace(0., 1., num_colors)
    colors = [cm.Set1(x) for x in color_x]
    for sec_type in sec_types_list:
        if len(cell.nodes[sec_type]) > 0:
            for branch in cell.nodes[sec_type]:
                if mech_name == 'Ra':
                    distances[sec_type].append(get_distance_to_node(cell, cell.tree.root, branch))
                    if sec_type == 'basal':
                        distances[sec_type][-1] *= -1
                    param_vals[sec_type].append(getattr(branch.sec, mech_name) * scale_factor)
                else:
                    for seg in [seg for seg in branch.sec if hasattr(seg, mech_name)]:
                        distances[sec_type].append(get_distance_to_node(cell, cell.tree.root, branch, seg.x))
                        if sec_type == 'basal':
                            distances[sec_type][-1] *= -1
                        param_vals[sec_type].append(getattr(seg, mech_name) * scale_factor)
    xmax0 = 0.1
    xmin0 = 0.
    for i, sec_type in enumerate(param_vals):
        axes.scatter(distances[sec_type], param_vals[sec_type], color=colors[i], label=sec_type, alpha=0.5)
        if maxval is None:
            maxval = max(param_vals[sec_type])
        else:
            maxval = max(maxval, max(param_vals[sec_type]))
        if minval is None:
            minval = min(param_vals[sec_type])
        else:
            minval = min(minval, min(param_vals[sec_type]))
        xmax0 = max(xmax0, max(distances[sec_type]))
        xmin0 = min(xmin0, min(distances[sec_type]))
    axes.set_xlabel('Distance to soma (um)')
    xmin = xmin0 - 0.01 * (xmax0 - xmin0)
    xmax = xmax0 + 0.01 * (xmax0 - xmin0)
    axes.set_xlim(xmin, xmax)
    axes.set_ylabel(ylabel + ' (' + yunits + ')')
    if (maxval is not None) and (minval is not None):
        buffer = 0.01 * (maxval - minval)
        axes.set_ylim(minval - buffer, maxval + buffer)
    if param_label is not None:
        axes.set_title(param_label, fontsize=mpl.rcParams['font.size'])
    axes.legend(loc='best', scatterpoints=1, frameon=False, framealpha=0.5, fontsize=mpl.rcParams['font.size'])
    clean_axes(axes)
    axes.tick_params(direction='out')
    if svg_title is not None:
        if param_label is not None:
            svg_title = svg_title + ' - ' + param_label + '.svg'
        else:
            svg_title = svg_title + ' - ' + mech_name + '_' + ' distribution.svg'
        fig.set_size_inches(5.27, 4.37)
        fig.savefig(data_dir + svg_title, format='svg', transparent=True)
    if show:
        plt.show()
    plt.close()
    if svg_title is not None:
        mpl.rcParams['font.size'] = remember_font_size

    if export is not None:
        export_file_path = data_dir + '/' + export
        if overwrite:
            if os.path.isfile(export_file_path):
                os.remove(export_file_path)
        f = h5py.File(export_file_path, 'a')
        if 'mech_file_path' in f.attrs:
            if not (f.attrs['mech_file_path'] == '{}'.format(cell.mech_file_path)):
                raise Exception('Specified mechanism filepath {} does not match the mechanism filepath '
                                'of the cell {}'.format(f.attrs['mech_file_path'], cell.mech_file_path))
        else:
            f.attrs['mech_file_path'] = '{}'.format(cell.mech_file_path)
        filetype = 'plot_mech_param'
        if filetype not in f:
            f.create_group(filetype)
        if len(f[filetype]) == 0:
            session_id = '0'
        else:
            session_id = str(len(f[filetype]))
        f[filetype].create_group(session_id)
        if description is not None:
            f[filetype][session_id].attrs['description'] = description
        f[filetype][session_id].create_group(mech_name)
        if param_label is not None:
            f[filetype][session_id][mech_name].attrs['param_label'] = param_label
        f[filetype][session_id][mech_name].attrs['gid'] = cell.gid
        if svg_title is not None:
            f[filetype][session_id][mech_name].attrs['svg_title'] = svg_title
        for sec_type in param_vals:
            f[filetype][session_id][mech_name].create_group(sec_type)
            f[filetype][session_id][mech_name][sec_type].create_dataset('values', data=param_vals[sec_type])
            f[filetype][session_id][mech_name][sec_type].create_dataset('distances', data=distances[sec_type])
        f.close()


def plot_mech_param_from_file(mech_name, param_name, filename, descriptions=None, param_label=None,
                              ylabel='Conductance density', yunits='pS/um2', svg_title=None, data_dir='data'):
    """
    Takes in a list of files, and superimposes plots of distance vs. the provided mechanism parameter for all sec_types
    found in each file.
    :param mech_name: str
    :param param_name: str
    :param filename: str (hdf filename)
    :param descriptions: list of str (descriptions of each session). If None, then plot all session_ids
    :param param_label: str
    :param ylabel: str
    :param yunits: str
    :param svg_title: str
    :param data_dir: str (path)
    """
    if svg_title is not None:
        remember_font_size = mpl.rcParams['font.size']
        mpl.rcParams['font.size'] = 20
    fig, axes = plt.subplots(1)
    max_param_val, min_param_val = 0., 0.
    max_dist, min_dist = 0., 0.
    num_colors = 10
    markers = mlines.Line2D.filled_markers
    color_x = np.linspace(0., 1., num_colors)
    colors = [cm.Set1(x) for x in color_x]
    marker_dict = {}
    file_path = data_dir + '/' + filename
    found = False
    if os.path.isfile(file_path):
        with h5py.File(file_path, 'r') as f:
            filetype = 'plot_mech_param'
            if filetype not in f:
                raise Exception('The file {} has the incorrect filetype; it is not plot_mech_param'.format(file))
            for s, session_id in enumerate(f[filetype]):
                if f[filetype][session_id].attrs.__contains__('description'):
                    description = f[filetype][session_id].attrs['description']
                    if descriptions is not None and description not in descriptions:
                        continue
                else:
                    description = None
                if mech_name in f[filetype][session_id] and \
                        (param_name is None or param_name in f[filetype][session_id][mech_name]):
                    found = True
                    if param_name is None:
                        if param_label is None and 'param_label' in f[filetype][session_id][mech_name].attrs:
                            param_label = f[filetype][session_id][mech_name].attrs['param_label']
                        group = f[filetype][session_id][mech_name]
                    else:
                        if param_label is None and \
                                'param_label' in f[filetype][session_id][mech_name][param_name].attrs:
                            param_label = f[filetype][session_id][mech_name][param_name].attrs['param_label']
                        group = f[filetype][session_id][mech_name][param_name]
                    for j, sec_type in enumerate(group):
                        if sec_type not in marker_dict:
                            m = len(marker_dict)
                            marker_dict[sec_type] = markers[m]
                        marker = marker_dict[sec_type]
                        param_vals = group[sec_type]['values'][:]
                        distances = group[sec_type]['distances'][:]
                        if description is None:
                            label = sec_type + ' session ' + session_id
                        else:
                            label = sec_type + ' ' + description
                        axes.scatter(distances, param_vals, color=colors[s], label=label, alpha=0.5, marker=marker)
                        if max_param_val is None:
                            max_param_val = max(param_vals)
                        else:
                            max_param_val = max(max_param_val, max(param_vals))
                        if min_param_val is None:
                            min_param_val = min(param_vals)
                        else:
                            min_param_val = min(min_param_val, min(param_vals))
                        if max_dist is None:
                            max_dist = max(distances)
                        else:
                            max_dist = max(max_dist, max(distances))
                        if min_dist is None:
                            min_dist = min(distances)
                        else:
                            min_dist = min(min_dist, min(distances))
    if not found:
        raise Exception('Specified mechanism: %s parameter: %s not found in the provided file: %s' %
                        (mech_name, param_name, file))
    axes.set_xlabel('Distance to soma (um)')
    min_dist = min(0., min_dist)
    xmin = min_dist - 0.01 * (max_dist - min_dist)
    xmax = max_dist + 0.01 * (max_dist - min_dist)
    axes.set_xlim(xmin, xmax)
    axes.set_ylabel(ylabel + ' (' + yunits + ')')
    if (max_param_val is not None) and (min_param_val is not None):
        buffer = 0.1 * (max_param_val - min_param_val)
    axes.set_ylim(min_param_val - buffer, max_param_val + buffer)
    if param_label is not None:
        axes.set_title(param_label, fontsize=mpl.rcParams['font.size'])
    axes.legend(loc='best', scatterpoints=1, frameon=False, framealpha=0.5, fontsize=mpl.rcParams['font.size'])
    clean_axes(axes)
    axes.tick_params(direction='out')
    if not svg_title is None:
        if param_label is not None:
            svg_title = svg_title + ' - ' + param_label + '.svg'
        elif param_name is None:
            svg_title = svg_title + ' - ' + mech_name + '_' + ' distribution.svg'
        else:
            svg_title = svg_title + ' - ' + mech_name + '_' + param_name + ' distribution.svg'
        fig.set_size_inches(5.27, 4.37)
        fig.savefig(data_dir + svg_title, format='svg', transparent=True)
    plt.show()
    plt.close()
    if svg_title is not None:
        mpl.rcParams['font.size'] = remember_font_size
        

def clean_axes(axes, left=True, right=False):
    """
    Remove top and right axes from pyplot axes object.
    :param axes: list of pyplot.Axes
    :param top: bool
    :param left: bool
    :param right: bool
    """
    if not type(axes) in [np.ndarray, list]:
        axes = [axes]
    elif type(axes) == np.ndarray:
        axes = axes.flatten()
    for axis in axes:
        axis.tick_params(direction='out')
        axis.spines['top'].set_visible(False)
        if not right:
            axis.spines['right'].set_visible(False)
        if not left:
            axis.spines['left'].set_visible(False)
        axis.get_xaxis().tick_bottom()
        axis.get_yaxis().tick_left()
        

def calculate_module_density(gid_module_assignments, gid_normed_distance):
    """
    TODO: context needs to be provided as an argument?
    :param gid_module_assignments:
    :param gid_normed_distance:
    :return:
    """

    module_bounds = [[1.0, 0.0] for _ in range(10)]
    module_counts = [0 for _ in range(10)]
    gid_module_assignments = context.gid_module_assignments
    gid_normed_distance    = context.gid_normed_distance
    
    for (gid,module) in list(gid_module_assignments.items()):
        normed_u, _, _, _ = gid_normed_distance[gid]
        if normed_u < module_bounds[module-1][0]:
            module_bounds[module - 1][0] = normed_u
        if normed_u > module_bounds[module - 1][1]:
            module_bounds[module - 1][1] = normed_u
        module_counts[module - 1] += 1

    module_widths  = [y-x for [x,y] in module_bounds]
    module_density = np.divide(module_counts, module_widths)
    return module_bounds, module_counts, module_density


def plot_module_assignment_histogram():
    """
    TODO: context needs to be provided as an argument?
    :return:
    """

    module_bounds, module_counts, module_density = calculate_module_density()

    fig, (ax1, ax2, ax3) = plt.subplots(3,1)

    ax1.bar(np.arange(10)+1, module_counts)
    ax1.set_xlabel('Module')
    ax1.set_ylabel('Count')

    ax2.bar(np.arange(10)+1, module_density)
    ax2.set_xlabel('Module')
    ax2.set_ylabel('Density')

    for (i, bounds) in enumerate(module_bounds):
        ax3.plot([bounds[0],bounds[1]], [i+1,i+1], label='%i' % (i+1))
    ax3.set_xlabel('Normalized Bounds')
    ax3.set_ylabel('Module')
    ax3.legend(frameon=False, framealpha=0.5, loc='center left')

    fig, (ax1, ax2) = plt.subplots(2,1)
    normalized_u_positions = [norm_u for (norm_u,_,_,_) in list(context.gid_normed_distance.values())]
    absolute_u_positions   = [u for (_,_,u,_) in list(context.gid_normed_distance.values())]
    absolute_v_positions   = [v for (_,_,_,v) in list(context.gid_normed_distance.values())]
    hist_norm, edges_norm  = np.histogram(normalized_u_positions, bins=25)
    hist_abs, edges_abs    = np.histogram(absolute_u_positions, bins=100)
    hist_v_abs, edges_v_abs = np.histogram(absolute_v_positions, bins=100)

    ax1.plot(edges_norm[1:], hist_norm)
    ax1.set_xlabel('Normalized septo-temporal position')
    ax1.set_ylabel('Cell count')

    ax2.plot(edges_abs[1:], hist_abs)
    ax2.set_xlabel('Absolute septo-temporal position')
    ax2.set_ylabel('Cell Count')

    fig, ax = plt.subplots()
    module_pos_dictionary = dict()
    for gid in context.gid_normed_distance:
        norm_u,_,_,_ = context.gid_normed_distance[gid]
        module       = context.gid_module_assignments[gid]
        if module in module_pos_dictionary:
            module_pos_dictionary[module].append(norm_u)
        else:
            module_pos_dictionary[module] = [norm_u]

    for module in module_pos_dictionary:
        positions = module_pos_dictionary[module]
        hist_pos, _ = np.histogram(positions, bins=edges_norm)
        hist_pos = hist_pos.astype('float32')
        ax.plot(edges_norm[1:], old_div(hist_pos, hist_norm))
    ax.legend(['%i' % (i+1) for i in range(10)])

    plt.show()


def plot_1D_rate_map(t, rate_map, peak_rate=None, spike_train=None, title=None, **kwargs):
    """

    :param t: array
    :param rate_map: array
    :param peak_rate: float
    :param spike_train: array
    :param title: str
    """
    fig_options = copy.copy(default_fig_options)
    fig_options.update(kwargs)

    if peak_rate is None:
        peak_rate = np.max(rate_map)
    fig, axes = plt.subplots()
    axes.plot(t, rate_map)
    if spike_train is not None:
        axes.plot(spike_train, np.ones_like(spike_train), 'k.')
    axes.set_ylim(0., peak_rate * 1.1)
    axes.set_ylabel('Firing rate (Hz)', fontsize=fig_options.fontSize)
    axes.set_xlabel('Time (ms)', fontsize=fig_options.fontSize)
    axes.set_title(title, fontsize=fig_options.fontSize)
    clean_axes(axes)

    if fig_options.saveFig is not None:
        save_figure(fig_options.saveFig, fig=fig, **fig_options())

    if fig_options.showFig:
        fig.show()


def plot_2D_rate_map(x, y, rate_map, peak_rate=None, title=None, **kwargs):
    """

    :param x: array
    :param y: array
    :param rate_map: array
    :param peak_rate: float
    :param title: str
    """
    fig_options = copy.copy(default_fig_options)
    fig_options.update(kwargs)

    if peak_rate is None:
        peak_rate = np.max(rate_map)
    fig, axes = plt.subplots()
    pc = axes.pcolor(x, y, rate_map, vmin=0., vmax=peak_rate)
    axes.set_aspect('equal')
    cbar = fig.colorbar(pc, ax=axes)
    cbar.set_label('Firing Rate (Hz)', rotation=270., labelpad=20., fontsize=fig_options.fontSize)
    axes.set_xlabel('X Position (cm)', fontsize=fig_options.fontSize)
    axes.set_ylabel('Y Position (cm)', fontsize=fig_options.fontSize)
    clean_axes(axes)
    if title is not None:
        axes.set_title(title, fontsize=fig_options.fontSize)

    if fig_options.saveFig is not None:
        save_figure(fig_options.saveFig, fig=fig, **fig_options())

    if fig_options.showFig:
        fig.show()
