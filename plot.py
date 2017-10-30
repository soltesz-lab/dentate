
import itertools
import numpy as np
from scipy import interpolate
from mpi4py import MPI
from neuroh5.io import read_cell_attributes, read_population_ranges, read_population_names
import matplotlib.pyplot as plt
from matplotlib import gridspec, mlab
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable

color_list = ["#000000", "#00FF00", "#0000FF", "#FF0000", "#01FFFE", "#FFA6FE",
              "#FFDB66", "#006401", "#010067", "#95003A", "#007DB5", "#FF00F6", "#FFEEE8", "#774D00",
              "#90FB92", "#0076FF", "#D5FF00", "#FF937E", "#6A826C", "#FF029D", "#FE8900", "#7A4782",
              "#7E2DD2", "#85A900", "#FF0056", "#A42400", "#00AE7E", "#683D3B", "#BDC6FF", "#263400",
              "#BDD393", "#00B917", "#9E008E", "#001544", "#C28C9F", "#FF74A3", "#01D0FF", "#004754",
              "#E56FFE", "#788231", "#0E4CA1", "#91D0CB", "#BE9970", "#968AE8", "#BB8800", "#43002C",
              "#DEFF74", "#00FFC6", "#FFE502", "#620E00", "#008F9C", "#98FF52", "#7544B1", "#B500FF",
              "#00FF78", "#FF6E41", "#005F39", "#6B6882", "#5FAD4E", "#A75740", "#A5FFD2", "#FFB167", 
              "#009BFF", "#E85EBE"]




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

def plot_in_degree(connectivity_path, coords_path, distances_namespace, destination_id, source=None, showFig = True, saveFig = False, verbose=True):
    """
    Plot connectivity in-degree with respect to septo-temporal position (longitudinal and transverse arc distances to reference points).

    :param connectivity_path:
    :param coords_path:
    :param distances_namespace: 
    :param destination_id: 

    """

    comm = MPI.COMM_WORLD

    (population_ranges, _) = read_population_ranges(comm, coords_path)

    target_start = population_ranges[target][0]
    target_count = population_ranges[target][1]

    with h5py.File(connectivity_path, 'r') as f:
        if source is None:
            in_degrees = f['Nodes']['Vertex Metrics']['Vertex indegree']['Attribute Value'][target_start:target_start+target_count]

    if verbose:
        print 'read in degrees (%i elements)' % len(in_degrees)
            
    distances = read_cell_attributes(comm, coords_path, target, namespace=distances_namespace)

    if verbose:
        print 'read distances (%i elements)' % len(distances.keys())
    
    soma_distances = { k: (v['U Distance'][0], v['V Distance'][0]) for (k,v) in distances.iteritems() }
    del distances
    
    fig1 = plt.figure(1, figsize=plt.figaspect(1.) * 2.)
    ax = plt.gca()

    distance_U = np.asarray([ soma_distances[v+target_start][0] for v in range(0,len(in_degrees)) ])
    distance_V = np.asarray([ soma_distances[v+target_start][1] for v in range(0,len(in_degrees)) ])


    distance_points = np.vstack((distance_U, distance_V))
    x_min = np.min(distance_U)
    x_max = np.max(distance_U)
    y_min = np.min(distance_V)
    y_max = np.max(distance_V)
    grid_x, grid_y = np.meshgrid(np.arange(x_min, x_max, 1),
                                np.arange(y_min, y_max, 1))
    in_degree_grid = interpolate.griddata(distance_points.T,
                                          in_degrees,
                                          (grid_x, grid_y),
                                          method='nearest')

    ###pcm = plt.pcolormesh (distance_U, distance_V, in_degree_grid, cmap='RdBu')

    plt.imshow(in_degree_grid, origin='lower', cmap='RdBu',
               vmin=np.min(in_degrees), vmax=np.max(in_degrees),
               extent=[x_min, x_max, y_min, y_max])
    plt.axis([x_min, x_max, y_min, y_max])
    plt.colorbar()
    
    ax.set_xlabel('Arc distance (septal - temporal) (um)')
    ax.set_ylabel('Arc distance (supra - infrapyramidal)  (um)')
    #ax.set_title(target+' <-- '+source+' (in degree)')
    #ax.set_aspect('equal', 'box')
    #clean_axes(ax)
    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes("right", size="3%", pad=0.05)
    #cbar = plt.colorbar(pcm, cax=cax)
    #cbar.ax.set_ylabel('Counts')

    if saveFig: 
        if isinstance(saveFig, basestring):
            filename = saveFig
        else:
            filename = namespace_id+' '+'in_degree.png'
            plt.savefig(filename)

    if showFig:
        show_figure()
    
    return ax



def plot_population_density(population, soma_coords, u, v, distance_U, distance_V, max_u, max_v, bin_size=100.):
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

    plt.show()
    plt.close()

## Plot spike histogram
def plot_spike_raster (input_file, namespace_id, timeRange = None, maxSpikes = int(1e6), orderInverse = False, labels = 'legend', popRates = False,
                       spikeHist = None, spikeHistBin = 5, lw = 3, marker = '|', figSize = (15,8), fontSize = 14, saveFig = None, 
                       showFig = True, verbose = False): 
    ''' 
    Raster plot of network spike times. Returns the figure handle.

    input_file: file with spike data
    namespace_id: attribute namespace for spike events
    timeRange ([start:stop]): Time range of spikes shown; if None shows all (default: None)
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

    if verbose:
        print('Reading spike data...')
    comm = MPI.COMM_WORLD

    (population_ranges, N) = read_population_ranges(comm, input_file)
    population_names  = read_population_names(comm, input_file)

    pop_num_cells = {}
    for k in population_names:
        pop_num_cells[k] = population_ranges[k][1]

    pop_colors = { pop_name: color_list[ipop%len(color_list)] for ipop, pop_name in enumerate(population_names) }

    spkindlst   = []
    spktlst     = []
    num_cell_spks = {}
    num_cells   = 0
    pop_active_cells = {}

    tmin = float('inf')
    tmax = 0.
    for pop_name in population_names:
 
        color   = pop_colors[pop_name]
        spkiter = read_cell_attributes(comm, input_file, pop_name, namespace=namespace_id)
        this_num_cell_spks = 0
        active_set = set([])

        pop_spkindlst = []
        pop_spktlst   = []
        
        # Time Range
        if timeRange is None:
            for spkind,spkts in spkiter:
                for spkt in spkts['t']:
                    pop_spkindlst.append(spkind)
                    pop_spktlst.append(spkt)
                    if spkt < tmin:
                        tmin = spkt
                    if spkt > tmax:
                        tmax = spkt
                    this_num_cell_spks += 1
                    active_set.add(spkind)
        else:
            for spkind,spkts in spkiter:
                for spkt in spkts['t']:
                    if timeRange[0] <= spkt <= timeRange[1]:
                        pop_spkindlst.append(spkind)
                        pop_spktlst.append(spkt)
                        if spkt < tmin:
                            tmin = spkt
                        if spkt > tmax:
                            tmax = spkt
                        this_num_cell_spks += 1
                        active_set.add(spkind)

        pop_active_cells[pop_name] = active_set
        num_cell_spks[pop_name] = this_num_cell_spks

        pop_spkts = np.asarray(pop_spktlst, dtype=np.float32)
        del(pop_spktlst)
        pop_spkinds = np.asarray(pop_spkindlst, dtype=np.uint32)
        del(pop_spkindlst)
                        
        # Limit to maxSpikes
        if (len(pop_spkts)>maxSpikes):
            if verbose:
                print('  Showing only randomly sampled %i out of %i spikes for population %s' % (maxSpikes, len(pop_spkts), pop_name))
            sample_inds = np.random.random_integers(0, len(pop_spkts)-1, size=int(maxSpikes))
            pop_spkts   = pop_spkts[sample_inds]
            pop_spkinds = pop_spkinds[sample_inds]
            tmax = max(tmax, max(pop_spkts))
                        

        spktlst.append(pop_spkts)
        spkindlst.append(pop_spkinds)
        
        if verbose:
            print 'Read %i spikes for population %s' % (this_num_cell_spks, pop_name)

    timeRange = [tmin, tmax]

    # Calculate spike histogram 
    if spikeHist:
        all_spkts = np.concatenate(spktlst, axis=0)
        histo = np.histogram(all_spkts, bins = np.arange(timeRange[0], timeRange[1], spikeHistBin))
        histoT = histo[1][:-1]+spikeHistBin/2
        histoCount = histo[0]

    if popRates:
        avg_rates = {}
        tsecs = (timeRange[1]-timeRange[0])/1e3 
        for i,pop_name in enumerate(population_names):
            pop_num = len(pop_active_cells[pop_name])
            if pop_num > 0:
                if num_cell_spks[pop_name] == 0:
                    avg_rates[pop_name] = 0
                else:
                    avg_rates[pop_name] = num_cell_spks[pop_name] / pop_num / tsecs

                    
    # Plot spikes
    fig, ax1 = plt.subplots(figsize=figSize)

    if verbose:
        print('Creating raster plot...')

    sctplots = []
    
    if spikeHist is None:

        for (pop_name, pop_spkinds, pop_spkts) in itertools.izip (population_names, spkindlst, spktlst):
            sctplots.append(ax1.scatter(pop_spkts, pop_spkinds, s=10, linewidths=lw, marker=marker, c=pop_colors[pop_name], alpha=0.5, label=pop_name))
        
        ax1.set_xlim(timeRange)

        ax1.set_xlabel('Time (ms)', fontsize=fontSize)
        ax1.set_ylabel('Cell Index', fontsize=fontSize)
        ax1.set_xlim([tmin, tmax])
        ax1.set_ylim(-1, N+1)    
        
    elif spikeHist == 'subplot':

        gs = gridspec.GridSpec(2, 1, height_ratios=[2,1])
        ax1=plt.subplot(gs[0])

        
        for (pop_name, pop_spkinds, pop_spkts) in itertools.izip (population_names, spkindlst, spktlst):
            sctplots.append(ax1.scatter(pop_spkts, pop_spkinds, s=10, linewidths=lw, marker=marker, c=pop_colors[pop_name], alpha=0.5, label=pop_name))
            
        ax1.set_xlim(timeRange)

        ax1.set_xlabel('Time (ms)', fontsize=fontSize)
        ax1.set_ylabel('Cell Index', fontsize=fontSize)
        ax1.set_xlim(timeRange)
        ax1.set_ylim(-1, N+1)

        # Add legend
        if popRates:
            pop_labels = [pop_name + ' (%i active; %.3g Hz)' % (len(pop_active_cells[pop_name]), avg_rates[pop_name]) for pop_name in population_names if pop_name in avg_rates]
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



## Plot spike histogram
def plot_spike_histogram (input_file, namespace_id, include = ['eachPop'], timeRange = None, maxSpikes=int(1e6), binSize = 5., 
                          overlay=True, graphType='bar', yaxis = 'rate', figSize = (15,8), fontSize = 14, lw = 3, 
                          saveFig = None, showFig = True, verbose = False): 
    ''' 
    Plots spike histogram. Returns figure handle.

        - input_file: file with spike data
        - namespace_id: attribute namespace for spike events
        - include (['eachPop'|<population name>]): List of data series to include. 
            (default: ['eachPop'] - expands to the name of each population)
        - timeRange ([start:stop]): Time range of spikes shown; if None shows all (default: None)
        - binSize (int): Size in ms of each bin (default: 5)
        - overlay (True|False): Whether to overlay the data lines or plot in separate subplots (default: True)
        - graphType ('line'|'bar'): Type of graph to use (line graph or bar plot) (default: 'line')
        - yaxis ('rate'|'count'): Units of y axis (firing rate in Hz, or spike count) (default: 'rate')
        - figSize ((width, height)): Size of figure (default: (10,8))
        - fontSize (integer): Size of text font (default: 14)
        - lw (integer): Line width for each spike (default: 3)
        - saveFig (None|True|'fileName'): File name where to save the figure;
            if set to True uses filename from simConfig (default: None)
        - showFig (True|False): Whether to show the figure or not (default: True)
    '''
    comm = MPI.COMM_WORLD

    (population_ranges, N) = read_population_ranges(comm, input_file)
    population_names  = read_population_names(comm, input_file)

    pop_num_cells = {}
    for k in population_names:
        pop_num_cells[k] = population_ranges[k][1]

    
    # Replace 'eachPop' with list of populations
    if 'eachPop' in include: 
        include.remove('eachPop')
        for pop in population_names:
            include.append(pop)

    spkpoplst   = []
    spkindlst   = []
    spktlst     = []
    num_cells   = 0
    num_cell_spks    = {}
    pop_active_cells = {}

    tmin = float('inf')
    tmax = 0.
    for pop_name in include:
 
        spkiter = read_cell_attributes(comm, input_file, pop_name, namespace=namespace_id)
        this_num_cell_spks = 0
        active_set = set([])

        pop_spkindlst = []
        pop_spktlst   = []
        
        # Time Range
        if timeRange is None:
            for spkind,spkts in spkiter:
                for spkt in spkts['t']:
                    pop_spkindlst.append(spkind)
                    pop_spktlst.append(spkt)
                    if spkt < tmin:
                        tmin = spkt
                    if spkt > tmax:
                        tmax = spkt
                    this_num_cell_spks += 1
                    active_set.add(spkind)
        else:
            for spkind,spkts in spkiter:
                for spkt in spkts['t']:
                    if timeRange[0] <= spkt <= timeRange[1]:
                        pop_spkindlst.append(spkind)
                        pop_spktlst.append(spkt)
                        if spkt < tmin:
                            tmin = spkt
                        if spkt > tmax:
                            tmax = spkt
                        this_num_cell_spks += 1
                        active_set.add(spkind)

        pop_active_cells[pop_name] = active_set
        num_cell_spks[pop_name] = this_num_cell_spks

        pop_spkts = np.asarray(pop_spktlst, dtype=np.float32)
        del(pop_spktlst)
        pop_spkinds = np.asarray(pop_spkindlst, dtype=np.uint32)
        del(pop_spkindlst)
                        
        # Limit to maxSpikes
        if (len(pop_spkts)>maxSpikes):
            if verbose:
                print('  Showing only randomly sampled %i out of %i spikes for population %s' % (maxSpikes, len(pop_spkts), pop_name))
            sample_inds = np.random.random_integers(0, len(pop_spkts)-1, size=int(maxSpikes))
            pop_spkts   = pop_spkts[sample_inds]
            pop_spkinds = pop_spkinds[sample_inds]
            tmax = max(tmax, max(pop_spkts))

        spkpoplst.append(pop_name)
        spktlst.append(pop_spkts)
        spkindlst.append(pop_spkinds)
        
        if verbose:
            print 'Read %i spikes for population %s' % (this_num_cell_spks, pop_name)

    timeRange = [tmin, tmax]
            
    # Y-axis label
    if yaxis == 'rate':
        yaxisLabel = 'Average cell firing rate (Hz)'
    elif yaxis == 'count':
        yaxisLabel = 'Spike count'
    else:
        print 'Invalid yaxis value %s', (yaxis)
        return

    # create fig
    fig, ax1 = plt.subplots(figsize=figSize)

    if verbose:
        print('Plotting spike histogram...')
        
    # Plot separate line for each entry in include
    histData = []
    for iplot, subset in enumerate(spkpoplst):

        spkts = spktlst[iplot]

        histo = np.histogram(spkts, bins = np.arange(timeRange[0], timeRange[1], binSize))
        histoT = histo[1][:-1]+binSize/2
        histoCount = histo[0] 
        
        histData.append(histoCount)

        if yaxis=='rate':
            histoCount = histoCount * (1000.0 / binSize) / (len(pop_active_cells[subset])) # convert to firing rate

        color = color_list[iplot%len(color_list)]

        if not overlay: 
            plt.subplot(len(spkpoplst),1,iplot+1)  # if subplot, create new subplot
            plt.title (str(subset), fontsize=fontSize)
   
        if graphType == 'line':
            plt.plot (histoT, histoCount, linewidth=lw, color = color)
        elif graphType == 'bar':
            plt.bar(histoT, histoCount, width = binSize, color = color)

        if iplot == 0: 
            plt.xlabel('Time (ms)', fontsize=fontSize)
            plt.ylabel(yaxisLabel, fontsize=fontSize) # add yaxis in opposite side
        plt.xlim(timeRange)

    if len(include) < 5:  # if apply tight_layout with many subplots it inverts the y-axis
        try:
            plt.tight_layout()
        except:
            pass

    # Add legend
    if overlay:
        for i,subset in enumerate(include):
            plt.plot(0,0,color=color_list[i%len(color_list)],label=str(subset))
        plt.legend(fontsize=fontSize, bbox_to_anchor=(1.04, 1), loc=2, borderaxespad=0.)
        maxLabelLen = min(10,max([len(str(l)) for l in include]))
        plt.subplots_adjust(right=(0.9-0.012*maxLabelLen))


    if saveFig: 
        if isinstance(saveFig, basestring):
            filename = saveFig
        else:
            filename = namespace_id+'_'+'spike_histogram.png'
        plt.savefig(filename)

    if showFig:
        show_figure()

    return fig


def plot_rate_PSD (input_file, namespace_id, include = ['eachPop'], timeRange = None,
                   maxSpikes = int(1e6), binSize = 5, Fs = 200, smooth = 0, overlay=True, 
                   figSize = (15,8), fontSize = 14, lw = 3, saveFig = None, showFig = True, verbose = False): 
    ''' 
    Plots firing rate power spectral density (PSD). Returns figure handle.
        - input_file: file with spike data
        - namespace_id: attribute namespace for spike events
        - include (['eachPop'|<population name>]): List of data series to include. 
            (default: ['eachPop'] - expands to the name of each population)
        - timeRange ([start:stop]): Time range of spikes shown; if None shows all (default: None)
        - binSize (int): Size in ms of each bin (default: 5)
        - Fs (float): frequency band
        - overlay (True|False): Whether to overlay the data lines or plot in separate subplots (default: True)
        - graphType ('line'|'bar'): Type of graph to use (line graph or bar plot) (default: 'line')
        - yaxis ('rate'|'count'): Units of y axis (firing rate in Hz, or spike count) (default: 'rate')
        - figSize ((width, height)): Size of figure (default: (10,8))
        - fontSize (integer): Size of text font (default: 14)
        - lw (integer): Line width for each spike (default: 3)
        - saveFig (None|True|'fileName'): File name where to save the figure;
            if set to True uses filename from simConfig (default: None)
        - showFig (True|False): Whether to show the figure or not (default: True)

    '''
    comm = MPI.COMM_WORLD

    (population_ranges, N) = read_population_ranges(comm, input_file)
    population_names  = read_population_names(comm, input_file)

    pop_num_cells = {}
    for k in population_names:
        pop_num_cells[k] = population_ranges[k][1]

    
    # Replace 'eachPop' with list of populations
    if 'eachPop' in include: 
        include.remove('eachPop')
        for pop in population_names:
            include.append(pop)

    spkpoplst   = []
    spkindlst   = []
    spktlst     = []
    num_cells   = 0
    num_cell_spks    = {}
    pop_active_cells = {}

    tmin = float('inf')
    tmax = 0.
    for pop_name in include:
 
        spkiter = read_cell_attributes(comm, input_file, pop_name, namespace=namespace_id)
        this_num_cell_spks = 0
        active_set = set([])

        pop_spkindlst = []
        pop_spktlst   = []
        
        # Time Range
        if timeRange is None:
            for spkind,spkts in spkiter:
                for spkt in spkts['t']:
                    pop_spkindlst.append(spkind)
                    pop_spktlst.append(spkt)
                    if spkt < tmin:
                        tmin = spkt
                    if spkt > tmax:
                        tmax = spkt
                    this_num_cell_spks += 1
                    active_set.add(spkind)
        else:
            for spkind,spkts in spkiter:
                for spkt in spkts['t']:
                    if timeRange[0] <= spkt <= timeRange[1]:
                        pop_spkindlst.append(spkind)
                        pop_spktlst.append(spkt)
                        if spkt < tmin:
                            tmin = spkt
                        if spkt > tmax:
                            tmax = spkt
                        this_num_cell_spks += 1
                        active_set.add(spkind)

        pop_active_cells[pop_name] = active_set
        num_cell_spks[pop_name] = this_num_cell_spks

        pop_spkts = np.asarray(pop_spktlst, dtype=np.float32)
        del(pop_spktlst)
        pop_spkinds = np.asarray(pop_spkindlst, dtype=np.uint32)
        del(pop_spkindlst)
                        
        # Limit to maxSpikes
        if (len(pop_spkts)>maxSpikes):
            if verbose:
                print('  Showing only randomly sampled %i out of %i spikes for population %s' % (maxSpikes, len(pop_spkts), pop_name))
            sample_inds = np.random.random_integers(0, len(pop_spkts)-1, size=int(maxSpikes))
            pop_spkts   = pop_spkts[sample_inds]
            pop_spkinds = pop_spkinds[sample_inds]
            tmax = max(tmax, max(pop_spkts))

        spkpoplst.append(pop_name)
        spktlst.append(pop_spkts)
        spkindlst.append(pop_spkinds)
        
        if verbose:
            print 'Read %i spikes for population %s' % (this_num_cell_spks, pop_name)

    timeRange = [tmin, tmax]

    # create fig
    fig, ax1 = plt.subplots(figsize=figSize)

    if verbose:
        print('Plotting firing rate power spectral density (PSD) ...')
    
    # Plot separate line for each entry in include
    histData = []
    for iplot, subset in enumerate(spkpoplst):

        spkts = spktlst[iplot]

        histo = np.histogram(spkts, bins = np.arange(timeRange[0], timeRange[1], binSize))
        histoT = histo[1][:-1]+binSize/2
        histoCount = histo[0] 
        
        histData.append(histoCount)
        
        power = mlab.psd(histoCount, Fs=Fs, NFFT=256, detrend=mlab.detrend_none, window=mlab.window_hanning, 
                         noverlap=0, pad_to=None, sides='default', scale_by_freq=None)

        if smooth:
            signal = smooth1d(10*np.log10(power[0]), smooth)
        else:
            signal = 10*np.log10(power[0])
            
        freqs = power[1]

        color = color_list[iplot%len(color_list)]

        if not overlay: 
            plt.subplot(len(spkpoplst),1,iplot+1)  # if subplot, create new subplot
            plt.title (str(subset), fontsize=fontSize)

        plt.plot(freqs, signal, linewidth=lw, color=color)

        plt.xlabel('Frequency (Hz)', fontsize=fontSize)
        plt.ylabel('Power Spectral Density (dB/Hz)', fontsize=fontSize) # add yaxis in opposite side
        plt.xlim([0, (Fs/2)-1])

    if len(include) < 5:  # if apply tight_layout with many subplots it inverts the y-axis
        try:
            plt.tight_layout()
        except:
            pass

    # Add legend
    if overlay:
        for i,subset in enumerate(include):
            color = color_list[i%len(color_list)]
            plt.plot(0,0,color=color,label=str(subset))
        plt.legend(fontsize=fontSize, bbox_to_anchor=(1.04, 1), loc=2, borderaxespad=0.)
        maxLabelLen = min(10,max([len(str(l)) for l in include]))
        plt.subplots_adjust(right=(0.9-0.012*maxLabelLen))

 
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

    return fig, power
