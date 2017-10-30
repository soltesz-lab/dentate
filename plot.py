
import itertools
import numpy as np
from mpi4py import MPI
from neuroh5.io import read_cell_attributes, read_population_ranges, read_population_names
import matplotlib.pyplot as plt
from matplotlib import gridspec, mlab

color_list = [[0.42,0.67,0.84], [0.90,0.76,0.00], [0.42,0.83,0.59], [0.90,0.32,0.00],
              [0.34,0.67,0.67], [0.90,0.59,0.00], [0.42,0.82,0.83], [1.00,0.85,0.00],
              [0.33,0.67,0.47], [1.00,0.38,0.60], [0.57,0.67,0.33], [0.5,0.2,0.0],
              [0.71,0.82,0.41], [0.0,0.2,0.5], [0.70,0.32,0.10],
              [0.42,0.67,0.84], [0.90,0.76,0.00], [0.42,0.83,0.59], [0.90,0.32,0.00],
              [0.34,0.67,0.67], [0.90,0.59,0.00], [0.42,0.82,0.83], [1.00,0.85,0.00],
              [0.33,0.67,0.47], [1.00,0.38,0.60], [0.57,0.67,0.33], [0.5,0.2,0.0],
              [0.71,0.82,0.41], [0.0,0.2,0.5], [0.70,0.32,0.10]] 



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


def plot_raster (input_file, namespace_id, timeRange = None, maxSpikes = int(1e6), orderInverse = False, labels = 'legend', popRates = False,
                 spikeHist = None, spikeHistBin = 5, syncLines = False, lw = 3, marker = '|', figSize = (15,8), fontSize = 14, saveFig = None, 
                showFig = True, verbose = False): 
    ''' 
    Raster plot of network spike times 
        - input_file: file with spike data
        - namespace_id: attribute namespace for spike events
        - timeRange ([start:stop]): Time range of spikes shown; if None shows all (default: None)
        - maxSpikes (int): maximum number of spikes that will be plotted  (default: 1e6)
        - orderInverse (True|False): Invert the y-axis order (default: False)
        - labels = ('legend', 'overlay'): Show population labels in a legend or overlayed on one side of raster (default: 'legend')
        - popRates = (True|False): Include population rates (default: False)
        - spikeHist (None|'overlay'|'subplot'): overlay line over raster showing spike histogram (spikes/bin) (default: False)
        - spikeHistBin (int): Size of bin in ms to use for histogram (default: 5)
        - syncLines (True|False): calculate synchorny measure and plot vertical lines for each spike to evidence synchrony (default: False)
        - lw (integer): Line width for each spike (default: 3)
        - marker (char): Marker for each spike (default: '|')
        - fontSize (integer): Size of text font (default: 14)
        - figSize ((width, height)): Size of figure (default: (15,8))
        - saveFig (None|True|'fileName'): File name where to save the figure (default: None)
            if set to True uses filename from simConfig (default: None)
        - showFig (True|False): Whether to show the figure or not (default: True)

        - Returns figure handle
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
    spkcolorlst = []
    num_cell_spks = {}
    num_cells   = 0

    tmin = float('inf')
    tmax = 0.
    for pop_name in population_names:
 
        color   = pop_colors[pop_name]
        spkiter = read_cell_attributes(comm, input_file, pop_name, namespace=namespace_id)
        this_num_cell_spks = 0

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
                    num_cells += 1
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
                        num_cells += 1

        pop_spkts = np.asarray(pop_spktlst, dtype=np.float32)
        del(pop_spktlst)
        pop_spkinds = np.asarray(pop_spkindlst, dtype=np.uint32)
        del(pop_spkindlst)

                        
        # Limit to maxSpikes
        if (len(spkts)>maxSpikes):
            if verbose:
                print('  Showing only randomly sampled %i out of %i spikes' % (maxSpikes, len(spkts)))
            sample_inds = np.random.random_integers(0, len(spkts)-1, size=int(maxSpikes))
            pop_spkts   = pop_spkts[sample_inds]
            pop_spkinds = pop_spkinds[sample_inds]
            timeRange[1] =  max(pop_spkts)
                        
        num_cell_spks[pop_name] = this_num_cell_spks

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
        for i,pop in enumerate(population_names):
            pop_num = pop_num_cells[pop]
            if pop_num > 0:
                if num_cell_spks[pop] == 0:
                    avg_rates[pop] = 0
                else:
                    avg_rates[pop] = num_cell_spks[pop] / pop_num / tsecs

                    
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
            pop_label_rates = [pop_name + ' (%.3g Hz)' % (avg_rates[pop_name]) for pop_name in population_names if pop_name in avg_rates]

        if labels == 'legend':
            lgd = plt.legend(sctplots, population_names, fontsize=fontSize, scatterpoints=1, markerscale=5.,
                             loc='upper right', bbox_to_anchor=(1.1, 1.0))
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

