
import numpy as np
from mpi4py import MPI
from neuroh5.io import read_cell_attributes, read_population_ranges, read_population_names
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import mlab

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


def save_fig_data(figData, fileName=None, type=''):
    if not fileName or not isinstance(fileName, basestring):
        fileName = sim.cfg.filename+'_'+type+'.json'

    elif fileName.endswith('.json'):  # save to json
        import json
        print('Saving figure data as %s ... ' % (fileName))
        with open(fileName, 'w') as fileObj:
            json.dump(figData, fileObj)
    else: 
        print 'File extension to save figure data not recognized'


def ifilternone(iterable):
    for x in iterable:
        if not (x is None):
            yield x
            
def flatten(iterables):
    return (elem for iterable in ifilternone(iterables) for elem in iterable)


def plot_raster (input_file, namespace_id, timeRange = None, maxSpikes = int(1e6), orderInverse = False, labels = 'legend', popRates = False,
                spikeHist = None, spikeHistBin = 5, syncLines = False, lw = 3, marker = '|', figSize = (10,8), saveFig = None,
                showFig = True, verbose = False): 
    ''' 
    Raster plot of network spike times 
        - input_file: file with spike data
        - namespace: attribute namespace for spike trains
        - timeRange ([start:stop]): Time range of spikes shown; if None shows all (default: None)
        - maxSpikes (int): maximum number of spikes that will be plotted  (default: 1e6)
        - orderInverse (True|False): Invert the y-axis order (default: False)
        - labels = ('legend', 'overlay'): Show population labels in a legend or overlayed on one side of raster (default: 'legend')
        - popRates = (True|False): Include population rates (default: False)
        - spikeHist (None|'overlay'|'subplot'): overlay line over raster showing spike histogram (spikes/bin) (default: False)
        - spikeHistBin (int): Size of bin in ms to use for histogram (default: 5)
        - syncLines (True|False): calculate synchorny measure and plot vertical lines for each spike to evidence synchrony (default: False)
        - lw (integer): Line width for each spike (default: 2)
        - marker (char): Marker for each spike (default: '|')
        - figSize ((width, height)): Size of figure (default: (10,8))
        - saveFig (None|True|'fileName'): File name where to save the figure (default: None)
            if set to True uses filename from simConfig (default: None)
        - showFig (True|False): Whether to show the figure or not (default: True)

        - Returns figure handle
    '''

    if verbose:
        print('Plotting raster...')
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

        # Time Range
        if timeRange is None:
            for spkind,spkts in spkiter:
                for spkt in spkts['t']:
                    spkindlst.append(spkind)
                    spktlst.append(spkt)
                    spkcolorlst.append(color)
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
                        spkindlst.append(spkind)
                        spktlst.append(spkt)
                        spkcolorlst.append(color)
                        if spkt < tmin:
                            tmin = spkt
                        if spkt > tmax:
                            tmax = spkt
                        this_num_cell_spks += 1
                        num_cells += 1
                        
        num_cell_spks[pop_name] = this_num_cell_spks

    spkts = np.asarray(spktlst, dtype=np.float32)
    if verbose:
        print 'Read %i spikes' % (len(spkts)) 
    del(spktlst)
    spkinds = np.asarray(spkindlst, dtype=np.uint32)
    del(spkindlst)
    spkcolors = np.asarray(spkcolorlst, dtype=np.float32)
    del(spkcolorlst)

    timeRange = [tmin, tmax]
    
    # Limit to maxSpikes
    if (len(spkts)>maxSpikes):
        if verbose:
            print('  Showing only randomly sampled %i out of %i spikes' % (maxSpikes, len(spkts)))
        sample_inds = np.random.random_integers(0, len(spkts), size=int(maxSpikes))
        spkts = spkts[sample_inds]
        spkinds = spkinds[sample_inds]
        timeRange[1] =  max(spkts)

    # Calculate spike histogram 
    if spikeHist:
        histo = np.histogram(spkts, bins = np.arange(timeRange[0], timeRange[1], spikeHistBin))
        histoT = histo[1][:-1]+spikeHistBin/2
        histoCount = histo[0]

    # Plot spikes
    fig,ax1 = plt.subplots(figsize=figSize)
    fontsiz = 14

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

    if spikeHist is None:

        ax1.scatter(spkts, spkinds, s=10, linewidths=lw, marker=marker, c=spkcolors, alpha=0.5) 
        ax1.set_xlim(timeRange)

        ax1.set_xlabel('Time (ms)', fontsize=fontsiz)
        ax1.set_ylabel('Cell Index', fontsize=fontsiz)
        ax1.set_xlim([tmin, tmax])
        ax1.set_ylim(-1, N+1)    
        
    elif spikeHist == 'subplot':

        gs = gridspec.GridSpec(2, 1, height_ratios=[2,1])
        ax1=plt.subplot(gs[0])
 
        ax1.scatter(spkts, spkinds, s=10, linewidths=lw, marker=marker, c=spkcolors, alpha=0.5)
        ax1.set_xlim(timeRange)

        ax1.set_xlabel('Time (ms)', fontsize=fontsiz)
        ax1.set_ylabel('Cell Index', fontsize=fontsiz)
        ax1.set_xlim(timeRange)
        ax1.set_ylim(-1, N+1)

        # Add legend
        if popRates:
            pop_label_rates = [pop_name + ' (%.3g Hz)' % (avg_rates[pop_name]) for pop_name in population_names if pop_name in avg_rates]

        if labels == 'legend':
            for ipop, pop_name in enumerate(population_names):
                label = pop_label_rates[ipop] if popRates else pop_name
                plt.plot(0,0,color=pop_colors[pop_name],label=label)
            plt.legend(fontsize=fontsiz, bbox_to_anchor=(1.04, 1), loc=2, borderaxespad=0.)
            maxLabelLen = max([len(l) for l in population_names])
            rightOffset = 0.85 if popRates else 0.9
            plt.subplots_adjust(right=(rightOffset-0.012*maxLabelLen))
    
        maxLabelLen = max([len(l) for l in population_names])
        plt.subplots_adjust(right=(1.0-0.011*maxLabelLen))

        # Plot spike hist
        if spikeHist == 'overlay':
            ax2 = ax1.twinx()
            ax2.plot (histoT, histoCount, linewidth=0.5)
            ax2.set_ylabel('Spike count', fontsize=fontsiz) # add yaxis label in opposite side
            ax2.set_xlim(timeRange)
        elif spikeHist == 'subplot':
            ax2=plt.subplot(gs[1])
            ax2.plot (histoT, histoCount, linewidth=1.0)
            ax2.set_xlabel('Time (ms)', fontsize=fontsiz)
            ax2.set_ylabel('Spike count', fontsize=fontsiz)
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

