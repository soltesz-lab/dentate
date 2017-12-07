import math
import itertools
from collections import defaultdict
import numpy as np
from neuroh5.io import read_cell_attributes, read_population_ranges, read_population_names

def read_spike_events(comm, input_file, population_names, namespace_id, timeVariable='t', timeRange = None, maxSpikes = None, verbose = False):

    spkpoplst        = []
    spkindlst        = []
    spktlst          = []
    num_cell_spks    = {}
    pop_active_cells = {}

    tmin = float('inf')
    tmax = 0.

    if verbose:
        print('Reading spike data...')

    for pop_name in population_names:
 
        spkiter = read_cell_attributes(comm, input_file, pop_name, namespace=namespace_id)
        this_num_cell_spks = 0
        active_set = set([])

        pop_spkindlst = []
        pop_spktlst   = []
        
        # Time Range
        if timeRange is None:
            for spkind,spkts in spkiter:
                for spkt in spkts[timeVariable]:
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
                for spkt in spkts[timeVariable]:
                    if timeRange[0] <= spkt <= timeRange[1]:
                        pop_spkindlst.append(spkind)
                        pop_spktlst.append(spkt)
                        if spkt < tmin:
                            tmin = spkt
                        if spkt > tmax:
                            tmax = spkt
                        this_num_cell_spks += 1
                        active_set.add(spkind)

        if not active_set:
            continue
                        
        pop_active_cells[pop_name] = active_set
        num_cell_spks[pop_name] = this_num_cell_spks

        pop_spkts = np.asarray(pop_spktlst, dtype=np.float32)
        del(pop_spktlst)
        pop_spkinds = np.asarray(pop_spkindlst, dtype=np.uint32)
        del(pop_spkindlst)
                        
        # Limit to maxSpikes
        if (maxSpikes is not None) and (len(pop_spkts)>maxSpikes):
            if verbose:
                print('  Reading only randomly sampled %i out of %i spikes for population %s' % (maxSpikes, len(pop_spkts), pop_name))
            sample_inds = np.random.randint(0, len(pop_spkinds)-1, size=int(maxSpikes))
            pop_spkts   = pop_spkts[sample_inds]
            pop_spkinds = pop_spkinds[sample_inds]
            tmax = max(tmax, max(pop_spkts))

        spkpoplst.append(pop_name)
        spktlst.append(pop_spkts)
        spkindlst.append(pop_spkinds)
        
        if verbose:
            print 'Read %i spikes for population %s' % (this_num_cell_spks, pop_name)

    return {'spkpoplst': spkpoplst, 'spktlst': spktlst, 'spkindlst': spkindlst, 'tmin': tmin, 'tmax': tmax,
            'pop_active_cells': pop_active_cells, 'num_cell_spks': num_cell_spks }



def mvcorrcoef(X,y):
    Xm = np.reshape(np.mean(X,axis=1),(X.shape[0],1))
    ym = np.mean(y)
    r_num = np.sum(np.multiply(X-Xm,y-ym),axis=1)
    r_den = np.sqrt(np.sum(np.square(X-Xm),axis=1)*np.sum(np.square(y-ym)))
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.true_divide(r_num, r_den)
        r[r == np.inf] = 0
        r = np.nan_to_num(r)
    return r


def histogram_correlation(spkdata, binSize=1., quantity='count', maxElems=None, verbose=False):
    """Compute correlation coefficients of the spike count or firing rate histogram of each population. """

    spkpoplst        = spkdata['spkpoplst']
    spkindlst        = spkdata['spkindlst']
    spktlst          = spkdata['spktlst']
    num_cell_spks    = spkdata['num_cell_spks']
    pop_active_cells = spkdata['pop_active_cells']
    tmin             = spkdata['tmin']
    tmax             = spkdata['tmax']

    bins  = np.arange(tmin, tmax, binSize)
    
    corr_dict = {}
    for subset, spkinds, spkts in itertools.izip(spkpoplst, spkindlst, spktlst):
        i = 0
        spk_dict = defaultdict(list)
        for spkind, spkt in itertools.izip(np.nditer(spkinds), np.nditer(spkts)):
            spk_dict[int(spkind)].append(spkt)
        x_lst = []
        for ind, lst in spk_dict.iteritems():
            spkv  = np.asarray(lst)
            count, bin_edges = np.histogram(spkv, bins = bins)
            if quantity == 'rate':
                q = count * (1000.0 / binSize) # convert to firing rate
            else:
                q = count
            x_lst.append(q)
            i = i+1

        x_matrix = np.matrix(x_lst)
        del(x_lst)
        
        # Limit to maxElems
        if (maxElems is not None) and (x_matrix.shape[0]>maxElems):
            if verbose:
                print('  Reading only randomly sampled %i out of %i cells for population %s' % (maxElems, x_matrix.shape[0], pop_name))
            sample_inds = np.random.randint(0, x_matrix.shape[0]-1, size=int(maxElems))
            x_matrix = x_matrix[sample_inds,:]


        corr_matrix = np.apply_along_axis(lambda y: mvcorrcoef(x_matrix, y), 1, x_matrix)

        corr_dict[subset] = corr_matrix

    
    return corr_dict

def autocorr (y, lag):
    leny = y.shape[1]
    a = y[0,0:leny-lag].reshape(-1)
    b = y[0,lag:leny].reshape(-1)
    m = np.vstack((a[0,:].reshape(-1), b[0,:].reshape(-1)))
    r = np.corrcoef(m)[0,1]
    if math.isnan(r):
        return 0.
    else:
        return r

def histogram_autocorrelation(spkdata, binSize=1., lag=1, quantity='count', maxElems=None, verbose=False):
    """Compute autocorrelation coefficients of the spike count or firing rate histogram of each population. """

    spkpoplst        = spkdata['spkpoplst']
    spkindlst        = spkdata['spkindlst']
    spktlst          = spkdata['spktlst']
    num_cell_spks    = spkdata['num_cell_spks']
    pop_active_cells = spkdata['pop_active_cells']
    tmin             = spkdata['tmin']
    tmax             = spkdata['tmax']

    bins  = np.arange(tmin, tmax, binSize)
    
    corr_dict = {}
    for subset, spkinds, spkts in itertools.izip(spkpoplst, spkindlst, spktlst):
        i = 0
        spk_dict = defaultdict(list)
        for spkind, spkt in itertools.izip(np.nditer(spkinds), np.nditer(spkts)):
            spk_dict[int(spkind)].append(spkt)
        x_lst = []
        for ind, lst in spk_dict.iteritems():
            spkv  = np.asarray(lst)
            count, bin_edges = np.histogram(spkv, bins = bins)
            if quantity == 'rate':
                q = count * (1000.0 / binSize) # convert to firing rate
            else:
                q = count
            x_lst.append(q)
            i = i+1

        x_matrix = np.matrix(x_lst)
        del(x_lst)
        
        # Limit to maxElems
        if (maxElems is not None) and (x_matrix.shape[0]>maxElems):
            if verbose:
                print('  Reading only randomly sampled %i out of %i cells for population %s' % (maxElems, x_matrix.shape[0], pop_name))
            sample_inds = np.random.randint(0, x_matrix.shape[0]-1, size=int(maxElems))
            x_matrix = x_matrix[sample_inds,:]


        corr_matrix = np.apply_along_axis(lambda y: autocorr(y, lag), 1, x_matrix)

        corr_dict[subset] = corr_matrix

    
    return corr_dict

