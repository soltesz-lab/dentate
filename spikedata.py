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

        sort_idxs = np.argsort(pop_spkts)
        spkpoplst.append(pop_name)
        spktlst.append(np.take(pop_spkts, sort_idxs))
        del pop_spkts
        spkindlst.append(np.take(pop_spkinds, sort_idxs))
        del pop_spkinds
        
        if verbose:
            print 'Read %i spikes for population %s' % (this_num_cell_spks, pop_name)

    return {'spkpoplst': spkpoplst, 'spktlst': spktlst, 'spkindlst': spkindlst, 'tmin': tmin, 'tmax': tmax,
            'pop_active_cells': pop_active_cells, 'num_cell_spks': num_cell_spks }


def make_spike_dict (spkinds, spkts):
    spk_dict = defaultdict(list)
    for spkind, spkt in itertools.izip(np.nditer(spkinds), np.nditer(spkts)):
        spk_dict[int(spkind)].append(spkt)
    return spk_dict

    
def interspike_intervals (spkdict):
    isi_dict = {}
    for ind, lst in spkdict.iteritems():
        isi_dict[ind] = np.diff(np.asarray(lst))
    return isi_dict


def spike_rates (spkdict, t_dflt):
    rate_dict = {}
    isidict = interspike_intervals(spkdict)
    for ind, isiv in isidict.iteritems():
        if isiv.size > 0:
            t = np.sum(isiv)
            rate = isiv.size / t / 1000.0
        elif len(spks) > 0:
            rate = 1.0 / t_dflt / 1000.0
        else:
            rate = 0.0
        rate_dict[ind] = rate
    return rate_dict


def spike_bin_inds (bins, spkts):
    bin_inds   = np.digitize(spkts, bins = bins)
    return bin_inds


def spike_bin_rates (bins, spkdict):
    spk_bin_dict = {}
    for (ind, lst) in spkdict.iteritems():
        spkts         = np.asarray(lst, dtype=np.float32)
        spkints       = np.diff(spkts)
        bin_inds      = np.digitize(spkts, bins = bins)
        isi_bin_inds  = bin_inds[1:]
        rate_bins     = []
        count_bins    = []
        if spkts.size > 0:
            t_prev     = spkts[0]
        else:
            t_prev = 0.0
        
        for ibin in xrange(1, len(bins)+1):
            bin_spks = spkts[bin_inds == ibin]
            bin_isi  = spkints[isi_bin_inds == ibin]
            count    = bin_spks.size
            if count > 1:
                t        = np.sum(bin_isi)
                rate     = bin_isi.size / t * 1000.0
                t_prev   = bin_spks[-1]
            elif count > 0:
                t = bin_spks[0] - t_prev
                if t > 0.:
                    rate = 1.0 / t * 1000.0
                else:
                    rate = 0.0
                t_prev = bin_spks[-1]
            else:
                rate = 0.0
            rate_bins.append(rate)
            count_bins.append(count)
            
        spk_bin_dict[ind] = (np.asarray(count_bins, dtype=np.uint32), np.asarray(rate_bins, dtype=np.float32))
        
    return spk_bin_dict
            

def spatial_information (trajectory, spkdict, timeRange, positionBinSize):

    tmin = timeRange[0]
    tmax = timeRange[1]
    
    (x, y, d, t)  = trajectory

    t_inds = np.where((t >= tmin) & (t <= tmax))
    t = t[t_inds]
    d = d[t_inds]
        
    d_extent       = np.max(d) - np.min(d)
    position_bins  = np.arange(np.min(d), np.max(d), positionBinSize)
    d_bin_inds     = np.digitize(d, bins = position_bins)
    t_bin_ind_lst  = [0]
    for ibin in xrange(1, len(position_bins)+1):
        bin_inds = np.where(d_bin_inds == ibin)
        t_bin_ind_lst.append(np.max(bin_inds))
    t_bin_inds = np.asarray(t_bin_ind_lst)
    time_bins  = t[t_bin_inds]

    d_bin_probs = {}
    prev_bin = np.min(d)
    for ibin in xrange(1, len(position_bins)+1):
        d_bin  = d[d_bin_inds == ibin]
        if d_bin.size > 0:
            bin_max = np.max(d_bin)
            d_prob = (bin_max - prev_bin) / d_extent
            d_bin_probs[ibin] = d_prob
            prev_bin = bin_max
        else:
            d_bin_probs[ibin] = 0.
            
    rate_bin_dict = spike_bin_rates(time_bins, spkdict)
    MI_dict = {}
    for ind, (count_bins, rate_bins) in rate_bin_dict.iteritems():
        MI = 0.
        rates = np.asarray(rate_bins)
        R     = np.mean(rates)

        if R > 0.:
            for ibin in xrange(1, len(position_bins)+1):
                p_i  = d_bin_probs[ibin]
                R_i  = rates[ibin-1]
                if R_i > 0.:
                    MI   += p_i * (R_i / R) * math.log(R_i / R, 2)
            
        MI_dict[ind] = MI
        
    return MI_dict
            
            

    


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

def activity_ratio(stimulus, response, binSize = 25.):
    result = np.power(np.sum(np.array(mean_rates),axis=0)/nstim,2) / (np.sum(np.power(mean_rates,2.0),axis=0)/nstim)
    return result


