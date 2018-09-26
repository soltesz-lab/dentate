import math, sys
import itertools
from collections import defaultdict
from pathos.multiprocessing import ProcessPool
import numpy as np
import neo, elephant
from quantities import s, ms, Hz
from dentate import utils
from utils import viewitems
from neuroh5.io import read_cell_attributes, write_cell_attributes, read_population_ranges, read_population_names

## This logger will inherit its setting from its root logger, dentate,
## which is created in module env
logger = utils.get_module_logger(__name__)

## Returns a list of arrays with consecutive values from data
def consecutive(data):
    return np.split(data, np.where(np.diff(data) != 1)[0]+1)

def read_spike_events(input_file, population_names, namespace_id, timeVariable='t', timeRange = None, maxSpikes = None):

    spkpoplst        = []
    spkindlst        = []
    spktlst          = []
    num_cell_spks    = {}
    pop_active_cells = {}

    tmin = float('inf')
    tmax = 0.


    for pop_name in population_names:

        if timeRange is None:
            logger.info('Reading spike data for population %s...' % pop_name)
        else:
            logger.info('Reading spike data for population %s in time range %s...' % (pop_name, str(timeRange)))

        spkiter = read_cell_attributes(input_file, pop_name, namespace=namespace_id)
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
            logger.warn('  Reading only randomly sampled %i out of %i spikes for population %s' % (maxSpikes, len(pop_spkts), pop_name))
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
        
        logger.warn('Read %i spikes for population %s' % (this_num_cell_spks, pop_name))

    return {'spkpoplst': spkpoplst, 'spktlst': spktlst, 'spkindlst': spkindlst, 'tmin': tmin, 'tmax': tmax,
            'pop_active_cells': pop_active_cells, 'num_cell_spks': num_cell_spks }


def make_spike_dict (spkinds, spkts):
    spk_dict = defaultdict(list)
    for spkind, spkt in zip(np.nditer(spkinds), np.nditer(spkts)):
        spk_dict[int(spkind)].append(spkt)
    return spk_dict

    
def interspike_intervals (spkdict):
    isi_dict = {}
    for ind, lst in viewitems(spkdict):
        isi_dict[ind] = np.diff(np.asarray(lst))
    return isi_dict


def spike_rates (spkdict, t):
    rate_dict = {}
    isidict = interspike_intervals(spkdict)
    for ind, isiv in viewitems(isidict):
        if isiv.size > 0:
            rate = isiv.size / (t / 1000.0)
        else:
            rate = 0.0
        rate_dict[ind] = rate
    return rate_dict

def spike_inst_rates_func (item,t_start,t_stop,sampling_period,sigma,kernel,bin_steps=5):
    (ind, lst) = item
    spkts = np.asarray(lst, dtype=np.float32)
    spktrain      = neo.core.SpikeTrain(spkts*ms, t_start=t_start*ms, t_stop=t_stop*ms)
    spkrates_r    = elephant.statistics.instantaneous_rate(spktrain, sampling_period, kernel=kernel).ravel()
    spkrates_x    = np.linspace(t_start, t_stop, spkrates_r.size)
    
    return (ind, { 'x': np.asarray(spkrates_x, dtype=np.float32),
                   'rate': np.asarray(spkrates_r, dtype=np.float32)
                 })


def spike_inst_rates (population, spkdict, timeRange, sampling_period=2.0*ms, sigma = 0.05, nprocs=1, saveData=False):

    kernel = elephant.kernels.GaussianKernel(sigma = sigma*s, invert = True)

    t_start = timeRange[0]
    t_stop = timeRange[1]

    pool = ProcessPool(nprocs)
    spk_rate_dict = dict(pool.map(lambda item: spike_inst_rates_func(item,t_start,t_stop,sampling_period,sigma,kernel), viewitems(spkdict)))

    if saveData:
        if isinstance(saveData, str):
            filename = saveData
        else:
            filename = '%s_spike_inst_rates.h5' % population

        write_cell_attributes(filename, population, spk_rate_dict, namespace='Instantaneous Rate')
        
    return spk_rate_dict
            

def spike_bin_counts(spkdict, bins):
    count_bin_dict = {}
    for (ind, lst) in viewitems(spkdict):

        spkts = np.asarray(lst, dtype=np.float32)
        bin_inds      = np.digitize(spkts, bins = bins)
        count_bins    = []
    
        for ibin in range(1, len(bins)+1):
            bin_spks  = spkts[bin_inds == ibin]
            count    = bin_spks.size
            count_bins.append(count)
        
        count_bin_dict[ind] = np.asarray(count_bins, dtype=np.uint32)

    return count_bin_dict


def spatial_information (population, trajectory, spkdict, timeRange, positionBinSize, saveData = False):

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
    for ibin in range(1, len(position_bins)+1):
        bin_inds = np.where(d_bin_inds == ibin)
        t_bin_ind_lst.append(np.max(bin_inds))
    t_bin_inds = np.asarray(t_bin_ind_lst)
    time_bins  = t[t_bin_inds]

    d_bin_probs = {}
    prev_bin = np.min(d)
    for ibin in range(1, len(position_bins)+1):
        d_bin  = d[d_bin_inds == ibin]
        if d_bin.size > 0:
            bin_max = np.max(d_bin)
            d_prob = (bin_max - prev_bin) / d_extent
            d_bin_probs[ibin] = d_prob
            prev_bin = bin_max
        else:
            d_bin_probs[ibin] = 0.
            
    rate_bin_dict = spike_inst_rates(population, spkdict, time_bins, timeRange, saveData=saveData)
    MI_dict = {}
    for ind, (count_bins, rate_bins) in viewitems(rate_bin_dict):
        MI = 0.
        rates = np.asarray(rate_bins)
        R     = np.mean(rates)

        if R > 0.:
            for ibin in range(1, len(position_bins)+1):
                p_i  = d_bin_probs[ibin]
                R_i  = rates[ibin-1]
                if R_i > 0.:
                    MI   += p_i * (R_i / R) * math.log(R_i / R, 2)
            
        MI_dict[ind] = MI

    if saveData:
        write_cell_attributes(saveData, population, MI_dict, namespace='Spatial Mutual Information')

    return MI_dict

def place_fields (population, bin_size, rate_dict, nstdev=1.5, binsteps=5, baseline_fraction=None, saveData = False):

    pf_dict = {}
    pf_total_count = 0
    cell_count = 0
    pf_min = sys.maxsize
    pf_max = 0
    for ind, valdict  in viewitems(rate_dict):
        x      = valdict['x']
        rate   = valdict['rate']
        m      = np.mean(rate)
        rate1  = np.subtract(rate, m)
        if baseline_fraction is None:
            s  = np.std(rate1)
        else:
            k = rate1.shape[0]/baseline_fraction
            s = np.std(rate1[np.argpartition(rate1,k)[:k]])
        tmin   = x[0]
        tmax   = x[-1]
        bins   = np.arange(tmin, tmax, bin_size)
        pf_bins  = []
        pf_rate = []
        pf_norm_rate = []
        for ibin in range(1, len(bins)-1):
            binx = np.linspace(bins[ibin-1],bins[ibin],binsteps)
            r_n  = np.mean(np.interp(binx,x,rate1))
            r    = np.mean(np.interp(binx,x,rate))
            if r_n > nstdev*s:
                  pf_bins.append(ibin)
                  pf_rate.append(r)
                  pf_norm_rate.append(r_n)

        pf_count = len(consecutive(pf_bins))
        pf_min = min(pf_count, pf_min)
        pf_max = max(pf_count, pf_max)
        cell_count += 1
        pf_total_count += pf_count
        pf_dict[ind] = { 'pf_count': np.asarray([pf_count], dtype=np.uint32), \
                         'pf_bins': np.asarray(pf_bins, dtype=np.uint32), \
                         'pf_rate': np.asarray(pf_rate, dtype=np.float32),
                         'pf_norm_rate': np.asarray(pf_norm_rate, dtype=np.float32) }

    print('%s place fields: min %i max %i mean %f\n' % (population, pf_min, pf_max, float(pf_total_count)/float(cell_count)))
    if saveData:
        write_cell_attributes(saveData, population, pf_dict, namespace='Place Fields')

    return pf_dict
            
            

    


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


def histogram_correlation(spkdata, binSize=1., quantity='count', maxElems=None):
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
    for subset, spkinds, spkts in zip(spkpoplst, spkindlst, spktlst):
        i = 0
        spk_dict = defaultdict(list)
        for spkind, spkt in zip(np.nditer(spkinds), np.nditer(spkts)):
            spk_dict[int(spkind)].append(spkt)
        x_lst = []
        for ind, lst in viewitems(spk_dict):
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
            logger.warn('  Reading only randomly sampled %i out of %i cells for population %s' % (maxElems, x_matrix.shape[0], subset))
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

def histogram_autocorrelation(spkdata, binSize=1., lag=1, quantity='count', maxElems=None):
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
    for subset, spkinds, spkts in zip(spkpoplst, spkindlst, spktlst):
        i = 0
        spk_dict = defaultdict(list)
        for spkind, spkt in zip(np.nditer(spkinds), np.nditer(spkts)):
            spk_dict[int(spkind)].append(spkt)
        x_lst = []
        for ind, lst in viewitems(spk_dict):
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
            logger.warn('  Reading only randomly sampled %i out of %i cells for population %s' % (maxElems, x_matrix.shape[0], subset))
            sample_inds = np.random.randint(0, x_matrix.shape[0]-1, size=int(maxElems))
            x_matrix = x_matrix[sample_inds,:]


        corr_matrix = np.apply_along_axis(lambda y: autocorr(y, lag), 1, x_matrix)

        corr_dict[subset] = corr_matrix

    
    return corr_dict
