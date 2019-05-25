import math, sys
import itertools
from collections import defaultdict
import numpy as np
from dentate import utils
from utils import viewitems, baks, akde
from neuroh5.io import read_cell_attributes, write_cell_attributes, read_population_ranges, read_population_names

## This logger will inherit its setting from its root logger, dentate,
## which is created in module env
logger = utils.get_module_logger(__name__)


def consecutive(data):
    """
    Returns a list of arrays with consecutive values from data.
    """
    return np.split(data, np.where(np.diff(data) != 1)[0]+1)

def mvcorrcoef(X,y):
    """
    Multivariate correlation coefficient.
    """
    Xm = np.reshape(np.mean(X,axis=1),(X.shape[0],1))
    ym = np.mean(y)
    r_num = np.sum(np.multiply(X-Xm,y-ym),axis=1)
    r_den = np.sqrt(np.sum(np.square(X-Xm),axis=1)*np.sum(np.square(y-ym)))
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.true_divide(r_num, r_den)
        r[r == np.inf] = 0
        r = np.nan_to_num(r)
    return r


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

def get_env_spike_dict(env, t_start=0.0):
    """
    Constructs  a dictionary with per-gid spike times from the output vectors with spike times and gids contained in env.
    """
    
    t_vec = np.array(env.t_vec, dtype=np.float32)
    id_vec = np.array(env.id_vec, dtype=np.uint32)

    if t_start > 0.0:
        inds = np.where(t_vec >= t_start)
        t_vec = t_vec[inds]
        id_vec = id_vec[inds]
    
    binlst  = []
    typelst = list(env.celltypes.keys())
    for k in typelst:
        binlst.append(env.celltypes[k]['start'])

    binvect  = np.array(binlst)
    sort_idx = np.argsort(binvect,axis=0)
    bins     = binvect[sort_idx][1:]
    types    = [ typelst[i] for i in sort_idx ]
    inds     = np.digitize(id_vec, bins)

    pop_spkdict = {}
    for i in range(0,len(types)):
        pop_name = types[i]
        spkdict  = {}
        sinds    = np.where(inds == i)
        if len(sinds) > 0:
            ids      = id_vec[sinds]
            ts       = t_vec[sinds]
            for j in range(0,len(ids)):
                id = ids[j]
                t  = ts[j]
                if id in spkdict:
                    spkdict[id].append(t)
                else:
                    spkdict[id]= [t]
            for j in list(spkdict.keys()):
                spkdict[j] = np.array(spkdict[j], dtype=np.float32)
        pop_spkdict[pop_name] = spkdict

    return pop_spkdict


def read_spike_events(input_file, population_names, namespace_id, time_variable='t', time_range = None, max_spikes = None):
    """
    Reads spike trains from a NeuroH5 file, and returns a dictionary with spike times and cell indices.
    """
    
    spkpoplst        = []
    spkindlst        = []
    spktlst          = []
    num_cell_spks    = {}
    pop_active_cells = {}

    tmin = float('inf')
    tmax = 0.


    for pop_name in population_names:

        if time_range is None:
            logger.info('Reading spike data for population %s...' % pop_name)
        else:
            logger.info('Reading spike data for population %s in time range %s...' % (pop_name, str(time_range)))

        spkiter = read_cell_attributes(input_file, pop_name, namespace=namespace_id)
        this_num_cell_spks = 0
        active_set = set([])

        pop_spkindlst = []
        pop_spktlst   = []

        # Time Range
        if time_range is None:
            for spkind,spkts in spkiter:
                for spkt in spkts[time_variable]:
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
                for spkt in spkts[time_variable]:
                    if time_range[0] <= spkt <= time_range[1]:
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
                        
        # Limit to max_spikes
        if (max_spikes is not None) and (len(pop_spkts)>max_spikes):
            logger.warn('  Reading only randomly sampled %i out of %i spikes for population %s' % (max_spikes, len(pop_spkts), pop_name))
            sample_inds = np.random.randint(0, len(pop_spkinds)-1, size=int(max_spikes))
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
    """
    Given arrays with cell indices and spike times, returns a dictionary with per-cell spike times.
    """
    spk_dict = defaultdict(list)
    for spkind, spkt in zip(np.nditer(spkinds), np.nditer(spkts)):
        spk_dict[int(spkind)].append(spkt)
    return spk_dict

    
def interspike_intervals (spkdict):
    """
    Calculates interspike intervals from the given spike dictionary.
    """
    isi_dict = {}
    for ind, lst in viewitems(spkdict):
        if len(lst) > 1:
            isi_dict[ind] = np.diff(np.asarray(lst))
        else:
            isi_dict[ind] = np.asarray([], dtype=np.float32)
    return isi_dict

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

def spike_rates (spkdict):
    """
    Calculates firing rates based on interspike intervals computed from the given spike dictionary.
    """
    rate_dict = {}
    isidict = interspike_intervals(spkdict)
    for ind, isiv in viewitems(isidict):
        if isiv.size > 0:
            rate = 1.0 / (np.mean(isiv) / 1000.0)
        else:
            rate = 0.0
        rate_dict[ind] = rate
    return rate_dict


def spike_density_estimate (population, spkdict, time_bins, save=False, progress=False, **kwargs):
    """
    Calculates spike density function for the given spike trains.
    """

    if progress:
        from tqdm import tqdm
    
    def make_spktrain (lst, t_start, t_stop):
        spkts = np.asarray(lst, dtype=np.float32)
        return spkts[(spkts >= t_start) & (spkts <= t_stop)]

    t_start = time_bins[0]
    t_stop = time_bins[-1]

    spktrains = { ind: make_spktrain(lst, t_start, t_stop) for (ind, lst) in viewitems(spkdict) }
    if progress:
        spk_rate_dict = { ind: baks(spkts / 1000., time_bins / 1000., **kwargs)[0].reshape((-1,))
                          for ind, spkts in tqdm(viewitems(spktrains)) if len(spkts) > 1 }
    else:
        spk_rate_dict = { ind: baks(spkts / 1000., time_bins / 1000., **kwargs)[0].reshape((-1,))
                          for ind, spkts in viewitems(spktrains) if len(spkts) > 1 }
        

    if save:
        if isinstance(save, str):
            filename = save
        else:
            filename = '%s_spike_density.h5' % population

        write_cell_attributes(filename, population, spk_rate_dict, namespace='Spike Density Function')

    result = { ind: { 'rate': rate, 'time': time_bins } for ind, rate in viewitems(spk_rate_dict) }
    return result
            


def spatial_information (population, trajectory, spkdict, time_range, position_bin_size, progress=False, save = False):
    """
    Calculates mutual information for the given spatial trajectory and spike trains.
    """

    tmin = time_range[0]
    tmax = time_range[1]
    
    (x, y, d, t)  = trajectory

    t_inds = np.where((t >= tmin) & (t <= tmax))
    t = t[t_inds]
    d = d[t_inds]
        
    d_extent       = np.max(d) - np.min(d)
    position_bins  = np.arange(np.min(d), np.max(d), position_bin_size)
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
            
    rate_bin_dict = spike_density_estimate(population, spkdict, time_bins, save=save, progress=progress)
    MI_dict = {}
    for ind, valdict in viewitems(rate_bin_dict):
        MI = 0.
        x      = valdict['time']
        rates  = valdict['rate']
        R      = np.mean(rates)

        if R > 0.:
            for ibin in range(1, len(position_bins)+1):
                p_i  = d_bin_probs[ibin]
                R_i  = rates[ibin-1]
                if R_i > 0.:
                    MI   += p_i * (R_i / R) * math.log(R_i / R, 2)
            
        MI_dict[ind] = MI

    if save:
        write_cell_attributes(save, population, MI_dict, namespace='Spatial Mutual Information')

    return MI_dict


def place_fields (population, bin_size, rate_dict, trajectory, nstdev=1.5, binsteps=5, baseline_fraction=None, min_pf_width=10., progress=False, save = False):
    """
    Estimates place fields from the given instantaneous spike rate dictionary.
    """

    if progress:
        from tqdm import tqdm
    
    (x, y, d, t) = trajectory

    pf_dict = {}
    pf_total_count = 0
    pf_cell_count = 0
    cell_count = 0
    pf_min = sys.maxsize
    pf_max = 0
    ncells = len(rate_dict)
    if progress:
        it = tqdm(viewitems(rate_dict))
    else:
        it = viewitems(rate_dict)
    for ind, valdict  in it:
        x      = valdict['time']
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
        bin_rates = []
        bin_norm_rates = []
        pf_ibins  = []
        for ibin in range(1, len(bins)):
            binx = np.linspace(bins[ibin-1],bins[ibin],binsteps)
            r_n  = np.mean(np.interp(binx,x,rate1))
            r    = np.mean(np.interp(binx,x,rate))
            bin_rates.append(r)
            bin_norm_rates.append(r_n)
            if r_n > nstdev*s:
                  pf_ibins.append(ibin-1)

        bin_rates = np.asarray(bin_rates)
        bin_norm_rates = np.asarray(bin_norm_rates)
                  
        if len(pf_ibins) > 0:
            pf_consecutive_ibins = []
            pf_consecutive_bins = []
            pf_widths = []
            for pf_ibin_array in consecutive(pf_ibins):
                pf_ibin_range = np.asarray([np.min(pf_ibin_array), np.max(pf_ibin_array)])
                pf_bin_range  = np.asarray([bins[pf_ibin_range[0]], bins[pf_ibin_range[1]]])
                pf_width = np.diff(np.interp(pf_bin_range, t, d))[0]
                pf_consecutive_ibins.append(pf_ibin_range)
                pf_consecutive_bins.append(pf_bin_range)
                pf_widths.append(pf_width)
            pf_filtered_ibins = [ pf_consecutive_ibins[i] for i, pf_width in enumerate(pf_widths) if pf_width >= min_pf_width ]
            pf_count = len(pf_filtered_ibins)
            pf_ibins =  [ xrange(pf_ibin[0], pf_ibin[1]+1) for pf_ibin in pf_filtered_ibins ]
            pf_mean_width = [] 
            pf_mean_rate = [] 
            pf_peak_rate = [] 
            pf_mean_norm_rate = [] 
            for pf_ibin_iter in pf_ibins:
                pf_ibin_array = list(pf_ibin_iter)
                pf_mean_width.append(np.mean(np.asarray([pf_width for pf_width in pf_widths if pf_width >= min_pf_width])))
                pf_mean_rate.append(np.mean(np.asarray(bin_rates[pf_ibin_array])))
                pf_peak_rate.append(np.max(np.asarray(bin_rates[pf_ibin_array])))
                pf_mean_norm_rate.append(np.mean(np.asarray(bin_norm_rates[pf_ibin_array])))

            pf_min = min(pf_count, pf_min)
            pf_max = max(pf_count, pf_max)
            pf_cell_count += 1
            pf_total_count += pf_count
        else:
            pf_count = 0
            pf_mean_width = []
            pf_mean_rate = []
            pf_peak_rate = []
            pf_mean_norm_rate = []

        cell_count += 1
        pf_dict[ind] = { 'pf_count': np.asarray([pf_count], dtype=np.uint32), \
                         'pf_mean_width': np.asarray(pf_mean_width, dtype=np.float32),
                         'pf_mean_rate': np.asarray(pf_mean_rate, dtype=np.float32),
                         'pf_peak_rate': np.asarray(pf_peak_rate, dtype=np.float32),
                         'pf_mean_norm_rate': np.asarray(pf_mean_norm_rate, dtype=np.float32) }

    logger.info('%s place fields: min %i max %i mean %f\n' % (population, pf_min, pf_max, float(pf_total_count)/float(cell_count)))
    if save:
        write_cell_attributes(save, population, pf_dict, namespace='Place Fields')

    return pf_dict
            

def activity_sequences (population, spkdict, time_bins):
    """
    Estimates activity ensembles from the given instantaneous spike rate dictionary.
    """

    import sklearn
    from sklearn.neighbors import BallTree

    def make_spkbin (lst, time_bins):
        spkts = np.asarray(lst, dtype=np.float32)
        bins = np.digitize(spkts, time_bins)

    spk_bin_dict = { ind: make_spkbin(lst, time_bins) for (ind, lst) in viewitems(spkdict) }
    acv_dict = { ind: np.asarray([ len(bin) for bin in bins ]) for ind, bins in viewitems(spkdict) }
    
    coactive_dict = defaultdict(set)
    for i, acv_i in viewitems(acv_dict):
        for j, acv_j in viewitems(acv_dict):
            if i != j:
                if np.dot(acv_i, acv_j) > 0:
                    coactive_dict[i].add(j)
                    coactive_dict[j].add(i)
    apvs = np.column_stack([ np.asarray([ acv_dict[ind] > 0.1 ], dtype=np.bool) for ind in sorted(acv_dict.keys()) ])
    
    tree = BallTree(apvs, leaf_size=2, metric='jaccard')
                    
    
    
    return tree
            
            

    



def histogram_correlation(spkdata, bin_size=1., quantity='count'):
    """Compute correlation coefficients of the spike count or firing rate histogram of each population. """

    spkpoplst        = spkdata['spkpoplst']
    spkindlst        = spkdata['spkindlst']
    spktlst          = spkdata['spktlst']
    num_cell_spks    = spkdata['num_cell_spks']
    pop_active_cells = spkdata['pop_active_cells']
    tmin             = spkdata['tmin']
    tmax             = spkdata['tmax']

    time_bins  = np.arange(tmin, tmax, bin_size)
    
    corr_dict = {}
    for subset, spkinds, spkts in zip(spkpoplst, spkindlst, spktlst):
        i = 0
        spk_dict = defaultdict(list)
        for spkind, spkt in zip(np.nditer(spkinds), np.nditer(spkts)):
            spk_dict[int(spkind)].append(spkt)
        x_lst = []
        for ind, lst in viewitems(spk_dict):
            spkts  = np.asarray(lst)
            if quantity == 'rate':
                q = akde(spkts / 1000., time_bins / 1000.)[0]
            else:
                count, bin_edges = np.histogram(spkts, bins = bins)
                q = count
            x_lst.append(q)
            i = i+1

        x_matrix = np.matrix(x_lst)
        
        corr_matrix = np.apply_along_axis(lambda y: mvcorrcoef(x_matrix, y), 1, x_matrix)
        corr_dict[subset] = corr_matrix
    
    return corr_dict


def histogram_autocorrelation(spkdata, bin_size=1., lag=1, quantity='count'):
    """Compute autocorrelation coefficients of the spike count or firing rate histogram of each population. """

    spkpoplst        = spkdata['spkpoplst']
    spkindlst        = spkdata['spkindlst']
    spktlst          = spkdata['spktlst']
    num_cell_spks    = spkdata['num_cell_spks']
    pop_active_cells = spkdata['pop_active_cells']
    tmin             = spkdata['tmin']
    tmax             = spkdata['tmax']

    bins  = np.arange(tmin, tmax, bin_size)
    
    corr_dict = {}
    for subset, spkinds, spkts in zip(spkpoplst, spkindlst, spktlst):
        i = 0
        spk_dict = defaultdict(list)
        for spkind, spkt in zip(np.nditer(spkinds), np.nditer(spkts)):
            spk_dict[int(spkind)].append(spkt)
        x_lst = []
        for ind, lst in viewitems(spk_dict):
            spkts  = np.asarray(lst)
            if quantity == 'rate':
                q = akde(spkts / 1000., time_bins / 1000.)[0]                
            else:
                count, bin_edges = np.histogram(spkts, bins = bins)
                q = count
            x_lst.append(q)
            i = i+1

        x_matrix = np.matrix(x_lst)
        
        corr_matrix = np.apply_along_axis(lambda y: autocorr(y, lag), 1, x_matrix)

        corr_dict[subset] = corr_matrix

    
    return corr_dict
