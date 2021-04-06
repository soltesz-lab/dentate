import sys, math, copy
from collections import defaultdict
import numpy as np
from scipy import interpolate
from neuroh5.io import scatter_read_cell_attributes, read_cell_attributes, read_population_names, read_population_ranges, write_cell_attributes
import dentate
from dentate.utils import get_module_logger, Struct, autocorr, baks, consecutive, mvcorrcoef, viewitems, zip, get_trial_time_ranges

## This logger will inherit its setting from its root logger, dentate,
## which is created in module env
logger = get_module_logger(__name__)

# Default spike analysis configuration
default_baks_analysis_options = Struct(**{'BAKS Alpha': 4.77,
                                          'BAKS Beta': None})
default_pf_analysis_options = Struct(**{'Minimum Width': 10.,
                                        'Minimum Rate': None})


def get_env_spike_dict(env, include_artificial=True):
    """
    Constructs  a dictionary with per-gid per-trial spike times from the output vectors with spike times and gids contained in env.
    """
    equilibration_duration = float(env.stimulus_config['Equilibration Duration'])
    n_trials = env.n_trials

    t_vec = np.array(env.t_vec, dtype=np.float32)
    id_vec = np.array(env.id_vec, dtype=np.uint32)

    trial_time_ranges = get_trial_time_ranges(env.t_rec.to_python(), env.n_trials)
    trial_time_bins = [ t_trial_start for t_trial_start, t_trial_end in trial_time_ranges ] 
    trial_dur = np.asarray([env.tstop + equilibration_duration] * n_trials, dtype=np.float32)
    
    binlst = []
    typelst = sorted(env.celltypes.keys())
    binvect = np.asarray([env.celltypes[k]['start'] for k in typelst ])
    sort_idx = np.argsort(binvect, axis=0)
    pop_names = [typelst[i] for i in sort_idx]
    bins = binvect[sort_idx][1:]
    inds = np.digitize(id_vec, bins)

    pop_spkdict = {}
    for i, pop_name in enumerate(pop_names):
        spkdict = {}
        sinds = np.where(inds == i)
        if len(sinds) > 0:
            ids = id_vec[sinds]
            ts = t_vec[sinds]
            for j in range(0, len(ids)):
                gid = ids[j]
                t = ts[j]
                if (not include_artificial) and (gid in env.artificial_cells[pop_name]):
                    continue
                if gid in spkdict:
                   spkdict[gid].append(t)
                else:
                   spkdict[gid] = [t]
            for gid in spkdict:
                spiketrain = np.array(spkdict[gid], dtype=np.float32)
                if gid in env.spike_onset_delay:
                    spiketrain -= env.spike_onset_delay[gid]
                trial_bins = np.digitize(spiketrain, trial_time_bins) - 1
                trial_spikes = [np.copy(spiketrain[np.where(trial_bins == trial_i)[0]])
                                for trial_i in range(env.n_trials)]
                for trial_i, trial_spiketrain in enumerate(trial_spikes):
                    trial_spiketrain -= np.sum(trial_dur[:(trial_i)]) + equilibration_duration
                spkdict[gid] = trial_spikes
        pop_spkdict[pop_name] = spkdict

    return pop_spkdict


def read_spike_events(input_file, population_names, namespace_id, spike_train_attr_name='t', time_range=None,
                      max_spikes=None, n_trials=-1, merge_trials=False, comm=None, io_size=0, include_artificial=True):
    """
    Reads spike trains from a NeuroH5 file, and returns a dictionary with spike times and cell indices.
    :param input_file: str (path to file)
    :param population_names: list of str
    :param namespace_id: str
    :param spike_train_attr_name: str
    :param time_range: list of float
    :param max_spikes: float
    :param n_trials: int
    :param merge_trials: bool
    :return: dict
    """
    assert((n_trials >= 1) | (n_trials == -1))

    trial_index_attr = 'Trial Index'
    trial_dur_attr = 'Trial Duration'
    artificial_attr = 'artificial'
    
    spkpoplst = []
    spkindlst = []
    spktlst = []
    spktrials = []
    num_cell_spks = {}
    pop_active_cells = {}

    tmin = float('inf')
    tmax = 0.

    for pop_name in population_names:

        if time_range is None or time_range[1] is None:
            logger.info('Reading spike data for population %s...' % pop_name)
        else:
            logger.info('Reading spike data for population %s in time range %s...' % (pop_name, str(time_range)))

        spike_train_attr_set = set([spike_train_attr_name, trial_index_attr, trial_dur_attr, artificial_attr])
        spkiter_dict = scatter_read_cell_attributes(input_file, pop_name, namespaces=[namespace_id], 
                                                    mask=spike_train_attr_set, comm=comm, io_size=io_size)
        spkiter = spkiter_dict[namespace_id]
        
        this_num_cell_spks = 0
        active_set = set([])

        pop_spkindlst = []
        pop_spktlst = []
        pop_spktriallst = []

        logger.info('Read spike cell attributes for population %s...' % pop_name)

        # Time Range
        if time_range is not None:
            if time_range[0] is None:
                time_range[0] = 0.0

        for spkind, spkattrs in spkiter:
            is_artificial_flag = spkattrs.get(artificial_attr, None)
            is_artificial = (is_artificial_flag[0] > 0) if is_artificial_flag is not None else None
            if is_artificial is not None:
                if is_artificial and (not include_artificial):
                    continue
            slen = len(spkattrs[spike_train_attr_name])
            trial_dur = spkattrs.get(trial_dur_attr, np.asarray([0.]))
            trial_ind = spkattrs.get(trial_index_attr, np.zeros((slen,),dtype=np.uint8))
            if n_trials == -1:
                n_trials = len(set(trial_ind))
            for spk_i, spkt in enumerate(spkattrs[spike_train_attr_name]):
                    trial_i = trial_ind[spk_i]
                    if trial_i >= n_trials:
                        continue
                    if time_range is not None:
                       if not ((spkt >= time_range[0]) and (spkt <= time_range[1])):
                           continue
                    if merge_trials:
                        spkt += np.sum(trial_dur[:trial_i])
                    pop_spkindlst.append(spkind)
                    pop_spktlst.append(spkt)
                    pop_spktriallst.append(trial_i)
                    if spkt < tmin:
                        tmin = spkt
                    if spkt > tmax:
                        tmax = spkt
                    this_num_cell_spks += 1
                    active_set.add(spkind)

        pop_active_cells[pop_name] = active_set
        num_cell_spks[pop_name] = this_num_cell_spks

        if not active_set:
            continue

        pop_spkts = np.asarray(pop_spktlst, dtype=np.float32)
        del (pop_spktlst)
        pop_spkinds = np.asarray(pop_spkindlst, dtype=np.uint32)
        del (pop_spkindlst)
        pop_spktrials = np.asarray(pop_spktriallst, dtype=np.uint32)
        del (pop_spktriallst)

        # Limit to max_spikes
        if (max_spikes is not None) and (len(pop_spkts) > max_spikes):
            logger.warn(' Reading only randomly sampled %i out of %i spikes for population %s' %
                        (max_spikes, len(pop_spkts), pop_name))
            sample_inds = np.random.randint(0, len(pop_spkinds) - 1, size=int(max_spikes))
            pop_spkts = pop_spkts[sample_inds]
            pop_spkinds = pop_spkinds[sample_inds]
            pop_spktrials = pop_spkinds[sample_inds]
            tmax = max(tmax, max(pop_spkts))

        spkpoplst.append(pop_name)
        pop_trial_spkindlst = []
        pop_trial_spktlst = []
        for trial_i in range(n_trials):
            trial_idxs = np.where(pop_spktrials == trial_i)[0]
            sorted_trial_idxs = np.argsort(pop_spkts[trial_idxs])
            pop_trial_spktlst.append(np.take(pop_spkts[trial_idxs], sorted_trial_idxs))
            pop_trial_spkindlst.append(np.take(pop_spkinds[trial_idxs], sorted_trial_idxs))
                
        del pop_spkts
        del pop_spkinds
        del pop_spktrials

        if merge_trials:
            pop_spkinds = np.concatenate(pop_trial_spkindlst)
            pop_spktlst = np.concatenate(pop_trial_spktlst)
            spkindlst.append(pop_spkinds)
            spktlst.append(pop_spktlst)
        else:
            spkindlst.append(pop_trial_spkindlst)
            spktlst.append(pop_trial_spktlst)
            

        logger.info(' Read %i spikes and %i trials for population %s' % (this_num_cell_spks, n_trials, pop_name))

    return {'spkpoplst': spkpoplst, 'spktlst': spktlst, 'spkindlst': spkindlst,
            'tmin': tmin, 'tmax': tmax,
            'pop_active_cells': pop_active_cells, 'num_cell_spks': num_cell_spks,
            'n_trials': n_trials}


def make_spike_dict(spkinds, spkts):
    """
    Given arrays with cell indices and spike times, returns a dictionary with per-cell spike times.
    """
    spk_dict = defaultdict(list)
    for spkind, spkt in zip(np.nditer(spkinds), np.nditer(spkts)):
        spk_dict[int(spkind)].append(float(spkt))
    return spk_dict


def interspike_intervals(spkdict):
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

    
def spike_bin_counts(spkdict, time_bins):
    bin_dict = {}
    for (ind, lst) in viewitems(spkdict):

        if len(lst) > 0:
            spkts = np.asarray(lst, dtype=np.float32)
            bins, bin_edges = np.histogram(spkts, bins=time_bins)
            
            bin_dict[ind] = bins

    return bin_dict



def spike_rates(spkdict):
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


def spike_covariate(population, spkdict, time_bins, nbins_before, nbins_after):
    """
    Creates the spike covariate matrix.

    X: a matrix of size nbins x nadj x ncells
    """
    
    spk_matrix = np.column_stack([ np.histogram(np.asarray(lst), bins=time_bins)[0]
                                   for i, (gid, lst) in enumerate(viewitems(spkdict[population])) if len(lst) > 1 ])

    nbins  = spk_matrix.shape[0]
    ncells = spk_matrix.shape[1]
    nadj   = nbins_before+nbins_after+1
    
    X      = np.empty([nbins, nadj, ncells]) 
    X[:]   = np.NaN
    
    start_idx=0
    for i in range(nbins-nbins_before-nbins_after): 
        end_idx=start_idx+nadj
        X[i+nbins_before,:,:] = spk_matrix[start_idx:end_idx,:] 
        start_idx=start_idx+1
        
    return X



def spike_density_estimate(population, spkdict, time_bins, arena_id=None, trajectory_id=None, output_file_path=None,
                            progress=False, inferred_rate_attr_name='Inferred Rate Map', **kwargs):
    """
    Calculates spike density function for the given spike trains.
    :param population:
    :param spkdict:
    :param time_bins:
    :param arena_id: str
    :param trajectory_id: str
    :param output_file_path:
    :param progress:
    :param inferred_rate_attr_name: str
    :param kwargs: dict
    :return: dict
    """
    if progress:
        from tqdm import tqdm

    analysis_options = copy.copy(default_baks_analysis_options)
    analysis_options.update(kwargs)

    def make_spktrain(lst, t_start, t_stop):
        spkts = np.asarray(lst, dtype=np.float32)
        return spkts[(spkts >= t_start) & (spkts <= t_stop)]

    
    t_start = time_bins[0]
    t_stop = time_bins[-1]

    spktrains = {ind: make_spktrain(lst, t_start, t_stop) for (ind, lst) in viewitems(spkdict)}
    baks_args = dict()
    baks_args['a'] = analysis_options['BAKS Alpha']
    baks_args['b'] = analysis_options['BAKS Beta']
    
    if progress:
        seq = tqdm(viewitems(spktrains))
    else:
        seq = viewitems(spktrains)
        
    spk_rate_dict = {ind: baks(spkts / 1000., time_bins / 1000., **baks_args)[0].reshape((-1,))
                     if len(spkts) > 1 else np.zeros(time_bins.shape)
                     for ind, spkts in seq}

    if output_file_path is not None:
        if arena_id is None or trajectory_id is None:
            raise RuntimeError('spike_density_estimate: arena_id and trajectory_id required to write Spike Density'
                               'Function namespace')
        namespace = 'Spike Density Function %s %s' % (arena_id, trajectory_id)
        attr_dict = {ind: {inferred_rate_attr_name: np.asarray(spk_rate_dict[ind], dtype='float32')}
                     for ind in spk_rate_dict}
        write_cell_attributes(output_file_path, population, attr_dict, namespace=namespace)

    result = {ind: {'rate': rate, 'time': time_bins} for ind, rate in viewitems(spk_rate_dict)}

        
    result = { ind: { 'rate': rate, 'time': time_bins }
              for ind, rate in viewitems(spk_rate_dict) }
    
    return result



def spatial_information(population, trajectory, spkdict, time_range, position_bin_size, arena_id=None,
                        trajectory_id=None, output_file_path=None, information_attr_name='Mutual Information',
                        progress=False, **kwargs):
    """
    Calculates mutual information for the given spatial trajectory and spike trains.
    :param population:
    :param trajectory:
    :param spkdict:
    :param time_range:
    :param position_bin_size:
    :param arena_id: str
    :param trajectory_id: str
    :param output_file_path: str (path to file)
    :param information_attr_name: str
    :return: dict
    """
    tmin = time_range[0]
    tmax = time_range[1]

    x, y, d, t = trajectory

    t_inds = np.where((t >= tmin) & (t <= tmax))
    t = t[t_inds]
    d = d[t_inds]

    d_extent = np.max(d) - np.min(d)
    position_bins = np.arange(np.min(d), np.max(d), position_bin_size)
    d_bin_inds = np.digitize(d, bins=position_bins)
    t_bin_ind_lst = [0]
    for ibin in range(1, len(position_bins) + 1):
        bin_inds = np.where(d_bin_inds == ibin)
        t_bin_ind_lst.append(np.max(bin_inds))
    t_bin_inds = np.asarray(t_bin_ind_lst)
    time_bins = t[t_bin_inds]

    d_bin_probs = {}
    prev_bin = np.min(d)
    for ibin in range(1, len(position_bins) + 1):
        d_bin = d[d_bin_inds == ibin]
        if d_bin.size > 0:
            bin_max = np.max(d_bin)
            d_prob = (bin_max - prev_bin) / d_extent
            d_bin_probs[ibin] = d_prob
            prev_bin = bin_max
        else:
            d_bin_probs[ibin] = 0.

    rate_bin_dict = spike_density_estimate(population, spkdict, time_bins, arena_id=arena_id,
                                           trajectory_id=trajectory_id, output_file_path=output_file_path,
                                           progress=progress, **kwargs)

    MI_dict = {}
    for ind, valdict in viewitems(rate_bin_dict):
        MI = 0.
        x = valdict['time']
        rates = valdict['rate']
        R = np.mean(rates)

        if R > 0.:
            for ibin in range(1, len(position_bins) + 1):
                p_i = d_bin_probs[ibin]
                R_i = rates[ibin - 1]
                if R_i > 0.:
                    MI += p_i * (R_i / R) * math.log((R_i / R), 2)

        MI_dict[ind] = MI

    if output_file_path is not None:
        if arena_id is None or trajectory_id is None:
            raise RuntimeError('spikedata.spatial_information: arena_id and trajectory_id required to write Spatial '
                               'Mutual Information namespace')
        namespace = 'Spatial Mutual Information %s %s' % (arena_id, trajectory_id)
        attr_dict = {ind: {information_attr_name: np.array(MI_dict[ind], dtype='float32')} for ind in MI_dict}
        write_cell_attributes(output_file_path, population, attr_dict, namespace=namespace)

    return MI_dict



def place_fields(population, bin_size, rate_dict, trajectory, arena_id=None, trajectory_id=None, nstdev=1.5,
                 binsteps=5, baseline_fraction=None, output_file_path=None, progress=False, **kwargs):
    """
    Estimates place fields from the given instantaneous spike rate dictionary.
    :param population: str
    :param bin_size: float
    :param rate_dict: dict
    :param trajectory: tuple of array
    :param arena_id: str
    :param trajectory_id: str
    :param nstdev: float
    :param binsteps: float
    :param baseline_fraction: float
    :param min_pf_width: float
    :param output_file_path: str (path to file)
    :param verbose: bool
    :return: dict
    """

    if progress:
        from tqdm import tqdm

    analysis_options = copy.copy(default_pf_analysis_options)
    analysis_options.update(kwargs)

    min_pf_width = analysis_options['Minimum Width']
    min_pf_rate = analysis_options['Minimum Rate']     

    (trj_x, trj_y, trj_d, trj_t) = trajectory

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
        t      = valdict['time']
        rate   = valdict['rate']
        m      = np.mean(rate)
        rate1  = np.subtract(rate, m)
        if baseline_fraction is None:
            s = np.std(rate1)
        else:
            k = rate1.shape[0] / baseline_fraction
            s = np.std(rate1[np.argpartition(rate1, k)[:k]])
        tmin = t[0]
        tmax = t[-1]
        bins = np.arange(tmin, tmax, bin_size)
        bin_rates = []
        bin_norm_rates = []
        pf_ibins = []
        for ibin in range(1, len(bins)):
            binx = np.linspace(bins[ibin - 1], bins[ibin], binsteps)
            interp_rate1 = np.interp(binx, t, np.asarray(rate1, dtype=np.float64))
            interp_rate = np.interp(binx, t, np.asarray(rate, dtype=np.float64))
            r_n = np.mean(interp_rate1)
            r = np.mean(interp_rate)
            bin_rates.append(r)
            bin_norm_rates.append(r_n)
            if r_n > nstdev * s:
                pf_ibins.append(ibin - 1)
                
        bin_rates = np.asarray(bin_rates)
        bin_norm_rates = np.asarray(bin_norm_rates)

        if len(pf_ibins) > 0:
            pf_consecutive_ibins = []
            pf_consecutive_bins = []
            pf_widths = []
            pf_rates = []
            for pf_ibin_array in consecutive(pf_ibins):
                pf_ibin_range = np.asarray([np.min(pf_ibin_array), np.max(pf_ibin_array)])
                pf_bin_range = np.asarray([bins[pf_ibin_range[0]], bins[pf_ibin_range[1]]])
                pf_bin_rates = [bin_rates[ibin] for ibin in pf_ibin_array]
                pf_width = np.diff(np.interp(pf_bin_range, trj_t, trj_d))[0]
                pf_consecutive_ibins.append(pf_ibin_range)
                pf_consecutive_bins.append(pf_bin_range)
                pf_widths.append(pf_width)
                pf_rates.append(np.mean(pf_bin_rates))
                
            if min_pf_rate is None:
                pf_filtered_ibins = [pf_consecutive_ibins[i] for i, pf_width in enumerate(pf_widths)
                                    if pf_width >= min_pf_width]
            else:
                pf_filtered_ibins = [pf_consecutive_ibins[i] for i, (pf_width, pf_rate) in enumerate(zip(pf_widths,pf_rates))
                                    if (pf_width >= min_pf_width) and (pf_rate >= min_pf_rate)]
                
            

            pf_count = len(pf_filtered_ibins)
            pf_ibins = [list(range(pf_ibin[0], pf_ibin[1] + 1)) for pf_ibin in pf_filtered_ibins]
            pf_mean_width = []
            pf_mean_rate = []
            pf_peak_rate = []
            pf_mean_norm_rate = []
            pf_x_locs = []
            pf_y_locs = []
            for pf_ibin_iter in pf_ibins:
                pf_ibin_array = list(pf_ibin_iter)
                pf_ibin_range = np.asarray([np.min(pf_ibin_array), np.max(pf_ibin_array)])
                pf_bin_range = np.asarray([bins[pf_ibin_range[0]], bins[pf_ibin_range[1]]])
                pf_mean_width.append(np.mean(
                    np.asarray([pf_width for pf_width in pf_widths if pf_width >= min_pf_width])))
                pf_mean_rate.append(np.mean(np.asarray(bin_rates[pf_ibin_array])))
                pf_peak_rate.append(np.max(np.asarray(bin_rates[pf_ibin_array])))
                pf_mean_norm_rate.append(np.mean(np.asarray(bin_norm_rates[pf_ibin_array])))
                pf_x_range = np.interp(pf_bin_range, trj_t, trj_x)
                pf_y_range = np.interp(pf_bin_range, trj_t, trj_y)
                pf_x_locs.append(np.mean(pf_x_range))
                pf_y_locs.append(np.mean(pf_y_range))
                
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
            pf_x_locs = []
            pf_y_locs = []

        cell_count += 1
        pf_dict[ind] = {'pf_count': np.asarray([pf_count], dtype=np.uint32),
                        'pf_mean_width': np.asarray(pf_mean_width, dtype=np.float32),
                        'pf_mean_rate': np.asarray(pf_mean_rate, dtype=np.float32),
                        'pf_peak_rate': np.asarray(pf_peak_rate, dtype=np.float32),
                        'pf_mean_norm_rate': np.asarray(pf_mean_norm_rate, dtype=np.float32),
                        'pf_x_locs': np.asarray(pf_x_locs),
                        'pf_y_locs': np.asarray(pf_y_locs)}

    logger.info('%s place fields: %i cells min %i max %i mean %f\n' %
                    (population, cell_count, pf_min, pf_max, float(pf_total_count) / float(cell_count)))
    if output_file_path is not None:
        if arena_id is None or trajectory_id is None:
            raise RuntimeError('spikedata.place_fields: arena_id and trajectory_id required to write %s namespace' %
                               'Place Fields')
        namespace = 'Place Fields %s %s' % (arena_id, trajectory_id)
        write_cell_attributes(output_file_path, population, pf_dict, namespace=namespace)

    return pf_dict


def coactive_sets (population, spkdict, time_bins, return_tree=False):
    """
    Estimates co-active activity ensembles from the given spike dictionary.
    """

    import sklearn
    from sklearn.neighbors import BallTree
    
    acv_dict = { gid: np.histogram(np.asarray(lst), bins=time_bins)[0] 
                 for (gid, lst) in viewitems(spkdict[population]) if len(lst) > 1 }
    n_features = len(time_bins)-1
    n_samples = len(acv_dict)

    active_gid = {}
    active_bins = np.zeros((n_samples, n_features),dtype=np.bool)
    for i, (gid, acv) in enumerate(viewitems(acv_dict)):
        active_bins[i,:] = acv > 0
        active_gid[i] = gid
    
    tree = BallTree(active_bins, metric='jaccard')
    qbins = np.zeros((n_features, n_features),dtype=np.bool)
    for ibin in range(n_features):
        qbins[ibin,ibin] = True

    nnrs, nndists = tree.query_radius(qbins, r=1, return_distance=True)

    fnnrs = []
    fnndists = []
    for i, (nns, nndist) in enumerate(zip(nnrs, nndists)):
        inds = [ inn for inn, nn in enumerate(nns) if np.any(np.logical_and(active_bins[nn,:], active_bins[i,:])) ] 
        fnns = np.asarray([ nns[inn] for inn in inds ])
        fdist = np.asarray([ nndist[inn] for inn in inds ])
        fnnrs.append(fnns)
        fnndists.append(fdist)

    if return_tree:
        return n_samples, fnnrs, fnndists, (tree, active_gid)
    else:
        return n_samples, fnnrs, fnndists


def spatial_coactive_sets (population, spkdict, time_bins, trajectory, return_tree=False):
    """
    Estimates spatially co-active activity ensembles from the given spike dictionary.
    """

    import sklearn
    from sklearn.neighbors import BallTree

    x, y, d, t = trajectory

    pch_x = interpolate.pchip(t, x)
    pch_y = interpolate.pchip(t, y)

    spatial_bins = np.column_stack([pch_x(time_bins[:-1]), pch_y(time_bins[:-1])])
    
    acv_dict = { gid: np.histogram(np.asarray(lst), bins=time_bins)[0] 
                 for (gid, lst) in viewitems(spkdict[population]) if len(lst) > 1 }
    n_features = len(time_bins)-1
    n_samples = len(acv_dict)

    active_gid = {}
    active_bins = np.zeros((n_samples, n_features),dtype=np.bool)
    for i, (gid, acv) in enumerate(viewitems(acv_dict)):
        active_bins[i,:] = acv > 0
        active_gid[i] = gid
    
    tree = BallTree(active_bins, metric='jaccard')
    qbins = np.zeros((n_features, n_features),dtype=np.bool)
    for ibin in range(n_features):
        qbins[ibin,ibin] = True

    nnrs, nndists = tree.query_radius(qbins, r=1, return_distance=True)

    fnnrs = []
    fnndists = []
    for i, (nns, nndist) in enumerate(zip(nnrs, nndists)):
        inds = [ inn for inn, nn in enumerate(nns) if np.any(np.logical_and(active_bins[nn,:], active_bins[i,:])) ] 
        fnns = np.asarray([ nns[inn] for inn in inds ])
        fdist = np.asarray([ nndist[inn] for inn in inds ])
        fnnrs.append(fnns)
        fnndists.append(fdist)

    if return_tree:
        return n_samples, spatial_bins, fnnrs, fnndists, (tree, active_gid)
    else:
        return n_samples, spatial_bins, fnnrs, fnndists



def histogram_correlation(spkdata, bin_size=1., quantity='count'):
    """Compute correlation coefficients of the spike count or firing rate histogram of each population. """

    spkpoplst = spkdata['spkpoplst']
    spkindlst = spkdata['spkindlst']
    spktlst = spkdata['spktlst']
    num_cell_spks = spkdata['num_cell_spks']
    pop_active_cells = spkdata['pop_active_cells']
    tmin = spkdata['tmin']
    tmax = spkdata['tmax']

    time_bins = np.arange(tmin, tmax, bin_size)

    corr_dict = {}
    for subset, spkinds, spkts in zip(spkpoplst, spkindlst, spktlst):
        i = 0
        spk_dict = defaultdict(list)
        for spkind, spkt in zip(np.nditer(spkinds), np.nditer(spkts)):
            spk_dict[int(spkind)].append(spkt)
        x_lst = []
        for ind, lst in viewitems(spk_dict):
            spkts = np.asarray(lst)
            if quantity == 'rate':
                q = akde(spkts / 1000., time_bins / 1000.)[0]
            else:
                count, bin_edges = np.histogram(spkts, bins=bins)
                q = count
            x_lst.append(q)
            i = i + 1

        x_matrix = np.matrix(x_lst)

        corr_matrix = np.apply_along_axis(lambda y: mvcorrcoef(x_matrix, y), 1, x_matrix)
        corr_dict[subset] = corr_matrix

    return corr_dict


def histogram_autocorrelation(spkdata, bin_size=1., lag=1, quantity='count'):
    """Compute autocorrelation coefficients of the spike count or firing rate histogram of each population. """

    spkpoplst = spkdata['spkpoplst']
    spkindlst = spkdata['spkindlst']
    spktlst = spkdata['spktlst']
    num_cell_spks = spkdata['num_cell_spks']
    pop_active_cells = spkdata['pop_active_cells']
    tmin = spkdata['tmin']
    tmax = spkdata['tmax']

    bins = np.arange(tmin, tmax, bin_size)

    corr_dict = {}
    for subset, spkinds, spkts in zip(spkpoplst, spkindlst, spktlst):
        i = 0
        spk_dict = defaultdict(list)
        for spkind, spkt in zip(np.nditer(spkinds), np.nditer(spkts)):
            spk_dict[int(spkind)].append(spkt)
        x_lst = []
        for ind, lst in viewitems(spk_dict):
            spkts = np.asarray(lst)
            if quantity == 'rate':
                q = akde(spkts / 1000., time_bins / 1000.)[0]
            else:
                count, bin_edges = np.histogram(spkts, bins=bins)
                q = count
            x_lst.append(q)
            i = i + 1

        x_matrix = np.matrix(x_lst)

        corr_matrix = np.apply_along_axis(lambda y: autocorr(y, lag), 1, x_matrix)

        corr_dict[subset] = corr_matrix

    return corr_dict
