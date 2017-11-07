
import itertools
import numpy as np
from neuroh5.io import read_cell_attributes, read_population_ranges, read_population_names

def read_spike_events(comm, input_file, population_names, namespace_id, timeVariable='t', timeRange = None, maxSpikes = int(1e6), verbose = False):

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
        if (len(pop_spkts)>maxSpikes):
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



def compute_activity_ratio(stimulus, response, binSize = 25.):

    
    
    result = numpy.power(numpy.sum(numpy.array(mean_rates),axis=0)/nstim,2) / (numpy.sum(numpy.power(mean_rates,2),axis=0)/nstim)

    return result

