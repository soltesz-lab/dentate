import math
import itertools
from collections import defaultdict
import numpy as np
from neuroh5.io import read_cell_attributes, read_population_ranges, read_population_names

def read_state(comm, input_file, population_names, namespace_id, timeVariable='t', variable='v', timeRange = None, maxUnits = None, unitNo = None, verbose = False):

    pop_state_dict = {}

    if verbose:
        print('Reading state data...')

    for pop_name in population_names:

        state_dict = {}
        valiter    = read_cell_attributes(comm, input_file, pop_name, namespace=namespace_id)
        
        if timeRange is None:
            if unitNo is None:
                for cellind, vals in valiter:
                    tlst = []
                    vlst = []
                    for (t,v) in itertools.izip(vals[timeVariable], vals[variable]):
                        tlst.append(t)
                        vlst.append(v)
                    state_dict[cellind] = (np.asarray(tlst,dtype=np.float32), np.asarray(vlst,dtype=np.float32))
            else:
                for cellind, vals in valiter:
                    if cellind == unitNo:
                        tlst = []
                        vlst = []
                        for (t,v) in itertools.izip(vals[timeVariable], vals[variable]):
                            tlst.append(t)
                            vlst.append(v)
                        state_dict[cellind] = (np.asarray(tlst,dtype=np.float32), np.asarray(vlst,dtype=np.float32))
                
        else:
            if unitNo is None:
                for cellind, vals in valiter:
                    tlst = []
                    vlst = []
                    for (t,v) in itertools.izip(vals[timeVariable], vals[variable]):
                        if timeRange[0] <= t <= timeRange[1]:
                            tlst.append(t)
                            vlst.append(v)
                    state_dict[cellind] = (np.asarray(tlst,dtype=np.float32), np.asarray(vlst,dtype=np.float32))
            else:
                for cellind, vals in valiter:
                    if cellind == unitNo:
                        tlst = []
                        vlst = []
                        for (t,v) in itertools.izip(vals[timeVariable], vals[variable]):
                            if timeRange[0] <= t <= timeRange[1]:
                                tlst.append(t)
                                vlst.append(v)
                        state_dict[cellind] = (np.asarray(tlst,dtype=np.float32), np.asarray(vlst,dtype=np.float32))

                
        # Limit to maxUnits
        if (unitNo is None) and (maxUnits is not None) and (len(state_dict)>maxUnits):
            if verbose:
                print('  Reading only randomly sampled %i out of %i units for population %s' % (maxUnits, len(state_dict), pop_name))
            sample_inds     = np.random.randint(0, len(state_dict)-1, size=int(maxUnits))
            state_dict_keys = state_dict.keys()
            sample_keys     = [state_dict_keys[i] for i in sample_inds]
            pop_state_dict[pop_name] = { k: state_dict[k] for k in sample_keys }
        else:
            pop_state_dict[pop_name] = state_dict
        
    return {'states': pop_state_dict, 'timeVariable': timeVariable, 'variable': variable }


