import math
import itertools
from collections import defaultdict
import numpy as np
from neuroh5.io import NeuroH5CellAttrGen, read_cell_attribute_info, read_population_ranges, read_population_names

def read_state(comm, input_file, population_names, namespace_id, timeVariable='t', variable='v', timeRange = None, maxUnits = None, unitNo = None, query = False, verbose = False):

    pop_state_dict = {}

    if verbose:
        print('Reading state data...')

    attr_info_dict = read_cell_attribute_info(input_file, populations=list(population_names), read_cell_index=True)

    if query:
        print attr_info_dict
        return
        
    for pop_name in population_names:
        cell_index = None
        pop_state_dict[pop_name] = {}
        for attr_name, attr_cell_index in attr_info_dict[pop_name][namespace_id]:
            if variable == attr_name: 
                cell_index = attr_cell_index
                
        # Limit to maxUnits
        if unitNo is None:
            if (maxUnits is not None) and (len(cell_index)>maxUnits):
                if verbose:
                    print('  Reading only randomly sampled %i out of %i units for population %s' % (maxUnits, len(cell_index), pop_name))
                sample_inds = np.random.randint(0, len(cell_index)-1, size=int(maxUnits))
                unitNo      = set([cell_index[i] for i in sample_inds])
            else:
                unitNo      = set(cell_index)
        else:
            unitNo = set(unitNo)

        print unitNo
            
        state_dict = {}
        valiter    = NeuroH5CellAttrGen(input_file, pop_name, namespace=namespace_id, comm=comm)
        
        if timeRange is None:
            if unitNo is None:
                for cellind, vals in valiter:
                    if cellind is not None:
                        tlst = []
                        vlst = []
                        for (t,v) in zip(vals[timeVariable], vals[variable]):
                            tlst.append(t)
                            vlst.append(v)
                        state_dict[cellind] = (np.asarray(tlst,dtype=np.float32), np.asarray(vlst,dtype=np.float32))
            else:
                for cellind, vals in valiter:
                    if (cellind is not None) and (cellind in unitNo):
                        tlst = []
                        vlst = []
                        for (t,v) in zip(vals[timeVariable], vals[variable]):
                            tlst.append(t)
                            vlst.append(v)
                        state_dict[cellind] = (np.asarray(tlst,dtype=np.float32), np.asarray(vlst,dtype=np.float32))
                
        else:
            if unitNo is None:
                for cellind, vals in valiter:
                    if cellind is not None:
                        tlst = []
                        vlst = []
                        for (t,v) in zip(vals[timeVariable], vals[variable]):
                            if timeRange[0] <= t <= timeRange[1]:
                                tlst.append(t)
                                vlst.append(v)
                        state_dict[cellind] = (np.asarray(tlst,dtype=np.float32), np.asarray(vlst,dtype=np.float32))
            else:
                for cellind, vals in valiter:
                    if cellind is not None:
                        if cellind in unitNo:
                            tlst = []
                            vlst = []
                            for (t,v) in zip(vals[timeVariable], vals[variable]):
                                if timeRange[0] <= t <= timeRange[1]:
                                    tlst.append(t)
                                    vlst.append(v)
                            state_dict[cellind] = (np.asarray(tlst,dtype=np.float32), np.asarray(vlst,dtype=np.float32))

        pop_state_dict[pop_name] = state_dict

                
        
    return {'states': pop_state_dict, 'timeVariable': timeVariable, 'variable': variable }


