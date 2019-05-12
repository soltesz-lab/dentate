import math
import itertools
from collections import defaultdict
from mpi4py import MPI
import numpy as np
from neuroh5.io import NeuroH5CellAttrGen, read_cell_attribute_info, read_population_ranges, read_population_names
from dentate import utils

## This logger will inherit its setting from its root logger, dentate,
## which is created in module env
logger = utils.get_module_logger(__name__)

def read_state(input_file, population_names, namespace_id, time_variable='t', variable='v', time_range = None, max_units = None, unit_no = None, query = False, comm = None):

    if comm is None:
        comm = MPI.COMM_WORLD
    pop_state_dict = {}

    logger.info('Reading state data...')

    attr_info_dict = read_cell_attribute_info(input_file, populations=population_names, read_cell_index=True)

    if query:
        print(attr_info_dict)
        return
        
    for pop_name in population_names:
        cell_index = None
        pop_state_dict[pop_name] = {}
        for attr_name, attr_cell_index in attr_info_dict[pop_name][namespace_id]:
            if variable == attr_name: 
                cell_index = attr_cell_index
                
        # Limit to max_units
        if unit_no is None:
            if (max_units is not None) and (len(cell_index)>max_units):
                logger.info('  Reading only randomly sampled %i out of %i units for population %s' % (max_units, len(cell_index), pop_name))
                sample_inds = np.random.randint(0, len(cell_index)-1, size=int(max_units))
                unit_no      = set([cell_index[i] for i in sample_inds])
            else:
                unit_no      = set(cell_index)
        else:
            unit_no = set(unit_no)

        state_dict = {}
        valiter    = NeuroH5CellAttrGen(input_file, pop_name, namespace=namespace_id, comm=comm)
        
        if time_range is None:
            if unit_no is None:
                for cellind, vals in valiter:
                    if cellind is not None:
                        tlst = []
                        vlst = []
                        for (t,v) in zip(vals[time_variable], vals[variable]):
                            tlst.append(t)
                            vlst.append(v)
                        state_dict[cellind] = (np.asarray(tlst,dtype=np.float32), np.asarray(vlst,dtype=np.float32))
            else:
                for cellind, vals in valiter:
                    if (cellind is not None) and (cellind in unit_no):
                        tlst = []
                        vlst = []
                        for (t,v) in zip(vals[time_variable], vals[variable]):
                            tlst.append(t)
                            vlst.append(v)
                        state_dict[cellind] = (np.asarray(tlst,dtype=np.float32), np.asarray(vlst,dtype=np.float32))
                
        else:
            if unit_no is None:
                for cellind, vals in valiter:
                    if cellind is not None:
                        tlst = []
                        vlst = []
                        for (t,v) in zip(vals[time_variable], vals[variable]):
                            if time_range[0] <= t <= time_range[1]:
                                tlst.append(t)
                                vlst.append(v)
                        state_dict[cellind] = (np.asarray(tlst,dtype=np.float32), np.asarray(vlst,dtype=np.float32))
            else:
                for cellind, vals in valiter:
                    if cellind is not None:
                        if cellind in unit_no:
                            tlst = []
                            vlst = []
                            for (t,v) in zip(vals[time_variable], vals[variable]):
                                if time_range[0] <= t <= time_range[1]:
                                    tlst.append(t)
                                    vlst.append(v)
                            state_dict[cellind] = (np.asarray(tlst,dtype=np.float32), np.asarray(vlst,dtype=np.float32))

        pop_state_dict[pop_name] = state_dict

                
        
    return {'states': pop_state_dict, 'time_variable': time_variable, 'variable': variable }


