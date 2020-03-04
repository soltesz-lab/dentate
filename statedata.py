
import numpy as np
from mpi4py import MPI

from dentate.utils import get_module_logger, zip
from neuroh5.io import read_cell_attributes, read_cell_attribute_selection, read_cell_attribute_info

## This logger will inherit its setting from its root logger, dentate,
## which is created in module env
logger = get_module_logger(__name__)


def query_state(input_file, population_names, namespace_ids=None):

    pop_state_dict = {}

    logger.info('Reading state data...')

    attr_info_dict = read_cell_attribute_info(input_file, populations=population_names, read_cell_index=True)

    for pop_name in attr_info_dict:
        cell_index = None
        pop_state_dict[pop_name] = {}
        if namespace_ids is None:
            namespace_id_lst = attr_info_dict[pop_name].keys()
        else:
            namespace_id_lst = namespace_ids
    return namespace_id_lst, attr_info_dict


def read_state(input_file, population_names, namespace_id, time_variable='t', state_variable='v', time_range=None,
               max_units=None, gid=None, comm=None):
    if comm is None:
        comm = MPI.COMM_WORLD
    pop_state_dict = {}

    logger.info('Reading state data from populations %s, namespace %s...' % (str(population_names), namespace_id))

    attr_info_dict = read_cell_attribute_info(input_file, populations=population_names, read_cell_index=True)

    for pop_name in population_names:
        cell_index = None
        pop_state_dict[pop_name] = {}
        for attr_name, attr_cell_index in attr_info_dict[pop_name][namespace_id]:
            if state_variable == attr_name:
                cell_index = attr_cell_index

        cell_set = set(cell_index)
                
        # Limit to max_units
        if gid is None:
            if (max_units is not None) and (len(cell_set) > max_units):
                logger.info('  Reading only randomly sampled %i out of %i units for population %s' % (
                max_units, len(cell_set), pop_name))
                sample_inds = np.random.randint(0, len(cell_set) - 1, size=int(max_units))
                cell_set_lst = list(cell_set)
                gid = set([cell_set_lst[i] for i in sample_inds])
            else:
                gid = cell_set
        else:
            gid = set(gid)

        state_dict = {}
        if gid is None:
            valiter = read_cell_attributes(input_file, pop_name, namespace=namespace_id, comm=comm)
        else:
            valiter = read_cell_attribute_selection(input_file, pop_name, namespace=namespace_id,
                                                    selection=list(gid), comm=comm)

        if time_range is None:
            for cellind, vals in valiter:
                if cellind is not None:
                    distance = vals.get('distance', [None])[0]
                    section = vals.get('section', [None])[0]
                    loc = vals.get('loc', [None])[0]
                    tlst = []
                    vlst = []
                    for (t, v) in zip(vals[time_variable], vals[state_variable]):
                        tlst.append(t)
                        vlst.append(v)
                    state_dict[cellind] = (np.asarray(tlst, dtype=np.float32),
                                           np.asarray(vlst, dtype=np.float32),
                                           distance, section, loc)
        else:
            for cellind, vals in valiter:
                if cellind is not None:
                    distance = vals.get('distance', [None])[0]
                    section = vals.get('section', [None])[0]
                    loc = vals.get('loc', [None])[0]
                    tlst = []
                    vlst = []
                    for (t, v) in zip(vals[time_variable], vals[state_variable]):
                        if time_range[0] <= t <= time_range[1]:
                            tlst.append(t)
                            vlst.append(v)
                    state_dict[cellind] = (np.asarray(tlst, dtype=np.float32),
                                           np.asarray(vlst, dtype=np.float32),
                                           distance, section, loc)

        pop_state_dict[pop_name] = state_dict

    return {'states': pop_state_dict, 'time_variable': time_variable, 'state_variable': state_variable}
