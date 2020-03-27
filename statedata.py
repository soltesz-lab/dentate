
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
               max_units=None, gid=None, comm=None, n_trials=-1):
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
                gid_set = set([cell_set_lst[i] for i in sample_inds])
            else:
                gid_set = cell_set
        else:
            gid_set = set(gid)

        state_dict = {}
        if gid is None:
            valiter = read_cell_attributes(input_file, pop_name, namespace=namespace_id, comm=comm)
        else:
            valiter = read_cell_attribute_selection(input_file, pop_name, namespace=namespace_id,
                                                    selection=list(gid_set), comm=comm)

        if time_range is None:
            for cellind, vals in valiter:
                if cellind is not None:
                    trial_dur = vals.get('trial duration', None)
                    distance = vals.get('distance', [None])[0]
                    section = vals.get('section', [None])[0]
                    loc = vals.get('loc', [None])[0]
                    tvals = np.asarray(vals[time_variable], dtype=np.float32)
                    svals = np.asarray(vals[state_variable], dtype=np.float32)
                    trial_bounds = np.where(np.isclose(tvals, tvals[0], atol=1e-4))[0]
                    if n_trials == -1 or n_trials > 1:
                        state_dict[cellind] = (np.split(tvals, trial_bounds[1:n_trials]),
                                               np.split(svals, trial_bounds[1:n_trials]),
                                               distance, section, loc)
                    else:
                        state_dict[cellind] = ([tvals[:trial_bounds[1]]],
                                               [svals[:trial_bounds[1]]],
                                               distance, section, loc)
                        
        else:
            for cellind, vals in valiter:
                if cellind is not None:
                    distance = vals.get('distance', [None])[0]
                    section = vals.get('section', [None])[0]
                    loc = vals.get('loc', [None])[0]
                    tinds = np.argwhere(vals[time_variable] <= time_range[1] &
                                        vals[time_variable] >= time_range[0])
                    tvals = np.asarray(vals[time_variable][tinds], dtype=np.float32)
                    svals = np.asarray(vals[state_variable][tinds], dtype=np.float32)
                    trial_bounds = np.where(np.isclose(tvals, tvals[0], atol=1e-4))[0][1:]
                    if n_trials == -1 or n_trials > 1:
                        state_dict[cellind] = (np.split(tvals, trial_bounds[1:n_trials]),
                                               np.split(svals, trial_bounds[1:n_trials]),
                                               distance, section, loc)
                    else:
                        state_dict[cellind] = ([tvals[:trial_bounds[1]]],
                                               [svals[:trial_bounds[1]]],
                                               distance, section, loc)


        pop_state_dict[pop_name] = state_dict

    return {'states': pop_state_dict, 'time_variable': time_variable, 'state_variable': state_variable}
