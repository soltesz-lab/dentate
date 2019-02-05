

import sys, os
from collections import defaultdict, deque
import dlib
import numpy as np
import dentate
from dentate.neuron_utils import *
from dentate import utils, spikedata, synapses
from utils import viewitems, mpi_mean

# This logger will inherit its settings from the root logger, created in dentate.env
logger = utils.get_module_logger(__name__)


class NetworkOptimizer():
    """
    Network optimizer based on dlib's global function search

    `<http://dlib.net/optimization.html#global_function_search>`

    Creates a global optimizer for optimizing the network firing rate as a
    function of synaptic conductances.  
    """
    def __init__(self, env, dt_opt=125.0, fname=None, objective=lambda (env, opt_targets): -pop_firing_distance(env, opt_targets), objreduce=mpi_mean):

        """
        Default constructor for the network optimizer.

        :param dict opt_params: parameters to optimize over.

        :param str fname: File name for restoring and/or saving results,
        progress and settings.
        """
        
        if fname is None:
            if opt_params is None:
                raise ValueError("No file name and no parameters specified")
        else:
            if not os.path.isfile(fname):
                if opt_params is None:
                    raise FileNotFoundError(fname)

        self.env = env
        self.fname = fname
        self.dt_opt = dt_opt
        self.t_start = h.t
        
        self.param_range_tuples = defaultdict(list)
        self.opt_targets = {}
        self.specs = {}
        self.min_values = {}
        self.max_values = {}
        self.pop_index = {}

        opt_params = env.netclamp_config.optimize_parameters
        for i, (pop_name, params) in enumerate(viewitems(opt_params)):
            param_ranges = params['Parameter ranges']
            opt_target = params['Target firing rate']
            param_range_tuples = []
            for source, source_dict in sorted(viewitems(param_ranges), key=lambda (k,v): k):
                for sec_type, sec_type_dict in sorted(viewitems(source_dict), key=lambda (k,v): k):
                    for syn_name, syn_mech_dict in sorted(viewitems(sec_type_dict), key=lambda (k,v): k):
                        for param_name, param_range in sorted(viewitems(syn_mech_dict), key=lambda (k,v): k):
                            param_range_tuples.append((source, sec_type, syn_name, param_name, param_range))

            self.param_range_tuples[pop_name] = param_range_tuples
            self.opt_targets[pop_name] = opt_target
            self.pop_index[pop_name] = i


        min_values = [ param_range[0]
                        for _, param_range_tuples in viewitems(self.param_range_ruples)
                           for source, sec_type, syn_name, param_name, param_range in param_range_tuples ]
        max_values = [ param_range[1]
                        for _, param_range_tuples in viewitems(self.param_range_ruples)
                           for source, sec_type, syn_name, param_name, param_range in param_range_tuples ]
        spec = dlib.function_spec(bound1=min_values, bound2=max_values)
            
        old_evals = [] ## TODO load from file
        optimizer = dlib.global_function_search(spec, initial_function_evals=old_evals)
        
        self.optimizer = optimizer
        self.spec = spec
        self.opt_coords = None
        self.fih = h.FInitializeHandler(1, self.run)
        self.objective = objective
        self.objreduce = objreduce
        
    def run(self):
        """
        Simulate the network; pause at every dt_opt ms, evaluate the
        firing rates and select new set of parameters.
        """
        if self.opt_coords:
            this_coords = self.opt_coords
            local_obj = self.objective(self.env, self.opt_targets)
            global_obj = self.objreduce(self.env.comm, local_obj)
            this_coords.set(global_obj)
            specs, evals = self.optimizer.get_function_evaluations() 
            e = evals[0]
            
        next_coords = self.optimizer.get_next_x()
        for pop_index, pop_name in viewitems(self.pop_index):
            biophys_cell_dict = self.env.biophys_cells[pop_name]
            params_tuples = self.from_param_vector(pop_name, next_coords.x)
            for gid, biophys_cell in viewitems(biophys_cell_dict):
                for source, sec_type, syn_name, param_name, param_value in params_tuples:
                    synapses.modify_syn_param(biophys_cell, self.env, sec_type, syn_name,
                                              param_name=param_name, value=param_value,
                                              filters={'sources': [source]},
                                              origin='soma', update_targets=True)
                cell = self.env.pc.gid2cell(gid)
        self.opt_coords = next_coords
        
        ## Add another event to the event queue, to 
        ## execute run again, dt_opt ms from now
        h.cvode.event(h.t + self.dt_opt, self.run)
        self.t_start = h.t

        
    def from_param_vector(self, pop_name, params):
        """
        Given a list of parameter values, use param_range_tuples
        structure to construct tuples of the form
        (source, section_type, syn_name, param_name, param_value).
        """
        result = []
        param_range_tuples = self.param_range_tuples[pop_name]
        assert(len(params) == len(param_range_tuples))
        for i, (source, sec_type, syn_name, param_name, param_range) in enumerate(param_range_tuples):
            result.append((source, sec_type, syn_name, param_name, params[i]))
        return result

    
    def to_param_vector(self, pop_name, params):
        """
        Given a list of parameter tuples, return a list of parameter values.
        """
        result = []
        for (source, sec_type, syn_name, param_name, param_value) in params:
            result.append(param_value)
        return result

    
def pop_firing_rates(env):
    """
    Computes the mean firing rate for each population in the network.
    """
    pop_spike_dict = spikedata.get_env_spike_dict(env)
    
    rate_dict = { pop_name: spikedata.spike_rates (spike_dict)
                  for pop_name, spike_dict in viewitems(pop_spike_dict) }

    return rate_dict
    
def pop_firing_distance(env, opt_targets):
    """
    Computes the distance vector between target firing rates
    and actual mean firing rates of the populations in the network.
    """

    rate_dict = pop_firing_rates(env)

    a = np.asarray([ opt_targets[pop_name] for pop_name in sorted(self.pop_index.keys()) ])
    sqdist = np.asarray([(a[self.pop_index[pop_name]] - rate) ** 2 for _, rate in viewitems(rate_dict[pop_name])
                        for pop_name in sorted(self.pop_index.keys()) ])
    
    return sqdist


def mpi_mean(comm, value):
    
    global_sum = np.zeros(value.shape, dtype='float32')
    comm.Allreduce(value, global_sum, op=MPI.SUM)
    count = comm.size
    return global_sum / count
