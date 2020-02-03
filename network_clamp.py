"""
Routines for Network Clamp simulation.
"""
import os, sys, copy, uuid, pprint
from collections import defaultdict
from mpi4py import MPI
import numpy as np
import click
from dentate import io_utils, spikedata, synapses, stimulus, cell_clamp
from dentate.cells import h, make_input_cell, register_cell, record_cell
from dentate.env import Env
from dentate.neuron_utils import h, configure_hoc_env, make_rec
from dentate.utils import Closure, mse, list_find, list_index, range, str, viewitems, zip_longest, get_module_logger
from dentate.cell_clamp import init_biophys_cell
from neuroh5.io import read_cell_attribute_selection

# This logger will inherit its settings from the root logger, created in dentate.env
logger = get_module_logger(__name__)


def generate_weights(env, weight_source_rules, this_syn_attrs):
    """
    Generates synaptic weights according to the rules specified in the
    Weight Generator section of network clamp configuration.
    """
    weights_dict = {}

    if len(weight_source_rules) > 0:

        for presyn_id, weight_rule in viewitems(weight_source_rules):
            source_syn_dict = defaultdict(list)

            for syn_id, syn in viewitems(this_syn_attrs):
                this_presyn_id = syn.source.population
                this_presyn_gid = syn.source.gid
                if this_presyn_id == presyn_id:
                    source_syn_dict[this_presyn_gid].append(syn_id)

            if weight_rule['class'] == 'Sparse':
                weights_name = weight_rule['name']
                rule_params = weight_rule['params']
                fraction = rule_params['fraction']
                seed_offset = int(env.model_config['Random Seeds']['Sparse Weights'])
                seed = int(seed_offset + 1)
                weights_dict[presyn_id] = \
                    synapses.generate_sparse_weights(weights_name, fraction, seed, source_syn_dict)
            elif weight_rule['class'] == 'Log-Normal':
                weights_name = weight_rule['name']
                rule_params = weight_rule['params']
                mu = rule_params['mu']
                sigma = rule_params['sigma']
                clip = None
                if 'clip' in rule_params:
                    clip = rule_params['clip']
                seed_offset = int(env.model_config['Random Seeds']['GC Log-Normal Weights 1'])
                seed = int(seed_offset + 1)
                weights_dict[presyn_id] = \
                    synapses.generate_log_normal_weights(weights_name, mu, sigma, seed, source_syn_dict, clip=clip)
            elif weight_rule['class'] == 'Normal':
                weights_name = weight_rule['name']
                rule_params = weight_rule['params']
                mu = rule_params['mu']
                sigma = rule_params['sigma']
                seed_offset = int(env.model_config['Random Seeds']['GC Normal Weights'])
                seed = int(seed_offset + 1)
                weights_dict[presyn_id] = \
                    synapses.generate_normal_weights(weights_name, mu, sigma, seed, source_syn_dict)
            else:
                raise RuntimeError('network_clamp.generate_weights: unknown weight generator rule class %s' % \
                                   weight_rule['class'])

    return weights_dict



def init(env, pop_name, gid, spike_events_path, generate_inputs_pops=set([]), generate_weights_pops=set([]),
         spike_events_namespace='Spike Events', t_var='t', t_min=None, t_max=None, write_cell=False, plot_cell=False):
    """
    Instantiates a cell and all its synapses and connections and loads
    or generates spike times for all synaptic connections.

    :param env: an instance of env.Env
    :param pop_name: population name
    :param gid: gid
    :param spike_events_path:

    """
    io_utils.mkout(env, env.results_file_path)

    env.cell_selection = {}
    
    ## If specified, presynaptic spikes that only fall within this time range
    ## will be loaded or generated
    if t_max is None:
        t_range = None
    else:
        if t_min is None:
            t_range = [0.0, t_max]
        else:
            t_range = [t_min, t_max]

    ## Attribute namespace that contains recorded spike events
    namespace_id = spike_events_namespace

    ## Determine presynaptic populations that connect to this cell type
    presyn_names = env.projection_dict[pop_name]

    ## Load cell gid and its synaptic attributes and connection data
    cell = init_biophys_cell(env, pop_name, gid, write_cell=write_cell)

    ## Load spike times of presynaptic cells
    spkdata = spikedata.read_spike_events(spike_events_path, \
                                          presyn_names, \
                                          spike_events_namespace, \
                                          spike_train_attr_name=t_var, \
                                          time_range=t_range)
    spkindlst = spkdata['spkindlst']
    spktlst = spkdata['spktlst']
    spkpoplst = spkdata['spkpoplst']

    ## Organize spike times by index of presynaptic population and gid
    input_source_dict = {}
    weight_source_dict = {}
    for presyn_name in presyn_names:
        presyn_index = int(env.Populations[presyn_name])
        spk_pop_index = list_index(presyn_name, spkpoplst)
        if spk_pop_index is None:
            logger.warning("No spikes found for population %s in file %s" % (presyn_name, spike_events_path))
            continue
        spk_inds = spkindlst[spk_pop_index]
        spk_ts = spktlst[spk_pop_index]
        spk_ts += float(env.stimulus_config['Equilibration Duration'])

        
        if presyn_name in generate_inputs_pops:
            if (presyn_name in env.netclamp_config.input_generators):
                spike_generator = env.netclamp_config.input_generators[presyn_name]
            else:
                raise RuntimeError('network_clamp.init: no input generator specified for population %s' % presyn_name)
        else:
            spike_generator = None

        input_source_dict[presyn_index] = {'gen': spike_generator,
                                           'spiketrains': {'gid': spk_inds, 't': spk_ts, }}

        if presyn_name in generate_weights_pops:
            if (presyn_name in env.netclamp_config.weight_generators[pop_name]):
                weight_rule = env.netclamp_config.weight_generators[pop_name][presyn_name]
            else:
                raise RuntimeError(
                    'network_clamp.init: no weights generator rule specified for population %s' % presyn_name)
        else:
            weight_rule = None

        if weight_rule is not None:
            weight_source_dict[presyn_index] = weight_rule

    min_delay = float('inf')
    syn_attrs = env.synapse_attributes
    this_syn_attrs = syn_attrs[gid]
    weight_params = defaultdict(dict)
    for syn_id, syn in viewitems(this_syn_attrs):
        presyn_id = syn.source.population
        presyn_gid = syn.source.gid
        delay = syn.source.delay
        if presyn_id in input_source_dict:
            ## Load presynaptic spike times into the VecStim for each synapse;
            ## if spike_generator_dict contains an entry for the respective presynaptic population,
            ## then use the given generator to generate spikes.
            if not (presyn_gid in env.gidset):
                cell = make_input_cell(env, presyn_gid, presyn_id, input_source_dict)
                register_cell(env, presyn_id, presyn_gid, cell)

    source_weight_params = generate_weights(env, weight_source_dict, this_syn_attrs)

    for presyn_id, weight_params in viewitems(source_weight_params):
        weights_syn_ids = weight_params['syn_id']
        for syn_name in (syn_name for syn_name in weight_params if syn_name != 'syn_id'):
            weights_values = weight_params[syn_name]
            syn_attrs.add_mech_attrs_from_iter(gid, syn_name, \
                                               zip_longest(weights_syn_ids, \
                                                           [{'weight': x} for x in weights_values]))
    synapses.config_biophys_cell_syns(env, gid, pop_name, insert=True, insert_netcons=True, verbose=True)
    record_cell(env, pop_name, gid)

    if plot_cell:
        import dentate.plot
        from dentate.plot import plot_synaptic_attribute_distribution
        syn_attrs = env.synapse_attributes
        biophys_cell = env.biophys_cells[pop_name][gid]
        syn_name = 'AMPA'
        syn_mech_name = syn_attrs.syn_mech_names[syn_name]
        for param_name in ['weight', 'g_unit']:
            param_label = '%s; %s; %s' % (syn_name, syn_mech_name, param_name)
            plot_synaptic_attribute_distribution(biophys_cell, env, syn_name, param_name, filters=None, from_mech_attrs=True,
                                                 from_target_attrs=True, param_label=param_label,
                                                 export='syn_params_%d.h5' % gid, description='network_clamp', show=False,
                                                 svg_title="Synaptic parameters for gid %d" % (gid),
                                                 output_dir=env.results_path)
        
        
    
    cell = env.pc.gid2cell(gid)
    for sec in list(cell.all):
        h.psection(sec=sec)

        
    env.pc.set_maxstep(10)
    h.stdinit()


def run(env):
    """
    Runs network clamp simulation. Assumes that procedure `init` has been
    called with the network configuration provided by the `env`
    argument.

    :param env:
    """

    rank = int(env.pc.id())
    nhosts = int(env.pc.nhost())

    if env.recording_profile is not None:
        rec_dt = env.recording_profile.get('dt', 0.1) 
        env.t_rec.record(h._ref_t, rec_dt)
    env.t_vec.resize(0)
    env.id_vec.resize(0)

    h.cvode_active(0)

    h.t = 0.0
    h.dt = env.dt
    h.tstop = env.tstop
    if 'Equilibration Duration' in env.stimulus_config:
        h.tstop += float(env.stimulus_config['Equilibration Duration'])
    h.finitialize(env.v_init)

    if rank == 0:
        logger.info("*** Running simulation with dt = %.03f and tstop = %.02f" % (h.dt, h.tstop))

    env.pc.barrier()
    env.pc.psolve(h.tstop)

    if rank == 0:
        logger.info("*** Simulation completed")
    env.pc.barrier()

    comptime = env.pc.step_time()
    cwtime = comptime + env.pc.step_wait()
    maxcw = env.pc.allreduce(cwtime, 2)
    avgcomp = env.pc.allreduce(comptime, 1) / nhosts
    maxcomp = env.pc.allreduce(comptime, 2)

    if rank == 0:
        logger.info("Host %i  ran simulation in %g seconds" % (rank, comptime))

    env.pc.runworker()
    env.pc.done()


def run_with(env, param_dict):
    """
    Runs network clamp simulation with the specified parameters for
    the given gid(s).  Assumes that procedure `init` has been called with
    the network configuration provided by the `env` argument.

    :param env:
    :param param_dict: dictionary { gid: params }

    """

    rank = int(env.pc.id())
    nhosts = int(env.pc.nhost())

    for pop_name, gid_param_dict in viewitems(param_dict):
        biophys_cell_dict = env.biophys_cells[pop_name]
        for gid, params_tuples in viewitems(gid_param_dict):
            biophys_cell = biophys_cell_dict[gid]
            for update_operator, destination, source, sec_type, syn_name, param_path, param_value in params_tuples:
                if isinstance(param_path, tuple):
                    p, s = param_path
                else:
                    p = param_path

                sources = None
                if isinstance(source, tuple):
                    sources = list(source)
                else:
                    if source is not None:
                        sources = [source]
                synapses.modify_syn_param(biophys_cell, env, sec_type, syn_name,
                                          param_name=p, value=param_value,
                                          filters={'sources': sources} if sources is not None else None,
                                          update_operator=update_operator,
                                          origin='soma', update_targets=True)
            cell = env.pc.gid2cell(gid)

    if env.recording_profile is not None:
        rec_dt = env.recording_profile.get('dt', 0.1) 
        env.t_rec.record(h._ref_t, rec_dt)

    env.t_vec.resize(0)
    env.id_vec.resize(0)

    h.cvode_active(0)

    h.t = 0.0
    h.tstop = env.tstop
    if 'Equilibration Duration' in env.stimulus_config:
        h.tstop += float(env.stimulus_config['Equilibration Duration'])

    h.dt = env.dt
    h.finitialize(env.v_init)

    if rank == 0:
        logger.info("*** Running simulation with dt = %.03f and tstop = %.02f" % (h.dt, h.tstop))
        
        logger.info("*** Parameters: %s" % pprint.pformat(param_dict))

    env.pc.barrier()
    env.pc.psolve(h.tstop)

    if rank == 0:
        logger.info("*** Simulation completed")
    env.pc.barrier()

    comptime = env.pc.step_time()
    cwtime = comptime + env.pc.step_wait()
    maxcw = env.pc.allreduce(cwtime, 2)
    avgcomp = env.pc.allreduce(comptime, 1) / nhosts
    maxcomp = env.pc.allreduce(comptime, 2)

    if rank == 0:
        logger.info("Host %i  ran simulation in %g seconds" % (rank, comptime))

    env.pc.runworker()
    env.pc.done()

    return spikedata.get_env_spike_dict(env, include_artificial=None)


def make_firing_rate_target(env, pop_name, gid, target_rate, from_param_vector):
    def gid_firing_rate(spkdict, gid):
        if gid in spkdict[pop_name]:
            spkdict1 = {gid: spkdict[pop_name][gid]}
        else:
            spkdict1 = {gid: np.asarray([], dtype=np.float32)}
        rate_dict = spikedata.spike_rates(spkdict1)
        if gid in spkdict[pop_name]:
            logger.info('firing rate objective: spikes times of gid %i: %s' % (gid, pprint.pformat(spkdict[pop_name][gid])))
        logger.info('firing rate objective: rate of gid %i is %.2f' % (gid, rate_dict[gid]))
        return rate_dict[gid]['rate']

    f = lambda *v: (abs(gid_firing_rate(run_with(env, {pop_name: {gid: from_param_vector(v)}}), gid) - target_rate))

    return f


def make_firing_rate_vector_target(env, pop_name, gid, target_rate_vector, time_bins, from_param_vector):
    def gid_firing_rate_vector(spkdict, gid):
        if gid in spkdict[pop_name]:
            spkdict1 = {gid: spkdict[pop_name][gid]}
        else:
            spkdict1 = {gid: np.asarray([], dtype=np.float32)}
        rate_dict = spikedata.spike_rates(spkdict1)
        spike_density_dict = spikedata.spike_density_estimate (pop_name, spkdict1, time_bins)
        if gid in spkdict[pop_name]:
            rate = spike_density_dict[gid]['rate']
            logger.info('firing rate objective: spike times of gid %i: %s' % (gid, str(spkdict[pop_name][gid])))
            logger.info('firing rate objective: firing rate of gid %i: %s' % (gid, str(rate)))
            logger.info('firing rate objective: min/max rates of gid %i are %.2f / %.2f Hz' % (gid, np.min(rate), np.max(rate)))
        return spike_density_dict[gid]['rate']
    logger.info("firing rate objective: target time bins: %s" % str(time_bins))
    logger.info("firing rate objective: target vector: %s" % str(target_rate_vector))
    logger.info("firing rate objective: target rate vector min/max is %.2f Hz (%.2f ms) / %.2f Hz (%.2f ms)" % (np.min(target_rate_vector), time_bins[np.argmin(target_rate_vector)], np.max(target_rate_vector), time_bins[np.argmax(target_rate_vector)]))
    f = lambda *v: (mse(gid_firing_rate_vector(run_with(env, {pop_name: {gid: from_param_vector(v)}}), gid), target_rate_vector))

    return f


def modify_scaled_syn_param(env, gid, syn_id, old_val, new_val):
    syn_name = env.syn_name
    syn_index = env.syn_index
    mech_name = env.mech_name
    syn = env.syn_id_attr_dict[gid][syn_id]
    syn_attr_dict = syn.attr_dict[syn_index]
    if env.param_name in syn_attr_dict:
        original_val = syn_attr_dict[env.param_name]
    else:
        original_val = old_val
        syn_attr_dict[env.param_name] = original_val
        logger.warning('modify_scaled_syn_param: gid %d syn %s is missing parameter %s; using value %s' % (gid, str(syn), env.param_name, str(old_val)))
    return original_val * new_val

def optimize_params(env, pop_name, param_type):
                        
    param_bounds = {}
    param_names = []
    param_initial_dict = {}
    param_range_tuples = []

    if param_type == 'synaptic':
        if (pop_name in env.netclamp_config.optimize_parameters['synaptic']):
            opt_params = env.netclamp_config.optimize_parameters['synaptic'][pop_name]
            param_ranges = opt_params['Parameter ranges']
        else:
            raise RuntimeError(
                "network_clamp.optimize_params: population %s does not have optimization configuration" % pop_name)
        update_operator = None
        keyfun = lambda kv: str(kv[0])
        for source, source_dict in sorted(viewitems(param_ranges), key=keyfun):
            for sec_type, sec_type_dict in sorted(viewitems(source_dict), key=keyfun):
                for syn_name, syn_mech_dict in sorted(viewitems(sec_type_dict), key=keyfun):
                    for param_fst, param_rst in sorted(viewitems(syn_mech_dict), key=keyfun):
                        if isinstance(param_rst, dict):
                            for const_name, const_range in sorted(viewitems(param_rst)):
                                param_path = (param_fst, const_name)
                                param_range_tuples.append((update_operator, pop_name, source, sec_type, syn_name, param_path, const_range))
                                param_key = '%s.%s.%s.%s.%s.%s' % (pop_name, str(source), sec_type, syn_name, param_fst, const_name)
                                param_initial_value = (const_range[1] - const_range[0]) / 2.0
                                param_initial_dict[param_key] = param_initial_value
                                param_bounds[param_key] = const_range
                                param_names.append(param_key)
                        else:
                            param_name = param_fst
                            param_range = param_rst
                            param_range_tuples.append((update_operator, pop_name, source, sec_type, syn_name, param_name, param_range))
                            param_key = '%s.%s.%s.%s.%s' % (pop_name, source, sec_type, syn_name, param_name)
                            param_initial_value = (param_range[1] - param_range[0]) / 2.0
                            param_initial_dict[param_key] = param_initial_value
                            param_bounds[param_key] = param_range
                            param_names.append(param_key)
                            
    elif param_type == 'scaling':
        if (pop_name in env.netclamp_config.optimize_parameters['scaling']):
            opt_params = env.netclamp_config.optimize_parameters['scaling'][pop_name]
            param_ranges = opt_params['Parameter ranges']
        else:
            raise RuntimeError(
                "network_clamp.optimize_params: population %s does not have optimization configuration" % pop_name)

        syn_attrs = env.synapse_attributes
        copy_syn_id_attr_dict = copy.deepcopy(syn_attrs.syn_id_attr_dict)

        keyfun = lambda kv: str(kv[0])
        for source, source_dict in sorted(viewitems(param_ranges), key=keyfun):
            for sec_type, sec_type_dict in sorted(viewitems(source_dict), key=keyfun):
                for syn_name, syn_param_dict in sorted(viewitems(sec_type_dict), key=keyfun):
                    for param_fst, param_rst in sorted(viewitems(syn_param_dict), key=keyfun):
                        if isinstance(param_rst, dict):
                            raise RuntimeError("network_clamp.optimize_params: dependent parameter expressions are not supported for parameter type %s" % param_type)
                        param_name = param_fst
                        param_range = param_rst
                        update_operator = Closure(modify_scaled_syn_param, param_name = param_name,
                                                  syn_name = syn_name, syn_index = syn_attrs.syn_name_index_dict[syn_name],
                                                  mech_name = syn_attrs.syn_mech_names[syn_name], 
                                                  syn_id_attr_dict = copy_syn_id_attr_dict) 
                        param_key = '%s.%s.%s.%s.%s' % (pop_name, str(source), sec_type, syn_name, param_name)
                        param_initial_value = (param_range[1] - param_range[0]) / 2.0
                        param_initial_dict[param_key] = param_initial_value
                        param_bounds[param_key] = param_range
                        param_names.append(param_key)
                        param_range_tuples.append((update_operator, pop_name, source, sec_type, syn_name, param_name, param_range))
                
        
    else:
        raise RuntimeError(
                "network_clamp.optimize_params: unknown parameter type %s" % param_type)

    return param_bounds, param_names, param_initial_dict, param_range_tuples
    
    
def optimize_rate(env, pop_name, gid, opt_iter=10, param_type='synaptic'):
    import dlib

    if (pop_name in env.netclamp_config.optimize_parameters):
        opt_params = env.netclamp_config.optimize_parameters[pop_name]
        param_ranges = opt_params['Parameter ranges']
        opt_target = opt_params['Targets']['firing rate']
    else:
        raise RuntimeError(
            "network_clamp.optimize_rate: population %s does not have optimization configuration" % pop_name)


    param_bounds, param_names, param_initial_dict, param_range_tuples = optimize_params(env, pop_name, param_type)

    def from_param_vector(params):
        result = []
        assert (len(params) == len(param_range_tuples))
        for i, (update_operator, pop_name, source, sec_type, syn_name, param_name, param_range) in enumerate(param_range_tuples):
            result.append((pop_name, source, sec_type, syn_name, param_name, params[i]))
        return result

    def to_param_vector(params):
        result = []
        for (update_operator, destination, source, sec_type, syn_name, param_name, param_value) in params:
            result.append(param_value)
        return result

    min_values = [(source, sec_type, syn_name, param_name, param_range[0]) for
                  update_operator, pop_name, source, sec_type, syn_name, param_name, param_range in param_range_tuples]
    max_values = [(source, sec_type, syn_name, param_name, param_range[1]) for
                  update_operator, pop_name, source, sec_type, syn_name, param_name, param_range in param_range_tuples]
    
    f_firing_rate = make_firing_rate_target(env, pop_name, gid, opt_target, time_bins, from_param_vector)
    opt_params, outputs = dlib.find_min_global(f_firing_rate, to_param_vector(min_values), to_param_vector(max_values),
                                               opt_iter)

    logger.info('Optimized parameters: %s' % pprint.pformat(from_param_vector(opt_params)))
    logger.info('Optimized objective function: %s' % pprint.pformat(outputs))

    return opt_params, outputs


def optimize_rate_dist(env, tstop, pop_name, gid, 
                       target_rate_map_path, target_rate_map_namespace,
                       target_rate_map_arena, target_rate_map_trajectory,
                       opt_iter=10, param_type='synaptic'):
    import dlib

    time_step = env.stimulus_config['Temporal Resolution']
    equilibration_duration = float(env.stimulus_config['Equilibration Duration'])

    input_namespace = '%s %s %s' % (target_rate_map_namespace, target_rate_map_arena, target_rate_map_trajectory)
    it = read_cell_attribute_selection(target_rate_map_path, pop_name, namespace=input_namespace,
                                        selection=[gid], mask=set(['Trajectory Rate Map']))
    trj_rate_map = dict(it)[gid]['Trajectory Rate Map']

    trj_x, trj_y, trj_d, trj_t = stimulus.read_trajectory(target_rate_map_path, target_rate_map_arena, target_rate_map_trajectory)

    time_range = (0., min(np.max(trj_t), tstop))
    
    interp_trj_t = np.arange(time_range[0], time_range[1], time_step)
    interp_trj_rate_map = np.interp(interp_trj_t, trj_t, trj_rate_map)
    
    param_bounds, param_names, param_initial_dict, param_range_tuples = optimize_params(env, pop_name, param_type)
    
    def from_param_vector(params):
        result = []
        assert (len(params) == len(param_range_tuples))
        for i, (update_operator, pop_name, source, sec_type, syn_name, param_name, param_range) in enumerate(param_range_tuples):
            result.append((update_operator, pop_name, source, sec_type, syn_name, param_name, params[i]))
        return result

    def to_param_vector(params):
        result = []
        for (update_operator, destination, source, sec_type, syn_name, param_name, param_value) in params:
            result.append(param_value)
        return result

    min_values = [(update_operator, pop_name, source, sec_type, syn_name, param_name, param_range[0]) for
                  update_operator, pop_name, source, sec_type, syn_name, param_name, param_range in
                      param_range_tuples]
    max_values = [(update_operator, pop_name, source, sec_type, syn_name, param_name, param_range[1]) for
                  update_operator, pop_name, source, sec_type, syn_name, param_name, param_range in
                      param_range_tuples]
    
    f_firing_rate_vector = make_firing_rate_vector_target(env, pop_name, gid, interp_trj_rate_map, interp_trj_t, from_param_vector)
    opt_params, outputs = dlib.find_min_global(f_firing_rate_vector, to_param_vector(min_values), to_param_vector(max_values),
                                               opt_iter)

    logger.info('Optimized parameters: %s' % pprint.pformat(from_param_vector(opt_params)))
    logger.info('Optimized objective function: %s' % pprint.pformat(outputs))

    return opt_params, outputs


def write_output(env):
    rank = env.comm.rank
    if rank == 0:
        logger.info("*** Writing spike data")
    io_utils.spikeout(env, env.results_file_path)
    if rank == 0:
        logger.info("*** Writing intracellular data")
    io_utils.recsout(env, env.results_file_path)


@click.group()
def cli():
    pass


@click.command()
@click.option("--config-file", '-c', required=True, type=str, help='model configuration file name')
@click.option("--population", '-p', required=True, type=str, default='GC', help='target population')
@click.option("--gid", '-g', required=True, type=int, default=0, help='target cell gid')
@click.option("--template-paths", type=str, required=True,
              help='colon-separated list of paths to directories containing hoc cell templates')
@click.option("--dataset-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              help='path to directory containing required neuroh5 data files')
@click.option("--config-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='config',
              help='path to directory containing network and cell mechanism config files')
@click.option("--results-path", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True), \
              help='path to directory where output files will be written')
@click.option("--spike-events-path", '-s', required=True, type=click.Path(),
              help='path to neuroh5 file containing spike times')
@click.option("--spike-events-namespace", type=str, default='Spike Events',
              help='namespace containing spike times')
@click.option("--spike-events-t", required=False, type=str, default='t',
              help='name of variable containing spike times')
@click.option('--plot-cell', is_flag=True, help='plot the distribution of weight and g_unit synaptic parameters')
@click.option('--write-cell', is_flag=True, help='write out selected cell tree morphology and connections')
@click.option('--profile-memory', is_flag=True, help='calculate and print heap usage after the simulation is complete')
@click.option('--recording-profile', type=str, default='Network clamp default', help='recording profile to use')

def show(config_file, population, gid, template_paths, dataset_prefix, config_prefix, results_path,
         spike_events_path, spike_events_namespace, spike_events_t, plot_cell, write_cell, profile_memory, recording_profile):
    """
    Show configuration for the specified cell.
    """

    comm = MPI.COMM_WORLD
    np.seterr(all='raise')

    verbose = True
    params = dict(locals())
    env = Env(**params)
    configure_hoc_env(env)

    init(env, population, gid, spike_events_path, \
         spike_events_namespace=spike_events_namespace, \
         t_var=spike_events_t, plot_cell=plot_cell, write_cell=write_cell)

    if env.profile_memory:
        profile_memory(logger)


@click.command()
@click.option("--config-file", '-c', required=True, type=str, help='model configuration file name')
@click.option("--population", '-p', required=True, type=str, default='GC', help='target population')
@click.option("--gid", '-g', required=True, type=int, default=0, help='target cell gid')
@click.option("--generate-inputs", '-e', required=False, type=str, multiple=True,
              help='generate spike trains for the given presynaptic population')
@click.option("--generate-weights", '-w', required=False, type=str, multiple=True,
              help='generate weights for the given presynaptic population')
@click.option("--tstop", '-t', type=float, default=150.0, help='simulation end time')
@click.option("--t-max", type=float)
@click.option("--t-min", type=float)
@click.option("--template-paths", type=str, required=True,
              help='colon-separated list of paths to directories containing hoc cell templates')
@click.option("--dataset-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              help='path to directory containing required neuroh5 data files')
@click.option("--config-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='config',
              help='path to directory containing network and cell mechanism config files')
@click.option("--spike-events-path", '-s', required=True, type=click.Path(),
              help='path to neuroh5 file containing spike times')
@click.option("--spike-events-namespace", type=str, default='Spike Events',
              help='namespace containing spike times')
@click.option("--spike-events-t", required=False, type=str, default='t',
              help='name of variable containing spike times')
@click.option("--results-path", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True), \
              help='path to directory where output files will be written')
@click.option("--results-file-id", type=str, required=False, default=None, \
              help='identifier that is used to name neuroh5 files that contain output spike and intracellular trace data')
@click.option("--results-namespace-id", type=str, required=False, default=None, \
              help='identifier that is used to name neuroh5 namespaces that contain output spike and intracellular trace data')
@click.option('--plot-cell', is_flag=True, help='plot the distribution of weight and g_unit synaptic parameters')
@click.option('--write-cell', is_flag=True, help='write out selected cell tree morphology and connections')
@click.option('--profile-memory', is_flag=True, help='calculate and print heap usage after the simulation is complete')
@click.option('--recording-profile', type=str, default='Network clamp default', help='recording profile to use')

def go(config_file, population, gid, generate_inputs, generate_weights, tstop, t_max, t_min,
       template_paths, dataset_prefix,
       config_prefix, spike_events_path, spike_events_namespace, spike_events_t,
       results_path, results_file_id, results_namespace_id, plot_cell, write_cell, profile_memory, recording_profile):

    """
    Runs network clamp simulation for the specified cell.
    """

    if results_file_id is None:
        results_file_id = uuid.uuid4()
    comm = MPI.COMM_WORLD
    np.seterr(all='raise')
    verbose = True
    params = dict(locals())
    env = Env(**params)
    configure_hoc_env(env)

    init(env, population, gid, spike_events_path, \
         generate_inputs_pops=set(generate_inputs), \
         generate_weights_pops=set(generate_weights), \
         spike_events_namespace=spike_events_namespace, \
         t_var=spike_events_t, t_min=t_min, t_max=t_max,
         plot_cell=plot_cell, write_cell=write_cell)

    run(env)
    write_output(env)

    if env.profile_memory:
        profile_memory(logger)


@click.command()
@click.option("--config-file", '-c', required=True, type=str, help='model configuration file name')
@click.option("--population", '-p', required=True, type=str, default='GC', help='target population')
@click.option("--gid", '-g', required=True, type=int, default=0, help='target cell gid')
@click.option("--generate-inputs", '-e', required=False, type=str, multiple=True,
              help='generate spike trains for the given presynaptic population')
@click.option("--generate-weights", '-w', required=False, type=str, multiple=True,
              help='generate weights for the given presynaptic population')
@click.option("--tstop", '-t', type=float, default=150.0, help='simulation end time')
@click.option("--t-max", type=float)
@click.option("--t-min", type=float)
@click.option("--opt-iter", type=int, default=10, help='number of optimization iterations')
@click.option("--template-paths", type=str, required=True,
              help='colon-separated list of paths to directories containing hoc cell templates')
@click.option("--dataset-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              help='path to directory containing required neuroh5 data files')
@click.option("--config-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='config',
              help='path to directory containing network and cell mechanism config files')
@click.option("--param-type", type=str, 
              help='parameter type for rate optimization (synaptic or scaling)')
@click.option('--recording-profile', type=str, help='recording profile to use')
@click.option("--results-path", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True), \
              help='path to directory where output files will be written')
@click.option("--spike-events-path", '-s', required=True, type=click.Path(),
              help='path to neuroh5 file containing spike times')
@click.option("--spike-events-namespace", type=str, required=False, default='Spike Events',
              help='namespace containing spike times')
@click.option("--spike-events-t", required=False, type=str, default='t',
              help='name of variable containing spike times')
@click.option("--target-rate-map-path", required=False, type=click.Path(),
              help='path to neuroh5 file containing target rate maps used for rate optimization')
@click.option("--target-rate-map-namespace", type=str, required=False, default='Input Spikes',
              help='namespace containing target rate maps used for rate optimization')
@click.option("--target-rate-map-arena", type=str, required=False, 
              help='name of arena used for rate optimization')
@click.option("--target-rate-map-trajectory", type=str, required=False, 
              help='name of trajectory used for rate optimization')
@click.argument('target')# help='rate, rate_dist'


def optimize(config_file, population, gid, generate_inputs, generate_weights, t_max, t_min, tstop, opt_iter,
             template_paths, dataset_prefix, config_prefix, spike_events_path, spike_events_namespace, spike_events_t,
             param_type, recording_profile, results_path, target_rate_map_path, target_rate_map_namespace, target_rate_map_arena, target_rate_map_trajectory, target):
    """
    Optimize the firing rate of the specified cell in a network clamp configuration.
    """

    results_file_id = uuid.uuid4()
    comm = MPI.COMM_WORLD
    np.seterr(all='raise')
    verbose = True
    cache_queries = True
    params = dict(locals())
    env = Env(**params)
    configure_hoc_env(env)

    init(env, population, gid, spike_events_path, 
         generate_inputs_pops=set(generate_inputs), 
         generate_weights_pops=set(generate_weights), 
         spike_events_namespace=spike_events_namespace, 
         t_var=spike_events_t, t_min=t_min, t_max=t_max)

    if target == 'rate':
        optimize_rate(env, population, gid, opt_iter=opt_iter, param_type=param_type)
    elif target == 'ratedist' or target == 'rate_dist':
        optimize_rate_dist(env, tstop, population, gid, 
                           target_rate_map_path, target_rate_map_namespace,
                           target_rate_map_arena, target_rate_map_trajectory,
                           opt_iter=opt_iter, param_type=param_type)
    else:
        raise RuntimeError('network_clamp.optimize: unknown optimization target %s' % \
                           target)


cli.add_command(show)
cli.add_command(go)
cli.add_command(optimize)

if __name__ == '__main__':
    cli(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
        standalone_mode=False)
