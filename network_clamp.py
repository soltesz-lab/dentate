"""
Routines for Network Clamp simulation.
"""
import os, sys, copy, uuid, pprint
from collections import defaultdict
from mpi4py import MPI
import numpy as np
import click
from dentate import io_utils, spikedata, synapses, stimulus, cell_clamp
from dentate.cells import h, make_input_cell, register_cell, record_cell, report_topology, is_cell_registered
from dentate.env import Env
from dentate.neuron_utils import h, configure_hoc_env, make_rec
from dentate.utils import is_interactive, Context, Closure, list_find, list_index, range, str, viewitems, zip_longest, get_module_logger, config_logging
from dentate.utils import get_trial_time_indices, get_trial_time_ranges, get_low_pass_filtered_trace
from dentate.cell_clamp import init_biophys_cell
from neuroh5.io import read_cell_attribute_selection, read_cell_attribute_info

# This logger will inherit its settings from the root logger, created in dentate.env
logger = get_module_logger(__name__)

context = Context()
env = None

def mpi_excepthook(type, value, traceback):
    """

    :param type:
    :param value:
    :param traceback:
    :return:
    """
    sys_excepthook(type, value, traceback)
    sys.stdout.flush()
    sys.stderr.flush()
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)

sys_excepthook = sys.excepthook
sys.excepthook = mpi_excepthook


def distgfs_reduce_fun(xs):
    return xs[0]

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


def init_inputs_from_spikes(env, presyn_sources, time_range,
                            spike_events_path, spike_events_namespace,
                            arena_id, trajectory_id, spike_train_attr_name='t', n_trials=1):

    populations = list(presyn_sources.keys())
    
    equilibration_duration = float(env.stimulus_config['Equilibration Duration'])

    this_spike_events_namespace = '%s %s %s' % (spike_events_namespace, arena_id, trajectory_id)
    ## Load spike times of presynaptic cells
    spkdata = spikedata.read_spike_events(spike_events_path,
                                          populations,
                                          this_spike_events_namespace,
                                          spike_train_attr_name=spike_train_attr_name,
                                          time_range=time_range, n_trials=n_trials,
                                          merge_trials=True)
    
    spkindlst = spkdata['spkindlst']
    spktlst = spkdata['spktlst']
    spkpoplst = spkdata['spkpoplst']

    ## Organize spike times by index of presynaptic population and gid
    input_source_dict = {}
    for population in sorted(populations):
        pop_index = int(env.Populations[population])
        spk_pop_index = list_index(population, spkpoplst)
        if spk_pop_index is None:
            logger.warning("No spikes found for population %s in file %s" % (population, spike_events_path))
            continue
        spk_inds = spkindlst[spk_pop_index]
        spk_ts = spktlst[spk_pop_index]

        spikes_attr_dict = {}
        gid_range = range(env.celltypes[population]['start'],
                          env.celltypes[population]['start'] + env.celltypes[population]['num'])
        for gid in gid_range:
            spk_inds = np.where(spk_inds == gid)[0]
            ts = spk_ts[spk_inds] + equilibration_duration
            spikes_attr_dict[gid] = { spike_train_attr_name: ts }  
        input_source_dict[pop_index] = {'spiketrains': spikes_attr_dict}

    return input_source_dict


def init_inputs_from_features(env, presyn_sources, time_range,
                              input_features_path, input_features_namespaces,
                              arena_id, trajectory_id, spike_train_attr_name='t', n_trials=1):

    if time_range is not None:
        if time_range[0] is None:
            time_range[0] = 0.0

    equilibration_duration = float(env.stimulus_config['Equilibration Duration'])
    spatial_resolution = float(env.stimulus_config['Spatial Resolution'])
    temporal_resolution = float(env.stimulus_config['Temporal Resolution'])
    
    this_input_features_namespaces = ['%s %s' % (input_features_namespace, arena_id)
                                      for input_features_namespace in input_features_namespaces]
    
    input_features_attr_names = ['Selectivity Type', 'Num Fields', 'Field Width', 'Peak Rate',
                                 'Module ID', 'Grid Spacing', 'Grid Orientation',
                                 'Field Width Concentration Factor', 
                                 'X Offset', 'Y Offset']
    
    selectivity_type_names = { i: n for n, i in viewitems(env.selectivity_types) }

    arena = env.stimulus_config['Arena'][arena_id]
    arena_x, arena_y = stimulus.get_2D_arena_spatial_mesh(arena=arena, spatial_resolution=spatial_resolution)
    
    trajectory = arena.trajectories[trajectory_id]
    t, x, y, d = stimulus.generate_linear_trajectory(trajectory,
                                                     temporal_resolution=temporal_resolution,
                                                     equilibration_duration=equilibration_duration)
    if time_range is not None:
        t_range_inds = np.where((t <= time_range[1]) & (t >= time_range[0] - equilibration_duration))[0] 
        t = t[t_range_inds]
        x = x[t_range_inds]
        y = y[t_range_inds]
        d = d[t_range_inds]
    trajectory = t, x, y, d

    equilibrate = stimulus.get_equilibration(env)

    input_source_dict = {}
    for population in sorted(presyn_sources):
        selection = list(presyn_sources[population])
        logger.info("generating spike trains for %d inputs from presynaptic population %s..." % (len(selection), population))
        pop_index = int(env.Populations[population])
        spikes_attr_dict = {}
        for input_features_namespace in this_input_features_namespaces:
            input_features_iter = read_cell_attribute_selection(input_features_path, population,
                                                                selection=selection,
                                                                namespace=input_features_namespace,
                                                                mask=set(input_features_attr_names), 
                                                                comm=env.comm)
            for gid, selectivity_attr_dict in input_features_iter:
                spikes_attr_dict[gid] = stimulus.generate_input_spike_trains(env, selectivity_type_names,
                                                                             trajectory, gid, selectivity_attr_dict,
                                                                             equilibrate=equilibrate,
                                                                             spike_train_attr_name=spike_train_attr_name,
                                                                             n_trials=n_trials,
                                                                             return_selectivity_features=False,
                                                                             merge_trials=True,
                                                                             time_range=time_range,
                                                                             comm=env.comm)
                spikes_attr_dict[gid][spike_train_attr_name] += equilibration_duration

        input_source_dict[pop_index] = {'spiketrains': spikes_attr_dict}

    return input_source_dict


def init(env, pop_name, gid_set, arena_id=None, trajectory_id=None, n_trials=1,
         spike_events_path=None, spike_events_namespace='Spike Events', spike_train_attr_name='t',
         input_features_path=None, input_features_namespaces=None,
         generate_weights_pops=set([]), t_min=None, t_max=None, write_cell=False, plot_cell=False):
    """
    Instantiates a cell and all its synapses and connections and loads
    or generates spike times for all synaptic connections.

    :param env: an instance of env.Env
    :param pop_name: population name
    :param gid_set: cell gids
    :param spike_events_path:

    """
    if env.results_file_path is not None:
        io_utils.mkout(env, env.results_file_path)

    if env.cell_selection is None:
        env.cell_selection = {}
    selection = env.cell_selection.get(pop_name, [])
    env.cell_selection[pop_name] = list(gid_set) + [selection]

    ## If specified, presynaptic spikes that only fall within this time range
    ## will be loaded or generated
    if t_max is None:
        t_range = None
    else:
        if t_min is None:
            t_range = [0., t_max]
        else:
            t_range = [t_min, t_max]
            
    ## Attribute namespace that contains recorded spike events
    namespace_id = spike_events_namespace

    ## Determine presynaptic populations that connect to this cell type
    presyn_names = env.projection_dict[pop_name]

    ## Load cell gid and its synaptic attributes and connection data
    for gid in gid_set:
        cell = init_biophys_cell(env, pop_name, gid, write_cell=write_cell)

    pop_index_dict = { ind: name for name, ind in viewitems(env.Populations) }
    
    weight_source_dict = {}
    for presyn_name in presyn_names:
        presyn_index = int(env.Populations[presyn_name])

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
    presyn_sources = defaultdict(set)

    for gid in gid_set:
        this_syn_attrs = syn_attrs[gid]
        for syn_id, syn in viewitems(this_syn_attrs):
            presyn_id = syn.source.population
            presyn_name = pop_index_dict[presyn_id]
            presyn_gid = syn.source.gid
            presyn_sources[presyn_name].add(presyn_gid)

    if spike_events_path is not None:
        input_source_dict = init_inputs_from_spikes(env, presyn_sources, t_range,
                                                    spike_events_path, spike_events_namespace,
                                                    arena_id, trajectory_id, spike_train_attr_name, n_trials)
    elif input_features_path is not None:
        input_source_dict = init_inputs_from_features(env, presyn_sources, t_range,
                                                      input_features_path, input_features_namespaces,
                                                      arena_id, trajectory_id, spike_train_attr_name, n_trials)
    else:
        raise RuntimeError('network_clamp.init: neither input spikes nor input features are provided')

    if t_range is not None:
        env.tstop = t_range[1] - t_range[0]

    for gid in gid_set:
        this_syn_attrs = syn_attrs[gid]
        for syn_id, syn in viewitems(this_syn_attrs):
            presyn_id = syn.source.population
            presyn_gid = syn.source.gid
            if presyn_id in input_source_dict:
                ## Load presynaptic spike times into the VecStim for each synapse;
                ## if spike_generator_dict contains an entry for the respective presynaptic population,
                ## then use the given generator to generate spikes.
                if not ((presyn_gid in env.gidset) or (is_cell_registered(env, presyn_gid))):
                    cell = make_input_cell(env, presyn_gid, presyn_id, input_source_dict,
                                           spike_train_attr_name=spike_train_attr_name)
                    register_cell(env, presyn_id, presyn_gid, cell)

    for gid in gid_set:
        this_syn_attrs = syn_attrs[gid]
        source_weight_params = generate_weights(env, weight_source_dict, this_syn_attrs)

        for presyn_id, weight_params in viewitems(source_weight_params):
            weights_syn_ids = weight_params['syn_id']
            for syn_name in (syn_name for syn_name in weight_params if syn_name != 'syn_id'):
                weights_values = weight_params[syn_name]
                syn_attrs.add_mech_attrs_from_iter(gid, syn_name, \
                                                   zip_longest(weights_syn_ids, \
                                                               [{'weight': x} for x in weights_values]))
    for gid in gid_set:
        synapses.config_biophys_cell_syns(env, gid, pop_name, insert=True, insert_netcons=True, verbose=True)
        record_cell(env, pop_name, gid)

    if plot_cell:
        import dentate.plot
        from dentate.plot import plot_synaptic_attribute_distribution
        syn_attrs = env.synapse_attributes
        syn_name = 'AMPA'
        syn_mech_name = syn_attrs.syn_mech_names[syn_name]
        for gid in gid_set:
            biophys_cell = env.biophys_cells[pop_name][gid]
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

    if is_interactive:
        context.update(locals())


def run(env, cvode=False, pc_runworker=True):
    """
    Runs network clamp simulation. Assumes that procedure `init` has been
    called with the network configuration provided by the `env`
    argument.

    :param env: instance of env.Env
    :param cvode: whether to use adaptive integration
    """

    rank = int(env.pc.id())
    nhosts = int(env.pc.nhost())

    rec_dt = 0.1
    if env.recording_profile is not None:
        rec_dt = env.recording_profile.get('dt', 0.1) 
    env.t_rec.record(h._ref_t, rec_dt)
    env.t_vec.resize(0)
    env.id_vec.resize(0)

    st_comptime = env.pc.step_time()

    h.cvode_active(1 if cvode else 0)
    
    h.t = 0.0
    h.dt = env.dt
    tstop = float(env.tstop)
    if 'Equilibration Duration' in env.stimulus_config:
        tstop += float(env.stimulus_config['Equilibration Duration'])
    h.tstop = float(env.n_trials) * tstop
    h.finitialize(env.v_init)

    if rank == 0:
        logger.info("*** Running simulation with dt = %.03f and tstop = %.02f" % (h.dt, h.tstop))

    env.pc.barrier()
    env.pc.psolve(h.tstop)

    if rank == 0:
        logger.info("*** Simulation completed")
    env.pc.barrier()

    comptime = env.pc.step_time() - st_comptime
    avgcomp = env.pc.allreduce(comptime, 1) / nhosts
    maxcomp = env.pc.allreduce(comptime, 2)

    if rank == 0:
        logger.info("Host %i  ran simulation in %g seconds" % (rank, comptime))

    if pc_runworker:
        env.pc.runworker()
    env.pc.done()

    return spikedata.get_env_spike_dict(env, include_artificial=None)


def run_with(env, param_dict, cvode=False, pc_runworker=True):
    """
    Runs network clamp simulation with the specified parameters for the given gid(s).
    Assumes that procedure `init` has been called with
    the network configuration provided by the `env` argument.

    :param env: instance of env.Env
    :param param_dict: dictionary { gid: params }
    :param cvode: whether to use adaptive integration
    """

    rank = int(env.pc.id())
    nhosts = int(env.pc.nhost())

    for pop_name, gid_param_dict in viewitems(param_dict):
        biophys_cell_dict = env.biophys_cells[pop_name]

        synapse_config = env.celltypes[pop_name]['synapses']
        weights_dict = synapse_config.get('weights', {})

        for gid, params_tuples in viewitems(gid_param_dict):
            biophys_cell = biophys_cell_dict[gid]
            for update_operator, destination, source, sec_type, syn_name, param_path, param_value in params_tuples:
                if isinstance(param_path, tuple):
                    p, s = param_path
                else:
                    p, s = param_path, None

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

    rec_dt = 0.1
    if env.recording_profile is not None:
        rec_dt = env.recording_profile.get('dt', 0.1) 
    env.t_rec.record(h._ref_t, rec_dt)

    env.t_vec.resize(0)
    env.id_vec.resize(0)

    st_comptime = env.pc.step_time()

    h.cvode.cache_efficient(1)
    h.cvode_active(1 if cvode else 0)

    h.t = 0.0
    tstop = float(env.tstop)
    if 'Equilibration Duration' in env.stimulus_config:
        tstop += float(env.stimulus_config['Equilibration Duration'])
    h.tstop = float(env.n_trials) * tstop

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

    comptime = env.pc.step_time() - st_comptime
    avgcomp = env.pc.allreduce(comptime, 1) / nhosts
    maxcomp = env.pc.allreduce(comptime, 2)

    if rank == 0:
        logger.info("Host %i  ran simulation in %g seconds" % (rank, comptime))

    if pc_runworker:
        env.pc.runworker()
    env.pc.done()
    
    return spikedata.get_env_spike_dict(env, include_artificial=None)



def optimize_params(env, pop_name, param_type, param_config_name):
                        
    param_bounds = {}
    param_names = []
    param_initial_dict = {}
    param_range_tuples = []

    synapse_config = env.celltypes[pop_name]['synapses']
    weights_dict = synapse_config.get('weights', {})

    if param_type == 'synaptic':
        if pop_name in env.netclamp_config.optimize_parameters['synaptic']:
            opt_params = env.netclamp_config.optimize_parameters['synaptic'][pop_name]
            param_ranges = opt_params['Parameter ranges'][param_config_name]
        else:
            raise RuntimeError(
                "network_clamp.optimize_params: population %s does not have optimization configuration" % pop_name)
        keyfun = lambda kv: str(kv[0])
        for source, source_dict in sorted(viewitems(param_ranges), key=keyfun):
            for sec_type, sec_type_dict in sorted(viewitems(source_dict), key=keyfun):
                for syn_name, syn_mech_dict in sorted(viewitems(sec_type_dict), key=keyfun):
                    for param_fst, param_rst in sorted(viewitems(syn_mech_dict), key=keyfun):
                        if isinstance(param_rst, dict):
                            update_operator = lambda gid, syn_id, old, new: old
                            for const_name, const_range in sorted(viewitems(param_rst)):
                                param_path = (param_fst, const_name)
                                param_range_tuples.append((update_operator, pop_name, source, sec_type, syn_name, param_path, const_range))
                                param_key = '%s.%s.%s.%s.%s.%s' % (pop_name, str(source), sec_type, syn_name, param_fst, const_name)
                                param_initial_value = (const_range[1] - const_range[0]) / 2.0
                                param_initial_dict[param_key] = param_initial_value
                                param_bounds[param_key] = const_range
                                param_names.append(param_key)
                        else:
                            update_operator = None
                            param_name = param_fst
                            param_range = param_rst
                            param_range_tuples.append((update_operator, pop_name, source, sec_type, syn_name, param_name, param_range))
                            param_key = '%s.%s.%s.%s.%s' % (pop_name, source, sec_type, syn_name, param_name)
                            param_initial_value = (param_range[1] - param_range[0]) / 2.0
                            param_initial_dict[param_key] = param_initial_value
                            param_bounds[param_key] = param_range
                            param_names.append(param_key)
        
    else:
        raise RuntimeError("network_clamp.optimize_params: unknown parameter type %s" % param_type)

    return param_bounds, param_names, param_initial_dict, param_range_tuples


def init_state_objfun(config_file, population, cell_index_set, arena_id, trajectory_id, generate_weights, t_max, t_min, opt_iter, template_paths, dataset_prefix, config_prefix, results_path, spike_events_path, spike_events_namespace, spike_events_t, input_features_path, input_features_namespaces, n_trials, param_type, param_config_name, recording_profile, state_variable, state_filter, target_value, worker, **kwargs):

    params = dict(locals())
    env = Env(**params)
    env.results_file_path = None
    configure_hoc_env(env)
    init(env, population, cell_index_set, arena_id, trajectory_id, n_trials,
         spike_events_path, spike_events_namespace=spike_events_namespace, 
         spike_train_attr_name=spike_events_t,
         input_features_path=input_features_path,
         input_features_namespaces=input_features_namespaces,
         generate_weights_pops=set(generate_weights), 
         t_min=t_min, t_max=t_max)

    time_step = env.stimulus_config['Temporal Resolution']
    equilibration_duration = float(env.stimulus_config['Equilibration Duration'])
    
    param_bounds, param_names, param_initial_dict, param_range_tuples = \
      optimize_params(env, population, param_type, param_config_name)
    
    def from_param_dict(params_dict):
        result = []
        for param_pattern, (update_operator, population, source, sec_type, syn_name, param_name, param_range) in zip(param_names, param_range_tuples):
            result.append((update_operator, population, source, sec_type, syn_name, param_name, params_dict[param_pattern]))
        return result

    def gid_state_values(spkdict, t_offset, n_trials, t_rec, state_recs_dict):
        t_vec = np.asarray(t_rec.to_python(), dtype=np.float32)
        t_trial_inds = get_trial_time_indices(t_vec, n_trials, t_offset)
        results_dict = {}
        filter_fun = None
        if state_filter == 'lowpass':
            filter_fun = lambda x, t: get_low_pass_filtered_trace(x, t)
        for gid in state_recs_dict:
            state_values = []
            state_recs = state_recs_dict[gid]
            for rec in state_recs:
                vec = np.asarray(rec['vec'].to_python(), dtype=np.float32)
                if filter_fun is None:
                    data = np.asarray([ np.mean(vec[t_inds])
                                            for t_inds in t_trial_inds ])
                else:
                    data = np.asarray([ np.mean(filter_fun(vec[t_inds], t_vec[t_inds]))
                                            for t_inds in t_trial_inds ])
                state_values.append(np.mean(data))
            m = np.mean(np.asarray(state_values))
            logger.info('state value objective: mean value of %s of gid %i is %.2f' % (state_variable, gid, m))
            results_dict[gid] = m
        return results_dict

    recording_profile = { 'label': 'network_clamp.state.%s' % state_variable,
                          'dt': 0.1,
                          'section quantity': {
                              state_variable: { 'swc types': ['soma'] }
                            }
                        }
    env.recording_profile = recording_profile
    state_recs_dict = {}
    for gid in cell_index_set:
        state_recs_dict[gid] = record_cell(env, population, gid, recording_profile=recording_profile)

    def f(v, **kwargs): 
        state_values_dict = gid_state_values(run_with(env, {population: {gid: from_param_dict(v[gid]) 
                                                                        for gid in cell_index_set}}), 
                                             equilibration_duration, 
                                             n_trials, env.t_rec, 
                                             state_recs_dict)
        return { gid: -abs(state_values_dict[gid] - target_value) for gid in cell_index_set }

    return f


def init_rate_objfun(config_file, population, cell_index_set, arena_id, trajectory_id, n_trials, generate_weights, t_max, t_min, opt_iter, template_paths, dataset_prefix, config_prefix, results_path, spike_events_path, spike_events_namespace, spike_events_t, input_features_path, input_features_namespaces, param_type, param_config_name, recording_profile, target_rate, worker, **kwargs):

    params = dict(locals())
    env = Env(**params)
    env.results_file_path = None
    configure_hoc_env(env)
    init(env, population, cell_index_set, arena_id, trajectory_id, n_trials,
         spike_events_path, spike_events_namespace=spike_events_namespace, 
         spike_train_attr_name=spike_events_t,
         input_features_path=input_features_path,
         input_features_namespaces=input_features_namespaces,
         generate_weights_pops=set(generate_weights),
         t_min=t_min, t_max=t_max)

    time_step = env.stimulus_config['Temporal Resolution']
    param_bounds, param_names, param_initial_dict, param_range_tuples = \
      optimize_params(env, population, param_type, param_config_name)
    
    def from_param_dict(params_dict):
        result = []
        for param_pattern, (update_operator, population, source, sec_type, syn_name, param_name, param_range) in zip(param_names, param_range_tuples):
            result.append((update_operator, population, source, sec_type, syn_name, param_name, params_dict[param_pattern]))
        return result

    def gid_firing_rate(spkdict, cell_index_set):
        rates_dict = defauldict(list)
        mean_rates_dict = {}
        for i in range(n_trials):
            spkdict1 = {}
            for gid in cell_index_set:
                if gid in spkdict[pop_name]:
                    spk_ts = spkdict[pop_name][gid][i]
                    spkdict1[gid] = spk_ts
                else:
                    spkdict1[gid] = np.asarray([], dtype=np.float32)

            rate_dict = spikedata.spike_rates(spkdict1)
            for gid in spkdict[pop_name]:
                logger.info('firing rate objective: spike times of gid %i: %s' % (gid, pprint.pformat(spkdict[pop_name][gid])))
                logger.info('firing rate objective: rate of gid %i is %.2f' % (gid, rate_dict[gid]))
            for gid in cell_index_set:
                rates_dict[gid].append(rate_dict[gid]['rate'])
        mean_rates_dict = { gid: np.mean(np.asarray(rates_dict[gid]))
                            for gid in cell_index_set }
        return mean_rates_dict

    logger.info("firing rate objective: target rate: %.02f" % target_rate)

    def f(v, **kwargs): 
        firing_rates_dict = gid_firing_rate(run_with(env, {population: {gid: from_param_dict(v[gid]) 
                                                                        for gid in cell_index_set}}), 
                                            cell_index_set)
        return { gid: -abs(firing_rates_dict[gid] - target_rate) for gid in cell_index_set }

    return f


def init_selectivity_features_objfun(config_file, population, cell_index_set, arena_id, trajectory_id, n_trials,
                                    generate_weights, t_max, t_min,
                                    opt_iter, template_paths, dataset_prefix, config_prefix, results_path,
                                    spike_events_path, spike_events_namespace, spike_events_t,
                                    input_features_path, input_features_namespaces,
                                    param_type, param_config_name, recording_profile,
                                    target_rate_map_path, target_rate_map_namespace,
			            target_rate_map_arena, target_rate_map_trajectory,  worker, **kwargs):
    
    rate_eps = kwargs.get('rate_eps', 1e-2)
    penalty_oof = kwargs.get('penalty_oof', 10)
    penalty_inf = kwargs.get('penalty_inf', 40)
    
    params = dict(locals())
    env = Env(**params)
    env.results_file_path = None
    configure_hoc_env(env)
    init(env, population, cell_index_set, arena_id, trajectory_id, n_trials,
         spike_events_path, spike_events_namespace=spike_events_namespace, 
         spike_train_attr_name=spike_events_t,
         input_features_path=input_features_path,
         input_features_namespaces=input_features_namespaces,
         generate_weights_pops=set(generate_weights), 
         t_min=t_min, t_max=t_max)

    time_step = env.stimulus_config['Temporal Resolution']

    input_namespace = '%s %s %s' % (target_rate_map_namespace, target_rate_map_arena, target_rate_map_trajectory)
    it = read_cell_attribute_selection(target_rate_map_path, population, namespace=input_namespace,
                                        selection=list(cell_index_set), mask=set(['Trajectory Rate Map']))
    trj_rate_maps = { gid: attr_dict['Trajectory Rate Map']
                      for gid, attr_dict in it }

    trj_x, trj_y, trj_d, trj_t = stimulus.read_trajectory(target_rate_map_path, target_rate_map_arena, target_rate_map_trajectory)

    time_range = (0., min(np.max(trj_t), t_max))
    
    time_bins = np.arange(time_range[0], time_range[1], time_step)
    target_rate_vector_dict = { gid: np.interp(time_bins, trj_t, trj_rate_maps[gid])
                                for gid in trj_rate_maps }
    for gid, target_rate_vector in viewitems(target_rate_vector_dict):
        target_rate_vector[np.abs(target_rate_vector) < rate_eps] = 0.

    target_spike_counts_dict = {}
    for gid, target_rate_vector in viewitems(target_rate_vector_dict):
        target_spike_counts = np.zeros((len(time_bins),))
        for i in range(len(time_bins)):
            if target_rate_vector[i] > 0.:
                target_spike_counts[i] = target_rate_vector[i] * 1e-3 * time_step
        target_spike_counts_dict[gid] = target_spike_counts
        
    param_bounds, param_names, param_initial_dict, param_range_tuples = \
      optimize_params(env, population, param_type, param_config_name)
    
    def from_param_dict(params_dict):
        result = []
        for param_pattern, (update_operator, population, source, sec_type, syn_name, param_name, param_range) in zip(param_names, param_range_tuples):
            result.append((update_operator, population, source, sec_type, syn_name, param_name, params_dict[param_pattern]))
        return result

    def gid_spike_counts(spkdict, cell_index_set):
        spike_counts_dict = defaultdict(list)
        mean_spike_counts_dict = {}
        for i in range(n_trials):
            spkdict1 = {}
            spike_bin_counts = {}
            for gid in cell_index_set:
                if gid in spkdict[population]:
                    spike_bin_counts, _ = np.histogram(spkdict[population][gid][i], bins=time_bins)
                else:
                    spike_bin_counts = [0.] * len(time_bins)
                spike_counts_dict[gid].append(spike_bin_counts)
            for gid in spkdict[population]:
                logger.info('selectivity features objective: trial %d spike times of gid %i: %s' % (i, gid, str(spkdict[population][gid])))
                logger.info('selectivity features objective: trial %d spike counts of gid %i: %s' % (i, gid, str(spike_counts_dict[gid][i])))
                logger.info('selectivity features objective: trial %d spike counts min/max of gid %i: %.02f / %.02f' %
                            (i, gid, np.min(spike_counts_dict[gid][i]), np.max(spike_counts_dict[gid][i])))

        mean_spike_counts_dict = { gid: np.mean(np.row_stack(spike_counts_dict[gid]), axis=0)
                                   for gid in cell_index_set }
        for gid in mean_spike_counts_dict:
            logger.info('selectivity features objective: mean spike count of gid %i: %s' % (gid, str(mean_spike_counts_dict[gid])))
            logger.info('selectivity features objective: mean spike count min/max of gid %i: %.02f / %.02f' %
                        (gid, np.min(mean_spike_counts_dict[gid]), np.max(mean_spike_counts_dict[gid])))
        return mean_spike_counts_dict

    def f(v, **kwargs):
        mean_spike_counts_dict = gid_spike_counts(run_with(env, {population: {gid: from_param_dict(v[gid]) for gid in cell_index_set}}), cell_index_set)
        result = {}
        for gid in cell_index_set:
            target_spike_counts = target_spike_counts_dict[gid]
            mean_spike_counts = mean_spike_counts_dict[gid]
            residual = []
            for i in range(len(time_bins)-1):
                target_count = target_spike_counts[i]
                mean_count = mean_spike_counts[i]
                if np.isclose(target_count, 0.) and mean_count > 0.:
                    residual.append(penalty_oof * (mean_count - target_count))
                elif np.isclose(mean_count, 0.) and target_count > 0.:
                    residual.append(penalty_inf * (mean_count - target_count))
                else:
                    residual.append(mean_count - target_count)
            result[gid] = -(np.square(np.asarray(residual)).mean())
        return result
    
    return f


def init_rate_dist_objfun(config_file, population, cell_index_set, arena_id, trajectory_id, n_trials,
                          generate_weights, t_max, t_min,
                          opt_iter, template_paths, dataset_prefix, config_prefix, results_path,
                          spike_events_path, spike_events_namespace, spike_events_t,
                          input_features_path, input_features_namespaces,
                          param_type, param_config_name, recording_profile,
                          target_rate_map_path, target_rate_map_namespace,
			  target_rate_map_arena, target_rate_map_trajectory,  worker, **kwargs):
    
    rate_eps = 1e-4
    
    params = dict(locals())
    env = Env(**params)
    env.results_file_path = None
    configure_hoc_env(env)
    init(env, population, cell_index_set, arena_id, trajectory_id, n_trials,
         spike_events_path, spike_events_namespace=spike_events_namespace, 
         spike_train_attr_name=spike_events_t,
         input_features_path=input_features_path,
         input_features_namespaces=input_features_namespaces,
         generate_weights_pops=set(generate_weights), 
         t_min=t_min, t_max=t_max)

    time_step = env.stimulus_config['Temporal Resolution']

    input_namespace = '%s %s %s' % (target_rate_map_namespace, target_rate_map_arena, target_rate_map_trajectory)
    it = read_cell_attribute_selection(target_rate_map_path, population, namespace=input_namespace,
                                        selection=list(cell_index_set), mask=set(['Trajectory Rate Map']))
    trj_rate_maps = { gid: attr_dict['Trajectory Rate Map']
                      for gid, attr_dict in it }

    trj_x, trj_y, trj_d, trj_t = stimulus.read_trajectory(target_rate_map_path, target_rate_map_arena, target_rate_map_trajectory)

    time_range = (0., min(np.max(trj_t), t_max))
    
    time_bins = np.arange(time_range[0], time_range[1], time_step)
    target_rate_vector_dict = { gid: np.interp(time_bins, trj_t, trj_rate_maps[gid])
                                for gid in trj_rate_maps }
    for gid, target_rate_vector in viewitems(target_rate_vector_dict):
        idxs = np.where(np.abs(target_rate_vector) < rate_eps)[0]
        target_rate_vector[idxs] = 0.
    
    param_bounds, param_names, param_initial_dict, param_range_tuples = \
      optimize_params(env, population, param_type, param_config_name)
    
    def from_param_dict(params_dict):
        result = []
        for param_pattern, (update_operator, population, source, sec_type, syn_name, param_name, param_range) in zip(param_names, param_range_tuples):
            result.append((update_operator, population, source, sec_type, syn_name, param_name, params_dict[param_pattern]))
        return result

    def gid_firing_rate_vectors(spkdict, cell_index_set):
        rates_dict = defaultdict(list)
        mean_rates_dict = {}
        for i in range(n_trials):
            spkdict1 = {}
            for gid in cell_index_set:
                if gid in spkdict[population]:
                    spkdict1[gid] = spkdict[population][gid][i]
                else:
                    spkdict1[gid] = np.asarray([], dtype=np.float32)
            spike_density_dict = spikedata.spike_density_estimate (population, spkdict1, time_bins)
            for gid in cell_index_set:
                rate_vector = spike_density_dict[gid]['rate']
                idxs = np.where(np.abs(rate_vector) < rate_eps)[0]
                rate_vector[idxs] = 0.
                rates_dict[gid].append(rate_vector)
            for gid in spkdict[population]:
                logger.info('firing rate objective: trial %d spike times of gid %i: %s' % (i, gid, str(spkdict[population][gid])))
                logger.info('firing rate objective: trial %d firing rate of gid %i: %s' % (i, gid, str(spike_density_dict[gid])))
                logger.info('firing rate objective: trial %d firing rate min/max of gid %i: %.02f / %.02f Hz' % (i, gid, np.min(rates_dict[gid]), np.max(rates_dict[gid])))

        mean_rate_vector_dict = { gid: np.mean(np.row_stack(rates_dict[gid]), axis=0)
                                  for gid in cell_index_set }
        for gid in mean_rate_vector_dict:
            logger.info('firing rate objective: mean firing rate vector of gid %i: %s' % (gid, str(mean_rate_vector_dict[gid])))
            logger.info('firing rate objective: mean firing rate min/max of gid %i: %.02f / %.02f Hz' % (gid, np.min(mean_rate_vector_dict[gid]), np.max(mean_rate_vector_dict[gid])))
        return mean_rate_vector_dict

    def f(v, **kwargs): 
        firing_rate_vectors_dict = gid_firing_rate_vectors(run_with(env, {population: {gid: from_param_dict(v[gid]) for gid in cell_index_set}}), cell_index_set)
        return { gid: -np.square(np.subtract(firing_rate_vectors_dict[gid], 
                                             target_rate_vector_dict[gid])).mean()
                 for gid in cell_index_set }
    
    return f


def optimize_run(env, pop_name, param_config_name, init_objfun,
                 opt_iter=10, solver_epsilon=1e-5, param_type='synaptic', init_params={}, 
                 results_file=None, verbose=False):
    import distgfs

    param_bounds, param_names, param_initial_dict, param_range_tuples = \
      optimize_params(env, pop_name, param_type, param_config_name)
    
    hyperprm_space = { param_pattern: [param_range[0], param_range[1]]
                       for param_pattern, (update_operator, pop_name, source, sec_type, syn_name, _, param_range) in
                           zip(param_names, param_range_tuples) }

    if results_file is None:
        if env.results_path is not None:
            file_path = '%s/distgfs.network_clamp.%s.h5' % (env.results_path, str(env.results_file_id))
        else:
            file_path = 'distgfs.network_clamp.%s.h5' % (str(env.results_file_id))
    else:
        file_path = '%s/%s' % (env.results_path, results_file)
    distgfs_params = {'opt_id': 'network_clamp.optimize',
                      'problem_ids': init_params.get('cell_index_set', None),
                      'obj_fun_init_name': init_objfun, 
                      'obj_fun_init_module': 'dentate.network_clamp',
                      'obj_fun_init_args': init_params,
                      'reduce_fun_name': 'distgfs_reduce_fun',
                      'reduce_fun_module': 'dentate.network_clamp',
                      'problem_parameters': {},
                      'space': hyperprm_space,
                      'file_path': file_path,
                      'save': True,
                      'n_iter': opt_iter,
                      'solver_epsilon': solver_epsilon }

    results_dict = distgfs.run(distgfs_params, verbose=verbose,
                               spawn_workers=True, nprocs_per_worker=1)
    logger.info('Optimized parameters and objective function: %s' % pprint.pformat(results_dict))

    return results_dict

    
def dist_ctrl(controller, init_params, cell_index_set):
    """Controller for distributed network clamp runs."""
    task_ids = []
    for gid in cell_index_set:
        task_id = controller.submit_call("dist_run", module_name="dentate.network_clamp",
                                         args=(init_params, gid,))
        task_ids.append(task_id)

    for task_id in task_ids: 
        task_id, res = controller.get_next_result()

    controller.info()

    
    
def dist_run(init_params, gid):
    """Initialize workers for distributed network clamp runs."""

    results_file_id = init_params.get('results_file_id', None)
    if results_file_id is None:
        results_file_id = uuid.uuid4()
        init_params['results_file_id'] = results_file_id

    global env
    if env is None:
        env = Env(**init_params)
        configure_hoc_env(env)
    env.clear()

    env.results_file_id = results_file_id
    env.results_file_path = "%s/%s_results_%s.h5" % (env.results_path, env.modelName, env.results_file_id)
    
    population = init_params['population']
    arena_id = init_params['arena_id']
    trajectory_id = init_params['trajectory_id']
    spike_events_path = init_params['spike_events_path']
    spike_events_namespace = init_params['spike_events_namespace']
    spike_events_t = init_params['spike_events_t']
    input_features_path = init_params['input_features_path']
    input_features_namespaces = init_params['input_features_namespaces']
    generate_weights = init_params.get('generate_weights', [])
    t_min = init_params['t_min']
    t_max = init_params['t_max']
    n_trials = init_params['n_trials']
    
    init(env, population, set([gid]), arena_id, trajectory_id, n_trials,
         spike_events_path, spike_events_namespace=spike_events_namespace, 
         spike_train_attr_name=spike_events_t,
         input_features_path=input_features_path,
         input_features_namespaces=input_features_namespaces,
         generate_weights_pops=set(generate_weights),
         t_min=t_min, t_max=t_max)

    run(env)
    write_output(env)

    return None
    

def write_output(env):
    rank = env.comm.rank
    if rank == 0:
        logger.info("*** Writing spike data")
    io_utils.spikeout(env, env.results_file_path)
    if rank == 0:
        logger.info("*** Writing intracellular data")
    io_utils.recsout(env, env.results_file_path,
                     write_cell_location_data=True,
                     write_trial_data=True)


@click.group()
def cli():
    pass


@click.command()
@click.option("--config-file", '-c', required=True, type=str, help='model configuration file name')
@click.option("--population", '-p', required=True, type=str, default='GC', help='target population')
@click.option("--gid", '-g', required=True, type=int, default=0, help='target cell gid')
@click.option("--arena-id", '-a', required=True, type=str, help='arena id for input stimulus')
@click.option("--trajectory-id", '-t', required=True, type=str, help='trajectory id for input stimulus')
@click.option("--template-paths", type=str, required=True,
              help='colon-separated list of paths to directories containing hoc cell templates')
@click.option("--dataset-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              help='path to directory containing required neuroh5 data files')
@click.option("--config-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='config',
              help='path to directory containing network and cell mechanism config files')
@click.option("--results-path", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True), \
              help='path to directory where output files will be written')
@click.option("--spike-events-path", '-s', type=click.Path(exists=True, dir_okay=False, file_okay=True),
              help='path to neuroh5 file containing spike times')
@click.option("--spike-events-namespace", type=str, default='Spike Events',
              help='namespace containing spike times')
@click.option("--spike-events-t", required=False, type=str, default='t',
              help='name of variable containing spike times')
@click.option("--input-features-path", required=False, type=click.Path(),
              help='path to neuroh5 file containing input selectivity features')
@click.option("--input-features-namespaces", type=str, multiple=True, required=False, default=['Place Selectivity', 'Grid Selectivity'],
              help='namespace containing input selectivity features')
@click.option('--plot-cell', is_flag=True, help='plot the distribution of weight and g_unit synaptic parameters')
@click.option('--write-cell', is_flag=True, help='write out selected cell tree morphology and connections')
@click.option('--profile-memory', is_flag=True, help='calculate and print heap usage after the simulation is complete')
@click.option('--recording-profile', type=str, default='Network clamp default', help='recording profile to use')

def show(config_file, population, gid, arena_id, trajectory_id, template_paths, dataset_prefix, config_prefix, results_path,
         spike_events_path, spike_events_namespace, spike_events_t, input_features_path, input_features_namespaces, plot_cell, write_cell, profile_memory, recording_profile):
    """
    Show configuration for the specified cell.
    """

    np.seterr(all='raise')

    verbose = True
    init_params = dict(locals())
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        comm0 = comm.Split(2 if rank == 0 else 1, 0)
    
        env = Env(**init_params, comm=comm0)
        configure_hoc_env(env)

        init(env, population, set([gid]), arena_id, trajectory_id, 
             spike_events_path=spike_events_path,
             spike_events_namespace=spike_events_namespace,
             spike_train_attr_name=spike_events_t,
             input_features_path=input_features_path,
             input_features_namespaces=input_features_namespaces,
             plot_cell=plot_cell, write_cell=write_cell)

        cell = env.biophys_cells[population][gid]
        logger.info(pprint.pformat(report_topology(cell, env)))
        
        if env.profile_memory:
            profile_memory(logger)
            
    comm.barrier()

@click.command()
@click.option("--config-file", '-c', required=True, type=str, help='model configuration file name')
@click.option("--population", '-p', required=True, type=str, default='GC', help='target population')
@click.option("--gid", '-g', required=False, type=int, help='target cell gid')
@click.option("--arena-id", '-a', required=True, type=str, help='arena id for input stimulus')
@click.option("--trajectory-id", '-t', required=True, type=str, help='trajectory id for input stimulus')
@click.option("--generate-weights", '-w', required=False, type=str, multiple=True,
              help='generate weights for the given presynaptic population')
@click.option("--t-max", '-t', type=float, default=150.0, help='simulation end time')
@click.option("--t-min", type=float)
@click.option("--template-paths", type=str, required=True,
              help='colon-separated list of paths to directories containing hoc cell templates')
@click.option("--dataset-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              help='path to directory containing required neuroh5 data files')
@click.option("--config-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='config',
              help='path to directory containing network and cell mechanism config files')
@click.option("--spike-events-path", '-s', type=click.Path(),
              help='path to neuroh5 file containing spike times')
@click.option("--spike-events-namespace", type=str, default='Spike Events',
              help='namespace containing spike times')
@click.option("--spike-events-t", required=False, type=str, default='t',
              help='name of variable containing spike times')
@click.option("--input-features-path", required=False, type=click.Path(),
              help='path to neuroh5 file containing input selectivity features')
@click.option("--input-features-namespaces", type=str, multiple=True, required=False, default=['Place Selectivity', 'Grid Selectivity'],
              help='namespace containing input selectivity features')
@click.option("--n-trials", required=False, type=int, default=1,
              help='number of trials for input stimulus')
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

def go(config_file, population, gid, arena_id, trajectory_id, generate_weights, t_max, t_min,
       template_paths, dataset_prefix, config_prefix,
       spike_events_path, spike_events_namespace, spike_events_t,
       input_features_path, input_features_namespaces, n_trials, 
       results_path, results_file_id, results_namespace_id, plot_cell, write_cell,
       profile_memory, recording_profile):

    """
    Runs network clamp simulation for the specified gid, or for all gids found in the input data file.
    """

    init_params = dict(locals())
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    np.seterr(all='raise')
    verbose = True
    init_params['verbose'] = verbose
    
    cell_index_set = set([])
    if gid is None:
        cell_index_data = None
        comm0 = comm.Split(2 if rank == 0 else 1, 0)
        if rank == 0:
            env = Env(**init_params, comm=comm0)
            attr_info_dict = read_cell_attribute_info(env.data_file_path, populations=[population],
                                                      read_cell_index=True, comm=comm0)
            cell_index = None
            attr_name, attr_cell_index = next(iter(attr_info_dict[population]['Trees']))
            cell_index_set = set(attr_cell_index)
        cell_index_set = comm.bcast(cell_index_set, root=0)
    else:
        cell_index_set.add(gid)

    comm.barrier()
        
    if size > 1:
        import distwq
        if distwq.is_controller:
            distwq.run(fun_name="dist_ctrl", module_name="dentate.network_clamp",
                       verbose=True, args=(init_params, cell_index_set),
                       spawn_workers=True, nprocs_per_worker=1)

        else:
            distwq.run(verbose=True, spawn_workers=True, nprocs_per_worker=1)
    else:
        if results_file_id is None:
            results_file_id = uuid.uuid4()
        init_params['results_file_id'] = results_file_id
        env = Env(**init_params, comm=comm)
        configure_hoc_env(env)
        for gid in cell_index_set:
            init(env, population, cell_index_set, arena_id, trajectory_id, n_trials,
                 spike_events_path, spike_events_namespace=spike_events_namespace,
                 spike_train_attr_name=spike_events_t,
                 input_features_path=input_features_path,
                 input_features_namespaces=input_features_namespaces,
                 generate_weights_pops=set(generate_weights),
                 t_min=t_min, t_max=t_max,
                 plot_cell=plot_cell, write_cell=write_cell)
            run(env)
            write_output(env)
        if env.profile_memory:
            profile_memory(logger)


@click.command()
@click.option("--config-file", '-c', required=True, type=str, help='model configuration file name')
@click.option("--population", '-p', required=True, type=str, default='GC', help='target population')
@click.option("--gid", '-g', type=int, help='target cell gid')
@click.option("--arena-id", '-a', type=str, required=True, help='arena id')
@click.option("--trajectory-id", '-t', type=str, required=True, help='trajectory id')
@click.option("--generate-weights", '-w', required=False, type=str, multiple=True,
              help='generate weights for the given presynaptic population')
@click.option("--t-max", '-t', type=float, default=150.0, help='simulation end time')
@click.option("--t-min", type=float)
@click.option("--opt-iter", type=int, default=10, help='number of optimization iterations')
@click.option("--template-paths", type=str, required=True,
              help='colon-separated list of paths to directories containing hoc cell templates')
@click.option("--dataset-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              help='path to directory containing required neuroh5 data files')
@click.option("--config-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='config',
              help='path to directory containing network and cell mechanism config files')
@click.option("--param-config-name", type=str, 
              help='parameter configuration name to use for optimization (defined in config file)')
@click.option("--param-type", type=str, default='synaptic',
              help='parameter type to use for optimization (synaptic)')
@click.option('--recording-profile', type=str, help='recording profile to use')
@click.option("--results-file", required=False, type=str, help='optimization results file')
@click.option("--results-path", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True), \
              help='path to directory where output files will be written')
@click.option("--spike-events-path", type=click.Path(), required=False,
              help='path to neuroh5 file containing spike times')
@click.option("--spike-events-namespace", type=str, required=False, default='Spike Events',
              help='namespace containing input spike times')
@click.option("--spike-events-t", required=False, type=str, default='t',
              help='name of variable containing spike times')
@click.option("--input-features-path", required=False, type=click.Path(),
              help='path to neuroh5 file containing input selectivity features')
@click.option("--input-features-namespaces", type=str, multiple=True, required=False, default=['Place Selectivity', 'Grid Selectivity'],
              help='namespace containing input selectivity features')
@click.option("--n-trials", required=False, type=int, default=1,
              help='number of trials for input stimulus')
@click.option("--target-rate-map-path", required=False, type=click.Path(),
              help='path to neuroh5 file containing target rate maps used for rate optimization')
@click.option("--target-rate-map-namespace", type=str, required=False, default='Input Spikes',
              help='namespace containing target rate maps used for rate optimization')
@click.option("--target-state-variable", type=str, required=False, 
              help='name of state variable used for state optimization')
@click.option("--target-state-filter", type=str, required=False, 
              help='optional filter for state values used for state optimization')
@click.argument('target')# help='rate, rate_dist, state'


def optimize(config_file, population, gid, arena_id, trajectory_id, generate_weights, t_max, t_min, opt_iter, 
             template_paths, dataset_prefix, config_prefix,
             param_config_name, param_type, recording_profile, results_file, results_path,
             spike_events_path, spike_events_namespace, spike_events_t, 
             input_features_path, input_features_namespaces, n_trials,
             target_rate_map_path, target_rate_map_namespace, target_state_variable,
             target_state_filter, target):
    """
    Optimize the firing rate of the specified cell in a network clamp configuration.
    """
    init_params = dict(locals())

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    results_file_id = None
    if rank == 0:
        results_file_id = uuid.uuid4()
        
    results_file_id = comm.bcast(results_file_id, root=0)
    
    np.seterr(all='raise')
    verbose = True
    cache_queries = True

    cell_index_set = set([])
    if gid is None:
        cell_index_data = None
        comm0 = comm.Split(2 if rank == 0 else 1, 0)
        if rank == 0:
            env = Env(**init_params, comm=comm0)
            attr_info_dict = read_cell_attribute_info(env.data_file_path, populations=[population],
                                                      read_cell_index=True, comm=comm0)
            cell_index = None
            attr_name, attr_cell_index = next(iter(attr_info_dict[population]['Trees']))
            cell_index_set = set(attr_cell_index)
        cell_index_set = comm.bcast(cell_index_set, root=0)
    else:
        cell_index_set.add(gid)
    init_params['cell_index_set'] = cell_index_set
    del(init_params['gid'])
    comm.barrier()

    params = dict(locals())
    env = Env(**params)
    if size == 1:
        configure_hoc_env(env)
        init(env, population, cell_index_set, arena_id, trajectory_id, n_trials,
             spike_events_path, spike_events_namespace=spike_events_namespace, 
             spike_train_attr_name=spike_events_t,
             input_features_path=input_features_path,
             input_features_namespaces=input_features_namespaces,
             generate_weights_pops=set(generate_weights), 
             t_min=t_min, t_max=t_max)
        
    if (population in env.netclamp_config.optimize_parameters[param_type]):
        opt_params = env.netclamp_config.optimize_parameters[param_type][population]
    else:
        raise RuntimeError(
            "network_clamp.optimize: population %s does not have optimization configuration" % population)

    if target == 'rate':
        opt_target = opt_params['Targets']['firing rate']
        init_params['target_rate'] = opt_target
        init_objfun_name = 'init_rate_objfun'
    elif target == 'ratedist' or target == 'rate_dist':
        init_params['target_rate_map_arena'] = arena_id
        init_params['target_rate_map_trajectory'] = trajectory_id
        init_objfun_name = 'init_rate_dist_objfun'
    elif target == 'selectivity':
        init_params['target_rate_map_arena'] = arena_id
        init_params['target_rate_map_trajectory'] = trajectory_id
        init_objfun_name = 'init_selectivity_features_objfun'
    elif target == 'state':
        opt_target = opt_params['Targets']['state'][target_state_variable]
        init_params['target_value'] = opt_target
        init_params['state_variable'] = target_state_variable
        init_params['state_filter'] = target_state_filter
        init_objfun_name = 'init_state_objfun'
    else:
        raise RuntimeError('network_clamp.optimize: unknown optimization target %s' % target) 
        
    optimize_run(env, population, param_config_name, init_objfun_name,
                 opt_iter=opt_iter, param_type=param_type,
                 init_params=init_params, results_file=results_file,
                 verbose=verbose)


cli.add_command(show)
cli.add_command(go)
cli.add_command(optimize)

if __name__ == '__main__':

    cli(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
        standalone_mode=False)
