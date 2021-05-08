#!/usr/bin/env python
"""
Dentate Gyrus place field optimization script for optimization with nested.optimize
"""
__author__ = 'See AUTHORS.md'
import os, sys, logging, pprint
import click
import numpy as np
from mpi4py import MPI
import dentate
from dentate import cells, synapses, spikedata, stimulus, optimization, utils, network_clamp
from dentate.env import Env
from dentate.neuron_utils import h, configure_hoc_env, make_rec
import nested
from nested.optimize_utils import *
from nested.parallel import get_parallel_interface
from neuroh5.io import read_cell_attribute_selection

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


context = Context()


@click.command()
@click.option("--optimize-config-file-path", type=str, help='optimization configuration file name',
              default='../config/DG_optimize_network_clamp_config.yaml')
@click.option("--output-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default='../data')
@click.option("--export", is_flag=True)
@click.option("--export-file-path", type=str, default=None)
@click.option("--label", type=str, default=None)
@click.option("--bin-size", type=float, default=5.0)
@click.option("--procs-per-worker", type=int, default=1)
@click.option("--verbose", is_flag=True)
def main(optimize_config_file_path, output_dir, export, export_file_path, label, procs_per_worker, verbose):
    """

    :param optimize_config_file_path: str
    :param output_dir: str
    :param export: bool
    :param export_file_path: str
    :param label: str
    :param verbose: bool
    """
    # requires a global variable context: :class:'Context'
    context.update(locals())
    interface=get_parallel_interface(procs_per_worker=procs_per_worker)
    config_optimize_interactive(__file__, config_file_path=optimize_config_file_path, output_dir=output_dir,
                                export=export, export_file_path=export_file_path, label=label, disp=verbose,
                                interface=interface)


    

    
def config_worker():
    """

    """
    utils.config_logging(context.verbose)
    context.logger = utils.get_script_logger(os.path.basename(__file__))

    context.init_params = context.kwargs
    context.init_params['target_rate_map_arena'] = context.init_params['arena_id']
    context.init_params['target_rate_map_trajectory'] = context.init_params['trajectory_id']
    context.gid = int(context.init_params['gid'])
    
    if 'results_file_id' not in context():
        context.results_file_id = 'DG_optimize_network_clamp_%s_%s' % \
                             (context.interface.worker_id, datetime.datetime.today().strftime('%Y%m%d_%H%M'))
    if 'env' not in context():
        try:
            init_env()
            init_network_clamp()
        except Exception as err:
            context.logger.exception(err)
            raise err

    param_bounds = {}
    param_names = []
    param_initial_dict = {}
    param_range_tuples = []

    opt_param_config = optimization_params(env.netclamp_config.optimize_parameters, [population], param_config_name, param_type)

    if (context.population in context.env.netclamp_config.optimize_parameters['synaptic']):
        opt_params = context.env.netclamp_config.optimize_parameters['synaptic'][context.population]
        param_ranges = opt_params['Parameter ranges']
    else:
        raise RuntimeError(
            "optimize_selectivity_subworlds: population %s does not have optimization configuration" % context.population)

    def from_param_vector(params):
        result = []
        assert (len(params) == len(param_range_tuples))
        for i, (param_pattern, (population, source, sec_type, syn_name, param_name, param_range)) in enumerate(zip(param_names, param_range_tuples)):
            result.append((population, source, sec_type, syn_name, param_name, params[i]))
        return result

    context.param_names = param_names
    context.bounds = [ param_bounds[key] for key in param_names ]
    context.x0 = param_initial_dict
    context.from_param_vector = from_param_vector

    def range_inds(rs):
        a = np.concatenate(list(rs))
        return a

    target_rate_vector = read_target_rate_vector(context)

    if 'state_filter' not in context():
        context.state_filter = None

    context.logger.info("target_rate_vector max: %s" % str(np.max(target_rate_vector)))
    context.logger.info("target_rate_vector > 0: %s" % str(target_rate_vector > 0))
    context.outfld_idxs = range_inds(utils.contiguous_ranges(target_rate_vector <= 0., return_indices=True))
    context.infld_idxs = range_inds(utils.contiguous_ranges(target_rate_vector > 0, return_indices=True))
    context.max_infld_target_rate = np.max(target_rate_vector[context.infld_idxs])
    context.min_infld_target_rate = np.min(target_rate_vector[context.infld_idxs])
    peak_pctile = np.percentile(target_rate_vector[context.infld_idxs], 80)
    trough_pctile = np.percentile(target_rate_vector[context.infld_idxs], 20)

    context.peak_idxs = range_inds(utils.contiguous_ranges(target_rate_vector >= peak_pctile, return_indices=True)) 
    context.trough_idxs = range_inds(utils.contiguous_ranges(np.logical_and(target_rate_vector > 0., target_rate_vector <= trough_pctile), return_indices=True))


    context.t_outfld_ranges = tuple( ( (context.time_bins[r[0]], context.time_bins[r[1]])
                                       for r in utils.contiguous_ranges(target_rate_vector <= 0.) ) )
    context.t_infld_ranges = tuple( ( (context.time_bins[r[0]], context.time_bins[r[1]])
                                      for r in utils.contiguous_ranges(target_rate_vector > 0) ) )
    context.t_peak_ranges = tuple( ( (context.time_bins[r[0]], context.time_bins[r[1]])
                                      for r in utils.contiguous_ranges(target_rate_vector > peak_pctile) ) )
    context.t_trough_ranges = tuple( ( (context.time_bins[r[0]], context.time_bins[r[1]])
                                      for r in utils.contiguous_ranges(np.logical_and(target_rate_vector > 0., target_rate_vector <= trough_pctile))) )

    target_state_variable = context.target_state_variable
    context.target_val['%s peak state value' % context.population] = opt_params['Targets']['state'][target_state_variable]['peak']
    context.target_val['%s baseline state value' % context.population] = opt_params['Targets']['state'][target_state_variable]['mean']
    context.target_val['%s baseline firing rate' % context.population] = float(opt_params['Targets']['baseline firing rate'])
    context.target_val['%s in field max firing rate' % context.population] = np.max(target_rate_vector[context.infld_idxs])
    context.target_val['%s in field min firing rate' % context.population] = np.min(target_rate_vector[context.infld_idxs])
    context.target_val['%s out of field mean firing rate' % context.population] = np.mean(target_rate_vector[context.outfld_idxs])

    
def config_controller():
    """

    """
    utils.config_logging(context.verbose)
    context.logger = utils.get_script_logger(os.path.basename(__file__))

    context.init_params = context.kwargs
    context.init_params['target_rate_map_arena'] = context.init_params['arena_id']
    context.init_params['target_rate_map_trajectory'] = context.init_params['trajectory_id']
    context.gid = int(context.init_params['gid'])
    context.target_val = {}
    
    if 'results_file_id' not in context():
        context.results_file_id = 'DG_optimize_pf_%s_%s' % \
                             (context.interface.worker_id, datetime.datetime.today().strftime('%Y%m%d_%H%M'))
    
    
def init_env():
    """

    """
    context.env = Env(comm=context.comm, results_file_id=context.results_file_id, **context.init_params)
    configure_hoc_env(context.env)
    context.gid = int(context.init_params['gid'])
    context.population = context.init_params['population']
    context.target_val = {}

    
def init_network_clamp():
    """

    """
    np.seterr(all='raise')
    if context.env is None:
        context.env = Env(comm=context.comm, results_file_id=context.results_file_id, **context.init_params)
        configure_hoc_env(env)

    context.gid = int(context.init_params['gid'])
    context.population = context.init_params['population']
    context.target_val = {}
    network_clamp.init(context.env,
                       context.init_params['population'],
                       set([context.gid]),
                       arena_id=context.init_params['arena_id'],
                       trajectory_id=context.init_params['trajectory_id'],
                       n_trials=int(context.init_params['n_trials']),
                       spike_events_path=context.init_params.get('spike_events_path', None),
                       spike_events_namespace=context.init_params.get('spike_events_namespace', None),
                       spike_train_attr_name=context.init_params.get('spike_train_attr_name', None),
                       input_features_path=context.init_params.get('input_features_path', None),
                       input_features_namespaces=context.init_params.get('input_features_namespaces', None),
                       t_min=0., t_max=context.init_params['tstop'])
    
    context.equilibration_duration = float(context.env.stimulus_config['Equilibration Duration'])
    
    state_variable = 'v'
    context.recording_profile = { 'label': 'optimize_network_clamp.state.%s' % state_variable,
                                  'dt': 0.1,
                                  'section quantity': {
                                      state_variable: { 'swc types': ['soma'] }
                                  }
                                }    
    context.state_recs_dict = {}
    context.state_recs_dict[context.gid] = cells.record_cell(context.env, context.population, context.gid,
                                                             recording_profile=context.recording_profile)



def update_network_clamp(x, context=None):
    """

    :param x: array
    :param context: :class:'Context'
    """
    if context is None:
        raise RuntimeError('update_network: missing required Context object')

    param_values = context.from_param_vector(x)
    context.logger.info("parameter values: %s" % str(param_values))

    biophys_cell_dict = context.env.biophys_cells[context.population]
    for gid, biophys_cell in viewitems(biophys_cell_dict):
        for destination, source, sec_type, syn_name, param_path, param_value in param_values:
            if context.population != destination:
                continue

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

            context.logger.info("gid %d: updating parameter %s with value %s" % (gid, str(p), str(param_value)))

            synapses.modify_syn_param(biophys_cell, context.env, sec_type, syn_name,
                                      param_name=p, 
                                      value={s: param_value} if (s is not None) else param_value,
                                      filters={'sources': sources} if sources is not None else None,
                                      origin='soma', update_targets=True)
                
    context.pop_spike_dict = network_clamp.run(context.env, pc_runworker=False)
    



def gid_firing_rate_vector(population, gid, time_bins, n_trials, spkdict):
    rates_dict = defaultdict(list)
    mean_rates_dict = {}

    for i in range(n_trials):
        spkdict1 = {}
        if gid in spkdict[population]:
            spkdict1[gid] = spkdict[population][gid][i]
        else:
            spkdict1[gid] = np.asarray([], dtype=np.float32)
        spike_density_dict = spikedata.spike_density_estimate (population, spkdict1, time_bins)
        rates_dict[gid].append(spike_density_dict[gid]['rate'])

            
    mean_rate_vector = np.mean(np.row_stack(rates_dict[gid]), axis=0)

    if gid in spkdict[population]:
        context.logger.info('firing rate objective: mean rate vector: %s' % (str(mean_rate_vector)))

    return mean_rate_vector


def gid_state_values(population, gid, t_offset, n_trials, t_rec, state_recs_dict):

    state_filter = context.state_filter
    
    t_vec = np.asarray(t_rec.to_python(), dtype=np.float32)
    t_trial_inds = utils.get_trial_time_indices(t_vec, n_trials, t_offset)
    results_dict = {}
    filter_fun = None
    
    if state_filter == 'lowpass':
        filter_fun = lambda x, t: utils.get_low_pass_filtered_trace(x, t)
        
    state_values = None
    state_recs = state_recs_dict[gid]
    assert(len(state_recs) == 1)
    rec = state_recs[0]
    vec = np.asarray(rec['vec'].to_python(), dtype=np.float32)
    if filter_fun is None:
        data = np.asarray([ vec[t_inds] for t_inds in t_trial_inds ])
    else:
        data = np.asarray([ filter_fun(vec[t_inds], t_vec[t_inds])
                                    for t_inds in t_trial_inds ])
        
    state_values = []
    max_len = np.max(np.asarray([len(a) for a in data]))
    for state_value_array in data:
        this_len = len(state_value_array)
        if this_len < max_len:
            a = np.pad(state_value_array, (0, max_len-this_len), 'edge')
        else:
            a = state_value_array
        state_values.append(a)

    state_value_array = np.row_stack(state_values)
    m = np.mean(state_value_array, axis=0)
        
    return t_vec[t_trial_inds[0]], m


def compute_features(x, n, export=False):
    """

    :param x: array
    :param export: bool
    :return: dict
    """
    results = dict()
    update_source_contexts(x, context)
    context.env.results_file_id = '%s_%s' % \
        (context.interface.worker_id, datetime.datetime.today().strftime('%Y%m%d_%H%M%S'))


    firing_rate_vector = gid_firing_rate_vector(context.population, context.gid,
                                                context.time_bins, context.env.n_trials,
                                                context.pop_spike_dict)
    
    max_infld_firing_rate = np.max(firing_rate_vector[context.infld_idxs])
    mean_peak_firing_rate = np.mean(firing_rate_vector[context.peak_idxs])
    mean_trough_firing_rate = np.mean(firing_rate_vector[context.trough_idxs])
    if context.outfld_idxs is not None:
        mean_outfld_firing_rate = np.mean(firing_rate_vector[context.outfld_idxs] )
    else:
        mean_outfld_firing_rate = None

    results['%s in field max firing rate' % context.population] = max_infld_firing_rate
    results['%s out of field mean firing rate' % context.population] = mean_outfld_firing_rate
    results['%s mean peak firing rate' % context.population] = mean_peak_firing_rate
    results['%s mean trough firing rate' % context.population] = mean_trough_firing_rate

    t_s, mean_state_values = gid_state_values(context.population, context.gid,
                                              context.equilibration_duration, 
                                              context.env.n_trials, context.env.t_rec, 
                                              context.state_recs_dict)
    
    t_infld_ranges = context.t_infld_ranges
    t_outfld_ranges = context.t_outfld_ranges
    t_peak_ranges = context.t_peak_ranges
    t_trough_ranges = context.t_trough_ranges

    t_peak_idxs = np.concatenate([ np.where(np.logical_and(t_s >= r[0], t_s < r[1]))[0] for r in t_peak_ranges ])
    t_trough_idxs = np.concatenate([ np.where(np.logical_and(t_s >= r[0], t_s < r[1]))[0] for r in t_trough_ranges ])
    t_infld_idxs = np.concatenate([ np.where(np.logical_and(t_s >= r[0], t_s < r[1]))[0] for r in t_infld_ranges ])
    if context.t_outfld_ranges is not None:
        t_outfld_idxs = np.concatenate([ np.where(np.logical_and(t_s >= r[0], t_s < r[1]))[0] for r in t_outfld_ranges ])
    else:
        t_outfld_idxs = None
            
    max_peak_state_value = np.max(mean_state_values[t_peak_idxs])
    min_infld_state_value = np.min(mean_state_values[t_infld_idxs])
    mean_infld_state_value = np.mean(mean_state_values[t_infld_idxs])
    if t_outfld_idxs is not None:
        mean_outfld_state_value = np.mean(mean_state_values[t_outfld_idxs])
    else:
        mean_outfld_state_value = None
    
    results['%s peak max state value' % context.population] = max_peak_state_value
    results['%s in field min state value' % context.population] = min_infld_state_value
    results['%s in field mean state value' % context.population] = mean_infld_state_value
    results['%s out of field mean state value' % context.population] = mean_outfld_state_value

    return results



def get_objectives(features, n, export=False):
    """

    :param features: dict
    :param export: bool
    :return: tuple of dict
    """
    objectives = dict()

    target_min_infld_rate = context.target_val['%s in field min firing rate' % context.population]
    target_max_infld_rate = context.target_val['%s in field max firing rate' % context.population]
    rate_baseline = context.target_val['%s baseline firing rate' % context.population]
    state_baseline = context.target_val['%s baseline state value' % context.population]
    state_peak = context.target_val['%s peak state value' % context.population]
    
    max_infld_rate = features['%s in field max firing rate' % context.population]
    mean_outfld_rate = features['%s out of field mean firing rate' % context.population]
    mean_peak_rate = features['%s mean peak firing rate' % context.population]
    mean_trough_rate = features['%s mean trough firing rate' % context.population]

    context.logger.info('rate selectivity objective: max infld rate: %.02f target max infld: %.02f' % (max_infld_rate, target_max_infld_rate))

    if max_infld_rate > target_max_infld_rate:
        rate_selectivity = None
        rate_selectivity_residual = 1.0
    elif mean_outfld_rate is None:
        rate_selectivity = ((mean_peak_rate - mean_trough_rate) ** 2.) / ((max(mean_trough_rate - target_min_infld_rate, 1.0)) ** 2.)
        rate_selectivity_range = context.max_infld_target_rate - context.min_infld_target_rate
        rate_selectivity_residual = (rate_selectivity_range**2 - rate_selectivity ) / (rate_selectivity_range**2)
        context.logger.info('rate selectivity objective: mean peak/mean trough/selectivity/residual: %.02f %.02f %.04f %.04f' % ( mean_peak_rate, mean_trough_rate, rate_selectivity, rate_selectivity_residual))
    else:
        rate_selectivity = (np.clip(max_infld_rate - mean_outfld_rate, 0., None) ** 2.)  / (max(mean_outfld_rate/rate_baseline, 1.0) ** 2.)
        rate_selectivity_range = context.max_infld_target_rate
        rate_selectivity_residual = (rate_selectivity_range**2 - rate_selectivity ) / (rate_selectivity_range**2)
        context.logger.info('rate selectivity objective: max in/mean out/selectivity/residual: %.02f %.02f %.04f %.04f' % (max_infld_rate, mean_outfld_rate, rate_selectivity, rate_selectivity_residual))

    max_peak_infld_state = features['%s peak max state value' % context.population] 
    min_infld_state = features['%s in field min state value' % context.population]
    mean_infld_state = features['%s in field mean state value' % context.population]
    mean_outfld_state = features['%s out of field mean state value' % context.population] 

    if mean_outfld_state is None:
        state_selectivity = np.clip(max_peak_infld_state - min_infld_state, 0., None) ** 2.
        state_selectivity_range = state_peak - state_baseline
        state_selectivity_residual = (state_selectivity_range**2 - state_selectivity) / (state_selectivity_range**2)
        context.logger.info('state selectivity objective: max/min/mean in: %.02f / %.02f / %.02f selectivity/residual: %.04f %.04f' % (max_peak_infld_state, min_infld_state, mean_infld_state, state_selectivity, state_residual))
    else:
        state_selectivity = np.clip(np.clip(max_peak_infld_state - mean_outfld_state, 0., None) - (mean_outfld_state - state_baseline), 0., None) ** 2.        
        state_selectivity_range = state_peak - state_baseline
        state_selectivity_residual = (state_selectivity_range**2 - state_selectivity) / (state_selectivity_range**2)

        context.logger.info('state selectivity objective: max in/mean in/mean out: %.02f / %.02f / %.02f selectivity/residual: %.04f %.04f' % (max_peak_infld_state, mean_infld_state, mean_outfld_state, state_selectivity, state_selectivity_residual))


    objectives['%s rate selectivity' % context.population] = rate_selectivity_residual
    objectives['%s state value selectivity' % context.population] = state_selectivity_residual

    return features, objectives


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv) + 1):])
