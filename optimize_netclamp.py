#!/usr/bin/env python
"""
Dentate Gyrus place field optimization script for optimization with nested.optimize
"""
__author__ = 'See AUTHORS.md'
import os, sys, logging
import click
import numpy as np
from mpi4py import MPI
import dentate
from dentate import cells, synapses, spikedata, stimulus, utils, network_clamp
from dentate.env import Env
import nested
from nested.optimize_utils import *
from nested.parallel import get_parallel_interface

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
              default='../config/DG_optimize_network_subworlds_config.yaml')
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
    if 'results_file_id' not in context():
        context.results_file_id = 'DG_optimize_pf_%s_%s' % \
                             (context.interface.worker_id, datetime.datetime.today().strftime('%Y%m%d_%H%M'))
    if 'env' not in context():
        try:
            init_network_clamp()
        except Exception as err:
            context.logger.exception(err)
            raise err

    param_bounds = {}
    param_names = []
    param_initial_dict = {}
    param_range_tuples = []
    opt_targets = {}

    param_bounds, param_names, param_initial_dict, param_range_tuples = \
        network_clamp.optimize_params(context.env, context.population, param_type, param_config_name)

    if (context.population in context.env.netclamp_config.optimize_parameters):
        opt_params = context.env.netclamp_config.optimize_parameters[context.population]
        param_ranges = opt_params['Parameter ranges']
    else:
        raise RuntimeError(
            "optimize_network_subworlds: population %s does not have optimization configuration" % context.population)

    def from_param_dict(params_dict):
        result = []
        for param_pattern, (update_operator, population, source, sec_type, syn_name, param_name, param_range) in zip(param_names, param_range_tuples):
            result.append((update_operator, population, source, sec_type, syn_name, param_name, params_dict[param_pattern]))
        return result

    context.param_names = param_names
    context.bounds = [ param_bounds[key] for key in param_names ]
    context.x0 = param_initial_dict
    context.from_param_dict = from_param_dict
    context.target_val = opt_targets
    context.target_range = opt_targets

    target_rate_map_path = context.kwargs['target_rate_map_path']
    target_rate_map_namespace = context.kwargs['target_rate_map_namespace']
    target_rate_map_arena = context.kwargs['target_rate_map_arena']
    target_rate_map_trajectory = context.kwargs['target_rate_map_trajectory']
    trj_x, trj_y, trj_d, trj_t = stimulus.read_trajectory(target_rate_map_path, target_rate_map_arena, target_rate_map_trajectory)

    time_range = (0., min(np.max(trj_t), context.kwargs['t_max']))
    time_step = context.env.stimulus_config['Temporal Resolution']
    context.time_bins = np.arange(time_range[0], time_range[1], time_step)

    input_namespace = '%s %s %s' % (target_rate_map_namespace, target_rate_map_arena, target_rate_map_trajectory)
    it = read_cell_attribute_selection(target_rate_map_path, context.population, namespace=input_namespace,
                                        selection=list(context.gid), mask=set(['Trajectory Rate Map']))
    _, attr_dict = next(it)
    trj_rate_map = attr_dict['Trajectory Rate Map']
    target_rate_vector = np.interp(context.time_bins, trj_t, trj_rate_map)
    
    context.target_val['%s firing rate vector' % context.population] = target_rate_vector


    
def config_controller():
    """

    """
    utils.config_logging(context.verbose)
    context.logger = utils.get_script_logger(os.path.basename(__file__))
    if 'results_file_id' not in context():
        context.results_file_id = 'DG_optimize_pf_%s_%s' % \
                             (context.interface.worker_id, datetime.datetime.today().strftime('%Y%m%d_%H%M'))
    if 'env' not in context():
        try:
            init_env()
        except Exception as err:
            context.logger.exception(err)
            raise err
    
        
def init_env():
    """

    """
    context.env = Env(comm=context.comm, results_file_id=context.results_file_id, **context.kwargs)

    
def init_network_clamp():
    """

    """
    np.seterr(all='raise')
    context.env = Env(comm=context.comm, results_file_id=context.results_file_id, **context.kwargs)
    context.gid = context.kwargs['gid']
    network_clamp.init(context.env,
                       context.kwargs['population'],
                       set([context.gid]),
                       arena_id=context.kwargs['arena_id'],
                       trajectory_id=context.kwargs['trajectory_id'],
                       n_trials=context.kwargs['n_trials'],
                       spike_events_path=context.kwargs['spike_events_path'],
                       spike_events_namespace=context.kwargs['spike_events_namespace'], 
                       spike_events_t=context.kwargs['spike_train_attr_name'],
                       input_features_path=context.kwargs['input_features_path'],
                       input_features_namespaces=context.kwargs['input_features_namespaces'],
                       t_min=context.kwargs['t_min'],
                       t_max=context.kwargs['t_max'])
    
    context.equilibration_duration = float(env.stimulus_config['Equilibration Duration'])
    

    context.recording_profile = { 'label': 'network_clamp.state.%s' % state_variable,
                                  'dt': 0.1,
                                  'section quantity': {
                                      state_variable: { 'swc types': ['soma'] }
                                  }
                                }    
    context.state_recs_dict = {}
    context.state_recs_dict[gid] = cells.record_cell(context.env, context.population,
                                                     context.gid, recording_profile=recording_profile)



def update_network_clamp(x, context=None):
    """

    :param x: array
    :param context: :class:'Context'
    """
    if context is None:
        raise RuntimeError('update_network: missing required Context object')

    param_values = context.from_param_vector(x)

    for destination, source, sec_type, syn_name, param_path, param_value in param_values:
        if context.population != destination:
            continue
        conn_params = context.env.connection_config[context.population][source].mechanisms
        sec_type_index = context.env.SWC_Types[sec_type]
        if 'default' in conn_params:
            mech_params = conn_params['default'][syn_name]
        else:
            mech_params = conn_params[sec_type_index][syn_name]
        if isinstance(param_path, tuple):
            p, s = param_path
            mech_param = mech_params[p]
            try:
                mech_param[s] = param_value
            except Exception as err:
                context.logger.exception('source: %s sec type: %s syn name: %s param path: %s mech params: %s' %
                                         (str(source), str(sec_type), str(syn_name), str(param_path), str(mech_params)))
                context.logger.exception(err)
                raise err

                
        biophys_cell_dict = context.env.biophys_cells[context.population]
        for gid, biophys_cell in viewitems(biophys_cell_dict):
            for destination, source, sec_type, syn_name, param_path, param_value in param_values:
                if context.population != destination:
                    continue
                
                if isinstance(param_path, tuple):
                    p, s = param_path
                else:
                    p = param_path
                synapses.modify_syn_param(biophys_cell, context.env, sec_type, syn_name,
                                          param_name=p, value=param_value,
                                          filters={'sources': [source]},
                                          origin='soma', update_targets=True)
                
    context.pop_spike_dict = network_clamp.run(context.env)
    



def gid_firing_rate_vectors(population, gid, time_bins, spkdict):
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
        logger.info('firing rate objective: trial %d spike times of gid %i: %s' % (i, gid, str(spkdict[population][gid])))
        logger.info('firing rate objective: trial %d firing rate of gid %i: %s' % (i, gid, str(spike_density_dict[gid])))
        logger.info('firing rate objective: trial %d firing rate min/max of gid %i: %.02f / %.02f Hz' % (i, gid, np.min(rates_dict[gid]), np.max(rates_dict[gid])))
            
    mean_rate_vector_dict = np.mean(np.row_stack(rates_dict[gid]), axis=0)

    logger.info('firing rate objective: mean firing rate vector of gid %i: %s' % (gid, str(mean_rate_vector_dict[gid])))
    logger.info('firing rate objective: mean firing rate min/max of gid %i: %.02f / %.02f Hz' % (gid, np.min(mean_rate_vector_dict[gid]), np.max(mean_rate_vector_dict[gid])))
        
    return mean_rate_vector_dict

def gid_state_values(t_offset, n_trials, t_rec, state_recs_dict):
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


def compute_features_firing_rate_vector(x, export=False):
    """

    :param x: array
    :param export: bool
    :return: dict
    """
    results = dict()
    update_source_contexts(x, context)
    context.env.results_file_id = '%s_%s' % \
        (context.interface.worker_id, datetime.datetime.today().strftime('%Y%m%d_%H%M%S'))

    firing_rate_vectors_dict = gid_firing_rate_vectors(context.population, context.gid,
                                                       context.time_bins, context.pop_spike_dict)
    
    firing_rate_vector =  firing_rate_vectors_dict[gid]
    
    results['%s firing rate vector' % context.population] = firing_rate_vector

    return results


def compute_features_state_vector_v(x, export=False):
    """

    :param x: array
    :param export: bool
    :return: dict
    """
    results = dict()
    update_source_contexts(x, context)
    context.env.results_file_id = '%s_%s' % \
                             (context.interface.worker_id, datetime.datetime.today().strftime('%Y%m%d_%H%M%S'))

    pop_spike_dict = network_clamp.run(context.env)

    time_bins = context.time_bins

    for pop_name in context.target_populations:

        state_values_dict = gid_state_values(context.equilibration_duration, 
                                             context.env.n_trials, context.env.t_rec, 
                                             context.state_recs_dict)

        mean_value =  state_values_dict[gid].mean()

        results['%s state vector v' % pop_name] = mean_value

    return results


def get_objectives(features, export=False):
    """

    :param features: dict
    :param export: bool
    :return: tuple of dict
    """
    objectives = dict()
    for feature_key in context.feature_names:
        objectives[feature_key] = ((features[feature_key] - context.target_val[feature_key]) / context.target_range[feature_key]) ** 2.

    return features, objectives


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv) + 1):])
