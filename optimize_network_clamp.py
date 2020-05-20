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
from dentate import cells, synapses, spikedata, stimulus, utils, network_clamp
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


#sys_excepthook = sys.excepthook
#sys.excepthook = mpi_excepthook


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



def read_target_rate_vector(context, eps=1e-2):
    """
    """

    target_rate_map_arena = context.init_params['target_rate_map_arena']
    target_rate_map_trajectory = context.init_params['target_rate_map_trajectory']
    target_rate_map_path = context.init_params['target_rate_map_path']
    target_rate_map_namespace = context.init_params.get('target_rate_map_namespace', 'Input Spikes')
    trj_x, trj_y, trj_d, trj_t = stimulus.read_trajectory(target_rate_map_path, target_rate_map_arena, target_rate_map_trajectory)

    time_range = (0., min(np.max(trj_t), context.init_params['tstop']))
    time_step = context.env.stimulus_config['Temporal Resolution']
    context.time_bins = np.arange(time_range[0], time_range[1], time_step)

    input_namespace = '%s %s %s' % (target_rate_map_namespace, target_rate_map_arena, target_rate_map_trajectory)
    it = read_cell_attribute_selection(target_rate_map_path, context.population, namespace=input_namespace,
                                        selection=[context.gid], mask=set(['Trajectory Rate Map']),
                                        comm=context.comm)
    _, attr_dict = next(it)
    trj_rate_map = attr_dict['Trajectory Rate Map']
    target_rate_vector = np.interp(context.time_bins, trj_t, trj_rate_map)

    target_rate_vector[np.abs(target_rate_vector) < eps] = 0.
    
    
    return target_rate_vector
    

    
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
    opt_targets = {}

    param_bounds, param_names, param_initial_dict, param_range_tuples = \
        network_clamp.optimize_params(context.env, context.population, 'synaptic',
                                      context.param_config_name)

    if (context.population in context.env.netclamp_config.optimize_parameters['synaptic']):
        opt_params = context.env.netclamp_config.optimize_parameters['synaptic'][context.population]
        param_ranges = opt_params['Parameter ranges']
    else:
        raise RuntimeError(
            "optimize_network_subworlds: population %s does not have optimization configuration" % context.population)

    def from_param_vector(params):
        result = []
        assert (len(params) == len(param_range_tuples))
        for i, (param_pattern, (update_operator, population, source, sec_type, syn_name, param_name, param_range)) in enumerate(zip(param_names, param_range_tuples)):
            result.append((update_operator, population, source, sec_type, syn_name, param_name, params[i]))
        return result

    context.param_names = param_names
    context.bounds = [ param_bounds[key] for key in param_names ]
    context.x0 = param_initial_dict
    context.from_param_vector = from_param_vector

    target_rate_vector = read_target_rate_vector(context)

    outfld_ranges = tuple(utils.contiguous_ranges(target_rate_vector <= 0.))
    context.outfld_idxs = np.r_[outfld_ranges]
    infld_ranges = tuple(utils.contiguous_ranges(target_rate_vector > 0.))
    context.infld_idxs = np.r_[infld_ranges]

    
    context.target_val['%s in field firing rate vector' % context.population] = target_rate_vector[context.infld_idxs]
    context.target_val['%s out of field firing rate vector' % context.population] = target_rate_vector[context.outfld_idxs]

    
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
        for update_operator, destination, source, sec_type, syn_name, param_path, param_value in param_values:
            if context.population != destination:
                continue

            if isinstance(param_path, tuple):
                p, _ = param_path
            else:
                p = param_path

            sources = None
            if isinstance(source, tuple):
                sources = list(source)
            else:
                if source is not None:
                    sources = [source]

            synapses.modify_syn_param(biophys_cell, context.env, sec_type, syn_name,
                                      param_name=p, value=param_value,
                                      filters={'sources': sources} if sources is not None else None,
                                      update_operator=update_operator,
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
        if gid in spkdict[population]:
            context.logger.info('firing rate objective: trial %d spike times of gid %i: %s' % (i, gid, str(spkdict[population][gid])))
        context.logger.info('firing rate objective: trial %d firing rate of gid %i: %s' % (i, gid, str(spike_density_dict[gid])))
        context.logger.info('firing rate objective: trial %d firing rate min/max of gid %i: %.02f / %.02f Hz' % (i, gid, np.min(rates_dict[gid]), np.max(rates_dict[gid])))
            
    mean_rate_vector = np.mean(np.row_stack(rates_dict[gid]), axis=0)

    context.logger.info('firing rate objective: mean firing rate vector of gid %i: %s' % (gid, str(mean_rate_vector)))
    context.logger.info('firing rate objective: mean firing rate min/max of gid %i: %.02f / %.02f Hz' % (gid, np.min(mean_rate_vector), np.max(mean_rate_vector)))
        
    return mean_rate_vector

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
        context.logger.info('state value objective: mean value of %s of gid %i is %.2f' % (state_variable, gid, m))
        results_dict[gid] = m
    return results_dict


def compute_features_firing_rate_vector(x, n, export=False):
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
    
    infld_firing_rate = firing_rate_vector[context.infld_idxs]
    outfld_firing_rate = firing_rate_vector[context.outfld_idxs]
    
    context.logger.info("in field firing rate: %s" % str(infld_firing_rate))
    context.logger.info("out of field firing rate: %s" % str(outfld_firing_rate))
    
    results['%s in field firing rate vector' % context.population] = infld_firing_rate
    results['%s out of field firing rate vector' % context.population] = outfld_firing_rate

    return results



def get_objectives(features, n, export=False):
    """

    :param features: dict
    :param export: bool
    :return: tuple of dict
    """
    objectives = dict()
    for feature_key in context.feature_names:
        objectives[feature_key] = np.square(np.subtract(features[feature_key],
                                                        context.target_val[feature_key])).mean()

    return features, objectives


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv) + 1):])
