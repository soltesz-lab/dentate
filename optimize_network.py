#!/usr/bin/env python
"""
Dentate Gyrus model optimization script for optimization with dmosopt
"""

import os, sys, logging, datetime, gc
from functools import partial
import click
import numpy as np
from mpi4py import MPI
from neuroh5.io import scatter_read_cell_attribute_selection, read_cell_attribute_info
from collections import defaultdict, namedtuple
import dentate
from dentate import network, network_clamp, synapses, spikedata, stimulus, utils
from dentate.env import Env
from dentate.utils import Context, read_from_yaml, list_find, viewitems, get_module_logger
import dmosopt

context = Context()
logger = get_module_logger(__name__)

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

SynParam = namedtuple('SynParam',
                      ['population',
                       'source',
                       'sec_type',
                       'syn_name',
                       'param_path',
                       'param_range'])

def syn_param_from_dict(d):
    return SynParam(*[d[key] for key in SynParam._fields])
             

OptConfig = namedtuple("OptConfig",
                       ['param_bounds', 
                        'param_names', 
                        'param_initial_dict', 
                        'param_tuples', 
                        'opt_targets'])


def dmosopt_broker_init(broker, *args):
    broker.sub_comm.barrier()
    broker.group_comm.barrier()
    logger.info(f"broker_init: broker {broker.group_comm.rank} done")

@click.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.option("--config-path", required=True, type=str, help='optimization configuration file name')
@click.option("--target-rate-map-path", required=False, type=click.Path(),
              help='path to neuroh5 file containing target rate maps used for rate optimization')
@click.option("--target-rate-map-namespace", type=str, required=False, default='Input Spikes',
              help='namespace containing target rate maps used for rate optimization')
@click.option("--optimize-file-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default='results')
@click.option("--optimize-file-name", type=click.Path(exists=False, file_okay=True, dir_okay=False), default='dmosopt.optimize_network.h5')
@click.option("--nprocs-per-worker", type=int, default=1)
@click.option("--n-iter", type=int, default=1)
@click.option("--verbose", '-v', is_flag=True)
def main(config_path, target_rate_map_path, target_rate_map_namespace, optimize_file_dir, optimize_file_name, nprocs_per_worker, n_iter, verbose):

    network_args = click.get_current_context().args
    network_config = {}
    for arg in network_args:
        kv = arg.split("=")
        if len(kv) > 1:
            k,v = kv
            network_config[k.replace('--', '').replace('-', '_')] = v
        else:
            k = kv[0]
            network_config[k.replace('--', '').replace('-', '_')] = True

    operational_config = read_from_yaml(config_path)
    if target_rate_map_path is not None:
        operational_config['target_rate_map_path'] = target_rate_map_path
    if target_rate_map_namespace is not None:
        operational_config['target_rate_map_namespace'] = target_rate_map_namespace

    network_config.update(operational_config.get('kwargs', {}))
    env = Env(**network_config)
    context.update({'env': env})

    objective_names = operational_config['objective_names']
    param_config_name = operational_config['param_config_name']
    target_populations = operational_config['target_populations']
    network_optimization_config = make_optimization_config(env, target_populations, param_config_name)

    opt_targets = network_optimization_config.opt_targets
    param_names = network_optimization_config.param_names
    param_tuples = network_optimization_config.param_tuples
    hyperprm_space = { param_pattern: [param_tuple.param_range[0], param_tuple.param_range[1]]
                       for param_pattern, param_tuple in 
                           zip(param_names, param_tuples) }

    init_objfun = 'init_network_objfun'
    init_params = { 'operational_config': operational_config,
                    'opt_targets': opt_targets,
                    'param_tuples': [ param_tuple._asdict() for param_tuple in param_tuples ],
                    'param_names': param_names
                    }
    init_params.update(network_config.items())
    

    # Create an optimizer
    dmosopt_params = {'opt_id': 'dmosopt_optimize_network',
                      'obj_fun_init_name': init_objfun, 
                      'obj_fun_init_module': 'dentate.optimize_network',
                      'obj_fun_init_args': init_params,
                      'reduce_fun_name': 'compute_objectives',
                      'reduce_fun_module': 'dentate.optimize_network',
                      'reduce_fun_args': (operational_config, opt_targets),
                      'problem_parameters': {},
                      'space': hyperprm_space,
                      'objective_names': objective_names,
                      'n_initial': 3,
                      'n_iter': n_iter,
                      'file_path': f'{optimize_file_dir}/{optimize_file_name}',
                      'save': True,
                      'save_eval': 5
                      }
    
    dmosopt_params['broker_fun_name'] = 'dmosopt_broker_init'
    dmosopt_params['broker_module_name'] = 'dentate.optimize_network'

    best = dmosopt.run(dmosopt_params, spawn_workers=True, 
                       nprocs_per_worker=nprocs_per_worker, 
                       verbose=True, worker_debug=True)
    

def make_optimization_config(env, pop_names, param_config_name, param_type='synaptic'):
    """Constructs a flat list representation of synaptic optimization parameters based on network clamp optimization configuration."""
    
    param_bounds = {}
    param_names = []
    param_initial_dict = {}
    param_tuples = []
    opt_targets = {}

    for pop_name in pop_names:
        if param_type == 'synaptic':
            if pop_name in env.netclamp_config.optimize_parameters['synaptic']:
                opt_params = env.netclamp_config.optimize_parameters['synaptic'][pop_name]
                param_ranges = opt_params['Parameter ranges'][param_config_name]
            else:
                raise RuntimeError(
                    "make_optimization_config: population %s does not have optimization configuration" % pop_name)
            for target_name, target_val in viewitems(opt_params['Targets']):
                opt_targets['%s %s' % (pop_name, target_name)] = target_val
            keyfun = lambda kv: str(kv[0])
            for source, source_dict in sorted(viewitems(param_ranges), key=keyfun):
                for sec_type, sec_type_dict in sorted(viewitems(source_dict), key=keyfun):
                    for syn_name, syn_mech_dict in sorted(viewitems(sec_type_dict), key=keyfun):
                        for param_fst, param_rst in sorted(viewitems(syn_mech_dict), key=keyfun):
                            if isinstance(param_rst, dict):
                                for const_name, const_range in sorted(viewitems(param_rst)):
                                    param_path = (param_fst, const_name)
                                    param_tuples.append(SynParam(pop_name, source, sec_type, syn_name, param_path, const_range))
                                    param_key = '%s.%s.%s.%s.%s.%s' % (pop_name, str(source), sec_type, syn_name, param_fst, const_name)
                                    param_initial_value = (const_range[1] - const_range[0]) / 2.0
                                    param_initial_dict[param_key] = param_initial_value
                                    param_bounds[param_key] = const_range
                                    param_names.append(param_key)
                            else:
                                param_name = param_fst
                                param_range = param_rst
                                param_tuples.append(SynParam(pop_name, source, sec_type, syn_name, param_name, param_range))
                                param_key = '%s.%s.%s.%s.%s' % (pop_name, source, sec_type, syn_name, param_name)
                                param_initial_value = (param_range[1] - param_range[0]) / 2.0
                                param_initial_dict[param_key] = param_initial_value
                                param_bounds[param_key] = param_range
                                param_names.append(param_key)
        
        else:
            raise RuntimeError("make_optimization_config: unknown parameter type %s" % param_type)

    return OptConfig(param_bounds=param_bounds, 
                     param_names=param_names, 
                     param_initial_dict=param_initial_dict, 
                     param_tuples=param_tuples, 
                     opt_targets=opt_targets)


def update_network_params(env, param_tuple_values):

    for param_tuple, param_value in param_tuple_values:

        pop_name = param_tuple.population
        biophys_cell_dict = env.biophys_cells[pop_name]
        synapse_config = env.celltypes[pop_name]['synapses']
        weights_dict = synapse_config.get('weights', {})

        for gid in biophys_cell_dict:

            biophys_cell = biophys_cell_dict[gid]
            is_reduced = False
            if hasattr(biophys_cell, 'is_reduced'):
                is_reduced = biophys_cell.is_reduced

            source = param_tuple.source
            sec_type = param_tuple.sec_type
            syn_name = param_tuple.syn_name
            param_path = param_tuple.param_path
            
            if isinstance(param_path, list) or isinstance(param_path, tuple):
                p, s = param_path
            else:
                p, s = param_path, None

            sources = None
            if isinstance(source, list) or isinstance(source, tuple):
                sources = source
            else:
                if source is not None:
                    sources = [source]

            if isinstance(sec_type, list) or isinstance(sec_type, tuple):
                sec_types = sec_type
            else:
                sec_types = [sec_type]
            for this_sec_type in sec_types:
                synapses.modify_syn_param(biophys_cell, env, this_sec_type, syn_name,
                                              param_name=p, 
                                              value={s: param_value} if (s is not None) else param_value,
                                              filters={'sources': sources} if sources is not None else None,
                                              origin=None if is_reduced else 'soma', 
                                              update_targets=True)


def rate_maps_from_features (env, pop_name, input_features_path, input_features_namespace, cell_index_set,
                             time_range=None, n_trials=1):
    
    """Initializes presynaptic spike sources from a file with input selectivity features represented as firing rates."""
        
    if time_range is not None:
        if time_range[0] is None:
            time_range[0] = 0.0

    spatial_resolution = float(env.stimulus_config['Spatial Resolution'])
    temporal_resolution = float(env.stimulus_config['Temporal Resolution'])
    
    this_input_features_namespace = '%s %s' % (input_features_namespace, env.arena_id)
    
    input_features_attr_names = ['Selectivity Type', 'Num Fields', 'Field Width', 'Peak Rate',
                                 'Module ID', 'Grid Spacing', 'Grid Orientation',
                                 'Field Width Concentration Factor', 
                                 'X Offset', 'Y Offset']
    
    selectivity_type_names = { i: n for n, i in viewitems(env.selectivity_types) }

    arena = env.stimulus_config['Arena'][env.arena_id]
    arena_x, arena_y = stimulus.get_2D_arena_spatial_mesh(arena=arena, spatial_resolution=spatial_resolution)
    
    trajectory = arena.trajectories[env.trajectory_id]
    t, x, y, d = stimulus.generate_linear_trajectory(trajectory, temporal_resolution=temporal_resolution)
    if time_range is not None:
        t_range_inds = np.where((t < time_range[1]) & (t >= time_range[0]))[0] 
        t = t[t_range_inds]
        x = x[t_range_inds]
        y = y[t_range_inds]
        d = d[t_range_inds]

    input_rate_map_dict = {}
    pop_index = int(env.Populations[pop_name])
    input_features_iter = scatter_read_cell_attribute_selection(input_features_path, pop_name,
                                                                selection=list(cell_index_set),
                                                                namespace=this_input_features_namespace,
                                                                mask=set(input_features_attr_names), 
                                                                comm=env.comm, io_size=env.io_size)
    for gid, selectivity_attr_dict in input_features_iter:

        this_selectivity_type = selectivity_attr_dict['Selectivity Type'][0]
        this_selectivity_type_name = selectivity_type_names[this_selectivity_type]
        input_cell_config = stimulus.get_input_cell_config(selectivity_type=this_selectivity_type,
                                                           selectivity_type_names=selectivity_type_names,
                                                           selectivity_attr_dict=selectivity_attr_dict)
        if input_cell_config.num_fields > 0:
            rate_map = input_cell_config.get_rate_map(x=x, y=y)
            input_rate_map_dict[gid] = rate_map
            
    return input_rate_map_dict



def init_network_objfun(operational_config, opt_targets, param_names, param_tuples, worker, **kwargs):

    param_tuples = [ syn_param_from_dict(param_tuple) for param_tuple in param_tuples ]

    objective_names = operational_config['objective_names']
    target_populations = operational_config['target_populations']
    target_rate_map_path = operational_config['target_rate_map_path']
    target_rate_map_namespace = operational_config['target_rate_map_namespace']

    kwargs['results_file_id'] = 'DG_optimize_network_%s_%s' % \
                                (worker.worker_id, datetime.datetime.today().strftime('%Y%m%d_%H%M'))

    logger = utils.get_script_logger(os.path.basename(__file__))
    env = init_network(comm=MPI.COMM_WORLD, kwargs=kwargs)
    gc.collect()

    t_start = 50.
    t_stop = env.tstop
    time_range = (t_start, t_stop)

    target_trj_rate_map_dict = {}
    target_rate_map_arena = env.arena_id
    target_rate_map_trajectory = env.trajectory_id
    for pop_name in target_populations:
        if ('%s target rate dist residual' % pop_name) not in objective_names:
            continue
        my_cell_index_set = set(env.biophys_cells[pop_name].keys())
        trj_rate_maps = {}
        trj_rate_maps = rate_maps_from_features(env, pop_name, target_rate_map_path, 
                                                target_rate_map_namespace,
                                                cell_index_set=list(my_cell_index_set),
                                                time_range=time_range)
        target_trj_rate_map_dict[pop_name] = trj_rate_maps

    def from_param_dict(params_dict):
        result = []
        for param_name, param_tuple in zip(param_names, param_tuples):
            result.append((param_tuple, params_dict[param_name]))
        return result

    env.comm.barrier()
    worker.parent_comm.barrier()
    return partial(network_objfun, env, operational_config, opt_targets,
                   target_trj_rate_map_dict, from_param_dict, t_start, t_stop, target_populations)

    
def init_network(comm, kwargs):
    np.seterr(all='raise')
    env = Env(comm=comm, **kwargs)
    network.init(env)
    return env


def network_objfun(env, operational_config, opt_targets,
                   target_trj_rate_map_dict, from_param_dict, t_start, t_stop, target_populations, x):

    param_tuple_values = from_param_dict(x)
    update_network_params(env, param_tuple_values)

    return compute_network_features(env, operational_config, opt_targets, target_trj_rate_map_dict, 
                                    t_start, t_stop, target_populations)


def compute_network_features(env, operational_config, opt_targets, target_trj_rate_map_dict, 
                             t_start, t_stop, target_populations):

    target_populations = operational_config['target_populations']

    features_dict = dict()
 
    temporal_resolution = float(env.stimulus_config['Temporal Resolution'])
    time_bins  = np.arange(t_start, t_stop, temporal_resolution)

    env.tstop = t_stop
    network.run(env, output=False, shutdown=False)

    pop_spike_dict = spikedata.get_env_spike_dict(env, include_artificial=False)

    for pop_name in target_populations:

        n_active = 0
        sum_mean_rate = 0.
        spike_density_dict = spikedata.spike_density_estimate (pop_name, pop_spike_dict[pop_name], time_bins)
        for gid, dens_dict in utils.viewitems(spike_density_dict):
            mean_rate = np.mean(dens_dict['rate'])
            sum_mean_rate += mean_rate
            if mean_rate > 0.:
                n_active += 1
        
        n_total = len(env.cells[pop_name]) - len(env.artificial_cells[pop_name])

        n_residual = 0
        sum_target_rate_dist_residual = None
        if pop_name in target_trj_rate_map_dict:
            pop_target_trj_rate_map_dict = target_trj_rate_map_dict[pop_name]
            n_residual = len(pop_target_trj_rate_map_dict)
            target_rate_dist_residuals = []
            for gid in pop_target_trj_rate_map_dict:
                target_trj_rate_map = pop_target_trj_rate_map_dict[gid]
                rate_map_len = len(target_trj_rate_map)
                if gid in spike_density_dict:
                    residual = np.mean(target_trj_rate_map - spike_density_dict[gid]['rate'][:rate_map_len])
                else:
                    residual = np.mean(target_trj_rate_map)
                target_rate_dist_residuals.append(residual)
            sum_target_rate_dist_residual = np.sum(target_rate_dist_residuals)
    
        pop_features_dict = {}
        pop_features_dict['n_total'] = n_total
        pop_features_dict['n_active'] = n_active
        pop_features_dict['n_residual'] = n_residual
        pop_features_dict['sum_mean_rate'] = sum_mean_rate
        
        if sum_target_rate_dist_residual is not None:
            pop_features_dict['sum_target_rate_dist_residual'] = sum_target_rate_dist_residual

        features_dict[pop_name] = pop_features_dict

    env.comm.barrier()
    return features_dict


def compute_objectives(features, operational_config, opt_targets):

    objectives_dict = {}

    target_populations = operational_config['target_populations']
    for pop_name in target_populations:
        
        pop_features_dicts = [ features_dict[0][pop_name] for features_dict in features ]

        has_rate_dist_residual = 'sum_target_rate_dist_residual' in pop_features_dicts[0]
        sum_target_rate_dist_residual = 0. if has_rate_dist_residual else None
        n_total = 0
        n_active = 0
        n_residual = 0
        sum_mean_rate = 0.
        for pop_feature_dict in pop_features_dicts:
            n_total += pop_feature_dict['n_total']
            n_active += pop_feature_dict['n_active']
            n_residual += pop_feature_dict['n_residual']
            sum_mean_rate += pop_feature_dict['sum_mean_rate']
            if has_rate_dist_residual:
                sum_target_rate_dist_residual += pop_feature_dict['sum_target_rate_dist_residual']

        if n_active > 0:
            mean_rate = sum_mean_rate / n_active
        else:
            mean_rate = 0.

        if n_total > 0:
            fraction_active = n_active / n_total
        else:
            fraction_active = 0.

        mean_target_rate_dist_residual = None
        if has_rate_dist_residual:
            mean_target_rate_dist_residual = sum_target_rate_dist_residual / n_residual

        logger.info(f'population {pop_name}: n_active = {n_active} n_total = {n_total} mean rate = {mean_rate}')
        logger.info(f'population {pop_name}: n_residual = {n_residual} sum_target_rate_dist_residual = {sum_target_rate_dist_residual}')

        objectives_dict['%s fraction active' % pop_name] = fraction_active
        objectives_dict['%s firing rate' % pop_name] = mean_rate
        if mean_target_rate_dist_residual is not None:
            objectives_dict['%s target rate dist residual' % pop_name] = mean_target_rate_dist_residual

    logger.info(f'objectives: {objectives_dict}')
    objective_names = operational_config['objective_names']
    target_vals = opt_targets
    target_ranges = opt_targets
    objectives = []
    for key in objective_names:
        objective_val = objectives_dict[key]
        logger.info(f'objective {key}: {objective_val}')
        if key in target_vals:
            logger.info(f'objective {key}: target: {target_vals[key]}')
            objective = (objective_val - target_vals[key]) ** 2.
        else:
            objective = objective_val ** 2.
        logger.info(f'normalized objective {key}: {objective}')
        objectives.append(objective)

    result = np.asarray(objectives)
    logger.info(f'objectives result: {result}')

    return {0: result}


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv) + 1):])
