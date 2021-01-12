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
from dentate import network, network_clamp, synapses, spikedata, stimulus, utils, optimization
from dentate.env import Env
from dentate.utils import read_from_yaml, list_find, viewitems, get_module_logger
from dentate.optimization import (SynParam, OptConfig, syn_param_from_dict, optimization_params, 
                                  update_network_params, rate_maps_from_features, network_features)
from dmosopt import dmosopt

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


def dmosopt_broker_init(broker, *args):
    broker.group_comm.barrier()
    logger.info(f"broker_init: broker {broker.group_comm.rank} done")

@click.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.option("--config-path", required=True, type=str, help='optimization configuration file name')
@click.option("--target-features-path", required=False, type=click.Path(),
              help='path to neuroh5 file containing target rate maps used for rate optimization')
@click.option("--target-features-namespace", type=str, required=False, default='Input Spikes',
              help='namespace containing target rate maps used for rate optimization')
@click.option("--optimize-file-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default='results')
@click.option("--optimize-file-name", type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option("--nprocs-per-worker", type=int, default=1)
@click.option("--n-iter", type=int, default=1)
@click.option("--n-initial", type=int, default=3)
@click.option("--population-size", type=int, default=100)
@click.option("--num-generations", type=int, default=200)
@click.option("--collective-mode", type=str, default='gather')
@click.option("--verbose", '-v', is_flag=True)
def main(config_path, target_features_path, target_features_namespace, optimize_file_dir, optimize_file_name, nprocs_per_worker, n_iter, n_initial, population_size, num_generations, collective_mode, verbose):

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

    run_ts = datetime.datetime.today().strftime('%Y%m%d_%H%M')

    if optimize_file_name is None:
        optimize_file_name=f"dmosopt.optimize_network_{run_ts}.h5"
    operational_config = read_from_yaml(config_path)
    operational_config['run_ts'] = run_ts
    if target_features_path is not None:
        operational_config['target_features_path'] = target_features_path
    if target_features_namespace is not None:
        operational_config['target_features_namespace'] = target_features_namespace

    network_config.update(operational_config.get('kwargs', {}))
    env = Env(**network_config)

    objective_names = operational_config['objective_names']
    param_config_name = operational_config['param_config_name']
    target_populations = operational_config['target_populations']
    opt_param_config = optimization_params(env.netclamp_config.optimize_parameters, target_populations, param_config_name)

    opt_targets = opt_param_config.opt_targets
    param_names = opt_param_config.param_names
    param_tuples = opt_param_config.param_tuples
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
    
    nworkers = env.comm.size-1
    resample_fraction = float(nworkers) / float(population_size)
    if resample_fraction > 1.0:
        resample_fraction = 1.0
    
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
                      'n_initial': n_initial,
                      'n_iter': n_iter,
                      'population_size': population_size,
                      'num_generations': num_generations,
                      'resample_fraction': resample_fraction,
                      'file_path': f'{optimize_file_dir}/{optimize_file_name}',
                      'save': True,
                      'save_eval': 5
                      }
    
    #dmosopt_params['broker_fun_name'] = 'dmosopt_broker_init'
    #dmosopt_params['broker_module_name'] = 'dentate.optimize_network'

    best = dmosopt.run(dmosopt_params, spawn_workers=True, sequential_spawn=False,
                       nprocs_per_worker=nprocs_per_worker,
                       collective_mode=collective_mode,
                       verbose=True, worker_debug=True)


def init_network_objfun(operational_config, opt_targets, param_names, param_tuples, worker, **kwargs):

    param_tuples = [ syn_param_from_dict(param_tuple) for param_tuple in param_tuples ]

    objective_names = operational_config['objective_names']
    target_populations = operational_config['target_populations']
    target_features_path = operational_config['target_features_path']
    target_features_namespace = operational_config['target_features_namespace']
    kwargs['results_file_id'] = 'DG_optimize_network_%d_%s' % \
                                (worker.worker_id, operational_config['run_ts'])

    logger = utils.get_script_logger(os.path.basename(__file__))
    env = init_network(comm=MPI.COMM_WORLD, kwargs=kwargs)
    gc.collect()

    t_start = 50.
    t_stop = env.tstop
    time_range = (t_start, t_stop)

    target_trj_rate_map_dict = {}
    target_features_arena = env.arena_id
    target_features_trajectory = env.trajectory_id
    for pop_name in target_populations:
        if ('%s target rate dist residual' % pop_name) not in objective_names:
            continue
        my_cell_index_set = set(env.biophys_cells[pop_name].keys())
        trj_rate_maps = {}
        trj_rate_maps = rate_maps_from_features(env, pop_name, target_features_path, 
                                                target_features_namespace,
                                                cell_index_set=list(my_cell_index_set),
                                                time_range=time_range)
        target_trj_rate_map_dict[pop_name] = trj_rate_maps

    def from_param_dict(params_dict):
        result = []
        for param_name, param_tuple in zip(param_names, param_tuples):
            result.append((param_tuple, params_dict[param_name]))
        return result

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

    env.tstop = t_stop
    network.run(env, output=False, shutdown=False)

    return network_features(env, target_trj_rate_map_dict, t_start, t_stop, target_populations)



def compute_objectives(features, operational_config, opt_targets):

    all_features = {}
    
    target_populations = operational_config['target_populations']
    for pop_name in target_populations:
        
        pop_features_dicts = [ features_dict[0][pop_name] for features_dict in features ]

        sum_mean_rate = 0.
        sum_target_rate_dist_residual = 0.
        n_total = 0
        n_active = 0
        n_target_rate_map = 0
        for pop_feature_dict in pop_features_dicts:

            n_active_local = pop_feature_dict['n_active']
            n_total_local = pop_feature_dict['n_total']
            n_target_rate_map_local = pop_feature_dict['n_target_rate_map']
            sum_mean_rate_local = pop_feature_dict['sum_mean_rate']
            sum_target_rate_dist_residual_local = pop_feature_dict['sum_target_rate_dist_residual']

            n_total += n_total_local
            n_active += n_active_local
            n_target_rate_map += n_target_rate_map_local
            sum_mean_rate += sum_mean_rate_local

            if sum_target_rate_dist_residual_local is not None:
                sum_target_rate_dist_residual += sum_target_rate_dist_residual_local

        if n_active > 0:
            mean_rate = sum_mean_rate / n_active
        else:
            mean_rate = 0.

        if n_total > 0:
            fraction_active = n_active / n_total
        else:
            fraction_active = 0.

        mean_target_rate_dist_residual = None
        if n_target_rate_map > 0:
            mean_target_rate_dist_residual = sum_target_rate_dist_residual / n_target_rate_map

        logger.info(f'population {pop_name}: n_active = {n_active} n_total = {n_total} mean rate = {mean_rate}')
        logger.info(f'population {pop_name}: n_target_rate_map = {n_target_rate_map} sum_target_rate_dist_residual = {sum_target_rate_dist_residual}')

        all_features['%s fraction active' % pop_name] = fraction_active
        all_features['%s firing rate' % pop_name] = mean_rate
        if mean_target_rate_dist_residual is not None:
            all_features['%s target rate dist residual' % pop_name] = mean_target_rate_dist_residual

    objective_names = operational_config['objective_names']
    target_vals = opt_targets
    target_ranges = opt_targets
    objectives = []
    for key in objective_names:
        feature_val = all_features[key]
        if key in target_vals:
            objective = (feature_val - target_vals[key]) ** 2.
            logger.info(f'objective {key}: {objective} target: {target_vals[key]} feature: {feature_val}')
        else:
            objective = feature_val ** 2.
            logger.info(f'objective {key}: {objective} feature: {feature_val}')
        objectives.append(objective)

    result = np.asarray(objectives)

    return {0: result}


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv) + 1):])