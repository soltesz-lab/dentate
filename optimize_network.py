#!/usr/bin/env python
"""
Dentate Gyrus model optimization script for optimization with dmosopt
"""

import os, sys, logging, datetime
from functools import partial
import click
import numpy as np
from neuroh5.io import scatter_read_cell_attribute_selection, read_cell_attribute_info
from mpi4py import MPI
from collections import defaultdict, namedtuple
import dentate
from dentate import network, network_clamp, synapses, spikedata, stimulus, utils
from dentate.env import Env
from dentate.utils import Context, read_from_yaml, list_find, viewitems
import dmosopt

context = Context()

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
                       'param_name',
                       'param_range'])

def syn_param_from_dict(d):
    SynParam(*[d[key] for key in SynParam._fields])
             

OptConfig = namedtuple("OptConfig",
                       ['param_bounds', 
                        'param_names', 
                        'param_initial_dict', 
                        'param_tuples', 
                        'opt_targets'])


@click.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.option("--config-path", type=str, help='optimization configuration file name',
              default='./config/DG_optimize_network.yaml')
@click.option("--target-rate-map-path", required=False, type=click.Path(),
              help='path to neuroh5 file containing target rate maps used for rate optimization')
@click.option("--target-rate-map-namespace", type=str, required=False, default='Input Spikes',
              help='namespace containing target rate maps used for rate optimization')
@click.option("--results-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default='results')
@click.option("--nprocs-per-worker", type=int, default=1)
@click.option("--verbose", is_flag=True)
def main(config_path, target_rate_map_path, target_rate_map_namespace, results_dir, nprocs_per_worker, verbose):

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

    param_config_name = operational_config['param_config_name']
    target_populations = operational_config['target_populations']
    network_optimization_config = make_optimization_config(env, target_populations, param_config_name)

    opt_targets = network_optimization_config.opt_targets
    param_names = network_optimization_config.param_names
    param_tuples = network_optimization_config.param_tuples
    hyperprm_space = { param_pattern: [param_tuple.param_range[0], param_tuple.param_range[1]]
                       for param_pattern, param_tuple in 
                           zip(param_names, param_tuples) }
    print(f"hyperprm_space: {hyperprm_space}")

    init_objfun = 'init_network_objfun'
    init_params = { 'operational_config': operational_config,
                    'opt_targets': opt_targets,
                    'param_tuples': [ param_tuple._asdict() for param_tuple in param_tuples ],
                    }
    init_params.update(network_config.items())
    

    # Create an optimizer
    dmosopt_params = {'opt_id': 'dmosopt_optimize_network',
                      'obj_fun_init_name': init_objfun, 
                      'obj_fun_init_module': 'dentate.optimize_network',
                      'obj_fun_init_args': init_params,
                      'problem_parameters': {},
                      'space': hyperprm_space,
                      'n_objectives': 3,
                      'n_initial': 10,
                      'n_iter': 10,
                      'file_path': f'{results_dir}/dmosopt.optimize_network.h5',
                      'save': True

                      }
    
    best = dmosopt.run(dmosopt_params, spawn_workers=True, 
                       nprocs_per_worker=nprocs_per_worker, 
                       verbose=True)
    

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
                    "optimization_params: population %s does not have optimization configuration" % pop_name)
            for target_name, target_val in viewitems(opt_params['Targets']):
                opt_targets['%s %s' % (pop_name, target_name)] = target_val
            keyfun = lambda kv: str(kv[0])
            print(f'opt_targets: {opt_targets}')
            print(f'param_ranges: {param_ranges}')
            for source, source_dict in sorted(viewitems(param_ranges), key=keyfun):
                for sec_type, sec_type_dict in sorted(viewitems(source_dict), key=keyfun):
                    for syn_name, syn_mech_dict in sorted(viewitems(sec_type_dict), key=keyfun):
                        for param_fst, param_rst in sorted(viewitems(syn_mech_dict), key=keyfun):
                            print(f'param_fst: {param_fst}')
                            print(f'param_rst: {param_rst}')
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

    print(f'param_tuples = {param_tuples}')
    return OptConfig(param_bounds=param_bounds, 
                     param_names=param_names, 
                     param_initial_dict=param_initial_dict, 
                     param_tuples=param_tuples, 
                     opt_targets=opt_targets)


def update_network_params(env, from_param_vector, param_values):

    param_tuples = from_param_vector(param_values)
    for param_tuple in param_tuples:

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
        rate_map = input_cell_config.get_rate_map(x=x, y=y)
        input_rate_map_dict[gid] = rate_map
            
    return input_rate_map_dict



def init_network_objfun(operational_config, opt_targets, param_tuples, worker, **kwargs):

    param_tuples = [ syn_param_from_dict(param_tuple) for param_tuple in param_tuples ]

    target_populations = operational_config['target_populations']
    target_rate_map_path = operational_config['target_rate_map_path']
    target_rate_map_namespace = operational_config['target_rate_map_namespace']

    kwargs['results_file_id'] = 'DG_optimize_network_%s_%s' % \
                                (worker.worker_id, datetime.datetime.today().strftime('%Y%m%d_%H%M'))

    logger = utils.get_script_logger(os.path.basename(__file__))
    env = init_network(comm=MPI.COMM_WORLD, kwargs=kwargs)

    t_start = 50.
    t_stop = env.tstop
    time_range = (t_start, t_stop)

    target_trj_rate_map_dict = {}
    target_rate_map_arena = env.arena_id
    target_rate_map_trajectory = env.trajectory_id
    for pop_name in target_populations:
        my_cell_index_set = set(env.biophys_cells[pop_name].keys())
        trj_rate_maps = rate_maps_from_features(env, pop_name, target_rate_map_path, 
                                                target_rate_map_namespace,
                                                cell_index_set=list(my_cell_index_set),
                                                time_range=time_range)
        if len(trj_rate_maps) > 0:
            target_trj_rate_map_dict[pop_name] = trj_rate_maps

    target_rate_vector_dict = { gid: trj_rate_maps[gid] for gid in trj_rate_maps }
    for gid, target_rate_vector in viewitems(target_rate_vector_dict):
        idxs = np.where(np.isclose(target_rate_vector, 0., atol=1e-4, rtol=1e-4))[0]
        target_rate_vector[idxs] = 0.


    def from_param_vector(param_values):
        result = []
        assert (len(param_values) == len(param_tuples))
        for i, param_tuple in enumerate(param_tuples):
            result.append((param_tuple.population,
                           param_tuple.source,
                           param_tuple.sec_type,
                           param_tuple.syn_name,
                           param_tuple.param_name,
                           param_values[i]))
        return result

    def to_param_vector(params):
        result = []
        for (source, sec_type, syn_name, param_name, param_value) in params:
            result.append(param_value)
        return result

    return partial(network_objfun, env, operational_config, opt_targets,
                   from_param_vector, to_param_vector, t_start, t_stop, target_populations)

    
def init_network(comm, kwargs):
    np.seterr(all='raise')
    env = Env(comm=comm, **kwargs)
    network.init(env)
    return env


def network_objfun(env, operational_config, opt_targets,
                   from_param_vector, to_param_vector, t_start, t_stop, target_populations, x):

    param_values = from_param_vector(x)

    update_network_params(env, param_values)

    return compute_network_features(env, operational_config, opt_targets,
                                    t_start, t_stop, target_populations)


def compute_network_features(env, operational_config, opt_targets, t_start, t_stop):

    target_populations = operational_config['target_populations']

    features_dict = dict()
 
    temporal_resolution = float(env.stimulus_config['Temporal Resolution'])
    time_bins  = np.arange(t_start, t_stop, temporal_resolution)

    network.run(env, output=False, shutdown=False)

    pop_spike_dict = spikedata.get_env_spike_dict(env, include_artificial=False)

    for pop_name in target_populations:

        n_active_local = 0
        mean_rate_sum_local = 0.
        spike_density_dict = spikedata.spike_density_estimate (pop_name, pop_spike_dict[pop_name], time_bins)
        for gid, dens_dict in utils.viewitems(spike_density_dict):
            mean_rate_sum_local += np.mean(dens_dict['rate'])
            if mean_rate_sum_local > 0.:
                n_active_local += 1
        mean_rate_sum = env.comm.allreduce(mean_rate_sum_local, op=MPI.SUM)
        env.comm.barrier()
        
        n_local = len(env.cells[pop_name]) - len(env.artificial_cells[pop_name])
        n_total = env.comm.allreduce(n_local, op=MPI.SUM)
        n_active = env.comm.allreduce(n_active_local, op=MPI.SUM)
        env.comm.barrier()
        
        if n_active > 0:
            mean_rate = mean_rate_sum / n_active
        else:
            mean_rate = 0.

        if n_total > 0:
            fraction_active = n_active / n_total
        else:
            fraction_active = 0.

        mean_target_rate_dist_residual = None
        if pop_name in target_trj_rate_map_dict:
            mean_target_rate_dist_residual = 0.
            if n_active > 0:
                target_trj_rate_map_dict = target_trj_rate_map_dict[pop_name]
                target_rate_dist_residuals = []
                for gid in spike_density_dict:
                    target_trj_rate_map = target_trj_rate_map_dict[gid]
                    residual = np.sum(target_trj_rate_map - spike_density_dict[gid]['rate'])
                    target_rate_dist_residuals.append(residual)
                residual_sum_local = np.sum(target_rate_dist_residuals)
                residual_sum = env.comm.allreduce(residual_sum_local, op=MPI.SUM)
                mean_target_rate_dist_residual = residual_sum / n_active
            env.comm.barrier()
                
        rank = int(env.pc.id())
        if env.comm.rank == 0:
            logger.info('population %s: n_active = %d n_total = %d mean rate = %s' %
                        (pop_name, n_active, n_total, str(mean_rate)))

        features_dict['%s firing rate' % pop_name] = mean_rate
        features_dict['%s fraction active' % pop_name] = fraction_active
        if mean_target_rate_dist_residual is not None:
            features_dict['%s target rate dist residual' % pop_name] = mean_target_rate_dist_residual
        
    return compute_objectives(env, operational_config, opt_targets, features_dict)


def compute_objectives(env, operational_config, opt_targets, features):

    objectives = []
    if env.comm.rank == 0:
        env.logger.info('features: %s' % str(features))
    objective_names = operational_config['objective_names']
    target_val = opt_targets
    target_range = opt_targets
    for key in objective_names:
        if key in target_val:
            objective = ((features[key] - target_val[key]) / target_range[key]) ** 2.
        else:
            objective = features[key] ** 2.
        objectives.append(objectives)
            

    return np.asarray(objectives)


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv) + 1):])
