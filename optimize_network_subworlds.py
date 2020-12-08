#!/usr/bin/env python
"""
Dentate Gyrus model optimization script for optimization with nested.optimize
"""
__author__ = 'See AUTHORS.md'
import os, sys, logging
import click
import numpy as np
from mpi4py import MPI
from collections import defaultdict, namedtuple
import dentate
from dentate import network, network_clamp, synapses, spikedata, stimulus, utils
from dentate.env import Env
from neuroh5.io import scatter_read_cell_attribute_selection, read_cell_attribute_info
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

SynParam = namedtuple('SynParam',
                      ['population',
                       'source',
                       'sec_type',
                       'syn_name',
                       'param_path',
                       'param_range'])

context = Context()


@click.command()
@click.option("--optimize-config-file-path", type=str, help='optimization configuration file name',
              default='../config/DG_optimize_network_subworlds_config.yaml')
@click.option("--output-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default='../data')
@click.option("--export", is_flag=True)
@click.option("--export-file-path", type=str, default=None)
@click.option("--label", type=str, default=None)
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


def optimization_params(env, pop_names, param_config_name, param_type='synaptic'):
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
            raise RuntimeError("optimization_params: unknown parameter type %s" % param_type)

    return param_bounds, param_names, param_initial_dict, param_tuples, opt_targets


def update_network_params(env, param_tuple_values):

    if context is None:
        raise RuntimeError('update_network: missing required Context object')
    
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
        rate_map = input_cell_config.get_rate_map(x=x, y=y)
        input_rate_map_dict[gid] = rate_map
            
    return input_rate_map_dict



def config_worker():
    """

    """
    utils.config_logging(context.verbose)
    context.logger = utils.get_script_logger(os.path.basename(__file__))
    if 'results_file_id' not in context():
        context.results_file_id = 'DG_optimize_network_subworlds_%s_%s' % \
                             (context.interface.worker_id, datetime.datetime.today().strftime('%Y%m%d_%H%M'))
    if 'env' not in context():
        try:
            context.comm = MPI.COMM_WORLD
            init_network()
        except Exception as err:
            context.logger.exception(err)
            raise err
    if 't_start' not in context():
        context.t_start = 50.
        context.t_stop = context.env.tstop
    time_range = (context.t_start, context.t_stop)

    context.target_trj_rate_map_dict = {}
    target_rate_map_path = context.target_rate_map_path
    target_rate_map_namespace = context.target_rate_map_namespace
    target_rate_map_arena = context.env.arena_id
    target_rate_map_trajectory = context.env.trajectory_id
    for pop_name in context.target_populations:
        my_cell_index_set = set(context.env.biophys_cells[pop_name].keys())
        trj_rate_maps = rate_maps_from_features(context.env, pop_name, target_rate_map_path, 
                                                target_rate_map_namespace,
                                                cell_index_set=list(my_cell_index_set),
                                                time_range=time_range)
        if len(trj_rate_maps) > 0:
            context.target_trj_rate_map_dict[pop_name] = trj_rate_maps

    target_rate_vector_dict = { gid: trj_rate_maps[gid] for gid in trj_rate_maps }
    for gid, target_rate_vector in viewitems(target_rate_vector_dict):
        idxs = np.where(np.isclose(target_rate_vector, 0., atol=1e-4, rtol=1e-4))[0]
        target_rate_vector[idxs] = 0.

    param_bounds, param_names, param_initial_dict, param_tuples, opt_targets = \
      optimization_params(context.env, context.target_populations, context.param_config_name)

    def from_param_vector(param_values):
        result = []
        assert (len(param_values) == len(param_tuples))
        for i, param_tuple in enumerate(param_tuples):
            result.append((param_tuple, param_values[i]))
        return result

    def to_param_vector(params):
        result = []
        for (source, sec_type, syn_name, param_name, param_value) in params:
            result.append(param_value)
        return result

    context.param_names = param_names
    context.bounds = [ param_bounds[key] for key in param_names ]
    context.x0 = param_initial_dict
    context.from_param_vector = from_param_vector
    context.to_param_vector = to_param_vector
    context.target_val = opt_targets
    context.target_range = opt_targets

    
def config_controller():
    """

    """
    utils.config_logging(context.verbose)
    context.logger = utils.get_script_logger(os.path.basename(__file__))
    if 'results_file_id' not in context():
        context.results_file_id = 'DG_optimize_network_subworlds_%s_%s' % \
                             (context.interface.worker_id, datetime.datetime.today().strftime('%Y%m%d_%H%M'))
    if 'env' not in context():
        try:
            context.comm = MPI.COMM_WORLD
            #init_env()
        except Exception as err:
            context.logger.exception(err)
            raise err
        
def init_env():
    """

    """
    context.env = Env(comm=context.comm, results_file_id=context.results_file_id, **context.kwargs)
    
def init_network():
    """

    """
    np.seterr(all='raise')
    context.comm.barrier()
    context.env = Env(comm=context.comm, results_file_id=context.results_file_id, **context.kwargs)
    network.init(context.env)
    context.comm.barrier()

def update_network(x, context=None):
    """

    :param x: array
    :param context: :class:'Context'
    """
    if context is None:
        raise RuntimeError('update_network: missing required Context object')

    param_tuple_values = context.from_param_vector(x)

    update_network_params(context.env, param_tuple_values)



def compute_network_features(x, model_id=None, export=False):
    """

    :param x: array
    :param export: bool
    :return: dict
    """
    results = dict()
    update_source_contexts(x, context)
    context.env.results_file_id = '%s_%s' % \
                             (context.interface.worker_id, datetime.datetime.today().strftime('%Y%m%d_%H%M%S'))

 
    temporal_resolution = float(context.env.stimulus_config['Temporal Resolution'])
    time_bins  = np.arange(context.t_start, context.t_stop, temporal_resolution)

    context.env.tstop = context.t_stop
    network.run(context.env, output=context.output_results, shutdown=False)

    pop_spike_dict = spikedata.get_env_spike_dict(context.env, include_artificial=False)

    for pop_name in context.target_populations:

        n_active_local = 0
        mean_rate_sum_local = 0.
        spike_density_dict = spikedata.spike_density_estimate (pop_name, pop_spike_dict[pop_name], time_bins)
        for gid, dens_dict in utils.viewitems(spike_density_dict):
            mean_rate = np.mean(dens_dict['rate'])
            mean_rate_sum_local += mean_rate
            if mean_rate > 0.:
                n_active_local += 1
        mean_rate_sum = context.env.comm.allreduce(mean_rate_sum_local, op=MPI.SUM)
        context.env.comm.barrier()
        
        n_local = len(context.env.cells[pop_name]) - len(context.env.artificial_cells[pop_name])
        n_total = context.env.comm.allreduce(n_local, op=MPI.SUM)
        n_active = context.env.comm.allreduce(n_active_local, op=MPI.SUM)
        context.env.comm.barrier()
        
        if n_active > 0:
            mean_rate = mean_rate_sum / n_active
        else:
            mean_rate = 0.

        if n_total > 0:
            fraction_active = n_active / n_total
        else:
            fraction_active = 0.

        mean_target_rate_dist_residual = None
        if pop_name in context.target_trj_rate_map_dict:
            mean_target_rate_dist_residual = 0.
            if n_active > 0:
                target_trj_rate_map_dict = context.target_trj_rate_map_dict[pop_name]
                target_rate_dist_residuals = []
                for gid in target_trj_rate_map_dict:
                    target_trj_rate_map = target_trj_rate_map_dict[gid]
                    rate_map_len = len(target_trj_rate_map)
                    if gid in spike_density_dict:
                        residual = np.sum(target_trj_rate_map - spike_density_dict[gid]['rate'][:rate_map_len])
                    else:
                        residual = np.sum(target_trj_rate_map)
                    target_rate_dist_residuals.append(residual)
                residual_sum_local = np.sum(target_rate_dist_residuals)
                residual_sum = context.env.comm.allreduce(residual_sum_local, op=MPI.SUM)
                mean_target_rate_dist_residual = residual_sum / len(target_trj_rate_map_dict)
            context.env.comm.barrier()
                
        rank = int(context.env.pc.id())
        if context.env.comm.rank == 0:
            context.logger.info('population %s: n_active = %d n_total = %d mean rate = %s' %
                                    (pop_name, n_active, n_total, str(mean_rate)))

        results['%s firing rate' % pop_name] = mean_rate
        results['%s fraction active' % pop_name] = fraction_active
        if mean_target_rate_dist_residual is not None:
            results['%s target rate dist residual' % pop_name] = mean_target_rate_dist_residual
        
    return results


def get_objectives(features, model_id=None, export=False):
    """

    :param features: dict
    :param export: bool
    :return: tuple of dict
    """
    objectives = dict()
    if context.env.comm.rank == 0:
        context.logger.info('features: %s' % str(features))
    for key in context.objective_names:
        if key in context.target_val:
            objectives[key] = ((features[key] - context.target_val[key]) / context.target_range[key]) ** 2.
        else:
            objectives[key] = features[key] ** 2.
            

    return features, objectives


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv) + 1):])
