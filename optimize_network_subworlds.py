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
from dentate.optimization import (SynParam, OptConfig, syn_param_from_dict, optimization_params, 
                                  update_network_params, network_features)
from dentate.stimulus import rate_maps_from_features
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


context = Context()


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True, ))
@click.option("--optimize-config-file-path", type=str, help='optimization configuration file name',
              default='../config/troubleshoot_DG_optimize_network_subworlds_config.yaml')
@click.option("--output-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default='../data')
@click.option("--export", is_flag=True)
@click.option("--export-file-path", type=str, default=None)
@click.option("--label", type=str, default=None)
@click.option("--verbose", is_flag=True)
@click.option("--debug", is_flag=True)
@click.pass_context
def main(cli, optimize_config_file_path, output_dir, export, export_file_path, label, verbose, debug):
    """

    :param optimize_config_file_path: str
    :param output_dir: str
    :param export: bool
    :param export_file_path: str
    :param label: str
    :param verbose: bool
    :param debug: bool
    """
    # requires a global variable context: :class:'Context'
    context.update(locals())
    kwargs = get_unknown_click_arg_dict(cli.args)
    context.disp = verbose > 0

    context.interface = get_parallel_interface(source_file=__file__, source_package=__package__, **kwargs)
    context.interface.start(disp=context.disp)
    context.interface.ensure_controller()
    config_optimize_interactive(__file__, config_file_path=optimize_config_file_path, output_dir=output_dir,
                                export=export, export_file_path=export_file_path, label=label,
                                disp=context.disp, interface=context.interface, verbose=verbose,
                                debug=debug, **kwargs)

    if not context.debug:
        model_id = 0
        if 'model_key' in context() and context.model_key is not None:
            model_label = context.model_key
        else:
            model_label = 'test'

        features = dict()

        # Stage 0:
        sequences = [[context.x0_array], [model_id], [context.export]]
        primitives = context.interface.map(compute_network_features, *sequences)
        this_features = {key: value for feature_dict in primitives for key, value in viewitems(feature_dict)}
        features.update(this_features)

        features, objectives = context.interface.execute(get_objectives, features, model_id, context.export)

        sys.stdout.flush()
        print('model_id: %i; model_labels: %s' % (model_id, model_label))
        print('params:')
        pprint.pprint(context.x0_dict)
        print('features:')
        pprint.pprint(features)
        print('objectives:')
        pprint.pprint(objectives)
        sys.stdout.flush()
        time.sleep(.1)

    context.interface.stop()





def from_param_vector(param_values, param_tuples):
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


def config_worker():
    """

    """
    if 'debug' not in context():
        context.debug = False

    if context.debug:
        if context.comm.rank == 1:
            print('# of parameters: %i' % len(context.param_names))
            print('param_names: ', context.param_names)
            print('target_val: ', context.target_val)
            print('target_range: ', context.target_range)
            print('param_tuples: ', context.param_tuples)
            sys.stdout.flush()

    utils.config_logging(context.verbose)
    context.logger = utils.get_script_logger(os.path.basename(__file__))
    # TODO: Do you want this to be identical on all ranks in a subworld? You can use context.comm.bcast
    if 'results_file_id' not in context():
        context.results_file_id = 'DG_optimize_network_subworlds_%s_%s' % \
                             (context.interface.worker_id, datetime.datetime.today().strftime('%Y%m%d_%H%M'))

    # 'env' might be in context on controller, but it needs to be re-built when the controller is in a worker subworld
    try:
        if context.debug:
            print('debug: config_worker; local_comm.rank: %i/%i; global_comm.rank: %i/%i' %
                  (context.comm.rank, context.comm.size, context.global_comm.rank, context.global_comm.size))
            sys.stdout.flush()
        init_network()
    except Exception as err:
        context.logger.exception(err)
        raise err

    if 't_start' not in context():
        context.t_start = 50.
    else:
        context.t_start = float(context.t_start)
    if 't_stop' not in context():
        context.t_stop = context.env.tstop
    else:
        context.t_stop = float(context.t_stop)
    time_range = (context.t_start, context.t_stop)

    try:
        if context.debug:
            if context.global_comm.rank == 0:
                print('t_start: %.1f, t_stop: %.1f' % (context.t_start, context.t_stop))
    except Exception as err:
        context.logger.exception(err)
        raise err

    context.target_trj_rate_map_dict = {}
    target_rate_map_path = context.target_rate_map_path
    target_rate_map_namespace = context.target_rate_map_namespace
    target_rate_map_arena = context.env.arena_id
    target_rate_map_trajectory = context.env.trajectory_id
    for pop_name in context.target_populations:
        my_cell_index_set = set(context.env.biophys_cells[pop_name].keys())
        trj_rate_maps = rate_maps_from_features(context.env, pop_name,
                                                cell_index_set=list(my_cell_index_set),
                                                input_features_path=target_rate_map_path, 
                                                input_features_namespace=target_rate_map_namespace,
                                                time_range=time_range)
        if len(trj_rate_maps) > 0:
            context.target_trj_rate_map_dict[pop_name] = trj_rate_maps


def config_controller():
    """

    """
    utils.config_logging(context.verbose)
    context.logger = utils.get_script_logger(os.path.basename(__file__))

    try:
        context.env = Env(comm=context.controller_comm, **context.kwargs)
    except Exception as err:
        context.logger.exception(err)
        raise err

    opt_param_config = optimization_params(context.env.netclamp_config.optimize_parameters, context.target_populations, context.param_config_name)
    param_bounds = opt_param_config.param_bounds
    param_names = opt_param_config.param_bounds
    param_initial_dict = opt_param_config.param_initial_dict
    param_tuples = opt_param_config.param_tuples
    opt_targets = opt_param_config.opt_targets

    context.param_names = param_names
    context.bounds = [param_bounds[key] for key in param_names]
    context.x0 = param_initial_dict
    context.target_val = opt_targets
    context.target_range = opt_targets
    context.param_tuples = param_tuples
    # These kwargs will be sent from the controller to each worker context
    context.kwargs['param_tuples'] = param_tuples


def init_network():
    """

    """
    np.seterr(all='raise')
    context.comm.barrier()
    context.env = Env(comm=context.comm, results_file_id=context.results_file_id, **context.kwargs)
    network.init(context.env)
    if context.debug:
        raise RuntimeError('config_worker: after network.init')
    context.comm.barrier()


def update_network(x, context=None):
    """

    :param x: array
    :param context: :class:'Context'
    """
    if context is None:
        raise RuntimeError('update_network: missing required Context object')

    param_tuple_values = from_param_vector(x, context.param_tuples)

    update_network_params(context.env, param_tuple_values)


def compute_network_features(x, model_id=None, export=False):
    """

    :param x: array
    :param export: bool
    :return: dict
    """
    results = dict()
    update_source_contexts(x, context)
    # TODO: Do you want this to be identical on all ranks in a subworld? You can use context.comm.bcast
    context.env.results_file_id = '%s_%s' % \
                             (context.interface.worker_id, datetime.datetime.today().strftime('%Y%m%d_%H%M%S'))

    temporal_resolution = float(context.env.stimulus_config['Temporal Resolution'])
    time_bins  = np.arange(context.t_start, context.t_stop, temporal_resolution)

    context.env.tstop = context.t_stop
    network.run(context.env, output=context.output_results, shutdown=False)

    local_network_features = network_features(context.env, context.target_trj_rate_map_dict, 
                                              context.t_start, context.t_stop, context.target_populations)
    results = {}
    for pop_name in context.target_populations:
        
        n_active_local = local_network_features[pop_name]['n_active']
        n_total_local = local_network_features[pop_name]['n_total']
        sum_mean_rate_local = local_network_features[pop_name]['sum_mean_rate']
        n_target_rate_map_local = local_network_features[pop_name]['n_target_rate_map']

        n_total = context.env.comm.allreduce(n_total_local, op=MPI.SUM)
        n_active = context.env.comm.allreduce(n_active_local, op=MPI.SUM)
        mean_rate_sum = context.env.comm.allreduce(sum_mean_rate_local, op=MPI.SUM)
        n_target_rate_map = context.env.comm.allreduce(n_target_rate_map_local, op=MPI.SUM)

        has_target_rate_map = n_target_rate_map > 0
        if has_target_rate_map:
            sum_target_rate_dist_residual_local = local_network_features[pop_name]['sum_target_rate_dist_residual']
            sum_target_rate_dist_residual = context.env.comm.allreduce(sum_target_rate_dist_residual_local, op=MPI.SUM)
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
        if has_target_rate_map:
            mean_target_rate_dist_residual = sum_target_rate_dist_residual / n_target_rate_map

        rank = int(context.env.pc.id())
        if context.env.comm.rank == 0:
            context.logger.info('population %s: n_active = %d n_total = %d mean rate = %s' %
                                    (pop_name, n_active, n_total, str(mean_rate)))

        results['%s fraction active' % pop_name] = fraction_active
        results['%s firing rate' % pop_name] = mean_rate
        if mean_target_rate_dist_residual is not None:
            results['%s target rate dist residual' % pop_name] = mean_target_rate_dist_residual

    return results
        

def get_objectives(features, model_id=None, export=False):
    """

    :param features: dict
    :param export: bool
    :return: tuple of dict
    """
    if context.env.comm.rank == 0:
        context.logger.info('features: %s' % str(features))

    objectives = dict()
    for key in context.objective_names:
        if key not in features:
            if context.env.comm.rank == 0:
                context.logger.info('get_objectives: model_id: %i failed - missing feature: %s' % (model_id, key))
            return dict(), dict()
        if key in context.target_val:
            objectives[key] = ((features[key] - context.target_val[key]) / context.target_range[key]) ** 2.
        else:
            objectives[key] = features[key] ** 2.

    return features, objectives


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv) + 1):])
