#!/usr/bin/env python
"""
Dentate Gyrus model optimization script for optimization with nested.optimize
"""
__author__ = 'See AUTHORS.md'
import os, sys, logging
import click
import numpy as np
from mpi4py import MPI
import dentate
from dentate import network, synapses, spikedata, utils
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
def main(optimize_config_file_path, output_dir, export, export_file_path, label, bin_size, procs_per_worker, verbose):
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
    if 'results_id' not in context():
        context.results_id = 'DG_optimize_network_subworlds_%s_%s' % \
                             (context.interface.worker_id, datetime.datetime.today().strftime('%Y%m%d_%H%M'))
    if 'env' not in context():
        try:
            init_network()
        except Exception as err:
            context.logger.exception(err)
            raise err
        context.bin_size = 5.0
    
    pop_name = context.target_population

    if (pop_name in context.env.netclamp_config.optimize_parameters):
        opt_params = context.env.netclamp_config.optimize_parameters[pop_name]
        param_ranges = opt_params['Parameter ranges']
        opt_targets = opt_params['Targets']
    else:
        raise RuntimeError(
            "optimize_network_subworlds: population %s does not have optimization configuration" % pop_name)

    param_bounds = {}
    param_names = []
    param_initial_dict = {}
    param_range_tuples = []
    for source, source_dict in sorted(viewitems(param_ranges), key=lambda k_v3: k_v3[0]):
        for sec_type, sec_type_dict in sorted(viewitems(source_dict), key=lambda k_v2: k_v2[0]):
            for syn_name, syn_mech_dict in sorted(viewitems(sec_type_dict), key=lambda k_v1: k_v1[0]):
                for param_name, param_range in sorted(viewitems(syn_mech_dict), key=lambda k_v: k_v[0]):
                    param_range_tuples.append((source, sec_type, syn_name, param_name, param_range))
                    param_key = '%s_%s_%s_%s' % (source, sec_type, syn_name, param_name)
                    param_initial_value = (param_range[1] - param_range[0]) / 2.0
                    param_initial_dict[param_key] = param_initial_value
                    param_bounds[param_key] = param_range
                    param_names.append(param_key)
                    
    def from_param_vector(params):
        result = []
        assert (len(params) == len(param_range_tuples))
        for i, (source, sec_type, syn_name, param_name, param_range) in enumerate(param_range_tuples):
            result.append((source, sec_type, syn_name, param_name, params[i]))
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
    if 'results_id' not in context():
        context.results_id = 'DG_optimize_network_subworlds_%s_%s' % \
                             (context.interface.worker_id, datetime.datetime.today().strftime('%Y%m%d_%H%M'))
    if 'env' not in context():
        try:
            init_env()
        except Exception as err:
            context.logger.exception(err)
            raise err
        context.bin_size = 5.0
    
    if (pop_name in context.env.netclamp_config.optimize_parameters):
        opt_params = context.env.netclamp_config.optimize_parameters[pop_name]
        param_ranges = opt_params['Parameter ranges']
        opt_targets = opt_params['Targets']
    else:
        raise RuntimeError(
            "optimize_network_subworlds: population %s does not have optimization configuration" % pop_name)

    param_bounds = {}
    param_names = []
    param_initial_dict = {}
    param_range_tuples = []
    for source, source_dict in sorted(viewitems(param_ranges), key=lambda k_v3: k_v3[0]):
        for sec_type, sec_type_dict in sorted(viewitems(source_dict), key=lambda k_v2: k_v2[0]):
            for syn_name, syn_mech_dict in sorted(viewitems(sec_type_dict), key=lambda k_v1: k_v1[0]):
                for param_name, param_range in sorted(viewitems(syn_mech_dict), key=lambda k_v: k_v[0]):
                    param_range_tuples.append((source, sec_type, syn_name, param_name, param_range))
                    param_key = '%s_%s_%s_%s' % (source, sec_type, syn_name, param_name)
                    param_initial_value = (param_range[1] - param_range[0]) / 2.0
                    param_initial_dict[param_key] = param_initial_value
                    param_bounds[param_key] = param_range
                    param_names.append(param_key)
                    
    def from_param_vector(params):
        result = []
        assert (len(params) == len(param_range_tuples))
        for i, (source, sec_type, syn_name, param_name, param_range) in enumerate(param_range_tuples):
            result.append((source, sec_type, syn_name, param_name, params[i]))
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
        
def init_env():
    """

    """
    context.env = Env(comm=context.comm, results_id=context.results_id, **context.kwargs)
    
def init_network():
    """

    """
    np.seterr(all='raise')
    context.env = Env(comm=context.comm, results_id=context.results_id, **context.kwargs)
    network.init(context.env)


def update_network(x, context=None):
    """

    :param x: array
    :param context: :class:'Context'
    """
    if context is None:
        raise RuntimeError('update_network: missing required Context object')

    pop_name = context.target_population
    param_values = context.from_param_vector(x)

    biophys_cell_dict = context.env.biophys_cells[pop_name]
    for gid, biophys_cell in viewitems(biophys_cell_dict):
        for source, sec_type, syn_name, param_name, param_value in param_values:
            synapses.modify_syn_param(biophys_cell, context.env, sec_type, syn_name,
                                      param_name=param_name, value=param_value,
                                      filters={'sources': [source]},
                                      origin='soma', update_targets=True)
                    
    


def compute_features_network_walltime(x, export=False):
    """

    :param x: array
    :param export: bool
    :return: dict
    """
    results = dict()
    start_time = time.time()
    update_source_contexts(x, context)
    results['modify_network_time'] = time.time() - start_time
    start_time = time.time()
    context.env.results_id = '%s_%s' % \
                             (context.interface.worker_id, datetime.datetime.today().strftime('%Y%m%d_%H%M%S'))
    network.run(context.env, output=context.output_results, shutdown=False)
    results['sim_network_time'] = time.time() - start_time

    return results


def compute_features_firing_rate(x, export=False):
    """

    :param x: array
    :param export: bool
    :return: dict
    """
    results = dict()
    update_source_contexts(x, context)
    context.env.results_id = '%s_%s' % \
                             (context.interface.worker_id, datetime.datetime.today().strftime('%Y%m%d_%H%M%S'))

    network.run(context.env, output=context.output_results, shutdown=False)

    pop_spike_dict = spikedata.get_env_spike_dict(context.env)

    t_start = 0.
    t_stop = context.env.tstop
    
    time_bins  = np.arange(t_start, t_stop, context.bin_size)

    pop_name = context.target_population

    mean_rate_sum = 0.
    spike_density_dict = spikedata.spike_density_estimate (pop_name, pop_spike_dict[pop_name], time_bins)
    for gid, dens_dict in utils.viewitems(spike_density_dict):
        mean_rate_sum += np.mean(dens_dict['rate'])

    n = len(spike_density_dict)
    if n > 0:
        mean_rate = mean_rate_sum / n
    else:
        mean_rate = 0.

    results['firing rate'] = mean_rate

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
