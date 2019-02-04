#!/usr/bin/env python
"""
Dentate Gyrus model simulation script for optimization with nested.optimize
"""
__author__ = 'See AUTHORS.md'
import sys, click, os
from mpi4py import MPI
import numpy as np
import dentate.network as network
from dentate.biophysics_utils import *
from nested.optimize_utils import *


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
              default='../config/DG_test_network_subworlds_config.yaml')
@click.option("--output-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default='../data')
@click.option("--export", is_flag=True)
@click.option("--export-file-path", type=str, default=None)
@click.option("--label", type=str, default=None)
@click.option("--verbose", is_flag=True)
def main(optimize_config_file_path, output_dir, export, export_file_path, label, verbose):
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
    config_optimize_interactive(__file__, config_file_path=optimize_config_file_path, output_dir=output_dir,
                                export=export, export_file_path=export_file_path, label=label, disp=verbose)


def config_worker():
    """

    """
    if 'results_id' not in context():
        context.results_id = 'DG_test_network_subworlds_%s_%s' % \
                             (context.interface.worker_id, datetime.datetime.today().strftime('%Y%m%d_%H%M'))
    if 'env' not in context():
        init_network()


def init_network():
    """

    """
    np.seterr(all='raise')
    context.env = Env(comm=context.comm, results_id=context.results_id, **context.kwargs)
    network.init(context.env, context.cleanup)


def update_network(x, context=None):
    """

    :param x: array
    :param context: :class:'Context'
    """
    if context is None:
        raise RuntimeError('update_network: missing required Context object')
    x_dict = param_array_to_dict(x, context.param_names)
    for postsyn_name in ['GC']:
        first_gid = True
        for gid in context.env.biophys_cells[postsyn_name]:
            if context.comm.rank == 0 and first_gid and context.verbose:
                verbose = True
                first_gid = False
            else:
                verbose = False
            cell = context.env.biophys_cells[postsyn_name][gid]
            for presyn_name, param_name, syn_name, syn_param_name in \
                    zip(['BC'], ['BC_GC.GABA_A.g_unit'], ['GABA_A'], ['g_unit']):
                sec_types = \
                    context.env.modelConfig['Connection Generator']['Synapses'][postsyn_name][presyn_name]['sections']
                layers = \
                    context.env.modelConfig['Connection Generator']['Synapses'][postsyn_name][presyn_name]['layers']
                syn_types = \
                    [context.env.modelConfig['Connection Generator']['Synapses'][postsyn_name][presyn_name]['type']]
                for sec_type in sec_types:
                    modify_syn_param(cell, context.env, sec_type, syn_name=syn_name, param_name=syn_param_name,
                                     filters={'syn_types': syn_types, 'sources': [presyn_name], 'layers': layers},
                                     value=x_dict[param_name], update_targets=True, verbose=verbose)


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


def get_objectives_network_walltime(features, export=False):
    """

    :param features: dict
    :param export: bool
    :return: tuple of dict
    """
    objectives = dict()
    for feature_key in context.feature_names:
        objectives[feature_key] = ((features[feature_key] - context.target_val[feature_key]) /
                                   context.target_range[feature_key]) ** 2.

    return features, objectives


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv) + 1):])
