#!/usr/bin/env python
"""
Dentate Gyrus model simulation script for interactive debugging.
"""
__author__ = 'See AUTHORS.md'
import sys, click, os
from mpi4py import MPI
import numpy as np
import dentate.network as network
from dentate.env import Env
from dentate.utils import list_find
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
    config_interactive(context, __file__, config_file_path=optimize_config_file_path, output_dir=output_dir,
                       export=export, export_file_path=export_file_path, label=label, disp=verbose)


def config_worker():
    """

    """
    if 'results_id' not in context():
        context.results_id = 'DG_test_network_subworlds_%s_%s' % \
                             (datetime.datetime.today().strftime('%Y%m%d_%H%M'), context.interface.worker_id)
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
    print x_dict


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
    network.run(context.env, output=context.output_results)
    results['sim_network_time'] = time.time() - start_time

    return results


def get_objectives_network_walltime(features):
    """

    :param features: dict
    :return: tuple of dict
    """
    objectives = dict()
    for feature_key in context.feature_names:
        objectives[feature_key] = ((features[feature_key] - context.target_val[feature_key]) /
                                   context.target_range[feature_key]) ** 2.

    return features, objectives


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv) + 1):])
