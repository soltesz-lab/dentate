"""
Dentate Gyrus model initialization script
"""
__author__ = 'Ivan Raikov, Aaron D. Milstein, Grace Ng'
import sys, click, os
from mpi4py import MPI
import numpy as np
import dentate.network as network
from dentate.env import Env
from nested.utils import Context


context = Context()


@click.command()
@click.option("--config-file", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='../config/Small_Scale_Control_log_normal_weights.yaml')
@click.option("--template-paths", type=str, default='../../dgc/Mateos-Aparicio2014:../templates')
@click.option("--hoc-lib-path", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='..')
@click.option("--dataset-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='../datasets')  # '/mnt/s')
@click.option("--tstop", type=int, default=1)
@click.option("--v-init", type=float, default=-75.0)
@click.option("--max-walltime-hours", type=float, default=0.1)
@click.option('--verbose', '-v', is_flag=True)
@click.option('--run-test', is_flag=True)
def main(config_file, template_paths, hoc_lib_path, dataset_prefix, tstop, v_init, max_walltime_hours, verbose,
         run_test):
    """

    :param config_file: str; model configuration file
    :param template_paths: str; colon-separated list of paths to directories containing hoc cell templates
    :param hoc_lib_path: str; path to directory containing required hoc libraries
    :param dataset_prefix: str; path to directory containing required neuroh5 data files
    :param tstop: int (ms)
    :param v_init: float (mV)
    :param max_walltime_hours: int (hrs)
    :param verbose: bool; print verbose diagnostic messages while constructing the network
    :param run_test: bool; run sim for duration tstop, do not save any output
    """
    comm = MPI.COMM_WORLD
    np.seterr(all='raise')
    env = Env(comm, config_file, template_paths, hoc_lib_path, dataset_prefix, tstop=tstop, verbose=verbose)
    context.update(locals())
    network.init(env)
    if run_test:
        network.run(env, output=False)


if __name__ == '__main__':
    main(args=sys.argv[(sys.argv.index(os.path.basename(__file__)) + 1):])
