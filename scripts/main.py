"""
Dentate Gyrus model initialization script
"""
__author__ = 'Ivan Raikov, Aaron D. Milstein, Grace Ng'

from mpi4py import MPI
import sys, click, os
import numpy as np
from dentate import network
from dentate.env import Env


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


@click.command()
@click.option("--config-file", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--template-paths", type=str, default='../templates')
@click.option("--hoc-lib-path", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='..')
@click.option("--dataset-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--results-path", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--results-id", type=str, required=False, default='')
@click.option("--node-rank-file", required=False, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--io-size", type=int, default=0)
@click.option("--vrecord-fraction", type=float, default=0.001)
@click.option("--coredat", is_flag=True)
@click.option("--tstop", type=int, default=1)
@click.option("--v-init", type=float, default=-75.0)
@click.option("--stimulus-onset", type=float, default=1.0)
@click.option("--max-walltime-hours", type=float, default=1.0)
@click.option("--results-write-time", type=float, default=360.0)
@click.option("--dt", type=float, default=0.025)
@click.option("--ldbal", is_flag=True)
@click.option("--lptbal", is_flag=True)
@click.option('--verbose', '-v', is_flag=True)
@click.option('--dry-run', is_flag=True)
def main(config_file, template_paths, hoc_lib_path, dataset_prefix, results_path, results_id, node_rank_file, io_size,
         vrecord_fraction, coredat, tstop, v_init, stimulus_onset, max_walltime_hours, results_write_time, dt, ldbal,
         lptbal, verbose, dry_run):
    """
    :param config_file: str; model configuration file
    :param template_paths: str; colon-separated list of paths to directories containing hoc cell templates
    :param hoc_lib_path: str; path to directory containing required hoc libraries
    :param dataset_prefix: str; path to directory containing required neuroh5 data files
    :param results_path: str; path to directory to export output files
    :param results_id: str; label for neuroh5 namespaces to write spike and voltage trace data
    :param node_rank_file: str; name of file specifying assignment of node gids to MPI ranks
    :param io_size: int; the number of MPI ranks to be used for I/O operations
    :param vrecord_fraction: float; fraction of cells to record intracellular voltage from
    :param coredat: bool; Save CoreNEURON data
    :param tstop: int; physical time to simulate (ms)
    :param v_init: float; initialization membrane potential (mV)
    :param stimulus_onset: float; starting time of stimulus (ms)
    :param max_walltime_hours: float; maximum wall time (hours)
    :param results_write_time: float; time to write out results at end of simulation
    :param dt: float; simulation time step
    :param ldbal: bool; estimate load balance based on cell complexity
    :param lptbal: bool; calculate load balance with LPT algorithm
    :param verbose: bool; print verbose diagnostic messages while constructing the network
    :param dry_run: bool; whether to actually execute simulation after building network
    """
    comm = MPI.COMM_WORLD
    np.seterr(all='raise')
    env = Env(comm, config_file, template_paths, hoc_lib_path, dataset_prefix, results_path, results_id,
              node_rank_file, io_size, vrecord_fraction, coredat, tstop, v_init, stimulus_onset, max_walltime_hours,
              results_write_time, dt, ldbal, lptbal, verbose)

    network.init(env)
    if not dry_run:
        network.run(env)


if __name__ == '__main__':
    main(args=sys.argv[(sys.argv.index(os.path.basename(__file__))+1):])
