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
@click.option("--cell-selection-path", required=False, type=click.Path(exists=True, file_okay=True, dir_okay=False),
              help='name of file specifying subset of cells gids to be instantiated')
@click.option("--config-file", required=True, type=str, help='model configuration file name')
@click.option("--template-paths", type=str, default='templates',
              help='colon-separated list of paths to directories containing hoc cell templates')
@click.option("--hoc-lib-path", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='.', help='path to directory containing required hoc libraries')
@click.option("--dataset-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
                  help='path to directory containing required neuroh5 data files')
@click.option("--config-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='config', help='path to directory containing network and cell mechanism config files')
@click.option("--results-path", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
                  help='path to directory where output files will be written')
@click.option("--results-id", type=str, required=False, default=None,
                  help='identifier that is used to name neuroh5 namespaces that contain output spike and '
                       'intracellular trace data')
@click.option("--node-rank-file", required=False, type=click.Path(exists=True, file_okay=True, dir_okay=False),
                  help='name of file specifying assignment of cell gids to MPI ranks')
@click.option("--io-size", type=int, default=0, help='the number of MPI ranks to be used for I/O operations')
@click.option("--vrecord-fraction", type=float, default=0.001,
              help='fraction of cells to record intracellular voltage from')
@click.option("--coredat", is_flag=True, help='Save CoreNEURON data')
@click.option("--tstop", type=int, default=1, help='physical time to simulate (ms)')
@click.option("--v-init", type=float, default=-75.0, help='initialization membrane potential (mV)')
@click.option("--stimulus-onset", type=float, default=1.0, help='starting time of stimulus (ms)')
@click.option("--max-walltime-hours", type=float, default=1.0, help='maximum wall time (hours)')
@click.option("--results-write-time", type=float, default=360.0, help='time to write out results at end of simulation')
@click.option("--spike-input-path", required=False, type=click.Path(exists=True, file_okay=True, dir_okay=False),
                  help='path to file for input spikes when cell selection is specified')
@click.option("--spike-input-namespace", required=False, type=str,
                  help='namespace for input spikes when cell selection is specified')
@click.option("--dt", type=float, default=0.025, help='')
@click.option("--ldbal", is_flag=True, help='estimate load balance based on cell complexity')
@click.option("--lptbal", is_flag=True, help='optimize load balancing assignment with LPT algorithm')
@click.option('--cleanup', type=bool, default=True,
              help='delete from memory the synapse attributes metadata after specifying connections')
@click.option('--verbose', '-v', is_flag=True, help='print verbose diagnostic messages while constructing the network')
@click.option('--run-test', is_flag=True, help='whether to actually execute simulation after building network')
def main(cell_selection_path, config_file, template_paths, hoc_lib_path, dataset_prefix, config_prefix,
         results_path, results_id, node_rank_file, io_size, vrecord_fraction, coredat, tstop, v_init,
         stimulus_onset, max_walltime_hours, results_write_time, spike_input_path, spike_input_namespace,
         dt, ldbal, lptbal, cleanup, verbose, run_test):
    """
    :param cell_selection_path: str; name of file specifying subset of cells gids to be instantiated
    :param config_file: str; model configuration file name
    :param template_paths: str; colon-separated list of paths to directories containing hoc cell templates
    :param hoc_lib_path: str; path to directory containing required hoc libraries
    :param dataset_prefix: str; path to directory containing required neuroh5 data files
    :param config_prefix: str; path to directory containing network and cell mechanism config files
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
    :param spike_input_path: str; path to file for input spikes when cell selection is specified
    :param spike_input_namespace: str;
    :param dt: float; simulation time step
    :param ldbal: bool; estimate load balance based on cell complexity
    :param lptbal: bool; calculate load balance with LPT algorithm
    :param cleanup: bool; whether to delete from memory the synapse attributes metadata after specifying connections
    :param verbose: bool; print verbose diagnostic messages while constructing the network
    :param run_test: bool; whether to actually execute simulation after building network
    """
    comm = MPI.COMM_WORLD
    np.seterr(all='raise')
    params = dict(locals())
    env = Env(**params)
    network.init(env)
    if run_test:
        network.run(env, output=False)


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv) + 1):])
