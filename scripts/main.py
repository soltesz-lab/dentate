#!/usr/bin/env python
"""
Dentate Gyrus model simulation script.


"""

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
                  help='identifier that is used to name neuroh5 namespaces that contain output spike and intracellular trace data')
@click.option("--node-rank-file", required=False, type=click.Path(exists=True, file_okay=True, dir_okay=False),
                  help='name of file specifying assignment of cell gids to MPI ranks')
@click.option("--io-size", type=int, default=0, help='the number of MPI ranks to be used for I/O operations')
@click.option("--vrecord-fraction", type=float, default=0.001, help='fraction of cells to record intracellular voltage from')
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
@click.option('--verbose', '-v', is_flag=True, help='print verbose diagnostic messages while constructing the network')
@click.option('--dry-run', is_flag=True, help='whether to actually execute simulation after building network')
def main(cell_selection_path, config_file, template_paths, hoc_lib_path, dataset_prefix, config_prefix,
         results_path, results_id, node_rank_file, io_size, vrecord_fraction, coredat, tstop, v_init,
         stimulus_onset, max_walltime_hours, results_write_time, spike_input_path, spike_input_namespace,
         dt, ldbal, lptbal, verbose, dry_run):
    comm = MPI.COMM_WORLD
    np.seterr(all='raise')
    params = dict(locals())
    env = Env(**params)
    network.init(env)
    if not dry_run:
        network.run(env)


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv)+1):])
