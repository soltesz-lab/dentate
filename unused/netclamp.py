#!/usr/bin/env python
"""
Dentate Gyrus model single-cell simulation script.


"""

import os
import sys

import numpy as np

import click
import dentate.network as network
from dentate.env import Env
from dentate.utils import list_find
from mpi4py import MPI


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
@click.option("--results-id", type=str, required=False, default='',
                  help='identifier that is used to name neuroh5 namespaces that contain output spike and intracellular trace data')
@click.option("--input-path", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              help='file containing input spike trains')
@click.option("--input-namespace", type=str, default='Spike Events', help='namespace containing input spike trains')
@click.option("--target-cell", type=(str,int), required=True, multiple=True, help='which cell to simulate')
@click.option("--tstop", type=int, default=1, help='physical time to simulate (ms)')
@click.option("--v-init", type=float, default=-75.0, help='initialization membrane potential (mV)')
@click.option("--stimulus-onset", type=float, default=1.0, help='starting time of stimulus (ms)')
@click.option("--max-walltime-hours", type=float, default=1.0, help='maximum wall time (hours)')
@click.option("--results-write-time", type=float, default=360.0, help='time to write out results at end of simulation')
@click.option("--dt", type=float, default=0.025, help='')
@click.option('--verbose', '-v', is_flag=True, help='print verbose diagnostic messages while constructing the network')
@click.option('--dry-run', is_flag=True, help='whether to actually execute simulation after building network')
def main(config_file, template_paths, hoc_lib_path, dataset_prefix, config_prefix, results_path, results_id, 
         input_path, input_namespace, target_cell, tstop, v_init, stimulus_onset, max_walltime_hours, 
         results_write_time, dt, ldbal, lptbal, verbose, dry_run):
    """
    :param config_file: str; model configuration file name
    :param template_paths: str; colon-separated list of paths to directories containing hoc cell templates
    :param hoc_lib_path: str; path to directory containing required hoc libraries
    :param dataset_prefix: str; path to directory containing required neuroh5 data files
    :param config_prefix: str; path to directory containing network and cell mechanism config files
    :param results_path: str; path to directory to export output files
    :param results_id: str; label for neuroh5 namespaces to write spike and voltage trace data
    :param tstop: int; physical time to simulate (ms)
    :param v_init: float; initialization membrane potential (mV)
    :param stimulus_onset: float; starting time of stimulus (ms)
    :param max_walltime_hours: float; maximum wall time (hours)
    :param results_write_time: float; time to write out results at end of simulation
    :param dt: float; simulation time step
    :param verbose: bool; print verbose diagnostic messages while constructing the network
    :param dry_run: bool; whether to actually execute simulation after building network
    """
    comm = MPI.COMM_WORLD
    np.seterr(all='raise')
    vrecord_fraction = 1.0
    env = Env(comm, config_file, template_paths, hoc_lib_path, dataset_prefix, config_prefix,
              results_path, results_id, vrecord_fraction, target_cell, tstop, v_init, stimulus_onset,
              max_walltime_hours, results_write_time, dt,
              cell_selection=target_cell, spike_input_path=input_path, spike_input_ns=input_namespace, 
              verbose=verbose)
    network.init(env)
    if not dry_run:
        network.run(env)


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv)+1):])
