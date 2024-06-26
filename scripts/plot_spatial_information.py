import os
import sys

import click
from dentate import plot
from dentate import utils
from mpi4py import MPI

script_name = os.path.basename(__file__)


@click.command()
@click.option("--config", required=True, type=str)
@click.option("--config-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='config')
@click.option("--spike-events-path", '-p', required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--spike-events-namespace", '-n', type=str, default='Spike Events')
@click.option("--spike-train-attr-name", type=str, default='t')
@click.option("--populations", '-i', type=str, multiple=True, default=None)
@click.option("--include-artificial/--exclude-artificial", type=bool, default=True, is_flag=True)
@click.option("--t-max", type=float)
@click.option("--t-min", type=float)
@click.option("--trajectory-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--arena-id", '-a', type=str, default='Default')
@click.option("--trajectory-id", '-t', type=str, default='Default')
@click.option("--threshold", type=float, default=0.01)
@click.option("--bin-size", '-b', type=float, default=10.0)
@click.option("--output-file-path", required=False, type=str, default=None)
@click.option("--save-fig", required=False, type=str, default=None)
@click.option("--save-fig-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default=None)
@click.option("--font-size", type=float, default=14)
@click.option("--fig-size", type=(float,float), default=(15,8))
@click.option("--fig-format", required=False, type=str, default='svg')
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
def main(config, config_prefix, spike_events_path, spike_events_namespace, spike_train_attr_name, populations, include_artificial, t_max,
         t_min, trajectory_path, arena_id, trajectory_id, threshold, bin_size, output_file_path, save_fig, save_fig_dir, font_size,
         fig_size, fig_format, verbose):
    """

    :param config: str (file name)
    :param config_prefix: str (path to dir)
    :param spike_events_path: str (path to file)
    :param spike_events_namespace: str
    :param spike_train_attr_name: str
    :param populations: list of str
    :param t_max: float
    :param t_min: float
    :param trajectory_path: str (path to file)
    :param arena_id: str
    :param trajectory_id: str
    :param output_file_path: str (path to file)
    :param save_fig: str (base file name)
    :param save_fig_dir: str (path to dir)
    :param font_size: float
    :param fig_format: str
    :param verbose: bool

    """
    utils.config_logging(verbose)

    if t_max is None:
        time_range = None
    else:
        if t_min is None:
            time_range = [0.0, t_max]
        else:
            time_range = [t_min, t_max]

    plot.plot_spatial_information(spike_events_path, spike_events_namespace, trajectory_path, arena_id, trajectory_id,
                                  populations=populations, include_artificial=include_artificial, threshold=threshold,
                                  position_bin_size=bin_size, spike_train_attr_name=spike_train_attr_name, time_range=time_range,
                                  fontSize=font_size, verbose=verbose, output_file_path=output_file_path,
                                  saveFig=save_fig, figFormat=fig_format, figSize=fig_size)


if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == script_name, sys.argv)+1):])
