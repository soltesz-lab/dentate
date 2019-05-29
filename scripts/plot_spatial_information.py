import sys, os
from mpi4py import MPI
import click
from dentate import utils, plot

script_name = os.path.basename(__file__)


@click.command()
@click.option("--spike-events-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--spike-events-namespace", '-n', type=str, default='Spike Events')
@click.option("--spike-train-attr-name", type=str, default='spiketrain')
@click.option("--populations", '-p', type=str, multiple=True, default=None)
@click.option("--t-max", type=float)
@click.option("--t-min", type=float)
@click.option("--trajectory-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--arena-id", '-a', type=str, default='Default')
@click.option("--trajectory-id", '-t', type=str, default='Default')
@click.option("--position-bin-size", '-b', type=float, default=10.0)
@click.option("--font-size", type=float, default=14)
@click.option("--output-file-path", required=False, type=str, default=None)
@click.option("--plot-dir-path", type=click.Path(exists=True, file_okay=False, dir_okay=True), default=None)
@click.option("--save-fig", required=False, type=str, default=None)
@click.option("--fig-format", required=False, type=str, default='svg')
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
def main(spike_events_path, spike_events_namespace, spike_train_attr_name, populations, t_max, t_min, trajectory_path,
         arena_id, trajectory_id, position_bin_size, font_size, output_file_path, plot_dir_path, save_fig, fig_format,
         verbose):
    """

    :param spike_events_path:
    :param spike_events_namespace:
    :param spike_train_attr_name: str
    :param populations: list of str
    :param t_attr_name: str
    :param t_max:
    :param t_min:
    :param trajectory_path:
    :param arena_id:
    :param trajectory_id:
    :param position_bin_size:
    :param font_size:
    :param output_file_path: str
    :param plot_dir_path: str (path to dir)
    :param save_fig: str (filename prefix)
    :param fig_format: str
    :param verbose: bool
    """
    utils.config_logging(verbose)

    plot.plot_spatial_information(spike_events_path, spike_events_namespace, trajectory_path, arena_id, trajectory_id,
                                  populations=populations, position_bin_size=position_bin_size,
                                  spike_train_attr_name=spike_train_attr_name, time_range=[t_min, t_max],
                                  fontSize=font_size, verbose=verbose, output_file_path=output_file_path,
                                  plot_dir_path=plot_dir_path, saveFig=save_fig, figFormat=fig_format,
                                  baks_alpha=utils.default_baks_alpha, baks_beta=utils.default_baks_beta)


if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == script_name, sys.argv)+1):])
