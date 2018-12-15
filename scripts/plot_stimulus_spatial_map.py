import sys, os
import click
import dentate.utils as utils
import dentate.plot as plot


@click.command()
@click.option("--features-path", '-p', required=True, type=click.Path())
@click.option("--features-namespace", '-n', type=str, default='Vector Stimulus')
@click.option("--trajectory-id", type=str, required=True)
@click.option("--coords-path", '-c', required=True, type=click.Path())
@click.option("--distances-namespace", '-d', type=str, default='Arc Distances')
@click.option("--include", '-i', type=str, multiple=True)
@click.option("--bin-size", type=float, default=100.)
@click.option("--from-spikes", type=bool, default=True)
@click.option("--normed", type=bool, default=False, is_flag=True)
@click.option("--font-size", type=float, default=14)
@click.option("--verbose", "-v", is_flag=True)
@click.option("--show-fig", is_flag=True)
@click.option("--save-fig", is_flag=True)
def main(features_path, coords_path, features_namespace, trajectory_id, distances_namespace, include, bin_size,
         from_spikes, normed, font_size, verbose, show_fig, save_fig):
    """

    :param features_path:
    :param coords_path:
    :param features_namespace:
    :param trajectory_id:
    :param distances_namespace:
    :param include:
    :param bin_size:
    :param from_spikes:
    :param normed:
    :param font_size:
    :param verbose:
    :param show_fig:
    :param save_fig:
    """
    utils.config_logging(verbose)
    logger = utils.get_script_logger(os.path.basename(__file__))

    plot.plot_stimulus_spatial_rate_map (features_path, coords_path, trajectory_id, features_namespace,
        distances_namespace, include, binSize=bin_size, fromSpikes=from_spikes, normed=normed, fontSize=font_size,
        saveFig=save_fig, showFig=show_fig, verbose=verbose)


if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv)+1):],
        standalone_mode = False)