import sys, os
import click
import dentate
from dentate import plot, utils

script_name = os.path.basename(__file__)


@click.command()
@click.option("--config", required=True, type=str)
@click.option("--config-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True))
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
def main(config, config_prefix, features_path, coords_path, features_namespace, trajectory_id, distances_namespace, include, bin_size,
         from_spikes, normed, font_size, verbose, show_fig, save_fig):

    utils.config_logging(verbose)
    
    logger = utils.get_script_logger(os.path.basename(script_name))

    env = Env(config_file=config, config_prefix=config_prefix)

    plot.plot_stimulus_spatial_rate_map (env, features_path, coords_path, trajectory_id, features_namespace,
        distances_namespace, include, bin_size=bin_size, from_spikes=from_spikes, normed=normed, fontSize=font_size,
        saveFig=save_fig, showFig=show_fig, verbose=verbose)


if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == script_name, sys.argv)+1):])
