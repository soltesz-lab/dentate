
import sys, gc, os
import click
import dentate
from dentate import utils, plot

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
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
<<<<<<< HEAD
def main(features_path, coords_path, features_namespace, trajectory_id, distances_namespace, include, bin_size, from_spikes, normed, font_size, verbose):

    plot.plot_stimulus_spatial_rate_map (features_path, coords_path, trajectory_id, features_namespace, \
    distances_namespace, include, binSize=bin_size, fromSpikes=from_spikes, normed=normed, fontSize=font_size, \
    saveFig=True, showFig=True, verbose=verbose)
=======
def main(features_path, coords_path, features_namespace, distances_namespace, include, normed, font_size, verbose):
<<<<<<< HEAD
    
    utils.config_logging(verbose)

    plot.plot_stimulus_spatial_rate_map (features_path, coords_path, features_namespace, distances_namespace,
                                         include, normed=normed, fontSize=font_size, saveFig=True)
=======

    utils.config_logging(verbose)
    logger = utils.get_script_logger(os.path.basename(__file__))

    plot.plot_stimulus_spatial_rate_map (features_path, coords_path, features_namespace, distances_namespace,
                                         include, normed=normed, fontSize=font_size, saveFig=True, showFig=True, verbose=verbose)
>>>>>>> dff6dcba2fa0f8a3f121e194eb6d2be7fb7e258c
>>>>>>> 55b41f830a7707f6b6832a31bcc3e6fc39b9f49c

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv)+1):])



    
