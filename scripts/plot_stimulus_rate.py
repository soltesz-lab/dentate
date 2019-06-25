import os
import sys

import click
import dentate
from dentate import plot
from dentate import utils

script_name = os.path.basename(__file__)

@click.command()
@click.option("--features-path", '-p', required=True, type=click.Path())
@click.option("--features-namespace", '-n', type=str)
@click.option("--arena-id", '-a', type=str)
@click.option("--trajectory-id", '-t', type=str)
@click.option("--include", '-i', type=str, multiple=True)
@click.option("--font-size", type=float, default=14)
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
@click.option("--save-fig", is_flag=True)
def main(features_path, features_namespace, arena_id, trajectory_id, include, font_size, verbose, save_fig):
    """
    
    :param features_path: 
    :param features_namespace: 
    :param trajectory_id: 
    :param include: 
    :param font_size: 
    :param verbose: 
    :param save_fig:  
    """
    utils.config_logging(verbose)
    
    for population in include:
        plot.plot_stimulus_rate(features_path, features_namespace, population,
                                arena_id=arena_id, trajectory_id=trajectory_id,
                                fontSize=font_size, saveFig=save_fig)


if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == script_name, sys.argv)+1):], 
         standalone_mode=False)
