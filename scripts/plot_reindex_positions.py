import os
import sys

import numpy as np

import click
import dentate
from dentate import plot
from dentate import utils

script_name = os.path.basename(__file__)


@click.command()
@click.option("--config", required=True, type=str)
@click.option("--config-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--coords-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--population", '-i', type=str, required=True)
@click.option("--distances-namespace", '-d', type=str, default='Arc Distances')
@click.option("--reindex-namespace", '-n', type=str, default='Tree Reindex')
@click.option("--reindex-attribute", '-a', type=str, default='New Cell Index')
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
def main(coords_path, population, distances_namespace, reindex_namespace, reindex_attribute, verbose):
        
    utils.config_logging(verbose)
    logger = utils.get_script_logger(os.path.basename(script_name))

    env = Env(config_file=config, config_prefix=config_prefix)

    plot.plot_reindex_positions (env, coords_path, population, distances_namespace, reindex_namespace, reindex_attribute)
        

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == os.path.basename(script_name), sys.argv)+1):])
