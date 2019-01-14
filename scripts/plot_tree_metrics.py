
import sys, gc
from mpi4py import MPI
import click
import dentate
from dentate import plot, utils

script_name = os.path.basename(__file__)


@click.command()
@click.option("--config", required=True, type=str)
@click.option("--config-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--forest-path", '-p', required=True, type=click.Path())
@click.option("--coords-path", '-c', required=True, type=click.Path())
@click.option("--population", type=str)
@click.option("--metric-namespace", type=str, default='Tree Measurements')
@click.option("--distances-namespace", '-t', type=str, default='Arc Distances')
@click.option("--metric", type=str, default='dendrite_length')
@click.option("--metric-index", type=int, default=0)
@click.option("--percentile", type=float)
#@click.option("--normed", type=bool, default=False, is_flag=True)
@click.option("--font-size", type=float, default=14)
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
def main(config, config_prefix, forest_path, coords_path, population, metric_namespace, distances_namespace, metric, metric_index, percentile, font_size, verbose):

    env = Env(config_file=config, config_prefix=config_prefix)

    plot.plot_tree_metrics (env, forest_path, coords_path, population, metric_namespace, distances_namespace,
                                metric=metric, metric_index=metric_index, percentile=percentile, fontSize=font_size, saveFig=True, verbose=verbose)

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])


    
