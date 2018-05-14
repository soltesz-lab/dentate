
import sys, gc
from mpi4py import MPI
import click
import dentate
from dentate import plot, utils

script_name = 'plot_tree_metrics.py'

@click.command()
@click.option("--forest-path", '-p', required=True, type=click.Path())
@click.option("--coords-path", '-c', required=True, type=click.Path())
@click.option("--population", type=str)
@click.option("--metric-namespace", type=str, default='Tree Measurements')
@click.option("--distances-namespace", '-t', type=str, default='Arc Distances')
@click.option("--metric", type=str, default='dendrite_length')
@click.option("--metric-index", type=int, default=0)
#@click.option("--normed", type=bool, default=False, is_flag=True)
@click.option("--font-size", type=float, default=14)
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
def main(forest_path, coords_path, population, metric_namespace, distances_namespace, metric, metric_index, font_size, verbose):
    plot.plot_tree_metrics (forest_path, coords_path, population, metric_namespace, distances_namespace,
                                metric=metric, metric_index=metric_index, fontSize=font_size, saveFig=True, verbose=verbose)

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])


    
