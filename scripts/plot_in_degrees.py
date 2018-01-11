
import sys, gc
from mpi4py import MPI
import click
import utils, plot

script_name = 'plot_in_degrees.py'

@click.command()
@click.option("--connectivity-path", '-p', required=True, type=click.Path())
@click.option("--coords-path", '-c', required=True, type=click.Path())
@click.option("--vertex-metrics-namespace", type=str, default='Vertex Metrics')
@click.option("--distances-namespace", '-t', type=str, default='Arc Distance')
@click.option("--destination", '-d', type=str)
@click.option("--sources", '-s', type=str, multiple=True)
@click.option("--normed", type=bool, default=False, is_flag=True)
@click.option("--font-size", type=float, default=14)
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
def main(connectivity_path, coords_path, vertex_metrics_namespace, distances_namespace, destination, sources, normed, font_size, verbose):
    plot.plot_in_degree (connectivity_path, coords_path, vertex_metrics_namespace, distances_namespace,
                         destination, sources, normed=normed, fontSize=font_size, saveFig=True, verbose=verbose)

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])


    
