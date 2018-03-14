
import sys, gc
from mpi4py import MPI
import click
import utils, plot

script_name = 'plot_connectivity.py'

@click.command()
@click.option("--connectivity-path", '-p', required=True, type=click.Path())
@click.option("--coords-path", '-c', required=True, type=click.Path())
@click.option("--indegree-namespace", '-n', type=str, required=True)
@click.option("--distances-namespace", '-d', type=str, default='Arc Distance')
@click.option("--destination", '-t', type=str, required=True)
@click.option("--font-size", type=float, default=14)
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
def main(connectivity_path, coords_path, indegree_namespace, distances_namespace, destination, font_size, verbose):
    plot.plot_in_degree (connectivity_path, coords_path, indegree_namespace, distances_namespace, destination,
                         fontSize=font_size, saveFig=True, verbose=verbose)

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])


    
