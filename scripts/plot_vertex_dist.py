
import sys, gc, math
from mpi4py import MPI
import click
import dentate, utils, plot

script_name = 'plot_vertex_dist.py'

@click.command()
@click.option("--connectivity-path", '-p', required=True, type=click.Path())
@click.option("--coords-path", '-c', required=True, type=click.Path())
@click.option("--distances-namespace", '-t', type=str, default='Arc Distances')
@click.option("--destination", '-d', type=str)
@click.option("--source", '-s', type=str)
@click.option("--bin-size", type=float, default=20.0)
@click.option("--font-size", type=float, default=14)
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
def main(connectivity_path, coords_path, distances_namespace, destination, source, bin_size, font_size, verbose):
    plot.plot_vertex_dist (connectivity_path, coords_path, distances_namespace,
                               destination, source, bin_size, fontSize=font_size,
                               saveFig=True, verbose=verbose)



if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])


    
