
import sys, gc
from mpi4py import MPI
import click
import utils, plot

script_name = 'plot_stimulus_spatial_map.py'

@click.command()
@click.option("--features-path", '-p', required=True, type=click.Path())
@click.option("--features-namespace", '-n', type=str, default='Vector Stimulus')
@click.option("--coords-path", '-c', required=True, type=click.Path())
@click.option("--distances-namespace", '-d', type=str, default='Arc Distance')
@click.option("--include", '-i', type=str, multiple=True)
@click.option("--font-size", type=float, default=14)
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
def main(features_path, coords_path, features_namespace, distances_namespace, include, font_size, verbose):
    plot.plot_stimulus_spatial_rate_map (features_path, coords_path, features_namespace, distances_namespace,
                                         include, fontSize=font_size, saveFig=True, verbose=verbose)

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])


    
