
import sys, os, gc
from mpi4py import MPI
import click
import dentate
from dentate import utils, plot



@click.command()
@click.option("--features-path", '-p', required=True, type=click.Path())
@click.option("--features-namespace", '-n', type=str, default='Vector Stimulus')
@click.option("--trajectory-id", '-t', type=int, default=0)
@click.option("--include", '-i', type=str, multiple=True)
@click.option("--font-size", type=float, default=14)
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
def main(features_path, features_namespace, trajectory_id, include, font_size, verbose):

    utils.config_logging(verbose)

    plot.plot_stimulus_rate (features_path, features_namespace, include, trajectory_id=trajectory_id, \
                             fontSize=font_size, saveFig=True)

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv)+1):])



    
