
import sys, os, gc, math
from mpi4py import MPI
import click
import dentate, utils, plot

@click.command()
@click.option("--connectivity-path", '-p', required=True, type=click.Path())
@click.option("--coords-path", '-c', required=True, type=click.Path())
@click.option("--distances-namespace", '-t', type=str, default='Arc Distances')
@click.option("--destination-gid", '-g', type=int)
@click.option("--destination", '-d', type=str)
@click.option("--source", '-s', type=str)
@click.option("--bin-size", type=float, default=20.0)
@click.option("--font-size", type=float, default=14)
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
def main(connectivity_path, coords_path, distances_namespace, destination_gid, destination, source, bin_size, font_size, verbose):

    utils.config_logging(verbose)
    logger = utils.get_script_logger(os.path.basename(__file__))

    plot.plot_single_vertex_dist (connectivity_path, coords_path, distances_namespace,
                                  destination_gid, destination, source, bin_size, fontSize=font_size,
                                  saveFig=True)
    

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv)+1):])
