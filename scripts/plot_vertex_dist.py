
import os, sys, gc, math
import click
import dentate
from dentate import plot, utils
from mpi4py import MPI

sys_excepthook = sys.excepthook
def mpi_excepthook(type, value, traceback):
    sys_excepthook(type, value, traceback)
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)
sys.excepthook = mpi_excepthook

script_name = os.path.basename(__file__)

@click.command()
@click.option("--connectivity-path", '-p', required=True, type=click.Path())
@click.option("--coords-path", '-c', required=True, type=click.Path())
@click.option("--distances-namespace", '-t', type=str, default='Arc Distances')
@click.option("--destination", '-d', type=str)
@click.option("--source", '-s', type=str, multiple=True)
@click.option("--bin-size", type=float, default=20.0)
@click.option("--font-size", type=float, default=14)
@click.option("--fig-size", type=(int,int), default=(15,5))
@click.option("--fig-format", type=str, default="png")
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
def main(connectivity_path, coords_path, distances_namespace, destination, source, bin_size, font_size, fig_size, fig_format, verbose):

    utils.config_logging(verbose)
    comm = MPI.COMM_WORLD

    plot.plot_vertex_distribution (connectivity_path, coords_path, distances_namespace,
                                   destination, source, bin_size, fontSize=font_size,
                                   saveFig=True, figSize=fig_size, figFormat=fig_format, comm=comm)


    

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == script_name, sys.argv)+1):])
