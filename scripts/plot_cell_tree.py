
import os, sys, click, re
import dentate
from dentate import plot, utils, statedata
from mpi4py import MPI

script_name = os.path.basename(__file__)

@click.command()
@click.option("--forest-path", '-f', required=True, type=click.Path())
@click.option("--population", '-i', type=str)
@click.option("--gid", type=int, default=None)
@click.option("--font-size", type=float, default=14)
@click.option("--colormap", type=str, default='coolwarm')
@click.option("--query", "-q", type=bool, default=False, is_flag=True)
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
def main(forest_path, population, gid, font_size, colormap, query, verbose):

    utils.config_logging(verbose)
        
    plot.plot_cell_tree (gid, population, forest_path, colormap=colormap, saveFig=True)
    

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == script_name, sys.argv)+1):])
