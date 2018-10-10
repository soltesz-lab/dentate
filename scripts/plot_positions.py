import sys, os, gc
import click
import dentate
from dentate import utils, plot

@click.command()
@click.option("--coords-path", '-c', required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--distances-namespace", '-d', type=str, default='Arc Distances')
@click.option("--population", '-i', type=str, required=True)
@click.option("--graph-type", type=str, default='kde')
@click.option("--bin-size", type=float, default=50.0)
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
def main(coords_path, distances_namespace, population, graph_type, bin_size, verbose):

    utils.config_logging(verbose)

    soma_distances = read_cell_attributes(coords_path, population, namespace=distances_namespace)
    
    plot.plot_positions (population, soma_distances, binSize=bin_size, graphType=graph_type, saveFig=True)
        

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv)+1):])
