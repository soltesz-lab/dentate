import os, sys
import numpy as np
import click
from dentate import plot, utils

script_name = os.path.basename(__file__)

@click.command()
@click.option("--coords-path", '-c', required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--population", '-i', type=str, required=True)
@click.option("--coords-namespace", '-n', required=False, type=str, default='Coordinates')
@click.option("--graph-type", type=str, default='scatter')
@click.option("--xyz", type=bool, default=False, is_flag=True)
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
def main(coords_path, population, coords_namespace, graph_type, xyz, verbose):
        
    plot.plot_coordinates (coords_path, population, coords_namespace, graphType=graph_type, xyz=xyz, verbose=verbose)
        

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == script_name, sys.argv)+1):])

