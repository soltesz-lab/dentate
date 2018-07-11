import sys
from mpi4py import MPI
import numpy as np
from dentate import plot
import click

def list_find (f, lst):
    i=0
    for x in lst:
        if f(x):
            return i
        else:
            i=i+1
    return None

@click.command()
@click.option("--coords-path", '-c', required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--population", '-i', type=str, required=True)
@click.option("--coords-namespace", required=False, type=str, default='Coordinates')
@click.option("--graph-type", type=str, default='scatter')
@click.option("--project", type=float, default=3.1)
@click.option("--rotate", type=float)
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
def main(coords_path, population, coords_namespace, graph_type, project, rotate, verbose):
        
    plot.plot_projected_coordinates (coords_path, population, coords_namespace, graphType=graph_type, project=project, rotate=rotate, verbose=verbose)
        

if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find("plot_projected_coordinates.py") != -1,sys.argv)+1):])
