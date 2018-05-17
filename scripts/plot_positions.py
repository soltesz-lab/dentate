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
@click.option("--distances-namespace", '-d', type=str, default='Arc Distances')
@click.option("--population", '-i', type=str, required=True)
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
def main(coords_path, distances_namespace, population, verbose):
        
    plot.plot_positions (coords_path, population, distances_namespace, verbose=verbose)
        

if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find("plot_positions.py") != -1,sys.argv)+1):])
