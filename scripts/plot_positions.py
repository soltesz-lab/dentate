import sys
from mpi4py import MPI
import numpy as np
from neuroh5.io import read_population_ranges, read_population_names, read_cell_attributes
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
@click.option("--bin-size", type=float, default=50.0)
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
def main(coords_path, distances_namespace, population, bin_size, verbose):

    soma_distances = read_cell_attributes(coords_path, population, namespace=distances_namespace)
    
    plot.plot_positions (population, soma_distances, binSize=bin_size, verbose=verbose, saveFig=True)
        

if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find("plot_positions.py") != -1,sys.argv)+1):])
