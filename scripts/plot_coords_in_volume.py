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
@click.option("--config", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--coords-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--coords-namespace", '-n', type=str, default='Coordinates')
@click.option("--population", '-i', type=str, required=True)
@click.option("--scale", type=float, default=25.0)
@click.option("--subvol", type=bool, default=False, is_flag=True)
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
def main(config, coords_path, coords_namespace, population, scale, subvol, verbose):
        
    plot.plot_coords_in_volume (population, coords_path, coords_namespace, config, \
                                subvol=subvol, scale=scale, verbose=verbose)
        

if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find("plot_coords_in_volume.py") != -1,sys.argv)+1):])
