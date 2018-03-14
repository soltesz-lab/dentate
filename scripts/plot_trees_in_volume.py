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
@click.option("--forest-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--population", '-i', type=str, required=True)
@click.option("--rotate", type=float)
@click.option("--width", type=float, default=3.0)
@click.option("--sample", type=float, default=0.05)
@click.option("--subvol", type=bool, default=False, is_flag=True)
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
def main(config, forest_path, population, rotate, width, sample, subvol, verbose):
        
    plot.plot_trees_in_volume (population, forest_path, config, \
                               sample=sample, subvol=subvol, width=width, rotate=rotate, \
                                verbose=verbose)
        

if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find("plot_trees_in_volume.py") != -1,sys.argv)+1):])
