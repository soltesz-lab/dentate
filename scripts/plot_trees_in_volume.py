import sys

import numpy as np

import click
from dentate import plot
from mpi4py import MPI


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
@click.option("--forest-path", '-p', required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--coords-path", '-c', required=False, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--population", '-i', type=str, required=True)
@click.option("--line-width", type=float, default=1.0)
@click.option("--sample", type=float, default=0.05)
@click.option("--longitudinal-extent", '-l', nargs=2, type=float)
@click.option("--gid", '-g', type=int, multiple=True)
@click.option("--volume-plot", type=str, default='subvol')
@click.option("--color-edge-scalars/--no-color-edge-scalars", default=True)
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
def main(config, forest_path, coords_path, population, line_width, sample, longitudinal_extent, gid, volume_plot, color_edge_scalars, verbose):


    if len(gid) == 0:
        plot.plot_trees_in_volume (population, forest_path, config, coords_path=coords_path, \
                                   sample=sample, longitudinal_extent=longitudinal_extent, \
                                   volume=volume_plot, line_width=line_width, \
                                   color_edge_scalars=color_edge_scalars, \
                                   verbose=verbose)
    else:
        plot.plot_trees_in_volume (population, forest_path, config, \
                                   sample=set(gid), volume=volume_plot, line_width=line_width,  \
                                   color_edge_scalars=color_edge_scalars, verbose=verbose)
        
        

if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find("plot_trees_in_volume.py") != -1,sys.argv)+1):])
