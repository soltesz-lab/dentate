
import sys, gc
from mpi4py import MPI
import click
from dentate import utils, plot

script_name = 'plot_EC_place_cells.py'

@click.command()
@click.option("--features-path", required=True, type=click.Path())
@click.option("--population", type=str, default='Spike Events')
@click.option("--nfields", type=int, default=1)
@click.option("--to-plot", type=int, default=100)
@click.option("--show_fig", type=int, default=1)
@click.option("--save_fig", type=str, default='Place-Fields.png')
def main(features_path, population, nfields, to_plot, show_fig, save_fig):
    plot.plot_place_cells(features_path, population, nfields, to_plot, show_fig, save_fig)

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])


    
