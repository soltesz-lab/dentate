
import sys, gc
from mpi4py import MPI
import click
import utils, plot

script_name = 'plot_selectivity_rate_map.py'

@click.command()
@click.option("--selectivity-path", '-p', required=True, type=click.Path())
@click.option("--selectivity-namespace", '-n', type=str, default='Vector Stimulus')
@click.option("--population", type=str)
@click.option("--trajectory-id", type=str)
@click.option("--cell-id", type=int, default=0)
@click.option("--font-size", type=float, default=14)
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
def main(selectivity_path, selectivity_namespace, population, cell_id, trajectory_id, font_size, verbose):
    plot.plot_selectivity_rate_map (selectivity_path, selectivity_namespace, population, cell_id=cell_id,
                                    trajectory_id=trajectory_id, fontSize=font_size, saveFig=True,
                                    verbose=verbose)

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])


    
