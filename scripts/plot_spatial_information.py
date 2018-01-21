
import sys, gc
from mpi4py import MPI
import click
import utils, plot

script_name = 'plot_spatial_information.py'

@click.command()
@click.option("--spike-events-path", '-p', required=True, type=click.Path())
@click.option("--spike-events-namespace", '-n', type=str, default='Spike Events')
@click.option("--include", '-i', type=str, multiple=True)
@click.option("--t-variable", type=str, default='t')
@click.option("--t-max", type=float)
@click.option("--t-min", type=float)
@click.option("--load-file", type=click.Path())
@click.option("--trajectory-path", '-t', required=True, type=click.Path())
@click.option("--trajectory-id", '-d', type=int, default=0)
@click.option("--position-bin-size", '-b', type=float, default=5.0)
@click.option("--font-size", type=float, default=14)
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
def main(spike_events_path, spike_events_namespace, include, t_variable, t_max, t_min, trajectory_path, trajectory_id, load_file, position_bin_size, font_size, verbose):
    if t_max is None:
        timeRange = None
    else:
        if t_min is None:
            timeRange = [0.0, t_max]
        else:
            timeRange = [t_min, t_max]

    if not include:
        include = ['eachPop']

    plot.plot_spatial_information (spike_events_path, spike_events_namespace, 
                                    trajectory_path, trajectory_id, include = include,
                                    loadFile = load_file, positionBinSize = position_bin_size,
                                    binCount = 10, timeVariable=t_variable, timeRange = timeRange, 
                                    fontSize = font_size, verbose = verbose)

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])


    
