
import sys, os
from mpi4py import MPI
import click
from dentate import utils, plot

script_name = os.path.basename(__file__)

@click.command()
@click.option("--spike-events-path", '-p', required=True, type=click.Path())
@click.option("--spike-events-namespace", '-n', type=str, default='Spike Events')
@click.option("--include", '-i', type=str, multiple=True)
@click.option("--t-variable", type=str, default='t')
@click.option("--t-max", type=float)
@click.option("--t-min", type=float)
@click.option("--trajectory-path", '-t', required=True, type=click.Path())
@click.option("--trajectory-id", '-s', type=str, default='Default')
@click.option("--position-bin-size", '-b', type=float, default=10.0)
@click.option("--font-size", type=float, default=14)
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
def main(spike_events_path, spike_events_namespace, include, t_variable, t_max, t_min, trajectory_path, trajectory_id, position_bin_size, font_size, verbose):

    utils.config_logging(verbose)

    if t_max is None:
        time_range = None
    else:
        if t_min is None:
            time_range = [0.0, t_max]
        else:
            time_range = [t_min, t_max]

    if not include:
        include = ['eachPop']

    plot.plot_spatial_information (spike_events_path, spike_events_namespace, 
                                    trajectory_path, trajectory_id, include = include,
                                    position_bin_size = position_bin_size, 
                                    time_variable=t_variable, time_range = time_range, 
                                    fontSize = font_size, verbose = verbose, saveData = True)

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == script_name, sys.argv)+1):])



    
