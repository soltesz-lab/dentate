
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
@click.option("--trajectory-id", '-d', type=str, default=0)
@click.option("--bin-size", '-b', type=float, default=50.0)
@click.option("--min-pf-width", '-m', type=float, default=10.)
@click.option("--font-size", type=float, default=14)
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
def main(spike_events_path, spike_events_namespace, include, t_variable, t_max, t_min, trajectory_path, trajectory_id, bin_size, min_pf_width, font_size, verbose):

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

    plot.plot_place_fields (spike_events_path, spike_events_namespace, 
                            trajectory_path, trajectory_id, include = include,
                            bin_size = bin_size, min_pf_width = min_pf_width,
                            time_variable = t_variable, time_range = time_range, 
                            fontSize = font_size, save_data = True)

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == script_name, sys.argv)+1):])
    


    
