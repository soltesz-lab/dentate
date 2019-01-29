
import sys, gc, os
from mpi4py import MPI
import click
import dentate
from dentate import utils, plot

script_name = os.path.basename(__file__)


@click.command()
@click.option("--spike-events-path", '-p', required=True, type=click.Path())
@click.option("--spike-events-namespace", '-n', type=str, default='Spike Events')
@click.option("--populations", '-i', type=str, multiple=True)
@click.option("--spike-hist-bin", type=float, default=5.0)
@click.option("--smooth", type=int)
@click.option("--t-variable", type=str, default='t')
@click.option("--t-max", type=float)
@click.option("--t-min", type=float)
@click.option("--quantity", type=str, default='rate')
@click.option("--font-size", type=float, default=14)
@click.option("--graph-type", type=str, default='bar')
@click.option("--overlay", type=bool, default=False, is_flag=True)
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
def main(spike_events_path, spike_events_namespace, populations, spike_hist_bin, smooth, t_variable, t_max, t_min, quantity, font_size, graph_type, overlay, verbose):

    utils.config_logging(verbose)

    if t_max is None:
        timeRange = None
    else:
        if t_min is None:
            timeRange = [0.0, t_max]
        else:
            timeRange = [t_min, t_max]

    if not populations:
        populations = ['eachPop']
        
    plot.plot_spike_histogram (spike_events_path, spike_events_namespace, include=populations, time_variable=t_variable,
                               time_range=time_range, pop_rates=True, kernel_size=kernel_size, bin_size=spike_hist_bin, smooth=smooth,
                               quantity=quantity, fontSize=font_size, overlay=overlay, graph_type=graph_type, saveFig=True)
    

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == script_name, sys.argv)+1):])



    
