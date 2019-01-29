
import sys, gc, os
from mpi4py import MPI
import click
import utils, plot

script_name = os.path.basename(__file__)

@click.command()
@click.option("--spike-events-path", '-p', required=True, type=click.Path())
@click.option("--spike-events-namespace", '-n', type=str, default='Spike Events')
@click.option("--populations", '-i', type=str, multiple=True)
@click.option("--spike-hist-bin", type=float, default=5.0)
@click.option("--t-variable", type=str, default='t')
@click.option("--t-max", type=float)
@click.option("--t-min", type=float)
@click.option("--lag", type=int, default=1)
@click.option("--max-cells", type=int)
@click.option("--graph-type", type=str)
@click.option("--font-size", type=float, default=14)
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
def main(spike_events_path, spike_events_namespace, populations, spike_hist_bin, t_variable, t_max, t_min, lag, max_cells, graph_type, font_size, verbose):

    utils.config_logging(verbose)

    if t_max is None:
        time_range = None
    else:
        if t_min is None:
            time_range = [0.0, t_max]
        else:
            time_range = [t_min, t_max]

    if not populations:
        populations = ['eachPop']

    if graph_type is None:
        graph_type = 'matrix'
        
    plot.plot_spike_histogram_autocorr (spike_events_path, spike_events_namespace, include=populations, time_range=time_range, time_variable=t_variable,  lag=lag,
                                        bin_size=spike_hist_bin, maxCells=max_cells, graph_type=graph_type, fontSize=font_size, saveFig=True)
        

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == script_name, sys.argv)+1):])



    
