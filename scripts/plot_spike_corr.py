
import gc
import os
import sys

import click
import plot
import utils
from mpi4py import MPI

script_name = os.path.basename(__file__)

@click.command()
@click.option("--spike-events-path", '-p', required=True, type=click.Path())
@click.option("--spike-events-namespace", '-n', type=str, default='Spike Events')
@click.option("--populations", '-i', type=str, multiple=True)
@click.option("--spike-hist-bin", type=float, default=5.0)
@click.option("--t-variable", type=str, default='t')
@click.option("--t-max", type=float)
@click.option("--t-min", type=float)
@click.option("--max-cells", type=int)
@click.option("--graph-type", type=str)
@click.option("--font-size", type=float, default=14)
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
def main(spike_events_path, spike_events_namespace, populations, spike_hist_bin, t_variable, t_max, t_min, max_cells, graph_type, font_size, verbose):

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

    if graph_type:
        graphType = graph_type
    else:
        graphType = 'matrix'
        
    plot.plot_spike_histogram_corr (spike_events_path, spike_events_namespace, include=populations, time_range=time_range, time_variable=t_variable, 
                                    bin_size=spike_hist_bin, max_cells=max_cells, graph_type=graph_type, fontSize=font_size, saveFig=True)
        
    

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == script_name, sys.argv)+1):])
