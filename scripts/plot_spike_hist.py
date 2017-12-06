
import sys, gc
from mpi4py import MPI
import click
import utils, plot

script_name = 'plot_spike_hist.py'

@click.command()
@click.option("--spike-events-path", '-p', required=True, type=click.Path())
@click.option("--spike-events-namespace", '-n', type=str, default='Spike Events')
@click.option("--populations", '-i', type=str, multiple=True)
@click.option("--spike-hist-bin", type=float, default=5.0)
@click.option("--t-variable", type=str, default='t')
@click.option("--t-max", type=float)
@click.option("--t-min", type=float)
@click.option("--y-axis", type=str, default='rate')
@click.option("--font-size", type=float, default=14)
@click.option("--graph-type", type=str, default='bar')
@click.option("--overlay", type=bool, default=False, is_flag=True)
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
def main(spike_events_path, spike_events_namespace, populations, spike_hist_bin, t_variable, t_max, t_min, y_axis, font_size, graph_type, overlay, verbose):
    if t_max is None:
        timeRange = None
    else:
        if t_min is None:
            timeRange = [0.0, t_max]
        else:
            timeRange = [t_min, t_max]

    if not populations:
        populations = ['eachPop']
        
    plot.plot_spike_histogram (spike_events_path, spike_events_namespace, include=populations, timeVariable=t_variable,
                               timeRange=timeRange, popRates=True, binSize=spike_hist_bin, yaxis=y_axis, fontSize=font_size,
                               overlay=overlay, graphType=graph_type, saveFig=True, verbose=verbose)
    

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])


    
