
import sys, gc, os
from mpi4py import MPI
import click
import utils, plot

@click.command()
@click.option("--spike-events-path", '-p', required=True, type=click.Path())
@click.option("--spike-events-namespace", '-n', type=str, default='Spike Events')
@click.option("--populations", '-i', type=str, multiple=True)
@click.option("--bin-size", type=float, default=50.0)
@click.option("--t-variable", type=str, default='t')
@click.option("--t-max", type=float)
@click.option("--t-min", type=float)
@click.option("--quantity", type=str, default='rate')
@click.option("--graph-type", type=str, default='point')
@click.option("--font-size", type=float, default=14)
@click.option("--overlay", type=bool, default=False, is_flag=True)
@click.option("--unit", type=str, default='cell')
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
def main(spike_events_path, spike_events_namespace, populations, bin_size, t_variable, t_max, t_min, quantity, graph_type, font_size, overlay, unit, verbose):

    utils.config_logging(verbose)
    logger = utils.get_script_logger(os.path.basename(__file__))
    
    if t_max is None:
        timeRange = None
    else:
        if t_min is None:
            timeRange = [0.0, t_max]
        else:
            timeRange = [t_min, t_max]

    if not populations:
        populations = ['eachPop']

    if unit == 'cell':
        plot.plot_spike_distribution_per_cell (spike_events_path, spike_events_namespace, include=populations, timeVariable=t_variable,
                                               timeRange=timeRange, quantity=quantity, fontSize=font_size, graphType = graph_type,
                                               overlay=overlay, saveFig=True)
    elif unit == 'time':
        plot.plot_spike_distribution_per_time (spike_events_path, spike_events_namespace, include=populations, timeVariable=t_variable,
                                               timeRange=timeRange, timeBinSize=bin_size, quantity=quantity, fontSize=font_size,
                                               overlay=overlay, saveFig=True)

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv)+1):])


    
