
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
@click.option("--bin-size", type=float, default=1.0)
@click.option("--window-size", type=int, default=1024)
@click.option("--overlap", type=float, default=0.5)
@click.option("--smooth", type=int)
@click.option("--t-max", type=float)
@click.option("--t-min", type=float)
@click.option("--font-size", type=float, default=14)
@click.option("--overlay", type=bool, default=False, is_flag=True)
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
def main(spike_events_path, spike_events_namespace, populations, bin_size, window_size, overlap, smooth, t_max, t_min, font_size, overlay, verbose):

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

    plot.plot_spike_PSD (spike_events_path, spike_events_namespace, populations, time_range=time_range, 
                         bin_size=bin_size, window_size=window_size, smooth=smooth, overlap=overlap,
                        fontSize=font_size, overlay=overlay, saveFig=True)
    

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == script_name, sys.argv)+1):])


    
