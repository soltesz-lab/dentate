
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
@click.option("--spike-hist-bin", type=float, default=1.0)
@click.option("--sliding-window", type=int, default=256)
@click.option("--overlap", type=float, default=0.5)
@click.option("--kernel-size", type=float, default=10.)
@click.option("--smooth", type=int)
@click.option("--t-max", type=float)
@click.option("--t-min", type=float)
@click.option("--font-size", type=float, default=14)
@click.option("--overlay", type=bool, default=False, is_flag=True)
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
def main(spike_events_path, spike_events_namespace, populations, spike_hist_bin, sliding_window, overlap, kernel_size, smooth, t_max, t_min, font_size, overlay, verbose):

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

    plot.plot_rate_PSD (spike_events_path, spike_events_namespace, populations, time_range=time_range, 
                        bin_size=spike_hist_bin, sliding_window=sliding_window, overlap=overlap,
                        kernel_size=kernel_size, smooth=smooth, fontSize=font_size, overlay=overlay, saveFig=True)
    

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == script_name, sys.argv)+1):])


    
