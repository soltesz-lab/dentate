
import sys, gc
from mpi4py import MPI
import click
import utils, plot

script_name = 'plot_rate_psd.py'

@click.command()
@click.option("--spike-events-path", '-p', required=True, type=click.Path())
@click.option("--spike-events-namespace", '-n', type=str, default='Spike Events')
@click.option("--populations", '-i', type=str, multiple=True)
@click.option("--max-spikes", type=int, default=int(1e6))
@click.option("--spike-hist-bin", type=float, default=5.0)
@click.option("--t-max", type=float)
@click.option("--t-min", type=float)
@click.option("--font-size", type=float, default=14)
@click.option("--overlay", type=bool, default=False, is_flag=True)
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
def main(spike_events_path, spike_events_namespace, populations, max_spikes, spike_hist_bin, t_max, t_min, font_size, overlay, verbose):
    if t_max is None:
        timeRange = None
    else:
        if t_min is None:
            timeRange = [0.0, t_max]
        else:
            timeRange = [t_min, t_max]
        
    plot.plot_rate_PSD (spike_events_path, spike_events_namespace, populations, timeRange=timeRange, 
                        maxSpikes=max_spikes, binSize=spike_hist_bin, fontSize=font_size,
                        overlay=overlay, saveFig=True, verbose=verbose)
    

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])


    
