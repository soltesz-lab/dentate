
import sys, gc, os
import click
import dentate
from dentate import utils, plot


@click.command()
@click.option("--spike-events-path", '-p', required=True, type=click.Path())
@click.option("--spike-events-namespace", '-n', type=str, default='Spike Events')
@click.option("--coords-path", '-c', required=True, type=click.Path())
@click.option("--distances-namespace", '-d', type=str, default='Arc Distances')
@click.option("--populations", '-i', type=str, multiple=True)
@click.option("--max-spikes", type=int, default=int(1e6))
@click.option("--t-variable", type=str, default='t')
@click.option("--t-max", type=float)
@click.option("--t-min", type=float)
@click.option("--font-size", type=float, default=14)
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
def main(spike_events_path, spike_events_namespace, coords_path, distances_namespace, populations, max_spikes, t_variable, t_max, t_min, font_size, verbose):

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
        
    plot.plot_spatial_spike_raster (spike_events_path, spike_events_namespace, coords_path, distances_namespace, include=populations, timeRange=timeRange, timeVariable=t_variable, maxSpikes=max_spikes, spikeHistBin=spike_hist_bin, fontSize=font_size, saveFig=True)
    

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv)+1):])
