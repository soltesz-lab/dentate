
import sys, gc
from mpi4py import MPI
import click
import utils, plot

script_name = 'plot_raster.py'

@click.command()
@click.option("--spike-events-path", '-p', required=True, type=click.Path())
@click.option("--spike-events-namespace", '-n', type=str, default='Spike Events')
@click.option("--max-spikes", type=int, default=int(1e6))
@click.option("--spike_hist_bin", type=float, default=5.0)
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
def main(spike_events_path, spike_events_namespace, max_spikes, spike_hist_bin, verbose):
    plot.plot_raster (spike_events_path, spike_events_namespace, popRates=True, spikeHist='subplot', maxSpikes=max_spikes, spikeHistBin=spike_hist_bin, saveFig=True, verbose=verbose)
    

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])

