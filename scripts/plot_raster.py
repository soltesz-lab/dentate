
import sys, gc
from mpi4py import MPI
import click
import utils, plot

script_name = 'plot_raster.py'

@click.command()
@click.option("--spike-events-path", '-p', required=True, type=click.Path())
@click.option("--spike-events-namespace", '-n', type=str, default='Spike Events')
def main(spike_events_path, spike_events_namespace):
    plot.plot_raster (spike_events_path, spike_events_namespace, popRates=True, spikeHist='subplot', saveFig=True, spikeHistBin=1)
    

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])

