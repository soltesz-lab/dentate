
import sys, os 
import click
import dentate
from dentate import utils, plot


@click.command()
@click.option("--input-path", '-p', required=True, type=click.Path())
@click.option("--spike-namespace", type=str, default='Spike Events')
@click.option("--state-namespace", type=str, default='Intracellular Voltage')
@click.option("--populations", '-i', type=str, multiple=True)
@click.option("--unit-no", '-u', type=int, required=True)
@click.option("--spike-hist-bin", type=float, default=5.0)
@click.option("--state-variable", type=str, default='v')
@click.option("--t-variable", type=str, default='t')
@click.option("--t-max", type=float)
@click.option("--t-min", type=float)
@click.option("--font-size", type=float, default=14)
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
def main(input_path, spike_namespace, state_namespace, populations, unit_no, spike_hist_bin, state_variable, t_variable, t_max, t_min, font_size, verbose):

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
        
    plot.plot_network_clamp (input_path, spike_namespace, state_namespace, unitNo=unit_no, include=populations, timeRange=timeRange, timeVariable=t_variable, intracellularVariable=state_variable, spikeHist='subplot', spikeHistBin=spike_hist_bin, fontSize=font_size, saveFig=True)
    

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv)+1):])
