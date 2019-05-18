
import sys, gc, os
import click
import dentate
from dentate import plot, utils

script_name = os.path.basename(__file__)

@click.command()
@click.option("--spike-events-path", '-p', required=True, type=click.Path())
@click.option("--spike-events-namespace", '-n', type=str, default='Spike Events')
@click.option("--populations", '-i', type=str, multiple=True)
@click.option("--t-variable", type=str, default='t')
@click.option("--t-max", type=float)
@click.option("--t-min", type=float)
@click.option("--bin-size", type=float, default=1.)
@click.option("--meansub", type=bool, default=False, is_flag=True)
@click.option("--progress", type=bool, default=False, is_flag=True)
@click.option("--font-size", type=float, default=14)
@click.option("--save-format", type=str, default='png')
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
def main(spike_events_path, spike_events_namespace, populations, t_variable, t_max, t_min, bin_size, meansub, progress, font_size, save_format, verbose):

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
        
    plot.plot_spike_rates (spike_events_path, spike_events_namespace, include=populations, time_range=time_range, time_variable=t_variable, meansub=meansub,
                           bin_size=bin_size, fontSize=font_size, saveFig=True, figFormat=save_format, progress=progress)
    

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == script_name, sys.argv)+1):])


    
