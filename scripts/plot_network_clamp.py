
import os
import sys

import click
import dentate
from dentate import plot
from dentate import utils

script_name = os.path.basename(__file__)

@click.command()
@click.option("--input-path", '-p', required=True, type=click.Path())
@click.option("--spike-namespace", type=str, default='Spike Events')
@click.option("--state-namespace", type=str, default='Intracellular soma')
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
        time_range = None
    else:
        if t_min is None:
            time_range = [0.0, t_max]
        else:
            time_range = [t_min, t_max]

    if not populations:
        populations = ['eachPop']
        
    plot.plot_network_clamp (input_path, spike_namespace, state_namespace, unit_no=unit_no, include=populations,
                             time_range=time_range, time_variable=t_variable, intracellular_variable=state_variable,
                             spike_hist='subplot', spike_hist_bin=spike_hist_bin, fontSize=font_size, saveFig=True)
    

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == script_name, sys.argv)+1):])
