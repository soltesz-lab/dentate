
import sys, os
from mpi4py import MPI
import click
import dentate
from dentate import utils, plot

script_name = os.path.basename(__file__)

@click.command()
@click.option("--config-path", '-c', required=True, type=click.Path())
@click.option("--input-path", '-p', required=True, type=click.Path())
@click.option("--t-max", type=float)
@click.option("--t-min", type=float)
@click.option("--psd", type=bool, default=False, is_flag=True)
@click.option("--font-size", type=float, default=14)
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
def main(config_path, input_path, t_max, t_min, psd, font_size, verbose):

    utils.config_logging(verbose)

    if t_max is None:
        time_range = None
    else:
        if t_min is None:
            time_range = [0.0, t_max]
        else:
            time_range = [t_min, t_max]

    plot.plot_lfp (config_path, input_path, time_range=time_range, compute_psd=psd, fontSize=font_size, saveFig=True)
    
    

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == script_name, sys.argv)+1):])
