
import sys, os
from mpi4py import MPI
import click
import dentate
from dentate import utils, plot


@click.command()
@click.option("--config-path", '-c', required=True, type=click.Path())
@click.option("--input-path", '-p', required=True, type=click.Path())
@click.option("--t-max", type=float)
@click.option("--t-min", type=float)
@click.option("--font-size", type=float, default=14)
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
def main(config_path, input_path, t_max, t_min, font_size, verbose):

    utils.config_logging(verbose)

    if t_max is None:
        timeRange = None
    else:
        if t_min is None:
            timeRange = [0.0, t_max]
        else:
            timeRange = [t_min, t_max]

    plot.plot_lfp (config_path, input_path, timeRange=timeRange, \
                   fontSize=font_size, saveFig=True)
    
    

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv)+1):])
