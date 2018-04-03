
import sys, gc
from mpi4py import MPI
import click
import dentate
from dentate import utils, plot

script_name = 'plot_state.py'

@click.command()
@click.option("--state-path", '-p', required=True, type=click.Path())
@click.option("--state-namespace", '-n', type=str, default='Intracellular Voltage')
@click.option("--populations", '-i', type=str, multiple=True)
@click.option("--max-units", type=int, default=None)
@click.option("--unit-no", type=int, default=None)
@click.option("--t-variable", type=str, default='t')
@click.option("--variable", type=str, default='v')
@click.option("--t-max", type=float)
@click.option("--t-min", type=float)
@click.option("--font-size", type=float, default=14)
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
def main(state_path, state_namespace, populations, max_units, unit_no, t_variable, variable, t_max, t_min, font_size, verbose):
    if t_max is None:
        timeRange = None
    else:
        if t_min is None:
            timeRange = [0.0, t_max]
        else:
            timeRange = [t_min, t_max]

    if not populations:
        populations = ['eachPop']
        
    plot.plot_intracellular_state (state_path, state_namespace, include=populations, timeRange=timeRange,
                                   timeVariable=t_variable, variable=variable,
                                   maxUnits=max_units, unitNo=unit_no,
                                   fontSize=font_size, saveFig=True, verbose=verbose)
    

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])


    
