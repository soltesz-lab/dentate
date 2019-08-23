
import os, sys, gc
import click
import dentate
from dentate import plot, utils

script_name = os.path.basename(__file__)

@click.command()
@click.option("--config-path", '-c', required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--spike-events-path", '-p', required=True, type=click.Path())
@click.option("--spike-events-namespace", '-n', type=str, default='Spike Events')
@click.option("--coords-path", '-d', required=True, type=click.Path())
@click.option("--coords-namespace", '-d', type=str, default='Coordinates')
@click.option("--populations", '-i', type=str, multiple=True)
@click.option("--compute-rates", type=bool, default=False, is_flag=True)
@click.option("--t-variable", type=str, default='t')
@click.option("--t-max", type=float)
@click.option("--t-min", type=float)
@click.option("--t-step", type=float, default=5.0)
@click.option("--font-size", type=float, default=14)
@click.option("--rotate-anim", type=bool, default=False, is_flag=True)
@click.option("--marker-scale", type=float, default=10.)
@click.option("--save-fig", type=bool, default=False, is_flag=True)
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
def main(config_path, spike_events_path, spike_events_namespace, coords_path, coords_namespace, populations, compute_rates, t_variable, t_max, t_min, t_step, font_size, rotate_anim, marker_scale, save_fig, verbose):

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
        
    plot.plot_spikes_in_volume (config_path, populations, coords_path, coords_namespace, spike_events_path, spike_events_namespace, \
                                time_range=time_range, time_variable=t_variable, time_step=t_step, compute_rates=compute_rates, \
                                fontSize=font_size, marker_scale=marker_scale, rotate_anim=rotate_anim, saveFig=save_fig)
    

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == script_name, sys.argv)+1):])
