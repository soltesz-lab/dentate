
import gc, os, sys
import click
import dentate
from dentate import plot
from dentate import utils

script_name = os.path.basename(__file__)

@click.command()
@click.option("--config-path", '-c', type=click.Path())
@click.option("--spike-events-path", '-p', required=True, type=click.Path())
@click.option("--spike-events-namespace", '-n', type=str, default='Spike Events')
@click.option("--populations", '-i', type=str, multiple=True)
@click.option("--arena-id", '-a', type=str, default='Default')
@click.option("--trajectory-id", '-t', type=str, default='Default')
@click.option("--target-rate-map-path", required=False, type=click.Path()9
@click.option("--target-rate-map-namespace", type=str, required=False, default='Input Spikes')
@click.option("--include-artificial/--exclude-artificial", type=bool, default=True, is_flag=True)
@click.option("--max-units", '-m', type=int)
@click.option("--t-variable", type=str, default='t')
@click.option("--t-max", type=float)
@click.option("--t-min", type=float)
@click.option("--threshold", type=float, default=0.01)
@click.option("--bin-size", type=float, default=1.)
@click.option("--meansub", type=bool, default=False, is_flag=True)
@click.option("--graph-type", type=str, default='raster2d')
@click.option("--progress", type=bool, default=False, is_flag=True)
@click.option("--font-size", type=float, default=14)
@click.option("--fig-size", type=(float,float), default=(15,8))
@click.option("--save-format", type=str, default='png')
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)


def main(config_path, spike_events_path, spike_events_namespace, populations, arena_id, trajectory_id, target_rate_map_path, target_rate_map_namespace, include_artificial, max_units, t_variable, t_max, t_min, threshold, bin_size, meansub, graph_type, progress, fig_size, font_size, save_format, verbose):

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
        
    plot.plot_spike_rates_with_features (spike_events_path, spike_events_namespace,
                                         arena_id=arena_id, trajectory_id=trajectory_id,
                                         target_rate_map_path=target_rate_map_path,
                                         target_rate_map_namespace=target_rate_map_namespace,
                                         config_path=config_path, include=populations,
                                         max_units=max_units, time_range=time_range, time_variable=t_variable,
                                         threshold=threshold, meansub=meansub, bin_size=bin_size,
                                         graph_type=graph_type, fontSize=font_size, figSize=fig_size,
                                         saveFig=True, figFormat=save_format, progress=progress,
                                         include_artificial=include_artificial)
    

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == script_name, sys.argv)+1):])
