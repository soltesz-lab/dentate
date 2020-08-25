import os, sys, click
import dentate
from dentate import plot, utils
from dentate.utils import Context, is_interactive

script_name = os.path.basename(__file__)

context = Context()


@click.command()
@click.option("--input-path", '-p', required=True, type=click.Path())
@click.option("--spike-namespace", type=str, default='Spike Events')
@click.option("--state-namespace", type=str, default='Intracellular soma')
@click.option("--populations", '-i', type=str, multiple=True)
@click.option("--gid", '-g', type=int)
@click.option("--n-trials", '-n', type=int, default=-1)
@click.option("--spike-hist-bin", type=float, default=5.0)
@click.option("--labels", type=str, default='overlay')
@click.option("--lowpass-plot-type", type=str, default='overlay')
@click.option("--state-variable", type=str, default='v')
@click.option("--t-variable", type=str, default='t')
@click.option("--t-max", type=float)
@click.option("--t-min", type=float)
@click.option("--font-size", type=float, default=14)
@click.option("--line-width", type=int, default=1)
@click.option("--verbose", "-v", is_flag=True)
def main(input_path, spike_namespace, state_namespace, populations, gid, n_trials, spike_hist_bin, 
         labels, lowpass_plot_type, state_variable, t_variable, t_max, t_min, font_size, line_width, verbose):
    """

    :param input_path:
    :param spike_namespace:
    :param state_namespace:
    :param populations:
    :param gid:
    :param spike_hist_bin:
    :param state_variable:
    :param t_variable:
    :param t_max:
    :param t_min:
    :param font_size:
    :param verbose: bool
    """
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
        
    plot.plot_network_clamp(input_path, spike_namespace, state_namespace, gid=gid, include=populations,
                            time_range=time_range, time_variable=t_variable, intracellular_variable=state_variable,
                            spike_hist_bin=spike_hist_bin, labels=labels, lowpass_plot_type=lowpass_plot_type,
                            n_trials=n_trials, fontSize=font_size, saveFig=True, lw=line_width)

    if is_interactive:
        context.update(locals())


if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == script_name, sys.argv)+1):],
         standalone_mode=False)
