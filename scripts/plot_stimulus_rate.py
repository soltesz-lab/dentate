import sys, os
import click
import dentate.utils as utils
import dentate.plot as plot


@click.command()
@click.option("--features-path", '-p', required=True, type=click.Path())
@click.option("--features-namespace", '-n', type=str, default='Vector Stimulus')
@click.option("--trajectory-id", '-t', type=int, default=0)
@click.option("--include", '-i', type=str, multiple=True)
@click.option("--font-size", type=float, default=14)
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
@click.option("--show-fig", is_flag=True)
@click.option("--save-fig", is_flag=True)
def main(features_path, features_namespace, trajectory_id, include, font_size, verbose, show_fig, save_fig):
    """
    
    :param features_path: 
    :param features_namespace: 
    :param trajectory_id: 
    :param include: 
    :param module: 
    :param font_size: 
    :param verbose: 
    :param show_fig: 
    :param save_fig:  
    """
    utils.config_logging(verbose)
    fig_filename = None
    for population in include:
        if save_fig:
            fig_filename = 'population-%s.png' % population
        plot.plot_stimulus_rate(features_path, features_namespace, population, trajectory_id=trajectory_id,
                                fontSize=font_size, showFig=show_fig, saveFig=fig_filename)


if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv)+1):], 
         standalone_mode=False)



    
