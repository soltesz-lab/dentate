
import sys, gc, os
import click
import dentate
from dentate import utils, plot

@click.command()
@click.option("--features-path", '-p', required=True, type=click.Path())
@click.option("--features-namespace", '-n', type=str, default='Vector Stimulus')
@click.option("--coords-path", '-c', required=True, type=click.Path())
@click.option("--distances-namespace", '-d', type=str, default='Arc Distances')
@click.option("--include", '-i', type=str, multiple=True)
@click.option("--normed", type=bool, default=False, is_flag=True)
@click.option("--font-size", type=float, default=14)
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
def main(features_path, coords_path, features_namespace, distances_namespace, include, normed, font_size, verbose):
    plot.plot_stimulus_spatial_rate_map (features_path, coords_path, features_namespace, distances_namespace,
                                         include, normed=normed, fontSize=font_size, saveFig=True, showFig=True, verbose=verbose)

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv)+1):])



    
