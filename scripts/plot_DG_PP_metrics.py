import os, sys, click
from dentate.plot import plot_PP_metrics
from dentate.utils import list_find

script_name = os.path.basename(__file__)


@click.command()
@click.option("--coords-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=True))
@click.option("--features-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=True))
@click.option("--distances-namespace", required=True, type=str)
@click.option("--population", required=True, type=str)
@click.option("--cell-type", required=True, type=str)
@click.option("--normed", type=int, default=0)
@click.option("--bin-size", type=float, default=400.)
@click.option("--save-fig", type=int, default=0)
@click.option("--show-fig", type=int, default=0)
def main(coords_path, features_path, distances_namespace, population, cell_type, normed, bin_size, save_fig, show_fig):
    """
    
    :param coords_path: 
    :param features_path: 
    :param distances_namespace: 
    :param population: 
    :param cell_type: 
    :param normed: 
    :param bin_size
    :param save_fig: 
    :param show_fig:  
    """
    plot_PP_metrics(coords_path, features_path, distances_namespace, population=population, cellType=cell_type, 
                    binSize=bin_size, metric="spacing", normed=normed, graphType="histogram2d", saveFig=save_fig,
                    showFig=show_fig)
    plot_PP_metrics(coords_path, features_path, distances_namespace, population=population, cellType=cell_type,
                    binSize=bin_size, metric="num-fields", normed=normed, graphType="histogram2d", saveFig=save_fig,
                    showFig=show_fig)
    plot_PP_metrics(coords_path, features_path, distances_namespace, population=population, cellType=cell_type,
                    binSize=bin_size, metric="orientation", normed=normed, graphType="histogram2d", saveFig=save_fig,
                    showFig=show_fig)


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(script_name) != -1, sys.argv)+1):], standalone_mode=False)
