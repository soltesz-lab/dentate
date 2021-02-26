
import gc, sys, click
import dentate
from dentate import plot
from dentate import utils
from mpi4py import MPI

script_name = os.path.basename(__file__)


@click.command()
@click.option("--config", required=True, type=str)
@click.option("--config-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--forest-path", '-p', required=True, type=click.Path())
@click.option("--population", type=str)
@click.option("--metric-namespace", type=str, default='Tree Measurements')
@click.option("--metric", type=str, default='dendrite_mean_diam_hist')
@click.option("--font-size", type=float, default=14)
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
def main(config, config_prefix, forest_path, population, metric_namespace, metric, font_size, verbose):

    env = Env(config_file=config, config_prefix=config_prefix)

    plot.plot_tree_metric_histogram (env, forest_path, population, metric_namespace,
                                     metric=metric, fontSize=font_size, saveFig=True,
                                     verbose=verbose)

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])
