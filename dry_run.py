"""
Dentate Gyrus model initialization script
"""
__author__ = 'Ivan Raikov, Aaron D. Milstein, Grace Ng'
from dentate.main import *
from nested.utils import Context


context = Context()
logging.basicConfig()

script_name = os.path.basename(__file__)
logger = logging.getLogger(script_name)


@click.command()
@click.option("--config-file", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='config/Small_Scale_Control_log_normal_weights.yaml')
@click.option("--template-paths", type=str, default='../dgc/Mateos-Aparicio2014:templates')
@click.option("--hoc-lib-path", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='.')
@click.option("--dataset-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='datasets')  # '/mnt/s')
@click.option('--verbose', '-v', is_flag=True)
def main(config_file, template_paths, hoc_lib_path, dataset_prefix, verbose):
    """
    :param config_file: str; model configuration file
    :param template_paths: str; colon-separated list of paths to directories containing hoc cell templates
    :param hoc_lib_path: str; path to directory containing required hoc libraries
    :param dataset_prefix: str; path to directory containing required neuroh5 data files
    :param verbose: bool; print verbose diagnostic messages while constructing the network
    """
    if verbose:
        logger.setLevel(logging.INFO)
    comm = MPI.COMM_WORLD

    np.seterr(all='raise')
    env = Env(comm, config_file, template_paths, hoc_lib_path, dataset_prefix, verbose=verbose)
    context.update(locals())
    init(env)


if __name__ == '__main__':
    main(args=sys.argv[(sys.argv.index(script_name)+1):])
