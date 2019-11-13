
import click
import copy, random
from mpi4py import MPI
import h5py
from dentate.env import Env
from dentate.stimulus import get_input_cell_config, generate_linear_trajectory, generate_input_spike_trains
from dentate.utils import *
from neuroh5.io import NeuroH5CellAttrGen, append_cell_attributes, read_population_ranges

logger = get_script_logger(os.path.basename(__file__))

context = Struct(**dict(locals()))

sys_excepthook = sys.excepthook
def mpi_excepthook(type, value, traceback):
    sys_excepthook(type, value, traceback)
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)
sys.excepthook = mpi_excepthook

@click.command()
@click.option("--config", required=True, type=str)
@click.option("--config-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='config')
@click.option("--selectivity-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--arena-id", type=str, default='A')
@click.option("--populations", '-p', type=str, multiple=True)
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--cache-size", type=int, default=50)
@click.option("--write-size", type=int, default=10000)
@click.option("--output-path", type=click.Path(file_okay=True, dir_okay=False), default=None)
@click.option("--verbose", '-v', is_flag=True)
@click.option("--dry-run", is_flag=True)
def main(config, config_prefix, selectivity_path, arena_id, populations, io_size, chunk_size,
         value_chunk_size, cache_size, write_size, output_path,  verbose, dry_run):
    """

    :param config: str (.yaml file name)
    :param config_prefix: str (path to dir)
    :param selectivity_path: str (path to file)
    :param arena_id: str
    :param populations: str
    :param io_size: int
    :param chunk_size: int
    :param value_chunk_size: int
    :param cache_size: int
    :param write_size: int
    :param output_path: str (path to file)
    :param verbose: bool
    :param dry_run: bool
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank

    config_logging(verbose)

    env = Env(comm=comm, config_file=config, config_prefix=config_prefix, template_paths=None)
    if io_size == -1:
        io_size = comm.size
    if rank == 0:
        logger.info('%i ranks have been allocated' % comm.size)


    population_ranges = read_population_ranges(selectivity_path, comm)[0]

    if len(populations) == 0:
        populations = ('MC', 'ConMC', 'LPP', 'GC', 'MPP', 'CA3c')

    if arena_id not in env.stimulus_config['Arena']:
        raise RuntimeError('Arena with ID: %s not specified by configuration at file path: %s' %
                           (arena_id, config_prefix + '/' + config))
    arena = env.stimulus_config['Arena'][arena_id]

    valid_selectivity_namespaces = dict()
    if rank == 0:
        for population in populations:
            if population not in population_ranges:
                raise RuntimeError('generate_DG_source_spike_trains: specified population: %s not found in '
                                   'provided selectivity_path: %s' % (population, selectivity_path))
            if population not in env.stimulus_config['Selectivity Type Probabilities']:
                raise RuntimeError('generate_DG_source_spike_trains: selectivity type not specified for '
                                   'population: %s' % population)
            valid_selectivity_namespaces[population] = []
            with h5py.File(selectivity_path, 'r') as selectivity_f:
                for this_namespace in selectivity_f['Populations'][population]:
                    if 'Selectivity %s' % arena_id in this_namespace:
                        valid_selectivity_namespaces[population].append(this_namespace)
                if len(valid_selectivity_namespaces[population]) == 0:
                    raise RuntimeError('generate_DG_source_spike_trains: no selectivity data in arena: %s found '
                                       'for specified population: %s in provided selectivity_path: %s' %
                                       (arena_id, population, selectivity_path))

    valid_selectivity_namespaces = comm.bcast(valid_selectivity_namespaces, root=0)
    selectivity_type_names = dict((val, key) for (key, val) in viewitems(env.selectivity_types))

    write_every = max(1, int(math.floor(write_size / comm.size)))
    
    for population in populations:

        remap_input_selectivity_features(env, arena, population,
                                         selectivity_path, selectivity_type_names,
                                         valid_selectivity_namespaces, 
                                         output_path, comm=comm, io_size=io_size,
                                         write_every=write_every, chunk_size=chunk_size,
                                         value_chunk_size=value_chunk_size,
                                         dry_run=dry_run)
    if interactive and rank == 0:
        context.update(locals())


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv) + 1):],
         standalone_mode=False)
