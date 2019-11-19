
import click
from mpi4py import MPI
import h5py
from neuroh5.io import NeuroH5CellAttrGen, append_cell_attributes, read_population_ranges
from dentate.env import Env
from dentate.stimulus import InputSelectivityConfig, choose_input_selectivity_type, get_2D_arena_spatial_mesh, \
    get_input_cell_config, generate_input_selectivity_features
from dentate.utils import *

logger = get_script_logger(os.path.basename(__file__))

context = Struct(**dict(locals()))

sys_excepthook = sys.excepthook
def mpi_excepthook(type, value, traceback):
    sys_excepthook(type, value, traceback)
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)
#sys.excepthook = mpi_excepthook


def debug_callback(context):
    from dentate.plot import plot_2D_rate_map
    fig_title = '%s %s cell %i' % (context.population, context.this_selectivity_type_name, context.gid)
    fig_options = copy.copy(context.fig_options)
    if context.save_fig is not None:
        fig_options.saveFig = '%s %s' % (context.save_fig, fig_title)
    plot_2D_rate_map(x=context.arena_x_mesh, y=context.arena_y_mesh, rate_map=context.rate_map,
                     peak_rate = context.env.stimulus_config['Peak Rate'][context.population][context.this_selectivity_type],
                     title='%s\nNormalized cell position: %.3f' % (fig_title, context.norm_u_arc_distance),
                     **fig_options())


@click.command()
@click.option("--config", required=True, type=str)
@click.option("--config-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='config')
@click.option("--coords-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--distances-namespace", '-n', type=str, default='Arc Distances')
@click.option("--output-path", type=click.Path(file_okay=True, dir_okay=False), default=None)
@click.option("--arena-id", type=str, default='A')
@click.option("--populations", '-p', type=str, multiple=True)
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--cache-size", type=int, default=50)
@click.option("--write-size", type=int, default=10000)
@click.option("--verbose", '-v', is_flag=True)
@click.option("--gather", is_flag=True)
@click.option("--interactive", is_flag=True)
@click.option("--debug", is_flag=True)
@click.option("--plot", is_flag=True)
@click.option("--show-fig", is_flag=True)
@click.option("--save-fig", required=False, type=str, default=None)
@click.option("--save-fig-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default=None)
@click.option("--font-size", type=float, default=14)
@click.option("--fig-format", required=False, type=str, default='svg')
@click.option("--dry-run", is_flag=True)
def main(config, config_prefix, coords_path, distances_namespace, output_path, arena_id, populations, io_size,
         chunk_size, value_chunk_size, cache_size, write_size, verbose, gather, interactive, debug, plot, show_fig,
         save_fig, save_fig_dir, font_size, fig_format, dry_run):
    """

    :param config: str (.yaml file name)
    :param config_prefix: str (path to dir)
    :param coords_path: str (path to file)
    :param distances_namespace: str
    :param output_path: str
    :param arena_id: str
    :param populations: tuple of str
    :param io_size: int
    :param chunk_size: int
    :param value_chunk_size: int
    :param cache_size: int
    :param write_size: int
    :param verbose: bool
    :param gather: bool; whether to gather population attributes to rank 0 for interactive analysis or plotting
    :param interactive: bool
    :param debug: bool
    :param plot: bool
    :param show_fig: bool
    :param save_fig: str (base file name)
    :param save_fig_dir:  str (path to dir)
    :param font_size: float
    :param fig_format: str
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

    if save_fig is not None:
        plot = True

    if plot:
        import matplotlib.pyplot as plt
        from dentate.plot import plot_2D_rate_map, default_fig_options, save_figure, clean_axes

        fig_options = copy.copy(default_fig_options)
        fig_options.saveFigDir = save_fig_dir
        fig_options.fontSize = font_size
        fig_options.figFormat = fig_format
        fig_options.showFig = show_fig

    if save_fig is not None:
        save_fig = '%s %s' % (save_fig, arena_id)
        fig_options.saveFig = save_fig

    if not dry_run and rank == 0:
        if output_path is None:
            raise RuntimeError('generate_DG_input_selectivity_features: missing output_path')
        if not os.path.isfile(output_path):
            input_file = h5py.File(coords_path, 'r')
            output_file = h5py.File(output_path, 'w')
            input_file.copy('/H5Types', output_file)
            input_file.close()
            output_file.close()
    comm.barrier()
    population_ranges = read_population_ranges(coords_path, comm)[0]

    if len(populations) == 0:
        populations = sorted(population_ranges.keys())

    reference_u_arc_distance_bounds = None
    if rank == 0:
        for population in populations:
            if population not in population_ranges:
                raise RuntimeError('generate_DG_input_selectivity_features: specified population: %s not found in '
                                   'provided coords_path: %s' % (population, coords_path))
            if population not in env.stimulus_config['Selectivity Type Probabilities']:
                raise RuntimeError('generate_DG_input_selectivity_features: selectivity type not specified for '
                                   'population: %s' % population)
            with h5py.File(coords_path, 'r') as coords_f:
                pop_size = population_ranges[population][1]
                unique_gid_count = len(set(
                    coords_f['Populations'][population][distances_namespace]['U Distance']['Cell Index'][:]))
                if pop_size != unique_gid_count:
                    raise RuntimeError('generate_DG_input_selectivity_features: only %i/%i unique cell indexes found '
                                       'for specified population: %s in provided coords_path: %s' %
                                       (unique_gid_count, pop_size, population, coords_path))

    if arena_id not in env.stimulus_config['Arena']:
        raise RuntimeError('Arena with ID: %s not specified by configuration at file path: %s' %
                           (arena_id, config_prefix + '/' + config))

    selectivity_type_names = dict([ (val, key) for (key, val) in viewitems(env.selectivity_types) ])
    selectivity_type_namespaces = dict()
    for this_selectivity_type in selectivity_type_names:
        this_selectivity_type_name = selectivity_type_names[this_selectivity_type]
        chars = list(this_selectivity_type_name)
        chars[0] = chars[0].upper()
        selectivity_type_namespaces[this_selectivity_type_name] = ''.join(chars) + ' Selectivity %s' % arena_id

    arena = env.stimulus_config['Arena'][arena_id]

    local_random = np.random.RandomState()
    selectivity_seed_offset = int(env.modelConfig['Random Seeds']['Input Selectivity'])
    local_random.seed(selectivity_seed_offset - 1)

    selectivity_config = InputSelectivityConfig(env.stimulus_config, local_random)
    if plot and rank == 0:
        selectivity_config.plot_module_probabilities(**fig_options())

    if (debug or interactive) and rank == 0:
        context.update(dict(locals()))

    if gather:
        pop_distances = {}
        rate_map_sum = {}

    write_every = max(1, int(math.floor(write_size / comm.size)))
    for population in populations:

        this_pop_norm_distances, this_rate_map_sum = \
          generate_input_selectivity_features(env, population, arena, 
                                              selectivity_config, selectivity_type_names,
                                              selectivity_type_namespaces,
                                              coords_path, output_path, 
                                              distances_namespace=distances_namespace,
                                              comm=comm, io_size=io_size, write_every=write_every,
                                              chunk_size=chunk_size, value_chunk_size=value_chunk_size,
                                              dry_run=dry_run, debug= (debug_callback, context) if debug else False)
                                              
                                              
        if gather:
            pop_distances[population] = this_pop_norm_distances
            rate_map_sum[population] = this_rate_map_sum
            
                
    if gather:
        arena_x_mesh, arena_y_mesh = None, None
        if rank == 0:
            arena_x_mesh, arena_y_mesh = \
              get_2D_arena_spatial_mesh(arena=arena, spatial_resolution=env.stimulus_config['Spatial Resolution'])
        arena_x_mesh = comm.bcast(arena_x_mesh, root=0)
        arena_y_mesh = comm.bcast(arena_y_mesh, root=0)

        pop_norm_distances = comm.gather(pop_norm_distances, root=0)
        rate_map_sum = dict([(key, dict(val.items())) for key, val in viewitems(rate_map_sum)])
        rate_map_sum = comm.gather(rate_map_sum, root=0)
        if rank == 0:
            merged_pop_norm_distances = defaultdict(list)
            for each_pop_norm_distances in pop_norm_distances:
                for population in each_pop_norm_distances:
                    merged_pop_norm_distances[population].extend(each_pop_norm_distances[population])
            merged_rate_map_sum = defaultdict(lambda: defaultdict(lambda: np.zeros_like(arena_x_mesh)))
            for each_rate_map_sum in rate_map_sum:
                for population in each_rate_map_sum:
                    for selectivity_type_name in each_rate_map_sum[population]:
                        merged_rate_map_sum[population][selectivity_type_name] = \
                            np.add(merged_rate_map_sum[population][selectivity_type_name],
                                   each_rate_map_sum[population][selectivity_type_name])
            if plot:
                for population in merged_pop_norm_distances:
                    hist, edges = np.histogram(merged_pop_norm_distances[population], bins=100)
                    fig, axes = plt.subplots(1)
                    axes.plot(edges[1:], hist)
                    axes.set_title('Population: %s' % population)
                    axes.set_xlabel('Normalized cell position')
                    axes.set_ylabel('Cell count')
                    clean_axes(axes)
                    if save_fig is not None:
                        save_figure('%s normalized distances histogram' % save_fig, fig=fig, **fig_options())
                    if fig_options.showFig:
                        fig.show()
                for population in merged_rate_map_sum:
                    for selectivity_type_name in merged_rate_map_sum[population]:
                        fig_title = '%s %s summed rate maps' % (population, this_selectivity_type_name)
                        if save_fig is not None:
                            fig_options.saveFig = '%s %s' % (save_fig, fig_title)
                        plot_2D_rate_map(x=arena_x_mesh, y=arena_y_mesh,
                                         rate_map=merged_rate_map_sum[population][selectivity_type_name],
                                         title='Summed rate maps\n%s %s cells' %
                                               (population, selectivity_type_name), **fig_options())

    if interactive and rank == 0:
        context.update(locals())


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv) + 1):],
         standalone_mode=False)
