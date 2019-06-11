"""
MEC is divided into discrete modules with distinct grid spacing and field width. Here we assume grid cells
sample uniformly from 10 modules with spacing that increases exponentially from 40 cm to 8 m. While organized
dorsal-ventrally, there is no organization in the transverse or septo-temporal extent of their projections to DG.
CA3 and LEC are assumed to exhibit place fields. Their field width varies septal-temporally. Here we assume a
continuous exponential gradient of field widths, with the same parameters as those controlling MEC grid width.
"""
from mpi4py import MPI
from neuroh5.io import NeuroH5CellAttrGen, append_cell_attributes, read_population_ranges
import h5py
from dentate.utils import *
from dentate.env import Env
from dentate.stimulus import SelectivityModuleConfig, get_input_cell_config, get_2D_arena_spatial_mesh, \
    choose_input_selectivity_type
from dentate.plot import plot_2D_rate_map
import click


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
@click.option("--spikes-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--spikes-namespace", '-n', type=str, default='Spike Events')
@click.option("--spike-train-attr-name", type=str, default='Spike Train')
@click.option("--output-path", type=click.Path(file_okay=True, dir_okay=False), default=None)
@click.option("--arena-id", type=str, default='A')
@click.option("--trajectory-id", type=str, default='Diag')
@click.option("--populations", '-p', type=str, multiple=True, default=None)
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
@click.option("--dry-run", is_flag=True)
def main(config, config_prefix, coords_path, output_path, distances_namespace, arena_id, populations, io_size,
         chunk_size, value_chunk_size, cache_size, write_size, verbose, gather, interactive, debug, plot, dry_run):
    """

    :param config: str (.yaml file name)
    :param config_prefix: str (path to dir)
    :param coords_path: str (path to file)
    :param output_path: str
    :param distances_namespace: str
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

    if plot:
        import matplotlib.pyplot as plt
        from dentate.plot import clean_axes

    if not dry_run and rank == 0:
        if output_path is None:
            raise RuntimeError('generate_DG_input_selectivity_parameters: missing output_path')
        if not os.path.isfile(output_path):
            input_file = h5py.File(coords_path, 'r')
            output_file = h5py.File(output_path, 'w')
            input_file.copy('/H5Types', output_file)
            input_file.close()
            output_file.close()
    comm.barrier()
    population_ranges = read_population_ranges(coords_path, comm)[0]

    if populations is None:
        populations = ['MC', 'ConMC', 'LPP', 'GC', 'MPP', 'CA3c']

    reference_u_arc_distance_bounds = None
    if rank == 0:
        for population in populations:
            if population not in population_ranges:
                raise RuntimeError('generate_DG_input_selectivity_parameters: specified population: %s not found in '
                                   'provided coords_path: %s' % (population, coords_path))
            if population not in env.input_config['Selectivity Type Probabilities']:
                raise RuntimeError('generate_DG_input_selectivity_parameters: selectivity type not specified for '
                                   'population: %s' % population)
            with h5py.File(coords_path, 'r') as coords_f:
                pop_size = population_ranges[population][1]
                unique_gid_count = len(set(
                    coords_f['Populations'][population][distances_namespace]['U Distance']['Cell Index'][:]))
                if pop_size != unique_gid_count:
                    raise RuntimeError('generate_DG_input_selectivity_parameters: only %i/%i unique cell indexes found '
                                       'for specified population: %s in provided coords_path: %s' %
                                       (unique_gid_count, pop_size, population, coords_path))
                if reference_u_arc_distance_bounds is None:
                    try:
                        reference_u_arc_distance_bounds = \
                            coords_f['Populations'][population][distances_namespace].attrs['Reference U Min'], \
                            coords_f['Populations'][population][distances_namespace].attrs['Reference U Max']
                    except Exception:
                        raise RuntimeError('generate_DG_input_selectivity_parameters: problem locating attributes '
                                           'containing reference bounds in namespace: %s for population: %s from '
                                           'coords_path: %s' % (distances_namespace, population, coords_path))
    reference_u_arc_distance_bounds = comm.bcast(reference_u_arc_distance_bounds, root=0)

    if arena_id not in env.input_config['Arena']:
        raise RuntimeError('Arena with ID: %s not specified by configuration at file path: %s' %
                           (arena_id, config_prefix + '/' + config))

    selectivity_type_names = dict((val, key) for (key, val) in viewitems(env.selectivity_types))
    selectivity_type_namespaces = dict()
    for this_selectivity_type in selectivity_type_names:
        this_selectivity_type_name = selectivity_type_names[this_selectivity_type]
        chars = list(this_selectivity_type_name)
        chars[0] = chars[0].upper()
        selectivity_type_namespaces[this_selectivity_type_name] = ''.join(chars) + ' Selectivity %s' % arena_id

    arena = env.input_config['Arena'][arena_id]
    arena_x_mesh, arena_y_mesh = None, None
    if rank == 0:
        arena_x_mesh, arena_y_mesh = \
            get_2D_arena_spatial_mesh(arena=arena, spatial_resolution=env.input_config['Spatial Resolution'])
    arena_x_mesh = comm.bcast(arena_x_mesh, root=0)
    arena_y_mesh = comm.bcast(arena_y_mesh, root=0)

    local_random = np.random.RandomState()
    selectivity_seed_offset = int(env.modelConfig['Random Seeds']['Input Spatial Selectivity'])
    local_random.seed(selectivity_seed_offset - 1)

    module_config = SelectivityModuleConfig(env.input_config, local_random)
    if plot and rank == 0:
        module_config.plot_module_probabilities()

    context.update(locals())

    if gather:
        pop_distances = defaultdict(list)
        rate_map_sum = defaultdict(lambda: defaultdict(lambda: np.zeros_like(arena_x_mesh)))

    write_every = max(1, int(math.floor(old_div(write_size, comm.size))))
    for population in populations:
        start_time = time.time()
        gid_count = defaultdict(lambda: 0)
        distances_attr_gen = \
            NeuroH5CellAttrGen(coords_path, population, namespace=distances_namespace, comm=comm, io_size=io_size,
                               cache_size=cache_size)
        selectivity_attr_dict = dict((key, dict()) for key in env.selectivity_types)
        for iter_count, (gid, distances_attr_dict) in enumerate(distances_attr_gen):
            if gid is not None:
                u_arc_distance = distances_attr_dict['U Distance'][0]
                norm_u_arc_distance = old_div((u_arc_distance - reference_u_arc_distance_bounds[0]), \
                                      (reference_u_arc_distance_bounds[1] - reference_u_arc_distance_bounds[0]))

                local_random.seed(int(selectivity_seed_offset + gid))
                this_selectivity_type = \
                    choose_input_selectivity_type(p=env.input_config['Selectivity Type Probabilities'][population],
                                                  local_random=local_random)
                cell_config = get_input_cell_config(selectivity_type=this_selectivity_type,
                                                    selectivity_type_names=selectivity_type_names,
                                                    population=population, input_config=env.input_config, arena=arena,
                                                    module_config=module_config, distance=norm_u_arc_distance,
                                                    local_random=local_random)
                this_selectivity_type_name = selectivity_type_names[this_selectivity_type]
                selectivity_attr_dict[this_selectivity_type_name][gid] = cell_config.get_selectivity_attr_dict()
                rate_map = cell_config.get_rate_map(x=arena_x_mesh, y=arena_y_mesh)
                if debug and plot and rank == 0:
                    plot_2D_rate_map(x=arena_x_mesh, y=arena_y_mesh, rate_map=rate_map,
                                     peak_rate = env.input_config['Peak Rate'][population][this_selectivity_type],
                                     title='%s %s cell %i\nNormalized cell position: %.3f' %
                                           (population, this_selectivity_type_name, gid, norm_u_arc_distance))
                selectivity_attr_dict[this_selectivity_type_name][gid]['Arena Rate Map'] = \
                    np.asarray(rate_map, dtype='float32').reshape(-1,)
                gid_count[this_selectivity_type_name] += 1
                if gather:
                    pop_distances[population].append(norm_u_arc_distance)
                    rate_map_sum[population][this_selectivity_type_name] = \
                        np.add(rate_map_sum[population][this_selectivity_type_name], rate_map)

            gid_count_dict = dict(list(gid_count.items()))
            gid_count_dict = comm.gather(gid_count_dict, root=0)
            if rank == 0:
                merged_gid_count = defaultdict(lambda: 0)
                for each_gid_count in gid_count_dict:
                    for selectivity_type_name in each_gid_count:
                        merged_gid_count[selectivity_type_name] += each_gid_count[selectivity_type_name]
                total_gid_count = np.sum(list(merged_gid_count.values()))

            if (iter_count > 0 and iter_count % write_every == 0) or (debug and iter_count == 10):
                if verbose and rank == 0:
                    logger.info('processed %i %s cells' % (total_gid_count, population))
                if not dry_run:
                    for selectivity_type_name in selectivity_attr_dict:
                        selectivity_type_namespace = selectivity_type_namespaces[selectivity_type_name]
                        append_cell_attributes(output_path, population, selectivity_attr_dict[selectivity_type_name],
                                               namespace=selectivity_type_namespace, comm=comm, io_size=io_size,
                                               chunk_size=chunk_size, value_chunk_size=value_chunk_size)
                del selectivity_attr_dict
                selectivity_attr_dict = dict((key, dict()) for key in env.selectivity_types)
            sys.stdout.flush()
            comm.barrier()
            if debug and iter_count == 10:
                break
        if not dry_run:
            for selectivity_type_name in selectivity_attr_dict:
                selectivity_type_namespace = selectivity_type_namespaces[selectivity_type_name]
                append_cell_attributes(output_path, population, selectivity_attr_dict[selectivity_type_name],
                                       namespace=selectivity_type_namespace, comm=comm, io_size=io_size,
                                       chunk_size=chunk_size, value_chunk_size=value_chunk_size)
        del selectivity_attr_dict
        if verbose and rank == 0:
            for selectivity_type_name in merged_gid_count:
                logger.info('processed %i/%i %s %s cells in %.2f s' %
                      (merged_gid_count[selectivity_type_name], total_gid_count, population, selectivity_type_name,
                       (time.time() - start_time)))
    if gather:
        pop_distances = dict(pop_distances.items())
        pop_distances = comm.gather(pop_distances, root=0)
        rate_map_sum = dict([(key, dict(val.items())) for key, val in viewitems(rate_map_sum)])
        rate_map_sum = comm.gather(rate_map_sum, root=0)
        if rank == 0:
            merged_pop_distances = defaultdict(list)
            for each_pop_distances in pop_distances:
                for population in each_pop_distances:
                    merged_pop_distances[population].extend(each_pop_distances[population])
            merged_rate_map_sum = defaultdict(lambda: defaultdict(lambda: np.zeros_like(arena_x_mesh)))
            for each_rate_map_sum in rate_map_sum:
                for population in each_rate_map_sum:
                    for selectivity_type_name in each_rate_map_sum[population]:
                        merged_rate_map_sum[population][selectivity_type_name] = \
                            np.add(merged_rate_map_sum[population][selectivity_type_name],
                                   each_rate_map_sum[population][selectivity_type_name])
            if plot:
                for population in merged_pop_distances:
                    hist, edges = np.histogram(merged_pop_distances[population], bins=100)
                    fig, axes = plt.subplots(1)
                    axes.plot(edges[1:], hist)
                    axes.set_title('Population: %s' % population)
                    axes.set_xlabel('Normalized cell position')
                    axes.set_ylabel('Cell count')
                    clean_axes(axes)
                    fig.show()
                for population in merged_rate_map_sum:
                    for selectivity_type_name in merged_rate_map_sum[population]:
                        plot_2D_rate_map(x=arena_x_mesh, y=arena_y_mesh,
                                         rate_map=merged_rate_map_sum[population][selectivity_type_name],
                                         title='Summed rate maps\n%s %s cells' %
                                               (population, selectivity_type_name))

    if interactive and rank == 0:
        context.update(locals())


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv) + 1):],
         standalone_mode=False)
