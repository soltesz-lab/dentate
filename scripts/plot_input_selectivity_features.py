import click
from mpi4py import MPI
import h5py
from neuroh5.io import NeuroH5CellAttrGen, read_population_ranges, bcast_cell_attributes
from dentate.env import Env
from dentate.plot import plot_2D_rate_map, default_fig_options, plot_2D_histogram, save_figure, close_figure
from dentate.stimulus import get_2D_arena_spatial_mesh, get_input_cell_config
from dentate.utils import *
import matplotlib.pyplot as plt

logger = get_script_logger(os.path.basename(__file__))

context = Context()


def mpi_excepthook(type, value, traceback):
    sys_excepthook(type, value, traceback)
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)

sys_excepthook = sys.excepthook
sys.excepthook = mpi_excepthook

def concatenate_ndarray(a, b, datatype):
    return np.concatenate((a,b))
mpi_op_concatenate_ndarray = MPI.Op.Create(concatenate_ndarray, commute=True)

def merge_count_dict(a, b, datatype):
    result = {}
    for k in a:
        result[k] = a[k]
    for k in b:
        result[k] = b[k] + result.get(k, 0)
    return result
mpi_op_merge_count_dict = MPI.Op.Create(merge_count_dict, commute=True)

def merge_list_dict(a, b, datatype):
    result = {}
    for k in a:
        result[k] = a[k]
    for k in b:
        result[k] = b[k] + result.get(k, [])
    return result
mpi_op_merge_list_dict = MPI.Op.Create(merge_list_dict, commute=True)


@click.command()
@click.option("--config", required=True, type=str)
@click.option("--config-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='config')
@click.option("--coords-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--distances-namespace", '-n', type=str, default='Arc Distances')
@click.option("--bin-distance", type=float, default=100.)
@click.option("--selectivity-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--selectivity-namespace", '-s', type=str, default='Selectivity')
@click.option("--spatial-resolution", type=float)
@click.option("--arena-id", type=str, default='A')
@click.option("--populations", '-p', type=str, multiple=True)
@click.option("--io-size", type=int, default=-1)
@click.option("--cache-size", type=int, default=50000)
@click.option("--verbose", '-v', is_flag=True)
@click.option("--debug", is_flag=True)
@click.option("--show-fig", is_flag=True)
@click.option("--save-fig", required=False, type=str, default=None)
@click.option("--save-fig-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default=None)
@click.option("--font-size", type=float, default=14)
@click.option("--fig-size", type=(float, float), default=(15, 8))
@click.option("--colormap", type=str)
@click.option("--fig-format", required=False, type=str, default='svg')
def main(config, config_prefix, coords_path, distances_namespace, bin_distance, selectivity_path, selectivity_namespace, spatial_resolution, arena_id,
         populations, io_size, cache_size, verbose, debug, show_fig, save_fig, save_fig_dir, font_size, fig_size, colormap,
         fig_format):
    """

    :param config: str (.yaml file name)
    :param config_prefix: str (path to dir)
    :param coords_path: str (path to file)
    :param distances_namespace: str
    :param bin_distance: float
    :param selectivity_path: str
    :param arena_id: str
    :param populations: tuple of str
    :param io_size: int
    :param cache_size: int
    :param verbose: bool
    :param debug: bool
    :param show_fig: bool
    :param save_fig: str (base file name)
    :param save_fig_dir:  str (path to dir)
    :param font_size: float
    :param fig_format: str
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank

    config_logging(verbose)

    env = Env(comm=comm, config_file=config, config_prefix=config_prefix, template_paths=None)
    if io_size == -1:
        io_size = comm.size
    if rank == 0:
        logger.info(f'{comm.size} ranks have been allocated')
    
    fig_options = copy.copy(default_fig_options)
    fig_options.saveFigDir = save_fig_dir
    fig_options.fontSize = font_size
    fig_options.figFormat = fig_format
    fig_options.showFig = show_fig
    fig_options.figSize = fig_size

    if save_fig is not None:
        save_fig = f'{save_fig} {arena_id}'
    fig_options.saveFig = save_fig
    
    population_ranges = read_population_ranges(selectivity_path, comm)[0]
    coords_population_ranges = read_population_ranges(coords_path, comm)[0]

    if len(populations) == 0:
        populations = ('MC', 'ConMC', 'LPP', 'GC', 'MPP', 'CA3c')

    valid_selectivity_namespaces = dict()
    if rank == 0:
        for population in populations:
            if population not in population_ranges:
                raise RuntimeError(f'plot_input_selectivity_features: specified population: {population} not found in '
                                   f'provided selectivity_path: {selectivity_path}')
            if population not in env.stimulus_config['Selectivity Type Probabilities']:
                raise RuntimeError('plot_input_selectivity_features: selectivity type not specified for '
                                   f'population: {population}')
            valid_selectivity_namespaces[population] = []
            with h5py.File(selectivity_path, 'r') as selectivity_f:
                for this_namespace in selectivity_f['Populations'][population]:
                    if f'{selectivity_namespace} {arena_id}' in this_namespace:
                        valid_selectivity_namespaces[population].append(this_namespace)
                if len(valid_selectivity_namespaces[population]) == 0:
                    raise RuntimeError(f'plot_input_selectivity_features: no selectivity data in arena: {arena_id} found '
                                       f'for specified population: {population} in provided selectivity_path: {selectivity_path}')

    valid_selectivity_namespaces = comm.bcast(valid_selectivity_namespaces, root=0)
    selectivity_type_names = dict((val, key) for (key, val) in viewitems(env.selectivity_types))
    
    reference_u_arc_distance_bounds = None
    reference_v_arc_distance_bounds = None
    if rank == 0:
        for population in populations:
            if population not in coords_population_ranges:
                raise RuntimeError(f'plot_input_selectivity_features: specified population: {population} not found in '
                                   f'provided coords_path: {coords_path}')
            with h5py.File(coords_path, 'r') as coords_f:
                pop_size = population_ranges[population][1]
                unique_gid_count = len(set(
                    coords_f['Populations'][population][distances_namespace]['U Distance']['Cell Index'][:]))
                if pop_size != unique_gid_count:
                    raise RuntimeError(f'plot_input_selectivity_features: only {unique_gid_count}/{pop_size} unique cell indexes found '
                                       f'for specified population: {population} in provided coords_path: {coords_path}')
                if reference_u_arc_distance_bounds is None:
                    try:
                        reference_u_arc_distance_bounds = \
                            coords_f['Populations'][population][distances_namespace].attrs['Reference U Min'], \
                            coords_f['Populations'][population][distances_namespace].attrs['Reference U Max']
                    except Exception:
                        raise RuntimeError('plot_input_selectivity_features: problem locating attributes '
                                           f'containing reference bounds in namespace: {distances_namespace} '
                                           f'for population: {population} from coords_path: {coords_path}')
                if reference_v_arc_distance_bounds is None:
                    try:
                        reference_v_arc_distance_bounds = \
                            coords_f['Populations'][population][distances_namespace].attrs['Reference V Min'], \
                            coords_f['Populations'][population][distances_namespace].attrs['Reference V Max']
                    except Exception:
                        raise RuntimeError('plot_input_selectivity_features: problem locating attributes '
                                           f'containing reference bounds in namespace: {distances_namespace} '
                                           f'for population: {population} from coords_path: {coords_path}')
    reference_u_arc_distance_bounds = comm.bcast(reference_u_arc_distance_bounds, root=0)
    reference_v_arc_distance_bounds = comm.bcast(reference_v_arc_distance_bounds, root=0)

    u_edges = np.arange(reference_u_arc_distance_bounds[0], reference_u_arc_distance_bounds[1] + bin_distance / 2.,
                        bin_distance)
    v_edges = np.arange(reference_v_arc_distance_bounds[0], reference_v_arc_distance_bounds[1] + bin_distance / 2.,
                        bin_distance)

    if arena_id not in env.stimulus_config['Arena']:
        raise RuntimeError(f'Arena with ID: {arena_id} not specified by configuration at file path: {config_prefix}/{config}')

    if spatial_resolution is None:
        spatial_resolution=env.stimulus_config['Spatial Resolution']    
    arena = env.stimulus_config['Arena'][arena_id]
    arena_x_mesh, arena_y_mesh = None, None
    if rank == 0:
        arena_x_mesh, arena_y_mesh = \
            get_2D_arena_spatial_mesh(arena=arena, spatial_resolution=spatial_resolution)
    arena_x_mesh = comm.bcast(arena_x_mesh, root=0)
    arena_y_mesh = comm.bcast(arena_y_mesh, root=0)
    x0_dict = {}
    y0_dict = {}

    for population in populations:

        start_time = time.time()
        u_distances_by_gid = dict()
        v_distances_by_gid = dict()
        distances_attr_gen = \
            bcast_cell_attributes(coords_path, population, root=0, namespace=distances_namespace, comm=comm)
        for gid, distances_attr_dict in distances_attr_gen:
            u_distances_by_gid[gid] = distances_attr_dict['U Distance'][0]
            v_distances_by_gid[gid] = distances_attr_dict['V Distance'][0]

        if rank == 0:
            logger.info(f'Reading {len(u_distances_by_gid)} cell positions for population {population} took '
                        f'{time.time() - start_time:.2f} s')

        for this_selectivity_namespace in valid_selectivity_namespaces[population]:
            start_time = time.time()
            if rank == 0:
                logger.info(f'Reading from {this_selectivity_namespace} namespace for population {population}...')
            gid_count = 0
            gathered_cell_attributes = defaultdict(list)
            gathered_component_attributes = defaultdict(list)
            u_distances_by_cell = list()
            v_distances_by_cell = list()
            u_distances_by_component = list()
            v_distances_by_component = list()
            rate_map_sum_by_module = defaultdict(lambda: np.zeros_like(arena_x_mesh))
            count_by_module = defaultdict(int)
            start_time = time.time()
            x0_list_by_module = defaultdict(list)
            y0_list_by_module = defaultdict(list)
            selectivity_attr_gen = NeuroH5CellAttrGen(selectivity_path, population,
                                                      namespace=this_selectivity_namespace, comm=comm, io_size=io_size,
                                                      cache_size=cache_size)
            for iter_count, (gid, selectivity_attr_dict) in enumerate(selectivity_attr_gen):
                if gid is not None:
                    gid_count += 1
                    this_selectivity_type = selectivity_attr_dict['Selectivity Type'][0]
                    this_selectivity_type_name = selectivity_type_names[this_selectivity_type]
                    input_cell_config = \
                        get_input_cell_config(selectivity_type=this_selectivity_type,
                                               selectivity_type_names=selectivity_type_names,
                                               selectivity_attr_dict=selectivity_attr_dict)
                    rate_map = input_cell_config.get_rate_map(x=arena_x_mesh, y=arena_y_mesh)
                    u_distances_by_cell.append(u_distances_by_gid[gid])
                    v_distances_by_cell.append(v_distances_by_gid[gid])
                    this_cell_attrs, component_count, this_component_attrs = input_cell_config.gather_attributes()
                    for attr_name, attr_val in viewitems(this_cell_attrs):
                        gathered_cell_attributes[attr_name].append(attr_val)
                    gathered_cell_attributes['Mean Rate'].append(np.mean(rate_map))
                    if component_count > 0:
                        u_distances_by_component.extend([u_distances_by_gid[gid]] * component_count)
                        v_distances_by_component.extend([v_distances_by_gid[gid]] * component_count)
                        for attr_name, attr_val in viewitems(this_component_attrs):
                            gathered_component_attributes[attr_name].extend(attr_val)
                    this_module_id = this_cell_attrs['Module ID']
                    if debug and rank == 0:
                        fig_title = f'{population} {this_selectivity_type_name} cell {gid}'
                        if save_fig is not None:
                            fig_options.saveFig = f'{save_fig} {fig_title}'
                        plot_2D_rate_map(x=arena_x_mesh, y=arena_y_mesh, rate_map=rate_map,
                                         peak_rate = env.stimulus_config['Peak Rate'][population][this_selectivity_type],
                                         title=f'{fig_title}\nModule: {this_module_id}',
                                         **fig_options())
                    x0_list_by_module[this_module_id].append(selectivity_attr_dict['X Offset'])
                    y0_list_by_module[this_module_id].append(selectivity_attr_dict['Y Offset'])
                    rate_map_sum_by_module[this_module_id] = np.add(rate_map, rate_map_sum_by_module[this_module_id])
                    count_by_module[this_module_id] += 1
                if debug and iter_count >= 10:
                    break

            if rank == 0:
                logger.info(f'Done reading from {this_selectivity_namespace} namespace for population {population}...')
                

            cell_count_hist, _, _ = np.histogram2d(u_distances_by_cell, v_distances_by_cell, bins=[u_edges, v_edges])
            component_count_hist, _, _ = np.histogram2d(u_distances_by_component, v_distances_by_component,
                                                 bins=[u_edges, v_edges])

            if debug:
                context.update(locals())

            gathered_cell_attr_hist = dict()
            gathered_component_attr_hist = dict()
            for key in gathered_cell_attributes:
                gathered_cell_attr_hist[key], _, _ = \
                    np.histogram2d(u_distances_by_cell, v_distances_by_cell, bins=[u_edges, v_edges],
                                   weights=gathered_cell_attributes[key])
            for key in gathered_component_attributes:
                gathered_component_attr_hist[key], _, _ = \
                    np.histogram2d(u_distances_by_component, v_distances_by_component, bins=[u_edges, v_edges],
                                   weights=gathered_component_attributes[key])
            gid_count = comm.gather(gid_count, root=0)
            cell_count_hist = comm.gather(cell_count_hist, root=0)
            component_count_hist = comm.gather(component_count_hist, root=0)
            gathered_cell_attr_hist = comm.gather(gathered_cell_attr_hist, root=0)
            gathered_component_attr_hist = comm.gather(gathered_component_attr_hist, root=0)
            x0_list_by_module = dict(x0_list_by_module)
            y0_list_by_module = dict(y0_list_by_module)
            x0_list_by_module = comm.reduce(x0_list_by_module, op=mpi_op_merge_list_dict, root=0)
            y0_list_by_module = comm.reduce(y0_list_by_module, op=mpi_op_merge_list_dict, root=0)
            rate_map_sum_by_module = dict(rate_map_sum_by_module)
            rate_map_sum_by_module = comm.gather(rate_map_sum_by_module, root=0)
            count_by_module = dict(count_by_module)
            count_by_module = comm.reduce(count_by_module, op=mpi_op_merge_count_dict, root=0)

            if rank == 0:
                gid_count = sum(gid_count)
                cell_count_hist = np.sum(cell_count_hist, axis=0)
                component_count_hist = np.sum(component_count_hist, axis=0)
                merged_cell_attr_hist = defaultdict(lambda: np.zeros_like(cell_count_hist))
                merged_component_attr_hist = defaultdict(lambda: np.zeros_like(component_count_hist))
                for each_cell_attr_hist in gathered_cell_attr_hist:
                    for key in each_cell_attr_hist:
                        merged_cell_attr_hist[key] = np.add(merged_cell_attr_hist[key], each_cell_attr_hist[key])
                for each_component_attr_hist in gathered_component_attr_hist:
                    for key in each_component_attr_hist:
                        merged_component_attr_hist[key] = np.add(merged_component_attr_hist[key],
                                                                 each_component_attr_hist[key])
                merged_rate_map_sum_by_module = defaultdict(lambda: np.zeros_like(arena_x_mesh))
                for each_rate_map_sum_by_module in rate_map_sum_by_module:
                    for this_module_id in each_rate_map_sum_by_module:
                        merged_rate_map_sum_by_module[this_module_id] = \
                            np.add(merged_rate_map_sum_by_module[this_module_id],
                                   each_rate_map_sum_by_module[this_module_id])

                logger.info(f'Processing {gid_count} {population} {this_selectivity_type_name} cells '
                            f'took {time.time() - start_time:.2f} s')

                if debug:
                    context.update(locals())

                fig_title = f'{population} {this_selectivity_type_name} field offsets'
                if save_fig is not None:
                        fig_options.saveFig = f'{save_fig} {fig_title}'
                
                for key in merged_cell_attr_hist:
                    fig_title = f'{population} {this_selectivity_type_name} cells {key} distribution'
                    if save_fig is not None:
                        fig_options.saveFig = f'{save_fig} {fig_title}'
                    if colormap is not None:
                        fig_options.colormap = colormap
                    title = f'{population} {this_selectivity_type_name} cells\n{key} distribution'
                    fig = plot_2D_histogram(merged_cell_attr_hist[key], x_edges=u_edges, y_edges=v_edges,
                                            norm=cell_count_hist, ylabel='Transverse position (um)',
                                            xlabel='Septo-temporal position (um)', title=title,
                                            cbar_label='Mean value per bin', cbar=True, **fig_options())
                    close_figure(fig)

                for key in merged_component_attr_hist:
                    fig_title = f'{population} {this_selectivity_type_name} cells {key} distribution'
                    if save_fig is not None:
                        fig_options.saveFig = f'{save_fig} {fig_title}'
                    title = f'{population} {this_selectivity_type_name} cells\n{key} distribution'
                    fig = plot_2D_histogram(merged_component_attr_hist[key], x_edges=u_edges, y_edges=v_edges,
                                            norm=component_count_hist, ylabel='Transverse position (um)',
                                            xlabel='Septo-temporal position (um)', title=title,
                                            cbar_label='Mean value per bin', cbar=True, **fig_options())
                    close_figure(fig)

                for this_module_id in merged_rate_map_sum_by_module:
                    num_cells = count_by_module[this_module_id]
                    x0 = np.concatenate(x0_list_by_module[this_module_id])
                    y0 = np.concatenate(y0_list_by_module[this_module_id])
                    fig_title = f'{population} {this_selectivity_type_name} Module {this_module_id} rate map'
                    if save_fig is not None:
                        fig_options.saveFig = f'{save_fig} {fig_title}'
                    fig = plot_2D_rate_map(x=arena_x_mesh, y=arena_y_mesh, x0=x0, y0=y0,
                                           rate_map=merged_rate_map_sum_by_module[this_module_id],
                                           title=(f'{population} {this_selectivity_type_name} rate map\n'
                                                  f'Module {this_module_id} ({num_cells} cells)'),
                                           **fig_options())
                    close_figure(fig)


    if is_interactive and rank == 0:
        context.update(locals())


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv) + 1):],
         standalone_mode=False)
