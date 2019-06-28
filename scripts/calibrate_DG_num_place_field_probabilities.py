"""
For "stimulus sources" from MEC and LEC, grid and place field widths, and grid spacing are assumed to be
topographically organized septo-temporally. Cells are assigned to one of ten discrete modules with distinct grid
spacing and field width. We assign stimulus_input_sources to cells probabilistically as a function of septo-temporal
position. The grid spacing across modules increases exponentially from 40 cm to 8 m. We then assume that GC, MC, and
CA3c neurons with place fields receive input from multiple discrete modules, and therefore have field widths that vary
with septo-temporal position, but are sampled from a continuous rather than a discrete distribution. Features are
imposed on these "proxy sources" for microcircuit clamp simulations.
"""
import click
from mpi4py import MPI
from dentate.env import Env
from dentate.plot import default_fig_options
from dentate.stimulus import InputSelectivityConfig, get_2D_arena_spatial_mesh, calibrate_num_place_field_probabilities
from dentate.utils import *

logger = get_script_logger(os.path.basename(__file__))

context = Struct(**dict(locals()))


@click.command()
@click.option("--config", required=True, type=str)
@click.option("--config-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='config')
@click.option("--arena-id", type=str, default='A')
@click.option("--populations", '-p', type=str, multiple=True)
@click.option("--module-ids", '-m', type=int, multiple=True)
@click.option("--target-fraction-active", type=float, default=None)
@click.option("--normalize-scale", type=bool, default=True)
@click.option("--verbose", '-v', is_flag=True)
@click.option("--interactive", is_flag=True)
@click.option("--debug", is_flag=True)
@click.option("--plot", is_flag=True)
@click.option("--show-fig", is_flag=True)
@click.option("--save-fig", required=False, type=str, default=None)
@click.option("--save-fig-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default=None)
@click.option("--font-size", type=float, default=14)
@click.option("--fig-format", required=False, type=str, default='svg')
def main(config, config_prefix, arena_id, populations, module_ids, target_fraction_active, normalize_scale, verbose,
         interactive, debug, plot, show_fig, save_fig, save_fig_dir, font_size, fig_format):
    """

    :param config: str (.yaml file name)
    :param config_prefix: str (path to dir)
    :param arena_id: str
    :param populations: tuple of str
    :param module_ids: tuple of int
    :param target_fraction_active: float
    :param normalize_scale: bool; whether to interpret the scale of the num_place_field_probabilities distribution
                                    as normalized to the scale of the mean place field width
    :param verbose: bool
    :param interactive: bool
    :param debug: bool
    :param plot: bool
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

    if plot:
        import matplotlib.pyplot as plt
        from dentate.plot import clean_axes

    fig_options = copy.copy(default_fig_options)
    fig_options.saveFigDir = save_fig_dir
    fig_options.fontSize = font_size
    fig_options.figFormat = fig_format
    fig_options.showFig = show_fig

    if save_fig is not None:
        save_fig = '%s %s' % (save_fig, arena_id)
    fig_options.saveFig = save_fig

    if len(populations) == 0:
        populations = ('MC', 'ConMC', 'LPP', 'GC', 'MPP', 'CA3c')

    if arena_id not in env.stimulus_config['Arena']:
        raise RuntimeError('Arena with ID: %s not specified by configuration at file path: %s' %
                           (arena_id, config_prefix + '/' + config))

    selectivity_type_names = dict((val, key) for (key, val) in viewitems(env.selectivity_types))

    arena = env.stimulus_config['Arena'][arena_id]
    arena_x_mesh, arena_y_mesh = \
        get_2D_arena_spatial_mesh(arena=arena, spatial_resolution=env.stimulus_config['Spatial Resolution'])

    local_random = np.random.RandomState()
    selectivity_seed_offset = int(env.modelConfig['Random Seeds']['Input Selectivity'])
    local_random.seed(selectivity_seed_offset - 1)

    selectivity_config = InputSelectivityConfig(env.stimulus_config, local_random)

    this_selectivity_type_name = 'place'
    this_selectivity_type = env.selectivity_types['place']

    if interactive:
        context.update(locals())

    if len(module_ids) == 0:
        module_ids = selectivity_config.module_ids
    elif not all([module_id in selectivity_config.module_ids for module_id in module_ids]):
        raise RuntimeError('calibrate_DG_num_place_field_probabilities: invalid module_ids provided: %s' %
                           str(module_ids))

    for population in populations:

        if population not in env.stimulus_config['Num Place Field Probabilities']:
            raise RuntimeError('calibrate_DG_num_place_field_probabilities: probabilities for number of place fields '
                               'not specified for population: %s' % population)
        num_place_field_probabilities = env.stimulus_config['Num Place Field Probabilities'][population]

        if population not in env.stimulus_config['Peak Rate'] or \
                this_selectivity_type not in env.stimulus_config['Peak Rate'][population]:
            raise RuntimeError('calibrate_DG_num_place_field_probabilities: peak rate not specified for population: '
                               '%s, selectivity type: %s' % (population, this_selectivity_type_name))
        peak_rate = env.stimulus_config['Peak Rate'][population][this_selectivity_type]

        start_time = time.time()
        for module_id in module_ids:
            field_width = selectivity_config.place_module_field_widths[module_id]
            logger.info(
                'Calibrating distribution of num_place_field_probabilities for population: %s, module: %i, '
                'field width: %.2f' % (population, module_id, field_width))
            modified_num_place_field_probabilities = \
                calibrate_num_place_field_probabilities(num_place_field_probabilities, field_width,
                                                        peak_rate=peak_rate, selectivity_type=this_selectivity_type,
                                                        arena=arena, normalize_scale=normalize_scale,
                                                        selectivity_config=selectivity_config,
                                                        target_fraction_active=target_fraction_active,
                                                        random_seed=selectivity_seed_offset + module_id,
                                                        plot=plot and show_fig)
            logger.info('Modified num_place_field_probabilities for population: %s, module: %i, field width: %.2f' %
                        (population, module_id, field_width))
            print_param_dict_like_yaml(modified_num_place_field_probabilities)
            sys.stdout.flush()
            if debug:
                context.update(locals())
                return

    if interactive:
        context.update(locals())


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv) + 1):],
         standalone_mode=False)
