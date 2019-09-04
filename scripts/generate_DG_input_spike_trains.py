
import click
import copy, random
from mpi4py import MPI
import h5py
from dentate.env import Env
from dentate.stimulus import get_input_cell_config, generate_linear_trajectory, generate_input_spike_trains
from dentate.utils import *
from dentate.plot import default_fig_options, save_figure, plot_1D_rate_map
from neuroh5.io import NeuroH5CellAttrGen, append_cell_attributes, read_population_ranges

logger = get_script_logger(os.path.basename(__file__))

context = Struct(**dict(locals()))

sys_excepthook = sys.excepthook
def mpi_excepthook(type, value, traceback):
    sys_excepthook(type, value, traceback)
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)
#sys.excepthook = mpi_excepthook

def debug_callback(context):
    fig_title = '%s %s cell %i' % (context.population, context.this_selectivity_type_name, context.gid)
    fig_options = copy.copy(context.fig_options)
    if context.save_fig is not None:
        fig_options.saveFig = '%s %s' % (context.save_fig, fig_title)
    plot_1D_rate_map(t=t, rate_map=context.rate_map,
                     peak_rate=context.env.stimulus_config['Peak Rate'][context.population][context.this_selectivity_type],
                     spike_train=context.spike_train, title=fig_title, **fig_options())

@click.command()
@click.option("--config", required=True, type=str)
@click.option("--config-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='config')
@click.option("--selectivity-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--arena-id", type=str, default='A')
@click.option("--trajectory-id", type=str, default='Diag')
@click.option("--populations", '-p', type=str, multiple=True)
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--cache-size", type=int, default=50)
@click.option("--write-size", type=int, default=10000)
@click.option("--output-path", type=click.Path(file_okay=True, dir_okay=False), default=None)
@click.option("--spikes-namespace", type=str, default='Input Spikes')
@click.option("--spike-train-attr-name", type=str, default='Spike Train')
@click.option("--gather", is_flag=True)
@click.option("--interactive", is_flag=True)
@click.option("--debug", is_flag=True)
@click.option("--plot", is_flag=True)
@click.option("--show-fig", is_flag=True)
@click.option("--save-fig", required=False, type=str, default=None)
@click.option("--save-fig-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default=None)
@click.option("--font-size", type=float, default=14)
@click.option("--fig-format", required=False, type=str, default='svg')
@click.option("--verbose", '-v', is_flag=True)
@click.option("--dry-run", is_flag=True)
def main(config, config_prefix, selectivity_path, arena_id, trajectory_id, populations, io_size, chunk_size,
         value_chunk_size, cache_size, write_size, output_path, spikes_namespace, spike_train_attr_name, gather,
         interactive, debug, plot, show_fig, save_fig, save_fig_dir, font_size, fig_format,
         verbose, dry_run):
    """

    :param config: str (.yaml file name)
    :param config_prefix: str (path to dir)
    :param selectivity_path: str (path to file)
    :param arena_id: str
    :param trajectory_id: str
    :param populations: str
    :param io_size: int
    :param chunk_size: int
    :param value_chunk_size: int
    :param cache_size: int
    :param write_size: int
    :param output_path: str (path to file)
    :param spikes_namespace: str
    :param spike_train_attr_name: str
    :param gather: bool
    :param interactive: bool
    :param debug: bool
    :param plot: bool
    :param show_fig: bool
    :param save_fig: str (base file name)
    :param save_fig_dir:  str (path to dir)
    :param font_size: float
    :param fig_format: str
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

    population_ranges = read_population_ranges(selectivity_path, comm)[0]

    if len(populations) == 0:
        populations = ('MC', 'ConMC', 'LPP', 'GC', 'MPP', 'CA3c')

    if arena_id not in env.stimulus_config['Arena']:
        raise RuntimeError('Arena with ID: %s not specified by configuration at file path: %s' %
                           (arena_id, config_prefix + '/' + config))
    arena = env.stimulus_config['Arena'][arena_id]

    if trajectory_id not in arena.trajectories:
        raise RuntimeError('Trajectory with ID: %s not specified by configuration at file path: %s' %
                           (trajectory_id, config_prefix + '/' + config))
    trajectory = arena.trajectories[trajectory_id]

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

    t, x, y, d = None, None, None, None
    if rank == 0:
        t, x, y, d = generate_linear_trajectory(trajectory,
                                                temporal_resolution=env.stimulus_config['Temporal Resolution'],
                                                equilibration_duration=env.stimulus_config['Equilibration Duration'])

    
    t = comm.bcast(t, root=0)
    x = comm.bcast(x, root=0)
    y = comm.bcast(y, root=0)
    d = comm.bcast(d, root=0)

    trajectory = t, x, y, d
    trajectory_namespace = 'Trajectory %s %s' % (arena_id, trajectory_id)
    this_spikes_namespace = '%s %s %s' % (spikes_namespace, arena_id, trajectory_id)

    if not dry_run and rank == 0:
        if output_path is None:
            raise RuntimeError('generate_DG_source_spike_trains: missing output_path')
        if not os.path.isfile(output_path):
            with h5py.File(output_path, 'w') as output_file:
                input_file = h5py.File(selectivity_path, 'r')
                input_file.copy('/H5Types', output_file)
                input_file.close()
        with h5py.File(output_path, 'a') as f:
            if trajectory_namespace not in f:
                logger.info('Appending %s datasets to file at path: %s' % (trajectory_namespace, output_path))
                group = f.create_group(trajectory_namespace)
                for key, value in zip(['t', 'x', 'y', 'd'], [t, x, y, d]):
                    dataset = group.create_dataset(key, data=value, dtype='float32')
            else:
                loaded_t = f[trajectory_namespace]['t'][:]
                if len(t) != len(loaded_t):
                    raise RuntimeError('generate_DG_source_spike_trains: file at path: %s already contains the '
                                       'namespace: %s, but the dataset sizes are inconsistent with the provided input'
                                       'configuration' % (output_path, trajectory_namespace))
    comm.barrier()

    if 'Equilibration Duration' in env.stimulus_config and env.stimulus_config['Equilibration Duration'] > 0.:
        equilibrate_len = int(env.stimulus_config['Equilibration Duration'] /
                              env.stimulus_config['Temporal Resolution'])
        from scipy.signal import hann
        equilibrate_hann = hann(2 * equilibrate_len)[:equilibrate_len]
        equilibrate = (equilibrate_hann, equilibrate_len)
    else:
        equilibrate = None

    if rank == 0:
        context.update(locals())

    spike_hist_sum_dict = {}
    spike_hist_resolution = 1000

    write_every = max(1, int(math.floor(write_size / comm.size)))
    for population in populations:

        this_spike_hist_sum = \
          generate_input_spike_trains(env, population, trajectory, selectivity_path, selectivity_type_names,
                                      valid_selectivity_namespaces,
                                      output_path, output_namespace=this_spikes_namespace,
                                      spike_train_attr_name=spike_train_attr_name,
                                      spike_hist_resolution=spike_hist_resolution,
                                      equilibrate=equilibrate,
                                      comm=comm, io_size=io_size,
                                      write_every=write_every, chunk_size=chunk_size,
                                      value_chunk_size=value_chunk_size,
                                      dry_run=dry_run, debug= (debug_callback, context) if debug else False)
        if gather:
            spike_hist_sum_dict[population] = this_spike_hist_sum


    if gather:
        this_spike_hist_sum = dict([(key, dict(val.items())) for key, val in viewitems(spike_hist_sum_dict)])
        spike_hist_sum = comm.gather(this_spike_hist_sum, root=0)
        if rank == 0:
            merged_spike_hist_sum = defaultdict(lambda: defaultdict(lambda: np.zeros(spike_hist_resolution)))
            for each_spike_hist_sum in spike_hist_sum:
                for population in each_spike_hist_sum:
                    for selectivity_type_name in each_spike_hist_sum[population]:
                        merged_spike_hist_sum[population][selectivity_type_name] = \
                            np.add(merged_spike_hist_sum[population][selectivity_type_name],
                                   each_spike_hist_sum[population][selectivity_type_name])
            if plot:
                spike_hist_edges = np.linspace(min(t), max(t), spike_hist_resolution + 1)
                for population, this_selectivity_type_name in viewitems(merged_spike_hist_sum):
                    for this_selectivity_type_name in merged_spike_hist_sum[population]:
                        fig_title = '%s %s summed spike PSTH' % (population, this_selectivity_type_name)
                        if save_fig is not None:
                            fig_options.saveFig = '%s %s' % (save_fig, fig_title)
                        fig, axes = plt.subplots()
                        axes.plot(spike_hist_edges[1:], merged_spike_hist_sum[population][selectivity_type_name])
                        axes.set_xlabel('Time (ms)', fontsize=fig_options.fontSize)
                        axes.set_ylabel('Population spike count', fontsize=fig_options.fontSize)
                        axes.set_ylim(0., np.max(merged_spike_hist_sum[population][selectivity_type_name]) * 1.1)
                        axes.set_title('Summed spike PSTH\n%s %s cells' % (population, selectivity_type_name),
                                       fontsize=fig_options.fontSize)
                        clean_axes(axes)

                        if fig_options.saveFig is not None:
                            save_figure(fig_options.saveFig, fig=fig, **fig_options())

                        if fig_options.showFig:
                            fig.show()
        comm.barrier()
    if interactive and rank == 0:
        context.update(locals())


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv) + 1):],
         standalone_mode=False)
