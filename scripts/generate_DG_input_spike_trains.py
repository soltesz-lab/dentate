from dentate.utils import *
from mpi4py import MPI
from neuroh5.io import NeuroH5CellAttrGen, append_cell_attributes, read_population_ranges
import h5py
from dentate.env import Env
from dentate.stimulus import get_input_cell_config, generate_linear_trajectory, plot_1D_rate_map
from dentate.stgen import get_inhom_poisson_spike_times_by_thinning
import random
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
@click.option("--selectivity-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--output-path", type=click.Path(file_okay=True, dir_okay=False), default=None)
@click.option("--arena-id", type=str, default='A')
@click.option("--trajectory-id", type=str, default='Diag')
@click.option("--spikes-namespace", type=str, default='Input Spikes')
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
def main(config, config_prefix, selectivity_path, output_path, arena_id, trajectory_id, spikes_namespace, populations,
         io_size, chunk_size, value_chunk_size, cache_size, write_size, verbose, gather, interactive, debug, plot,
         dry_run):
    """

    :param config: str (.yaml file name)
    :param config_prefix: str (path to dir)
    :param selectivity_path: str (path to file)
    :param output_path: str
    :param arena_id: str
    :param trajectory_id: str
    :param spikes_namespace: str
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

    population_ranges = read_population_ranges(selectivity_path, comm)[0]

    if populations is None:
        populations = ['MC', 'ConMC', 'LPP', 'GC', 'MPP', 'CA3c']

    if arena_id not in env.input_config['Arena']:
        raise RuntimeError('Arena with ID: %s not specified by configuration at file path: %s' %
                           (arena_id, config_prefix + '/' + config))
    arena = env.input_config['Arena'][arena_id]

    if trajectory_id not in arena.trajectories:
        raise RuntimeError('Trajectory with ID: %s not specified by configuration at file path: %s' %
                           (trajectory_id, config_prefix + '/' + config))
    trajectory = arena.trajectories[trajectory_id]

    valid_selectivity_namespaces = dict()
    if rank == 0:
        for population in populations:
            if population not in population_ranges:
                raise RuntimeError('generate_DG_input_spike_trains: specified population: %s not found in '
                                   'provided selectivity_path: %s' % (population, selectivity_path))
            if population not in env.input_config['Selectivity Type Probabilities']:
                raise RuntimeError('generate_DG_input_spike_trains: selectivity type not specified for '
                                   'population: %s' % population)
            valid_selectivity_namespaces[population] = []
            with h5py.File(selectivity_path, 'r') as selectivity_f:
                for this_namespace in selectivity_f['Populations'][population]:
                    if 'Selectivity %s' % arena_id in this_namespace:
                        valid_selectivity_namespaces[population].append(this_namespace)
                if len(valid_selectivity_namespaces[population]) == 0:
                    raise RuntimeError('generate_DG_input_spike_trains: no selectivity data in arena: %s found '
                                       'for specified population: %s in provided selectivity_path: %s' %
                                       (arena_id, population, selectivity_path))

    valid_selectivity_namespaces = comm.bcast(valid_selectivity_namespaces, root=0)
    selectivity_type_names = dict((val, key) for (key, val) in env.selectivity_types.items())

    t, x, y, d = None, None, None, None
    if rank == 0:
        t, x, y, d = generate_linear_trajectory(trajectory,
                                                temporal_resolution=env.input_config['Temporal Resolution'],
                                                equilibration_duration=env.input_config['Equilibration Duration'])
    t = comm.bcast(t, root=0)
    x = comm.bcast(x, root=0)
    y = comm.bcast(y, root=0)
    d = comm.bcast(d, root=0)

    trajectory_namespace = 'Trajectory %s %s' % (arena_id, trajectory_id)
    this_spikes_namespace = '%s %s %s' % (spikes_namespace, arena_id, trajectory_id)

    if not dry_run and rank == 0:
        if output_path is None:
            raise RuntimeError('generate_DG_input_spike_trains: missing output_path')
        if not os.path.isfile(output_path):
            input_file = h5py.File(selectivity_path, 'r')
            output_file = h5py.File(output_path, 'w')
            input_file.copy('/H5Types', output_file)
            input_file.close()
            output_file.close()
        with h5py.File(output_path, 'a') as f:
            if trajectory_namespace not in f:
                logger.info('Appending %s datasets to file at output_path: %s' % (trajectory_namespace, output_path))
                group = f.create_group(trajectory_namespace)
                for key, value in zip(['t', 'x', 'y', 'd'], [t, x, y, d]):
                    dataset = group.create_dataset(key, data=value, dtype='float32')
            else:
                loaded_t = f[trajectory_namespace]['t'][:]
                if len(t) != len(loaded_t):
                    raise RuntimeError('generate_DG_input_spike_trains: file at output_path already contains the '
                                       'namespace: %s, but the dataset sizes are inconsistent with the provided input'
                                       'configuration')
    comm.barrier()

    if 'Equilibration Duration' in env.input_config and env.input_config['Equilibration Duration'] > 0.:
        equilibrate_len = int(env.input_config['Equilibration Duration'] / env.input_config['Temporal Resolution'])
        from scipy.signal import hann
        equilibrate_hann = hann(2 * equilibrate_len)[:equilibrate_len]
    else:
        equilibrate_hann = None

    local_random = random.Random()
    input_spike_train_offset = int(env.modelConfig['Random Seeds']['Input Spiketrains'])

    context.update(locals())

    if gather:
        spike_hist_resolution = 1000
        spike_hist_edges = np.linspace(min(t), max(t), spike_hist_resolution + 1)
        spike_hist_sum = defaultdict(lambda: defaultdict(lambda: np.zeros(spike_hist_resolution)))

    write_every = max(1, int(math.floor(write_size / comm.size)))
    for population in populations:
        gid_count = defaultdict(lambda: 0)
        process_time = dict()
        for this_selectivity_namespace in valid_selectivity_namespaces[population]:
            start_time = time.time()
            selectivity_attr_gen = NeuroH5CellAttrGen(selectivity_path, population,
                                                      namespace=this_selectivity_namespace, comm=comm, io_size=io_size,
                                                      cache_size=cache_size)
            spikes_attr_dict = dict()
            for iter_count, (gid, selectivity_attr_dict) in enumerate(selectivity_attr_gen):
                if gid is not None:
                    this_selectivity_type = selectivity_attr_dict['Selectivity Type'][0]
                    this_selectivity_type_name = selectivity_type_names[this_selectivity_type]
                    cell_config = get_input_cell_config(selectivity_type=this_selectivity_type,
                                                        selectivity_type_names=selectivity_type_names,
                                                        selectivity_attr_dict=selectivity_attr_dict)
                    rate_map = cell_config.get_rate_map(x=x, y=y)
                    if equilibrate_hann is not None:
                        rate_map[:equilibrate_len] = np.multiply(rate_map[:equilibrate_len], equilibrate_hann)
                    local_random.seed(int(input_spike_train_offset + gid))
                    spike_train = get_inhom_poisson_spike_times_by_thinning(rate_map, t, dt=0.025,
                                                                            generator=local_random)
                    if debug and plot and rank == 0:
                        plot_1D_rate_map(t=t, rate_map=rate_map,
                                         peak_rate=env.input_config['Peak Rate'][population][this_selectivity_type],
                                         spike_train=spike_train,
                                         title='%s %s cell %i' % (population, this_selectivity_type_name, gid))
                    spikes_attr_dict[gid] = dict()
                    spikes_attr_dict[gid]['Selectivity Type'] = np.array([this_selectivity_type], dtype='uint8')
                    spikes_attr_dict[gid]['Trajectory Rate Map'] = np.asarray(rate_map, dtype='float32')
                    spikes_attr_dict[gid]['Spike Train'] = np.asarray(spike_train, dtype='float32')
                    gid_count[this_selectivity_type_name] += 1
                    if gather:
                        hist, edges = np.histogram(spike_train, bins=spike_hist_edges)
                        spike_hist_sum[population][this_selectivity_type_name] = \
                            np.add(spike_hist_sum[population][this_selectivity_type_name], hist)

                gid_count_dict = dict(gid_count.items())
                gid_count_dict = comm.gather(gid_count_dict, root=0)
                if rank == 0:
                    merged_gid_count = defaultdict(lambda: 0)
                    for each_gid_count in gid_count_dict:
                        for selectivity_type_name in each_gid_count:
                            merged_gid_count[selectivity_type_name] += each_gid_count[selectivity_type_name]
                    total_gid_count = np.sum(merged_gid_count.values())

                if (iter_count > 0 and iter_count % write_every == 0) or (debug and iter_count == 10):
                    if verbose and rank == 0:
                        logger.info('processed %i %s %s cells' %
                                    (merged_gid_count[this_selectivity_type_name], population,
                                     this_selectivity_type_name))
                    if not dry_run:
                        append_cell_attributes(output_path, population, spikes_attr_dict,
                                               namespace=this_spikes_namespace, comm=comm, io_size=io_size,
                                               chunk_size=chunk_size, value_chunk_size=value_chunk_size)
                    del spikes_attr_dict
                    spikes_attr_dict = dict()
                sys.stdout.flush()
                comm.barrier()
                if debug and iter_count == 10:
                    break
            if not dry_run:
                append_cell_attributes(output_path, population, spikes_attr_dict,
                                       namespace=this_spikes_namespace, comm=comm, io_size=io_size,
                                       chunk_size=chunk_size, value_chunk_size=value_chunk_size)
            del spikes_attr_dict
            process_time[this_selectivity_type_name] = time.time() - start_time
        if verbose and rank == 0:
            for selectivity_type_name in merged_gid_count:
                logger.info('processed %i/%i %s %s cells in %.2f s' %
                            (merged_gid_count[selectivity_type_name], total_gid_count, population,
                             selectivity_type_name, process_time[selectivity_type_name]))

    if gather:
        spike_hist_sum = dict([(key, dict(val.items())) for key, val in spike_hist_sum.items()])
        spike_hist_sum = comm.gather(spike_hist_sum, root=0)
        if rank == 0:
            merged_spike_hist_sum = defaultdict(lambda: defaultdict(lambda: np.zeros(spike_hist_resolution)))
            for each_spike_hist_sum in spike_hist_sum:
                for population in each_spike_hist_sum:
                    for selectivity_type_name in each_spike_hist_sum[population]:
                        merged_spike_hist_sum[population][selectivity_type_name] = \
                            np.add(merged_spike_hist_sum[population][selectivity_type_name],
                                   each_spike_hist_sum[population][selectivity_type_name])
            if plot:
                for population in merged_spike_hist_sum:
                    for selectivity_type_name in merged_spike_hist_sum[population]:
                        fig, axes = plt.subplots()
                        axes.plot(spike_hist_edges[1:], merged_spike_hist_sum[population][selectivity_type_name])
                        axes.set_xlabel('Time (ms)')
                        axes.set_ylabel('Population spike count')
                        axes.set_ylim(0., np.max(merged_spike_hist_sum[population][selectivity_type_name]) * 1.1)
                        axes.set_title('Summed rate maps\n%s %s cells' % (population, selectivity_type_name))
                        clean_axes(axes)
                        fig.show()

    if interactive and rank == 0:
        context.update(locals())


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv) + 1):],
         standalone_mode=False)
