
import sys, time, gc, random, click, logging
import numpy as np
import mpi4py
from mpi4py import MPI
import h5py
from neuroh5.io import NeuroH5CellAttrGen, append_cell_attributes, read_population_ranges
import dentate
from dentate.env import Env
from dentate import stimulus, stgen, utils

script_name = 'generate_DG_PP_spiketrains.py'


@click.command()
@click.option("--config", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--features-path", "-p", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--cache-size", type=int, default=50)
@click.option("--stimulus-id", type=int, default=0)
@click.option("--features-namespaces", "-n", type=str, multiple=True, default=['Grid Input Features','Place Input Features'])
@click.option("--stimulus-namespace", type=str, default='Vector Stimulus')
@click.option("--verbose", '-v', is_flag=True)
@click.option("--dry-run", is_flag=True)
def main(config, features_path, io_size, chunk_size, value_chunk_size, cache_size, stimulus_id, features_namespaces,
         stimulus_namespace, verbose, dry_run):
    """

    :param features_path: str
    :param io_size: int
    :param chunk_size: int
    :param value_chunk_size: int
    :param cache_size: int
    :param stimulus_id: int
    :param features_namespace: str
    :param stimulus_namespace: str
    :param dry_run: bool
    """
    utils.config_logging(verbose)
    logger = utils.get_script_logger(script_name)

    comm = MPI.COMM_WORLD
    rank = comm.rank

    env = Env(comm=comm, configFile=config)
    
    if io_size == -1:
        io_size = comm.size
    if rank == 0:
        print('%i ranks have been allocated' % comm.size)

    local_random = random.Random()
    input_spiketrain_offset = int(env.modelConfig['Random Seeds']['Input Spiketrains'])

    input_config = env.inputConfig[stimulus_id]
    feature_type_dict = input_config['feature type']

    arena_dimension = int(input_config['trajectory']['Distance to boundary'])  # minimum distance from origin to boundary (cm)

    arena_dimension = int(input_config['trajectory']['Distance to boundary'])  # minimum distance from origin to boundary (cm)
    default_run_vel = int(input_config['trajectory']['Default run velocity'])  # cm/s
    spatial_resolution = float(input_config['trajectory']['Spatial resolution'])  # cm

    trajectory_namespace = 'Trajectory %s' % str(stimulus_id)
    stimulus_id_namespace = '%s %s' % (stimulus_namespace, str(stimulus_id))

    if rank == 0:
        with h5py.File(features_path, 'a') as f:
            if trajectory_namespace not in f:
                logger.info('Rank: %i; Creating %s datasets' % (rank, trajectory_namespace))
                group = f.create_group(trajectory_namespace)
                t, x, y, d = stimulus.generate_trajectory(arena_dimension=arena_dimension, velocity=default_run_vel,
                                                          spatial_resolution=spatial_resolution)
                for key, value in zip(['x', 'y', 'd', 't'], [x, y, d, t]):
                    dataset = group.create_dataset(key, (value.shape[0],), dtype='float32')
                    dataset[:] = value.astype('float32', copy=False)
            else:
                logger.info('Rank: %i; Reading %s datasets' % (rank, trajectory_namespace))
                group = f[trajectory_namespace]
                dataset = group['x']
                x = dataset[:]
                dataset = group['y']
                y = dataset[:]
                dataset = group['d']
                d = dataset[:]
                dataset = group['t']
                t = dataset[:]
    else:
        x = None
        y = None
        d = None
        t = None
    x = comm.bcast(x, root=0)
    y = comm.bcast(y, root=0)
    d = comm.bcast(d, root=0)
    t = comm.bcast(t, root=0)

    population_ranges = read_population_ranges(features_path, comm=comm)[0]

    for population in ['MPP', 'LPP']:
        population_start = population_ranges[population][0]

        count = 0
        start_time = time.time()

        for features_type, features_namespace in enumerate(features_namespaces):
            attr_gen = NeuroH5CellAttrGen(features_path, population, namespace=features_namespace,
                                              comm=comm, io_size=io_size, cache_size=cache_size)
            for gid, features_dict in attr_gen:
                response_dict = {}
                response = None
                if gid is None:
                    logger.info('Rank %i gid is None' % rank)
                else:
                    logger.info('Rank %i received attributes for gid %i' % (rank, gid))
                    local_time = time.time()
                    response = stimulus.generate_spatial_ratemap(features_type, features_dict, t, x, y, 
                                                                grid_peak_rate=20., place_peak_rate=20.)
                    local_random.seed(int(input_spiketrain_offset + gid))
                    spiketrain = stgen.get_inhom_poisson_spike_times_by_thinning(response, t, generator=local_random)
                    response_dict[gid] = {'rate': response, \
                                          'spiketrain': np.asarray(spiketrain, dtype='float32')}
                    baseline = np.mean(response[np.where(response <= np.percentile(response, 10.))[0]])
                    peak = np.mean(response[np.where(response >= np.percentile(response, 90.))[0]])
                    modulation = 0. if peak <= 0.1 else (peak - baseline) / peak
                    peak_index = np.where(response == np.max(response))[0][0]
                    response_dict[gid]['modulation'] = np.array([modulation], dtype='float32')
                    response_dict[gid]['peak index'] = np.array([peak_index], dtype='uint32')
                    logger.info( 'Rank %i; source: %s; generated spike trains for gid %i in %.2f s' % \
                                    (rank, population, gid, time.time() - local_time))
                    count += 1
                if not dry_run:
                    append_cell_attributes(features_path, population, response_dict,
                                            namespace=stimulus_id_namespace, comm=comm,
                                            io_size=io_size, chunk_size=chunk_size,
                                            value_chunk_size=value_chunk_size)
                del response
                response_dict.clear()
                gc.collect()

        global_count = comm.gather(count, root=0)
        if rank == 0:
            logger.info('%i ranks generated spike trains for %i cells in %.2f s' % (comm.size, np.sum(global_count),
                                                                                     time.time() - start_time))


if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])
