
import sys, time, os, gc, random, click, logging
import numpy as np
import mpi4py
from mpi4py import MPI
import h5py
from neuroh5.io import NeuroH5CellAttrGen, append_cell_attributes, read_population_ranges
import dentate
from dentate.env import Env
from dentate import stimulus, stgen, utils, InputCell

sys_excepthook = sys.excepthook
def mpi_excepthook(type, value, traceback):
    sys_excepthook(type, value, traceback)
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)
sys.excepthook = mpi_excepthook


@click.command()
@click.option("--config", required=True, type=str)
@click.option("--config-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True), default='config')
@click.option("--features-path", "-p", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--output-path", "-o", required=True, type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--cache-size", type=int, default=50)
@click.option("--arena-id", type=str, default='A')
@click.option("--trajectory-id", type=str, default='Dflt')
@click.option("--features-namespaces", "-n", type=str, multiple=True, default=['Grid Input Features','Place Input Features'])
@click.option("--stimulus-namespace", type=str, default='Vector Stimulus')
@click.option("--verbose", '-v', is_flag=True)
@click.option("--dry-run", is_flag=True)
def main(config, config_prefix, features_path, output_path, io_size, chunk_size, value_chunk_size, cache_size, arena_id, trajectory_id,
         features_namespaces, stimulus_namespace, verbose, dry_run):
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
    logger = utils.get_script_logger(os.path.basename(__file__))

    comm = MPI.COMM_WORLD
    rank = comm.rank

    env = Env(comm=comm, config_file=config, config_prefix=config_prefix)
    
    if io_size == -1:
        io_size = comm.size
    if rank == 0:
        print('%i ranks have been allocated' % comm.size)

    local_random = random.Random()
    input_spiketrain_offset = int(env.modelConfig['Random Seeds']['Input Spiketrains'])

    input_config = env.input_config
    spatial_resolution = input_config['Spatial Resolution']
    feature_type_dict = input_config['Feature Distribution']

    arena = input_config['Arena'][arena_id]

    trajectory_namespace = 'Trajectory %s %s' % (arena_id, str(trajectory_id))
    stimulus_id_namespace = '%s %s %s' % (stimulus_namespace, str(arena_id), str(trajectory_id))

    generate_trajectory = stimulus.generate_linear_trajectory
    
    if rank == 0:
        if (not dry_run):
            if not os.path.isfile(output_path):
                input_file  = h5py.File(features_path,'r')
                output_file = h5py.File(output_path,'w')
                input_file.copy('/H5Types',output_file)
                input_file.close()
                output_file.close()

        with h5py.File(features_path, 'a') as f:
            if trajectory_namespace not in f:
                logger.info('Rank: %i; Creating %s datasets' % (rank, trajectory_namespace))
                group = f.create_group(trajectory_namespace)
                t, x, y, d = generate_trajectory(arena, trajectory_id, spatial_resolution=spatial_resolution)
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

            output_file = h5py.File(output_path, 'a')
            f.copy(trajectory_namespace, output_file)
            output_file.close()
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
            attr_gen = NeuroH5CellAttrGen(features_path, population,
                                          namespace='%s %s' % (features_namespace, str(arena_id)),
                                          comm=comm, io_size=io_size, cache_size=cache_size)
                
            response_dict = {}
            for gid, features_dict in attr_gen:
                response = None
                if gid is None:
                    logger.info('Rank %i gid is None' % rank)
                else:
                    if verbose:
                        logger.info('Rank %i received attributes for gid %i' % (rank, gid))
                    local_time = time.time()
                    del(features_dict['gid'])
                    cell = InputCell.make_input_cell(gid, features_type, features_dict)
                    response = cell.generate_spatial_ratemap(x, y)
                    local_random.seed(int(input_spiketrain_offset + gid))
                    spiketrain = stgen.get_inhom_poisson_spike_times_by_thinning(response, t, generator=local_random)
                    if len(spiketrain) > 0:
                        if np.min(spiketrain) < 0:
                            logger.info("Rank %i gid %i: response = %s" % (rank, gid, str(response)))
                            logger.info("Rank %i gid %i: t = %s" % (rank, gid, str(t)))
                            logger.info("Rank %i gid %i: spiketrain min = %f" % (rank, gid, np.min(spiketrain)))
                    response_dict[gid] = cell.return_attr_dict()
                    response_dict[gid]['spiketrain'] = np.asarray(spiketrain, dtype='float32')
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
                    append_cell_attributes(output_path, population, response_dict,
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
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv)+1):])

