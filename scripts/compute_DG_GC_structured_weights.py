from function_lib import *
from collections import Counter
from mpi4py import MPI
from neurotrees.io import NeurotreeAttrGen
from neurotrees.io import append_cell_attributes
from neurotrees.io import bcast_cell_attributes
from neurotrees.io import population_ranges
import click


try:
    import mkl
    mkl.set_num_threads(1)
except:
    pass


# log_normal weights: 1./(sigma * x * np.sqrt(2. * np.pi)) * np.exp(-((np.log(x)-mu)**2.)/(2. * sigma**2.))

script_name = 'compute_DG_GC_structured_weights.py'

local_random = np.random.RandomState()
# yields a distribution of synaptic weights with mean  ~>1., and tail ~2.-4.
mu = 0.
sigma = 0.35


plasticity_mask_sigma = 90. / 3. / np.sqrt(2.)  # cm
plasticity_mask = lambda d, d_offset: np.exp(-((d-d_offset)/plasticity_mask_sigma) ** 2.)
plasticity_mask = np.vectorize(plasticity_mask, excluded=[1])

#  custom data type for type of feature selectivity
selectivity_grid = 0
selectivity_place_field = 1

a = 0.55
b = -1.5
u = lambda ori: (np.cos(ori), np.sin(ori))
ori_array = 2. * np.pi * np.array([-30., 30., 90.]) / 360.  # rads
g = lambda x: np.exp(a * (x - b)) - 1.
scale_factor = g(3.)
grid_peak_rate = 20.  # Hz
grid_rate = lambda grid_spacing, ori_offset, x_offset, y_offset: \
    lambda x, y: grid_peak_rate / scale_factor * \
                 g(np.sum([np.cos(4. * np.pi / np.sqrt(3.) /
                                  grid_spacing * np.dot(u(theta - ori_offset), (x - x_offset, y - y_offset)))
                           for theta in ori_array]))

place_peak_rate = 20.  # Hz
place_rate = lambda field_width, x_offset, y_offset: \
    lambda x, y: place_peak_rate * np.exp(-((x - x_offset) / (field_width / 3. / np.sqrt(2.))) ** 2.) * \
                 np.exp(-((y - y_offset) / (field_width / 3. / np.sqrt(2.))) ** 2.)


@click.command()
@click.option("--features-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--weights-path", type=click.Path(exists=True, file_okay=True, dir_okay=False), default=None)
@click.option("--weights-namespace", type=str, default='Weights')
@click.option("--connectivity-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--connectivity-namespace", type=str, default='Connectivity')
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--cache-size", type=int, default=50)
@click.option("--trajectory-id", type=int, default=0)
@click.option("--seed", type=int, default=6)
@click.option("--debug", is_flag=True)
def main(features_path, weights_path, weights_namespace, connectivity_path, connectivity_namespace, io_size, chunk_size,
         value_chunk_size, cache_size, trajectory_id, seed, debug):
    """

    :param features_path:
    :param weights_path:
    :param weights_namespace:
    :param connectivity_path:
    :param connectivity_namespace:
    :param io_size:
    :param chunk_size:
    :param value_chunk_size:
    :param cache_size:
    :param trajectory_id:
    :param seed:
    :param debug:
    """
    # make sure random seeds are not being reused for various types of stochastic sampling
    weights_seed_offset = int(seed * 2e6)

    comm = MPI.COMM_WORLD
    rank = comm.rank

    if io_size == -1:
        io_size = comm.size
    if rank == 0:
        print '%i ranks have been allocated' % comm.size
    sys.stdout.flush()

    population_range_dict = population_ranges(MPI._addressof(comm), connectivity_path)

    features_dict = {}
    for population in ['MPP', 'LPP']:
        features_dict[population] = bcast_cell_attributes(MPI._addressof(comm), 0, features_path, population,
                                                          namespace='Feature Selectivity')

    if weights_path is None:
        weights_path = features_path

    arena_dimension = 100.  # minimum distance from origin to boundary (cm)

    run_vel = 30.  # cm/s
    spatial_resolution = 1.  # cm
    x = np.arange(-arena_dimension, arena_dimension, spatial_resolution)
    y = np.arange(-arena_dimension, arena_dimension, spatial_resolution)
    distance = np.insert(np.cumsum(np.sqrt(np.sum([np.diff(x) ** 2., np.diff(y) ** 2.], axis=0))), 0, 0.)
    interp_distance = np.arange(distance[0], distance[-1], spatial_resolution)
    t = interp_distance / run_vel * 1000.  # ms
    interp_x = np.interp(interp_distance, distance, x)
    interp_y = np.interp(interp_distance, distance, y)

    with h5py.File(features_path, 'a', driver='mpio', comm=comm) as f:
        if 'Trajectories' not in f:
            f.create_group('Trajectories')
        if str(trajectory_id) not in f['Trajectories']:
            f['Trajectories'].create_group(str(trajectory_id))
            f['Trajectories'][str(trajectory_id)].create_dataset('x', dtype='float32', data=interp_x)
            f['Trajectories'][str(trajectory_id)].create_dataset('y', dtype='float32', data=interp_y)
            f['Trajectories'][str(trajectory_id)].create_dataset('d', dtype='float32', data=interp_distance)
            f['Trajectories'][str(trajectory_id)].create_dataset('t', dtype='float32', data=t)
        x = f['Trajectories'][str(trajectory_id)]['x'][:]
        y = f['Trajectories'][str(trajectory_id)]['y'][:]
        d = f['Trajectories'][str(trajectory_id)]['d'][:]

    prediction_namespace = 'Response Prediction ' + str(trajectory_id)

    target_population = 'GC'
    count = 0
    start_time = time.time()
    attr_gen = NeurotreeAttrGen(MPI._addressof(comm), connectivity_path, target_population, io_size=io_size,
                                cache_size=cache_size, namespace=connectivity_namespace)
    if debug:
        attr_gen = [attr_gen.next() for i in xrange(2)]
    for gid, connectivity_dict in attr_gen:
        local_time = time.time()
        weights_dict = {}
        syn_ids = np.array([], dtype='uint32')
        weights = np.array([], dtype='float32')
        if gid is not None:
            for population in ['MPP', 'LPP']:
                indexes = np.where((connectivity_dict[connectivity_namespace]['source_gid'] >=
                                    population_range_dict[population][0]) &
                                   (connectivity_dict[connectivity_namespace]['source_gid'] <
                                    population_range_dict[population][0] + population_range_dict[population][1]))[0]
                syn_ids = np.append(syn_ids,
                                    connectivity_dict[connectivity_namespace]['syn_id'][indexes]).astype('uint32',
                                                                                                         copy=False)
            local_random.seed(gid + weights_seed_offset)
            weights = np.append(weights, local_random.lognormal(mu, sigma, len(syn_ids))).astype('float32', copy=False)
            weights_dict[gid] = {'syn_id': syn_ids, 'weight': weights}
            print 'Rank %i: took %.2f s to compute synaptic weights for %s gid %i' % \
                  (rank, time.time() - local_time, target_population, gid)
            count += 1
        if not debug:
            append_cell_attributes(MPI._addressof(comm), weights_path, target_population, weights_dict,
                                   namespace=weights_namespace, io_size=io_size, chunk_size=chunk_size,
                                   value_chunk_size=value_chunk_size)
        sys.stdout.flush()
        del syn_ids
        del weights
        del weights_dict
        gc.collect()

    global_count = comm.gather(count, root=0)
    if rank == 0:
        print '%i ranks took %.2f s to compute synaptic weights for %i %s cells' % \
              (comm.size, time.time() - start_time, np.sum(global_count), target_population)


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])