from function_lib import *
from itertools import izip
from collections import defaultdict
from mpi4py import MPI
from neurotrees.io import NeurotreeAttrGen
from neurotrees.io import append_cell_attributes
from neurotrees.io import bcast_cell_attributes
from neurotrees.io import population_ranges
import click

"""
features_path: contains 'Feature Selectivity' namespace describing inputs
weights_path: contains existing 'Weights' namespace; write 'Structured Weights' namespace to this path
connectivity_path: contains existing mapping of syn_id to source_gid

10% of GCs will have a subset of weights modified according to a slow timescale plasticity rule, the rest are inherited
    from the previous set of weights
    
TODO: Rather than choosing peak_locs randomly, have the peak_locs depend on the previous weight distribution.
"""


try:
    import mkl
    mkl.set_num_threads(1)
except:
    pass


# log_normal weights: 1./(sigma * x * np.sqrt(2. * np.pi)) * np.exp(-((np.log(x)-mu)**2.)/(2. * sigma**2.))

script_name = 'compute_DG_GC_structured_weights.py'

local_random = np.random.RandomState()

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
@click.option("--structured-weights-namespace", type=str, default='Structured Weights')
@click.option("--connectivity-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--connectivity-namespace", type=str, default='Connectivity')
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--cache-size", type=int, default=50)
@click.option("--trajectory-id", type=int, default=0)
@click.option("--seed", type=int, default=6)
@click.option("--target-sparsity", type=float, default=0.05)
@click.option("--debug", is_flag=True)
def main(features_path, weights_path, weights_namespace, structured_weights_namespace, connectivity_path,
         connectivity_namespace, io_size, chunk_size, value_chunk_size, cache_size, trajectory_id, seed,
         target_sparsity, debug):
    """

    :param features_path:
    :param weights_path:
    :param weights_namespace:
    :param structured_weights_namespace:
    :param connectivity_path:
    :param connectivity_namespace:
    :param io_size:
    :param chunk_size:
    :param value_chunk_size:
    :param cache_size:
    :param trajectory_id:
    :param seed:
    :param target_sparsity:
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
    source_population_list = ['MPP', 'LPP']
    for population in source_population_list:
        features_dict[population] = bcast_cell_attributes(MPI._addressof(comm), 0, features_path, population,
                                                          namespace='Feature Selectivity')

    if weights_path is None:
        weights_path = features_path

    arena_dimension = 100.  # minimum distance from origin to boundary (cm)

    run_vel = 30.  # cm/s
    spatial_resolution = 1.  # cm
    plasticity_window_dur = 3.  # s
    plasticity_kernel_sigma = plasticity_window_dur * run_vel / 3. / np.sqrt(2.)  # cm
    plasticity_kernel = lambda d, d_offset: np.exp(-((d - d_offset) / plasticity_kernel_sigma) ** 2.)
    plasticity_kernel = np.vectorize(plasticity_kernel, excluded=[1])

    with h5py.File(features_path, 'a', driver='mpio', comm=comm) as f:
        try:
            x = f['Trajectories'][str(trajectory_id)]['x'][:]
            y = f['Trajectories'][str(trajectory_id)]['y'][:]
            d = f['Trajectories'][str(trajectory_id)]['d'][:]
        except:
            raise AttributeError('compute_DG_GC_structured_weights: features file does not contain trajectory')

    target_population = 'GC'
    count = 0
    start_time = time.time()
    connectivity_gen = NeurotreeAttrGen(MPI._addressof(comm), connectivity_path, target_population, io_size=io_size,
                                        cache_size=cache_size, namespace=connectivity_namespace)
    weights_gen = NeurotreeAttrGen(MPI._addressof(comm), weights_path, target_population, io_size=io_size,
                                   cache_size=cache_size, namespace=weights_namespace)
    if debug:
        attr_gen = ((connectivity_gen.next(), weights_gen.next()) for i in xrange(2))
    else:
        attr_gen = izip(connectivity_gen, weights_gen)
    for (gid, connectivity_dict), (weights_gid, weights_dict) in attr_gen:
        local_time = time.time()
        source_map = {}
        weight_map = {}
        structured_weight_map = {}
        structured_weights_dict = {}
        if gid is not None:
            if gid != weights_gid:
                raise Exception('gid %i from connectivity_gen does not match gid %i from weights_gen')
            local_random.seed(gid + weights_seed_offset)
            if local_random.uniform() <= target_sparsity:
                weight_map = dict(zip(weights_dict[weights_namespace]['syn_id'],
                                      weights_dict[weights_namespace]['weight']))
                print 'Rank: %i, %s gid: %i, loaded %i inputs' % (rank, target_population, gid, len(weight_map))
                for population in source_population_list:
                    source_map[population] = defaultdict(list)
                for i in xrange(len(connectivity_dict[connectivity_namespace]['source_gid'])):
                    source_gid = connectivity_dict[connectivity_namespace]['source_gid'][i]
                    population = gid_in_population_list(source_gid, source_population_list, population_range_dict)
                    if population is not None:
                        syn_id = connectivity_dict[connectivity_namespace]['syn_id'][i]
                        source_map[population][source_gid].append(syn_id)
                peak_loc = local_random.choice(d)
                this_plasticity_kernel = plasticity_kernel(d, peak_loc)
                plasticity_kernel_area = np.sum(this_plasticity_kernel) * spatial_resolution
                plasticity_signal_list = []
                for population in source_population_list:
                    for source_gid in source_map[population]:
                        this_feature_dict = features_dict[population][source_gid]
                        selectivity_type = this_feature_dict['Selectivity Type'][0]
                        if selectivity_type == selectivity_grid:
                            rate_threshold = grid_peak_rate / 10.
                            ori_offset = this_feature_dict['Grid Orientation'][0]
                            grid_spacing = this_feature_dict['Grid Spacing'][0]
                            x_offset = this_feature_dict['X Offset'][0]
                            y_offset = this_feature_dict['Y Offset'][0]
                            rate = np.vectorize(grid_rate(grid_spacing, ori_offset, x_offset, y_offset))
                        elif selectivity_type == selectivity_place_field:
                            rate_threshold = place_peak_rate / 10.
                            field_width = this_feature_dict['Field Width'][0]
                            x_offset = this_feature_dict['X Offset'][0]
                            y_offset = this_feature_dict['Y Offset'][0]
                            rate = np.vectorize(place_rate(field_width, x_offset, y_offset))
                        this_plasticity_signal = np.sum(np.multiply(rate(x, y), this_plasticity_kernel)) * \
                                                 spatial_resolution
                        if this_plasticity_signal / plasticity_kernel_area > rate_threshold:
                            if population not in structured_weight_map:
                                structured_weight_map[population] = {}
                            structured_weight_map[population][source_gid] = this_plasticity_signal
                            plasticity_signal_list.append(this_plasticity_signal)
                plasticity_signal_min = np.min(plasticity_signal_list)
                plasticity_signal_amp = np.max(plasticity_signal_list) - plasticity_signal_min
                for population in structured_weight_map:
                    for source_gid in structured_weight_map[population]:
                        weight = 1.5 * (structured_weight_map[population][source_gid] - plasticity_signal_min) \
                                 / plasticity_signal_amp + 1.
                        for syn_id in source_map[population][source_gid]:
                            weight_map[syn_id] = weight
                print 'Rank: %i, %s gid: %i, ended with %i inputs' % (rank, target_population, gid, len(weight_map))
                structured_weights_dict[gid] = {'syn_id': np.array(weight_map.keys()).astype('uint32', copy=False),
                                                'weight': np.array(weight_map.values()).astype('float32', copy=False)}
                print 'Rank %i: took %.2f s to compute structured weights for %s gid %i (%i inputs)' % \
                      (rank, time.time() - local_time, target_population, gid, len(plasticity_signal_list))
                del plasticity_signal_list
            else:
                structured_weights_dict[gid] = weights_dict[weights_namespace]
                print 'Rank %i: took %.2f s; %s gid %i was not selected for structured weights' % \
                      (rank, time.time() - local_time, target_population, gid)
            count += 1
        if not debug:
            append_cell_attributes(MPI._addressof(comm), weights_path, target_population, structured_weights_dict,
                                   namespace=structured_weights_namespace, io_size=io_size, chunk_size=chunk_size,
                                   value_chunk_size=value_chunk_size)
        sys.stdout.flush()
        del source_map
        del weight_map
        del weights_dict
        del structured_weights_dict
        gc.collect()

    global_count = comm.gather(count, root=0)
    if rank == 0:
        print '%i ranks took %.2f s to compute synaptic weights for %i %s cells' % \
              (comm.size, time.time() - start_time, np.sum(global_count), target_population)


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])