from function_lib import *
from itertools import izip
from collections import defaultdict
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


script_name = 'predict_DG_GC_feature_selectivity_weighted.py'

example_features_path = '../morphologies/dentate_Full_Scale_Control_selectivity_20170615.h5'
example_connectivity_path = '../morphologies/DGC_forest_connectivity_20170427.h5'

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
@click.option("--debug", is_flag=True)
def main(features_path, weights_path, weights_namespace, connectivity_path, connectivity_namespace, io_size,
         chunk_size, value_chunk_size, cache_size, trajectory_id, debug):
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
    :param debug:
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank

    if io_size == -1:
        io_size = comm.size
    if rank == 0:
        print '%i ranks have been allocated' % comm.size
    sys.stdout.flush()

    population_range_dict = population_ranges(MPI._addressof(comm), features_path)

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

    prediction_namespace = 'Response Prediction '+str(trajectory_id)

    target_population = 'GC'
    source_population_list = ['MPP', 'LPP']
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
        response_dict = {}
        response = np.zeros_like(d, dtype='float32')
        if gid is not None:
            if gid != weights_gid:
                raise Exception('gid %i from connectivity_gen does not match gid %i from weights_gen') % \
                      (gid, weights_gid)
            weight_map = dict(zip(weights_dict[weights_namespace]['syn_id'],
                                  weights_dict[weights_namespace]['weight']))
            for population in source_population_list:
                source_map[population] = defaultdict(list)
            for i in xrange(len(connectivity_dict[connectivity_namespace]['source_gid'])):
                source_gid = connectivity_dict[connectivity_namespace]['source_gid'][i]
                population = gid_in_population_list(source_gid, source_population_list, population_range_dict)
                if population is not None:
                    syn_id = connectivity_dict[connectivity_namespace]['syn_id'][i]
                    source_map[population][source_gid].append(syn_id)
            for population in source_population_list:
                for source_gid in source_map[population]:
                    this_feature_dict = features_dict[population][source_gid]
                    selectivity_type = this_feature_dict['Selectivity Type'][0]
                    if selectivity_type == selectivity_grid:
                        ori_offset = this_feature_dict['Grid Orientation'][0]
                        grid_spacing = this_feature_dict['Grid Spacing'][0]
                        x_offset = this_feature_dict['X Offset'][0]
                        y_offset = this_feature_dict['Y Offset'][0]
                        rate = np.vectorize(grid_rate(grid_spacing, ori_offset, x_offset, y_offset))
                    elif selectivity_type == selectivity_place_field:
                        field_width = this_feature_dict['Field Width'][0]
                        x_offset = this_feature_dict['X Offset'][0]
                        y_offset = this_feature_dict['Y Offset'][0]
                        rate = np.vectorize(place_rate(field_width, x_offset, y_offset))
                    this_rate = rate(x, y)
                    for syn_id in source_map[population][source_gid]:
                        weight = weight_map[syn_id]
                        response = np.add(response, weight * this_rate, dtype='float32')
            response_dict[gid] = {'waveform': response}
            baseline = np.mean(response[np.where(response <= np.percentile(response, 10.))[0]])
            peak = np.mean(response[np.where(response >= np.percentile(response, 90.))[0]])
            modulation = 0. if peak <= 0.1 else (peak - baseline) / peak
            peak_index = np.where(response == np.max(response))[0][0]
            response_dict[gid]['modulation'] = np.array([modulation], dtype='float32')
            response_dict[gid]['peak_index'] = np.array([peak_index], dtype='uint32')
            print 'Rank %i: took %.2f s to compute predicted response for %s gid %i' % \
                  (rank, time.time() - local_time, target_population, gid)
            count += 1
        if not debug:
            append_cell_attributes(MPI._addressof(comm), features_path, target_population, response_dict,
                            namespace=prediction_namespace, io_size=io_size, chunk_size=chunk_size,
                            value_chunk_size=value_chunk_size)
        sys.stdout.flush()
        del response
        del response_dict
        del source_map
        del weight_map
        gc.collect()

    global_count = comm.gather(count, root=0)
    if rank == 0:
        print '%i ranks took %.2f s to compute predicted response parameters for %i %s cells' % \
              (comm.size, time.time() - start_time, np.sum(global_count), target_population)


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])
