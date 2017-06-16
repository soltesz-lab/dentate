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


script_name = 'predict_DG_GC_feature_selectivity.py'

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
@click.option("--connectivity-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--connectivity-namespace", type=str, default='Connectivity')
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--cache-size", type=int, default=50)
@click.option("--trajectory-id", type=int, default=0)
def main(features_path, connectivity_path, connectivity_namespace, io_size, chunk_size, value_chunk_size, cache_size,
         trajectory_id):
    """

    :param features_path:
    :param connectivity_path:
    :param connectivity_namespace:
    :param io_size:
    :param chunk_size:
    :param value_chunk_size:
    :param cache_size:
    :param trajectory_id:
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

    run_vel = 30.  # cm/s
    spatial_resolution = 1.  # cm
    x = np.arange(-100., 100., 1.)
    y = np.arange(-100., 100., 1.)
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
    count = 0
    start_time = time.time()
    for gid, connectivity_dict in NeurotreeAttrGen(MPI._addressof(comm), connectivity_path, target_population,
                                                       io_size=io_size, cache_size=cache_size,
                                                       namespace=connectivity_namespace):
        local_time = time.time()
        source_gid_counts = {}
        response_dict = {}
        if gid is not None:
            for population in ['MPP', 'LPP']:
                indexes = np.where((connectivity_dict[connectivity_namespace]['source_gid'][:] >=
                                    population_range_dict[population][0]) &
                                   (connectivity_dict[connectivity_namespace]['source_gid'][:] <
                                    population_range_dict[population][0] + population_range_dict[population][1]))[0]
                source_gid_counts[population] = \
                    Counter(connectivity_dict[connectivity_namespace]['source_gid'][:][indexes])
            response = np.zeros_like(d, dtype='float32')
            for population in ['MPP', 'LPP']:
                for source_gid in (source_gid for source_gid in source_gid_counts[population]
                                   if source_gid in features_dict[population]):
                    this_feature_dict = features_dict[population][source_gid]
                    selectivity_type = this_feature_dict['Selectivity Type'][0]
                    contact_count = source_gid_counts[population][source_gid]
                    if selectivity_type == selectivity_grid:
                        ori_offset = this_feature_dict['Grid Orientation'][0]
                        grid_spacing = this_feature_dict['Grid Spacing'][0]
                        x_offset = this_feature_dict['X Offset'][0]
                        y_offset = this_feature_dict['Y Offset'][0]
                        rate = np.vectorize(grid_rate(grid_spacing, ori_offset, x_offset, y_offset))
                        response = np.add(response, contact_count * rate(x, y), dtype='float32')
                    elif selectivity_type == selectivity_place_field:
                        field_width = this_feature_dict['Field Width'][0]
                        x_offset = this_feature_dict['X Offset'][0]
                        y_offset = this_feature_dict['Y Offset'][0]
                        rate = np.vectorize(place_rate(field_width, x_offset, y_offset))
                        response = np.add(response, contact_count * rate(x, y), dtype='float32')
            response_dict[gid] = {'waveform': response}
            print 'Rank %i: took %.2f s to compute predicted response for %s gid %i' % \
                  (rank, time.time() - local_time, target_population, gid)
            count += 1
        append_cell_attributes(MPI._addressof(comm), features_path, target_population, response_dict,
                            namespace=prediction_namespace, io_size=io_size, chunk_size=chunk_size,
                            value_chunk_size=value_chunk_size)
        sys.stdout.flush()
        del response
        del response_dict
        gc.collect()

    global_count = comm.gather(count, root=0)
    if rank == 0:
        print '%i ranks took %.2f s to compute selectivity parameters for %i %s cells' % \
              (comm.size, time.time() - start_time, np.sum(global_count), target_population)


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])