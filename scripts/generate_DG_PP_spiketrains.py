
import sys, time, gc
import numpy as np
import mpi4py
from mpi4py import MPI
from neuroh5.io import NeuroH5CellAttrGen, append_cell_attributes, read_population_ranges
import click
import stimulus, stgen, utils

try:
    import mkl
    mkl.set_num_threads(1)
except:
    pass


script_name = 'generate_DG_PP_spiketrains.py'

selectivity_type_dict = {'MPP': stimulus.selectivity_grid, 'LPP': stimulus.selectivity_place_field}


@click.command()
@click.option("--features-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--cache-size", type=int, default=50)
@click.option("--trajectory-id", type=int, default=0)
@click.option("--seed", type=int, default=3)
@click.option("--debug", is_flag=True)
def main(features_path, io_size, chunk_size, value_chunk_size, cache_size, trajectory_id, seed, debug):
    """

    :param features_path:
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

    interp_x, interp_y, d, t = stimulus.generate_trajectory()
    
    trajectory_namespace = 'Trajectory %s' % str(trajectory_id)
    
    # with h5py.File(features_path, 'a', driver='mpio', comm=comm) as f:
    #     if 'Trajectories' not in f:
    #         f.create_group('Trajectories')
    #     if str(trajectory_id) not in f['Trajectories']:
    #         f['Trajectories'].create_group(str(trajectory_id))
    #         f['Trajectories'][str(trajectory_id)].create_dataset('x', dtype='float32', data=interp_x)
    #         f['Trajectories'][str(trajectory_id)].create_dataset('y', dtype='float32', data=interp_y)
    #         f['Trajectories'][str(trajectory_id)].create_dataset('d', dtype='float32', data=interp_distance)
    #         f['Trajectories'][str(trajectory_id)].create_dataset('t', dtype='float32', data=t)
    #     x = f['Trajectories'][str(trajectory_id)]['x'][:]
    #     y = f['Trajectories'][str(trajectory_id)]['y'][:]
    #     d = f['Trajectories'][str(trajectory_id)]['d'][:]

    t_stop = t[-1]

    features_namespace = 'Feature Selectivity'
    spiketrain_namespace = 'Vector Stimulus %s' % str(trajectory_id)

    populations = selectivity_type_dict.keys()

    population_ranges = read_population_ranges(comm, features_path)[0]
    stg = stgen.StGen(seed=seed)

    for population in ['MPP', 'LPP']:
        (population_start, _) = population_ranges[population]

        count = 0
        start_time = time.time()
        
        selectivity_type = selectivity_type_dict[population]
        attr_gen = NeuroH5CellAttrGen(comm, features_path, population, io_size=io_size,
                                      cache_size=cache_size, namespace=features_namespace)
        if debug:
            attr_gen_wrapper = (attr_gen.next() for i in xrange(2))
        else:
            attr_gen_wrapper = attr_gen
        for gid, features_dict in attr_gen_wrapper:
            local_time = time.time()
            response_dict = {}
            if gid is not None:
                response = stimulus.generate_spatial_ratemap(selectivity_type, features_dict, interp_x, interp_y, d)
                # TODO: replace with generator with refractory period, reset seed for each gid
                spiketrain = stg.inh_poisson_generator(response, t, t_stop)
                response_dict[gid-population_start] = {'rate': response,
                                                       'spiketrain': np.asarray(spiketrain, dtype='float32')}
                baseline = np.mean(response[np.where(response <= np.percentile(response, 10.))[0]])
                peak = np.mean(response[np.where(response >= np.percentile(response, 90.))[0]])
                modulation = 0. if peak <= 0.1 else (peak - baseline) / peak
                peak_index = np.where(response == np.max(response))[0][0]
                response_dict[gid-population_start]['modulation'] = np.array([modulation], dtype='float32')
                response_dict[gid-population_start]['peak_index'] = np.array([peak_index], dtype='uint32')
                print 'Rank %i: took %.2f s to compute spike trains for %s gid %i' % \
                      (rank, time.time() - local_time, population, gid)
                count += 1
            if not debug:
                append_cell_attributes(comm, features_path, population, response_dict,
                                       namespace=spiketrain_namespace, io_size=io_size, chunk_size=chunk_size,
                                       value_chunk_size=value_chunk_size)
            sys.stdout.flush()
            del response
            del response_dict
            gc.collect()

        global_count = comm.gather(count, root=0)
        if rank == 0:
            print '%i ranks took %.2f s to compute spike trains for %i %s cells' % \
                  (comm.size, time.time() - start_time, np.sum(global_count), population)


if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])
