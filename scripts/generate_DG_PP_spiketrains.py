
import sys, time, gc
import numpy as np
from mpi4py import MPI
import h5py
from neuroh5.io import NeuroH5CellAttrGen, append_cell_attributes, read_population_ranges
import random
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
@click.option("--selectivity-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--cache-size", type=int, default=50)
@click.option("--trajectory-id", type=int, default=0)
@click.option("--selectivity-namespace", type=str, default='Feature Selectivity')
@click.option("--stimulus-namespace", type=str, default='Vector Stimulus')
@click.option("--seed-offset", type=int, default=None)
@click.option("--debug", is_flag=True)
def main(selectivity_path, io_size, chunk_size, value_chunk_size, cache_size, trajectory_id, selectivity_namespace,
         stimulus_namespace, seed_offset, debug):
    """

    :param selectivity_path: str
    :param io_size: int
    :param chunk_size: int
    :param value_chunk_size: int
    :param cache_size: int
    :param trajectory_id: int
    :param selectivity_namespace: str
    :param stimulus_namespace: str
    :param seed_offset: int
    :param debug: bool
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank

    if io_size == -1:
        io_size = comm.size
    if rank == 0:
        print '%i ranks have been allocated' % comm.size
    sys.stdout.flush()

    local_random = random.Random()
    if seed_offset is None:
        seed_offset = 9 * 2e6

    arena_dimension = 100.  # minimum distance from origin to boundary (cm)
    spatial_resolution = 1.  # cm
    default_run_vel = 30.  # cm/s

    trajectory_namespace = 'Trajectory %s' % str(trajectory_id)
    stimulus_namespace += ' ' + str(trajectory_id)

    with h5py.File(selectivity_path, 'a') as f:
        if trajectory_namespace not in f:
            f.create_group(trajectory_namespace)
            x, y, d, t = stimulus.generate_trajectory(arena_dimension=arena_dimension, velocity=default_run_vel,
                                                      spatial_resolution=spatial_resolution)
            f[trajectory_namespace].create_group(str(trajectory_id))
            f[trajectory_namespace].create_dataset('x', dtype='float32', data=x)
            f[trajectory_namespace].create_dataset('y', dtype='float32', data=y)
            f[trajectory_namespace].create_dataset('d', dtype='float32', data=d)
            f[trajectory_namespace].create_dataset('t', dtype='float32', data=t)
        else:
            x = f[trajectory_namespace]['x'][:]
            y = f[trajectory_namespace]['y'][:]
            d = f[trajectory_namespace]['d'][:]
            t = f[trajectory_namespace]['t'][:]

    population_ranges = read_population_ranges(comm, selectivity_path)[0]

    for population in ['MPP', 'LPP']:
        population_start = population_ranges[population][0]

        count = 0
        start_time = time.time()

        selectivity_type = selectivity_type_dict[population]
        attr_gen = NeuroH5CellAttrGen(comm, selectivity_path, population, io_size=io_size,
                                      cache_size=cache_size, namespace=selectivity_namespace)
        if debug:
            attr_gen_wrapper = (attr_gen.next() for i in xrange(2))
        else:
            attr_gen_wrapper = attr_gen
        for gid, selectivity_dict in attr_gen_wrapper:
            local_time = time.time()
            response_dict = {}
            if gid is not None:
                response = stimulus.generate_spatial_ratemap(selectivity_type, selectivity_dict, x, y, d,
                                                             grid_peak_rate=20., place_peak_rate=20.)
                local_random.seed(int(seed_offset + gid))
                spiketrain = stgen.get_inhom_poisson_spike_times_by_thinning(response, t, generator=local_random)
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
                append_cell_attributes(comm, selectivity_path, population, response_dict,
                                       namespace=stimulus_namespace, io_size=io_size, chunk_size=chunk_size,
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
