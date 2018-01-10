
import sys, time, gc
from mpi4py import MPI
import h5py
from neuroh5.io import NeuroH5CellAttrGen, append_cell_attributes, read_population_ranges, bcast_cell_attributes, \
    NeuroH5ProjectionGen
from neuroh5_io_utils import *
import numpy as np
from collections import defaultdict
import click
from utils import *
import stimulus
from itertools import izip_longest

"""
stimulus_path: contains namespace with 1D spatial rate map attribute ('rate')
weights_path: contains namespace with initial weights ('Weights'), applied plasticity rule and writes new weights to
 'Structured Weights' namespace
connections_path: contains existing mapping of syn_id to source_gid

10% of GCs will have a subset of weights modified according to a slow time-scale plasticity rule, the rest inherit the
    unaltered initial log-normal weights
    
TODO: Rather than choosing peak_locs randomly, have the peak_locs depend on the previous weight distribution.
"""

try:
    import mkl
    mkl.set_num_threads(1)
except:
    pass

script_name = 'generate_DG_GC_structured_weights_as_cell_attr.py'

local_random = np.random.RandomState()

# look up table for type of feature selectivity
selectivity_type_dict = {'MPP': stimulus.selectivity_grid, 'LPP': stimulus.selectivity_place_field}
peak_rate_dict = {'MPP': 20., 'LPP': 20.}  # Hz


@click.command()
@click.option("--stimulus-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--stimulus-namespace", type=str, default='Vector Stimulus')
@click.option("--weights-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--initial-weights-namespace", type=str, default='Weights')
@click.option("--structured-weights-namespace", type=str, default='Structured Weights')
@click.option("--connections-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--cache-size", type=int, default=50)
@click.option("--trajectory-id", type=int, default=0)
@click.option("--seed-offset", type=int, default=6)
@click.option("--target-sparsity", type=float, default=0.1)
@click.option("--debug", is_flag=True)
def main(stimulus_path, stimulus_namespace, weights_path, initial_weights_namespace, structured_weights_namespace,
         connections_path, io_size, chunk_size, value_chunk_size, cache_size, trajectory_id, seed_offset,
         target_sparsity, debug):
    """

    :param stimulus_path: str
    :param stimulus_namespace: str
    :param weights_path: str
    :param initial_weights_namespace: str
    :param structured_weights_namespace: str
    :param connections_path: str
    :param io_size: int
    :param chunk_size: int
    :param value_chunk_size: int
    :param cache_size: int
    :param trajectory_id: int
    :param seed_offset: int
    :param target_sparsity: float
    :param debug:  bool
    """
    # make sure random seeds are not being reused for various types of stochastic sampling
    seed_offset *= 2e6

    comm = MPI.COMM_WORLD
    rank = comm.rank

    if io_size == -1:
        io_size = comm.size
    if rank == 0:
        print '%s: %i ranks have been allocated' % (script_name, comm.size)
    sys.stdout.flush()

    stimulus_namespace += ' ' + str(trajectory_id)

    stimulus_attrs = {}
    source_population_list = ['MPP', 'LPP']
    for source in source_population_list:
        stimulus_attr_gen = bcast_cell_attributes(comm, 0, stimulus_path, source, namespace=stimulus_namespace)
        stimulus_attrs[source] = {gid: attr_dict for gid, attr_dict in stimulus_attr_gen}

    trajectory_namespace = 'Trajectory %s' % str(trajectory_id)

    arena_dimension = 100.  # minimum distance from origin to boundary (cm)
    default_run_vel = 30.  # cm/s
    spatial_resolution = 1.  # cm

    with h5py.File(stimulus_path, 'a', driver='mpio', comm=comm) as f:
        if trajectory_namespace not in f:
            print 'Rank: %i; Creating %s datasets' % (rank, trajectory_namespace)
            group = f.create_group(trajectory_namespace)
            t, x, y, d = stimulus.generate_trajectory(arena_dimension=arena_dimension, velocity=default_run_vel,
                                                      spatial_resolution=spatial_resolution)
            for key, value in zip(['x', 'y', 'd', 't'], [x, y, d, t]):
                dataset = group.create_dataset(key, (value.shape[0],), dtype='float32')
                with dataset.collective:
                    dataset[:] = value.astype('float32', copy=False)
        else:
            print 'Rank: %i; Reading %s datasets' % (rank, trajectory_namespace)
            group = f[trajectory_namespace]
            dataset = group['x']
            with dataset.collective:
                x = dataset[:]
            dataset = group['y']
            with dataset.collective:
                y = dataset[:]
            dataset = group['d']
            with dataset.collective:
                d = dataset[:]
            dataset = group['t']
            with dataset.collective:
                t = dataset[:]

    plasticity_window_dur = 4.  # s
    plasticity_kernel_sigma = plasticity_window_dur * default_run_vel / 3. / np.sqrt(2.)  # cm
    plasticity_kernel = lambda d, d_offset: np.exp(-((d - d_offset) / plasticity_kernel_sigma) ** 2.)
    plasticity_kernel = np.vectorize(plasticity_kernel, excluded=[1])
    max_plasticity_kernel_area = np.sum(plasticity_kernel(d, np.max(d) / 2.)) * spatial_resolution

    target = 'GC'

    pop_ranges, pop_size = read_population_ranges(comm, stimulus_path)
    target_gid_offset = pop_ranges[target][0]

    count = 0
    structured_count = 0
    start_time = time.time()

    gid_index_map = get_cell_attributes_gid_index_map(comm, weights_path, target, initial_weights_namespace)

    connection_gen_list = []
    for source in source_population_list:
        connection_gen_list.append(NeuroH5ProjectionGen(comm, connections_path, source, target, io_size=io_size,
                                                        cache_size=cache_size, namespaces=['Synapses']))

    maxiter = 100 if debug else None
    for itercount, attr_gen_package in enumerate(izip_longest(*connection_gen_list)):
        local_time = time.time()
        syn_weight_map = {}
        source_syn_map = defaultdict(list)
        syn_peak_index_map = {}
        structured_weights_dict = {}
        modulated_inputs = 0
        source_gid_array = None
        conn_attr_dict = None
        target_gid = attr_gen_package[0][0]
        if not all([attr_gen_items[0] == target_gid for attr_gen_items in attr_gen_package]):
            raise Exception('Rank: %i; target: %s; target_gid not matched across multiple attribute generators: %s' %
                            (rank, target, [attr_gen_items[0] for attr_gen_items in attr_gen_package]))
            sys.stdout.flush()
        # else:
        #    print 'Rank: %i; received target: %s; target_gid: %s' % (rank, target, str(target_gid))
        initial_weights_dict = get_cell_attributes_by_gid(target_gid, comm, weights_path, gid_index_map, target,
                                                          initial_weights_namespace, target_gid_offset)
        if target_gid is not None:
            if initial_weights_dict is None:
                raise Exception('Rank: %i; target: %s; target_gid: %s; get_cell_attributes_by_gid didn\'t work' %
                                (rank, target, str(target_gid)))
            local_random.seed(int(target_gid + seed_offset))
            syn_weight_map = dict(zip(initial_weights_dict['syn_id'], initial_weights_dict['weight']))
            for this_target_gid, (source_gid_array, conn_attr_dict) in attr_gen_package:
                for i in xrange(len(source_gid_array)):
                    this_source_gid = source_gid_array[i]
                    this_syn_id = conn_attr_dict['Synapses'][0][i]
                    source_syn_map[this_source_gid].append(this_syn_id)
            if local_random.uniform() <= target_sparsity:
                modify_weights = True
                peak_loc = local_random.choice(d)
                this_plasticity_kernel = plasticity_kernel(d, peak_loc)
            else:
                modify_weights = False
            for source in stimulus_attrs:
                peak_rate = peak_rate_dict[source]
                for this_source_gid in stimulus_attrs[source]:
                    peak_index = stimulus_attrs[source][this_source_gid]['peak_index'][0]
                    if modify_weights:
                        norm_rate = stimulus_attrs[source][this_source_gid]['rate'] / peak_rate
                        this_plasticity_signal = np.sum(np.multiply(norm_rate, this_plasticity_kernel)) * \
                                                 spatial_resolution / max_plasticity_kernel_area
                        delta_weight = 2. * this_plasticity_signal
                    else:
                        delta_weight = 0.
                    for this_syn_id in source_syn_map[this_source_gid]:
                        syn_peak_index_map[this_syn_id] = peak_index
                        if delta_weight >= 0.1:
                            modulated_inputs += 1
                        syn_weight_map[this_syn_id] += delta_weight
            structured_weights_dict[target_gid - target_gid_offset] = \
                {'syn_id': np.array(syn_peak_index_map.keys()).astype('uint32', copy=False),
                 'weight': np.array([syn_weight_map[syn_id] for syn_id in syn_peak_index_map]).astype('float32',
                                                                                                      copy=False),
                 'peak_index': np.array(syn_peak_index_map.values()).astype('uint32', copy=False),
                 'structured': np.array([int(modify_weights)], dtype='uint32')}
            if modify_weights:
                print 'Rank %i; target: %s; gid %i; generated structured weights for %i/%i inputs in %.2f s' % \
                      (rank, target, target_gid, modulated_inputs, len(syn_weight_map), time.time() - local_time)
                structured_count += 1
            else:
                print 'Rank %i; target: %s; gid %i; calculated input peak_locs for %i inputs in %.2f s (not selected ' \
                      'for structured weights)' % (rank, target, target_gid, len(syn_weight_map),
                                                   time.time() - local_time)
            count += 1
        else:
            print 'Rank: %i received target_gid as None' % rank
        if not debug:
            append_cell_attributes(comm, weights_path, target, structured_weights_dict,
                                   namespace=structured_weights_namespace, io_size=io_size, chunk_size=chunk_size,
                                   value_chunk_size=value_chunk_size)
        sys.stdout.flush()
        del syn_weight_map
        del source_syn_map
        del syn_peak_index_map
        del structured_weights_dict
        del modulated_inputs
        del source_gid_array
        del conn_attr_dict
        gc.collect()
        if debug:
            comm.barrier()
            if maxiter is not None and itercount > maxiter:
                break
    if debug:
        print 'Rank: %i exited the loop' % rank
    global_count = comm.gather(count, root=0)
    global_structured_count = comm.gather(structured_count, root=0)
    if rank == 0:
        print 'target: %s; %i ranks processed %i cells (%i assigned structured weights) in %.2f s' % \
              (target, comm.size, np.sum(global_count), np.sum(global_structured_count), time.time() - start_time)


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])
