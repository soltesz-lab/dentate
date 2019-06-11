import sys, os, time, gc, click, logging
from collections import defaultdict
# from itertools import zip_longest
import numpy as np
from mpi4py import MPI
import neuroh5
from neuroh5.io import append_cell_attributes, read_population_ranges, bcast_cell_attributes, \
    read_cell_attribute_selection, NeuroH5ProjectionGen
from dentate.env import Env
from dentate import utils, stimulus
from dentate.utils import *


"""
stimulus_path: contains namespace with 1D spatial rate map attribute ('rate')
weights_path: contains namespace with initial weights ('Weights'), applied plasticity rule and writes new weights to
 'Structured Weights' namespace
connections_path: contains existing mapping of syn_id to source_gid

10% of GCs will have a subset of weights modified according to a slow time-scale plasticity rule, the rest inherit the
    unaltered initial log-normal weights
    
TODO: Rather than choosing peak_locs randomly, have the peak_locs depend on the previous weight distribution.
"""

script_name = 'generate_structured_weights_as_cell_attr.py'

local_random = np.random.RandomState()

# look up table for type of feature selectivity
selectivity_type_dict = {'MPP': stimulus.selectivity_grid, 'LPP': stimulus.selectivity_place_field}
peak_rate_dict = {'MPP': 20., 'LPP': 20.}  # Hz


@click.command()
@click.option("--config", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--stimulus-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--stimulus-namespace", type=str, default='Vector Stimulus')
@click.option("--weights-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--initial-weights-namespace", type=str, default='Weights')
@click.option("--structured-weights-namespace", type=str, default='Structured Weights')
@click.option("--connections-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--destination", '-d', type=str)
@click.option("--sources", '-s', type=str, multiple=True)
@click.option("--stimulus-id", type=int, default=0)
@click.option("--target-sparsity", type=float, default=0.1)
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--cache-size", type=int, default=50)
@click.option("--write-size", type=int, default=1)
@click.option("--verbose", "-v", is_flag=True)
@click.option("--dry-run", is_flag=True)
def main(config, stimulus_path, stimulus_namespace, weights_path, initial_weights_namespace,
         structured_weights_namespace, connections_path, destination, sources, stimulus_id, target_sparsity, io_size,
         chunk_size, value_chunk_size, cache_size, write_size, verbose, dry_run):
    """

    :param config: str (path to .yaml file)
    :param stimulus_path: str (path to .h5 file)
    :param stimulus_namespace: str
    :param weights_path: str (path to .h5 file)
    :param initial_weights_namespace: str
    :param structured_weights_namespace: str
    :param connections_path: str (path to .h5 file)
    :param destination: str
    :param sources: list of str
    :param stimulus_id: int
    :param target_sparsity: float
    :param io_size:
    :param chunk_size:
    :param value_chunk_size:
    :param cache_size:
    :param write_size:
    :param verbose:
    :param dry_run:
    :return:
    """
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
    :param stimulus_id: int
    :param target_sparsity: float
    :param dry_run:  bool
    """

    utils.config_logging(verbose)
    logger = utils.get_script_logger(script_name)

    comm = MPI.COMM_WORLD
    rank = comm.rank

    env = Env(comm=comm, config_file=config)

    if io_size == -1:
        io_size = comm.size
    if rank == 0:
        logger.info('%s: %i ranks have been allocated' % (script_name, comm.size))

    stimulus_namespace += ' ' + str(stimulus_id)

    stimulus_attrs = {}
    for source in sources:
        stimulus_attr_gen = bcast_cell_attributes(stimulus_path, source, namespace=stimulus_namespace, root=0, comm=comm)
        stimulus_attrs[source] = { gid: attr_dict for gid, attr_dict in stimulus_attr_gen }

    trajectory_namespace = 'Trajectory %s' % str(stimulus_id)

    seed_offset = int(env.modelConfig['Random Seeds']['GC Structured Weights'])

    input_config = env.inputConfig[stimulus_id]
    arena_dimension = int(input_config['trajectory']['Distance to boundary'])  # minimum distance from origin to boundary (cm)
    default_run_vel = input_config['trajectory']['Default run velocity']  # cm/s
    spatial_resolution = input_config['trajectory']['Spatial resolution']  # cm

    if rank == 0:
        import h5py
        with h5py.File(stimulus_path) as f:
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
    comm.barrier()

    x = comm.bcast(x, root=0)
    y = comm.bcast(y, root=0)
    d = comm.bcast(d, root=0)
    t = comm.bcast(t, root=0)
    
    plasticity_window_dur = 4.  # s
    plasticity_kernel_sigma = plasticity_window_dur * default_run_vel / 3. / np.sqrt(2.)  # cm
    plasticity_kernel = lambda d, d_offset: np.exp(-(old_div((d - d_offset), plasticity_kernel_sigma)) ** 2.)
    plasticity_kernel = np.vectorize(plasticity_kernel, excluded=[1])
    max_plasticity_kernel_area = np.sum(plasticity_kernel(d, np.max(d) / 2.)) * spatial_resolution

    count = 0
    gid_count = 0
    structured_count = 0
    start_time = time.time()

    connection_gen_list = []
    for source in sources:
        connection_gen_list.append(NeuroH5ProjectionGen(connections_path, source, destination, namespaces=['Synapses'],
                                                        comm=comm))

    structured_weights_dict = {}
    for itercount, attr_gen_package in enumerate(zip_longest(*connection_gen_list)):
        local_time = time.time()
        syn_weight_dict = {}
        source_syn_map = defaultdict(list)
        syn_peak_index_map = {}
        modulated_inputs = 0
        source_gid_array = None
        conn_attr_dict = None
        destination_gid = attr_gen_package[0][0]
        if not all([attr_gen_items[0] == destination_gid for attr_gen_items in attr_gen_package]):
            raise Exception('Rank: %i; destination: %s; destination_gid not matched across multiple attribute '
                            'generators: %s' % (rank, destination,
                                                [attr_gen_items[0] for attr_gen_items in attr_gen_package]))
            sys.stdout.flush()
        
        if destination_gid is not None:
            itval = read_cell_attribute_selection(weights_path, destination, selection=[destination_gid],
                                                   namespace=initial_weights_namespace, comm=comm)
            _, initial_weights_dict = next(itval)
            syn_weight_dict = { int(syn_id): float(weight) for (syn_id, weight) in 
                                zip(np.nditer(initial_weights_dict['syn_id']),
                                     np.nditer(initial_weights_dict['weight'])) }
            local_random.seed(int(destination_gid + seed_offset))
            for this_destination_gid, (source_gid_array, conn_attr_dict) in attr_gen_package:
                for i in range(len(source_gid_array)):
                    this_source_gid = source_gid_array[i]
                    this_syn_id = conn_attr_dict['Synapses']['syn_id'][i]
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
                    peak_index = stimulus_attrs[source][this_source_gid]['peak index'][0]
                    if modify_weights:
                        norm_rate = old_div(stimulus_attrs[source][this_source_gid]['rate'], peak_rate)
                        this_plasticity_signal = old_div(np.sum(np.multiply(norm_rate, this_plasticity_kernel)) * \
                                                 spatial_resolution, max_plasticity_kernel_area)
                        delta_weight = 2. * this_plasticity_signal
                    else:
                        delta_weight = 0.
                    for this_syn_id in source_syn_map[this_source_gid]:
                        syn_peak_index_map[this_syn_id] = peak_index
                        if delta_weight >= 0.1:
                            modulated_inputs += 1
                        syn_weight_dict[this_syn_id] += delta_weight
            structured_weights_dict[destination_gid] = \
                {'syn_id': np.array(list(syn_peak_index_map.keys())).astype('uint32', copy=False),
                 'weight': np.array([syn_weight_dict[syn_id] for syn_id in syn_peak_index_map]).astype('float32',
                                                                                                      copy=False),
                 'peak_index': np.array(list(syn_peak_index_map.values())).astype('uint32', copy=False),
                 'structured': np.array([int(modify_weights)], dtype='uint32')}
            if modify_weights:
                logger.info('Rank %i; destination: %s; gid %i; generated structured weights for %i/%i inputs in %.2f '
                            's' % (rank, destination, destination_gid, modulated_inputs, len(syn_weight_dict),
                                   time.time() - local_time))
                structured_count += 1
            else:
                logger.info('Rank %i; destination: %s; gid %i; calculated input peak_locs for %i inputs in %.2f s (not'
                            ' selected for structured weights)' %
                            (rank, destination, destination_gid, len(syn_weight_dict), time.time() - local_time))
            count += 1
        else:
            logger.info('Rank: %i received destination_gid as None' % rank)
        gid_count += 1
        if gid_count % write_size == 0:
            if not dry_run:
                append_cell_attributes(weights_path, destination, structured_weights_dict,
                                       namespace=structured_weights_namespace, comm=comm, io_size=io_size,
                                       chunk_size=chunk_size, value_chunk_size=value_chunk_size)
            structured_weights_dict.clear()
        del syn_weight_dict
        del source_syn_map
        del syn_peak_index_map
        del modulated_inputs
        del source_gid_array
        del conn_attr_dict
        # gc.collect()
    if not dry_run:
        append_cell_attributes(weights_path, destination, structured_weights_dict,
                               namespace=structured_weights_namespace, comm=comm, io_size=io_size,
                               chunk_size=chunk_size, value_chunk_size=value_chunk_size)
    global_count = comm.gather(count, root=0)
    global_structured_count = comm.gather(structured_count, root=0)
    if rank == 0:
        logger.info('destination: %s; %i ranks processed %i cells (%i assigned structured weights) in %.2f s' %
                    (destination, comm.size, np.sum(global_count), np.sum(global_structured_count),
                     time.time() - start_time))
    MPI.Finalize()


if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])
