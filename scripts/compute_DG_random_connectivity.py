from function_lib import *
from mpi4py import MPI
from neurotrees.io import append_cell_attributes
from neurotrees.io import NeurotreeAttrGen
from neurotrees.io import bcast_cell_attributes
from neurotrees.io import population_ranges
import click


try:
    import mkl
    mkl.set_num_threads(1)
except:
    pass


"""
Determine synaptic connectivity onto DG GCs randomly (serving as a null control for distance-dependent connectivity).

Algorithm:
1. For each cell population:
    i. Load the soma locations of each cell population from a NeuroIO file. The (U, V) coordinates of each cell are
        projected onto a plane in the middle of the granule cell layer (L = -1), and used to calculate the orthogonal 
        arc distances (S-T and M-L) between the projected soma locations.
2. For each cell, for each type of connection:
    i. Randomly sample (with replacement) connections from all possible sources with uniform probability.
    ii. Load from a NeuroIO file the synapses attributes, including layer, type, syn_loc, sec_type, and unique indexes 
        for each synapse.
    ii. Write to a NeuroIO file the source_gids and synapse_indexes for all the connections that have been
        specified in this step. Iterating through connection types will keep appending to this data structure.
    TODO: Implement a parallel write method to the separate connection edge data structure, use that instead.

swc_type_enumerator = {'soma': 1, 'axon': 2, 'basal': 3, 'apical': 4, 'trunk': 5, 'tuft': 6}
syn_type_enumerator = {'excitatory': 0, 'inhibitory': 1, 'neuromodulatory': 2}

"""

script_name = 'compute_DG_random_connectivity.py'


layer_Hilus = 0
layer_GCL   = 1
layer_IML   = 2
layer_MML   = 3
layer_OML   = 4 

layers = {'GC': {'MPP':  [layer_MML], 'LPP': [layer_OML], 'MC': [layer_IML],
                 'NGFC': [layer_MML, layer_OML], 'AAC': [layer_GCL], 'BC': [layer_GCL, layer_IML],
                 'MOPP': [layer_MML, layer_OML], 'HCC': [layer_IML], 'HC': [layer_MML, layer_OML]}}

syn_Exc = 0
syn_Inh = 1

syn_types = {'GC': {'MPP': [syn_Exc], 'LPP': [syn_Exc], 'MC': [syn_Exc],
                    'NGFC': [syn_Inh, syn_Inh], 'AAC': [syn_Inh], 'BC': [syn_Inh, syn_Inh],
                    'MOPP': [syn_Inh, syn_Inh], 'HCC': [syn_Inh], 'HC': [syn_Inh, syn_Inh]}}

swc_Dend = 4
swc_Axon = 2
swc_Soma = 1

swc_types = {'GC': {'MPP': [swc_Dend], 'LPP': [swc_Dend], 'MC': [swc_Dend],
                    'NGFC': [swc_Dend, swc_Dend], 'AAC': [swc_Axon], 'BC': [swc_Soma, swc_Dend],
                    'MOPP': [swc_Dend, swc_Dend], 'HCC': [swc_Dend], 'HC': [swc_Dend, swc_Dend]}}

# fraction of synapses matching this layer, syn_type, sec_type, and source
proportions = {'GC': {'MPP': [1.], 'LPP': [1.], 'MC': [1.],
                      'NGFC': [0.15, 0.15], 'AAC': [1.], 'BC': [1., 0.4],
                      'MOPP': [0.3, 0.3], 'HCC': [0.6], 'HC': [0.55, 0.55]}}


local_np_random = np.random.RandomState()
# make sure random seeds are not being reused for various types of stochastic sampling
connectivity_seed_offset = int(3 * 2e6)


def get_array_index_func(val_array, this_val):
    """

    :param val_array: array 
    :param this_val: float
    :return: int
    """
    indexes = np.where(val_array >= this_val)[0]
    if np.any(indexes):
        return indexes[0]
    else:
        return len(val_array) - 1


get_array_index = np.vectorize(get_array_index_func, excluded=[0])


def filter_sources(target, layer, swc_type, syn_type):
    """
    
    :param target: str 
    :param layer: int
    :param swc_type: int
    :param syn_type: int
    :return: list
    """
    source_list = []
    proportion_list = []
    for source in layers[target]:
        for i, this_layer in enumerate(layers[target][source]):
            if this_layer == layer:
                if swc_types[target][source][i] == swc_type:
                    if syn_types[target][source][i] == syn_type:
                        source_list.append(source)
                        proportion_list.append(proportions[target][source][i])
    if proportion_list and np.sum(proportion_list) != 1.:
        raise Exception('Proportions of synapses to target: %s, layer: %i, swc_type: %i, '
                        'syn_type: %i do not sum to 1' % (target, layer, swc_type, syn_type))
    return source_list, proportion_list


def filter_synapses(synapse_dict, layer, swc_type, syn_type):
    """
    
    :param synapse_dict: 
    :param layer: int 
    :param swc_type: int
    :param syn_type: int
    :return: 
    """
    return np.where((synapse_dict['layer'] == layer) & (synapse_dict['swc_type'] == swc_type)
                    & (synapse_dict['syn_type'] == syn_type))[0]


@click.command()
@click.option("--forest-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--connectivity-namespace", type=str, default='Random Connectivity')
@click.option("--coords-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--coords-namespace", type=str, default='Sorted Coordinates')
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--cache-size", type=int, default=50)
@click.option("--debug", is_flag=True)
def main(forest_path, connectivity_namespace, coords_path, coords_namespace, io_size, chunk_size, value_chunk_size,
         cache_size, debug):
    """

    :param forest_path:
    :param connectivity_namespace:
    :param coords_path:
    :param coords_namespace:
    :param io_size:
    :param chunk_size:
    :param value_chunk_size:
    :param cache_size:
    """
    # troubleshooting
    if False:
        forest_path = '../morphologies/DGC_forest_connectivity_20170508.h5'
        coords_path = '../morphologies/dentate_Full_Scale_Control_coords_selectivity_20170615.h5'
        coords_namespace = 'Coordinates'
        io_size = -1
        chunk_size = 1000
        value_chunk_size = 1000
        cache_size = 50

    comm = MPI.COMM_WORLD
    rank = comm.rank  # The process ID (integer 0-3 for 4-process run)

    if io_size == -1:
        io_size = comm.size
    if rank == 0:
        print '%i ranks have been allocated' % comm.size
    sys.stdout.flush()

    start_time = time.time()

    soma_coords = {}
    source_populations = population_ranges(MPI._addressof(comm), coords_path).keys()
    for population in source_populations:
        soma_coords[population] = bcast_cell_attributes(MPI._addressof(comm), 0, coords_path, population,
                                                            namespace=coords_namespace)

    target = 'GC'

    layer_set, swc_type_set, syn_type_set = set(), set(), set()
    for source in layers[target]:
        layer_set.update(layers[target][source])
        swc_type_set.update(swc_types[target][source])
        syn_type_set.update(syn_types[target][source])

    count = 0
    attr_gen = NeurotreeAttrGen(MPI._addressof(comm), forest_path, target, io_size=io_size, cache_size=cache_size,
                                namespace='Synapse_Attributes')
    if debug:
        attr_gen = [attr_gen.next()]
    for target_gid, attributes_dict in attr_gen:
        last_time = time.time()
        connection_dict = {}
        p_dict = {}
        source_gid_dict = {}
        if target_gid is None:
            print 'Rank %i target gid is None' % rank
        else:
            print 'Rank %i received attributes for target: %s, gid: %i' % (rank, target, target_gid)
            synapse_dict = attributes_dict['Synapse_Attributes']
            connection_dict[target_gid] = {}
            local_np_random.seed(target_gid + connectivity_seed_offset)
            connection_dict[target_gid]['source_gid'] = np.array([], dtype='uint32')
            connection_dict[target_gid]['syn_id'] = np.array([], dtype='uint32')

            for layer in layer_set:
                for swc_type in swc_type_set:
                    for syn_type in syn_type_set:
                        sources, this_proportions = filter_sources(target, layer, swc_type, syn_type)
                        if sources:
                            if rank == 0 and count == 0:
                                source_list_str = '[' + ', '.join(['%s' % xi for xi in sources]) + ']'
                                print 'Connections to target: %s in layer: %i ' \
                                    '(swc_type: %i, syn_type: %i): %s' % \
                                    (target, layer, swc_type, syn_type, source_list_str)
                            p, source_gid = np.array([]), np.array([])
                            for source, this_proportion in zip(sources, this_proportions):
                                if source not in source_gid_dict:
                                    this_source_gid = soma_coords[source].keys()
                                    this_p = np.ones(len(this_source_gid)) / float(len(this_source_gid))
                                    source_gid_dict[source] = this_source_gid
                                    p_dict[source] = this_p
                                else:
                                    this_source_gid = source_gid_dict[source]
                                    this_p = p_dict[source]
                                p = np.append(p, this_p * this_proportion)
                                source_gid = np.append(source_gid, this_source_gid)
                            syn_indexes = filter_synapses(synapse_dict, layer, swc_type, syn_type)
                            connection_dict[target_gid]['syn_id'] = \
                                np.append(connection_dict[target_gid]['syn_id'],
                                          synapse_dict['syn_id'][syn_indexes]).astype('uint32', copy=False)
                            this_source_gid = local_np_random.choice(source_gid, len(syn_indexes), p=p)
                            connection_dict[target_gid]['source_gid'] = \
                                np.append(connection_dict[target_gid]['source_gid'],
                                          this_source_gid).astype('uint32', copy=False)
            count += 1
            print 'Rank %i took %.2f s to compute connectivity for target: %s, gid: %i' % (rank,
                                                                                           time.time() - last_time,
                                                                                           target, target_gid)
            sys.stdout.flush()
        if not debug:
            append_cell_attributes(MPI._addressof(comm), forest_path, target, connection_dict,
                                   namespace=connectivity_namespace, io_size=io_size, chunk_size=chunk_size,
                                   value_chunk_size=value_chunk_size)
        sys.stdout.flush()
        del connection_dict
        del p_dict
        del source_gid_dict
        gc.collect()

    global_count = comm.gather(count, root=0)
    if rank == 0:
        print '%i ranks took took %.2f s to compute connectivity for %i cells' % (comm.size, time.time() - start_time,
                                                                                  np.sum(global_count))


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])
