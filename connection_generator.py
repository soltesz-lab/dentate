
import itertools
import numpy as np
from mpi4py import MPI
from neuroh5.io import NeurotreeAttrGen, bcast_cell_attributes, population_ranges, append_graph
import click
import utils

try:
    import mkl
    mkl.set_num_threads(1)
except:
    pass



def generate_uv_distance_connections(comm, destination, forest_path, connection_layers,
                                     synapse_namespace, connectivity_namespace,
                                     coords_path, coords_namespace,
                                     connectivity_seed,
                                     io_size, chunk_size, value_chunk_size, cache_size):
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
    rank = comm.rank  # The process ID (integer 0-3 for 4-process run)

    if io_size == -1:
        io_size = comm.size
    if rank == 0:
        print '%i ranks have been allocated' % comm.size
    sys.stdout.flush()

    start_time = time.time()

    soma_coords = {}
    source_populations = population_ranges(comm, coords_path).keys()
    for population in source_populations:
        soma_coords[population] = bcast_cell_attributes(comm, 0, coords_path, population,
                                                        namespace=coords_namespace)

    for population in soma_coords:
        for cell in soma_coords[population].itervalues():
            cell['u_index'] = get_array_index(u, cell['U Coordinate'][0])
            cell['v_index'] = get_array_index(v, cell['V Coordinate'][0])

    layer_set, swc_type_set, syn_type_set = set(), set(), set()
    for source in connection_layers[destination]:
        layer_set.update(connection_layers[destination][source])
        swc_type_set.update(swc_types[destination][source])
        syn_type_set.update(syn_types[destination][source])

    count = 0
    for destination_gid, attributes_dict in NeurotreeAttrGen(comm, forest_path, destination, io_size=io_size,
                                                             cache_size=cache_size, namespace=synapse_namespace):
        last_time = time.time()
        connection_dict = {}
        p_dict = {}
        source_gid_dict = {}
        if destination_gid is None:
            print 'Rank %i destination gid is None' % rank
        else:
            print 'Rank %i received attributes for destination: %s, gid: %i' % (rank, destination, destination_gid)
            synapse_dict = attributes_dict['Synapse Attributes']
            connection_dict[destination_gid] = {}
            local_np_random.seed(destination_gid + connectivity_seed)
            connection_dict[destination_gid]['source_gid'] = np.array([], dtype='uint32')
            connection_dict[destination_gid]['syn_id']     = np.array([], dtype='uint32')

            for layer in layer_set:
                for swc_type in swc_type_set:
                    for syn_type in syn_type_set:
                        sources, this_proportions = filter_sources(destination, layer, swc_type, syn_type)
                        if sources:
                            if rank == 0 and count == 0:
                                source_list_str = '[' + ', '.join(['%s' % xi for xi in sources]) + ']'
                                print 'Connections to destination: %s in layer: %i ' \
                                    '(swc_type: %i, syn_type: %i): %s' % \
                                    (destination, layer, swc_type, syn_type, source_list_str)
                            p, source_gid = np.array([]), np.array([])
                            for source, this_proportion in zip(sources, this_proportions):
                                if source not in source_gid_dict:
                                    this_p, this_source_gid = p_connect.get_p(destination, source, destination_gid, soma_coords,
                                                                              distance_U, distance_V)
                                    source_gid_dict[source] = this_source_gid
                                    p_dict[source] = this_p
                                else:
                                    this_source_gid = source_gid_dict[source]
                                    this_p = p_dict[source]
                                p = np.append(p, this_p * this_proportion)
                                source_gid = np.append(source_gid, this_source_gid)
                            syn_indexes = filter_synapses(synapse_dict, layer, swc_type, syn_type)
                            connection_dict[destination_gid]['syn_id'] = \
                                np.append(connection_dict[destination_gid]['syn_id'],
                                          synapse_dict['syn_id'][syn_indexes]).astype('uint32', copy=False)
                            this_source_gid = local_np_random.choice(source_gid, len(syn_indexes), p=p)
                            connection_dict[destination_gid]['source_gid'] = \
                                np.append(connection_dict[destination_gid]['source_gid'],
                                          this_source_gid).astype('uint32', copy=False)
            count += 1
            print 'Rank %i took %i s to compute connectivity for destination: %s, gid: %i' % (rank, time.time() - last_time,
                                                                                         destination, destination_gid)
            sys.stdout.flush()
        last_time = time.time()
        append_graph(comm, connectivity_path, source, destination, connection_dict, io_size=io_size)
        if rank == 0:
            print 'Appending connectivity for destination: %s took %i s' % (destination, time.time() - last_time)
        sys.stdout.flush()
        del connection_dict
        del p_dict
        del source_gid_dict
        gc.collect()

    global_count = comm.gather(count, root=0)
    if rank == 0:
        print '%i ranks took took %i s to compute connectivity for %i cells' % (comm.size, time.time() - start_time,
                                                                                np.sum(global_count))


