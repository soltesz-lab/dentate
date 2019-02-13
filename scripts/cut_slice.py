
import sys, os, time, gc, click, logging, yaml, pprint
from collections import defaultdict
import numpy as np
from mpi4py import MPI
from neuroh5.io import read_graph_selection, read_population_ranges
import h5py
from dentate.env import Env
import dentate.utils as utils
import dentate.synapses as synapses

sys_excepthook = sys.excepthook
def mpi_excepthook(type, value, traceback):
    sys_excepthook(type, value, traceback)
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)
sys.excepthook = mpi_excepthook

@click.command()
@click.option("--config", '-c', required=True, type=str)
@click.option("--config-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='config', help='path to directory containing network config files')
@click.option("--dataset-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
                help='path to directory containing required neuroh5 data files')
@click.option("--distances-namespace", '-n', type=str, default='Arc Distances')
@click.option("--distance-limits", type=(float,float), default=(-1,-1))
@click.option("--output-path", '-o', required=True, type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option("--io-size", type=int, default=-1)
@click.option("--verbose", "-v", is_flag=True)
def main(config, config_prefix, dataset_prefix, distances_namespace, distance_limits, output_path, population, io_size, verbose):

    utils.config_logging(verbose)
    logger = utils.get_script_logger(os.path.basename(__file__))

    comm = MPI.COMM_WORLD
    rank = comm.rank

    env = Env(comm=comm, config_file=config, config_prefix=config_prefix, dataset_prefix=dataset_prefix)

    if io_size == -1:
        io_size = comm.size
    if rank == 0:
        logger.info('%i ranks have been allocated' % comm.size)

    pop_ranges, pop_size = read_population_ranges(env.connectivity_file_path, comm=comm)

    distance_U_dict = {}
    distance_V_dict = {}
    range_U_dict = {}
    range_V_dict = {}

    gid_ranges = {}
    
    for population in pop_ranges.keys():
        distances = read_cell_attributes(coords_path, population, namespace=distances_namespace)
        soma_distances = { k: (v['U Distance'][0], v['V Distance'][0]) for (k,v) in distances }
        del distances
        
        logger.info('read distances (%i elements)' % len(list(soma_distances.keys())))
        distance_U_array = np.asarray([soma_distances[gid][0] for gid in soma_distances])
        distance_V_array = np.asarray([soma_distances[gid][1] for gid in soma_distances])

        U_min = np.min(distance_U_array)
        U_max = np.max(distance_U_array)
        V_min = np.min(distance_V_array)
        V_max = np.max(distance_V_array)

        range_U_dict[population] = (U_min, U_max)
        range_V_dict[population] = (V_min, V_max)
        
        distance_U = { gid: soma_distances[gid][0] for gid in soma_distances }
        distance_V = { gid: soma_distances[gid][1] for gid in soma_distances }

        distance_U_dict[population] = distance_U
        distance_V_dict[population] = distance_V

        min_dist = U_min
        if distance_limits[0] > -1:
            min_dist = distance_limits[0]
        max_dist = U_max 
        if distance_limits[1] > -1:
            max_dist = distance_limits[1]

        gid_ranges[population] = [ k if (distance_U[k] >= min_dist) and
                                    (distance_U[k] <= max_dist) for k in distance_U ]

    output_dict = defaultdict(set)
    for population in gid_ranges:
        postsyn_name = population
        presyn_names = env.projection_dict[population]
        selection_projections = [ (presyn_name, postsyn_name) for presyn_name in presyn_names ]
        
        (graph, a) = read_graph_selection(env.connectivity_file_path, 
                                          selection=gid_ranges[population], comm=env.comm)


    
        for presyn_name in presyn_names:
            edge_iter = graph[postsyn_name][presyn_name]
            for (postsyn_gid, edges) in edge_iter:
                presyn_gids, edge_attrs = edges
                output_dict[postsyn_name].add(int(postsyn_gid))
                output_dict[presyn_name].update(np.asarray(presyn_gids,dtype=np.intp).tolist())

    yaml_output_dict = {}
    for k, v in utils.viewitems(output_dict):
        logger.info('Rank %d: population %s: %d cells' % (comm.rank, k, len(v)))
        yaml_output_dict[k] = list(v)
         
    with open(output_path, 'w') as outfile:
        yaml.dump(yaml_output_dict, outfile)

    MPI.Finalize()


if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv)+1):])

