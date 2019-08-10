import sys, os, gc, pprint, time, click
from collections import defaultdict
from mpi4py import MPI
import numpy as np
import yaml
import dentate
import dentate.synapses as synapses
import dentate.utils as utils
from dentate.env import Env
from neuroh5.io import read_cell_attributes, read_graph_selection, read_population_ranges
import h5py

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
@click.option("--coords-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--coords-namespace", type=str, default='Coordinates')
@click.option("--distances-namespace", '-n', type=str, default='Arc Distances')
@click.option("--distance-limits", type=(float,float))
@click.option("--output-path", '-o', required=True, type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option("--io-size", type=int, default=-1)
@click.option("--verbose", "-v", is_flag=True)
def main(config, config_prefix, dataset_prefix, coords_path, coords_namespace, distances_namespace, distance_limits, output_path, io_size, verbose):

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

    output_dict = defaultdict(set)
    
    for population in pop_ranges:
        distances = read_cell_attributes(coords_path, population, namespace=distances_namespace, comm=comm, io_size=io_size)
        soma_distances = { k: (v['U Distance'][0], v['V Distance'][0]) for (k,v) in distances }
        del distances
        
        logger.info('read %s distances (%i elements)' % (population, len(list(soma_distances.keys()))))

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
        max_dist = U_max 
        if distance_limits:
            min_dist = distance_limits[0]
            max_dist = distance_limits[1]

        output_dict[population] = set([ k for k in distance_U if (distance_U[k] >= min_dist) and 
                                                   (distance_U[k] <= max_dist)  ])
    
    yaml_output_dict = {}
    for k, v in utils.viewitems(output_dict):
        logger.info('Rank %d: population %s: %d cells' % (comm.rank, k, len(v)))
        yaml_output_dict[k] = list(v)
         
    with open(output_path, 'w') as outfile:
        yaml.dump(yaml_output_dict, outfile)



if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv)+1):])
