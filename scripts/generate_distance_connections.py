##
## Generates distance-weighted random connectivity between the specified populations.
##

import gc, logging, os, os.path, sys
import click
from mpi4py import MPI
import h5py
import numpy as np
import dentate
import dentate.utils as utils
import rbf
import rbf.basis
from rbf.interpolate import RBFInterpolant
from dentate.connection_generator import ConnectionProb, generate_uv_distance_connections
from dentate.env import Env
from dentate.geometry import make_distance_interpolant, measure_distances
from dentate.neuron_utils import configure_hoc_env
from neuroh5.io import bcast_cell_attributes, read_cell_attributes, read_population_names, read_population_ranges

sys_excepthook = sys.excepthook
def mpi_excepthook(type, value, traceback):
     sys_excepthook(type, value, traceback)
     if MPI.COMM_WORLD.size > 1:
         MPI.COMM_WORLD.Abort(1)
#sys.excepthook = mpi_excepthook


@click.command()
@click.option("--config", required=True, type=str)
@click.option("--config-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True), default='config')
@click.option("--forest-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--connectivity-path", required=True, type=click.Path())
@click.option("--connectivity-namespace", type=str, default='Connectivity')
@click.option("--coords-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--coords-namespace", type=str, default='Coordinates')
@click.option("--synapses-namespace", type=str, default='Synapse Attributes')
@click.option("--distances-namespace", type=str, default='Arc Distances')
@click.option("--resolution", type=(int,int,int), default=(30,30,10))
@click.option("--interp-chunk-size", type=int, default=1000)
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--cache-size", type=int, default=50)
@click.option("--write-size", type=int, default=1)
@click.option("--verbose", "-v", is_flag=True)
@click.option("--dry-run", is_flag=True)
def main(config, config_prefix, forest_path, connectivity_path, connectivity_namespace, coords_path, 
         coords_namespace, synapses_namespace, distances_namespace, resolution, interp_chunk_size, io_size,
         chunk_size, value_chunk_size, cache_size, write_size, verbose, dry_run):

    utils.config_logging(verbose)
    logger = utils.get_script_logger(os.path.basename(__file__))
    
    comm = MPI.COMM_WORLD
    rank = comm.rank

    env = Env(comm=comm, config_file=config, config_prefix=config_prefix)
    configure_hoc_env(env)

    connection_config = env.connection_config
    extent      = {}
    soma_coords = {}

    if (not dry_run) and (rank==0):
        if not os.path.isfile(connectivity_path):
            input_file  = h5py.File(coords_path,'r')
            output_file = h5py.File(connectivity_path,'w')
            input_file.copy('/H5Types',output_file)
            input_file.close()
            output_file.close()
    comm.barrier()
    
    population_ranges = read_population_ranges(coords_path)[0]
    populations = sorted(list(population_ranges.keys()))

    soma_distances = {}
    for population in populations:
        if rank == 0:
            logger.info('Reading %s population coordinates...' % population)
        coords_iter = bcast_cell_attributes(coords_path, population, 0, namespace=coords_namespace)
        distances_iter = bcast_cell_attributes(coords_path, population, 0, namespace=distances_namespace)

        soma_coords[population] = { k: (v['U Coordinate'][0], v['V Coordinate'][0], v['L Coordinate'][0]) for (k,v) in coords_iter }

        distances = { k: (v['U Distance'][0], v['V Distance'][0]) for (k,v) in distances_iter }

        if len(distances) > 0:
            soma_distances[population] = distances
        
        gc.collect()

    destination_populations = sorted(read_population_names(forest_path))

    if len(soma_distances) == 0:
        (origin_ranges, ip_dist_u, ip_dist_v) = make_distance_interpolant(env, resolution=resolution, nsample=nsample)
        ip_dist = (origin_ranges, ip_dist_u, ip_dist_v)
        soma_distances = measure_distances(env, soma_coords, ip_dist, resolution=resolution)


    for destination_population in destination_populations:

        if rank == 0:
            logger.info('Generating connection probabilities for population %s...' % destination_population)

        connection_prob = ConnectionProb(destination_population, soma_coords, soma_distances, \
                                         env.connection_extents)

        synapse_seed = int(env.modelConfig['Random Seeds']['Synapse Projection Partitions'])
        
        connectivity_seed = int(env.modelConfig['Random Seeds']['Distance-Dependent Connectivity'])
        cluster_seed = int(env.modelConfig['Random Seeds']['Connectivity Clustering'])

        if rank == 0:
            logger.info('Generating connections for population %s...' % destination_population)

        populations_dict = env.modelConfig['Definitions']['Populations']
        generate_uv_distance_connections(comm, populations_dict,
                                         connection_config,
                                         connection_prob, forest_path,
                                         synapse_seed, connectivity_seed, cluster_seed,
                                         synapses_namespace, connectivity_namespace, connectivity_path,
                                         io_size, chunk_size, value_chunk_size, cache_size, write_size,
                                         dry_run=dry_run)
    MPI.Finalize()

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv)+1):])
