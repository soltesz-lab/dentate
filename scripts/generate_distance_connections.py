##
## Generates distance-weighted random connectivity between the specified populations.
##

import sys, os, gc, click, logging
from mpi4py import MPI
from neuroh5.io import read_population_ranges, read_population_names, bcast_cell_attributes, read_cell_attributes
import h5py
import numpy as np
import rbf
from rbf.interpolate import RBFInterpolant
import rbf.basis
import dentate
from dentate.connection_generator import ConnectionProb, generate_uv_distance_connections
from dentate.geometry import measure_distances
from dentate.env import Env
import dentate.utils as utils

sys_excepthook = sys.excepthook
def mpi_excepthook(type, value, traceback):
    sys_excepthook(type, value, traceback)
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)
sys.excepthook = mpi_excepthook

script_name = 'generate_distance_connections.py'

@click.command()
@click.option("--config", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--forest-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--connectivity-path", required=True, type=click.Path())
@click.option("--connectivity-namespace", type=str, default='Connectivity')
@click.option("--coords-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--coords-namespace", type=str, default='Sorted Coordinates')
@click.option("--synapses-namespace", type=str, default='Synapse Attributes')
@click.option("--resolution", type=(int,int,int), default=(30,30,10))
@click.option("--interp-chunk-size", type=int, default=1000)
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--cache-size", type=int, default=50)
@click.option("--write-size", type=int, default=1)
@click.option("--verbose", "-v", is_flag=True)
@click.option("--dry-run", is_flag=True)
def main(config, forest_path, connectivity_path, connectivity_namespace, coords_path, coords_namespace,
         synapses_namespace, resolution, interp_chunk_size, io_size,
         chunk_size, value_chunk_size, cache_size, write_size, verbose, dry_run):

    utils.config_logging(verbose)
    logger = utils.get_script_logger(script_name)
    
    comm = MPI.COMM_WORLD
    rank = comm.rank

    env = Env(comm=comm, configFile=config)
    connectivity_synapse_types = env.modelConfig['Connection Generator']['Synapses']
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
    populations = population_ranges.keys()
    
    if rank == 0:
        logger.info('Reading population coordinates...')

    for population in populations:
        coords = bcast_cell_attributes(coords_path, population, 0, \
                                       namespace=coords_namespace)

        soma_coords[population] = { k: (v['U Coordinate'][0], v['V Coordinate'][0], v['L Coordinate'][0]) for (k,v) in coords }
        del coords
        gc.collect()
        extent[population] = { 'width': env.modelConfig['Connection Generator']['Axon Width'][population],
                               'offset': env.modelConfig['Connection Generator']['Axon Offset'][population] }

    soma_distances = measure_distances(env, comm, soma_coords)
    
    for destination_population in populations:

        if rank == 0:
            logger.info('Generating connection probabilities for population %s...' % destination_population)

        connection_prob = ConnectionProb(destination_population, soma_coords, soma_distances, extent)

        synapse_seed        = int(env.modelConfig['Random Seeds']['Synapse Projection Partitions'])
        
        connectivity_seed = int(env.modelConfig['Random Seeds']['Distance-Dependent Connectivity'])
        cluster_seed = int(env.modelConfig['Random Seeds']['Connectivity Clustering'])
        connectivity_namespace = 'Connections'

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
    main(args=sys.argv[(utils.list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])

