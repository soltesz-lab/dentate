##
## Generates distance-weighted random connectivity between the specified populations.
##

import sys, os, gc
from mpi4py import MPI
from neuroh5.io import read_population_ranges, read_population_names, bcast_cell_attributes, read_cell_attributes
import h5py
import numpy as np
import rbf
from rbf.interpolate import RBFInterpolant
import rbf.basis
import dentate
from dentate.connection_generator import ConnectionProb, generate_uv_distance_connections, get_volume_distances, get_soma_distances
from dentate.DG_volume import make_volume
from dentate.env import Env
import dentate.utils as utils
import click
import logging
logging.basicConfig()

script_name = 'generate_distance_connections.py'
logger = logging.getLogger(script_name)

@click.command()
@click.option("--config", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--forest-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--connectivity-path", required=True, type=click.Path())
@click.option("--connectivity-namespace", type=str, default='Connectivity')
@click.option("--coords-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--coords-namespace", type=str, default='Sorted Coordinates')
@click.option("--synapses-namespace", type=str, default='Synapse Attributes')
@click.option("--resample-volume", type=int, default=5)
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--cache-size", type=int, default=50)
@click.option("--write-size", type=int, default=1)
@click.option("--verbose", "-v", is_flag=True)
def main(config, forest_path, connectivity_path, connectivity_namespace, coords_path, coords_namespace,
         synapses_namespace, resample_volume, io_size, chunk_size, value_chunk_size, cache_size, write_size, verbose):

    if verbose:
        logger.setLevel(logging.INFO)
    
    comm = MPI.COMM_WORLD
    rank = comm.rank

    env = Env(comm=comm, configFile=config)

    extent      = {}
    soma_coords = {}

    if rank==0:
        if not os.path.isfile(connectivity_path):
            input_file  = h5py.File(coords_path,'r')
            output_file = h5py.File(connectivity_path,'w')
            input_file.copy('/H5Types',output_file)
            input_file.close()
            output_file.close()
    comm.barrier()

    if rank == 0:
        logger.info('Reading population coordinates...')
    
    population_ranges = read_population_ranges(coords_path)[0]
    for population in population_ranges.keys():
        coords = bcast_cell_attributes(coords_path, population, 0, \
                                       namespace=coords_namespace)

        soma_coords[population] = { k: (v['U Coordinate'][0], v['V Coordinate'][0], v['L Coordinate'][0]) for (k,v) in coords }
        del coords
        gc.collect()
        extent[population] = { 'width': env.modelConfig['Connection Generator']['Axon Width'][population],
                               'offset': env.modelConfig['Connection Generator']['Axon Offset'][population] }

    vol_dist = None
    if rank == 0:
        logger.info('Creating volume...')
        ip_volume = make_volume(-3.95, 3.2, ures=16, vres=15, lres=10)
        logger.info('Computing volume distances...')
        vol_dist = get_volume_distances(ip_volume, res=resample_volume, verbose=True)
        logger.info('Broadcasting volume distances...')
        
    vol_dist = comm.bcast(vol_dist, root=0)
    if rank == 0:
        logger.info('Computing volume distance interpolants...')

    step=1
    (dist_u, obs_dist_u, dist_v, obs_dist_v) = vol_dist
    ip_dist_u = RBFInterpolant(obs_dist_u,dist_u,order=1,basis=rbf.basis.phs3,extrapolate=True)
    del dist_u, obs_dist_u

    sample_inds = np.arange(0, obs_dist_v.shape[0]-1, step)
    ip_dist_v = RBFInterpolant(obs_dist_v,dist_v,order=1,basis=rbf.basis.phs3,extrapolate=True)
    del dist_v, obs_dist_v

    if rank == 0:
        logger.info('Computing soma distances...')
    soma_distances = get_soma_distances(comm, ip_dist_u, ip_dist_v, soma_coords)
    
    connectivity_synapse_types = env.modelConfig['Connection Generator']['Synapse Types']

    populations = read_population_names(forest_path)
    
    for destination_population in populations:

        if rank == 0:
            logger.info('Generating connection probabilities for population %s...' % destination_population)

        connection_prob = ConnectionProb(destination_population, soma_coords, soma_distances, extent)

        synapse_seed        = int(env.modelConfig['Random Seeds']['Synapse Projection Partitions'])
        
        connectivity_seed = int(env.modelConfig['Random Seeds']['Distance-Dependent Connectivity'])
        connectivity_namespace = 'Connections'

        if rank == 0:
            logger.info('Generating connections for population %s...' % destination_population)

        populations_dict = env.modelConfig['Definitions']['Populations']
        generate_uv_distance_connections(comm, populations_dict,
                                         env.connection_generator,
                                         connection_prob, forest_path,
                                         synapse_seed, synapses_namespace, 
                                         connectivity_seed, connectivity_namespace, connectivity_path,
                                         io_size, chunk_size, value_chunk_size, cache_size, write_size,
                                         verbose=verbose)


if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])

