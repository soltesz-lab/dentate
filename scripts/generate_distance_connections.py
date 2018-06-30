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
from dentate.connection_generator import ConnectionProb, generate_uv_distance_connections, get_volume_distances, get_soma_distances, interp_soma_distances
from dentate.geometry import make_volume
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
@click.option("--resample", type=int, default=2)
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
         synapses_namespace, resample, resolution, interp_chunk_size, io_size,
         chunk_size, value_chunk_size, cache_size, write_size, verbose, dry_run):

    utils.config_logging(verbose)
    logger = utils.get_script_logger(script_name)
    
    comm = MPI.COMM_WORLD
    rank = comm.rank

    env = Env(comm=comm, configFile=config)

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

    if rank == 0:
        logger.info('Reading population coordinates...')
    
    min_l = float('inf')
    max_l = 0.0
    population_ranges = read_population_ranges(coords_path)[0]
    for population in population_ranges.keys():
        min_extent = env.geometry['Cell Layers']['Minimum Extent'][population]
        max_extent = env.geometry['Cell Layers']['Maximum Extent'][population]
        min_l = min(min_extent[2], min_l)
        max_l = max(max_extent[2], max_l)
        coords = bcast_cell_attributes(coords_path, population, 0, \
                                       namespace=coords_namespace)

        soma_coords[population] = { k: (v['U Coordinate'][0], v['V Coordinate'][0], v['L Coordinate'][0]) for (k,v) in coords }
        del coords
        gc.collect()
        extent[population] = { 'width': env.modelConfig['Connection Generator']['Axon Width'][population],
                               'offset': env.modelConfig['Connection Generator']['Axon Offset'][population] }

    rotate = env.geometry['Parametric Surface']['Rotation']
        
    obs_uv = None
    coeff_dist_u = None
    coeff_dist_v = None
    origin_uvl = None
    
    vol_res = volume_resolution

    interp_penalty = 0.001
    interp_basis = 'ga'
    interp_order = 1

    ## This parameter is used to expand the range of L and avoid
    ## situations where the endpoints of L end up outside of the range
    ## of the distance interpolant
    safety = 0.01

    if rank == 0:
        logger.info('Creating volume: min_l = %f max_l = %f...' % (min_l, max_l))
    
    if rank == 0:
        logger.info('Creating volume...')
        ip_volume = make_volume(min_l-safety, max_l+safety, resolution=resolution, rotate=rotate)
        span_U, span_V, span_L  = ip_volume._resample_uvl(resample, resample, resample)

        origin_u = np.median(span_U)
        origin_v = np.median(span_V)
        origin_l = np.max(span_L)-safety

        origin_uvl = np.asarray([origin_u, origin_v, origin_l])
        
        logger.info('Computing volume distances...')
        vol_dist = get_volume_distances(ip_volume, res=resample, verbose=verbose)
        (obs_uv, dist_u, dist_v) = vol_dist
        logger.info('Computing U volume distance interpolants...')
        ip_dist_u = RBFInterpolant(obs_uv,dist_u,order=interp_order,basis=interp_basis,\
                                   penalty=interp_penalty)
        coeff_dist_u = ip_dist_u._coeff
        logger.info('Computing V volume distance interpolants...')
        ip_dist_v = RBFInterpolant(obs_uv,dist_v,order=interp_order,basis=interp_basis,\
                                   penalty=interp_penalty)
        coeff_dist_v = ip_dist_v._coeff
        logger.info('Broadcasting volume distance interpolants...')
        
    obs_uv = comm.bcast(obs_uv, root=0)
    coeff_dist_u = comm.bcast(coeff_dist_u, root=0)
    coeff_dist_v = comm.bcast(coeff_dist_v, root=0)
    origin_uvl = comm.bcast(origin_uvl, root=0)
    
    ip_dist_u = RBFInterpolant(obs_uv,coeff=coeff_dist_u,order=interp_order,basis=interp_basis,\
                               penalty=interp_penalty)
    ip_dist_v = RBFInterpolant(obs_uv,coeff=coeff_dist_v,order=interp_order,basis=interp_basis,\
                               penalty=interp_penalty)

    if rank == 0:
        logger.info('Computing soma distances...')

    
    populations = read_population_names(forest_path)

    soma_distances = interp_soma_distances(comm, ip_dist_u, ip_dist_v, origin_uvl, soma_coords, population_extents, \
                                           interp_chunk_size=interp_chunk_size, allgather=True)
    
    connectivity_synapse_types = env.modelConfig['Connection Generator']['Synapse Types']

    
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
                                         env.connection_config,
                                         connection_prob, forest_path,
                                         synapse_seed, synapses_namespace, 
                                         connectivity_seed, cluster_seed, connectivity_namespace, connectivity_path,
                                         io_size, chunk_size, value_chunk_size, cache_size, write_size,
                                         verbose=verbose, dry_run=dry_run)
    MPI.Finalize()

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])

