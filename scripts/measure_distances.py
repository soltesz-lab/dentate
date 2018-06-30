
import sys, os, gc, click, logging
from mpi4py import MPI
from neuroh5.io import read_population_ranges, read_population_names, bcast_cell_attributes, append_cell_attributes
import h5py
import numpy as np
import rbf
from rbf.interpolate import RBFInterpolant
import rbf.basis
import dentate
from dentate.geometry import make_volume, get_volume_distances, get_soma_distances, interp_soma_distances
from dentate.env import Env
import dentate.utils as utils

sys_excepthook = sys.excepthook
def mpi_excepthook(type, value, traceback):
    sys_excepthook(type, value, traceback)
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)
sys.excepthook = mpi_excepthook

script_name = 'measure_distances.py'

@click.command()
@click.option("--config", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--coords-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--coords-namespace", type=str, default='Sorted Coordinates')
@click.option("--populations", '-i', required=True, multiple=True, type=str)
@click.option("--interpolate", is_flag=True)
@click.option("--interp-chunk-size", type=int, default=1000)
@click.option("--alpha-radius", type=float, default=120.)
@click.option("--resample", type=int, default=7)
@click.option("--resolution", type=(int,int,int), default=(30,30,10))
@click.option("--ndist", type=int, default=1)
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--cache-size", type=int, default=50)
@click.option("--verbose", "-v", is_flag=True)
def main(config, coords_path, coords_namespace, populations, interpolate, interp_chunk_size, resample, resolution, alpha_radius, ndist, io_size, chunk_size, value_chunk_size, cache_size, verbose):

    utils.config_logging(verbose)
    logger = utils.get_script_logger(script_name)
    
    comm = MPI.COMM_WORLD
    rank = comm.rank

    env = Env(comm=comm, configFile=config)
    output_path = coords_path

    soma_coords = {}

    if rank == 0:
        logger.info('Reading population coordinates...')

    rotate = env.geometry['Parametric Surface']['Rotation']
    min_l = float('inf')
    max_l = 0.0
    population_ranges = read_population_ranges(coords_path)[0]
    population_extents = {}
    for population in population_ranges.keys():
        min_extent = env.geometry['Cell Layers']['Minimum Extent'][population]
        max_extent = env.geometry['Cell Layers']['Maximum Extent'][population]
        min_l = min(min_extent[2], min_l)
        max_l = max(max_extent[2], max_l)
        population_extents[population] = (min_extent, max_extent)
        
    for population in populations:
        coords = bcast_cell_attributes(coords_path, population, 0, \
                                       namespace=coords_namespace)

        soma_coords[population] = { k: (v['U Coordinate'][0], v['V Coordinate'][0], v['L Coordinate'][0]) for (k,v) in coords }
        del coords
        gc.collect()

    obs_uv = None
    coeff_dist_u = None
    coeff_dist_v = None

    interp_penalty = 0.001
    interp_basis = 'ga'
    interp_order = 1

    ## This parameter is used to expand the range of L and avoid
    ## situations where the endpoints of L end up outside of the range
    ## of the distance interpolant
    safety = 0.01
    
    if rank == 0:
        logger.info('Creating volume: min_l = %f max_l = %f...' % (min_l, max_l))
    if interpolate:
        obs_uv = None
        coeff_dist_u = None
        coeff_dist_v = None
        origin_uvl = None
        if rank == 0:
            ip_volume = make_volume(min_l-safety, max_l+safety, \
                                    resolution=resolution, \
                                    rotate=rotate)

            logger.info('Computing volume distances...')
            vol_dist = get_volume_distances (ip_volume, origin_coords=origin_uvl, rotate=rotate, res=resample, alpha_radius=alpha_radius)
            (obs_uv, dist_u, dist_v) = vol_dist

            
                
            logger.info('Computing U volume distance interpolants...')
            ip_dist_u = RBFInterpolant(obs_uv,dist_u,order=interp_order,basis=interp_basis,\
                                       penalty=interp_penalty, extrapolate=False)
            coeff_dist_u = ip_dist_u._coeff
            del dist_u
            gc.collect()
            logger.info('Computing V volume distance interpolants...')
            ip_dist_v = RBFInterpolant(obs_uv,dist_v,order=interp_order,basis=interp_basis,\
                                       penalty=interp_penalty, extrapolate=False)
            coeff_dist_v = ip_dist_v._coeff
            del dist_v
            gc.collect()
            logger.info('Broadcasting volume distance interpolants...')
        
        obs_uv = comm.bcast(obs_uv, root=0)
        coeff_dist_u = comm.bcast(coeff_dist_u, root=0)
        coeff_dist_v = comm.bcast(coeff_dist_v, root=0)
        origin_uvl = comm.bcast(origin_uvl, root=0)

        ip_dist_u = RBFInterpolant(obs_uv,coeff=coeff_dist_u,order=interp_order,basis=interp_basis,\
                                   penalty=interp_penalty, extrapolate=False)
        ip_dist_v = RBFInterpolant(obs_uv,coeff=coeff_dist_v,order=interp_order,basis=interp_basis,\
                                   penalty=interp_penalty, extrapolate=False)
    else:
        ip_volume = make_volume(min_l-safety, max_l+safety, \
                                resolution=resolution, \
                                rotate=rotate)

        span_U, span_V, span_L  = ip_volume._resample_uvl(resample, resample, resample)

        origin_u = np.median(span_U)
        origin_v = np.median(span_V)
        origin_l = np.max(span_L)-safety
        origin_uvl = np.asarray([origin_u, origin_v, origin_l])
                                       
    for population in populations:

        if interpolate:
            soma_distances = interp_soma_distances(comm, ip_dist_u, ip_dist_v, soma_coords, population_extents, populations=[population], \
                                                   ndist=ndist, interp_chunk_size=interp_chunk_size, allgather=False)
        else:
            soma_distances = get_soma_distances(comm, ip_volume, origin_uvl, soma_coords, population_extents, populations=[population], \
                                                ndist=ndist, allgather=False)
            

        if rank == 0:
            logger.info('Writing distances for population %s...' % population)

        dist_dict = soma_distances[population]
        attr_dict = {}
        for k, v in dist_dict.iteritems():
            attr_dict[k] = { 'U Distance': np.asarray([v[0]],dtype=np.float32), \
                             'V Distance': np.asarray([v[1]],dtype=np.float32) }
        append_cell_attributes(output_path, population, attr_dict,
                               namespace='Arc Distances', comm=comm,
                               io_size=io_size, chunk_size=chunk_size,
                               value_chunk_size=value_chunk_size, cache_size=cache_size)


if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])

    
