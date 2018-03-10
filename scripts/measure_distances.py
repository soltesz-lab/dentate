
import sys, os, gc
from mpi4py import MPI
from neuroh5.io import read_population_ranges, read_population_names, bcast_cell_attributes, append_cell_attributes
import h5py
import numpy as np
import rbf
from rbf.interpolate import RBFInterpolant
import rbf.basis
import dentate
from dentate.connection_generator import get_volume_distances, get_soma_distances
from dentate.DG_volume import make_volume
from dentate.env import Env
import dentate.utils as utils
import click
import logging
logging.basicConfig()

script_name = 'measure_distances.py'
logger = logging.getLogger(script_name)

@click.command()
@click.option("--config", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--coords-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--coords-namespace", type=str, default='Sorted Coordinates')
@click.option("--resample-volume", type=int, default=2)
@click.option("--populations", '-i', required=True, multiple=True, type=str)
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--cache-size", type=int, default=50)
@click.option("--verbose", "-v", is_flag=True)
def main(config, coords_path, coords_namespace, resample_volume, populations, io_size, chunk_size, value_chunk_size, cache_size, verbose):

    if verbose:
        logger.setLevel(logging.INFO)
    
    comm = MPI.COMM_WORLD
    rank = comm.rank

    env = Env(comm=comm, configFile=config)

    soma_coords = {}

    if rank == 0:
        logger.info('Reading population coordinates...')
    
    population_ranges = read_population_ranges(coords_path)[0]
    for population in populations:
        coords = bcast_cell_attributes(coords_path, population, 0, \
                                       namespace=coords_namespace)

        soma_coords[population] = { k: (v['U Coordinate'][0], v['V Coordinate'][0], v['L Coordinate'][0]) for (k,v) in coords }
        del coords
        gc.collect()

    vol_dist = None
    if rank == 0:
        logger.info('Creating volume...')
        ip_volume = make_volume(-3.95, 3.2, ures=16, vres=15, lres=10)
        logger.info('Computing volume distances...')
        vol_dist = get_volume_distances(ip_volume, res=resample_volume, verbose=True)
        del ip_volume
        logger.info('Broadcasting volume distances...')
        
    vol_dist = comm.bcast(vol_dist, root=0)
    if rank == 0:
        logger.info('Computing volume distance interpolants...')

    (dist_u, obs_dist_u, dist_v, obs_dist_v) = vol_dist
    del vol_dist
        
    ip_dist_u = RBFInterpolant(obs_dist_u,dist_u,order=1,basis=rbf.basis.phs3,extrapolate=True)
    del dist_u, obs_dist_u

    ip_dist_v = RBFInterpolant(obs_dist_v,dist_v,order=1,basis=rbf.basis.phs3,extrapolate=True)
    del dist_v, obs_dist_v

    if rank == 0:
        logger.info('Computing soma distances...')
    soma_distances = get_soma_distances(comm, ip_dist_u, ip_dist_v, soma_coords)
    
    output_path = coords_path
    for population in soma_distances.keys():

        dist_dict = soma_distances[population]
        attr_dict = {}
        for k, v in dist_dict.iteritems():
            attr_dict[k] = { 'U Distance': v[0],  'V Distance': v[1] }
        append_cell_attributes(output_path, population, attr_dict,
                               namespace='Arc Distances', comm=comm,
                               io_size=io_size, chunk_size=chunk_size,
                               value_chunk_size=value_chunk_size, cache_size=cache_size)


if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])

