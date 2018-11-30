
import sys, os, gc, click, logging
from mpi4py import MPI
from neuroh5.io import read_population_ranges, read_population_names, bcast_cell_attributes, append_cell_attributes
import h5py
import numpy as np
import rbf
from rbf.interpolate import RBFInterpolant
import rbf.basis
import dentate
from dentate.geometry import measure_distances
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
@click.option("--interp-chunk-size", type=int, default=1000)
@click.option("--alpha-radius", type=float, default=120.)
@click.option("--resolution", type=(int,int,int), default=(30,30,10))
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--cache-size", type=int, default=50)
@click.option("--verbose", "-v", is_flag=True)
def main(config, coords_path, coords_namespace, populations, interp_chunk_size, resolution, alpha_radius, io_size, chunk_size, value_chunk_size, cache_size, verbose):

    utils.config_logging(verbose)
    logger = utils.get_script_logger(script_name)
    
    comm = MPI.COMM_WORLD
    rank = comm.rank

    env = Env(comm=comm, config_file=config)
    output_path = coords_path

    soma_coords = {}

    if rank == 0:
        logger.info('Reading population coordinates...')
        
    for population in populations:
        coords = bcast_cell_attributes(coords_path, population, 0, \
                                       namespace=coords_namespace)

        soma_coords[population] = { k: (v['U Coordinate'][0], v['V Coordinate'][0], v['L Coordinate'][0]) for (k,v) in coords }
        del coords
        gc.collect()

    soma_distances = measure_distances(env, comm, soma_coords, resolution=resolution)
                                       
    for population in list(soma_distances.keys()):
            

        if rank == 0:
            logger.info('Writing distances for population %s...' % population)

        dist_dict = soma_distances[population]
        attr_dict = {}
        for k, v in dist_dict.items():
            attr_dict[k] = { 'U Distance': np.asarray([v[0]],dtype=np.float32), \
                             'V Distance': np.asarray([v[1]],dtype=np.float32) }
        append_cell_attributes(output_path, population, attr_dict,
                               namespace='Arc Distances', comm=comm,
                               io_size=io_size, chunk_size=chunk_size,
                               value_chunk_size=value_chunk_size, cache_size=cache_size)


if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])

    
