
import sys, os, gc, click, logging
from mpi4py import MPI
from neuroh5.io import read_population_ranges, read_population_names, bcast_cell_attributes, append_cell_attributes
import h5py
import numpy as np
import dentate
from dentate.geometry import icp_transform
from dentate.env import Env
import dentate.utils as utils

script_name = 'project_somas.py'

sys_excepthook = sys.excepthook
def mpi_excepthook(type, value, traceback):
    sys_excepthook(type, value, traceback)
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)
sys.excepthook = mpi_excepthook

@click.command()
@click.option("--config", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--coords-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--coords-namespace", type=str, default='Sorted Coordinates')
@click.option("--resample", type=int, default=2)
@click.option("--resolution", type=int, default=16)
@click.option("--populations", '-i', required=True, multiple=True, type=str)
@click.option("--projection-depth", '-l', required=True, multiple=True, type=float)
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--cache-size", type=int, default=50)
@click.option("--verbose", "-v", is_flag=True)
def main(config, coords_path, coords_namespace, resample, resolution, populations, projection_depth, io_size, chunk_size, value_chunk_size, cache_size, verbose):

    utils.config_logging(verbose)
    logger = utils.get_script_logger(script_name)
    
    comm = MPI.COMM_WORLD
    rank = comm.rank

    env = Env(comm=comm, config_file=config)

    soma_coords = {}

    if rank == 0:
        logger.info('Reading population coordinates...')

    rotate = env.geometry['Parametric Surface']['Rotation']
    min_l = float('inf')
    max_l = 0.0
    population_ranges = read_population_ranges(coords_path)[0]
    population_extents = {}
    for population in list(population_ranges.keys()):
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
    
    output_path = coords_path
    soma_coords = icp_transform(comm, soma_coords, projection_depth, population_extents, \
                                populations=populations, rotate=rotate, verbose=verbose)
    
    for population in populations:

        if rank == 0:
            logger.info('Writing transformed coordinates for population %s...' % population)

        append_cell_attributes(output_path, population, soma_coords[population],
                               namespace='Soma Projections', comm=comm,
                               io_size=io_size, chunk_size=chunk_size,
                               value_chunk_size=value_chunk_size, cache_size=cache_size)


if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])

    
