
import sys, os, gc, click, logging
from mpi4py import MPI
from neuroh5.io import read_population_ranges, read_population_names, bcast_cell_attributes
import h5py
import numpy as np
import dentate
from dentate.env import Env
import dentate.utils as utils

sys_excepthook = sys.excepthook
def mpi_excepthook(type, value, traceback):
    sys_excepthook(type, value, traceback)
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)
sys.excepthook = mpi_excepthook

@click.command()
@click.option("--config", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--coords-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--coords-namespace", type=str, default='Coordinates')
@click.option("--distances-namespace", type=str, default='Arc Distances')
@click.option("--populations", '-i', required=True, multiple=True, type=str)
@click.option("--verbose", "-v", is_flag=True)
def main(config, coords_path, coords_namespace, distances_namespace, populations, verbose):

    utils.config_logging(verbose)
    logger = utils.get_script_logger(__file__)
    
    comm = MPI.COMM_WORLD
    rank = comm.rank

    env = Env(comm=comm, config_file=config)
    output_path = coords_path

    soma_coords = {}
    soma_distances = {}

    if rank == 0:
        logger.info('Reading population coordinates and distances...')
        
    for population in populations:

        coords = bcast_cell_attributes(coords_path, population, 0, namespace=coords_namespace, comm=comm)
        soma_coords[population] = { k: (v['U Coordinate'][0], v['V Coordinate'][0], v['L Coordinate'][0]) for (k,v) in coords }
        del coords
        gc.collect()

        distances = bcast_cell_attributes(coords_path, population, 0, namespace=distances_namespace, comm=comm)
        soma_distances = { k: (v['U Distance'][0], v['V Distance'][0]) for (k,v) in distances }
        del distances
        gc.collect()
        

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv)+1):])
