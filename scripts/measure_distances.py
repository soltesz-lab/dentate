
import os, sys, gc, logging
import click
import h5py
import numpy as np
import rbf
import rbf.basis
from rbf.interpolate import RBFInterpolant
from mpi4py import MPI
import dentate
import dentate.utils as utils
from dentate.env import Env
from dentate.geometry import measure_distances
from dentate.utils import viewitems
from neuroh5.io import append_cell_attributes, bcast_cell_attributes, read_population_names, read_population_ranges

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
    logger = utils.get_script_logger(__file__)
    
    comm = MPI.COMM_WORLD
    rank = comm.rank

    env = Env(comm=comm, config_file=config)
    output_path = coords_path

    soma_coords = {}

    if rank == 0:
        logger.info('Reading population coordinates...')
        
    for population in populations:
        coords = bcast_cell_attributes(coords_path, population, 0, \
                                       namespace=coords_namespace, comm=comm)

        soma_coords[population] = { k: (v['U Coordinate'][0], v['V Coordinate'][0], v['L Coordinate'][0]) for (k,v) in coords }
        del coords
        gc.collect()

    origin_ranges, soma_distances = measure_distances(env, soma_coords, resolution=resolution)
                                       
    for population in list(soma_distances.keys()):
            

        if rank == 0:
            logger.info('Writing distances for population %s...' % population)

        dist_dict = soma_distances[population]
        attr_dict = {}
        for k, v in viewitems(dist_dict):
            attr_dict[k] = { 'U Distance': np.asarray([v[0]],dtype=np.float32), \
                             'V Distance': np.asarray([v[1]],dtype=np.float32) }
        append_cell_attributes(output_path, population, attr_dict,
                               namespace='Arc Distances', comm=comm,
                               io_size=io_size, chunk_size=chunk_size,
                               value_chunk_size=value_chunk_size, cache_size=cache_size)
        if rank == 0:
            f = h5py.File(output_path, 'a')
            f['Populations'][population]['Arc Distances'].attrs['Reference U Min'] = origin_ranges[0][0]
            f['Populations'][population]['Arc Distances'].attrs['Reference U Max'] = origin_ranges[0][1]
            f['Populations'][population]['Arc Distances'].attrs['Reference V Min'] = origin_ranges[1][0]
            f['Populations'][population]['Arc Distances'].attrs['Reference V Max'] = origin_ranges[1][1]
            f.close()
        comm.Barrier()

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv)+1):])
