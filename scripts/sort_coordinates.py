
import sys, os, gc, click, logging
from mpi4py import MPI
from neuroh5.io import read_population_ranges, read_population_names, bcast_cell_attributes, append_cell_attributes
import h5py
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
@click.option("--coords-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
def main(coords_path, io_size, chunk_size, value_chunk_size):

    utils.config_logging(verbose)
    logger = utils.get_script_logger(__file__)
    
    comm = MPI.COMM_WORLD
    rank = comm.rank

    env = Env(comm=comm, config_file=config)
    output_path = coords_path

    if io_size == -1:
        io_size = comm.size
    if rank == 0:
        logger.info('%i ranks have been allocated' % comm.size)

    source_population_ranges = read_population_ranges(coords_path)
    source_populations = list(source_population_ranges.keys())

    for population in source_populations:
        if rank == 0:
            logger.info('population: ',population)
        soma_coords = bcast_cell_attributes(0, coords_path, population,
                                            namespace='Interpolated Coordinates', comm=comm)
        #print soma_coords.keys()
        u_coords = []
        gids = []
        for gid, attrs in soma_coords.items():
            u_coords.append(attrs['U Coordinate'])
            gids.append(gid)
        u_coordv = np.asarray(u_coords, dtype=np.float32)
        gidv     = np.asarray(gids, dtype=np.uint32)
        sort_idx = np.argsort(u_coordv, axis=0)
        offset   = source_population_ranges[population][0]
        sorted_coords_dict = {}
        for i in range(0,sort_idx.size):
            sorted_coords_dict[offset+i] = soma_coords[gidv[sort_idx[i][0]]]
        
        append_cell_attributes(coords_path, population, sorted_coords_dict,
                                namespace='Sorted Coordinates', io_size=io_size, chunk_size=chunk_size,
                                value_chunk_size=value_chunk_size, comm=comm)

        


if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv)+1):])
