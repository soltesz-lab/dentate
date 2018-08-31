import sys
from mpi4py import MPI
import numpy as np
from neurotrees.io import append_cell_attributes, bcast_cell_attributes, population_ranges
import click



def list_find (f, lst):
    i=0
    for x in lst:
        if f(x):
            return i
        else:
            i=i+1
    return None

@click.command()
@click.option("--coords-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
def main(coords_path, io_size, chunk_size, value_chunk_size):

    comm = MPI.COMM_WORLD
    rank = comm.rank

    if io_size == -1:
        io_size = comm.size
    if rank == 0:
        print('%i ranks have been allocated' % comm.size)
    sys.stdout.flush()

    source_population_ranges = population_ranges(MPI._addressof(comm), coords_path)
    source_populations = list(source_population_ranges.keys())
    for population in source_populations:
        if rank == 0:
            print('population: ',population)
        soma_coords = bcast_cell_attributes(MPI._addressof(comm), 0, coords_path, population,
                                            namespace='Interpolated Coordinates')
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
        
        append_cell_attributes(MPI._addressof(comm), coords_path, population, sorted_coords_dict,
                                namespace='Sorted Coordinates', io_size=io_size, chunk_size=chunk_size,
                                value_chunk_size=value_chunk_size)

        

if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find("sort_coordinates.py") != -1,sys.argv)+1):])
