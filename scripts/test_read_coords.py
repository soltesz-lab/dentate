import sys
from mpi4py import MPI
from neuroh5.io import read_population_ranges, read_population_names, bcast_cell_attributes, append_cell_attributes
import numpy as np
import click

def list_find (f, lst):
    i=0
    for x in lst:
        if f(x):
            return i
        else:
            i=i+1
    return None

script_name = 'test_read_coords.py'

@click.command()
@click.option("--coords-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--coords-namespace", type=str, default='Sorted Coordinates')
@click.option("--io-size", type=int, default=-1)
def main(coords_path, coords_namespace, io_size):

    comm = MPI.COMM_WORLD
    rank = comm.rank


    population_ranges = read_population_ranges(comm, coords_path)[0]
    
    soma_coords = {}
    for population in population_ranges.keys():
        soma_coords[population] = bcast_cell_attributes(comm, 0, coords_path, population,
                                                        namespace=coords_namespace)
        
    for population in population_ranges:
        (population_start, _) = population_ranges[population]
        coords = soma_coords[population]

        for (cell_gid, cell_coords_dict) in coords.iteritems():

            cell_u = cell_coords_dict['U Coordinate']
            cell_v = cell_coords_dict['V Coordinate']

            print 'Rank %i: gid = %i u = %f v = %f' % (rank, cell_gid, cell_u, cell_v)


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])

