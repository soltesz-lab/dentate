## Import X Y Z coordinates sampled by DG_sample_pos script
##
##


import sys
from mpi4py import MPI
import h5py
import numpy as np
from neuroh5.io import append_cell_attributes
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
@click.option("--output-path", required=True, type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option("--input-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
def main(input_path, output_path, io_size, chunk_size, value_chunk_size):

    comm = MPI.COMM_WORLD
    rank = comm.rank

    if io_size == -1:
        io_size = comm.size
    if rank == 0:
        print '%i ranks have been allocated' % comm.size
    sys.stdout.flush()

    input_file = h5py.File(input_path)
    source_populations = input_file.keys()
    
    for population in source_populations:
        if rank == 0:
            print 'population: ',population
        population_grp = input_file[population]
        
        X_coords = population_grp['X Coordinate']
        Y_coords = population_grp['Y Coordinate']
        Z_coords = population_grp['Z Coordinate']
        
        coords_dict = {}
        for i in xrange(rank,X_coords.size,comm.size):
            coords_dict[i] = {'X Coordinate': np.array([X_coords[0,i]],dtype=np.float32),
                              'Y Coordinate': np.array([Y_coords[0,i]],dtype=np.float32),
                              'Z Coordinate': np.array([Z_coords[0,i]],dtype=np.float32) }

        append_cell_attributes(MPI._addressof(comm), output_path, population, coords_dict,
                                namespace='Sampled Coordinates', io_size=io_size, chunk_size=chunk_size,
                                value_chunk_size=value_chunk_size)

    input_file.close()
        

if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find("import_sampled_coordinates.py") != -1,sys.argv)+1):])
