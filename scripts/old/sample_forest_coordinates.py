##
## Import dendritic X Y Z coordinates from a forest of morphologies into 'Sampled Coordinates' namespace.
##


import sys
from mpi4py import MPI
import h5py
import numpy as np
from neuroh5.io import scatter_read_trees, append_cell_attributes
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
@click.option("--config", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--input-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--output-path", required=True, type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option("--populations", '-i', type=str, multiple=True)
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
def main(input_path, output_path, populations, io_size, chunk_size, value_chunk_size):

    comm = MPI.COMM_WORLD
    rank = comm.rank

    if io_size == -1:
        io_size = comm.size
    if rank == 0:
        print '%i ranks have been allocated' % comm.size
    sys.stdout.flush()

    env = Env(comm=comm, configFile=config)
    
    swc_type_apical = env.SWC_Types['apical']
    
    if rank==0:
        input_file  = h5py.File(input_path,'r')
        output_file = h5py.File(output_path,'w')
        input_file.copy('/H5Types',output_file)
        input_file.close()
        output_file.close()
    comm.barrier()

    for population in populations:

        if rank == 0:
            print 'population: ',population

        (trees, forestSize) = scatter_read_trees(input_path, population, io_size=io_size)

        coords_dict = {}
        for (gid, tree) in trees:

            vx       = tree['x']
            vy       = tree['y']
            vz       = tree['z']
            swc_type = tree['swc_type']

            dend_idxs = np.where(swc_type == swc_type_apical)[0]

            x_coord = vx[dend_idxs[0]]
            y_coord = vy[dend_idxs[0]]
            z_coord = vz[dend_idxs[0]]
            coords_dict[gid] = {'X Coordinate': np.asarray([x_coord],dtype=np.float32),
                                'Y Coordinate': np.asarray([y_coord],dtype=np.float32),
                                'Z Coordinate': np.asarray([z_coord],dtype=np.float32) }

        append_cell_attributes(output_path, population, coords_dict,
                                namespace='Sampled Coordinates',
                                io_size=io_size, chunk_size=chunk_size,
                                value_chunk_size=value_chunk_size)

        

if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find("sample_forest_coordinates.py") != -1,sys.argv)+1):])
