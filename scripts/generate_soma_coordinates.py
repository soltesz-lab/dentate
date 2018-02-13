##
## Generate soma coordinates within layer-specific volume.
##


import sys, itertools
from mpi4py import MPI
import h5py
import numpy as np
import math
from neuroh5.io import read_population_ranges, append_cell_attributes
import click
from env import Env
from DG_volume import make_volume
import rbf
from rbf.nodes import snap_to_boundary,disperse,menodes
from rbf.geometry import contains
from alphavol import alpha_shape

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
@click.option("--types-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--output-path", required=True, type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option("--output-namespace", type=str, default='Generated Coordinates')
@click.option("--populations", '-i', type=str, multiple=True)
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--verbose", '-v', type=bool, default=False, is_flag=True)
def main(config, types_path, output_path, output_namespace, populations, io_size, chunk_size, value_chunk_size, verbose):

    comm = MPI.COMM_WORLD
    rank = comm.rank

    if io_size == -1:
        io_size = comm.size
    if rank == 0:
        print '%i ranks have been allocated' % comm.size
    sys.stdout.flush()

    if rank==0:
        input_file  = h5py.File(types_path,'r')
        output_file = h5py.File(output_path,'w')
        input_file.copy('/H5Types',output_file)
        input_file.close()
        output_file.close()
    comm.barrier()

    env = Env(comm=comm, configFile=config)

    max_extents = env.geometry['Parametric Surface']['Minimum Extent']
    min_extents = env.geometry['Parametric Surface']['Maximum Extent']

    population_ranges = read_population_ranges(output_path, comm)[0]

    for population in populations:

        if verbose and (rank == 0):
            print 'population: ',population

        (population_start, population_count) = population_ranges[population]

        coords = []
        coords_dict = {}

        pop_max_extent = None
        pop_min_extent = None
        for ((layer_name,max_extent),(_,min_extent)) in itertools.izip(max_extents.iteritems(),min_extents.iteritems()):

            layer_count = env.geometry['Cell Layer Counts'][population][layer_name]

            if layer_count > 0:
                if pop_max_extent is None:
                    pop_max_extent = np.asarray(max_extent)
                else:
                    pop_max_extent = np.maximum(pop_max_extent, np.asarray(max_extent))
                if pop_min_extent is None:
                    pop_min_extent = np.asarray(min_extent)
                else:
                    pop_min_extent = np.minimum(pop_min_extent, np.asarray(min_extent))

        vol = make_volume(pop_min_extent[2], pop_max_extent[2])
            
        tri = vol.create_triangulation()
        alpha = alpha_shape([], 120., tri=tri)
    
        vert = alpha.points
        smp  = np.asarray(alpha.bounds, dtype=np.int64)

        N = population_count*2 # total number of nodes
        node_count = 0
        itr = 10

        while node_count < population_count:
            # create N quasi-uniformly distributed nodes
            nodes, smpid = menodes(N,vert,smp,itr=itr)
    
            # remove nodes outside of the domain
            in_nodes = nodes[contains(nodes,vert,smp)]
                              
            node_count = len(in_nodes)
            itr = int(itr / 2)

        sampled_idxs  = np.random.randint(0, node_count-1, size=int(population_count))

        xyz_coords = (in_nodes[sampled_idxs]).reshape(-1,3)
        uvl_coords = vol.inverse(xyz_coords)
            
        xyz_error = np.asarray([0.0, 0.0, 0.0])
        for i in xrange(0,population_count):
            xyz_coords1 = vol(uvl_coords[i,0],uvl_coords[i,1],uvl_coords[i,2]).ravel()
            xyz_error   = np.add(xyz_error, np.abs(np.subtract(xyz_coords[i,:], xyz_coords1)))
            coords.append((xyz_coords1[0],xyz_coords1[1],xyz_coords1[2],\
                           uvl_coords[i,0],uvl_coords[i,1],uvl_coords[i,2]))
                           
        xyz_error = np.divide(xyz_error, np.asarray([population_count, population_count, population_count], dtype=np.float))
        if verbose:
            print "mean XYZ error: ", xyz_error
                               

        coords.sort(key=lambda coord: coord[3]) ## sort on U coordinate
        coords_dict = { population_start+i :  { 'X Coordinate': np.asarray([x_coord],dtype=np.float32),
                                    'Y Coordinate': np.asarray([y_coord],dtype=np.float32),
                                    'Z Coordinate': np.asarray([z_coord],dtype=np.float32),
                                    'U Coordinate': np.asarray([u_coord],dtype=np.float32),
                                    'V Coordinate': np.asarray([v_coord],dtype=np.float32),
                                    'L Coordinate': np.asarray([l_coord],dtype=np.float32) }
                        for i,(x_coord,y_coord,z_coord,u_coord,v_coord,l_coord) in enumerate(coords) }

        append_cell_attributes(output_path, population, coords_dict,
                                namespace=output_namespace,
                                io_size=io_size, chunk_size=chunk_size,
                                value_chunk_size=value_chunk_size,comm=comm)

        

if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find("generate_soma_coordinates.py") != -1,sys.argv)+1):])
