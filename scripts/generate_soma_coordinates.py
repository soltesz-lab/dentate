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
def main(config, types_path, output_path, output_namespace, populations, io_size, chunk_size, value_chunk_size):

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

    layer_vols = []
    for ((layer_name,max_extent),(_,min_extent)) in itertools.izip(max_extents.iteritems(),min_extents.iteritems()):
        if rank == 0:
            print 'creating volume for layer %s' % layer_name
        vol = make_volume(min_extent[2], max_extent[2])
        layer_vols.append((layer_name, vol))
        
    population_ranges = read_population_ranges(output_path, comm)[0]


    for population in populations:

        if rank == 0:
            print 'population: ',population

        (population_start, population_count) = population_ranges[population]

        coords_count = 0
        coords = []
        coords_dict = {}
        for (layer_name,layer_vol) in layer_vols:

            layer_count = env.geometry['Cell Layer Counts'][population][layer_name]

            if layer_count <= 0:
                continue
            
            tri = layer_vol.create_triangulation()
            alpha = alpha_shape([], 120., tri=tri)
    
            vert = alpha.points
            smp  = np.asarray(alpha.bounds, dtype=np.int64)

            N = layer_count*2 # total number of nodes
            node_count = 0
            itr = 10

            while node_count < layer_count:
                # create N quasi-uniformly distributed nodes
                nodes, smpid = menodes(N,vert,smp,itr=itr)
    
                # remove nodes outside of the domain
                in_nodes = nodes[contains(nodes,vert,smp)]
                              
                node_count = len(in_nodes)
                itr = int(itr / 2)

            sampled_idxs  = np.random.randint(0, node_count-1, size=int(layer_count))

            xyz_coords = (in_nodes[sampled_idxs]).reshape(-1,3)
            uvl_coords = layer_vol.inverse(xyz_coords)
            
            for i in xrange(0,layer_count):
                xyz_coords1 = layer_vol(uvl_coords[i,0],uvl_coords[i,1],uvl_coords[i,2]).ravel()
                coords.append((xyz_coords1[0],xyz_coords1[1],xyz_coords1[2],\
                               uvl_coords[i,0],uvl_coords[i,1],uvl_coords[i,2]))
                               
            coords_count += layer_count
        assert(coords_count == population_count)

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
