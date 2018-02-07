##
## Generate soma X Y Z coordinates within layer-specific volume.
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
@click.option("--populations", '-i', type=str, multiple=True)
@click.option("--spatial-resolution", type=float, default=1.0)
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
def main(config, types_path, output_path, populations, spatial_resolution, io_size, chunk_size, value_chunk_size):

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

        count = 0
        coords = []
        coords_dict = {}
        for (layer_name,layer_vol) in layers_vols:

            layer_count = env.geometry['Cell Layer Counts'][population][layer_name]

            layer_count = env.geometry['Cell Layer Counts'][population][layer_name]
            
            tri = layer_vol.create_triangulation()
            alpha = alpha_shape([], 120., tri=tri)
    
            vert = alpha.points
            smp  = np.asarray(alpha.bounds, dtype=np.int64)

            N = layer_count*2 # total number of nodes

            node_count = 0
            while node_count < layer_count:
                # create N quasi-uniformly distributed nodes
                nodes, smpid = menodes(N,vert,smp,itr=20)
    
                # remove nodes outside of the domain
                in_nodes = nodes[contains(nodes,vert,smp)]

                node_count = len(in_nodes)
                itr = int(itr / 2)

            sampled_idxs  = np.random.randint(0, node_count-1, size=int(layer_count))

            for i in sampled_idxs:

                xyz_coords = in_nodes[sampled_idxs[i]]
                uvl_coords = layer_vol.inverse(xyz_coords)

                coords.append((xyz_coords[0],xyz_coords[1],xyz_coords[2],\
                               uvl_coords[0],uvl_coords[1],uvl_coords[2]))
                count += layer_count
        assert(count == population_count)

        coords.sort(key=lambda coord: coord[3]) ## sort on U coordinate
        coords_dict = { population_start+i :  { 'X Coordinate': np.asarray([x_coord],dtype=np.float32),
                                    'Y Coordinate': np.asarray([y_coord],dtype=np.float32),
                                    'Z Coordinate': np.asarray([z_coord],dtype=np.float32),
                                    'U Coordinate': np.asarray([u_coord],dtype=np.float32),
                                    'V Coordinate': np.asarray([v_coord],dtype=np.float32),
                                    'L Coordinate': np.asarray([l_coord],dtype=np.float32) }
                        for i,(x_coord,y_coord,z_coord,u_coord,v_coord,l_coord) in enumerate(coords) }

        append_cell_attributes(output_path, population, coords_dict,
                                namespace='Generated Coordinates',
                                io_size=io_size, chunk_size=chunk_size,
                                value_chunk_size=value_chunk_size,comm=comm)

        

if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find("generate_soma_coordinates.py") != -1,sys.argv)+1):])
