##
## Generate soma coordinates within layer-specific volume.
##


import sys, itertools, os.path
from mpi4py import MPI
import h5py
import numpy as np
import math, random
from neuroh5.io import read_population_ranges, append_cell_attributes
import click
from dentate.utils import list_find
from dentate.env import Env
from dentate.geometry import make_volume, DG_volume, make_uvl_distance
import dlib
import rbf
from rbf.nodes import snap_to_boundary,disperse,menodes
from rbf.geometry import contains
from alphavol import alpha_shape
import logging

logging.basicConfig()

script_name = "generate_soma_coordinates.py"
logger = logging.getLogger(script_name)

def random_subset( iterator, K ):
    result = []
    N = 0

    for item in iterator:
        N += 1
        if len( result ) < K:
            result.append( item )
        else:
            s = int(random.random() * N)
            if s < K:
                result[ s ] = item

    return result



@click.command()
@click.option("--config", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--types-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--output-path", required=True, type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option("--output-namespace", type=str, default='Generated Coordinates')
@click.option("--populations", '-i', type=str, multiple=True)
@click.option("--alpha-radius", type=float, default=120.)
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--verbose", '-v', type=bool, default=False, is_flag=True)
def main(config, types_path, output_path, output_namespace, populations, alpha_radius, io_size, chunk_size, value_chunk_size, verbose):
    if verbose:
        logger.setLevel(logging.INFO)

    comm = MPI.COMM_WORLD
    rank = comm.rank

    if io_size == -1:
        io_size = comm.size
    if rank == 0:
        logger.info('%i ranks have been allocated' % comm.size)
    sys.stdout.flush()

    optiter = 200
    
    if rank==0:
        if not os.path.isfile(output_path):
            input_file  = h5py.File(types_path,'r')
            output_file = h5py.File(output_path,'w')
            input_file.copy('/H5Types',output_file)
            input_file.close()
            output_file.close()
    comm.barrier()

    env = Env(comm=comm, configFile=config)

    min_extents = env.geometry['Parametric Surface']['Minimum Extent']
    max_extents = env.geometry['Parametric Surface']['Maximum Extent']
    rotate = env.geometry['Rotation']

    population_ranges = read_population_ranges(output_path, comm)[0]

    for population in populations:

        if verbose and (rank == 0):
            logger.info( 'population: %s' % population )

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

        if verbose and (rank == 0):
            logger.info('min extent: %f %f %f' % (pop_min_extent[0],pop_min_extent[1],pop_min_extent[2]))
            logger.info('max extent: %f %f %f' % (pop_max_extent[0],pop_max_extent[1],pop_max_extent[2]))

        if verbose:
            logger.info("Constructing volume...")
        vol = make_volume(pop_min_extent[2], pop_max_extent[2], rotate=rotate)
        if verbose:
            logger.info("Volume constructed")
            
        if verbose:
            logger.info("Constructing alpha shape...")
        tri = vol.create_triangulation()
        alpha = alpha_shape([], alpha_radius, tri=tri)
        if verbose:
            logger.info("Alpha shape constructed")
    
        vert = alpha.points
        smp  = np.asarray(alpha.bounds, dtype=np.int64)

        N = population_count*2 # total number of nodes
        node_count = 0
        itr = 1

        if verbose:
            logger.info("Generating %i nodes..." % N)
        while node_count < population_count:
            # create N quasi-uniformly distributed nodes
            nodes, smpid = menodes(N,vert,smp,itr=itr)
    
            # remove nodes outside of the domain
            in_nodes = nodes[contains(nodes,vert,smp)]
                              
            node_count = len(in_nodes)
            itr = int(itr / 2)
        if verbose:
            logger.info("%i nodes generated" % node_count)


        xyz_coords = in_nodes.reshape(-1,3)
        uvl_coords_interp = vol.inverse(xyz_coords)
            
        xyz_error = np.asarray([0.0, 0.0, 0.0])
        for i in xrange(0,xyz_coords.shape[0]):
            xyz_coords_interp = vol(uvl_coords_interp[i,0],uvl_coords_interp[i,1],uvl_coords_interp[i,2]).ravel()
            xyz_error_interp  = np.abs(np.subtract(xyz_coords[i,:], xyz_coords_interp))

            f_uvl_distance = make_uvl_distance(xyz_coords[i,:],rotate=rotate)
            uvl_coords_opt,dist = dlib.find_min_global(f_uvl_distance, pop_min_extent.tolist(), pop_max_extent.tolist(), optiter)
            xyz_coords_opt = DG_volume(uvl_coords_opt[0], uvl_coords_opt[1], uvl_coords_opt[2], rotate=rotate)[0]
            xyz_error_opt  = np.abs(np.subtract(xyz_coords[i,:], xyz_coords_opt))

            if np.all (np.less (xyz_error_interp, xyz_error_opt)):
                uvl_coords  = uvl_coords_interp[i,:]
                xyz_coords1 = xyz_coords_interp
                xyz_error  = xyz_error_interp
            else:
                uvl_coords  = uvl_coords_opt
                xyz_coords1 = xyz_coords_opt
                xyz_error  = xyz_error_opt

            xyz_error   = np.add(xyz_error, np.abs(np.subtract(xyz_coords[i,:], xyz_coords1)))

            if verbose:
                logger.info('cell %i: %f %f %f' % (i, uvl_coords[0], uvl_coords[1], uvl_coords[2]))
            if ((uvl_coords[0] <= pop_max_extent[0]) and (uvl_coords[0] >= pop_min_extent[0]) and 
                (uvl_coords[1] <= pop_max_extent[1]) and (uvl_coords[1] >= pop_min_extent[1]) and
                (uvl_coords[2] <= pop_max_extent[2]) and (uvl_coords[2] >= pop_min_extent[2])):
                coords.append((xyz_coords1[0],xyz_coords1[1],xyz_coords1[2],\
                               uvl_coords[0],uvl_coords[1],uvl_coords[2]))
                           
        xyz_error = np.divide(xyz_error, np.asarray([population_count, population_count, population_count], dtype=np.float))
        if verbose:
            logger.info("mean XYZ error: %f %f %f " % (xyz_error[0], xyz_error[1], xyz_error[2]))
                               
        sampled_coords = random_subset(coords, int(population_count))

        coords.sort(key=lambda coord: coord[3]) ## sort on U coordinate
        coords_dict = { population_start+i :  { 'X Coordinate': np.asarray([x_coord],dtype=np.float32),
                                                'Y Coordinate': np.asarray([y_coord],dtype=np.float32),
                                                'Z Coordinate': np.asarray([z_coord],dtype=np.float32),
                                                'U Coordinate': np.asarray([u_coord],dtype=np.float32),
                                                'V Coordinate': np.asarray([v_coord],dtype=np.float32),
                                                'L Coordinate': np.asarray([l_coord],dtype=np.float32) }
                        for i,(x_coord,y_coord,z_coord,u_coord,v_coord,l_coord) in enumerate(sampled_coords) }

        append_cell_attributes(output_path, population, coords_dict,
                                namespace=output_namespace,
                                io_size=io_size, chunk_size=chunk_size,
                                value_chunk_size=value_chunk_size,comm=comm)

        

if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])
