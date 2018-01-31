##
## Import dendritic X Y Z coordinates from a forest of morphologies into 'Sampled Coordinates' namespace.
##


import sys, itertools
from mpi4py import MPI
import h5py
import numpy as np
import math
from neuroh5.io import read_population_ranges, append_cell_attributes
import click
from env import Env
from DG_surface import make_surface

def list_find (f, lst):
    i=0
    for x in lst:
        if f(x):
            return i
        else:
            i=i+1
    return None

##
## Create hexagonal packing of cell soma positions
##
def cellgrid(env, population):

    xmin    =   -4000.
    xmax    =   4000.
    ymin    =   -500.
    ymax    =   5000.
    zmin    =   -900.
    zmax    =   3000.

    DistH = env.geometry['Cell Grid Spacing'][population]['H']
    DistV = env.geometry['Cell Grid Spacing'][population]['V']

    
    Kepler      =   math.pi / (3. * math.sqrt(2.))
    
    ## Round limits to nearest multiple of distances
    xmax2    = ((xmax-xmin)/DistH)*DistH + xmin
    ymax2    = ((ymax-ymin)/DistH)*DistH + ymin
    zmax2    = ((zmax-zmin)/(Kepler*DistV))*(Kepler*DistV) + zmin
    zlayers  = ((zmax2-zmin)/(Kepler*DistV))
    
    ## Create square packed somata within limits
    xlin        = np.linspace(xmin,xmax2,(xmax2-xmin)/DistH)
    ylin        = np.linspace(ymin,ymax2,(ymax2-ymin)/DistH)
    zlin        = np.linspace(zmin,zmax2,zlayers)
    xx,yy,zz    = np.meshgrid(xlin,ylin,zlin,indexing='ij')

    nxy        = xlin.size * ylin.size

    ## Shift every other z-layer to create hexagonal packing
    for i in xrange(0,int(zlayers),2):
        startpt                     = nxy*i
        endpt                       = (nxy*(i+1))-1
        xx[startpt:endpt]           = xx[startpt:endpt]-DistH

    return np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))


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

    pclouds = []
    for ((layer_name,max_extent),(_,min_extent)) in itertools.izip(max_extents.iteritems(),min_extents.iteritems()):
        if rank == 0:
            print 'creating surfaces for layer %s' % layer_name
        max_srf = make_surface(l=max_extent[2], spatial_resolution=spatial_resolution)
        min_srf = make_surface(l=min_extent[2], spatial_resolution=spatial_resolution)
        if rank == 0:
            print 'creating point clouds for layer %s' % layer_name
        max_pcloud = max_srf.point_cloud()
        min_pcloud = min_srf.point_cloud()
        pclouds.append((layer_name,min_pcloud,max_pcloud))

    population_ranges = read_population_ranges(output_path, comm)[0]


    for population in populations:

        if rank == 0:
            print 'population: ',population

        (population_start, population_count) = population_ranges[population]

        coords_dict = {}

        points = cellgrid(env, population)
        
        count = 0
        
        coords = []
        for (layer_name,min_pcloud,max_pcloud) in pclouds:

            layer_count = env.geometry['Cell Layer Counts'][population][layer_name]

            max_extents = env.geometry['Parametric Surface']['Maximum Extent'][layer_name]
            min_extents = env.geometry['Parametric Surface']['Minimum Extent'][layer_name]

            l_min = min_extents[2]
            l_max = max_extents[2]

            if layer_count > 0:
                
                min_dd, min_ii = min_pcloud.tree.query(points,k=1,n_jobs=-1)
                max_dd, max_ii = max_pcloud.tree.query(points,k=1,n_jobs=-1)

                layer_point_sqdiff = (max_pcloud.tree.data[max_ii,:] - min_pcloud.tree.data[min_ii,:])**2
                layer_point_dist   = np.sqrt(layer_point_sqdiff.sum(axis=-1))

                min_z = min_pcloud.tree.data[min_ii,2]
                max_z = max_pcloud.tree.data[max_ii,2]
                
                in_points_idxs = np.where((min_dd < layer_point_dist) & (max_dd < layer_point_dist) &
                                          (points[:,2] > min_z) & (points[:,2] < max_z))[0]

                sampled_idxs  = np.random.randint(0, in_points_idxs.size-1, size=int(layer_count))

                for i in sampled_idxs:

                    pt_index = in_points_idxs[i]
                    pt       = points[pt_index,:]

                    x_coord = pt[0]
                    y_coord = pt[1]
                    z_coord = pt[2]

                    if min_dd[pt_index] < max_dd[pt_index]:
                        u_coord = min_pcloud.U[min_ii[pt_index]]
                        v_coord = min_pcloud.V[min_ii[pt_index]]
                    else:
                        u_coord = max_pcloud.U[max_ii[pt_index]]
                        v_coord = max_pcloud.V[max_ii[pt_index]]
                    l_frac = min_dd[pt_index] / (min_dd[pt_index] + max_dd[pt_index])
                    l_coord = l_min + ((l_max - l_min) * l_frac)


                    coords.append((x_coord,y_coord,z_coord,u_coord,v_coord,l_coord))
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
                                namespace='Sampled Coordinates',
                                io_size=io_size, chunk_size=chunk_size,
                                value_chunk_size=value_chunk_size,comm=comm)

        

if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find("sample_grid_coordinates.py") != -1,sys.argv)+1):])
