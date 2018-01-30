##
## Import dendritic X Y Z coordinates from a forest of morphologies into 'Sampled Coordinates' namespace.
##


import sys
from mpi4py import MPI
import h5py
import numpy as np
from neuroh5.io import read_population_ranges, append_cell_attributes
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
        if layer_name in layers:
            max_srf = make_surface(l=max_extent, spatial_resolution=spatial_resolution)
            min_srf = make_surface(l=min_extent, spatial_resolution=spatial_resolution)
            max_pcloud = max_srf.point_cloud()
            min_pcloud = min_srf.point_cloud()
            pclouds.append((min_pcloud,max_pcloud))
    
    population_ranges = read_population_ranges(output_path)[0]
    
    for population in populations:

        if rank == 0:
            print 'population: ',population

        (population_start, population_count) = population_ranges[population]

        coords_dict = {}

        points = somagrid(config, population)

        start = population_start

        for (min_pcloud,max_pcloud) in pclouds:

            if layer_count > 0:
                
                min_dd, min_ii = min_pcloud.tree.query(points,k=1,n_jobs=-1)
                max_dd, max_ii = max_pcloud.tree.query(points,k=1,n_jobs=-1)
                
                layer_point_dist = point_distance(min_pcloud.tree.data[min_ii], min_pcloud.tree.data[max_ii])
                
                in_points = points[np.where((min_dd < layer_point_dist) & (max_dd < layer_point_dist))]
            
                sampled_idxs  = np.random.randint(0, in_points.shape[0]-1, size=int(layer_count))

                k = 0
                for i in sampled_idxs:

                    pt      = points[i]
                    x_coord = pt[0]
                    y_coord = pt[1]
                    z_coord = pt[2]

                    if min_dd[i] < max_dd[i]:
                        u_coord = min_pcloud.U[min_ii]
                        v_coord = min_pcloud.V[min_ii]
                    else:
                        u_coord = max_pcloud.U[max_ii]
                        v_coord = max_pcloud.V[max_ii]
                        
                    coords_dict[start+k] = { 'X Coordinate': x_coord,
                                             'Y Coordinate': y_coord,
                                             'Z Coordinate': z_coord,
                                             'U Coordinate': u_coord,
                                             'V Coordinate': v_coord }
                    k = k+1
                
                start += layer_count

        assert(start == population_count)
        append_cell_attributes(output_path, population, coords_dict,
                                namespace='Sampled Coordinates',
                                io_size=io_size, chunk_size=chunk_size,
                                value_chunk_size=value_chunk_size)

        

if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find("sample_grid_coordinates.py") != -1,sys.argv)+1):])
