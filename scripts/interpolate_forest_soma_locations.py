import sys, time, gc, itertools
from mpi4py import MPI
import numpy as np
from neuroh5.io import read_population_ranges, scatter_read_trees, append_cell_attributes
from rbf_volume import rotate3d
from dentate.DG_volume import make_volume
from dentate.env import Env
from dentate.utils import list_find, list_argsort
import random
import click  # CLI argument processing
import logging
logging.basicConfig()

script_name = 'interpolate_forest_soma_locations.py'
logger = logging.getLogger(script_name)

def list_concat(a, b, datatype):
    return a+b

concatOp = MPI.Op.Create(list_concat, commute=True)

@click.command()
@click.option("--config", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--forest-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--coords-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--populations", '-i', required=True, multiple=True, type=str)
@click.option("--rotate", type=float)
@click.option("--io-size", type=int, default=-1)
@click.option("--verbose", "-v", is_flag=True)
def main(config, forest_path, coords_path, populations, rotate, io_size, verbose):
    if verbose:
        logger.setLevel(logging.INFO)

    comm = MPI.COMM_WORLD
    rank = comm.rank  

    env = Env(comm=comm, configFile=config)

    swc_type_soma   = env.SWC_Types['soma']

    if rotate is not None:
        a = float(np.deg2rad(rotate))
        rot = rotate3d([1,0,0], a)

    if io_size==-1:
        io_size = comm.size

    if rank == 0:
        logger.info('%i ranks have been allocated' % comm.size)

    if rank == 0:
        logger.info('Reading population coordinates...')
        
    (pop_ranges, _)  = read_population_ranges(forest_path)
    

    if rank == 0:
        color = 1
    else:
        color = 0

    ## comm0 includes only rank 0
    comm0 = comm.Split(color, 0)
    
    for population in populations:
        min_extent = env.geometry['Cell Layers']['Minimum Extent'][population]
        max_extent = env.geometry['Cell Layers']['Maximum Extent'][population]
        
        if rank == 0:
            logger.info('Reading forest for population %s...' % population)
            
        (trees, forestSize) = scatter_read_trees(forest_path, population, io_size=io_size, comm=comm)
        (population_start, _) = pop_ranges[population]

        if rank == 0:
            logger.info('Constructing volume...')
        ip_volume = make_volume(min_extent[2], max_extent[2])

        if rank == 0:
            logger.info('Interpolating forest coordinates...')
            
        count = 0
        coords = []
        coords_dict = {}
        start_time = time.time()
        for (gid, morph_dict) in trees:

            swc_type = morph_dict['swc_type']
            xs       = morph_dict['x']
            ys       = morph_dict['y']
            zs       = morph_dict['z']

            px       = xs[0]
            py       = ys[0]
            pz       = zs[0]
            pts      = np.array([px,py,pz]).reshape(3,1)

            if rotate:
                xyz_coords = np.dot(rot, pts).T
            else:
                xyz_coords = pts.T
            
            uvl_coords  = ip_volume.inverse(xyz_coords)
            xyz_coords1 = ip_volume(uvl_coords[0,0],uvl_coords[0,1],uvl_coords[0,2]).ravel()
            xyz_error   = np.abs(np.subtract(xyz_coords, xyz_coords1))

            coords_dict[gid] = { 'X Coordinate': np.array([xyz_coords1[0]], dtype='float32'),
                                 'Y Coordinate': np.array([xyz_coords1[1]], dtype='float32'),
                                 'Z Coordinate': np.array([xyz_coords1[2]], dtype='float32'),
                                 'U Coordinate': np.array([uvl_coords[0,0]], dtype='float32'),
                                 'V Coordinate': np.array([uvl_coords[0,1]], dtype='float32'),
                                 'L Coordinate': np.array([uvl_coords[0,2]], dtype='float32'),
                                 'Interpolation Error': np.asarray(xyz_error[0], dtype='float32') }

            if (uvl_coords[0,0] <= max_extent[0]) and (uvl_coords[0,0] >= min_extent[0]) and \
                (uvl_coords[0,1] <= max_extent[1]) and (uvl_coords[0,1] >= min_extent[1]):
                    coords.append((gid, uvl_coords[0,0], uvl_coords[0,1], uvl_coords[0,2]))

            count += 1
            
        append_cell_attributes(coords_path, population, coords_dict,
                               namespace='Interpolated Coordinates', io_size=io_size, comm=comm)

        global_count = comm.gather(count, root=0)
        if rank == 0:
            if global_count > 0:
                logger.info('Interpolation of %i %s cells took %i s' % (np.sum(global_count), \
                                                                        population, \
                                                                        time.time()-start_time))
        del coords_dict
        gc.collect()

        all_coords = comm.reduce(coords, root=0, op=concatOp)
            
        if rank == 0:
            if len(all_coords) > 0:
                coords_sort_idxs = list_argsort(lambda coords: coords[1], all_coords) ## sort on U coordinate
                reindex_dict = { coords[0]: { 'New Cell Index' : np.array([(i+population_start)], dtype='uint32') }
                                     for (i, coords) in itertools.izip (coords_sort_idxs, all_coords) }
                append_cell_attributes(coords_path, population, reindex_dict,
                                    namespace='Tree Reindex', io_size=1, comm=comm0)
            
        comm0.Barrier()
            

if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])

