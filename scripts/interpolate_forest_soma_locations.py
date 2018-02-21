from mpi4py import MPI
from neuroh5.io import population_ranges, append_cell_attributes
from dentate.DG_volume import make_volume
from dentate.env import Env
import random
import click  # CLI argument processing
import logging

script_name = 'interpolate_forest_soma_locations.py'
logger = logging.getLogger(script_name)

@click.command()
@click.option("--forest-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--coords-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--spatial-resolution", type=float, default=1.0)
@click.option("--io-size", type=int, default=-1)
@click.option("--verbose", "-v", is_flag=True)
def main(forest_path, coords_path, spatial_resolutin, io_size, verbose):
    if verbose:
        logger.setLevel(logging.INFO)

    comm = MPI.COMM_WORLD
    rank = comm.rank  

    if io_size==-1:
        io_size = comm.size

    if rank == 0:
        logger.info('%i ranks have been allocated' % comm.size)

    if rank == 0:
        logger.info('Reading population coordinates...')
    populations = population_ranges(forest_path).keys()
    
    if rank == 0:
        logger.info('Creating volume...')
    ip_volume = make_volume(-3.95, 3.2)

    
    for population in populations:
        (trees, forestSize) = scatter_read_trees(forest_path, population, io_size=io_size, comm=comm)

        for (gid, morph_dict) in trees:

            px       = morph_dict['x'][0]
            py       = morph_dict['y'][0]
            pz       = morph_dict['z'][0]

            xyz_coords  = np.array([px,py,pz]).reshape(1,3)
            uvl_coords  = ip_volume.inverse(xyz_coords)
            xyz_coords1 = ip_volume(uvl_coords[i,0],uvl_coords[i,1],uvl_coords[i,2]).ravel()
            xyz_error   = np.abs(np.subtract(xyz_coords, xyz_coords1))
            
            coords_dict[gid]['X Coordinate'] = np.array([xyz_coords1[0]], dtype='float32')
            coords_dict[gid]['Y Coordinate'] = np.array([xyz_coords1[1]], dtype='float32')
            coords_dict[gid]['Z Coordinate'] = np.array([xyz_coords1[2]], dtype='float32')
            coords_dict[gid]['U Coordinate'] = np.array([uvl_coords1[0]], dtype='float32')
            coords_dict[gid]['V Coordinate'] = np.array([uvl_coords1[1]], dtype='float32')
            coords_dict[gid]['L Coordinate'] = np.array([uvl_coords1[2]], dtype='float32')

            coords_dict[gid]['Interpolation Error'] = np.asarray(xyz_error, dtype='float32')
            
        append_cell_attributes(coords_path, population, coords_dict,
                               namespace='Interpolated Coordinates', io_size=io_size, comm=comm)
        del coords_dict
        gc.collect()
        global_count = comm.gather(count, root=0)
        if rank == 0:
            logger.info('Interpolation of %i %s cells took %i s' % (np.sum(global_count), \
                                                                    population, \
                                                                    time.time()-start_time))

if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])

