import os, sys, itertools, logging, time
import click
from mpi4py import MPI
import numpy as np
import dentate
from dentate.env import Env
from dentate.geometry import DG_volume, make_uvl_distance, make_volume, get_total_extents, uvl_in_bounds, optimize_inverse_uvl_coords
from dentate.utils import *
from neuroh5.io import append_cell_attributes, read_population_ranges, scatter_read_trees

sys_excepthook = sys.excepthook
def mpi_excepthook(type, value, traceback):
    sys_excepthook(type, value, traceback)
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)
#sys.excepthook = mpi_excepthook

def list_concat(a, b, datatype):
    return a+b

mpi_op_concat = MPI.Op.Create(list_concat, commute=True)
    

@click.command()
@click.option("--config", required=True, type=str)
@click.option("--config-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True), default='config')
@click.option("--forest-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--coords-path", required=True, type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option("--populations", '-i', required=True, multiple=True, type=str)
@click.option("--resolution", type=(int,int,int), default=(30,30,10))
@click.option("--reltol", type=float, default=10.)
@click.option("--optiter", type=int, default=250)
@click.option("--io-size", type=int, default=-1)
@click.option("--verbose", "-v", is_flag=True)
@click.option("--dry-run", is_flag=True)
def main(config, config_prefix, forest_path, coords_path, populations, resolution, reltol, optiter, io_size, verbose, dry_run):

    config_logging(verbose)
    logger = get_script_logger(__file__)

    comm = MPI.COMM_WORLD
    rank = comm.rank  

    env = Env(comm=comm, config_file=config, config_prefix=config_prefix)
    swc_type_soma   = env.SWC_Types['soma']

    if io_size==-1:
        io_size = comm.size

    if rank==0:
        import h5py
        if not os.path.isfile(coords_path):
            input_file  = h5py.File(forest_path,'r')
            output_file = h5py.File(coords_path,'w')
            input_file.copy('/H5Types',output_file)
            input_file.close()
            output_file.close()
    comm.barrier()

    (pop_ranges, _)  = read_population_ranges(forest_path)
    

    if rank == 0:
        color = 1
    else:
        color = 0

    ## comm0 includes only rank 0
    comm0 = comm.Split(color, 0)

    rotate = env.geometry['Parametric Surface']['Rotation']
    origin = env.geometry['Parametric Surface']['Origin']
    layer_extents = env.geometry['Parametric Surface']['Layer Extents']

    ((min_u, max_u), (min_v, max_v), (min_l, max_l)) = get_total_extents(layer_extents)

    ## This parameter is used to expand the range of L and avoid
    ## situations where the endpoints of L end up outside of the range
    ## of the distance interpolant
    safety = 0.001

    ip_volume = None
    if rank == 0:
        ip_volume = make_volume((min_u-safety, max_u+safety), \
                                (min_v-safety, max_v+safety), \
                                (min_l-safety, max_l+safety), \
                                resolution=resolution, rotate=rotate)

    ip_volume = env.comm.bcast(ip_volume, root=0)
    
    for population in populations:
        pop_layers = env.geometry['Cell Distribution'][population]        

        if rank == 0:
            logger.info('Reading forest for population %s...' % population)
            
        (trees, forestSize) = scatter_read_trees(forest_path, population, io_size=io_size, comm=comm)
        (population_start, _) = pop_ranges[population]

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

            xyz_coords = np.array([px,py,pz]).reshape(3,1).T
            uvl_coords_interp = ip_volume.inverse(xyz_coords)[0]
            xyz_coords_interp = DG_volume(uvl_coords_interp[0], uvl_coords_interp[1], uvl_coords_interp[2], 
                                           rotate=rotate)[0]
            xyz_error_interp   = np.abs(np.subtract(xyz_coords, xyz_coords_interp))[0]

            uvl_coords_opt = optimize_inverse_uvl_coords(xyz_coords, rotate, layer_extents, pop_layers, optiter=optiter)
            if uvl_coords_opt:
                xyz_coords_opt = DG_volume(uvl_coords_opt[0], uvl_coords_opt[1], uvl_coords_opt[2], 
                                            rotate=rotate)[0]
                xyz_error_opt   = np.abs(np.subtract(xyz_coords, xyz_coords_opt))[0]
            else:
                xyz_coords_opt = None
                xyz_error_opt = None

            if (xyz_error_opt is not None) and \
               (xyz_error_opt[0] < xyz_error_interp[0]) and \
               (xyz_error_opt[1] < xyz_error_interp[1]) and \
               (xyz_error_opt[2] < xyz_error_interp[2]):
                uvl_coords = uvl_coords_opt
                xyz_error = xyz_error_opt
                xyz_coords1 = xyz_coords_opt
            else:
                uvl_coords = uvl_coords_interp
                xyz_error = xyz_error_interp
                xyz_coords1 = xyz_coords_interp
            
            if rank == 0:
                logger.info('xyz_coords: %s' % str(xyz_coords))
                logger.info('uvl_coords_interp: %s' % str(uvl_coords_interp))
                logger.info('xyz_coords_interp: %s' % str(xyz_coords_interp))
                logger.info('xyz_error_interp: %s' % str(xyz_error_interp))
                logger.info('uvl_coords_opt: %s' % str(uvl_coords_opt))
                logger.info('xyz_coords_opt: %s' % str(xyz_coords_opt))
                logger.info('xyz_error_opt: %s' % str(xyz_error_opt))
                logger.info('uvl_in_bounds: %s' % str(uvl_in_bounds(uvl_coords, layer_extents, pop_layers)))

                
            coords_dict[gid] = { 'X Coordinate': np.array([xyz_coords1[0]], dtype='float32'),
                                 'Y Coordinate': np.array([xyz_coords1[1]], dtype='float32'),
                                 'Z Coordinate': np.array([xyz_coords1[2]], dtype='float32'),
                                 'U Coordinate': np.array([uvl_coords[0]], dtype='float32'),
                                 'V Coordinate': np.array([uvl_coords[1]], dtype='float32'),
                                 'L Coordinate': np.array([uvl_coords[2]], dtype='float32'),
                                 'Interpolation Error': np.asarray(xyz_error, dtype='float32') }

            if uvl_in_bounds(uvl_coords, layer_extents, pop_layers) and \
               (xyz_error[0] <= reltol) and (xyz_error[1] <= reltol) and (xyz_error[2] <= reltol):
                coords.append((gid, uvl_coords[0], uvl_coords[1], uvl_coords[2]))
            else:
                logger.warning("Rank %d: uvl coords %s not added to new index (in bounds: %s; error: %s)" % \
                               (rank, str(uvl_coords), str(uvl_in_bounds(uvl_coords, layer_extents, pop_layers)), str(xyz_error)))

            count += 1

        if not dry_run:
            append_cell_attributes(coords_path, population, coords_dict,
                                   namespace='Interpolated Coordinates', 
                                   io_size=io_size, comm=comm)

        global_count = np.sum(np.asarray(comm.gather(count, root=0)))
        if rank == 0:
            if global_count > 0:
                logger.info('Interpolation of %i %s cells took %i s' % (np.sum(global_count), \
                                                                        population, \
                                                                        time.time()-start_time))
        all_coords = comm.reduce(coords, root=0, op=mpi_op_concat)
            
        if rank == 0:
            if len(all_coords) > 0:
                coords_sort_idxs = list_argsort(lambda coords: coords[1], all_coords) ## sort on U coordinate
                reindex_dict = { coords[0]: { 'New Cell Index' : np.array([(i+population_start)], dtype='uint32') }
                                     for (i, coords) in zip (coords_sort_idxs, all_coords) }
                append_cell_attributes(coords_path, population, reindex_dict,
                                       namespace='Tree Reindex', comm=comm0)
            
        comm.barrier()


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv)+1):])
