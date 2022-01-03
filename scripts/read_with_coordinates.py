##
## Reads the specified attribute namespace along with corresponding coordinates.
##

import gc, logging, os, os.path, sys
import click
from mpi4py import MPI
import h5py
import numpy as np
import dentate
import dentate.utils as utils
from dentate.geometry import make_distance_interpolant, measure_distances
from neuroh5.io import read_cell_attributes, read_population_names, read_population_ranges

sys_excepthook = sys.excepthook
def mpi_excepthook(type, value, traceback):
     sys_excepthook(type, value, traceback)
     sys.stdout.flush()
     sys.stderr.flush()
     if MPI.COMM_WORLD.size > 1:
         MPI.COMM_WORLD.Abort(1)
sys.excepthook = mpi_excepthook


@click.command()
@click.option("--include", '-i', type=str, multiple=True)
@click.option("--coords-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--coords-namespace", type=str, default='Coordinates')
@click.option("--distances-namespace", type=str, default='Arc Distances')
@click.option("--resolution", type=(int,int,int), default=(30,30,10))
@click.option("--io-size", type=int, default=-1)
@click.option("--verbose", "-v", is_flag=True)
def main(include, coords_path, coords_namespace, distances_namespace, resolution, io_size,
         verbose):

    utils.config_logging(verbose)
    logger = utils.get_script_logger(os.path.basename(__file__))
    
    comm = MPI.COMM_WORLD
    rank = comm.rank

    connection_config = env.connection_config
    extent      = {}
    
    population_ranges = read_population_ranges(coords_path)[0]
    populations = sorted(list(population_ranges.keys()))

    color = 0
    if rank == 0:
         color = 1
    comm0 = comm.Split(color, 0)

    soma_distances = {}
    soma_coords = {}
    for population in populations:
        if rank == 0:
            logger.info(f'Reading {population} coordinates...')
            coords_iter = read_cell_attributes(coords_path, population, comm=comm0,
                                               mask=set(['U Coordinate', 'V Coordinate', 'L Coordinate']),
                                               namespace=coords_namespace)
            distances_iter = read_cell_attributes(coords_path, population, comm=comm0,
                                                  mask=set(['U Distance', 'V Distance']),
                                                  namespace=distances_namespace)

            soma_coords[population] = { k: (float(v['U Coordinate'][0]), 
                                            float(v['V Coordinate'][0]), 
                                            float(v['L Coordinate'][0])) for (k,v) in coords_iter }

            distances = { k: (float(v['U Distance'][0]), 
                              float(v['V Distance'][0])) for (k,v) in distances_iter }
            
            if len(distances) > 0:
                 soma_distances[population] = distances
        
            gc.collect()

    comm.barrier()
    comm0.Free()

    soma_distances = comm.bcast(soma_distances, root=0)
    soma_coords = comm.bcast(soma_coords, root=0)

    extra_columns_list = extra_columns.split(",")
    columns = ['Field Width', 'X Offset', 'Y Offset']+extra_columns_list
    df_dict = {}
    it = read_cell_attributes(features_path, population, 
                              namespace=features_namespace)

    for cell_gid, features_dict in it:
        cell_field_width = features_dict['Field Width'][0]
        cell_xoffset = features_dict['X Offset'][0]
        cell_yoffset = features_dict['Y Offset'][0]
        
        df_dict[cell_gid] = [cell_field_width, cell_xoffset, cell_yoffset]

        
    df = pd.DataFrame.from_dict(df_dict, orient='index', columns=columns)
    df = df.reindex(sorted(df_dict.keys()))
    df.to_csv('features.%s.csv' % population)


    MPI.Finalize()

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv)+1):])
