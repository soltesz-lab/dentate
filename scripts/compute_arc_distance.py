import sys
from mpi4py import MPI
from neuroh5.io import read_population_ranges, read_population_names, bcast_cell_attributes, append_cell_attributes
from DG_surface import make_surface
import numpy as np
import utils
import click

script_name = 'compute_arc_distance.py'

@click.command()
@click.option("--coords-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--coords-namespace", type=str, default='Sorted Coordinates')
@click.option("--npoints", type=int, default=12000)
@click.option("--io-size", type=int, default=-1)
def main(coords_path, coords_namespace, npoints, io_size):

    comm = MPI.COMM_WORLD
    rank = comm.rank


    population_ranges = read_population_ranges(comm, coords_path)[0]
    
    soma_coords = {}
    for population in population_ranges.keys():
        soma_coords[population] = bcast_cell_attributes(comm, 0, coords_path, population,
                                                        namespace=coords_namespace)
        
    origin_u = 0.0
    origin_v = 0.0

    ip_surface  = make_surface(l=3.0, spatial_resolution=10.) ## Corresponds to OML boundary
    
    for population in population_ranges:
        (population_start, _) = population_ranges[population]
        coords = soma_coords[population]

        for (cell_gid, cell_coords_dict) in coords.iteritems():

            arc_distance_dict = {}
            
            cell_u = cell_coords_dict['U Coordinate']
            cell_v = cell_coords_dict['V Coordinate']

            U = np.linspace(origin_u, cell_u, npts)
            V = np.linspace(origin_v, cell_v, npts)

            arc_distance_u = ip_surface.point_distance(U, cell_v, normalize_uv=True)
            arc_distance_v = ip_surface.point_distance(cell_u, V, normalize_uv=True)

            arc_distance_dict[cell_gid-population_start] = {'U Distance': np.asarray([arc_distance_u], dtype='float32'),
                                                            'V Distance': np.asarray([arc_distance_u], dtype='float32') }

            print 'Rank %i: gid = %i u = %f v = %f dist u = %f dist v = %f' % (rank, cell_gid, cell_u, cell_v, arc_distance_u, arc_distance_v)

            append_cell_attributes(comm, coords_path, population, arc_distance_dict,
                                   namespace='Arc Distance', io_size=io_size)


if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])

