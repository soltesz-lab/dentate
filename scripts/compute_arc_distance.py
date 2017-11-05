##
## Computes arc distance approximations relative to beginning of U and V coordinate ranges.
##

import sys
from mpi4py import MPI
import itertools
from neuroh5.io import read_population_ranges, read_population_names, append_cell_attributes, NeuroH5CellAttrGen
from DG_surface import make_surface
import numpy as np
import click


def list_find (f, lst):
    i=0
    for x in lst:
        if f(x):
            return i
        else:
            i=i+1
    return None


script_name = 'compute_arc_distance.py'


@click.command()
@click.option("--config", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--coords-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--coords-namespace", type=str, default='Sorted Coordinates')
@click.option("--distance-namespace", type=str, default='Arc Distance')
@click.option("--npoints", type=int, default=12000)
@click.option("--origin-u", type=float, default=0.0)
@click.option("--origin-v", type=float, default=0.0)
@click.option("--spatial-resolution", type=float, default=1.0)
@click.option("--io-size", type=int, default=-1)
def main(config_path, coords_path, coords_namespace, distance_namespace, npoints, origin_u, origin_v, spatial_resolution, layers, io_size):

    comm = MPI.COMM_WORLD
    rank = comm.rank

    env = Env(comm=comm, configFile=config)

    layers = []
    max_extents = env.layers['Parametric Surface']['Minimum Extent']
    min_extents = env.layers['Parametric Surface']['Maximum Extent']

    for (max_extent,min_extent) in itertools.izip(max_extents,min_extents):
        mid = (max_extent[2] - min_extent) / 2.
        layers.append(mid)
    
    population_ranges = read_population_ranges(comm, coords_path)[0]
    
    ip_surfaces = []
    for layer in layers:
        ip_surfaces.append(make_surface(l=layer, spatial_resolution=spatial_resolution))
    
    for population in population_ranges:
        (population_start, _) = population_ranges[population]

        for (layer_index, (layer, ip_surface)) in enumerate(itertools.izip(layers, ip_surfaces)):
            
            for cell_gid, attr_dict in NeuroH5CellAttrGen(comm, coords_path, population, io_size=io_size,
                                                        namespace=coords_namespace):
                arc_distance_dict = {}
                if cell_gid is None:
                    print 'Rank %i cell gid is None' % rank
                else:
                    cell_coords_dict = attr_dict[coords_namespace]

            
                    cell_u = cell_coords_dict['U Coordinate']
                    cell_v = cell_coords_dict['V Coordinate']
                    
                    U = np.linspace(origin_u, cell_u, npoints)
                    V = np.linspace(origin_v, cell_v, npoints)
                    
                    arc_distance_u = ip_surface.point_distance(U, cell_v, normalize_uv=True)
                    arc_distance_v = ip_surface.point_distance(cell_u, V, normalize_uv=True)
                    
                    arc_distance_dict[cell_gid-population_start] = {'U Distance': np.asarray([arc_distance_u], dtype='float32'),
                                                                    'V Distance': np.asarray([arc_distance_v], dtype='float32') }
                    
                    print 'Rank %i: gid = %i u = %f v = %f dist u = %f dist v = %f' % (rank, cell_gid, cell_u, cell_v, arc_distance_u, arc_distance_v)

                append_cell_attributes(comm, coords_path, population, arc_distance_dict,
                                       namespace='%s Layer %i' % (distance_namespace, layer_index),
                                       io_size=io_size)
                comm.Barrier()
            
if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])

