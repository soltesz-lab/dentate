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
import utils
from env import Env

script_name = 'compute_arc_distance.py'


@click.command()
@click.option("--config", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--coords-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--coords-namespace", type=str, default='Sorted Coordinates')
@click.option("--distance-namespace", type=str, default='Arc Distance')
@click.option("--layers", '-l', type=str, multiple=True)
@click.option("--npoints", type=int, default=12000)
@click.option("--spatial-resolution", type=float, default=1.0)
@click.option("--io-size", type=int, default=-1)
@click.option("--verbose", "-v", is_flag=True)
def main(config, coords_path, coords_namespace, distance_namespace, layers, npoints, spatial_resolution, io_size, verbose):

    comm = MPI.COMM_WORLD
    rank = comm.rank

    env = Env(comm=comm, configFile=config)
    
    max_extents = env.geometry['Parametric Surface']['Minimum Extent']
    min_extents = env.geometry['Parametric Surface']['Maximum Extent']

    layer_mids = []
    for ((layer_name,max_extent),(_,min_extent)) in itertools.izip(max_extents.iteritems(),min_extents.iteritems()):
        if layer_name in layers:
            mid = (max_extent[2] - min_extent[2]) / 2.
            layer_mids.append(mid)
    
    population_ranges = read_population_ranges(comm, coords_path)[0]
    
    ip_surfaces = []
    for layer in layer_mids:
        ip_surfaces.append(make_surface(l=layer, spatial_resolution=spatial_resolution))
    
    for population in population_ranges:
        (population_start, _) = population_ranges[population]

        for (layer_index, (layer_name, layer_mid, ip_surface)) in enumerate(itertools.izip(layers, layer_mids, ip_surfaces)):

            origin_u = np.min(ip_surface.su[0])
            origin_v = np.min(ip_surface.sv[0])
            
            for cell_gid, cell_coords_dict in NeuroH5CellAttrGen(comm, coords_path, population, io_size=io_size,
                                                                 namespace=coords_namespace):
                arc_distance_dict = {}
                if cell_gid is None:
                    print 'Rank %i cell gid is None' % rank
                else:
                    cell_u = cell_coords_dict['U Coordinate']
                    cell_v = cell_coords_dict['V Coordinate']

                    U = np.linspace(origin_u, cell_u, npoints)
                    V = np.linspace(origin_v, cell_v, npoints)
                    
                    arc_distance_u = ip_surface.point_distance(U, cell_v, normalize_uv=True)
                    arc_distance_v = ip_surface.point_distance(cell_u, V, normalize_uv=True)
                    
                    arc_distance_dict[cell_gid-population_start] = {'U Distance': np.asarray([arc_distance_u], dtype='float32'),
                                                                    'V Distance': np.asarray([arc_distance_v], dtype='float32') }

                    if verbose:
                        print 'Rank %i: gid = %i u = %f v = %f dist u = %f dist v = %f' % (rank, cell_gid, cell_u, cell_v, arc_distance_u, arc_distance_v)

                append_cell_attributes(comm, coords_path, population, arc_distance_dict,
                                       namespace='%s Layer %s' % (distance_namespace, layer_name),
                                       io_size=io_size)
                comm.Barrier()
            
if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])

